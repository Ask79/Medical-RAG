# File: src/rag_med/linkers/build_rxnorm_chunks.py
"""
Build compact RxNorm "fact chunks" for retrieval.

Merges:
  - rx_meta.jsonl         (TTY + has_ndc)                 [REQUIRED]
  - RXNCONSO.RRF          (preferred name + synonyms)     [REQUIRED]
  - rx_link_keys.jsonl    (NDCs -> ndc_count)             [REQUIRED]
  - daily_rx_links.jsonl  (SPL<->RXCUI raw links)         [OPTIONAL]
  - daily_rx_links_resolved.jsonl (SPL->primary RXCUI)    [OPTIONAL]

Output JSONL per RXCUI (default caps: synonyms<=20, sample_spl_set_ids<=3):
{
  "rxcui": "12345",
  "tty": "SCD",
  "preferred_name": "Acetaminophen 500 MG Oral Tablet",
  "synonyms": ["Tylenol 500 mg", "Acetaminophen Tablet 500 mg"],
  "has_ndc": true,
  "ndc_count": 37,
  "ingredients": ["161"],   # only if present in rx_meta
  "route": "oral",          # optional heuristic from names (basic)
  "dose_form": "tablet",     # optional heuristic from names (basic)
  "strengths": ["500 mg"],   # optional heuristic from names (basic)
  "spl_count": 12,
  "sample_spl_set_ids": ["ABD6ECF0-...", "AAE8B7A4-...", "dffb4544-..."]
}

Usage example:
  python -u src/rag_med/linkers/build_rxnorm_chunks.py \
    --rx-meta-in data/links/rx_meta.jsonl \
    --conso-in   data/raw/.../rrf/RXNCONSO.RRF \
    --rx-keys-in data/links/rx_link_keys.jsonl \
    --links-in   data/links/daily_rx_links.jsonl \
    --out        data/links/rxnorm_chunks.jsonl

Or sample SPLs from RESOLVED links (preferred):
  python -u src/rag_med/linkers/build_rxnorm_chunks.py \
    --rx-meta-in data/links/rx_meta.jsonl \
    --conso-in   data/raw/.../rrf/RXNCONSO.RRF \
    --rx-keys-in data/links/rx_link_keys.jsonl \
    --resolved-in data/links/daily_rx_links_resolved.jsonl \
    --out        data/links/rxnorm_chunks.jsonl \
    --use-resolved true
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

try:
    import orjson as _json
    def dumps(obj):
        return _json.dumps(obj).decode()
except Exception:
    def dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rx-meta-in', required=True)
    p.add_argument('--conso-in', required=True)
    p.add_argument('--rx-keys-in', required=True)
    p.add_argument('--links-in')
    p.add_argument('--resolved-in')
    p.add_argument('--use-resolved', type=str, default='true', help='true/false: sample SPLs from resolved file when provided')
    p.add_argument('--out', required=True)
    p.add_argument('--max-synonyms', type=int, default=20)
    p.add_argument('--sample-spls', type=int, default=3)
    p.add_argument('--progress-every', type=int, default=500000)
    return p.parse_args()

# ------------- Loaders ----------------

def load_rx_meta(path: str) -> Dict[str, Dict]:
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rxcui = str(row.get('rxcui') or '').strip()
            if not rxcui:
                continue
            m = {
                'tty': (row.get('tty') or '').upper(),
                'has_ndc': bool(row.get('has_ndc')),
            }
            if isinstance(row.get('ingredients'), list):
                m['ingredients'] = [str(x) for x in row['ingredients'] if x]
            meta[rxcui] = m
    print(f"Loaded rx_meta: {len(meta):,} RXCUIs", file=sys.stderr)
    return meta


def load_ndc_counts_from_rx_keys(path: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            rxcui = str(row.get('rxcui') or '').strip()
            if not rxcui:
                continue
            ndcs = row.get('ndc11s') or []
            counts[rxcui] += len(ndcs)
    print(f"Derived ndc_count for {sum(1 for v in counts.values() if v):,} RXCUIs from rx_link_keys (rows read {total:,})", file=sys.stderr)
    return counts


def invert_links_from_raw(path: Optional[str], progress_every: int=500000) -> Dict[str, Counter]:
    """Return rxcui -> Counter({spl: count})."""
    inv: Dict[str, Counter] = defaultdict(Counter)
    if not path or not os.path.exists(path):
        return inv
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            spl = (row.get('spl_set_id') or '').strip()
            rxcui = str(row.get('rxcui') or '').strip()
            if not spl or not rxcui:
                continue
            inv[rxcui][spl] += 1
            if progress_every and total % progress_every == 0:
                print(f".. raw links read {total:,} | RXCUIs={len(inv):,}", file=sys.stderr)
    print(f"Built raw rxcui->SPL map for {len(inv):,} RXCUIs from daily_rx_links", file=sys.stderr)
    return inv


def invert_links_from_resolved(path: Optional[str], progress_every: int=500000, include_secondaries: bool=False) -> Dict[str, Counter]:
    """Return rxcui -> Counter({spl: 1}) using resolved (primary by default)."""
    inv: Dict[str, Counter] = defaultdict(Counter)
    if not path or not os.path.exists(path):
        return inv
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            spl = (row.get('spl_set_id') or '').strip()
            if not spl:
                continue
            primary = row.get('primary') or {}
            pr = str(primary.get('rxcui') or '').strip()
            if pr:
                inv[pr][spl] += 1
            if include_secondaries:
                for sec in row.get('secondary') or []:
                    sr = str(sec.get('rxcui') or '').strip()
                    if sr:
                        inv[sr][spl] += 1
            if progress_every and total % progress_every == 0:
                print(f".. resolved links read {total:,} | RXCUIs={len(inv):,}", file=sys.stderr)
    print(f"Built resolved rxcui->SPL map for {len(inv):,} RXCUIs from daily_rx_links_resolved", file=sys.stderr)
    return inv

# ------------- CONSO parsing -------------
# RXNCONSO.RRF (per RxNorm "prescribe" RRF):
# RXCUI(0)|LAT(1)|TS(2)|LUI(3)|STT(4)|SUI(5)|ISPREF(6)|RXAUI(7)|SAUI(8)|SCUI(9)|SDUI(10)|SAB(11)|TTY(12)|CODE(13)|STR(14)|SUPPRESS(15)|CVF(16)

def load_conso_names(path: str, progress_every: int=500000) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Return (preferred_name, synonyms, tty_by_rxcui_from_conso)."""
    pref: Dict[str, str] = {}
    syns: Dict[str, List[str]] = defaultdict(list)
    tty_seen: Dict[str, str] = {}
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            parts = line.rstrip('\n').split('|')
            if len(parts) < 15:
                continue
            rxcui = parts[0].strip()
            tty = parts[12].strip().upper()
            s = parts[14].strip()
            if not rxcui or not s:
                continue
            # record a TTY seen in CONSO (fallback if rx_meta missing)
            if rxcui not in tty_seen:
                tty_seen[rxcui] = tty
            # prefer the first STR we encounter as candidate preferred; we'll overwrite later if better
            if rxcui not in pref:
                pref[rxcui] = s
            # collect synonyms (we'll dedup later)
            syns[rxcui].append(s)
            if progress_every and total % progress_every == 0:
                print(f".. CONSO read {total:,} lines | RXCUIs with names={len(pref):,}", file=sys.stderr)
    print(f"Loaded names/synonyms for {len(pref):,} RXCUIs from RXNCONSO", file=sys.stderr)
    return pref, syns, tty_seen

# ------------ Light NLP heuristics -------------
_strength_pat = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b", re.I)
_route_words = [
    'oral','topical','ophthalmic','otic','nasal','inhalation','subcutaneous','intravenous','intramuscular','rectal','vaginal','sublingual','transdermal'
]
_dose_forms = ['tablet','capsule','solution','suspension','cream','ointment','gel','lotion','patch','aerosol','spray','injection','kit','powder']

def heuristics_from_name(name: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    if not name:
        return None, None, []
    lower = name.lower()
    route = None
    for w in _route_words:
        if w in lower:
            route = w
            break
    dose_form = None
    for df in _dose_forms:
        if df in lower:
            dose_form = df
            break
    strengths = [f"{m.group(1)} {m.group(2)}" for m in _strength_pat.finditer(name)]
    return route, dose_form, strengths

# ------------- Main build -------------

def main():
    args = parse_args()

    use_resolved = str(args.use_resolved).lower() in ('1','true','yes','y')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.isdir(args.out):
        raise RuntimeError(f"--out path is a directory, not a file: {args.out}")

    rx_meta = load_rx_meta(args.rx_meta_in)
    ndc_counts = load_ndc_counts_from_rx_keys(args.rx_keys_in)

    # Names & synonyms
    preferred, synonyms, tty_from_conso = load_conso_names(args.conso_in, progress_every=args.progress_every)

    # SPL sampling maps
    inv_resolved = invert_links_from_resolved(args.resolved_in, include_secondaries=False) if use_resolved else {}
    inv_raw = {} if use_resolved else invert_links_from_raw(args.links_in)

    written = 0
    with open(args.out, 'w', encoding='utf-8') as w:
        for rxcui in preferred.keys():
            meta = rx_meta.get(rxcui, {})
            tty = meta.get('tty') or tty_from_conso.get(rxcui) or ''
            has_ndc = bool(meta.get('has_ndc'))
            ings = meta.get('ingredients')

            # names
            pref_name = preferred.get(rxcui)
            syn_list = list(dict.fromkeys(synonyms.get(rxcui, [])))  # dedupe preserve order
            if pref_name and syn_list and syn_list[0] != pref_name:
                # ensure preferred appears first
                if pref_name in syn_list:
                    syn_list.remove(pref_name)
                syn_list.insert(0, pref_name)
            # cap synonyms
            if args.max_synonyms and len(syn_list) > args.max_synonyms:
                syn_list = syn_list[:args.max_synonyms]

            # counts
            ndc_count = int(ndc_counts.get(rxcui, 0))

            # SPL sampling
            spl_counter = inv_resolved.get(rxcui) if use_resolved else inv_raw.get(rxcui)
            sample_spls: List[str] = []
            spl_count = 0
            if spl_counter:
                spl_count = sum(spl_counter.values()) if not use_resolved else len(spl_counter)
                # pick top-N by count
                for spl, _cnt in spl_counter.most_common(args.sample_spls):
                    sample_spls.append(spl)

            # heuristics from name
            route = dose_form = None
            strengths: List[str] = []
            if pref_name:
                route, dose_form, strengths = heuristics_from_name(pref_name)

            row = {
                'rxcui': rxcui,
                'tty': tty,
                'preferred_name': pref_name,
                'synonyms': syn_list,
                'has_ndc': has_ndc,
                'ndc_count': ndc_count,
            }
            if ings:
                row['ingredients'] = ings
            if route:
                row['route'] = route
            if dose_form:
                row['dose_form'] = dose_form
            if strengths:
                row['strengths'] = strengths
            if spl_count:
                row['spl_count'] = spl_count
            if sample_spls:
                row['sample_spl_set_ids'] = sample_spls

            w.write(dumps(row) + "\n")
            written += 1

    print(f"Wrote {written:,} RxNorm chunks → {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
