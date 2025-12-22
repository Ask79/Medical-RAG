# File: src/rag_med/linkers/resolve_links.py
"""
Resolve DailyMed ↔ RxNorm link conflicts by selecting a PRIMARY RXCUI per SPL,
while retaining all other matches as SECONDARY.

Inputs:
  --links-in     : data/links/daily_rx_links.jsonl              (REQUIRED)
  --out          : data/links/daily_rx_links_resolved.jsonl     (REQUIRED)
  --report-out   : data/links/reports/daily_rx_links_resolved_report.txt (REQUIRED)

Optional:
  --rx-meta-in   : data/links/rx_meta.jsonl   (per-RXCUI metadata: {rxcui, tty, has_ndc})
  --rx-keys-in   : data/links/rx_link_keys.jsonl  (fallback source for has_ndc if rx_meta not provided)

Scoring (higher is better):
  +100  if tty in {SBD, SCD}
   +60  if has_ndc == True
   -50  if tty in {GPCK, BPCK}
   -80  if tty in {IN, MIN, PIN, DF}
   + 5  if via == 'spl_set_id'  (tiny nudge)
   + 0  otherwise
Tie-breaker: lower numeric RXCUI first; then lexicographic.

Outputs:
  - daily_rx_links_resolved.jsonl: one row per SPL_SET_ID
     {
       "spl_set_id": "...",
       "primary": {"rxcui": "...", "tty": "SCD", "score": 165, "via": "spl_set_id", "has_ndc": true},
       "secondary": [{...}, ...]
     }
  - report with summary metrics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import orjson as _json
    def dumps(obj):
        return _json.dumps(obj).decode()
except Exception:
    def dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

# ------------------------------ CLI -----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--links-in', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report-out', required=True)
    p.add_argument('--rx-meta-in')
    p.add_argument('--rx-keys-in')
    p.add_argument('--progress-every', type=int, default=200000)
    return p.parse_args()

# ---------------------------- Loaders ---------------------------------------

def load_rx_meta(path: Optional[str]) -> Dict[str, Dict]:
    meta = {}
    if not path:
        return meta
    if not os.path.exists(path):
        print(f"WARN: --rx-meta-in not found: {path}", file=sys.stderr)
        return meta
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rxcui = str(row.get('rxcui') or '').strip()
            if not rxcui:
                continue
            tty = (row.get('tty') or '').upper() if isinstance(row.get('tty'), str) else ''
            has_ndc = bool(row.get('has_ndc'))
            ings = row.get('ingredients') if isinstance(row.get('ingredients'), list) else None
            meta[rxcui] = {
                'tty': tty,
                'has_ndc': has_ndc,
                'ingredients': ings,
            }
    print(f"Loaded rx_meta: {len(meta):,} RXCUIs", file=sys.stderr)
    return meta


def infer_has_ndc_from_rx_keys(path: Optional[str]) -> Dict[str, bool]:
    """If rx_meta doesn't exist, we can still infer has_ndc from rx_link_keys."""
    has_ndc = {}
    if not path or not os.path.exists(path):
        if path:
            print(f"WARN: --rx-keys-in not found: {path}", file=sys.stderr)
        return has_ndc
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            rxcui = str(row.get('rxcui') or '').strip()
            if not rxcui:
                continue
            ndcs = row.get('ndc11s') or []
            has_ndc[rxcui] = bool(ndcs)
    print(f"Derived has_ndc from rx_link_keys for {sum(1 for v in has_ndc.values() if v):,} RXCUIs", file=sys.stderr)
    return has_ndc


def load_links_grouped(path: str, progress_every: int=200000) -> Dict[str, List[Tuple[str, str]]]:
    """Return map: spl_set_id -> list of (rxcui, via)."""
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            spl = (row.get('spl_set_id') or '').strip()
            rxcui = str(row.get('rxcui') or '').strip()
            via = (row.get('via') or '').strip() or 'spl_set_id'
            if not spl or not rxcui:
                continue
            groups[spl].append((rxcui, via))
            if progress_every and total % progress_every == 0:
                print(f".. links read {total:,} rows | unique SPLs={len(groups):,}", file=sys.stderr)
    print(f"Loaded links: {total:,} rows → {len(groups):,} SPLs", file=sys.stderr)
    return groups

# -------------------------- Scoring logic -----------------------------------

_TTY_WEIGHT = {
    'SBD': 100,  # Semantic Brand Drug
    'SCD': 100,  # Semantic Clinical Drug
    'GPCK': -50, 'BPCK': -50,  # packs
    'IN': -80, 'MIN': -80, 'PIN': -80,  # ingredient levels
    'DF': -80,  # dose form
}


def score_candidate(rxcui: str, via: str, meta: Dict[str, Dict], has_ndc_map: Dict[str, bool]) -> Tuple[int, Dict]:
    info = meta.get(rxcui, {})
    tty = info.get('tty', '')
    has_ndc = info.get('has_ndc')
    if has_ndc is None:
        has_ndc = has_ndc_map.get(rxcui, False)

    score = 0
    score += _TTY_WEIGHT.get(tty, 0)
    if has_ndc:
        score += 60
    if via == 'spl_set_id':
        score += 5
    return score, {'rxcui': rxcui, 'tty': tty, 'has_ndc': has_ndc, 'via': via, 'score': score}


def rxcui_key_for_tiebreak(rxcui: str):
    # Prefer numeric ordering when possible, fall back to string
    try:
        return (0, int(rxcui))
    except Exception:
        return (1, rxcui)

# ----------------------------- Main -----------------------------------------

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    if os.path.isdir(args.out):
        raise RuntimeError(f"--out path is a directory, not a file: {args.out}")

    # Load metadata
    rx_meta = load_rx_meta(args.rx_meta_in)
    has_ndc_map = infer_has_ndc_from_rx_keys(args.rx_keys_in) if not rx_meta else {}

    # Group raw links by SPL
    groups = load_links_grouped(args.links_in, progress_every=args.progress_every)

    # Resolve per SPL
    written = 0
    prim_ttys = Counter()
    sec_count = []

    with open(args.out, 'w', encoding='utf-8') as w:
        for spl, pairs in groups.items():
            # score all candidates
            enriched = []
            for rxcui, via in pairs:
                s, payload = score_candidate(rxcui, via, rx_meta, has_ndc_map)
                enriched.append(payload)

            # pick best
            enriched.sort(key=lambda d: (-d['score'], rxcui_key_for_tiebreak(d['rxcui'])))
            primary = enriched[0]
            secondary = enriched[1:]
            prim_ttys[primary.get('tty') or ''] += 1
            sec_count.append(len(secondary))

            out_row = {
                'spl_set_id': spl,
                'primary': primary,
                'secondary': secondary,
            }
            w.write(dumps(out_row) + "\n")
            written += 1

    # Report
    total_spls = len(groups)
    avg_secondaries = (sum(sec_count) / total_spls) if total_spls else 0.0
    top_ttys = ', '.join(f"{k or 'UNKNOWN'}={v}" for k, v in prim_ttys.most_common())

    report_lines = [
        f"Resolved SPLs: {total_spls:,}",
        f"Rows written: {written:,}",
        f"Primary TTY distribution: {top_ttys}",
        f"Average secondary count per SPL: {avg_secondaries:.2f}",
    ]
    with open(args.report_out, 'w', encoding='utf-8') as r:
        r.write("\n".join(report_lines) + "\n")
    print("\n".join(report_lines), file=sys.stderr)


if __name__ == '__main__':
    main()
