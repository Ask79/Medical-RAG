# File: src/rag_med/linkers/build_daily_rx_links.py

import argparse
import json
import sys
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

try:
    import orjson as _json
    def dumps(obj):
        return _json.dumps(obj).decode()
except Exception:
    def dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

from ndc_utils import normalize_ndc11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dailymed-in', required=True)
    p.add_argument('--rx-keys-in', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report-dir', required=True)
    p.add_argument('--progress-every', type=int, default=50000)
    return p.parse_args()


def load_rx_keys(path: str):
    """Load inverted maps: SPL_SET_ID -> {RXCUI}, NDC11 -> {RXCUI}."""
    spl_to_rxcui = defaultdict(set)
    ndc_to_rxcui = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rxcui = str(row.get('rxcui') or '').strip()
            if not rxcui:
                continue
            for spl in row.get('spl_set_ids', []) or []:
                if isinstance(spl, str) and spl.strip():
                    spl_to_rxcui[spl.strip()].add(rxcui)
            for ndc in row.get('ndc11s', []) or []:
                ndc_norm = normalize_ndc11(ndc)
                if ndc_norm:
                    ndc_to_rxcui[ndc_norm].add(rxcui)
    return spl_to_rxcui, ndc_to_rxcui


def iter_dm_records(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


# ---------- Robust key extraction (recursive) ----------

_SPL_KEY_CANDIDATES = {
    "spl_set_id", "dm_spl_id", "splsetid", "spl_setid", "set_id", "splset_id"
}

def _iter_items(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (keypath, value) for dicts/lists scalars recursively (limited depth)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield (key, v)
            # recurse a couple levels deep to catch common nestings
            if isinstance(v, (dict, list, tuple)):
                for kv in _iter_items(v, key):
                    yield kv
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield (key, v)
            if isinstance(v, (dict, list, tuple)):
                for kv in _iter_items(v, key):
                    yield kv
    else:
        yield (prefix, obj)


def _first_spl_value(row: Dict[str, Any]) -> str:
    """Find first plausible SPL_SET_ID/DM_SPL_ID anywhere in the row."""
    for keypath, val in _iter_items(row):
        leaf_key = keypath.split(".")[-1].lower().replace("-", "_")
        if leaf_key in _SPL_KEY_CANDIDATES:
            if isinstance(val, str) and val.strip():
                return val.strip()
            # Sometimes stored as single-element list
            if isinstance(val, list) and val and isinstance(val[0], str):
                s = val[0].strip()
                if s:
                    return s
    return ""


def _collect_ndcs(row: Dict[str, Any]) -> List[str]:
    """Collect all NDC-like fields anywhere whose key contains 'ndc' (case-insensitive)."""
    ndcs: List[str] = []
    for keypath, val in _iter_items(row):
        leaf = keypath.split(".")[-1].lower()
        if "ndc" not in leaf:
            continue
        if isinstance(val, str):
            n = normalize_ndc11(val)
            if n:
                ndcs.append(n)
        elif isinstance(val, (list, tuple)):
            for x in val:
                if isinstance(x, str):
                    n = normalize_ndc11(x)
                    if n:
                        ndcs.append(n)
    # dedupe but keep order
    seen = set()
    out = []
    for n in ndcs:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def extract_dm_keys(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (SPL_SET_ID-like, [NDC11s])."""
    spl = _first_spl_value(row)
    ndcs = _collect_ndcs(row)
    return spl, ndcs


# ---------- Main ----------

def main():
    args = parse_args()
    spl_to_rxcui, ndc_to_rxcui = load_rx_keys(args.rx_keys_in)

    # Ensure output/report dirs exist and that --out is not a directory
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    if os.path.isdir(args.out):
        raise RuntimeError(f"--out path is a directory, not a file: {args.out}")

    out_path = args.out
    written = 0
    dm_total = dm_linked_any = dm_via_spl = dm_via_ndc = dm_conflicts = 0

    # light debug sampler
    debug_shown = 0

    with open(out_path, 'w', encoding='utf-8') as w:
        for dm_row in iter_dm_records(args.dailymed_in):
            dm_total += 1
            spl, ndcs = extract_dm_keys(dm_row)

            if debug_shown < 3:
                print(f"[debug] row#{dm_total} extracted SPL='{spl}' NDCs={ndcs[:3]}{'...' if len(ndcs)>3 else ''}", file=sys.stderr)
                debug_shown += 1

            links: Set[Tuple[str, str]] = set()

            # 1) Prefer SPL
            if spl:
                rxs = spl_to_rxcui.get(spl)
                if rxs:
                    for rxcui in rxs:
                        links.add((rxcui, 'spl_set_id'))

            # 2) Fallback via NDC
            if not links and ndcs:
                for ndc in ndcs:
                    rxs = ndc_to_rxcui.get(ndc)
                    if rxs:
                        for rxcui in rxs:
                            links.add((rxcui, 'ndc11'))

            if links:
                dm_linked_any += 1
                if any(via == 'spl_set_id' for _, via in links):
                    dm_via_spl += 1
                if any(via == 'ndc11' for _, via in links):
                    dm_via_ndc += 1
                if len({r for r, _ in links}) > 1:
                    dm_conflicts += 1
                for rxcui, via in links:
                    rec = {'spl_set_id': spl or None, 'rxcui': rxcui, 'via': via}
                    w.write(dumps(rec) + "\n")
                    written += 1

            if args.progress_every and dm_total % args.progress_every == 0:
                print(
                    f".. DM read {dm_total:,} | linked={dm_linked_any:,} "
                    f"(via SPL={dm_via_spl:,}, via NDC={dm_via_ndc:,})",
                    file=sys.stderr,
                )

    if dm_total and not dm_linked_any:
        print(
            "WARN: 0 DailyMed rows linked. Check SPL/NDC presence in your master; see [debug] lines above for extracted values.",
            file=sys.stderr,
        )

    report = [
        f"DailyMed records read: {dm_total:,}",
        f"Linked to >=1 RXCUI: {dm_linked_any:,}",
        f" - via SPL_SET_ID: {dm_via_spl:,}",
        f" - via NDC11: {dm_via_ndc:,}",
        f"Conflicts: {dm_conflicts:,}",
        f"Link rows written: {written:,}",
    ]
    rep_path = args.report_dir.rstrip('/\\') + '/daily_rx_links_report.txt'
    with open(rep_path, 'w', encoding='utf-8') as r:
        r.write("\n".join(report) + "\n")
    print("\n".join(report), file=sys.stderr)


if __name__ == '__main__':
    main()
