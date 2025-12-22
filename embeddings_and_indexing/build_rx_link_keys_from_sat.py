# File: src/rag_med/linkers/build_rx_link_keys_from_sat.py

import argparse
import json
import sys
import os
from collections import defaultdict
from typing import Dict, Set, Any, Iterable

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
    p.add_argument('--sat-in', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--progress-every', type=int, default=200000)
    return p.parse_args()


# ---- helpers ---------------------------------------------------------------

_SPL_KEYS = ('SPL_SET_ID', 'DM_SPL_ID', 'spl_set_id', 'dm_spl_id')
_NDC_KEYS = ('NDC11', 'NDC', 'ndc11', 'ndc11s', 'ndc_list', 'ndcs', 'ndc_code')

def _coerce_iter(val: Any) -> Iterable:
    """Return an iterable of values from str/list/tuple/None."""
    if val is None:
        return ()
    if isinstance(val, (list, tuple)):
        return val
    return (val,)

def _lower_keys(d: Any):
    if isinstance(d, dict):
        return { (k.lower() if isinstance(k, str) else k): v for k, v in d.items() }
    return d

def _add_spl(spl_set, val):
    if isinstance(val, str):
        s = val.strip()
        if s:
            spl_set.add(s)

def _add_ndc(ndc_set, val):
    ndc_norm = normalize_ndc11(val)
    if ndc_norm:
        ndc_set.add(ndc_norm)


# ---- main -----------------------------------------------------------------

def main():
    args = parse_args()

    # Guard: ensure parent dir exists and out is not a directory
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(args.out):
        raise RuntimeError(f"--out path is a directory, not a file: {args.out}")

    spl_by_rxcui: Dict[str, Set[str]] = defaultdict(set)
    ndc_by_rxcui: Dict[str, Set[str]] = defaultdict(set)

    # counters for debugging
    total = 0
    agg_dict_rows = 0
    agg_list_rows = 0
    raw_rows = 0
    rxcui_missing = 0
    spl_added = 0
    ndc_added = 0

    with open(args.sat_in, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            # tolerate rxcui/RXCUI/int
            rxcui = str(row.get('rxcui') or row.get('RXCUI') or '').strip()
            if not rxcui:
                rxcui_missing += 1
                continue

            # 1) Aggregated dict style: {"rxcui":..., "attrs": {...}}
            attrs = row.get('attrs')
            if isinstance(attrs, dict):
                agg_dict_rows += 1
                # SPLs
                for k in _SPL_KEYS:
                    for v in _coerce_iter(attrs.get(k)):
                        before = len(spl_by_rxcui[rxcui])
                        _add_spl(spl_by_rxcui[rxcui], v)
                        spl_added += (len(spl_by_rxcui[rxcui]) - before)
                # NDCs
                # accept any of the ndc keys, keep first one that exists if there are multiple variants
                picked = None
                for k in _NDC_KEYS:
                    if k in attrs:
                        picked = attrs.get(k)
                        break
                if picked is not None:
                    for v in _coerce_iter(picked):
                        before = len(ndc_by_rxcui[rxcui])
                        _add_ndc(ndc_by_rxcui[rxcui], v)
                        ndc_added += (len(ndc_by_rxcui[rxcui]) - before)
                continue

            # 2) Aggregated list style: {"rxcui":..., "attrs": [{"ATN":..., "ATV":...}, ...]}
            if isinstance(attrs, list):
                agg_list_rows += 1
                for a in attrs:
                    if not isinstance(a, dict):
                        continue
                    al = _lower_keys(a)
                    atn = str(al.get('ATN') or al.get('atn') or al.get('attribute_name') or al.get('name') or '').upper()
                    atv = al.get('ATV') or al.get('atv') or al.get('attribute_value') or al.get('value')
                    if atn in ('SPL_SET_ID', 'DM_SPL_ID'):
                        before = len(spl_by_rxcui[rxcui])
                        _add_spl(spl_by_rxcui[rxcui], atv if isinstance(atv, str) else str(atv))
                        spl_added += (len(spl_by_rxcui[rxcui]) - before)
                    elif atn in ('NDC', 'NDC11'):
                        before = len(ndc_by_rxcui[rxcui])
                        _add_ndc(ndc_by_rxcui[rxcui], atv if isinstance(atv, str) else str(atv))
                        ndc_added += (len(ndc_by_rxcui[rxcui]) - before)
                continue

            # 3) Raw row style: {"rxcui":..., "ATN":..., "ATV":...} (case/alias tolerant)
            rl = _lower_keys(row)
            atn = str(rl.get('ATN') or rl.get('atn') or rl.get('attribute_name') or rl.get('name') or '').upper()
            atv = rl.get('ATV') or rl.get('atv') or rl.get('attribute_value') or rl.get('value')
            if atn or atv:
                raw_rows += 1
                if atn in ('SPL_SET_ID', 'DM_SPL_ID') and isinstance(atv, str):
                    before = len(spl_by_rxcui[rxcui])
                    _add_spl(spl_by_rxcui[rxcui], atv)
                    spl_added += (len(spl_by_rxcui[rxcui]) - before)
                elif atn in ('NDC', 'NDC11') and atv:
                    before = len(ndc_by_rxcui[rxcui])
                    _add_ndc(ndc_by_rxcui[rxcui], atv if isinstance(atv, str) else str(atv))
                    ndc_added += (len(ndc_by_rxcui[rxcui]) - before)

            if args.progress_every and total % args.progress_every == 0:
                print(f".. SAT read {total:,} lines | RXCUIs_seen={len(spl_by_rxcui) + len(ndc_by_rxcui):,} | "
                      f"agg_dict={agg_dict_rows:,} agg_list={agg_list_rows:,} raw={raw_rows:,} | "
                      f"spl_added={spl_added:,} ndc_added={ndc_added:,}", file=sys.stderr)

    # Emit
    written = 0
    with open(args.out, 'w', encoding='utf-8') as w:
        for rxcui in sorted(set(list(spl_by_rxcui.keys()) + list(ndc_by_rxcui.keys())),
                            key=lambda x: int(x) if x.isdigit() else x):
            rec = {
                'rxcui': rxcui,
                'spl_set_ids': sorted(spl_by_rxcui.get(rxcui, set())),
                'ndc11s': sorted(ndc_by_rxcui.get(rxcui, set())),
            }
            w.write(dumps(rec) + "\n")
            written += 1

    # Summary
    print(
        "\n".join([
            f"Wrote {written:,} RXCUI link-key rows → {args.out}",
            f"SAT lines read: {total:,}",
            f"Rows w/ aggregated attrs (dict): {agg_dict_rows:,}",
            f"Rows w/ aggregated attrs (list): {agg_list_rows:,}",
            f"Raw rows (ATN/ATV found): {raw_rows:,}",
            f"RXCUIs missing in rows: {rxcui_missing:,}",
            f"SPL links added: {spl_added:,}",
            f"NDC links added: {ndc_added:,}",
        ]),
        file=sys.stderr
    )


if __name__ == '__main__':
    main()
