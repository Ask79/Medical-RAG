# File: src/rag_med/linkers/build_rx_meta_from_conso.py
"""
Build a compact RxNorm metadata file from RXNCONSO.RRF (and optionally RXNREL.RRF)
that the resolver can use to prioritize concept types.

Outputs JSONL rows like:
  {"rxcui":"12345","tty":"SCD","has_ndc":true}

Inputs:
  --conso-in   : path to RXNCONSO.RRF (pipe-delimited RRF)
  --out        : path to write rx_meta.jsonl
Optional:
  --rx-keys-in : rx_link_keys.jsonl to set has_ndc = True for RXCUIs w/ any NDCs

Notes:
- Correct RXNCONSO field mapping (RxNorm spec):
  RXCUI(0)|LAT(1)|TS(2)|LUI(3)|STT(4)|SUI(5)|ISPREF(6)|RXAUI(7)|SAUI(8)|SCUI(9)|SDUI(10)|SAB(11)|TTY(12)|CODE(13)|STR(14)|SUPPRESS(15)|CVF(16)
- We select a single representative TTY per RXCUI using a priority ordering.
"""

import argparse
import os
import sys
import json
from collections import defaultdict
from typing import Dict, Optional

try:
    import orjson as _json
    def dumps(obj):
        return _json.dumps(obj).decode()
except Exception:
    def dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--conso-in', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rx-keys-in')
    p.add_argument('--progress-every', type=int, default=500000)
    return p.parse_args()

# TTY priority (choose a representative when multiple TTYs exist for same RXCUI)
TTY_PRIORITY = {
    'SBD': 100,  # Semantic Brand Drug
    'SCD': 95,   # Semantic Clinical Drug
    'BN':  80,   # Brand Name
    'IN':  60, 'MIN': 60, 'PIN': 60,  # ingredient levels
    'DF':  40,   # dose form
    'GPCK': 30, 'BPCK': 30,           # packs
}

def better_tty(cur: Optional[str], new: str) -> bool:
    if cur is None:
        return True
    return TTY_PRIORITY.get(new, 0) > TTY_PRIORITY.get(cur, 0)

def load_has_ndc_from_rx_keys(path: Optional[str]) -> Dict[str, bool]:
    has_ndc = {}
    if not path or not os.path.exists(path):
        if path:
            print(f"WARN: --rx-keys-in not found: {path}", file=sys.stderr)
        return has_ndc
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
            has_ndc[rxcui] = bool(ndcs)
    print(f"Loaded has_ndc for {sum(1 for v in has_ndc.values() if v):,} RXCUIs from rx_link_keys", file=sys.stderr)
    return has_ndc

def load_conso_best_tty(path: str, progress_every: int = 500000) -> Dict[str, str]:
    """
    Parse RXNCONSO.RRF and choose one representative TTY per RXCUI,
    using the correct column indices: RXCUI=0, SAB=11, TTY=12.
    Prefer higher-priority TTYs per TTY_PRIORITY.
    """
    best_tty: Dict[str, str] = {}
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            parts = line.rstrip('\n').split('|')
            if len(parts) < 13:  # need at least up to TTY (index 12)
                continue
            rxcui = parts[0].strip()   # RXCUI
            sab   = parts[11].strip()  # SAB (not used for filtering here)
            tty   = parts[12].strip().upper()  # TTY
            if not rxcui or not tty:
                continue

            if rxcui in best_tty:
                if better_tty(best_tty[rxcui], tty):
                    best_tty[rxcui] = tty
            else:
                best_tty[rxcui] = tty

            if progress_every and total % progress_every == 0:
                print(f".. CONSO read {total:,} lines | RXCUIs seen={len(best_tty):,}", file=sys.stderr)

    print(f"Loaded TTY for {len(best_tty):,} RXCUIs from RXNCONSO", file=sys.stderr)
    return best_tty

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.isdir(args.out):
        raise RuntimeError(f"--out path is a directory, not a file: {args.out}")

    best_tty = load_conso_best_tty(args.conso_in, progress_every=args.progress_every)
    has_ndc_map = load_has_ndc_from_rx_keys(args.rx_keys_in)

    written = 0
    with open(args.out, 'w', encoding='utf-8') as w:
        for rxcui, tty in best_tty.items():
            row = {
                'rxcui': rxcui,
                'tty': tty,
                'has_ndc': bool(has_ndc_map.get(rxcui, False)),
            }
            w.write(dumps(row) + "\n")
            written += 1
    print(f"Wrote {written:,} rx_meta rows → {args.out}", file=sys.stderr)

if __name__ == '__main__':
    main()
