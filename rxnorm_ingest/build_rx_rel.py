#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Builds a canonical REL sidecar from RXNREL.RRF.

- CUI<->CUI only (drop AUI edges)
- Default SAB filter: RXNORM (override with --keep-sab)
- Canonicalizes direction onto a small rela set:
    has_ingredient, has_dose_form, has_tradename, isa, has_part (optional)
- Drops SUPPRESS='Y' if requested
- De-dupes edges

Output JSONL per edge:
{
  "src_rxcui": "string",
  "rela": "has_ingredient|has_dose_form|has_tradename|isa|has_part",
  "dst_rxcui": "string",
  "sab": "RXNORM"
}

Usage:
python -u src\rxnorm_ingestion_key\build_rx_rel.py ^
  --in data\raw\RxNorm_full_prescribe_08042025\rrf\RXNREL.RRF ^
  --out data\processed\normalized\rxnorm\rel\rel_edges_v2.jsonl ^
  --dedupe --drop-suppress --progress-every 200000
"""

import argparse, io, json, os, sys
from typing import Iterable, Optional, Set, Tuple

IDX_RXCUI1 = 0
IDX_STYPE1 = 2
IDX_REL    = 3
IDX_RXCUI2 = 4
IDX_STYPE2 = 6
IDX_RELA   = 7
IDX_SAB    = 10
IDX_SUP    = 14

CANON = {
    # keep as-is
    "HAS_INGREDIENT": ("has_ingredient",  1),  # src->dst
    "HAS_DOSE_FORM":  ("has_dose_form",   1),
    "HAS_TRADENAME":  ("has_tradename",   1),
    "ISA":            ("isa",             1),

    # flip direction
    "INGREDIENT_OF":  ("has_ingredient", -1),  # src<-dst
    "DOSE_FORM_OF":   ("has_dose_form",  -1),
    "TRADENAME_OF":   ("has_tradename",  -1),
    "INVERSE_ISA":    ("isa",            -1),

    # optional structure
    "CONSISTS_OF":    ("has_part",        1),
    "CONSTITUTES":    ("has_part",       -1)
}

def _open_text(path: str) -> io.TextIOBase:
    return open(path, 'r', encoding='utf-8', errors='replace', newline='')

def _split_rrf_line(line: str) -> list:
    parts = line.rstrip('\n\r').split('|')
    if parts and parts[-1] == '':
        parts = parts[:-1]
    return parts

def _get(parts: list, idx: int) -> str:
    try:
        return parts[idx]
    except IndexError:
        return ''

def _is_num(s: str) -> bool:
    return s.isdigit() if s else False

def iter_rel_edges(path: str,
                   keep_sab: Optional[Set[str]] = None,
                   drop_suppress: bool = False,
                   dedupe: bool = False,
                   progress_every: int = 0) -> Iterable[dict]:
    seen = set()
    total_in = 0
    total_out = 0

    with _open_text(path) as f:
        for line in f:
            total_in += 1
            if progress_every and total_in % progress_every == 0:
                print(f".. read {total_in:,} lines, emitted {total_out:,}", file=sys.stderr)

            if not line.strip():
                continue
            p = _split_rrf_line(line)

            # require CUI<->CUI rows
            st1 = _get(p, IDX_STYPE1).upper()
            st2 = _get(p, IDX_STYPE2).upper()
            if not (st1 == "CUI" and st2 == "CUI"):
                continue

            rxcui1 = _get(p, IDX_RXCUI1).strip()
            rxcui2 = _get(p, IDX_RXCUI2).strip()
            if not (_is_num(rxcui1) and _is_num(rxcui2)):
                continue

            sab = _get(p, IDX_SAB).strip().upper() or ""
            if keep_sab and sab not in keep_sab:
                continue

            suppress = (_get(p, IDX_SUP) or "").upper()
            if drop_suppress and suppress.startswith("Y"):
                continue

            rela_raw = (_get(p, IDX_RELA) or "").upper()
            if not rela_raw:
                # ignore bare REL when RELA is missing; these are underspecified
                continue
            if rela_raw not in CANON:
                # skip RELAs outside our schema
                continue

            rela_canon, direction = CANON[rela_raw]
            if direction == 1:
                src, dst = rxcui1, rxcui2
            else:
                src, dst = rxcui2, rxcui1

            row = {"src_rxcui": src, "rela": rela_canon, "dst_rxcui": dst, "sab": sab}

            if dedupe:
                key = (row["src_rxcui"], row["rela"], row["dst_rxcui"], row["sab"])
                if key in seen:
                    continue
                seen.add(key)

            yield row
            total_out += 1

def main():
    ap = argparse.ArgumentParser(description="Build canonical REL edges from RXNREL.RRF")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--keep-sab", type=str, default="RXNORM", help="Comma-separated SAB whitelist (default=RXNORM)")
    ap.add_argument("--drop-suppress", action="store_true")
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--progress-every", type=int, default=0)
    args = ap.parse_args()

    keep_sab = {s.strip().upper() for s in (args.keep_sab or "").split(",") if s.strip()} or None

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    n = 0
    with open(args.out_path, "w", encoding="utf-8") as w:
        for row in iter_rel_edges(args.in_path,
                                  keep_sab=keep_sab,
                                  drop_suppress=bool(args.drop_suppress),
                                  dedupe=bool(args.dedupe),
                                  progress_every=int(args.progress_every or 0)):
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n:,} rows → {args.out_path}")

if __name__ == "__main__":
    main()
