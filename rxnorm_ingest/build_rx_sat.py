#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Builds a compact SAT sidecar from RXNSAT.RRF.

Inputs
------
--in  : path to RXNSAT.RRF
--out : path to output JSONL (one row per attribute)

Optional
--------
--sample N             : write only first N valid rows (0 = all)
--keep-atns A,B,C      : only keep attributes where ATN ∈ {A,B,C}
--keep-sab S1,S2       : only keep rows where SAB ∈ {S1,S2}
--drop-suppress        : drop rows where SUPPRESS flag is 'Y' (case-insensitive)
--dedupe               : de-duplicate by (rxcui, atn, atv, sab)
--normalize-ndc11      : attempt to normalize NDC/NDC11 to 11 digits (best-effort)
--only-valid-ndc11     : when normalizing, keep only rows that end up with exactly 11 digits
--progress-every N     : print a progress line every N input rows

Output JSONL schema (sidecar)
-----------------------------
{
  "rxcui": "string",          # RXCUI owner of the attribute
  "atn": "string",            # attribute name (uppercased)
  "atv": "string",            # attribute value (trimmed, original case)
  "sab": "string",            # source vocabulary (uppercased)
  "suppress": "string|null"   # SUPPRESS flag from RRF (often 'N' or 'Y')
}

Common ATNs you might care about
--------------------------------
Link keys:    NDC, NDC11, SPL_SET_ID, SPL_ID, UNII
Descriptors:  ROUTE, RXN_ROUTE, RXN_DOSE_FORM, DOSE_FORM, STRENGTH, BRAND_NAME
Classing:     ATC, ATC_CODE, VA_CLASS
Misc:         TTY (term type), RXN_* attributes depending on release

RXNSAT.RRF fields (no header; tolerant parsing)
----------------------------------------------
0 RXCUI | 1 LUI | 2 SUI | 3 RXAUI | 4 STYPE | 5 CODE | 6 ATUI | 7 SATUI |
8 ATN | 9 SAB | 10 ATV | 11 SUPPRESS | 12 CVF

Usage (Windows CMD)
-------------------
python -u src\rag_med\rxnorm\build_rx_sat.py ^
  --in data\raw\RxNorm_full_prescribe_08042025\rrf\RXNSAT.RRF ^
  --out data\processed\rxnorm\sat\rx_sat_v1.jsonl ^
  --dedupe --drop-suppress --progress-every 200000

# Keep only the attributes you need for links and descriptors
python -u src\rag_med\rxnorm\build_rx_sat.py ^
  --in data\raw\RxNorm_full_prescribe_08042025\rrf\RXNSAT.RRF ^
  --out data\processed\rxnorm\sat\rx_sat_v1.jsonl ^
  --keep-atns NDC,NDC11,SPL_SET_ID,SPL_ID,UNII,ROUTE,RXN_ROUTE,RXN_DOSE_FORM,STRENGTH,ATC,ATC_CODE ^
  --normalize-ndc11 --dedupe --drop-suppress --progress-every 200000
"""

import argparse
import io
import json
import os
import re
import sys
from typing import Iterable, Optional, Set, Tuple

_DIGITS_RE = re.compile(r"\d+")
_NON_DIGIT_RE = re.compile(r"\D+")


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


def _is_numeric_str(s: str) -> bool:
    return s.isdigit() if s else False


def _normalize_atn(s: str) -> str:
    return (s or '').strip().upper()


def _normalize_sab(s: str) -> str:
    return (s or '').strip().upper()


def _trim_atv(s: str) -> str:
    return (s or '').strip()

# --- NDC normalization helpers ---

def _normalize_ndc11_value(value: str) -> str:
    """Best-effort normalization to 11-digit NDC.
    Rules:
      - If already 11 digits: keep as-is (digits only)
      - If hyphenated 10-digit: try FDA segment padding (5-4-1, 4-4-2, 5-3-2)
      - If plain 10-digit: return digits (10) — caller may drop if --only-valid-ndc11
      - Otherwise: return digits-only string (may be != 10/11)
    """
    raw = value or ''
    if not raw:
        return ''
    # Extract digits and hyphenated segments if any
    if '-' in raw:
        segs = raw.split('-')
        if len(segs) == 3 and all(segs):
            a, b, c = segs
            if a.isdigit() and b.isdigit() and c.isdigit():
                # Try patterns
                # 4-4-2 -> pad a to 5
                if len(a) == 4 and len(b) == 4 and len(c) == 2:
                    return f"{a.zfill(5)}{b}{c}"
                # 5-3-2 -> pad b to 4
                if len(a) == 5 and len(b) == 3 and len(c) == 2:
                    return f"{a}{b.zfill(4)}{c}"
                # 5-4-1 -> pad c to 2
                if len(a) == 5 and len(b) == 4 and len(c) == 1:
                    return f"{a}{b}{c.zfill(2)}"
                # If already 5-4-2
                if len(a) == 5 and len(b) == 4 and len(c) == 2:
                    return f"{a}{b}{c}"
        # Fallback: strip hyphens to digits
        digits = _NON_DIGIT_RE.sub('', raw)
        if len(digits) == 11:
            return digits
        return digits
    # No hyphens: just digits
    digits = _NON_DIGIT_RE.sub('', raw)
    return digits


def iter_sat_rows(rrf_path: str,
                  keep_atns: Optional[Set[str]] = None,
                  keep_sab: Optional[Set[str]] = None,
                  drop_suppress: bool = False,
                  dedupe: bool = False,
                  normalize_ndc11: bool = False,
                  only_valid_ndc11: bool = False,
                  progress_every: int = 0,
                  sample: int = 0) -> Iterable[dict]:
    seen: Set[Tuple[str, str, str, str]] = set()
    total_in = 0
    total_out = 0

    with _open_text(rrf_path) as f:
        for line in f:
            total_in += 1
            if progress_every and (total_in % progress_every == 0):
                print(f".. read {total_in:,} lines, emitted {total_out:,}", file=sys.stderr)

            if not line.strip():
                continue
            parts = _split_rrf_line(line)

            rxcui = _get(parts, 0).strip()
            atn = _normalize_atn(_get(parts, 8))
            sab = _normalize_sab(_get(parts, 9))
            atv = _trim_atv(_get(parts, 10))
            suppress = _get(parts, 11).strip() or None

            if not _is_numeric_str(rxcui):
                continue
            if keep_atns and atn not in keep_atns:
                continue
            if keep_sab and sab not in keep_sab:
                continue
            if drop_suppress and (suppress or '').upper().startswith('Y'):
                continue

            # Best-effort NDC normalization
            if normalize_ndc11 and atn in {"NDC", "NDC11"}:
                normalized = _normalize_ndc11_value(atv)
                if only_valid_ndc11 and len(normalized) != 11:
                    continue
                atn = "NDC11"  # collapse to NDC11
                atv = normalized

            row = {
                'rxcui': rxcui,
                'atn': atn,
                'atv': atv,
                'sab': sab,
                'suppress': suppress,
            }

            if dedupe:
                key = (row['rxcui'], row['atn'], row['atv'], row['sab'])
                if key in seen:
                    continue
                seen.add(key)

            yield row
            total_out += 1
            if sample and total_out >= sample:
                break


def main():
    ap = argparse.ArgumentParser(description="Build SAT sidecar from RXNSAT.RRF")
    ap.add_argument('--in', dest='in_path', required=True, help='Path to RXNSAT.RRF')
    ap.add_argument('--out', dest='out_path', required=True, help='Path to output JSONL')
    ap.add_argument('--sample', type=int, default=0, help='Write only first N valid rows (0=all)')
    ap.add_argument('--keep-atns', type=str, default='', help='Comma-separated whitelist of ATN values (uppercased internally)')
    ap.add_argument('--keep-sab', type=str, default='', help='Comma-separated whitelist of SAB values (uppercased)')
    ap.add_argument('--drop-suppress', action='store_true', help="Drop rows with SUPPRESS='Y'")
    ap.add_argument('--dedupe', action='store_true', help='De-duplicate by (rxcui,atn,atv,sab)')
    ap.add_argument('--normalize-ndc11', action='store_true', help='Normalize NDC/NDC11 to 11-digit best-effort')
    ap.add_argument('--only-valid-ndc11', action='store_true', help='When normalizing, keep only rows with exactly 11 digits')
    ap.add_argument('--progress-every', type=int, default=0, help='Print progress every N lines read')

    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    sample = int(args.sample or 0)
    keep_atns = {s.strip().upper() for s in args.keep_atns.split(',') if s.strip()} or None
    keep_sab = {s.strip().upper() for s in args.keep_sab.split(',') if s.strip()} or None
    drop_suppress = bool(args.drop_suppress)
    dedupe = bool(args.dedupe)
    normalize_ndc11 = bool(args.normalize_ndc11)
    only_valid_ndc11 = bool(args.only_valid_ndc11)
    progress_every = int(args.progress_every or 0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for row in iter_sat_rows(
            in_path,
            keep_atns=keep_atns,
            keep_sab=keep_sab,
            drop_suppress=drop_suppress,
            dedupe=dedupe,
            normalize_ndc11=normalize_ndc11,
            only_valid_ndc11=only_valid_ndc11,
            progress_every=progress_every,
            sample=sample,
        ):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total:,} rows → {out_path}")


if __name__ == '__main__':
    main()
