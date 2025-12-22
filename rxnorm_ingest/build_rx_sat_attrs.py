# -*- coding: utf-8 -*-
"""
build_rx_sat_attrs.py

Purpose
-------
Parse RXNSAT.RRF and emit a long/row JSONL sidecar with concept-related
attributes (routes, dose forms, strengths, ATC, UNII, SPL IDs, etc.)
in the same JSON schema you've been using:

  {"rxcui","atn","atv","sab","suppress"}

Key features
------------
- Fully respects --keep-atns (no hidden allowlist).
- Optional --keep-sabs filter.
- Optional --normalize-ndc11 (convert NDC/NDC11 values to 11-digit NDC11).
- Optional --emit-ndc to ALSO keep original NDC rows while normalizing.
- Optional --drop-suppress (skip rows with SUPPRESS != "N").
- Optional --dedupe with stable (rxcui, atn, atv, sab) key (or --dedupe-global
  if you want to ignore SAB when deduping).
- Streams line-by-line, reports progress.

Column assumptions
------------------
RXNSAT.RRF pipe-delimited columns (based on the samples you provided):

  0: RXCUI
  1: LUI             (often empty in RxNorm SAT)
  2: SUI             (often empty in RxNorm SAT)
  3: RXAUI           (e.g., 6344362)
  4: STYPE           (e.g., "AUI")
  5: CODE            (e.g., "49035-268")
  6: ATUI            (may be empty)
  7: SATUI           (may be empty)
  8: ATN             (attribute name, e.g., "SPL_SET_ID", "NDC", "RXN_STRENGTH")
  9: SAB             (e.g., "MTHSPL", "RXNORM")
 10: ATV             (attribute value)
 11: SUPPRESS        ("N" or "Y")
 12: CVF             (e.g., "4096")
 13+: (ignore extra)

If your drop has different positions, tweak the indices below.

Usage
-----
Example (attrs-rich pass):

  python -u src/rxnorm_ingestion_key/build_rx_sat_attrs.py ^
    --in data/raw/RxNorm_full_prescribe_08042025/rrf/RXNSAT.RRF ^
    --out data/processed/normalized/rxnorm/sat/rx_sat_attrs_v2.jsonl ^
    --keep-atns NDC,NDC11,SPL_SET_ID,SPL_ID,UNII,ROUTE,RXN_ROUTE,RXN_DOSE_FORM,STRENGTH,RXN_STRENGTH,RXN_AVAILABLE_STRENGTH,ATC,ATC_CODE,RXTERM_FORM,LABELER,LABEL_TYPE,MARKETING_CATEGORY,MARKETING_STATUS,MARKETING_EFFECTIVE_TIME_LOW,MARKETING_EFFECTIVE_TIME_HIGH,DM_SPL_ID,ANDA,BLA,SCORE,SHAPE,SHAPETEXT,SIZE,COLOR,COLORTEXT ^
    --normalize-ndc11 --emit-ndc --dedupe --drop-suppress --progress-every 200000
"""

import argparse
import json
import os
import re
import sys
from typing import Iterable, Optional, Set, Tuple

DIGITS_RE = re.compile(r"\D+")

def to_ndc11(value: str) -> Optional[str]:
    """
    Convert an NDC (10- or 11-digit in any dashed format) into a strict 11-digit string.
    Strategy:
      - Strip non-digits.
      - If 11 digits => accept as-is.
      - If 10 digits => attempt zero-padding heuristics by common segment layouts.
        Since we don't have segment context here, we assume 5-3-2 pattern is common;
        as a conservative fallback, left-pad with a leading '0'.
      - Otherwise => return None (un-normalizable).
    """
    if not value:
        return None
    digits = DIGITS_RE.sub("", value)
    if len(digits) == 11:
        return digits
    if len(digits) == 10:
        # Conservative fallback: left-pad to 11
        return "0" + digits
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build concept-attrs SAT JSONL from RXNSAT.RRF")
    p.add_argument("--in", dest="in_path", required=True, help="Path to RXNSAT.RRF")
    p.add_argument("--out", dest="out_path", required=True, help="Path to output .jsonl")
    p.add_argument("--keep-atns", type=str, default="",
                   help="Comma-separated ATNs to keep. If omitted, keep ALL ATNs.")
    p.add_argument("--keep-sabs", type=str, default="",
                   help="Comma-separated SABs to keep (e.g., RXNORM,MTHSPL). If omitted, keep ALL SABs.")
    p.add_argument("--normalize-ndc11", action="store_true",
                   help="Normalize NDC and NDC11 values to 11-digit NDC11; rename NDC to NDC11.")
    p.add_argument("--emit-ndc", action="store_true",
                   help="When --normalize-ndc11 is on, ALSO emit the original NDC rows (alongside normalized NDC11).")
    p.add_argument("--drop-suppress", action="store_true",
                   help="Drop any rows with SUPPRESS != 'N'.")
    p.add_argument("--dedupe", action="store_true",
                   help="Deduplicate by (rxcui, atn, atv, sab).")
    p.add_argument("--dedupe-global", action="store_true",
                   help="Deduplicate by (rxcui, atn, atv) ignoring SAB.")
    p.add_argument("--progress-every", type=int, default=200000,
                   help="Report progress every N input lines.")
    return p.parse_args()


def iter_rxnsat_lines(fp: Iterable[str]) -> Iterable[Tuple[str, str, str, str, str]]:
    """
    Yield tuples (rxcui, atn, atv, sab, suppress) from RXNSAT lines using the index map above.
    Guard against short lines; skip if mandatory fields are missing.
    """
    for raw in fp:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        parts = raw.split("|")
        # Ensure we have at least up to index 12
        if len(parts) < 13:
            continue

        rxcui    = parts[0].strip()
        atn      = parts[8].strip()
        sab      = parts[9].strip()
        atv      = parts[10].strip()
        suppress = parts[11].strip()

        # Minimal sanity checks
        if not rxcui or not atn or not sab:
            continue

        yield (rxcui, atn, atv, sab, suppress)


def main():
    args = parse_args()

    keep_atns: Optional[Set[str]] = None
    if args.keep_atns:
        keep_atns = {a.strip() for a in args.keep_atns.split(",") if a.strip()}

    keep_sabs: Optional[Set[str]] = None
    if args.keep_sabs:
        keep_sabs = {s.strip() for s in args.keep_sabs.split(",") if s.strip()}

    if args.dedupe and args.dedupe_global:
        print("WARNING: --dedupe and --dedupe-global both set. Using --dedupe-global.", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    total = 0
    written = 0
    seen: Set[Tuple[str, str, str, Optional[str]]] = set() if (args.dedupe or args.dedupe_global) else None

    with open(args.in_path, "r", encoding="utf-8") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:

        for (rxcui, atn, atv, sab, suppress) in iter_rxnsat_lines(fin):
            total += 1

            if args.progress_every and (total % args.progress_every == 0):
                print(f".. read {total:,} lines, emitted {written:,}", file=sys.stderr, flush=True)

            # Filter SUPPRESS
            if args.drop_suppress and suppress != "N":
                continue

            # Filter by SAB
            if keep_sabs is not None and sab not in keep_sabs:
                continue

            # Filter by ATN
            if keep_atns is not None and atn not in keep_atns:
                # If we are going to normalize NDC to NDC11, it's still an ATN we might want to transform.
                # But if atn isn't NDC/NDC11 and it's not in the keep list, skip now.
                if not (args.normalize_ndc11 and atn in {"NDC", "NDC11"}):
                    continue

            # Normalize NDC / NDC11 if requested
            rows_to_emit = []
            if args.normalize_ndc11 and atn in {"NDC", "NDC11"}:
                ndc11 = to_ndc11(atv)
                if ndc11:
                    # Emit normalized NDC11
                    rows_to_emit.append(("NDC11", ndc11))
                    # Optionally ALSO emit the original (NDC or NDC11) row
                    if args.emit_ndc and atn == "NDC":
                        rows_to_emit.append(("NDC", atv))
                else:
                    # Could not normalize; either emit raw NDC if allowed, or skip
                    if args.emit_ndc and atn == "NDC":
                        rows_to_emit.append(("NDC", atv))
                    else:
                        # Skip un-normalizable NDC values
                        pass
            else:
                # Regular attribute, pass-through
                rows_to_emit.append((atn, atv))

            # Write rows, applying dedupe if needed
            for (emit_atn, emit_atv) in rows_to_emit:
                # If a keep list exists, make sure the *emitted* ATN is allowed.
                if keep_atns is not None and emit_atn not in keep_atns:
                    continue

                if seen is not None:
                    key = (rxcui, emit_atn, emit_atv, None if args.dedupe_global else sab)
                    if key in seen:
                        continue
                    seen.add(key)

                rec = {
                    "rxcui": rxcui,
                    "atn": emit_atn,
                    "atv": emit_atv,
                    "sab": sab,
                    "suppress": suppress
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written:,} rows → {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
