#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, json, os, sys, argparse, unicodedata
from collections import defaultdict
from datetime import datetime

# RXNCONSO columns (pipe-delimited)
#  0 RXCUI | 1 LAT | 2 TS | 3 LUI | 4 STT | 5 SUI | 6 ISPREF | 7 RXAUI | 8 SAUI
#  9 SCUI  |10 SDUI|11 SAB |12 TTY |13 CODE|14 STR |15 SRL    |16 SUPPRESS|17 CVF

# Priority tuned for product-first selection (yours, with tiny clarifications)
TTY_PRIORITY = {
    "SCD": 10, "SBD": 9, "GPCK": 8, "BPCK": 7, "PIN": 6,
    "IN": 5, "BN": 4, "DF": 3, "MIN": 3, "PSN": 3,
    "SY": 2, "TMSY": 1
}
def tty_score(tty: str) -> int:
    return TTY_PRIORITY.get(tty, 1)

# Map TTY to brand_generic + node_kind hints
TTY_BUCKETS = {
    # products
    "SCD": ("generic",  "product"),
    "SBD": ("brand",    "product"),
    "GPCK":(None,       "pack"),
    "BPCK":(None,       "pack"),
    # identity/ingredients/brands/forms
    "IN":  ("ingredient","ingredient"),
    "MIN": ("ingredient","ingredient"),
    "PIN": ("ingredient","ingredient"),
    "BN":  ("brand",    "brand"),
    "DF":  (None,       "dose_form"),
    # names
    "PSN": (None,       None),
    "SY":  (None,       None),
    "TMSY":(None,       None),
}

def brand_generic_from_tty(tty: str):
    return TTY_BUCKETS.get(tty, (None, None))[0]

def node_kind_from_tty(tty: str):
    return TTY_BUCKETS.get(tty, (None, None))[1]

def normalize_name(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s.lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conso", required=True, help="Path to RXNCONSO.RRF")
    ap.add_argument("--out", required=True, help="Output JSONL for master")
    ap.add_argument("--min-syns", type=int, default=0)
    args = ap.parse_args()

    groups = defaultdict(list)

    # Stream RXNCONSO
    with open(args.conso, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row or len(row) < 18:
                continue
            rxcui = row[0]
            sab   = (row[11] or "").upper()
            tty   = (row[12] or "").upper()
            s     = row[14] or ""
            ispref= (row[6] or "").upper()
            suppress = (row[16] or "").upper()

            if not rxcui or not s:
                continue
            if suppress in ("Y","O"):  # dropped/outdated
                continue

            groups[rxcui].append({
                "SAB": sab, "TTY": tty, "STR": s, "ISPREF": ispref
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for rxcui, rows in groups.items():
            if not rows:
                continue

            # Prefer SAB=RXNORM where possible
            rxnorm_rows = [r for r in rows if r["SAB"] == "RXNORM"]
            rank_pool = rxnorm_rows if rxnorm_rows else rows

            # tie-breaker favors higher TTY rank, then ISPREF=Y, then shorter STR
            def rank_key(r):
                return (
                    -tty_score(r["TTY"]),
                    0 if r["ISPREF"] == "Y" else 1,
                    len(r["STR"])
                )
            preferred = sorted(rank_pool, key=rank_key)[0]
            preferred_term = preferred["STR"]
            preferred_tty  = preferred["TTY"]

            # synonyms (dedupe by normalized form)
            seen_norm = set()
            synonyms = []
            sabs = set()
            for r in rows:
                sabs.add(r["SAB"])
                nm = r["STR"]
                norm = normalize_name(nm)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    synonyms.append(nm)

            brand_generic = brand_generic_from_tty(preferred_tty)
            node_kind     = node_kind_from_tty(preferred_tty)

            record = {
                "rxcui": rxcui,
                "preferred_term": preferred_term,
                "tty": preferred_tty,
                "synonyms": synonyms,
                "normalized_name": normalize_name(preferred_term),
                "brand_generic": brand_generic,          # NEW
                "node_kind": node_kind,                  # NEW (helps downstream glue)
                "sab_sources": sorted(sabs),
                "source": "rxnorm",
                "last_updated": now_iso
            }
            # Require RXNORM participation in SABs
            if "RXNORM" not in sabs:
                continue
            if len(synonyms) >= args.min_syns:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} records to {args.out}")

if __name__ == "__main__":
    main()
