#!/usr/bin/env python3
"""
SNOMED CT: Relationship Snapshot → JSONL

Input  : One or more tab-delimited "sct2_Relationship_Snapshot_*.txt" files
Output : JSONL with one object per relationship row

Required columns in input (tab-separated):
  id  effectiveTime  active  moduleId  sourceId  destinationId
  relationshipGroup  typeId  characteristicTypeId  modifierId

Notes:
- All SNOMED identifiers are written as *strings* (do not coerce to ints).
- Dates are converted from YYYYMMDD to YYYY-MM-DD.
- `active` is converted to boolean.
- Adds convenience flag `is_a` (True when typeId == "116680003").
"""

from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Set

# Be generous for very long fields (defensive; relationships are small but harmless)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # On some platforms sys.maxsize is too large for the C long; fall back.
    csv.field_size_limit(2_147_483_647)


IS_A_TYPE_ID = "116680003"


def yyyymmdd_to_iso(d: str) -> str:
    d = (d or "").strip()
    if not d or len(d) != 8 or not d.isdigit():
        return d  # leave as-is if unexpected
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def as_bool01(v: str) -> bool:
    return str(v).strip() == "1"


def short_hash(obj: dict) -> str:
    """
    Stable short content hash for change-detection/debugging.
    8-byte hex from BLAKE2s over a canonicalized JSON payload.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=8).hexdigest()


def iter_relationship_rows(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = {
            "id",
            "effectiveTime",
            "active",
            "moduleId",
            "sourceId",
            "destinationId",
            "relationshipGroup",
            "typeId",
            "characteristicTypeId",
            "modifierId",
        }
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing required columns: {sorted(missing)}")

        for row in reader:
            yield row


def transform_row(row: dict) -> dict:
    # Keep all IDs as strings; relationshipGroup as int if possible
    rel_id = str(row["id"]).strip()
    out = {
        "doc_id": f"snomed:rel:{rel_id}",
        "table": "Relationship",
        "relationship_id": rel_id,
        "effective_time": yyyymmdd_to_iso(row.get("effectiveTime", "")),
        "active": as_bool01(row.get("active", "")),
        "module_id": str(row.get("moduleId", "")).strip(),
        "sourceId": str(row.get("sourceId", "")).strip(),
        "destinationId": str(row.get("destinationId", "")).strip(),
        "relationshipGroup": int(row.get("relationshipGroup", "0") or 0),
        "typeId": str(row.get("typeId", "")).strip(),
        "characteristicTypeId": str(row.get("characteristicTypeId", "")).strip(),
        "modifierId": str(row.get("modifierId", "")).strip(),
    }
    out["is_a"] = (out["typeId"] == IS_A_TYPE_ID)

    # Provenance
    out["source"] = "snomed_ct"
    out["last_ingested"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    out["record_hash"] = short_hash(
        {
            k: out[k]
            for k in (
                "relationship_id",
                "effective_time",
                "active",
                "module_id",
                "sourceId",
                "destinationId",
                "relationshipGroup",
                "typeId",
                "characteristicTypeId",
                "modifierId",
            )
        }
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ingest SNOMED Relationship Snapshot (TSV) to JSONL."
    )
    ap.add_argument("input_glob", help="Glob for input files, e.g. data/.../sct2_Relationship_Snapshot_*.txt")
    ap.add_argument("out_jsonl", help="Output JSONL file")
    ap.add_argument("--only-active", type=int, choices=(0, 1), default=1, help="Keep only active=1 rows (default 1)")
    ap.add_argument(
        "--type",
        dest="type_ids",
        action="append",
        default=None,
        help="Filter to one or more typeId values (repeat flag for multiple). Example: --type 116680003",
    )
    ap.add_argument("--max", type=int, default=0, help="Stop after N rows (for smoke tests)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        print(f"No files matched: {args.input_glob}", file=sys.stderr)
        sys.exit(2)

    type_filter: Optional[Set[str]] = set(map(str, args.type_ids)) if args.type_ids else None
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as w:
        for p in paths:
            print(p)
            for row in iter_relationship_rows(p):
                total_in += 1

                if args.only_active == 1 and str(row.get("active", "")).strip() != "1":
                    continue
                if type_filter and str(row.get("typeId", "")).strip() not in type_filter:
                    continue

                out_obj = transform_row(row)
                w.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                total_out += 1

                if args.max and total_out >= args.max:
                    break
            if args.max and total_out >= args.max:
                break

    print(f"In rows  : {total_in}")
    print(f"Out rows : {total_out}")
    if args.only_active == 1:
        print("Filter   : only active")
    if type_filter:
        print(f"Filter   : typeId in {sorted(type_filter)}")


if __name__ == "__main__":
    main()
