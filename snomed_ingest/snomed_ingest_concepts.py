import csv, json, sys, glob, hashlib
from pathlib import Path
from datetime import datetime

DEF_PRIMITIVE = "900000000000074008"
DEF_DEFINED   = "900000000000073002"

def to_iso(date_yyyymmdd: str) -> str:
    s = date_yyyymmdd.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s  # leave as-is if unexpected

def rec_hash(d: dict) -> str:
    m = hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
    return m[:16]

def ingest_concepts(input_paths, out_jsonl, max_n=None):
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = active = defined = primitive = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for p in input_paths:
            with Path(p).open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                header = next(r)
                idx = {name:i for i,name in enumerate(header)}
                required = ["id","effectiveTime","active","moduleId","definitionStatusId"]
                for key in required:
                    if key not in idx:
                        raise RuntimeError(f"Missing column '{key}' in {p}")

                for row in r:
                    total += 1
                    concept_id = row[idx["id"]]
                    eff_raw = row[idx["effectiveTime"]]
                    is_active = (row[idx["active"]] == "1")
                    module_id = row[idx["moduleId"]]
                    def_id = row[idx["definitionStatusId"]]
                    def_status = ("defined" if def_id == DEF_DEFINED
                                  else "primitive" if def_id == DEF_PRIMITIVE
                                  else "unknown")

                    if is_active: active += 1
                    if def_status == "defined":   defined += 1
                    if def_status == "primitive": primitive += 1

                    rec = {
                        "doc_id": f"snomed:concept:{concept_id}",
                        "snomed_id": concept_id,
                        "table": "Concept",
                        "effective_time": to_iso(eff_raw),
                        "active": is_active,
                        "module_id": module_id,
                        "definitionStatusId": def_id,
                        "definition_status": def_status,
                        "source": "snomed_ct",
                        "last_ingested": datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    }
                    rec["record_hash"] = rec_hash(rec)

                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if max_n and written >= max_n:
                        break
            if max_n and written >= max_n:
                break

    print(f"[concepts] files={len(input_paths)} rows_in={total} written={written} "
          f"active={active} defined={defined} primitive={primitive}")
    print(f"[done] → {out_jsonl}")

def main():
    if len(sys.argv) < 3:
        print("Usage:\n  python -u src\\rag_med\\ingest\\snomed_ingest_concepts.py <input_glob> <out_jsonl> [--max N]")
        sys.exit(1)

    input_glob = sys.argv[1]
    out_jsonl = sys.argv[2]
    max_n = None
    if len(sys.argv) >= 5 and sys.argv[3] == "--max":
        max_n = int(sys.argv[4])

    paths = sorted(glob.glob(input_glob))
    if not paths:
        print(f"No files match: {input_glob}")
        sys.exit(2)

    ingest_concepts(paths, out_jsonl, max_n)

if __name__ == "__main__":
    main()
