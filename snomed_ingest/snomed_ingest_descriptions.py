import csv, json, sys, glob, hashlib, unicodedata
from pathlib import Path
from datetime import datetime, timezone

# ⬇️ allow very long fields (Python default ~128 KB)
csv.field_size_limit(2**31 - 1)

TYPEID_FSN      = "900000000000003001"
TYPEID_SYNONYM  = "900000000000013009"
TYPEID_TEXTDEF  = "900000000000550004"

CASE_ID_MAP = {
    "900000000000448009": "entire_term_case_insensitive",
    "900000000000020002": "initial_character_case_insensitive",
    "900000000000017005": "entire_term_case_sensitive",
}
CASE_LABEL_MAP = {
    "ENTIRE_TERM_CASE_INSENSITIVE": "entire_term_case_insensitive",
    "INITIAL_CHARACTER_CASE_INSENSITIVE": "initial_character_case_insensitive",
    "ENTIRE_TERM_CASE_SENSITIVE": "entire_term_case_sensitive",
    "CASE_INSENSITIVE": "entire_term_case_insensitive",
    "CASE_SENSITIVE": "entire_term_case_sensitive",
}

def to_iso(s: str) -> str:
    s = (s or "").strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s

def clean_term(t: str) -> str:
    t = (t or "").replace("\u00A0", " ").strip()
    return " ".join(t.split())

def fold_ascii_lower(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return t.lower()

def rec_hash(d: dict) -> str:
    m = hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return m[:16]

def main():
    if len(sys.argv) < 3:
        print("Usage: python -u src\\snomed_ingest\\snomed_ingest_descriptions.py <input_glob> <out_jsonl> [--lang en] [--only-active 0|1] [--max N]")
        sys.exit(1)

    input_glob = sys.argv[1]
    out_jsonl  = sys.argv[2]
    lang = "en"
    only_active = False
    max_n = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--lang":
            lang = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "--only-active":
            only_active = sys.argv[i+1] in ("1","true","True"); i += 2
        elif sys.argv[i] == "--max":
            max_n = int(sys.argv[i+1]); i += 2
        else:
            i += 1

    paths = sorted(glob.glob(input_glob))
    if not paths:
        print(f"No files match: {input_glob}"); sys.exit(2)

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for p in paths:
            # newline="" is recommended by csv module; utf-8-sig handles BOM safely
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                r = csv.reader(f, delimiter="\t")
                header = next(r)
                idx = {h:i for i,h in enumerate(header)}
                req = ["id","effectiveTime","active","moduleId","conceptId","languageCode","typeId","term","caseSignificanceId"]
                for k in req:
                    if k not in idx:
                        raise RuntimeError(f"Missing column {k} in {p}")

                for row in r:
                    if lang and row[idx["languageCode"]] != lang:
                        continue
                    if only_active and row[idx["active"]] != "1":
                        continue

                    desc_id   = row[idx["id"]]
                    conceptId = row[idx["conceptId"]]
                    typeId    = row[idx["typeId"]]
                    term      = clean_term(row[idx["term"]])
                    cs_raw    = row[idx["caseSignificanceId"]].strip()

                    cs_label = CASE_ID_MAP.get(cs_raw) or CASE_LABEL_MAP.get(cs_raw.upper(), "unknown")
                    kind = ("fsn" if typeId == TYPEID_FSN else
                            "synonym" if typeId == TYPEID_SYNONYM else
                            "textdef" if typeId == TYPEID_TEXTDEF else
                            "other")

                    rec = {
                        "doc_id": f"snomed:desc:{desc_id}",
                        "table": "Description",
                        "description_id": desc_id,
                        "conceptId": conceptId,
                        "effective_time": to_iso(row[idx["effectiveTime"]]),
                        "active": row[idx["active"]] == "1",
                        "module_id": row[idx["moduleId"]],
                        "languageCode": row[idx["languageCode"]],
                        "typeId": typeId,
                        "kind": kind,
                        "term": term,
                        "caseSignificanceId": cs_raw,
                        "case_significance": cs_label,
                        "term_folded": fold_ascii_lower(term),
                        "source": "snomed_ct",
                        "last_ingested": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
                    }
                    rec["record_hash"] = rec_hash(rec)

                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if max_n and written >= max_n: break
            if max_n and written >= max_n: break

    print(f"[descriptions] files={len(paths)} written={written} -> {out_jsonl}")

if __name__ == "__main__":
    main()
