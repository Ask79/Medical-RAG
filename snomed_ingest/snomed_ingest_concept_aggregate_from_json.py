#!/usr/bin/env python3
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

# SCT constants
FSN_TYPE_ID       = "900000000000003001"
SYNONYM_TYPE_ID   = "900000000000013009"
ISA_TYPE_ID       = "116680003"
CHAR_INFERRED     = "900000000000011006"
CHAR_STATED       = "900000000000010007"
DEF_FULLY_DEFINED = "900000000000073002"
DEF_PRIMITIVE     = "900000000000074008"

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            yield json.loads(line)

def to_date_key(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s)
    return "".join(ch for ch in s if ch.isdigit())  # handles 'YYYYMMDD' and 'YYYY-MM-DD'

def strip_semantic_tag(term: Optional[str]) -> Optional[str]:
    if term is None:
        return term
    t = term.strip()
    if t.endswith(")") and " (" in t:
        base, _ = t.rsplit(" (", 1)
        return base
    return t

# ------------------ Robust field readers ------------------

# ADDED support for 'snomed_id'/'snomedId' and deriving from 'doc_id: "snomed:concept:<cid>"'
CID_KEYS = ("conceptId", "concept_id", "cid", "sctid", "id", "snomed_id", "snomedId")

def get_cid(r) -> str:
    # flat keys first
    for k in CID_KEYS:
        v = r.get(k)
        if v:
            return str(v).strip()
    # derive from doc_id like "snomed:concept:12345"
    doc_id = r.get("doc_id") or r.get("docId") or ""
    if isinstance(doc_id, str) and doc_id.startswith("snomed:concept:"):
        return doc_id.split("snomed:concept:", 1)[1]
    # nested shapes e.g. {"concept": {...}} or {"data": {...}}
    c = r.get("concept") or r.get("data") or {}
    if isinstance(c, dict):
        for k in CID_KEYS:
            v = c.get(k)
            if v:
                return str(v).strip()
        # nested doc_id fallback too
        nested_doc = c.get("doc_id") or c.get("docId") or ""
        if isinstance(nested_doc, str) and nested_doc.startswith("snomed:concept:"):
            return nested_doc.split("snomed:concept:", 1)[1]
    return ""

def get_active(r) -> bool:
    # prefer explicit active, otherwise accept variations, otherwise default True
    v = r.get("active", None)
    if v is not None:
        if isinstance(v, str):
            vs = v.strip().lower()
            return vs in ("1","true","t","yes","y")
        if isinstance(v, (int, bool)):
            return bool(v)
    v = r.get("isActive", None)
    if v is not None:
        if isinstance(v, str):
            return v.strip().lower() in ("1","true","t","yes","y")
        return bool(v)
    v = r.get("status", None)  # e.g., "active" / "inactive"
    if isinstance(v, str):
        return v.strip().lower() in ("active","current","enabled")
    # default to True if missing (your masters often pre-filtered to active)
    return True

def get_effective_time(r) -> str:
    return str(r.get("effective_time") or r.get("effectiveTime") or r.get("effectiveDate") or r.get("released") or "")

def get_module_id(r) -> str:
    return str(r.get("moduleId") or r.get("module_id") or r.get("module") or "")

def get_def_status_id(r) -> str:
    ds = r.get("definitionStatusId") or r.get("definition_status_id")
    if ds:
        return str(ds)
    # derive from boolean if present
    fd = r.get("is_fully_defined")
    if fd is not None:
        if isinstance(fd, str):
            fdv = fd.strip().lower() in ("1","true","t","yes","y")
        else:
            fdv = bool(fd)
        return DEF_FULLY_DEFINED if fdv else DEF_PRIMITIVE
    return ""

# ------------------ Loaders FROM JSONL ------------------

def load_concepts_from_jsonl(path: Path, include_inactive: bool = False) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for r in read_jsonl(path):
        cid = get_cid(r)
        if not cid:
            continue
        et = get_effective_time(r)
        prev = latest.get(cid)
        if not prev or to_date_key(et) >= to_date_key(prev.get("effective_time","")):
            rec = {
                "active": get_active(r),
                "moduleId": get_module_id(r),
                "definitionStatusId": get_def_status_id(r),
                "effective_time": et,
            }
            latest[cid] = rec
    if not include_inactive:
        latest = {k:v for k,v in latest.items() if v.get("active")}
    return latest

def load_descriptions_from_jsonl(path: Path):
    per_concept: Dict[str, dict] = defaultdict(lambda: {"fsn": None, "synonyms": []})
    for r in read_jsonl(path):
        if not get_active(r):
            continue
        did = str(r.get("id") or r.get("description_id") or "").strip()
        cid = str(r.get("conceptId") or r.get("concept_id") or "").strip()
        if not did or not cid:
            continue
        typeId = str(r.get("typeId") or "")
        term = str(r.get("term") or "")
        lang = str(r.get("languageCode") or r.get("lang") or "en")
        et   = get_effective_time(r)
        rec = {"id": did, "conceptId": cid, "term": term, "lang": lang, "typeId": typeId, "effective_time": et}
        bundle = per_concept[cid]
        if typeId == FSN_TYPE_ID:
            cur = bundle["fsn"]
            if not cur or to_date_key(et) >= to_date_key(cur.get("effective_time","")):
                bundle["fsn"] = rec
        elif typeId == SYNONYM_TYPE_ID:
            bundle["synonyms"].append(rec)
    # de-dupe synonyms by (term, lang) newest
    for cid, bundle in per_concept.items():
        best = {}
        for s in bundle["synonyms"]:
            key = (s["term"], s["lang"])
            cur = best.get(key)
            if not cur or to_date_key(s["effective_time"]) >= to_date_key(cur.get("effective_time","")):
                best[key] = s
        bundle["synonyms"] = list(best.values())
    return per_concept

def load_relationships_from_jsonl(path: Path, characteristic_filter: Optional[str] = CHAR_INFERRED):
    per_concept: Dict[str, dict] = defaultdict(lambda: {"parents": []})
    for r in read_jsonl(path):
        if not get_active(r):
            continue
        # type filter: allow explicit typeId or an 'is_a' boolean convenience flag from your rel master
        if str(r.get("typeId") or "") != ISA_TYPE_ID and not r.get("is_a", False):
            continue
        ctype = str(r.get("characteristicTypeId") or "")
        if characteristic_filter and ctype != characteristic_filter:
            continue
        sid = str(r.get("sourceId") or "").strip()
        did = str(r.get("destinationId") or "").strip()
        if not sid or not did:
            continue
        et = get_effective_time(r)
        rec = {"destinationId": did, "effective_time": et, "moduleId": get_module_id(r)}
        per_concept[sid]["parents"].append(rec)
    # de-dupe by destinationId newest
    for cid, bundle in per_concept.items():
        best = {}
        for p in bundle["parents"]:
            did = p["destinationId"]
            cur = best.get(did)
            if not cur or to_date_key(p["effective_time"]) >= to_date_key(cur.get("effective_time","")):
                best[did] = p
        bundle["parents"] = list(best.values())
    return per_concept

# ------------------ Compose ------------------

def compose_docs(concepts, descs_per_concept, inferred_parents, stated_parents, include_stated):
    ts = now_utc_iso()
    for cid, cmeta in concepts.items():
        descs = descs_per_concept.get(cid, {})
        fsn_obj = None
        if descs.get("fsn"):
            fsn_obj = {"id": descs["fsn"]["id"], "term": descs["fsn"]["term"], "lang": descs["fsn"]["lang"]}
        pt_map = {}
        if fsn_obj:
            pt_map["default"] = {"id": fsn_obj["id"], "term": strip_semantic_tag(fsn_obj["term"])}
        syns = [{"id": s["id"], "term": s["term"], "lang": s["lang"]} for s in descs.get("synonyms", [])]
        inh = inferred_parents.get(cid, {"parents": []})
        parents = inh.get("parents", [])
        parent_ids = sorted({p["destinationId"] for p in parents})
        doc = {
            "doc_id": f"snomed:concept:{cid}",
            "table": "ConceptAggregate",
            "conceptId": str(cid),
            "active": bool(cmeta.get("active", True)),
            "moduleId": str(cmeta.get("moduleId") or ""),
            "definitionStatusId": str(cmeta.get("definitionStatusId") or ""),
            "is_fully_defined": str(cmeta.get("definitionStatusId") or "") == DEF_FULLY_DEFINED,
            "effective_time": cmeta.get("effective_time") or "",
            "terms": {"fsn": fsn_obj, "pt": pt_map, "synonyms": syns},
            "hierarchy": {
                "parents": parents,
                "parent_ids": parent_ids,
                "parent_count": len(parent_ids),
                "characteristic": "inferred",
            },
            "attributes": {"roles": []},
            "source": "snomed_ct",
            "last_ingested": ts,
        }
        if include_stated and stated_parents is not None:
            st = stated_parents.get(cid, {"parents": []})
            s_parents = st.get("parents", [])
            s_ids = sorted({p["destinationId"] for p in s_parents})
            doc["hierarchy"]["stated_parents"] = s_parents
            doc["hierarchy"]["stated_parent_ids"] = s_ids
            doc["hierarchy"]["stated_parent_count"] = len(s_ids)
        tmp = dict(doc); tmp.pop("record_hash", None)
        doc["record_hash"] = hashlib.sha1(
            json.dumps(tmp, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        yield doc

# ------------------ CLI ------------------

def run(args):
    concept_path = Path(args.concept_json)
    desc_path    = Path(args.description_json)
    rel_path     = Path(args.relationship_json)
    out_path     = Path(args.out)

    for p in (concept_path, desc_path, rel_path):
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    print("Loading concepts (from JSONL)...")
    concepts = load_concepts_from_jsonl(concept_path, include_inactive=args.include_inactive)
    print(f"  concepts kept: {len(concepts):,}")

    print("Loading descriptions (from JSONL)...")
    per_concept = load_descriptions_from_jsonl(desc_path)
    print(f"  concepts with descriptions: {len(per_concept):,}")

    print("Loading relationships (inferred is-a, from JSONL)...")
    inferred = load_relationships_from_jsonl(rel_path, characteristic_filter=CHAR_INFERRED)
    print(f"  concepts with inferred parents: {len(inferred):,}")

    stated = None
    if args.include_stated:
        print("Loading relationships (stated is-a, from JSONL)...")
        stated = load_relationships_from_jsonl(rel_path, characteristic_filter=CHAR_STATED)
        print(f"  concepts with stated parents: {len(stated):,}")

    # ---- Fallback: if no concepts survived, synthesize from desc/rel ----
    if not concepts:
        print("No concepts recognized from concept JSONL; synthesizing concept list from descriptions/relationships...")
        all_cids = set(per_concept.keys()) | set(inferred.keys())
        if stated:
            all_cids |= set(stated.keys())
        concepts = {cid: {
            "active": True,                 # assume active; you prefiltered earlier
            "moduleId": "",
            "definitionStatusId": "",
            "effective_time": "",
        } for cid in all_cids}
        print(f"  synthesized concepts: {len(concepts):,}")

    print("Composing documents...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc in compose_docs(concepts, per_concept, inferred, stated, args.include_stated):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n:,} concept documents -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate SNOMED masters (JSONL) into concept-centric JSONL")
    ap.add_argument("--concept-json",      required=True, help="Path to concepts_master.jsonl")
    ap.add_argument("--description-json",  required=True, help="Path to descriptions_master.jsonl")
    ap.add_argument("--relationship-json", required=True, help="Path to relationships_master.jsonl")
    ap.add_argument("--out",               required=True, help="Output JSONL path")
    ap.add_argument("--include-stated",    action="store_true", help="Also include stated parents")
    ap.add_argument("--include-inactive",  action="store_true", help="Include inactive concepts")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
