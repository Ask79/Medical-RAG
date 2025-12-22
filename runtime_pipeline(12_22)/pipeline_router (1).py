from __future__ import annotations
import os, re, json, sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from sentence_transformers import SentenceTransformer

# Local imports
from rag_med.qa_parse import parse_query
from rag_med.router.spl_rxcui import spl_to_rxcui
from rag_med.collector import collect_evidence

# -----------------
# Paths
# -----------------

def _p(key: str, default: str) -> str:
    return os.getenv(key) or default

DM_META = _p("DM_META_PATH", "data/interim/embeddings/dailymed/chunks_v5_clean__BAAI_bge-small-en-v1.meta.jsonl")
DM_IDX  = _p("DM_IDX_PATH",  "data/interim/embeddings/dailymed/chunks_v5_clean__BAAI_bge-small-en-v1.faiss")
DM_EMB  = _p("DM_EMB_PATH",  "data/interim/embeddings/dailymed/chunks_v5_clean__BAAI_bge-small-en-v1.npy")
DM_MASTER = _p("DM_MASTER_PATH", "data/processed/normalized/dailymed/master_final.jsonl")
SN_META = _p("SN_META_PATH", "data/interim/embeddings/snomed/snomed_chunks_master__BAAI_bge-small-en-v1.meta.jsonl")
SN_IDX  = _p("SN_IDX_PATH",  "data/interim/embeddings/snomed/snomed_chunks_master__BAAI_bge-small-en-v1.faiss")
SN_EMB  = _p("SN_EMB_PATH",  "data/interim/embeddings/snomed/snomed_chunks_master__BAAI_bge-small-en-v1.npy")
SN_MASTER = _p("SN_MASTER_PATH", "data/interim/snomed/snomed_chunks_master.jsonl")
SPL_RXCUI_MAP = _p("SPL_RXCUI_PATH", "data/links/daily_rx_links_resolved.jsonl")

# -----------------
# Debug helpers
# -----------------

def _print_paths():
    print("[paths]\n",
          "DM_META =", DM_META, "\n",
          "DM_IDX =", DM_IDX, "\n",
          "DM_EMB =", DM_EMB, "\n",
          "DM_MASTER =", DM_MASTER, "\n",
          "SN_META =", SN_META, "\n",
          "SN_IDX =", SN_IDX, "\n",
          "SN_EMB =", SN_EMB, "\n",
          "SN_MASTER =", SN_MASTER, "\n",
          "SPL_RXCUI_MAP =", SPL_RXCUI_MAP,
          sep="")

# -----------------
# Drug lexicon (from master + meta for robustness)
# -----------------

_UUID_RX = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_STOP = set("""
 the and or for of in to with by on as an a
 is are was were be being been do does did done doing
 what whats what's which who whom whose when where why how tell me about give info information details please pls kindly
 side effects effect adverse indications indication usage use uses contraindication contraindications dose dosing dosage warning warnings boxed black box mechanism moa interaction interactions pregnancy lactation breastfeeding can may take i
""".split())

_NAME_KEYS = [
    "display_name","rxcui_name","drug_name","product_name","proprietary_name","brand_name","generic_name","name","title"
]

def _canon_drug_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(usp|hcl|hydrochloride|tablet[s]?|capsule[s]?|injection|syrup|oral|solution|suspension|extended\s*release|immediate\s*release)\b", " ", s)
    s = re.sub(r"\b(?:mg|mcg|g|ml)\b", " ", s)
    s = re.sub(r"\bor\b", " ", s)
    s = re.sub(r"[()\[\],.;:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_setid(o: Dict[str, Any]) -> Optional[str]:
    for k in ("spl_set_id","SPL_SET_ID","setid","SetId","SETID","set_id","setId","set_id_string","setIdString"):
        v = o.get(k)
        if isinstance(v, str):
            m = _UUID_RX.search(v)
            if m:
                return m.group(0)
    for v in o.values():
        if isinstance(v, str):
            m = _UUID_RX.search(v)
            if m:
                return m.group(0)
    return None

def build_drug_lexicon(dm_meta_path: Optional[str], dm_master_path: Optional[str]) -> Dict[str, str]:
    lex: Dict[str, str] = {}

    # 1) Harvest from MASTER (strongest signal)
    p = dm_master_path or DM_MASTER
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                sid = _find_setid(o)
                if not sid:
                    continue
                for k, v in o.items():
                    if not isinstance(v, str):
                        continue
                    if not any(kw in k.lower() for kw in ("name","title","proprietary","generic","brand","display")):
                        continue
                    key = _canon_drug_key(v)
                    if not key or key in _STOP or len(key) <= 2:
                        continue
                    lex.setdefault(key, sid)

    # 2) Harvest from META (if any helpful names exist)
    p = dm_meta_path or DM_META
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                sid = (o.get("spl_set_id") or o.get("SPL_SET_ID") or "").strip()
                if not sid:
                    continue
                for k in _NAME_KEYS:
                    v = o.get(k)
                    if isinstance(v, str) and v.strip():
                        key = _canon_drug_key(v)
                        if key and key not in _STOP and len(key) > 2:
                            lex.setdefault(key, sid)

    # prune accidental stopwords / tiny tokens
    lex = {k: v for k, v in lex.items() if len(k) > 2 and k not in _STOP and sum(c.isalpha() for c in k) >= 2}
    return lex

# -----------------
# Condition resolver (stub for now)
# -----------------
@dataclass
class ConditionResolution:
    query_text: str
    sctid: Optional[str]
    preferred_term: Optional[str]
    confidence: float
    note: str = ""

_SNOMED_LEX = {
    "arrhythmia": ("698247007", "Cardiac arrhythmia (disorder)", 0.92),
    "amoxicillin": ("372687004", "Amoxicillin (substance)", 0.92),
}

def resolve_condition(text: Optional[str]) -> ConditionResolution:
    q = (text or "").strip()
    if not q:
        return ConditionResolution(query_text="", sctid=None, preferred_term=None, confidence=0.0, note="No condition provided")
    k = q.lower()
    if k in _SNOMED_LEX:
        s, pt, conf = _SNOMED_LEX[k]
        return ConditionResolution(query_text=q, sctid=s, preferred_term=pt, confidence=conf, note="Lexicon-exact")
    return ConditionResolution(query_text=q, sctid=None, preferred_term=None, confidence=0.0, note="Embedding")

# -----------------
# Links
# -----------------
def build_links(drug: Dict[str, Any], condition: ConditionResolution) -> Dict[str, str]:
    links: Dict[str, str] = {}
    spl = (drug.get("spl_set_id") if isinstance(drug, dict) else None) or ""
    rxcui = (drug.get("rxcui") if isinstance(drug, dict) else None) or ""
    if spl:
        links["dailymed"] = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl}"
        links["dailymed_api"] = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{spl}.json"
    if rxcui:
        links["rxnav"] = f"https://rxnav.nlm.nih.gov/search?searchBy=RXCUI&term={rxcui}"
        links["rxnav_api"] = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}.json"
    if condition and condition.sctid:
        links["snomed"] = f"https://browser.ihtsdotools.org/?perspective=full&conceptId1={condition.sctid}"
    return links

# -----------------
# Pipeline
# -----------------
def _encoder() -> SentenceTransformer:
    enc = SentenceTransformer("BAAI/bge-small-en")
    print("[encoder] Loaded model: BAAI/bge-small-en")
    return enc

def _resolve_drug(drug_text: Optional[str], lex: Dict[str, str]) -> Dict[str, Any]:
    q = (drug_text or "").strip()
    if not q:
        return {"query_text": drug_text, "matched_name": None, "spl_set_id": None, "rxcui": None, "confidence": 0.0, "note": ""}
    key = q.lower()
    sid = lex.get(key) or lex.get(_canon_drug_key(key))
    note = "lexicon" if sid else "no-hit"
    rxcui = spl_to_rxcui(sid) if sid else None
    return {"query_text": q, "matched_name": key if sid else None, "spl_set_id": sid, "rxcui": rxcui, "confidence": 1.0 if sid else 0.0, "note": note}

def _cond_for_filter(cond: ConditionResolution) -> str:
    if not cond:
        return ""
    q = (cond.query_text or "").strip()
    pt = (cond.preferred_term or "").strip()
    if cond.confidence >= 0.85 and q and q.lower() in pt.lower():
        return pt
    return q

def run_pipeline(query: str) -> Dict[str, Any]:
    _print_paths()
    enc = _encoder()

    # Build lexicon once per run
    lex = build_drug_lexicon(DM_META, DM_MASTER)

    # Parse
    parsed = parse_query(query, lex)
    intent = parsed.get("intent") or "overview"

    # Resolve entities
    drug_obj = _resolve_drug(parsed.get("drug_text"), lex)
    cond_obj = resolve_condition(parsed.get("condition_text"))

    # Evidence mining (two-stage done inside collector)
    cond_text = _cond_for_filter(cond_obj)
    evidence = collect_evidence(query=query, payload={"intent": intent, "drug": drug_obj}, encoder=enc, top_k=6, condition_text=cond_text)

    # Verdict heuristic (placeholder)
    verdict = None
    if intent in {"interaction", "contraindication"} and not evidence:
        verdict = "insufficient-evidence"

    payload: Dict[str, Any] = {
        "query": query,
        "intent": intent,
        "drug": drug_obj,
        "condition": asdict(cond_obj),
        "graph": {"link_types": [intent], "edges": []},
        "evidence": evidence,
        "sources": {"entity_links": build_links(drug_obj, cond_obj), "evidence_links": [e.get("links", {}).get("dailymed") for e in evidence if e.get("links", {}).get("dailymed")]},
        "verdict": verdict,
    }
    return payload

# -----------------
# CLI
# -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        _print_paths()

    payload = run_pipeline(args.query)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload)

if __name__ == "__main__":
    main()
