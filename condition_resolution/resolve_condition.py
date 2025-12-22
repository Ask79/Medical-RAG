# rag_med/router/resolve_condition.py

from __future__ import annotations

from dataclasses import dataclass
import json, os, re
from typing import Optional, List, Dict, Tuple, Any

@dataclass
class ConditionResolution:
    query_text: Optional[str]
    sctid: Optional[str]
    preferred_term: Optional[str]
    confidence: float
    note: str

__all__ = [
    "ConditionResolution",
    "resolve_condition",
    "build_condition_lexicon_any",
    "build_id_to_pt_map",
]

# ---------- Helpers ----------

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _extract_sctid(obj: dict) -> Optional[str]:
    # direct variants
    for k in ("sctid", "conceptId", "concept_id", "conceptid", "id"):
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return str(int(v))
        if isinstance(v, str) and v.strip():
            return v.strip()
    # derive from doc_id / chunk_id like "snomed:concept:<id>"
    for k in ("doc_id", "chunk_id"):
        v = obj.get(k)
        if isinstance(v, str):
            m = re.search(r"snomed:concept:(\d+)", v)
            if m:
                return m.group(1)
    return None

def _collect_names_from_master_terms(terms: dict) -> List[str]:
    names: List[str] = []
    if not isinstance(terms, dict):
        return names

    # fsn
    fsn = terms.get("fsn")
    if isinstance(fsn, dict):
        t = fsn.get("term")
        if isinstance(t, str) and t.strip():
            names.append(t.strip())

    # pt (preferred term)
    pt = terms.get("pt")
    if isinstance(pt, dict):
        d = pt.get("default")
        if isinstance(d, dict):
            t = d.get("term")
            if isinstance(t, str) and t.strip():
                names.append(t.strip())
        # non-default variants
        for v in pt.values():
            if isinstance(v, dict):
                t = v.get("term")
                if isinstance(t, str) and t.strip():
                    names.append(t.strip())

    # synonyms list
    syns = terms.get("synonyms")
    if isinstance(syns, list):
        for s in syns:
            if isinstance(s, dict):
                t = s.get("term")
                if isinstance(t, str) and t.strip():
                    names.append(t.strip())
    return names

def _collect_names_flat(obj: dict) -> List[str]:
    names: List[str] = []
    for k in ("preferred_term", "preferredTerm", "pt", "term", "fsn", "FSN", "name", "Fully Specified Name"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            names.append(v.strip())
    for k in ("synonyms", "terms_list", "syns"):
        vs = obj.get(k)
        if isinstance(vs, list):
            names.extend([x.strip() for x in vs if isinstance(x, str) and x.strip()])
    return names

def _collect_names_from_chunk_text(obj: dict) -> Tuple[List[str], Optional[str]]:
    """
    Parse chunk-style records with fields:
      section: "core" | "synonyms"
      text:    free text containing lines like:
                 FSN: <name>
                 Synonyms:
                   - <syn1>
                   - <syn2>
    Returns (names, pt_candidate)
    """
    names: List[str] = []
    pt_candidate: Optional[str] = None
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        return names, pt_candidate

    # FSN line
    m = re.search(r"(?im)^\s*FSN:\s*(.+?)\s*$", text)
    if m:
        fsn = m.group(1).strip()
        if fsn:
            names.append(fsn)
            pt_candidate = pt_candidate or fsn

    # Lines after "Synonyms:" with leading '- '
    syn_block = re.split(r"(?im)^\s*Synonyms:\s*$", text)
    if len(syn_block) > 1:
        for line in syn_block[1].splitlines():
            m2 = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if m2:
                nm = m2.group(1).strip()
                if nm:
                    names.append(nm)

    # Also try a simple "PT:" if present
    mpt = re.search(r"(?im)^\s*PT:\s*(.+?)\s*$", text)
    if mpt:
        pt = mpt.group(1).strip()
        if pt:
            names.append(pt)
            pt_candidate = pt

    # de-dup normalized
    seen, out = set(), []
    for n in names:
        k = _norm(n)
        if k and k not in seen:
            seen.add(k)
            out.append(n)
    return out, pt_candidate

# ---------- Semantic tags & gating ----------

# Allowed tags per intent (lowercased)
_ALLOWED_TAGS = {
    "indication": {"disorder", "situation with explicit context"},
    "contraindication": {"disorder", "situation with explicit context"},
    "adverse_effect": {"disorder", "finding"},
    # others default to permissive
}

_TAG_FSN_RE = re.compile(r"\(([^()]+)\)\s*$")  # FSN typically ends with " (disorder)" etc.

def _semantic_tag_from_row(row: dict) -> Optional[str]:
    # Explicit semanticTag
    tag = row.get("semanticTag")
    if isinstance(tag, str) and tag.strip():
        return tag.strip().lower()

    # Nested terms.fsn.term
    terms = row.get("terms")
    if isinstance(terms, dict):
        fsn = terms.get("fsn")
        if isinstance(fsn, dict):
            t = fsn.get("term")
            if isinstance(t, str) and t.strip():
                m = _TAG_FSN_RE.search(t.strip())
                if m:
                    return m.group(1).strip().lower()

    # Flat FSN
    fsn_str = row.get("fsn")
    if isinstance(fsn_str, str) and fsn_str.strip():
        m = _TAG_FSN_RE.search(fsn_str.strip())
        if m:
            return m.group(1).strip().lower()

    # Chunk text: FSN:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        m = re.search(r"(?im)^\s*FSN:\s*(.+?)\s*$", text)
        if m:
            t = m.group(1).strip()
            m2 = _TAG_FSN_RE.search(t)
            if m2:
                return m2.group(1).strip().lower()

    return None

def _is_tag_allowed(intent: Optional[str], tag: Optional[str]) -> bool:
    if not intent:
        return True
    allowed = _ALLOWED_TAGS.get(intent.lower())
    if not allowed:
        return True
    if not tag:
        # If unknown: be conservative for indication/contraindication; permissive for AEs
        return (intent.lower() == "adverse_effect")
    return tag.lower() in allowed

def _prefer_more_specific_name(a: Optional[str], b: Optional[str]) -> str:
    """Heuristic: prefer the longer name as 'more specific'."""
    a = a or ""
    b = b or ""
    return b if len(b) > len(a) else a

# ---------- Lexicon builders ----------

def build_condition_lexicon_any(jsonl_paths: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Build dict[name_lower -> (conceptId, display_name)].
    Works with both:
      • SNOMED ConceptAggregate master (nested 'terms')
      • Chunked records with 'text' + 'section'
    Stops at the first path that yields entries.
    """
    lex: Dict[str, Tuple[str, str]] = {}
    for p in jsonl_paths:
        if not p or not os.path.exists(p):
            continue
        count_before = len(lex)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                cid = _extract_sctid(o)
                if not cid:
                    continue

                names = []
                # 1) nested ConceptAggregate terms
                terms = o.get("terms")
                if isinstance(terms, dict):
                    names.extend(_collect_names_from_master_terms(terms))
                # 2) flat fields
                names.extend(_collect_names_flat(o))
                # 3) chunk-style text
                tx_names, _ = _collect_names_from_chunk_text(o)
                names.extend(tx_names)

                for nm in names:
                    key = _norm(nm)
                    if key and key not in lex:
                        lex[key] = (cid, nm)
        if len(lex) > count_before:
            break
    return lex

def build_id_to_pt_map(jsonl_paths: List[str]) -> Dict[str, str]:
    """
    Map conceptId -> preferred term.
    Priority: terms.pt.default.term > terms.fsn.term > PT in chunk text > FSN in chunk text > first available name.
    """
    id2pt: Dict[str, str] = {}
    for p in jsonl_paths:
        if not p or not os.path.exists(p):
            continue
        count_before = len(id2pt)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                cid = _extract_sctid(o)
                if not cid:
                    continue

                pt_term = None
                # 1) ConceptAggregate nested PT / FSN
                terms = o.get("terms")
                if isinstance(terms, dict):
                    pt = terms.get("pt")
                    if isinstance(pt, dict):
                        d = pt.get("default")
                        if isinstance(d, dict):
                            t = d.get("term")
                            if isinstance(t, str) and t.strip():
                                pt_term = t.strip()
                    if not pt_term:
                        fsn = terms.get("fsn")
                        if isinstance(fsn, dict):
                            t = fsn.get("term")
                            if isinstance(t, str) and t.strip():
                                pt_term = t

                # 2) chunk text: prefer PT:, else FSN:
                if not pt_term:
                    _, pt_from_chunk = _collect_names_from_chunk_text(o)
                    if pt_from_chunk:
                        pt_term = pt_from_chunk

                # 3) last resort: a flat name
                if not pt_term:
                    flat = _collect_names_flat(o)
                    if flat:
                        pt_term = flat[0]

                if pt_term and cid not in id2pt:
                    id2pt[cid] = pt_term
        if len(id2pt) > count_before:
            break
    return id2pt

# ---------- Lexicon lookups ----------

def _lexicon_exact(q: str, lex: dict):
    return lex.get(_norm(q))

def _lexicon_fuzzy(q: str, lex: dict, cutoff: int = 88):
    if not lex:
        return None
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return None
    cand, score, _ = process.extractOne(_norm(q), list(lex.keys()), scorer=fuzz.token_set_ratio) or (None, 0, None)
    if cand and score >= cutoff:
        cid, name = lex[cand]
        return cid, name, score / 100.0
    return None

# ---------- Meta row access ----------

def _read_meta_row(meta_path: str, row_id: int) -> Optional[dict]:
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_id:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None

# ---------- Resolver ----------

def resolve_condition(query_text: str,
                      qvec,
                      snomed_index,
                      snomed_meta_path: str,
                      lexicon: dict,
                      id_to_pt: Optional[Dict[str, str]] = None,
                      debug: bool = False,
                      **kwargs) -> ConditionResolution:
    """
    Resolves a condition concept for the user's query.

    Non-breaking change: 'intent' may be provided via kwargs; if present,
    we apply semantic-tag whitelisting on embedding-based matches.
    """
    intent = (kwargs.get("intent") or "").strip().lower()

    qt = (query_text or "").strip()
    if not qt:
        return ConditionResolution(query_text, None, None, 0.0, "Empty query")

    # ---- 1) Lexicon exact
    hit = _lexicon_exact(qt, lexicon)
    if hit:
        cid, name = hit
        if id_to_pt and id_to_pt.get(cid):
            name = id_to_pt[cid]
        return ConditionResolution(query_text, cid, name, 0.92, "Lexicon-exact")

    # ---- 2) Lexicon fuzzy
    fhit = _lexicon_fuzzy(qt, lexicon, cutoff=88)
    if fhit:
        cid, name, conf = fhit
        if id_to_pt and id_to_pt.get(cid):
            name = id_to_pt[cid]
        return ConditionResolution(query_text, cid, name, max(0.85, conf), "Lexicon-fuzzy")

    # ---- 3) Embedding → nearest meta row(s) → extract ID → PT
    try:
        D, I = snomed_index.search(qvec, 5)
    except Exception:
        return ConditionResolution(query_text, None, None, 0.0, "Index error")

    if I is None or I.size == 0 or I[0][0] < 0:
        return ConditionResolution(query_text, None, None, 0.0, "No hit")

    # Gather top rows and apply semantic-tag gating if intent provided
    candidates: List[Tuple[str, str, float, str]] = []  # (cid, pt, conf, note)
    best_name_for_cid: Dict[str, str] = {}

    width = I.shape[1] if hasattr(I, "shape") else len(I[0])
    for j in range(min(5, width)):
        row_id = int(I[0][j])
        row = _read_meta_row(snomed_meta_path, row_id)
        if not row:
            continue

        cid = _extract_sctid(row)
        if not cid:
            continue

        # Preferred term
        name = (id_to_pt or {}).get(cid)
        if not name:
            # fallback: fsn or flat fields
            name = None
            terms = row.get("terms")
            if isinstance(terms, dict):
                pt = terms.get("pt")
                if isinstance(pt, dict):
                    d = pt.get("default")
                    if isinstance(d, dict) and isinstance(d.get("term"), str):
                        name = d["term"].strip()
                if not name:
                    fsn = terms.get("fsn")
                    if isinstance(fsn, dict) and isinstance(fsn.get("term"), str):
                        name = fsn["term"].strip()
            if not name:
                for k in ("preferred_term", "pt", "term", "fsn", "name"):
                    v = row.get(k)
                    if isinstance(v, str) and v.strip():
                        name = v.strip(); break
            if not name:
                # As a last resort, try chunk text PT/FSN
                tx_names, pt_candidate = _collect_names_from_chunk_text(row)
                name = pt_candidate or (tx_names[0] if tx_names else None)

        # Confidence from distance
        conf = 0.7
        try:
            d0 = float(D[0][j])
            conf = max(0.5, min(0.95, 0.5 + 0.5 * (1.0 / (1.0 + abs(d0)))))
        except Exception:
            pass

        # Semantic tag gating
        tag = _semantic_tag_from_row(row)
        if not _is_tag_allowed(intent, tag):
            if debug:
                print(f"[resolve_condition] filtered by tag: cid={cid} tag={tag} intent={intent}")
            continue

        # Track the most specific label for a cid
        prev = best_name_for_cid.get(cid)
        best_name_for_cid[cid] = _prefer_more_specific_name(prev, name or "") if name else (prev or "")

        note = "Embedding"
        if tag:
            note += f" ({tag})"
        candidates.append((cid, best_name_for_cid[cid], conf, note))

    if not candidates:
        return ConditionResolution(query_text, None, None, 0.0, "No tag-eligible hit" if intent else "No hit")

    # Choose highest confidence; tie-break by longer (more specific) name
    candidates.sort(key=lambda t: (t[2], len(t[1] or "")), reverse=True)
    cid, name, conf, note = candidates[0]
    return ConditionResolution(query_text, cid, name, conf, note)
