import re
from dataclasses import dataclass

@dataclass
class Route:
    intent: str           # "drug", "condition", "safety"
    drug_text: str|None   # raw drug mention (if any)
    cond_text: str|None   # raw condition mention (if any)

# very light keyword heuristics; you’ll ML-ify later
_DRUG_SYNONYMS = r"(drug|med|medicine|medication|tablet|capsule|dose|formulation|brand|generic)"
_COND_SYNONYMS = r"(condition|disease|diagnos(?:is|e[sd])|symptom|syndrome)"
_SAFETY_SYNONYMS = r"(contraindicat|warning|precaution|avoid|risk|black box|interaction|interact)"
_TREATMENT_SYNONYMS = r"(treat|therapy|management|best (?:med|drug)|first[- ]line)"

# cheap entity pattern (single-token or simple multiword); you’ll replace w/ proper NER later
_ENTITY = r"[A-Za-z][A-Za-z0-9\-\s]{1,60}"

def route_query(q: str) -> Route:
    s = q.lower().strip()

    # Safety-centric has priority (contains safety words)
    if re.search(_SAFETY_SYNONYMS, s):
        # try to pull a drug and a condition-ish phrase near the safety term
        drug = _extract_drug_like(q)
        cond = _extract_condition_like(q)
        return Route("safety", drug, cond)

    # Condition-centric (“what treats Z?” etc.)
    if re.search(_TREATMENT_SYNONYMS, s) or re.search(r"what (?:helps|treats) ", s):
        cond = _extract_condition_like(q)
        return Route("condition", None, cond if cond else _fallback_condition_guess(q))

    # Drug-centric (default if a drug-like token present)
    drug = _extract_drug_like(q)
    if drug:
        return Route("drug", drug, None)

    # Else assume condition-centric if a condition-ish phrase is present
    cond = _extract_condition_like(q)
    if cond:
        return Route("condition", None, cond)

    # Fallback: if nothing detected, treat like drug Q&A (pipeline can still handle generic)
    return Route("drug", None, None)

def _extract_drug_like(q: str) -> str|None:
    # prioritizes tokens next to common drug Q words (side effects, dose, etc.)
    m = re.search(rf"(?:side effect|dose|dosing|indicat(?:ion|ed)|{_DRUG_SYNONYMS})\s+of\s+({_ENTITY})", q, flags=re.I)
    if m: return m.group(1).strip(" ?.,")
    m = re.search(rf"\b({_ENTITY})\b(?:\s+(?:side effects|dose|dosing|indication|contraindication))", q, flags=re.I)
    if m: return m.group(1).strip(" ?.,")
    # plain fallback: first capitalized token sequence
    m = re.search(r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,3})\b", q)
    return m.group(1).strip(" ?.,") if m else None

def _extract_condition_like(q: str) -> str|None:
    m = re.search(rf"(?:for|with|in)\s+({_ENTITY})", q, flags=re.I)
    if m: return m.group(1).strip(" ?.,")
    # common constructions
    m = re.search(r"treat(?:s|ment|ing)?\s+(?:of|for)\s+([A-Za-z0-9\-\s]{2,60})", q, flags=re.I)
    if m: return m.group(1).strip(" ?.,")
    return None

def _fallback_condition_guess(q: str) -> str|None:
    # last-resort: longest lowercase phrase at the end
    m = re.search(r"(?:for|with)\s+([a-z0-9\-\s]{2,60})$", q)
    return m.group(1).strip(" ?.,") if m else None
