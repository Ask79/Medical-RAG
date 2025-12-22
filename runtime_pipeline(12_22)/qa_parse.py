# rag_med/qa_parse.py
from __future__ import annotations
import re
from typing import Iterable, Optional, Dict

# Lightweight English stopwords so we never treat these as drug names
_STOP = {
    "a","an","the","and","or","of","for","to","in","on","with","without","from","by",
    "is","are","was","were","be","being","been","do","does","did","done","doing",
    "what","whats","what's","which","who","whom","whose","when","where","why","how",
    "tell","me","about","give","info","information","details","please","pls","kindly",
    "side","effects","effect","adverse","indications","indication","usage","use","uses",
    "contraindication","contraindications","dose","dosing","dosage","warning","warnings",
    "boxed","black","box","mechanism","moa","interaction","interactions","pregnancy",
    "lactation","breastfeeding","can","may","take","i"
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*", re.I)

# -----------------------------
# Intent detection (richer)
# -----------------------------
_INTENT_PATTERNS = [
    (r"\b(indications?(?:\s*&\s*usage)?|indicated\s+for|uses?)\b", "indication"),
    (r"\b(side\s*effects?|adverse\s*effects?|aes?)\b", "adverse_effect"),
    (r"\bcontraindicat(?:ed|ion|ions)?\b|\bavoid\s+(?:use|using)\b|\bnot\s+(?:to\s+)?use\b", "contraindication"),
    (r"\b(dose|dosage|dosing)\b", "dosage"),
    (r"\b(black\s*box|boxed|warnings?|precautions?)\b", "warning"),
    (r"\b(pregnancy|lactation|breast\s*feeding|breastfeeding)\b", "pregnancy"),
    (r"\b(mechanism(?:\s*of\s*action)?|moa)\b", "mechanism"),
]

# Interaction: “can I take X with Y”, “X with Y”, “use X with Y”
_INTERACTION_PATTERNS = [
    r"\b(?:can|may)?\s*(?:i\s+)?(?:take|use)\s+(.+?)\s+with\s+(.+?)\??$",
    r"\b([A-Z][A-Za-z0-9\-']+(?:\s+[A-Z][A-Za-z0-9\-']+){0,3})\s+with\s+([A-Z][A-Za-z0-9\-']+(?:\s+[A-Z][A-Za-z0-9\-']+){0,3})\b",
    r"\b([a-z][a-z0-9\-']+(?:\s+[a-z0-9\-']+){0,3})\s+with\s+([a-z][a-z0-9\-']+(?:\s+[a-z0-9\-']+){0,3})\b",
]


def detect_intent(query: str) -> str:
    q = query.strip()
    # Interaction first (it’s easy to miss otherwise)
    for pat in _INTERACTION_PATTERNS:
        if re.search(pat, q, re.I):
            return "interaction"
    ql = q.lower()
    for pat, intent in _INTENT_PATTERNS:
        if re.search(pat, ql):
            return intent
    if re.match(r"^\s*what(?:'s|\s+is)\b", ql) or re.match(r"^\s*tell\s+me\s+about\b", ql):
        return "overview"
    return "overview"

# -----------------------------
# Drug lexicon helpers (optional input)
# -----------------------------

def _names_view(drug_lex: Iterable[str] | Dict[str, str] | None) -> set[str]:
    if not drug_lex:
        return set()
    try:
        keys = drug_lex.keys()
    except AttributeError:
        keys = drug_lex
    return {str(k).lower() for k in keys}


def _strip_prefixes(q: str) -> str:
    s = q.strip()
    s = re.sub(r"^\s*what(?:'s|\s+is)?\s+", "", s, flags=re.I)
    s = re.sub(r"^\s*tell\s+me\s+about\s+", "", s, flags=re.I)
    s = re.sub(r"^\s*give\s+me\s+(?:the\s+)?(?:info|information)\s+(?:on|about)\s+", "", s, flags=re.I)
    s = re.sub(r"^\s*can\s+i\s+", "", s, flags=re.I)
    return s.strip(" ?!.,;:").strip()


def _longest_ngram_in_lex(core: str, names: set[str], max_n: int = 6) -> Optional[str]:
    toks = _TOKEN_RE.findall(core.lower())
    max_n = min(max_n, len(toks))
    for n in range(max_n, 0, -1):
        for i in range(len(toks) - n + 1):
            phrase = " ".join(toks[i:i + n])
            # skip pure-stopword phrases
            if phrase in _STOP or all(t in _STOP for t in toks[i:i+n]):
                continue
            if phrase in names:
                return phrase
    return None


def _capitalized_fallback(core: str, names: set[str]) -> Optional[str]:
    caps = re.findall(r"\b[A-Z][A-Za-z0-9\-']+(?:\s+[A-Z][A-Za-z0-9\-']+){0,3}\b", core)
    for cand in caps:
        if cand.lower() in names and cand.lower() not in _STOP:
            return cand
    return None


def _any_token_in_lex(core: str, names: set[str]) -> Optional[str]:
    for t in _TOKEN_RE.findall(core.lower()):
        if t in _STOP:
            continue
        if t in names:
            return t
    return None


def _best_drug_match(core: str, drug_lex: Iterable[str] | Dict[str, str] | None) -> Optional[str]:
    names = _names_view(drug_lex)
    if names:
        cand = (
            _longest_ngram_in_lex(core, names)
            or _capitalized_fallback(core, names)
            or _any_token_in_lex(core, names)
        )
        if cand and cand.lower() in _STOP:
            cand = None
        return cand
    caps = re.findall(r"\b([A-Z][A-Za-z0-9\-']+)\b", core)
    return caps[0] if caps else None

# -----------------------------
# Condition extraction helpers
# -----------------------------

def _span_after_for_in_with(q: str) -> Optional[str]:
    m = re.search(r"(?i)\b(?:for|in|with)\s+([A-Za-z][A-Za-z0-9 \-']{2,})", q)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip(" ?!.,;:'\"")).strip()
    return None


# -----------------------------
# Main
# -----------------------------

def parse_query(query: str, drug_lex: Iterable[str] | Dict[str, str] | None = None) -> Dict[str, Optional[str]]:
    """Return a dict with keys: intent, drug_text, condition_text."""
    q = query or ""
    intent = detect_intent(q)
    core = _strip_prefixes(q)

    # Interaction
    if intent == "interaction":
        for pat in _INTERACTION_PATTERNS:
            m = re.search(pat, q, re.I)
            if not m:
                continue
            g1, g2 = m.group(1), m.group(2)
            left, right = (g1 or "").strip(), (g2 or "").strip()
            drug_left = _best_drug_match(left, drug_lex)
            drug_right = _best_drug_match(right, drug_lex)
            if drug_left and not drug_right:
                return {"intent": intent, "drug_text": drug_left, "condition_text": right}
            if drug_right and not drug_left:
                return {"intent": intent, "drug_text": drug_right, "condition_text": left}
            # both or neither: keep literals to avoid bad SNOMED guesses later
            return {"intent": intent, "drug_text": drug_left or left, "condition_text": drug_right or right}

    # Adverse effects
    if intent == "adverse_effect":
        m = re.search(r"(?i)\b(?:side\s*effects?|adverse\s*effects?)\s+of\s+(.+)$", core)
        if m:
            drug = _best_drug_match(m.group(1), drug_lex)
            return {"intent": intent, "drug_text": drug or m.group(1).strip(), "condition_text": None}
        drug = _best_drug_match(core, drug_lex)
        return {"intent": intent, "drug_text": drug, "condition_text": None}

    # Contraindication: pick a drug lex hit anywhere, condition is span after in/with/for
    if intent == "contraindication":
        drug = _best_drug_match(core, drug_lex)
        cond = _span_after_for_in_with(q)
        return {"intent": intent, "drug_text": drug, "condition_text": cond}

    # Indication: special-case "indications ... for X" to set X as drug
    if intent == "indication":
        m = re.search(r"(?i)indications?(?:\s*&\s*usage)?\s+for\s+(.+)$", q)
        if m:
            x = m.group(1).strip(" ?!.,;:")
            drug = _best_drug_match(x, drug_lex) or x
            return {"intent": intent, "drug_text": drug, "condition_text": None}
        # otherwise, try generic parse
        drug = _best_drug_match(core, drug_lex)
        cond = _span_after_for_in_with(q)
        return {"intent": intent, "drug_text": drug, "condition_text": cond}

    # Others
    drug = _best_drug_match(core, drug_lex)
    cond = _span_after_for_in_with(q)
    return {"intent": intent, "drug_text": drug, "condition_text": cond}
