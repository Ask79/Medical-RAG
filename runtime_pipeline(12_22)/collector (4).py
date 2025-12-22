from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional, Iterable, Tuple

# -----------------------------
# Cosine
# -----------------------------
def _cos(a, b) -> float:
    try:
        import numpy as np  # type: ignore
        a = (a[0] if isinstance(a, (list, tuple)) and len(a) == 1 else a)
        b = (b[0] if isinstance(b, (list, tuple)) and len(b) == 1 else b)
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a) + 1e-9; nb = np.linalg.norm(b) + 1e-9
        return float((a @ b) / (na * nb))
    except Exception:
        return 0.0

# -----------------------------
# Paths & helpers
# -----------------------------
def _dm_master_path() -> Optional[str]:
    return os.getenv("DM_MASTER_PATH") or "data/processed/normalized/dailymed/master_final.jsonl"

_UUID_RX = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

SECTION_KEYS = ["section","section_name","Section","SECTION","heading","title","sectionTitle"]
TEXT_KEYS    = ["text","chunk","content","raw_text","section_text","body","value","paragraph","sentences","line","lines","children"]

# -----------------------------
# Intent → preferred sections (expanded)
# -----------------------------
_DEF_SECTIONS = {
    "adverse_effect": [
        "adverse reactions","adverse events","side effects","most common adverse reactions",
        "undesirable effects"
    ],
    "interaction": [
        "drug interactions","interactions","drug-drug interactions","interactions with other drugs",
        "interactions and contraindications","interactions/contraindications"
    ],
    "contraindication": ["contraindications","contraindication"],
    "indication": ["indications and usage","indications","uses","therapeutic indications"],
    "dosage": ["dosage and administration","recommended dosage","recommended dosing schedule","dosage"],
    "warning": ["boxed warning","black box warning","warnings and precautions","warnings","precautions"],
    "mechanism": ["mechanism of action"],
    "pregnancy": ["pregnancy","use in specific populations"],
}

# -----------------------------
# Synonyms & intent cues
# -----------------------------
_SYNONYMS = {
    "maoi": [
        "maoi","maois","mao inhibitor","mao inhibitors",
        "monoamine oxidase inhibitor","monoamine oxidase inhibitors",
        "monoamine-oxidase inhibitor","monoamine-oxidase inhibitors",
        "within 14 days","within fourteen days","concomitant use",
        "serotonin syndrome","do not use with maoi",
    ],
    "interaction": [
        "bleeding","hemorrhage","inr","antiplatelet","anticoagulant","nsaid","nsaids",
        "warfarin","coumadin","vitamin k antagonist",
        "cyp2d6","cyp3a4","cyp1a2","cyp2c9","cyp2c19","p-gp","pgp","p glycoprotein",
        "concomitant use","co-administration","avoid concomitant","monitor","dose adjustment",
        "strong inhibitor","strong inducer","contraindicated with",
        # classic warfarin offenders (help catch open-ended Q)
        "trimethoprim","sulfamethoxazole","tmp-smx","bactrim","metronidazole",
        "fluconazole","voriconazole","ketoconazole","azole",
        "erythromycin","clarithromycin","macrolide",
        "ciprofloxacin","levofloxacin","fluoroquinolone",
        "amiodarone","rifampin","phenytoin","carbamazepine","miconazole"
    ],
    "warning": [
        "boxed warning","black box warning","warnings and precautions","precaution",
        "serious","life-threatening","fatal","mortality","risk of death",
        "agranulocytosis","neutropenia","severe neutropenia","myocarditis","seizure","rems"
    ],
    "adverse_effect": [
        "adverse reaction","adverse reactions","adverse events","side effect","most common adverse",
        "incidence","occurrence","frequency","nausea","diarrhea","vomiting","headache",
        "dizziness","abdominal pain","injection site","constipation","fatigue",
        "upper respiratory tract infection","urticaria","rash"
    ],
    "contraindication": [
        "contraindicated","contraindication","do not use","hypersensitivity",
        "concomitant use with nitrates","severe hepatic impairment","pregnancy","pregnant"
    ],
    "indication": [
        "is indicated for","indicated for","adjunct to","management of","treatment of","prevention of","reduce the risk",
    ],
}

_INTENT_CUES: Dict[str, List[re.Pattern]] = {
    "interaction": [
        re.compile(r"\b(contraindicated|do not use)\b.*\b(maoi|monoamine oxidase)\b", re.I),
        re.compile(r"\bserotonin syndrome\b", re.I),
        re.compile(r"\bwarfarin|coumadin|vitamin\s*k\s*antagonist|inr\b", re.I),
        re.compile(r"\b(cyp[0-9a-z\-]+|p-?gp)\b", re.I),
        re.compile(r"\b(concomitant|co[- ]administration|use with|avoid concomitant)\b", re.I),
        re.compile(r"\b(bleeding|hemorrhage|increase[s]? (?:inr|levels?)|reduce[s]? levels?|potentiate[s]?|anticoagulant effect)\b", re.I),
        re.compile(r"\b(tmp[- ]?smx|trimethoprim|sulfamethoxazole|metronidazole|azole|amiodarone|rifampin|macrolide|fluoroquinolone|phenytoin|carbamazepine|miconazole|nsaid)\b", re.I),
    ],
    "contraindication": [
        re.compile(r"\bcontraindicated\b", re.I),
        re.compile(r"\bdo not use\b", re.I),
        re.compile(r"\bhypersensitivity\b", re.I),
        re.compile(r"\bsevere (?:renal|hepatic) impairment\b", re.I),
        re.compile(r"\bconcomitant use with\b.*\bnitrates?\b", re.I),
        re.compile(r"\bpregnan", re.I),
    ],
    "warning": [
        re.compile(r"\bboxed warning\b|\bblack box\b|\brems\b", re.I),
        re.compile(r"\bwarning[s]?\b|\bprecaution[s]?\b", re.I),
        re.compile(r"\bserious (?:risks?|hazards?)\b", re.I),
        re.compile(r"\bfatal|mortality|life[- ]threatening\b", re.I),
        re.compile(r"\bagranulocytosis|neutropenia|severe neutropenia|myocarditis|seizure\b", re.I),
    ],
    "adverse_effect": [
        re.compile(r"\badverse reactions?\b|\bside effects?\b", re.I),
        re.compile(r"\bmost common\b|\bmost frequently\b", re.I),
        re.compile(r"\b\d{1,2}\s?%\b", re.I),
        re.compile(r"\bnausea|vomit|diarrhea|headache|dizziness|abdominal pain|injection site|constipation|fatigue\b", re.I),
    ],
    "indication": [
        re.compile(r"\bis indicated for\b", re.I),
        re.compile(r"\bfor the (?:treatment|prevention|reduction)\b", re.I),
        re.compile(r"\bindication[s]?\b", re.I),
    ],
}

# -----------------------------
# Headings, noise & utilities
# -----------------------------
_HEADING_INFER = [
    ("indications and usage", ["indications and usage","indications","uses","therapeutic indications"]),
    ("contraindications", ["contraindications","contraindication"]),
    ("warnings and precautions", ["warnings and precautions","warnings","precautions","boxed warning","black box warning"]),
    ("drug interactions", ["drug interactions","interactions","drug-drug interactions","interactions with other drugs"]),
    ("adverse reactions", ["adverse reactions","adverse events","side effects","most common adverse reactions"]),
    ("dosage and administration", [
        "dosage and administration","recommended dosage","recommended dosing schedule","dosage ","dose adjustment"
    ]),
    ("boxed warning", ["boxed warning","boxed warnings","black box warning"]),
]
_GENERIC_HDR_RX = re.compile(r"\b(tablet|tablets|capsule|capsules|inhalation|aerosol|usp|oral|injection|solution|suspension)\b", re.I)
_NOISE_PATTERNS = [
    _UUID_RX,
    re.compile(r"\b\d{4,5}-\d{3,4}-\d{1,2}\b"),
    re.compile(r"\bNDC\b[:\s]?\d[\d\-]+\b", re.I),
    re.compile(r"\b\d{8}\b"),
    re.compile(r"\b\d{5,}\b"),
    re.compile(r"\bORAL\b|\bUSP\b|\bSPRAY\b|\bAEROSOL\b", re.I),
    re.compile(r"\b(for oral inhalation only)\b", re.I),
]
_LETTERS_ONLY = re.compile(r"[^a-z]")
def _norm_letters(s: str) -> str: return _LETTERS_ONLY.sub("", (s or "").lower())

# -----------------------------
# Dosage cues & helpers
# -----------------------------
UNITS_RX = re.compile(r"\b\d+\s*(?:mg|mcg|g|units)\b", re.I)
FREQ_RX  = re.compile(r"\b(?:once daily|twice daily|three times|four times|bid|tid|qid|q\d+h|every\s+\d+\s*(?:hours|days|weeks))\b", re.I)
MEAL_RX  = re.compile(r"\bwith meals?\b", re.I)
VERB_RX  = re.compile(r"\b(?:starting dose|titrate|titration|maximum(?:\s+daily)?\s+dose)\b", re.I)
CHEM_NOISE = re.compile(r"\b(?:antihyperglycemic|chemically|pharmacologically|N,\s?N-|imidodicarbonimidic|diamid|USP)\b", re.I)
_DOSE_TERMS_RX = re.compile(
    r"\b(?:\d+\s*(?:mg|mcg|g|units)\b|once daily|twice daily|three times|four times|bid|tid|qid|q\d+h|every\s+\d+\s*(?:hours|days|weeks)|with meals|starting dose|titrate|maximum)\b",
    re.I
)

def _split_sentences(t: str) -> List[str]:
    t = re.sub(r"\s+", " ", (t or "")).strip()
    if not t: return []
    return re.split(r"(?<=[.!?])\s+", t)

def _normalize_phrase_key(p: str) -> str:
    p = (p or "").strip().lower()
    if not p: return p
    return p.replace("monoamine oxidase inhibitors","maoi").replace("monoamine oxidase inhibitor","maoi")

def _make_snippet(s: str) -> str:
    t = " ".join((s or "").split())
    for rx in _NOISE_PATTERNS: t = rx.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    good = [p.strip() for p in parts if len(re.findall(r"[A-Za-z0-9'-]+", p)) >= 6]
    if not good: return t
    out = " ".join(good[:2])
    if len(out) < 200 and len(good) >= 3: out = " ".join(good[:3])
    if len(out) > 600: out = out[:600].rsplit(" ", 1)[0] + "…"
    return out.strip()

_GUIDE_PATTERNS = [
    r"(?:usual|recommended)\s+starting\s+dose[^.]{0,220}\.",
    r"\binitial\s+dose[^.]{0,220}\.",
    r"\bstarting\s+dose[^.]{0,220}\.",
    r"\btitrate[^.]{0,220}\.",
    r"\bmaximum(?:\s+daily)?\s+dose[^.]{0,220}\.",
    r"\b(?:give|given)\s+with\s+meals[^.]{0,220}\.",
]
def _extract_best_dose_guideline(text: str) -> Optional[str]:
    if not text: return None
    low = text.lower()
    for pat in _GUIDE_PATTERNS:
        m = re.search(pat, low, re.I)
        if m: return text[m.start():m.end()].strip()
    return None
def _make_dose_snippet(text: str) -> str:
    g = _extract_best_dose_guideline(text)
    if g: return _make_snippet(g)
    dose_sents = [s for s in _split_sentences(text) if (UNITS_RX.search(s) and (FREQ_RX.search(s) or MEAL_RX.search(s) or VERB_RX.search(s)))]
    if dose_sents: return _make_snippet(" ".join(dose_sents[:2]))
    s = re.sub(r"[ \t]*\n[ \t]*", " ", (text or ""))
    windows: List[str] = []
    for m in UNITS_RX.finditer(s):
        lo = max(0, m.start() - 100); hi = min(len(s), m.end() + 100)
        chunk = s[lo:hi]
        if FREQ_RX.search(chunk) or MEAL_RX.search(chunk) or VERB_RX.search(chunk):
            windows.append(chunk.strip())
    return _make_snippet(" ".join(windows[:2])) if windows else _make_snippet(text)

# -----------------------------
# Intent signals & snippets
# -----------------------------
def _row_has_intent_signal(intent: str, text: str) -> bool:
    if intent == "dosage":
        return bool(_DOSE_TERMS_RX.search(text))
    pats = _INTENT_CUES.get(intent, [])
    if not pats: return True
    for rx in pats:
        if rx.search(text or ""): return True
    return False

def _make_intent_snippet(intent: str, text: str) -> str:
    if intent == "dosage":
        return _make_dose_snippet(text)
    sents = _split_sentences(text)
    pats = _INTENT_CUES.get(intent, [])
    if sents and pats:
        hits = []
        for s in sents:
            for rx in pats:
                if rx.search(s):
                    hits.append(s.strip()); break
        if hits: return _make_snippet(" ".join(hits[:2]))
    if intent == "adverse_effect":
        pri = [s for s in sents if re.search(r"\b\d{1,2}\s?%|\bmost (?:common|frequently)\b", s, re.I)]
        if pri: return _make_snippet(" ".join(pri[:2]))
    if intent == "indication":
        pri = [s for s in sents if re.search(r"\bindicated\b|\bfor the treatment\b", s, re.I)]
        if pri: return _make_snippet(" ".join(pri[:2]))
    return _make_snippet(text)

# -----------------------------
# String gatherers
# -----------------------------
def _gather_strings_pref(obj: Any, prefer_keys: List[str]) -> List[str]:
    out: List[str] = []
    def walk(o: Any, parent_key: str = ""):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if kl in (pk.lower() for pk in prefer_keys):
                    if isinstance(v, str) and v.strip(): out.append(v)
                    elif isinstance(v, (list, dict)):   walk(v, kl)
                else:
                    walk(v, kl)
        elif isinstance(o, list):
            for it in o: walk(it, parent_key)
        elif isinstance(o, (str,)):
            if o.strip() and parent_key and parent_key.lower() in (pk.lower() for pk in prefer_keys):
                out.append(o)
    walk(obj); return out

def _gather_strings_all(obj: Any) -> List[str]:
    out: List[str] = []
    def walk(o: Any):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for it in o: walk(it)
        elif isinstance(o, (str,)):
            if o.strip(): out.append(o)
    walk(obj)
    return out

def _dedupe_join(chunks: List[str]) -> str:
    seen, out = set(), []
    for s in chunks:
        t = " ".join((s or "").split())
        if t and t not in seen:
            seen.add(t); out.append(t)
    return " ".join(out)

# -----------------------------
# Row parsing
# -----------------------------
def _row_setid(o: Dict[str, Any]) -> Optional[str]:
    for k in ("spl_set_id","SPL_SET_ID","setid","SetId","SETID","set_id","setId","set_id_string","setIdString"):
        v = o.get(k)
        if isinstance(v, str): return v.strip()
    return None

def _row_section(o: Dict[str, Any]) -> str:
    for k in SECTION_KEYS:
        v = o.get(k)
        if isinstance(v, str) and v.strip(): return v.strip().lower()
    return ""

def _row_text(o: Dict[str, Any]) -> str:
    pref = _gather_strings_pref(o, TEXT_KEYS)
    return _dedupe_join(pref) if pref else _dedupe_join(_gather_strings_all(o))

def _iter_master_rows(path: str) -> Iterable[Tuple[str, str, str]]:
    if not path or not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            sid = _row_setid(o)
            if not sid: continue
            sec = _row_section(o)
            txt = _row_text(o)
            tl = (txt or "").lower()
            if not sec or _GENERIC_HDR_RX.search(sec or ""):
                for canonical, variants in _HEADING_INFER:
                    if any(v in tl for v in variants):
                        sec = canonical; break
            yield sid, (sec or ""), (txt or "")

# -----------------------------
# Intent & scoring
# -----------------------------
def _allow_sections(intent: str) -> List[str]:
    return _DEF_SECTIONS.get((intent or "").lower(), [])

def _phrase_hits(txt: str, phrase: str) -> bool:
    if not phrase: return True
    txt_l = (txt or "").lower()
    key = _normalize_phrase_key(phrase)
    if key in _SYNONYMS and any(s in txt_l for s in _SYNONYMS[key]): return True
    toks = re.findall(r"[A-Za-z0-9'-]+", key)
    return any(t in txt_l for t in toks if len(t) > 2)

def _section_ok(sec: str, allow: List[str], intent: str = "") -> bool:
    if not allow: return True
    if not sec:   return False
    if intent == "dosage" and _norm_letters(sec) == _norm_letters("dosage forms and strengths"):
        return False
    sl = _norm_letters(sec)
    for a in allow:
        al = _norm_letters(a)
        if sl == al or al in sl or sl in al: return True
    return False

def _score(qv, txt: str, sec: str, allow: List[str], phrase_ok: bool, encoder, intent: str = "", intent_hit: bool = False) -> float:
    sv = encoder.encode([txt])
    base = _cos(qv, sv)
    section_bonus = 1.0 if _section_ok(sec, allow, intent) else 0.0
    phrase_bonus  = 1.0 if phrase_ok else 0.0
    dose_bonus    = 0.25 if (intent == "dosage" and _DOSE_TERMS_RX.search(txt or "")) else 0.0
    intent_bonus  = 0.30 if intent_hit and intent in ("interaction","contraindication","warning","adverse_effect","indication") else 0.0
    return 0.38*base + 0.38*section_bonus + 0.12*phrase_bonus + 0.07 + dose_bonus + intent_bonus

# -----------------------------
# Ultra rescue (dosage)
# -----------------------------
def _load_master_objects_for_sid(path: str, spl_set_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path): return out
    sid_l = spl_set_id.lower()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            sid = (o.get("spl_set_id") or o.get("SPL_SET_ID") or "").strip().lower()
            if sid == sid_l:
                out.append(o)
    return out

def _flatten_strings(obj: Any) -> List[str]:
    acc: List[str] = []
    def walk(x: Any):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            s = x.strip()
            if s: acc.append(s)
    walk(obj); return acc

def _best_windows(text: str, token_rx: re.Pattern, window: int = 80) -> List[str]:
    s = re.sub(r"[ \t]*\n[ \t]*", " ", (text or ""))
    spans = []
    for m in token_rx.finditer(s):
        lo = max(0, m.start() - window); hi = min(len(s), m.end() + window)
        spans.append(s[lo:hi])
    seen, out = set(), []
    for t in spans:
        t2 = re.sub(r"\s+", " ", t).strip()
        if t2 and t2 not in seen:
            seen.add(t2); out.append(t2)
    return out[:4]

def _dosage_ultra_rescue_v2(*, query: str, spl_set_id: str, encoder, top_k: int) -> List[Dict[str, Any]]:
    master = _dm_master_path()
    if not master or not os.path.exists(master): return []
    objs = _load_master_objects_for_sid(master, spl_set_id)
    if not objs: return []
    long_text = " ".join(_flatten_strings(o) for o in objs)
    chunks = _best_windows(long_text, UNITS_RX, window=100)
    if not chunks: return []
    qv = encoder.encode([query])
    items: List[Dict[str, Any]] = []
    for ch in chunks:
        snippet = _make_dose_snippet(ch)
        sv = encoder.encode([snippet]); base = _cos(qv, sv)
        score = 0.65 * base + 0.35
        items.append({
            "spl_set_id": spl_set_id,
            "section": "dosage and administration",
            "snippet": snippet,
            "score": score,
            "stage": 26,
            "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl_set_id}"},
        })
    items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return items[:top_k]

# -----------------------------
# Rescues: warning / adverse / interaction
# -----------------------------
WARNING_HARD_RX = re.compile(r"\bboxed warning\b|\bblack box\b|\brems\b|agranulocytosis|neutropenia|severe neutropenia|myocarditis|seizure|fatal|mortality|life[- ]threatening", re.I)
ADVERSE_HINT_RX = re.compile(r"\b(adverse reactions?|side effects?)\b|\bmost (?:common|frequently)\b|\b\d{1,2}\s?%\b|nausea|diarrhea|vomiting|headache|dizziness|abdominal pain|injection site|rash|urticaria", re.I)

def _warning_rescue(query: str, spl_set_id: str, encoder, top_k: int) -> List[Dict[str, Any]]:
    master = _dm_master_path()
    if not master or not os.path.exists(master): return []
    qv = encoder.encode([query])
    items: List[Dict[str, Any]] = []
    for sid, sec, txt in _iter_master_rows(master):
        if sid.lower() != spl_set_id.lower(): continue
        if not txt: continue
        if WARNING_HARD_RX.search(txt):
            snip = _make_intent_snippet("warning", txt)
            sv = encoder.encode([snip]); base = _cos(qv, sv)
            score = 0.55*base + 0.45
            items.append({
                "spl_set_id": sid, "section": sec or "warnings and precautions", "snippet": snip,
                "score": score, "stage": 24,
                "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"},
            })
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

def _adverse_rescue(query: str, spl_set_id: str, encoder, top_k: int) -> List[Dict[str, Any]]:
    master = _dm_master_path()
    if not master or not os.path.exists(master): return []
    qv = encoder.encode([query])
    items: List[Dict[str, Any]] = []
    for sid, sec, txt in _iter_master_rows(master):
        if sid.lower() != spl_set_id.lower(): continue
        if not txt: continue
        if ADVERSE_HINT_RX.search(txt):
            snip = _make_intent_snippet("adverse_effect", txt)
            sv = encoder.encode([snip]); base = _cos(qv, sv)
            score = 0.5*base + 0.5
            items.append({
                "spl_set_id": sid, "section": sec or "adverse reactions", "snippet": snip,
                "score": score, "stage": 23,
                "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"},
            })
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

def _interaction_rescue(query: str, spl_set_id: str, drug_name: str, encoder, top_k: int) -> List[Dict[str, Any]]:
    master = _dm_master_path()
    if not master or not os.path.exists(master): return []
    qv = encoder.encode([query])

    items: List[Dict[str, Any]] = []

    # (A) relaxed within the SPL
    for sid, sec, txt in _iter_master_rows(master):
        if sid.lower() != spl_set_id.lower(): continue
        if not txt: continue
        if _row_has_intent_signal("interaction", txt):
            snip = _make_intent_snippet("interaction", txt)
            sv = encoder.encode([snip]); base = _cos(qv, sv)
            section_bonus = 0.25 if _norm_letters(sec).startswith("druginteractions") else 0.0
            score = 0.5*base + 0.35 + section_bonus
            items.append({
                "spl_set_id": sid, "section": sec or "drug interactions", "snippet": snip,
                "score": score, "stage": 22,
                "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"},
            })

    # (B) cross-label scan (need drug name)
    dn = (drug_name or "").lower()
    if dn:
        for sid, sec, txt in _iter_master_rows(master):
            if not txt: continue
            tl = (txt or "").lower()
            if dn in tl and _row_has_intent_signal("interaction", tl):
                snip = _make_intent_snippet("interaction", txt)
                sv = encoder.encode([snip]); base = _cos(qv, sv)
                score = 0.52*base + 0.33
                items.append({
                    "spl_set_id": sid, "section": sec or "drug interactions", "snippet": snip,
                    "score": score, "stage": 21,
                    "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"},
                })

    if not items: return []
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

# -----------------------------
# SPL-scoped mining
# -----------------------------
def _mine_spl(*, query: str, spl_set_id: str, intent: str, condition_text: str,
              encoder, top_k: int, require_phrase: bool, require_section: bool,
              stage_tag: int) -> List[Dict[str, Any]]:

    master = _dm_master_path()
    if not master or not os.path.exists(master): return []

    allow = _allow_sections(intent)
    qv = encoder.encode([query])
    items: List[Dict[str, Any]] = []

    for sid, sec, txt in _iter_master_rows(master):
        if sid.lower() != spl_set_id.lower(): continue
        if not txt: continue

        tl = (txt or "").lower()
        if intent == "dosage" and any(v in tl for v in ["dosage and administration","recommended dosage","recommended dosing schedule"]):
            sec = "dosage and administration"

        raw = txt.strip()
        is_table_like = raw.lower().startswith("table ") or raw.count("\n") > 8
        sec_ok = _section_ok(sec, allow, intent)

        # Allow table-like for dosage, warnings, adverse effects (many labels use tables for these)
        if is_table_like and intent in {"dosage","warning","adverse_effect"} and sec_ok:
            pass
        elif is_table_like:
            continue

        if require_section and ((not sec_ok) or (not sec) or _GENERIC_HDR_RX.search(sec or "")):
            continue

        intent_hit = _row_has_intent_signal(intent, txt)
        if require_section and intent in _INTENT_CUES and not intent_hit:
            continue

        phit = _phrase_hits(txt, condition_text) if condition_text else True
        if require_phrase and not phit:
            continue

        snippet = _make_intent_snippet(intent, txt)
        score = _score(qv, snippet, sec or "", allow, phit, encoder, intent, intent_hit)
        items.append({
            "spl_set_id": sid, "section": sec or "", "snippet": snippet,
            "score": score, "stage": stage_tag,
            "links": {"dailymed": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"},
        })

    items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return items[:top_k]

# -----------------------------
# Name-first and SPL-first wrappers (+ rescues)
# -----------------------------
def _collect_by_name_master(query: str, drug_name: str, intent: str, condition_text: str, encoder, top_k: int) -> List[Dict[str, Any]]:
    master = _dm_master_path()
    if not master or not os.path.exists(master): return []
    lowered = (drug_name or "").lower()
    if not lowered: return []

    counts: Dict[str, int] = {}
    for sid, _sec, txt in _iter_master_rows(master):
        if lowered in (txt or "").lower():
            counts[sid] = counts.get(sid, 0) + 1
    if not counts: return []
    best_sid = max(counts.items(), key=lambda kv: kv[1])[0]

    items = _mine_spl(query=query, spl_set_id=best_sid, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=True, require_section=True, stage_tag=1)
    if items: return items

    items = _mine_spl(query=query, spl_set_id=best_sid, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=False, require_section=True, stage_tag=2)
    if items: return items

    if intent == "dosage":
        items = _dosage_ultra_rescue_v2(query=query, spl_set_id=best_sid, encoder=encoder, top_k=top_k)
        if items: return items
    elif intent == "warning":
        items = _warning_rescue(query, best_sid, encoder, top_k)
        if items: return items
    elif intent == "adverse_effect":
        items = _adverse_rescue(query, best_sid, encoder, top_k)
        if items: return items
    elif intent == "interaction":
        items = _interaction_rescue(query, best_sid, lowered, encoder, top_k)
        if items: return items

    items = _mine_spl(query=query, spl_set_id=best_sid, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=False, require_section=False, stage_tag=3)
    return items

def _collect_by_spl_master(query: str, spl_set_id: str, intent: str, condition_text: str, encoder, top_k: int, drug_name_for_cross: str = "") -> List[Dict[str, Any]]:
    items = _mine_spl(query=query, spl_set_id=spl_set_id, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=True, require_section=True, stage_tag=1)
    if items: return items
    items = _mine_spl(query=query, spl_set_id=spl_set_id, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=False, require_section=True, stage_tag=2)
    if items: return items

    if intent == "dosage":
        items = _dosage_ultra_rescue_v2(query=query, spl_set_id=spl_set_id, encoder=encoder, top_k=top_k)
        if items: return items
    elif intent == "warning":
        items = _warning_rescue(query, spl_set_id, encoder, top_k)
        if items: return items
    elif intent == "adverse_effect":
        items = _adverse_rescue(query, spl_set_id, encoder, top_k)
        if items: return items
    elif intent == "interaction":
        # now we pass the drug name so cross-label scan can fire if needed
        items = _interaction_rescue(query, spl_set_id, drug_name_for_cross, encoder, top_k)
        if items: return items

    items = _mine_spl(query=query, spl_set_id=spl_set_id, intent=intent, condition_text=condition_text,
                      encoder=encoder, top_k=top_k, require_phrase=False, require_section=False, stage_tag=3)
    return items

# -----------------------------
# Public API
# -----------------------------
def collect_evidence(*, query: str, payload: Dict[str, Any], encoder, top_k: int = 6,
                     per_edge: int = 2, id_to_pt: Optional[Dict[str, str]] = None,
                     condition_text: str = "") -> List[Dict[str, Any]]:

    intent = (payload.get("intent") or "").lower()
    drug = payload.get("drug") or {}
    spl  = (drug.get("spl_set_id") or "").strip() if isinstance(drug, dict) else ""
    dn   = (drug.get("matched_name") or drug.get("query_text") or "").strip() if isinstance(drug, dict) else ""

    items: List[Dict[str, Any]] = []
    if spl:
        items = _collect_by_spl_master(query, spl, intent, condition_text, encoder, top_k, drug_name_for_cross=dn)
    if not items and dn:
        items = _collect_by_name_master(query, dn, intent, condition_text, encoder, top_k)

    # extra fallbacks per intent
    if intent == "adverse_effect" and not items and spl:
        items = _adverse_rescue(query, spl, encoder, top_k)
    if intent == "warning" and not items and spl:
        items = _warning_rescue(query, spl, encoder, top_k)
    if intent == "interaction" and not items:
        items = _interaction_rescue(query, spl or "", dn.lower(), encoder, top_k)

    # Keep to same SPL unless crossing labels is needed (interactions may cross)
    if spl and intent not in ("interaction",):
        items = [it for it in items if str(it.get("spl_set_id","")).lower() == spl.lower()]

    return items[:top_k]
