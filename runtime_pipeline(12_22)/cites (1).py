# rag_med/cites.py
from __future__ import annotations

import re
from typing import Optional, Dict

# -----------------------------
# Helpers
# -----------------------------

_UUID_RX = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

def _safe_str(x: object) -> str:
    return "" if x is None else str(x).strip()

def _norm_uuid(u: Optional[str]) -> str:
    s = _safe_str(u)
    m = _UUID_RX.search(s)
    return m.group(0).lower() if m else ""

def _norm_rxcui(x: Optional[str]) -> str:
    s = _safe_str(x)
    s = re.sub(r"[^0-9]", "", s)
    return s

def _norm_sctid(x: Optional[str]) -> str:
    s = _safe_str(x)
    s = re.sub(r"[^0-9]", "", s)
    return s

# -----------------------------
# DailyMed
# -----------------------------

# We keep anchors optional because section IDs differ by label rendering.
_SECTION_ANCHORS = {
    # Common FDA SPL headings → typical anchors (not guaranteed)
    # "indications and usage": "#section-12",
    # "contraindications": "#section-4",
    # "warnings and precautions": "#section-5",
    # "drug interactions": "#section-7",
    # "adverse reactions": "#section-6",
    # "dosage and administration": "#section-2",
    # "boxed warning": "#section-5.1",
    # "mechanism of action": "#section-12.1",
    # "use in specific populations": "#section-8",
    # "pregnancy": "#section-8.1",
}

def build_dailymed_link(spl_set_id: Optional[str], section: Optional[str] = None) -> Optional[str]:
    sid = _norm_uuid(spl_set_id)
    if not sid:
        return None
    base = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}"
    sec = _safe_str(section).lower()
    frag = _SECTION_ANCHORS.get(sec, "")
    return base + (frag or "")

def build_dailymed_api(spl_set_id: Optional[str]) -> Optional[str]:
    sid = _norm_uuid(spl_set_id)
    if not sid:
        return None
    return f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{sid}.json"

# -----------------------------
# RxNav (RxNorm)
# -----------------------------

def build_rxnav_ui(rxcui: Optional[str]) -> Optional[str]:
    rx = _norm_rxcui(rxcui)
    if not rx:
        return None
    # UI deep search (SPA): use hash-bang style
    return f"https://rxnav.nlm.nih.gov/#!search={rx}"

def build_rxnav_api(rxcui: Optional[str]) -> Optional[str]:
    rx = _norm_rxcui(rxcui)
    if not rx:
        return None
    return f"https://rxnav.nlm.nih.gov/REST/rxcui/{rx}.json"

# -----------------------------
# SNOMED CT
# -----------------------------

def build_snomed_browser(sctid: Optional[str]) -> Optional[str]:
    sc = _norm_sctid(sctid)
    if not sc:
        return None
    return f"https://browser.ihtsdotools.org/?perspective=full&conceptId1={sc}"

# -----------------------------
# Public: bundle
# -----------------------------

def build_links(
    spl_set_id: Optional[str] = None,
    rxcui: Optional[str] = None,
    sctid: Optional[str] = None,
    section: Optional[str] = None,
) -> Dict[str, str]:
    """
    Returns a dict of available links. Keys only appear if values exist.
      - dailymed, dailymed_api
      - rxnav (UI), rxnav_api
      - snomed
    """
    out: Dict[str, str] = {}

    dm = build_dailymed_link(spl_set_id, section=section)
    if dm:
        out["dailymed"] = dm
    dm_api = build_dailymed_api(spl_set_id)
    if dm_api:
        out["dailymed_api"] = dm_api

    rx_ui = build_rxnav_ui(rxcui)
    if rx_ui:
        out["rxnav"] = rx_ui
    rx_api = build_rxnav_api(rxcui)
    if rx_api:
        out["rxnav_api"] = rx_api

    sn = build_snomed_browser(sctid)
    if sn:
        out["snomed"] = sn

    return out
