# rag_med/router/resolve_drug.py
from dataclasses import dataclass
from functools import lru_cache
import json, os, re
from typing import Optional, List, Tuple

# Prefer ingredient or clinical drug when we have to pick from 'secondary'
_TTY_PREF = ("IN", "SCD", "SBD", "PIN", "BPCK", "GPCK", "DP", "SU")

@dataclass
class DrugResolution:
    query_text: Optional[str]
    rxcui: Optional[str]
    confidence: float
    spl_set_id: Optional[str]
    matched_name: Optional[str]
    note: str

# ------------------------
# SPL↔RxCUI LINK LOADER
# ------------------------
def _default_links_path() -> str:
    return os.getenv(
        "SPL_RXCUI_PATH",
        os.path.join("data", "processed", "links", "daily_rx_links_resolved.jsonl"),
    )

@lru_cache(maxsize=1)
def _load_spl_to_rxcui_map_cached(path: str) -> dict:
    """
    Loads SPL_SET_ID -> (rxcui, tty) from your schema:
      {"spl_set_id": "...",
       "primary": {"rxcui": "...", "tty": "...", ...},
       "secondary": [{"rxcui": "...", "tty": "..."}, ...]}
    Keys normalized to lowercase for SPLs.
    """
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            spl = (o.get("spl_set_id") or "").strip().lower()
            if not spl:
                continue

            pri = o.get("primary") or {}
            rx  = pri.get("rxcui")
            tty = pri.get("tty")

            # If no primary rxcui, choose best from secondary by TTY preference
            if not rx:
                sec = o.get("secondary") or []
                best = None
                # try preference order
                for pref in _TTY_PREF:
                    for cand in sec:
                        if cand.get("rxcui") and str(cand.get("tty", "")).upper() == pref:
                            best = cand
                            break
                    if best:
                        break
                # if still none, take the first valid
                if not best:
                    for cand in sec:
                        if cand.get("rxcui"):
                            best = cand
                            break
                if best:
                    rx = best.get("rxcui")
                    tty = best.get("tty")

            if rx:
                m[spl] = (str(rx), str(tty) if tty else "")

    return m

def _load_spl_to_rxcui_map(path: Optional[str]) -> dict:
    """Public wrapper; lets caller pass None to use env/default path."""
    path = path or _default_links_path()
    return _load_spl_to_rxcui_map_cached(path)

# ---------------------------------------------
# FAST META ROW ACCESS (cached byte offsets)
# ---------------------------------------------
@lru_cache(maxsize=8)
def _meta_line_offsets(meta_path: str) -> List[int]:
    """
    Build a byte-offset index so row_id→line is O(1) seek.
    NOTE: For very large files, this index can be big; consider persisting
    an .idx file in the future. For now, we cache in-memory once per path.
    """
    offsets: List[int] = []
    pos = 0
    with open(meta_path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets

def _read_meta_rows_by_ids(meta_path: str, row_ids: List[int]) -> List[dict]:
    """Random-access read of JSONL rows by numeric ids using cached offsets."""
    if not os.path.exists(meta_path):
        return []
    offs = _meta_line_offsets(meta_path)
    out: List[dict] = []
    with open(meta_path, "rb") as f:
        for rid in row_ids:
            if rid < 0 or rid >= len(offs):
                continue
            f.seek(offs[rid])
            line = f.readline()
            try:
                out.append(json.loads(line.decode("utf-8")))
            except Exception:
                continue
    return out

# ------------------------------------------------
# NEAREST NEIGHBOR LOOKUP (DailyMed → SPL meta)
# ------------------------------------------------
def _nearest_dailymed(qvec, dm_index, dm_meta_path: str, k: int = 5) -> List[dict]:
    """
    qvec: (1, d) numpy array
    dm_index: FAISS index
    dm_meta_path: path to meta JSONL (aligned with embeddings)
    """
    D, I = dm_index.search(qvec, k)
    if I is None or I.size == 0:
        return []
    row_ids = [int(x) for x in I[0].tolist()]
    return _read_meta_rows_by_ids(dm_meta_path, row_ids)

# ------------------------
# MAIN RESOLVER
# ------------------------
def resolve_drug(query_text: str,
                 qvec, dm_index,
                 dm_meta_path: str,
                 spl_to_rxcui_jsonl: Optional[str] = None) -> DrugResolution:
    """
    Strategy:
      1) NN search → top meta rows → read SPL_SET_ID(s)
      2) Map SPL_SET_ID → RxCUI using your daily_rx_links_resolved.jsonl
         (prefers 'primary', falls back via TTY order)
      3) Return DrugResolution with best match
    """
    # Normalize (kept for future lexicon hooks; not strictly needed here)
    _ = re.sub(r"[^A-Za-z0-9 ]+", " ", query_text or "").strip().lower()

    # Load map once (cached)
    spl_map = _load_spl_to_rxcui_map(spl_to_rxcui_jsonl)

    # Get nearest DailyMed chunks
    hits = _nearest_dailymed(qvec, dm_index, dm_meta_path, k=5)
    if not hits:
        return DrugResolution(query_text, None, 0.0, None, None, "No hit")

    # Try to find a mapped SPL among the hits
    for h in hits:
        spl = (h.get("spl_set_id") or h.get("doc_id") or h.get("source_id") or "").lower()
        title = h.get("title") or h.get("section_title") or h.get("drug_name")
        if not spl:
            continue
        rx_pair = spl_map.get(spl)
        if rx_pair:
            rxcui, tty = rx_pair
            note = f"NN→SPL→RxCUI (primary/fallback TTY={tty})"
            return DrugResolution(query_text, rxcui, 0.85, spl, title, note)

    # If no SPL had a mapping, still return the top SPL so upstream can proceed
    top = hits[0]
    spl0 = (top.get("spl_set_id") or top.get("doc_id") or top.get("source_id") or "").lower()
    title0 = top.get("title") or top.get("section_title") or top.get("drug_name")
    if spl0:
        return DrugResolution(query_text, None, 0.6, spl0, title0, "NN→SPL (no RxCUI map)")
    return DrugResolution(query_text, None, 0.0, None, None, "No hit")
