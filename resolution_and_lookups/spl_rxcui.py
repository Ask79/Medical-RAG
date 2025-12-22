# rag_med/router/spl_rxcui.py
from __future__ import annotations
import os, json, itertools
from functools import lru_cache
from typing import Optional

_DEF1 = "data/links/daily_rx_links_resolved.jsonl"
_DEF2 = "data/processed/links/daily_rx_links_resolved.jsonl"


def _map_path() -> Optional[str]:
    cand = os.getenv("SPL_RXCUI_PATH")
    if cand and os.path.exists(cand):
        return cand
    for p in (_DEF1, _DEF2):
        if os.path.exists(p):
            return p
    return cand or _DEF1


@lru_cache(maxsize=1)
def _load_map() -> dict[str, Optional[str]]:
    path = _map_path()
    out: dict[str, Optional[str]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in itertools.islice(f, 2_000_000):
            try:
                o = json.loads(line)
            except Exception:
                continue
            sid = (o.get("spl_set_id") or o.get("SPL_SET_ID") or "").lower().strip()
            if not sid:
                continue
            # Prefer nested primary.rxcui, then top-level rxcui, then first secondary[].rxcui
            rx = (o.get("primary") or {}).get("rxcui")
            if not rx:
                rx = o.get("rxcui") or o.get("RXCUI")
            if not rx:
                sec = o.get("secondary")
                if isinstance(sec, list) and sec:
                    rx = (sec[0] or {}).get("rxcui")
            if rx is not None:
                out[sid] = str(rx)
            else:
                # cache explicit None to avoid re-scanning file later
                out.setdefault(sid, None)
    return out


def spl_to_rxcui(spl_set_id: str) -> Optional[str]:
    """Return RxCUI string for a given SPL set id (case-insensitive), or None."""
    if not spl_set_id:
        return None
    m = _load_map()
    return m.get(str(spl_set_id).lower().strip())
