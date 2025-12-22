import os, json, math
from collections import defaultdict
from typing import Dict, List, Optional, Iterable

# ---------- config defaults (paths) ----------
def _default_paths(repo: str):
    rollup = os.path.join(repo, "data", "links", "rxcui_condition_rollup.jsonl")
    r2c    = os.path.join(repo, "data", "links", "rxcui_to_conditions.jsonl")
    c2r    = os.path.join(repo, "data", "links", "condition_to_rxcui.jsonl")
    block  = os.path.join(repo, "data", "links", "_eval", "generic_blocklist.txt")
    return rollup, r2c, c2r, block

# ---------- helpers ----------
def score_edge(max_conf: float, spl_count: int) -> float:
    # 0.6*max_conf + 0.4*log1p(spl_count)
    return 0.6*float(max_conf) + 0.4*math.log1p(int(spl_count))

def _read_blocklist(path: Optional[str]) -> set:
    s = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith("#"):
                    s.add(t)
    return s

def _all_examples_negated(examples: list) -> Optional[bool]:
    """Return True if at least one example exists and all are negated.
       Return False if any example is not negated.
       Return None if negation info is unavailable."""
    if not examples:
        return None
    saw = False
    for ex in examples:
        if isinstance(ex, dict) and "negated" in ex:
            saw = True
            if not ex.get("negated", False):
                return False
    return True if saw else None

# ---------- main index ----------
class GraphIndex:
    """
    Loads RxCUI<->SNOMED edges from your link layer.
    Prefers the aggregated rollup; falls back to the two directional files.
    Keeps provenance fields (sections, spl_set_ids, examples, chunk_count, last_updated).
    """
    def __init__(self,
                 repo_root: str,
                 rollup_path: Optional[str] = None,
                 r2c_path: Optional[str] = None,
                 c2r_path: Optional[str] = None,
                 generic_blocklist_path: Optional[str] = None):
        d_rollup, d_r2c, d_c2r, d_block = _default_paths(repo_root)
        self.rollup_path = rollup_path if rollup_path else d_rollup
        self.r2c_path    = r2c_path    if r2c_path    else d_r2c
        self.c2r_path    = c2r_path    if c2r_path    else d_c2r
        self.blocklist   = _read_blocklist(generic_blocklist_path if generic_blocklist_path else d_block)

        # internal stores
        self.r2c: Dict[str, List[dict]] = defaultdict(list)
        self.c2r: Dict[str, List[dict]] = defaultdict(list)

        if os.path.exists(self.rollup_path):
            self._load_rollup(self.rollup_path)
        else:
            # fallback to the two directional files (nested shapes)
            if os.path.exists(self.r2c_path): self._load_r2c_nested(self.r2c_path)
            if os.path.exists(self.c2r_path): self._load_c2r_nested(self.c2r_path)

    # ---------- loaders ----------
    def _load_rollup(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                rxcui = str(o.get("rxcui", "")).strip()
                sctid = str(o.get("snomed_id", o.get("sctid", ""))).strip()
                ltype = str(o.get("link_type", "")).lower().strip()
                if not (rxcui and sctid and ltype): continue
                e = {
                    "rxcui": rxcui,
                    "sctid": sctid,
                    "link_type": ltype,
                    "max_confidence": float(o.get("max_confidence", 0.0) or 0.0),
                    "spl_count": int(o.get("spl_count", 0) or 0),
                    # provenance (optional)
                    "sections": o.get("sections"),
                    "spl_set_ids": o.get("spl_set_ids"),
                    "chunk_count": o.get("chunk_count"),
                    "examples": o.get("examples"),
                    "last_updated": o.get("last_updated"),
                }
                self.r2c[rxcui].append(e)
                self.c2r[sctid].append(e)

    def _load_r2c_nested(self, path: str):
        # shape: {"rxcui": "...", "conditions": [ { "snomed_id": ..., "link_type": ..., "max_confidence": ..., "spl_count": ... , "examples": [...] }, ...]}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: root = json.loads(line)
                except Exception: continue
                rxcui = str(root.get("rxcui", "")).strip()
                conds = root.get("conditions") or []
                for o in conds:
                    sctid = str(o.get("snomed_id", o.get("sctid",""))).strip()
                    ltype = str(o.get("link_type","")).lower().strip()
                    if not (rxcui and sctid and ltype): continue
                    e = {
                        "rxcui": rxcui,
                        "sctid": sctid,
                        "link_type": ltype,
                        "max_confidence": float(o.get("max_confidence", 0.0) or 0.0),
                        "spl_count": int(o.get("spl_count", 0) or 0),
                        "sections": o.get("sections"),
                        "spl_set_ids": None,   # not present in nested file
                        "chunk_count": o.get("chunk_count"),
                        "examples": o.get("examples"),
                        "last_updated": None,
                        # pass through useful labels if present
                        "label": o.get("label"),
                        "tag": o.get("tag"),
                    }
                    self.r2c[rxcui].append(e)
                    self.c2r[sctid].append(e)

    def _load_c2r_nested(self, path: str):
        # shape: {"snomed_id": "...", "drugs": [ { "rxcui": ..., "link_type": ..., "max_confidence": ..., "spl_count": ... , "examples": [...] }, ...]}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: root = json.loads(line)
                except Exception: continue
                sctid = str(root.get("snomed_id", root.get("sctid",""))).strip()
                drugs = root.get("drugs") or []
                for o in drugs:
                    rxcui = str(o.get("rxcui","")).strip()
                    ltype = str(o.get("link_type","")).lower().strip()
                    if not (rxcui and sctid and ltype): continue
                    e = {
                        "rxcui": rxcui,
                        "sctid": sctid,
                        "link_type": ltype,
                        "max_confidence": float(o.get("max_confidence", 0.0) or 0.0),
                        "spl_count": int(o.get("spl_count", 0) or 0),
                        "sections": o.get("sections"),
                        "spl_set_ids": None,
                        "chunk_count": o.get("chunk_count"),
                        "examples": o.get("examples"),
                        "last_updated": None,
                        "tty": o.get("tty"),
                        "has_ndc": o.get("has_ndc"),
                    }
                    self.r2c[rxcui].append(e)
                    self.c2r[sctid].append(e)

    # ---------- query ----------
    def _filter_rank(self, edges: List[dict],
                     link_types: Iterable[str],
                     min_conf: float = 0.80,
                     min_spl: int = 2,
                     blocklist: Optional[set] = None,
                     drop_all_negated: bool = True,
                     top_n: int = 10) -> List[dict]:
        lt = set(t.lower() for t in link_types) if link_types else None
        blk = blocklist or set()
        rows = []
        for e in edges or []:
            if lt and e.get("link_type","") not in lt:    # fast path: already lower-cased
                continue
            if e.get("sctid") in blk:
                continue
            mc = float(e.get("max_confidence", 0.0) or 0.0)
            sc = int(e.get("spl_count", 0) or 0)
            if mc < float(min_conf): continue
            if sc < int(min_spl):    continue
            if drop_all_negated:
                neg = _all_examples_negated(e.get("examples") or [])
                if neg is True:
                    continue
            rows.append({**e, "score": score_edge(mc, sc)})
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:top_n]

    def conditions_for_rxcui(self, rxcui: str,
                             link_types: Iterable[str],
                             min_conf: float = 0.80,
                             min_spl: int = 2,
                             top_n: int = 10) -> List[dict]:
        return self._filter_rank(self.r2c.get(str(rxcui), []),
                                 link_types, min_conf, min_spl,
                                 blocklist=self.blocklist, top_n=top_n)

    def rxcuis_for_condition(self, sctid: str,
                             link_types: Iterable[str],
                             min_conf: float = 0.80,
                             min_spl: int = 2,
                             top_n: int = 10) -> List[dict]:
        return self._filter_rank(self.c2r.get(str(sctid), []),
                                 link_types, min_conf, min_spl,
                                 blocklist=self.blocklist, top_n=top_n)
