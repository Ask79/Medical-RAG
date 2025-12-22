#!/usr/bin/env python3
import argparse, json, uuid, re, time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# --- sections -> link_type
SECTION_MAP = {
    "indications_and_usage": "indication",
    "contraindications": "contraindication",
    "adverse_reactions": "adverse_effect",
    # optional:
    "warnings_and_precautions": "adverse_effect",
}

ALLOWED_TAGS = {
    "disorder","finding","disease","injury","syndrome","infection","poisoning",
    "neoplasm","abnormality","deficiency","symptom","complication","event"
}

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def strip_semantic_tag(term: str) -> str:
    if not term: return term
    t = term.strip()
    if t.endswith(")") and " (" in t:
        return t.rsplit(" (", 1)[0]
    return t

def infer_tag_from_fsn(fsn: str) -> str:
    m = re.search(r"\(([^)]+)\)\s*$", fsn or "")
    return (m.group(1).strip().lower() if m else "")

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)

def build_lexicon(snomed_master: Path, min_len: int = 3):
    """Return dict: term_lc -> list of {snomed_id, surface} limited to condition-like concepts."""
    lex = defaultdict(list)
    kept = 0
    dedup = set()  # (cid, term_lc)
    for r in read_jsonl(snomed_master):
        cid = str(r.get("conceptId") or "")
        if not cid or not r.get("terms"): 
            continue
        fsn = (r["terms"].get("fsn") or {}).get("term") or ""
        tag = infer_tag_from_fsn(fsn)
        if tag not in ALLOWED_TAGS:
            continue

        # add PT from FSN
        pt = strip_semantic_tag(fsn)
        if pt:
            k = pt.lower().strip()
            if len(k) >= min_len and any(c.isalpha() for c in k) and (cid, k) not in dedup:
                lex[k].append({"snomed_id": cid, "surface": pt})
                dedup.add((cid, k))

        # add synonyms
        for s in (r["terms"].get("synonyms") or []):
            term = (s.get("term") or "").strip()
            if not term:
                continue
            k = term.lower()
            if len(k) >= min_len and any(c.isalpha() for c in k) and (cid, k) not in dedup:
                lex[k].append({"snomed_id": cid, "surface": term})
                dedup.add((cid, k))

        kept += 1
    return lex, kept

# ---------------- matcher (built once) ----------------

def build_matcher(keys_iterable):
    keys = list(keys_iterable)
    print("Preparing matcher...")
    t0 = time.time()
    try:
        import ahocorasick  # pip install pyahocorasick
        A = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEYSTRING)
        for k in keys:
            A.add_word(k, k)  # store term itself
        A.make_automaton()
        print(f"  Aho-Corasick ready for {len(keys):,} terms in {time.time()-t0:.1f}s")
        return ("aho", A)
    except Exception:
        from flashtext import KeywordProcessor  # pip install flashtext
        K = KeywordProcessor(case_sensitive=False)
        for k in keys:
            K.add_keyword(k, k)
        print(f"  FlashText ready for {len(keys):,} terms in {time.time()-t0:.1f}s")
        return ("flash", K)

def iter_matches(text_lc: str, matcher):
    kind, M = matcher
    if kind == "aho":
        for end_idx, term in M.iter(text_lc):
            start = end_idx - len(term) + 1
            yield start, end_idx + 1, term  # end exclusive
    else:
        # FlashText returns (result, start, end)
        for term, start, end in M.extract_keywords(text_lc, span_info=True):
            yield start, end, term

# ---------------- heuristics ----------------

def detect_negation(text_lc, start, end):
    # very light heuristic in ±8 words window (~80 chars)
    window = text_lc[max(0, start-80): min(len(text_lc), end+80)]
    return any(w in window for w in (" not ", " no ", " without ", " denies "))

def confidence_for(surface):
    n = len(surface.strip())
    if n < 4: return 0.60
    if n < 8: return 0.78
    return 0.90

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Build SNOMED←→DailyMed chunk_condition_links.jsonl (fast matcher)")
    ap.add_argument("--snomed-combined-jsonl", required=True, help="snomed_combined_master.jsonl")
    ap.add_argument("--dailymed-chunks-jsonl", required=True, help="DailyMed chunks (e.g., chunks_v5_clean.jsonl)")
    ap.add_argument("--out", required=True, help="Output chunk_condition_links.jsonl")
    ap.add_argument("--sections", nargs="*", default=list(SECTION_MAP.keys()),
                    help="DailyMed sections to scan (defaults to keys in SECTION_MAP)")
    ap.add_argument("--min-term-len", type=int, default=3, help="Min length of lexicon terms")
    ap.add_argument("--limit", type=int, default=0, help="Scan only first N chunks (debug)")
    args = ap.parse_args()

    sm_path  = Path(args.snomed_combined_jsonl)
    dm_path  = Path(args.dailymed_chunks_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building SNOMED condition lexicon...")
    lex, kept = build_lexicon(sm_path, min_len=args.min_term_len)
    print(f"  condition concepts kept: {kept:,}")
    print(f"  term variants: {len(lex):,}")

    matcher = build_matcher(lex.keys())

    n_links = 0
    n_chunks = 0
    now = now_utc_iso()
    secs = set(s.lower() for s in args.sections)
    t_scan = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for chunk in read_jsonl(dm_path):
            n_chunks += 1
            if args.limit and n_chunks > args.limit:
                break

            section = (chunk.get("section") or "").strip().lower()
            if section not in secs:
                continue
            link_type = SECTION_MAP.get(section)
            if not link_type:
                continue

            text = chunk.get("text") or ""
            if not text.strip():
                continue

            chunk_id = chunk.get("chunk_id") or ""
            spl_set  = chunk.get("spl_set_id") or ""
            if not chunk_id or not spl_set:
                continue

            text_lc = text.lower()
            seen = set()
            for s, e, key in iter_matches(text_lc, matcher):
                if (s, e, key) in seen:
                    continue
                seen.add((s, e, key))
                entries = lex.get(key)
                if not entries:
                    continue
                neg = detect_negation(text_lc, s, e)
                mt = text[s:e]
                for entry in entries:
                    rec = {
                        "link_id": str(uuid.uuid4()),
                        "spl_set_id": str(spl_set),
                        "chunk_id": str(chunk_id),
                        "section": section,
                        "snomed_id": entry["snomed_id"],
                        "match_text": mt,
                        "char_start": int(s),
                        "char_end": int(e),
                        "link_type": link_type,
                        "confidence": round(confidence_for(entry["surface"]), 3),
                        "method": "dictionary_match.v2",
                        "qualifiers": {"negated": neg},
                        "last_updated": now,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_links += 1

            if n_chunks % 10000 == 0:
                elapsed = time.time() - t_scan
                print(f"  scanned {n_chunks:,} chunks, wrote {n_links:,} links [{elapsed:.1f}s]")

    print(f"Wrote {n_links:,} links from {n_chunks:,} chunks -> {out_path} in {time.time()-t_scan:.1f}s")

if __name__ == "__main__":
    main()
