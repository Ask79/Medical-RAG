#!/usr/bin/env python3
import argparse, json, hashlib, math, re
from pathlib import Path
from datetime import datetime, timezone

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("//"): continue
            yield json.loads(line)

def strip_semantic_tag(term: str) -> str:
    if not term: return term
    t = term.strip()
    if t.endswith(")") and " (" in t:
        return t.rsplit(" (",1)[0]
    return t

def estimate_tokens(text: str) -> int:
    # cheap & decent: ~4 chars/token
    return max(1, math.ceil(len(text) / 4))

def stable_id(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def best_name(doc) -> str:
    pt = doc.get("terms", {}).get("pt") or {}
    # prefer en-US / en-GB / default in that order
    for k in ("en-US","en-GB","default"):
        if k in pt and pt[k] and pt[k].get("term"):
            return strip_semantic_tag(pt[k]["term"])
    fsn = doc.get("terms", {}).get("fsn") or {}
    if fsn.get("term"):
        return strip_semantic_tag(fsn["term"])
    return f"SNOMED {doc.get('conceptId','')}"

def build_display_name_index(snomed_path: Path):
    names = {}
    for doc in read_jsonl(snomed_path):
        cid = str(doc.get("conceptId","")).strip()
        if not cid: continue
        names[cid] = best_name(doc)
    return names

def chunk_synonyms(syns, base_prefix, max_tokens):
    """Yield text chunks containing synonyms, respecting max_tokens."""
    buf = []
    cur = base_prefix
    for s in syns:
        piece = f"- {s}\n"
        if estimate_tokens(cur + piece) > max_tokens:
            if buf:
                yield cur
            cur = base_prefix + piece
            buf = [piece]
        else:
            cur += piece
            buf.append(piece)
    if cur.strip():
        yield cur

def run(args):
    in_path  = Path(args.snomed_combined_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Indexing display names...")
    name_idx = build_display_name_index(in_path)
    print(f"  names indexed: {len(name_idx):,}")

    print("Emitting chunks...")
    n_chunks = 0
    ts = now_utc_iso()
    max_toks = int(args.max_tokens)

    # second streaming pass to avoid holding all docs
    with out_path.open("w", encoding="utf-8") as out:
        for doc in read_jsonl(in_path):
            cid = str(doc.get("conceptId","")).strip()
            if not cid: continue

            # core fields
            title = best_name(doc)
            fsn_obj = (doc.get("terms",{}) or {}).get("fsn") or {}
            fsn = fsn_obj.get("term") or ""
            def_status = "fully_defined" if doc.get("is_fully_defined") else "primitive"
            parents = (doc.get("hierarchy",{}) or {}).get("parent_ids", []) or []
            parent_lines = []
            for pid in parents:
                pname = name_idx.get(str(pid), "")
                parent_lines.append(f"- {pid}: {pname}" if pname else f"- {pid}")

            core_text = (
                f"{title}\n"
                f"SNOMED CT conceptId: {cid}\n"
                f"Definition status: {def_status}\n"
                + (f"FSN: {fsn}\n" if fsn else "")
                + ("Parents:\n" + "\n".join(parent_lines) + "\n" if parent_lines else "")
            ).strip() + "\n"

            # write core chunk
            core_chunk = {
                "chunk_id": f"snomed:chunk:{cid}:{stable_id(cid,'core')}",
                "doc_id": f"snomed:concept:{cid}",
                "conceptId": cid,
                "section": "core",
                "text": core_text,
                "token_estimate": estimate_tokens(core_text),
                "order": 0,
                "source": "snomed_ct",
                "last_updated": ts,
            }
            out.write(json.dumps(core_chunk, ensure_ascii=False) + "\n")
            n_chunks += 1

            # synonyms batching
            syns = []
            for s in (doc.get("terms",{}) or {}).get("synonyms", []):
                term = s.get("term")
                if term:
                    syns.append(term)
            syns = sorted(set(syns), key=str.lower)

            if syns:
                base = "Synonyms:\n"
                idx = 1
                for syn_text in chunk_synonyms(syns, base, max_toks):
                    chunk = {
                        "chunk_id": f"snomed:chunk:{cid}:{stable_id(cid,'syn',idx)}",
                        "doc_id": f"snomed:concept:{cid}",
                        "conceptId": cid,
                        "section": "synonyms",
                        "text": syn_text,
                        "token_estimate": estimate_tokens(syn_text),
                        "order": idx,
                        "source": "snomed_ct",
                        "last_updated": ts,
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    idx += 1

    print(f"Wrote {n_chunks:,} chunks -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Chunk SNOMED ConceptAggregate JSONL into RAG-friendly chunks")
    ap.add_argument("--snomed-combined-jsonl", required=True, help="data/interim/snomed/snomed_combined_master.jsonl")
    ap.add_argument("--out", required=True, help="Output JSONL, e.g., data/interim/snomed/snomed_chunks_master.jsonl")
    ap.add_argument("--max-tokens", type=int, default=300, help="Approx token cap per chunk (default: 300)")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
