#!/usr/bin/env python3
import argparse, json, os, sys, time, hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List
import numpy as np
from tqdm import tqdm

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)

def pick_text(r: Dict[str, Any]) -> str:
    text = (r.get("text") or "").strip()
    pre = []
    sec = (r.get("section") or "").strip()
    if sec: pre.append(f"[section:{sec}]")
    title = (r.get("title") or r.get("name") or r.get("doc_id") or "").strip()
    if title and title != r.get("chunk_id"):
        pre.append(title)
    return ("\n".join(pre) + ("\n" if pre else "")) + text

def safe_id(r: Dict[str, Any]) -> str:
    cid = str(r.get("chunk_id") or r.get("id") or "")
    if cid: return cid
    raw = json.dumps(r, sort_keys=True, ensure_ascii=False)
    return "anon:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(description="Embed chunk JSONLs into .npy (+ optional FAISS)")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input JSONL files")
    ap.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="HF model name")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-per-file", type=int, default=100000, help="0 = no cap")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize (use IP cosine search)")
    ap.add_argument("--build-faiss", dest="build_faiss", action="store_true", help="Write FAISS index")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {args.model} on {args.device}")
    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()

    for in_path_str in args.inputs:
        in_path = Path(in_path_str)
        if not in_path.exists():
            print(f"[SKIP] missing: {in_path}", file=sys.stderr)
            continue

        base = in_path.stem
        tag = args.model.replace("/", "_")
        out_base = out_dir / f"{base}__{tag}"

        # FIX: split the annotated assignments
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        texts: List[str] = []

        n = 0
        print(f"\nReading {in_path} …")
        for r in read_jsonl(in_path):
            txt = pick_text(r)
            if not txt:
                continue
            ids.append(safe_id(r))
            metas.append({
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "section": r.get("section"),
                "source": r.get("source"),
            })
            texts.append(txt)
            n += 1
            if args.limit_per_file and n >= args.limit_per_file:
                break

        if not ids:
            print(f"[WARN] no chunks collected from {in_path}")
            continue

        print(f"Embedding {len(texts):,} chunks … (batch={args.batch_size})")
        vecs: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch = texts[i:i+args.batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            vecs.append(emb.astype("float32"))
        X = np.vstack(vecs)

        if args.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / norms

        npy_path  = out_base.with_suffix(".npy")
        ids_path  = out_base.with_suffix(".ids.txt")
        meta_path = out_base.with_suffix(".meta.jsonl")

        np.save(npy_path, X)
        with ids_path.open("w", encoding="utf-8") as f:
            for _id in ids: f.write(f"{_id}\n")
        with meta_path.open("w", encoding="utf-8") as f:
            for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"Saved: {npy_path}  (shape={X.shape})")
        print(f"Saved: {ids_path}")
        print(f"Saved: {meta_path}")

        if args.build_faiss:
            try:
                import faiss
                index = faiss.IndexFlatIP(dim) if args.normalize else faiss.IndexFlatL2(dim)
                index.add(X)
                faiss_path = out_base.with_suffix(".faiss")
                faiss.write_index(index, str(faiss_path))
                print(f"Saved: {faiss_path}  (metric={'IP' if args.normalize else 'L2'})")
            except Exception as e:
                print(f"[WARN] FAISS not written: {e}")

if __name__ == "__main__":
    main()
