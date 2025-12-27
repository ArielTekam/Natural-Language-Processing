# search.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import load_jsonl


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic search over FAISS index.")
    p.add_argument("--query", required=True, help="Search query text.")
    p.add_argument("--top_k", type=int, default=5, help="Number of results.")
    p.add_argument("--artifacts_dir", default="artifacts", help="Directory containing faiss.index/meta.jsonl/config.json")
    p.add_argument("--min_score", type=float, default=None, help="Filter results below this score.")
    p.add_argument("--show_text", action="store_true", help="Print text snippet.")
    p.add_argument("--max_text_len", type=int, default=200, help="Max characters of text to show.")
    p.add_argument("--json", action="store_true", help="Output JSON.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    index_path = os.path.join(args.artifacts_dir, "faiss.index")
    meta_path = os.path.join(args.artifacts_dir, "meta.jsonl")
    cfg_path = os.path.join(args.artifacts_dir, "config.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError(
            "Artifacts not found. Expected faiss.index, meta.jsonl, config.json in artifacts_dir."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    normalize = bool(cfg.get("normalize", False))
    model_name = cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print(f"[1/3] Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)

    print(f"[2/3] Loading meta: {meta_path}")
    meta = load_jsonl(meta_path)
    print(f"Meta records: {len(meta)}")

    print(f"[3/3] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    q = args.query
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=normalize).astype(np.float32)

    # Search
    scores, ids = index.search(q_emb, args.top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(meta):
            continue
        if args.min_score is not None and score < args.min_score:
            continue

        r = {
            "rank": rank,
            "id": meta[idx].get("id", idx),
            "label": meta[idx].get("label", "unknown"),
            "score": float(score),
            "title": meta[idx].get("title", ""),
        }

        if args.show_text:
            text = meta[idx].get("text", "")
            if args.max_text_len and len(text) > args.max_text_len:
                text = text[: args.max_text_len] + "…"
            r["text"] = text

        results.append(r)

    if args.json:
        out = {"query": q, "top_k": args.top_k, "min_score": args.min_score, "results": results}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("\n=== QUERY ===")
    print(q)
    print("\n=== RESULTS ===")
    if normalize:
        print("score ∈ [0,1] approx — cosine similarity (IP on L2-normalized embeddings)")
    else:
        print("score — L2 distance (smaller is better)")

    for r in results:
        print(f"\n#{r['rank']} | score={r['score']:.4f} | id={r['id']} | label={r['label']}")
        print(f"Title: {r['title']}")
        if args.show_text:
            print(f"Text : {r.get('text','')}")


if __name__ == "__main__":
    main()
