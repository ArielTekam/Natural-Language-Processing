# index.py
from __future__ import annotations

import argparse
import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import iter_text_gz, save_jsonl


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build FAISS index for semantic search (text.gz).")
    p.add_argument("--input", required=True, help="Path to input .gz file (contains plain text).")
    p.add_argument("--out_dir", default="artifacts", help="Directory to store index + meta + config.")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  help="SentenceTransformer model name.")
    p.add_argument("--limit", type=int, default=None, help="Max number of documents to index.")
    p.add_argument("--mode", choices=["line", "paragraph"], default="line",
                  help="How to split documents from text: line or paragraph.")
    p.add_argument("--min_chars", type=int, default=30, help="Skip docs shorter than this.")
    p.add_argument("--batch_size", type=int, default=64, help="Encoding batch size.")
    p.add_argument("--normalize", action="store_true", help="L2-normalize vectors for cosine similarity.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] Loading documents: {args.input} (mode={args.mode})")
    docs = list(iter_text_gz(args.input, mode=args.mode, min_chars=args.min_chars, limit=args.limit))
    print(f"Loaded docs: {len(docs)}")

    if len(docs) == 0:
        raise RuntimeError(
            "0 documents loaded. Your .gz probably contains very short lines or empty text. "
            "Try --mode paragraph and/or reduce --min_chars."
        )

    print(f"[2/4] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = [d["text"] for d in docs]

    print(f"[3/4] Encoding (batch_size={args.batch_size})...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,  # SentenceTransformers can normalize directly
    ).astype(np.float32)

    # If user didn't use normalize_embeddings but wants normalization:
    if args.normalize:
        # already normalized above; kept for clarity
        pass

    dim = emb.shape[1]
    print(f"Embeddings shape: {emb.shape}")

    print("[4/4] Building FAISS index...")
    # cosine similarity with normalized vectors == inner product
    index = faiss.IndexFlatIP(dim) if args.normalize else faiss.IndexFlatL2(dim)
    index.add(emb)

    index_path = os.path.join(args.out_dir, "faiss.index")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    cfg_path = os.path.join(args.out_dir, "config.json")

    faiss.write_index(index, index_path)

    # meta: on stocke title/label/text (texte utile pour montrer l’extrait)
    save_jsonl(meta_path, docs)

    cfg = {
        "input": args.input,
        "split_mode": args.mode,
        "min_chars": args.min_chars,
        "limit": args.limit,
        "model": args.model,
        "normalize": bool(args.normalize),
        "batch_size": args.batch_size,
        "index_type": "IP" if args.normalize else "L2",
        "dim": dim,
        "count": len(docs),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Indexed docs: {len(docs)}")
    print(f"Index:  {index_path}")
    print(f"Meta:   {meta_path}")
    print(f"Config: {cfg_path}")


if __name__ == "__main__":
    main()
