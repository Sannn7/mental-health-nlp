"""
src/search/semantic_search.py
Module 5 – Semantic Search (Dense Retrieval)

Covers:
  - SBERT sentence embeddings (all-MiniLM-L6-v2)
  - FAISS vector index (flat L2 + IVFFlat for scale)
  - Dense retrieval: query → top-K most semantically similar posts
  - Comparison vs BM25 (keyword baseline)
  - Interactive query loop from the command line

Usage:
    # Build index (one-time):
    python -m src.search.semantic_search --build

    # Query interactively:
    python -m src.search.semantic_search --query "sleep problems and medication"
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    DATA_PROC, DATA_EMB, MODELS_DIR,
    SBERT_MODEL, FAISS_INDEX_PATH, FAISS_META_PATH, TOP_K
)


# ── 1. Build SBERT + FAISS index ─────────────────────────────────────────────

def build_index(batch_size: int = 64) -> None:
    """
    Encodes all posts with SBERT and builds a FAISS flat index.
    Saved to data/embeddings/ for later reuse.
    """
    DATA_EMB.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_PROC / "cleaned.csv"
    df       = pd.read_csv(csv_path)
    df.dropna(subset=["text_clean_deep"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    texts    = df["text_clean_deep"].tolist()
    labels   = df["label"].tolist() if "label" in df.columns else ["unknown"] * len(texts)

    print(f"[search] Loading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)

    print(f"[search] Encoding {len(texts)} posts ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True   # cosine similarity via inner product
    )
    embeddings = embeddings.astype(np.float32)
    dim        = embeddings.shape[1]

    # FAISS flat index (exact search — for < 100K posts this is fine)
    # For > 100K: switch to IndexIVFFlat with nlist=100
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (since normalized)
    index.add(embeddings)
    print(f"[search] FAISS index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"[search] Index saved → {FAISS_INDEX_PATH}")

    # Save metadata (original text + label for display)
    meta = pd.DataFrame({
        "idx":    range(len(texts)),
        "text":   df["text"].tolist() if "text" in df.columns else texts,
        "label":  labels,
        "source": df["source"].tolist() if "source" in df.columns else ["unknown"] * len(texts)
    })
    meta.to_csv(FAISS_META_PATH, index=False)
    print(f"[search] Metadata saved → {FAISS_META_PATH}")


# ── 2. Load index ─────────────────────────────────────────────────────────────

def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run with --build first: python -m src.search.semantic_search --build"
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta  = pd.read_csv(FAISS_META_PATH)
    model = SentenceTransformer(SBERT_MODEL)
    return index, meta, model


# ── 3. Query ──────────────────────────────────────────────────────────────────

def search(query: str, index, meta: pd.DataFrame, model,
           k: int = TOP_K, label_filter: str = None) -> pd.DataFrame:
    """
    Returns top-K most semantically similar posts to the query.

    Args:
        query        : natural language search query
        label_filter : optional — restrict to posts with a specific label
                       e.g. label_filter='depression'
    """
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, k * 3)   # over-fetch for filtering

    results = meta.iloc[indices[0]].copy()
    results["similarity"] = scores[0]

    if label_filter:
        results = results[results["label"] == label_filter]

    results = results.head(k).reset_index(drop=True)
    return results


def display_results(query: str, results: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")
    for _, row in results.iterrows():
        snippet = str(row.get("text", ""))[:200].replace("\n", " ")
        print(f"\n[{row.get('label', '?'):15s}] score={row['similarity']:.4f}")
        print(f"  {snippet} ...")


# ── 4. BM25 baseline comparison ──────────────────────────────────────────────

def bm25_search(query: str, corpus: list[str], k: int = TOP_K):
    """
    Keyword-based BM25 retrieval as a baseline to compare against dense search.
    Requires: pip install rank_bm25
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[search] rank_bm25 not installed. Run: pip install rank-bm25")
        return []

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25             = BM25Okapi(tokenized_corpus)
    scores           = bm25.get_scores(query.lower().split())
    top_idx          = np.argsort(scores)[::-1][:k]
    return [(corpus[i], scores[i]) for i in top_idx]


# ── 5. Interactive query loop ─────────────────────────────────────────────────

def interactive_loop():
    print("\n[search] Loading index ...")
    index, meta, model = load_index()
    corpus = meta["text"].tolist()

    print("[search] Index ready. Type a query (or 'quit' to exit).")
    print("  Tips: 'sleep problems after medication'")
    print("        'feeling hopeless and isolated'")
    print("        'how to tell family about depression'\n")

    while True:
        query = input("Query > ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        # Dense retrieval
        results = search(query, index, meta, model)
        display_results(query, results)

        # BM25 for comparison
        print(f"\n── BM25 top-3 (keyword baseline) ──")
        bm25_res = bm25_search(query, corpus, k=3)
        for snippet, score in bm25_res:
            print(f"  score={score:.2f} | {str(snippet)[:150]} ...")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mental Health NLP – Semantic Search")
    parser.add_argument("--build", action="store_true",
                        help="Build the FAISS index from the cleaned dataset")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query and print results")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        index, meta, model = load_index()
        results = search(args.query, index, meta, model)
        display_results(args.query, results)
    else:
        interactive_loop()
