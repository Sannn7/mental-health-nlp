"""
src/summarization/summarizer.py
Module 4 – Abstractive Summarization

Covers:
  - BART-large-cnn for abstractive summarization
  - Extractive summarization baseline (TF-IDF sentence ranking)
  - Summarize entire topic clusters → human-readable theme descriptions
  - ROUGE evaluation against reference summaries
  - Output: one summary per topic cluster, saved to results/

Usage:
    python -m src.summarization.summarizer
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    DATA_PROC, RESULTS_DIR, MODELS_DIR,
    SUMMARIZER_MODEL, SUMMARY_MAX_LEN, SUMMARY_MIN_LEN
)


# ── 1. Load summarization pipeline ──────────────────────────────────────────

def load_summarizer():
    print(f"[summarizer] Loading {SUMMARIZER_MODEL} ...")
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        device=-1   # CPU; set to 0 for GPU
    )
    print("[summarizer] Model loaded.")
    return summarizer


# ── 2. Extractive baseline: TF-IDF sentence ranking ─────────────────────────

def extractive_summarize(text: str, n_sentences: int = 3) -> str:
    """
    Simple extractive summary: picks the top-N sentences by TF-IDF score.
    Serves as a cheap baseline to compare against BART.
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    if len(sentences) <= n_sentences:
        return text

    tfidf  = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    scores = np.asarray(tfidf.sum(axis=1)).flatten()
    top_idx = np.argsort(scores)[::-1][:n_sentences]
    # Return in original order
    summary_sentences = [sentences[i] for i in sorted(top_idx)]
    return " ".join(summary_sentences)


# ── 3. BART abstractive summarization ───────────────────────────────────────

def abstractive_summarize(text: str, summarizer) -> str:
    """
    Truncates to 1024 tokens (BART limit) and runs abstractive summarization.
    """
    truncated = " ".join(text.split()[:900])   # rough word-level truncation
    if len(truncated.split()) < 50:
        return truncated  # too short to summarize

    try:
        result = summarizer(
            truncated,
            max_length=SUMMARY_MAX_LEN,
            min_length=SUMMARY_MIN_LEN,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception as e:
        print(f"[summarizer] Warning: {e}")
        return truncated[:300]


# ── 4. Cluster-level summarization ───────────────────────────────────────────

def summarize_clusters(df: pd.DataFrame, cluster_col: str, text_col: str,
                        summarizer, n_posts_per_cluster: int = 20) -> pd.DataFrame:
    """
    For each topic cluster:
      1. Concatenates a sample of posts into a 'mega-document'
      2. Runs extractive + abstractive summarization
      3. Returns a DataFrame with one row per cluster
    """
    clusters = sorted(df[cluster_col].dropna().unique())
    records  = []

    for cluster_id in tqdm(clusters, desc="[summarizer] Summarizing clusters"):
        cluster_posts = df[df[cluster_col] == cluster_id][text_col].dropna().tolist()
        sample        = cluster_posts[:n_posts_per_cluster]
        mega_text     = " ".join(sample)

        extractive  = extractive_summarize(mega_text, n_sentences=3)
        abstractive = abstractive_summarize(mega_text, summarizer)

        records.append({
            "cluster_id":           cluster_id,
            "n_posts":              len(cluster_posts),
            "extractive_summary":   extractive,
            "abstractive_summary":  abstractive
        })
        print(f"\n── Cluster {cluster_id} ({len(cluster_posts)} posts) ──────────")
        print(f"   BART: {abstractive}")

    return pd.DataFrame(records)


# ── 5. ROUGE evaluation ──────────────────────────────────────────────────────

def evaluate_rouge(generated: list[str], references: list[str]) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L between generated and reference summaries.
    If you don't have ground-truth summaries, compare extractive vs abstractive.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg    = {"rouge1": [], "rouge2": [], "rougeL": []}

    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        for key in agg:
            agg[key].append(scores[key].fmeasure)

    return {k: round(np.mean(v), 4) for k, v in agg.items()}


# ── 6. Master runner ─────────────────────────────────────────────────────────

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data — prefer with_vader.csv (has sentiment) or fall back to cleaned
    csv_path = DATA_PROC / "with_vader.csv"
    if not csv_path.exists():
        csv_path = DATA_PROC / "cleaned.csv"

    df = pd.read_csv(csv_path)
    df.dropna(subset=["text_clean_deep"], inplace=True)
    print(f"[summarizer] Loaded {len(df)} rows from {csv_path.name}")

    # Use 'label' as a proxy cluster column if BERTopic hasn't run yet
    if "bertopic_topic" in df.columns:
        cluster_col = "bertopic_topic"
    else:
        cluster_col = "label"
        print(f"[summarizer] Using '{cluster_col}' as cluster column.")

    summarizer = load_summarizer()

    cluster_summaries = summarize_clusters(
        df, cluster_col=cluster_col,
        text_col="text_clean_deep",
        summarizer=summarizer,
        n_posts_per_cluster=15
    )

    # ── ROUGE: extractive vs abstractive ─────────────────────────────────────
    rouge = evaluate_rouge(
        generated=cluster_summaries["abstractive_summary"].tolist(),
        references=cluster_summaries["extractive_summary"].tolist()
    )
    print(f"\n[summarizer] ROUGE scores (abstractive vs extractive baseline):")
    for k, v in rouge.items():
        print(f"  {k}: {v}")

    # Save
    out = RESULTS_DIR / "cluster_summaries.csv"
    cluster_summaries.to_csv(out, index=False)
    print(f"\n[summarizer] Saved → {out}")

    with open(RESULTS_DIR / "rouge_scores.json", "w") as f:
        json.dump(rouge, f, indent=2)

    return cluster_summaries


if __name__ == "__main__":
    run()
