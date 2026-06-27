"""
run_pipeline.py
Master runner — executes all 5 modules in sequence.

Usage:
    python run_pipeline.py --all          # full pipeline
    python run_pipeline.py --module 1     # only Module 1 (preprocessing)
    python run_pipeline.py --module 2     # only Module 2 (classical NLP)
    python run_pipeline.py --module 3     # only Module 3 (transformers + NER)
    python run_pipeline.py --module 4     # only Module 4 (summarization)
    python run_pipeline.py --module 5     # only Module 5 (semantic search)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_module_1():
    print("\n" + "="*65)
    print("MODULE 1 — Data Loading & Preprocessing")
    print("="*65)
    from src.preprocessing.data_loader import download_dreaddit, merge_and_save

    # Note: Kaggle download requires credentials.
    # If you placed the CSV manually, data_loader skips the API call.
    try:
        from src.preprocessing.data_loader import download_kaggle_dataset
        kaggle_df = download_kaggle_dataset()
    except Exception as e:
        print(f"[pipeline] Kaggle download skipped ({e}). Using Dreaddit only.")
        import pandas as pd
        kaggle_df = pd.DataFrame(columns=["post", "label"])

    dreaddit_df = download_dreaddit()
    merge_and_save(kaggle_df, dreaddit_df)

    from src.preprocessing.cleaner import run_cleaning_pipeline
    run_cleaning_pipeline()
    print("\n[pipeline] ✓ Module 1 complete")


def run_module_2():
    print("\n" + "="*65)
    print("MODULE 2 — Classical NLP (TF-IDF, Word2Vec, VADER, LDA)")
    print("="*65)
    from src.classical.tfidf_baseline import run as run_tfidf
    from src.classical.embeddings_topics import run as run_embeddings
    run_tfidf()
    run_embeddings()
    print("\n[pipeline] ✓ Module 2 complete")


def run_module_3():
    print("\n" + "="*65)
    print("MODULE 3 — Transformers (NER + DistilBERT Classifier)")
    print("="*65)
    from src.transformers.ner_extractor import run_ner_pipeline
    from src.transformers.classifier import train

    run_ner_pipeline(sample_size=2000)   # sample for speed; remove limit for full run
    train()
    print("\n[pipeline] ✓ Module 3 complete")


def run_module_4():
    print("\n" + "="*65)
    print("MODULE 4 — Summarization (BART + ROUGE)")
    print("="*65)
    from src.summarization.summarizer import run as run_summarizer
    run_summarizer()
    print("\n[pipeline] ✓ Module 4 complete")


def run_module_5():
    print("\n" + "="*65)
    print("MODULE 5 — Semantic Search (SBERT + FAISS)")
    print("="*65)
    from src.search.semantic_search import build_index
    build_index()
    print("\n[pipeline] ✓ Module 5 complete")
    print("[pipeline] You can now run interactive search:")
    print("           python -m src.search.semantic_search --query 'your query here'")


MODULE_MAP = {
    1: run_module_1,
    2: run_module_2,
    3: run_module_3,
    4: run_module_4,
    5: run_module_5,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mental Health NLP Pipeline")
    parser.add_argument("--all",    action="store_true", help="Run all modules")
    parser.add_argument("--module", type=int, choices=[1,2,3,4,5],
                        help="Run a single module (1–5)")
    args = parser.parse_args()

    if args.all:
        for m_id, m_fn in MODULE_MAP.items():
            t0 = time.time()
            m_fn()
            elapsed = time.time() - t0
            print(f"[pipeline] Module {m_id} finished in {elapsed:.1f}s")
    elif args.module:
        MODULE_MAP[args.module]()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  Step 1: python run_pipeline.py --module 1")
        print("  Step 2: python run_pipeline.py --module 2")
        print("  ...and so on")
