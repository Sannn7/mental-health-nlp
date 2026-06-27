"""
src/preprocessing/data_loader.py
Downloads and merges the two free Reddit mental health datasets:
  1. Kaggle  – neelghoshal/reddit-mental-health-data
  2. HuggingFace – dreaddit (stress detection)

Usage:
    python -m src.preprocessing.data_loader
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import DATA_RAW, DATA_PROC, KAGGLE_CSV, DREADDIT_NAME, RANDOM_SEED


def download_kaggle_dataset() -> pd.DataFrame:
    """
    Downloads the Kaggle Reddit mental health CSV via the Kaggle API.
    Requires ~/.kaggle/kaggle.json to be set up, OR set env vars:
        KAGGLE_USERNAME and KAGGLE_KEY
    Instructions: https://www.kaggle.com/docs/api
    """
    if KAGGLE_CSV.exists():
        print(f"[data_loader] Kaggle CSV already exists at {KAGGLE_CSV}. Skipping download.")
    else:
        print("[data_loader] Downloading Kaggle dataset ...")
        import kaggle  # triggers auth check
        kaggle.api.dataset_download_files(
            "neelghoshal/reddit-mental-health-data",
            path=str(DATA_RAW),
            unzip=True
        )
        print(f"[data_loader] Saved to {DATA_RAW}")

    df = pd.read_csv(KAGGLE_CSV)
    print(f"[data_loader] Kaggle dataset shape: {df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df


def download_dreaddit() -> pd.DataFrame:
    """
    Downloads the Dreaddit stress-detection dataset from HuggingFace.
    No credentials required.
    """
    print("[data_loader] Loading Dreaddit from HuggingFace ...")
    ds = load_dataset(DREADDIT_NAME)

    frames = []
    for split_name, split_data in ds.items():
        tmp = split_data.to_pandas()
        tmp["_split"] = split_name
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Dreaddit shape: {df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df


def merge_and_save(kaggle_df: pd.DataFrame, dreaddit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonises column names across both datasets and merges them into a
    single dataframe saved to data/processed/merged_raw.csv.

    Unified schema:
        text   – the post body
        label  – string category (depression / anxiety / stress / etc.)
        source – 'kaggle' or 'dreaddit'
    """
    # ── Kaggle normalisation ───────────────────────────────────────────────────
    # Inspect first; rename based on what you see printed above.
    # Common column names in this dataset: 'post', 'label'
    kaggle_cols = kaggle_df.columns.tolist()
    text_col_k  = "post"   if "post"   in kaggle_cols else kaggle_cols[0]
    label_col_k = "label"  if "label"  in kaggle_cols else kaggle_cols[1]

    kaggle_clean = pd.DataFrame({
        "text":   kaggle_df[text_col_k].astype(str),
        "label":  kaggle_df[label_col_k].astype(str),
        "source": "kaggle"
    })

    # ── Dreaddit normalisation ─────────────────────────────────────────────────
    # Dreaddit uses 'text' and 'label' (0 = not stressed, 1 = stressed)
    dreaddit_clean = pd.DataFrame({
        "text":   dreaddit_df["text"].astype(str),
        "label":  dreaddit_df["label"].map({0: "not_stressed", 1: "stress"}),
        "source": "dreaddit"
    })

    merged = pd.concat([kaggle_clean, dreaddit_clean], ignore_index=True)
    merged.drop_duplicates(subset="text", inplace=True)
    merged.dropna(subset=["text", "label"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    out_path = DATA_PROC / "merged_raw.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n[data_loader] Merged dataset: {merged.shape[0]} rows saved to {out_path}")
    print(merged["label"].value_counts())
    return merged


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    kaggle_df   = download_kaggle_dataset()
    dreaddit_df = download_dreaddit()
    merged      = merge_and_save(kaggle_df, dreaddit_df)
    print("\n[data_loader] Done. Run src/preprocessing/cleaner.py next.")
