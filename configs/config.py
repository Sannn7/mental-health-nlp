"""
configs/config.py
Central configuration for the Mental Health NLP pipeline.
Change paths and model names here — nothing else needs to be touched.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
DATA_EMB      = ROOT / "data" / "embeddings"
OUTPUTS       = ROOT / "outputs"
MODELS_DIR    = OUTPUTS / "models"
RESULTS_DIR   = OUTPUTS / "results"
VIZ_DIR       = OUTPUTS / "visualizations"

# ── Dataset ────────────────────────────────────────────────────────────────────
# Kaggle: neelghoshal/reddit-mental-health-data  → place CSV here after download
KAGGLE_CSV    = DATA_RAW / "reddit_mental_health.csv"
DREADDIT_NAME = "dreaddit"          # HuggingFace dataset id

# Column names in the Kaggle CSV (adjust if different after download)
TEXT_COL      = "post"
LABEL_COL     = "label"             # e.g. depression / anxiety / etc.

# ── Preprocessing ──────────────────────────────────────────────────────────────
MAX_TOKEN_LEN = 512
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15

# ── Classical NLP ─────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 10_000
LDA_N_TOPICS        = 10
LDA_N_WORDS_SHOW    = 8
W2V_VECTOR_SIZE     = 300
W2V_WINDOW          = 5
W2V_MIN_COUNT       = 3
W2V_EPOCHS          = 10

# ── Transformer ───────────────────────────────────────────────────────────────
CLASSIFIER_MODEL    = "distilbert-base-uncased"
CLASSIFIER_EPOCHS   = 3
CLASSIFIER_LR       = 2e-5
CLASSIFIER_BATCH    = 16

# ── Summarization ─────────────────────────────────────────────────────────────
SUMMARIZER_MODEL    = "facebook/bart-large-cnn"
SUMMARY_MAX_LEN     = 130
SUMMARY_MIN_LEN     = 40

# ── Semantic Search ───────────────────────────────────────────────────────────
SBERT_MODEL         = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH    = DATA_EMB / "faiss_index.bin"
FAISS_META_PATH     = DATA_EMB / "faiss_meta.csv"
TOP_K               = 10
