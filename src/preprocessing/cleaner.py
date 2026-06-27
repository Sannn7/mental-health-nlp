"""
src/preprocessing/cleaner.py
Module 1 – Text Preprocessing

Covers:
  - Reddit-specific noise removal (markdown, URLs, usernames, subreddit tags)
  - Lowercasing, punctuation stripping
  - Tokenization comparison  : spaCy  vs  NLTK  vs  Whitespace
  - Stopword removal
  - Stemming   (PorterStemmer)
  - Lemmatization (spaCy)
  - POS tagging  → keeps only adjectives + nouns (for symptom extraction)
  - Saves both a 'clean' version (for deep learning) and a
    'preprocessed' version (for classical ML with stemming/lemmatization)

Usage:
    python -m src.preprocessing.cleaner
"""

import re
import sys
import string
import pandas as pd
import spacy
import nltk
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import DATA_PROC

# ── One-time NLTK downloads ─────────────────────────────────────────────────
nltk.download("punkt",        quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ── spaCy model (run: python -m spacy download en_core_web_sm) ──────────────
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("[cleaner] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()


# ── 1. Reddit noise removal ──────────────────────────────────────────────────

def remove_reddit_noise(text: str) -> str:
    """Strip URLs, subreddit refs, usernames, markdown, HTML entities."""
    text = re.sub(r"http\S+|www\.\S+",        "",  text)   # URLs
    text = re.sub(r"r/\w+",                   "",  text)   # subreddit names
    text = re.sub(r"u/\w+",                   "",  text)   # usernames
    text = re.sub(r"\[.*?\]\(.*?\)",          "",  text)   # markdown links
    text = re.sub(r"[>*_~`]",                 "",  text)   # markdown symbols
    text = re.sub(r"&\w+;",                   "",  text)   # HTML entities
    text = re.sub(r"\s+",                     " ", text)   # collapse whitespace
    return text.strip()


def basic_clean(text: str) -> str:
    """Lowercase + remove punctuation after noise removal."""
    text = remove_reddit_noise(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# ── 2. Tokenization (three approaches shown for comparison) ─────────────────

def tokenize_whitespace(text: str) -> list[str]:
    return text.split()


def tokenize_nltk(text: str) -> list[str]:
    return word_tokenize(text)


def tokenize_spacy(text: str) -> list[str]:
    if nlp is None:
        return tokenize_nltk(text)
    return [tok.text for tok in nlp(text)]


# ── 3. Stopword removal ──────────────────────────────────────────────────────

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ── 4. Stemming ──────────────────────────────────────────────────────────────

def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens]


# ── 5. Lemmatization (spaCy) ─────────────────────────────────────────────────

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Re-join tokens → run spaCy → extract lemmas."""
    if nlp is None:
        return tokens
    doc = nlp(" ".join(tokens))
    return [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct]


# ── 6. POS tagging → keep adjectives + nouns (symptom signal) ───────────────

def extract_adj_noun(text: str) -> list[str]:
    """
    Returns only adjectives (JJ*) and nouns (NN*) from a text.
    These carry the most symptom-descriptive signal.
    E.g. "I feel exhausted and worthless every morning"
         → ['exhausted', 'worthless', 'morning']
    """
    tokens = tokenize_nltk(basic_clean(text))
    tagged = nltk.pos_tag(tokens)
    keep   = {"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"}
    return [word for word, tag in tagged if tag in keep]


# ── 7. Master pipeline ───────────────────────────────────────────────────────

def preprocess_for_classical(text: str) -> str:
    """
    For TF-IDF / Word2Vec / LDA:
    clean → tokenize (NLTK) → remove stopwords → lemmatize → rejoin
    """
    cleaned = basic_clean(text)
    tokens  = tokenize_nltk(cleaned)
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_for_deep(text: str) -> str:
    """
    For BERT / DistilBERT: only noise removal + lowercase.
    The transformer tokenizer handles everything else.
    """
    return basic_clean(text)


# ── 8. Run on merged dataset ─────────────────────────────────────────────────

def run_cleaning_pipeline(input_csv: Path = None, output_csv: Path = None) -> pd.DataFrame:
    input_csv  = input_csv  or DATA_PROC / "merged_raw.csv"
    output_csv = output_csv or DATA_PROC / "cleaned.csv"

    print(f"[cleaner] Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    print(f"[cleaner] {len(df)} rows loaded.")

    tqdm.pandas(desc="[cleaner] Cleaning for deep learning")
    df["text_clean_deep"] = df["text"].progress_apply(preprocess_for_deep)

    tqdm.pandas(desc="[cleaner] Cleaning for classical NLP")
    df["text_clean_classical"] = df["text"].progress_apply(preprocess_for_classical)

    tqdm.pandas(desc="[cleaner] Extracting adj+noun tokens (symptom signal)")
    df["tokens_adj_noun"] = df["text"].progress_apply(
        lambda t: " ".join(extract_adj_noun(t))
    )

    # Drop rows where cleaning left an empty string
    df = df[df["text_clean_deep"].str.strip().astype(bool)]
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"\n[cleaner] Saved cleaned dataset → {output_csv}")
    print(f"[cleaner] Final shape: {df.shape}")
    return df


# ── Quick demo ───────────────────────────────────────────────────────────────

def demo():
    sample = (
        "I've been feeling so **worthless** lately. u/throwaway12345 "
        "mentioned r/depression but I can't even get out of bed "
        "https://t.co/xyz. Everything feels hopeless &amp; exhausting."
    )

    print("=" * 60)
    print("ORIGINAL:\n", sample)
    print("\nAFTER NOISE REMOVAL:\n", remove_reddit_noise(sample))
    print("\nBASIC CLEAN:\n", basic_clean(sample))
    print("\nNLTK TOKENS:", tokenize_nltk(basic_clean(sample))[:10], "...")
    print("\nAFTER STOPWORD REMOVAL:", remove_stopwords(tokenize_nltk(basic_clean(sample)))[:10])
    print("\nCLASSICAL PIPELINE OUTPUT:\n", preprocess_for_classical(sample))
    print("\nDEEP LEARNING PIPELINE OUTPUT:\n", preprocess_for_deep(sample))
    print("\nADJ + NOUN TOKENS:", extract_adj_noun(sample))
    print("=" * 60)


if __name__ == "__main__":
    demo()
    # Uncomment after running data_loader.py:
    # run_cleaning_pipeline()
