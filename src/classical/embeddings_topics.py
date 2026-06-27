"""
src/classical/embeddings_topics.py
Module 2b – Word Embeddings, VADER Sentiment & Topic Modelling

Covers:
  - Word2Vec (gensim) training on corpus
  - Semantic similarity between symptom phrases
  - VADER rule-based sentiment analysis
  - LDA topic modelling via gensim (10 latent topics)
  - BERTopic (neural topic modelling) – comparison
  - Visualizations: topic word clouds, VADER distribution

Usage:
    python -m src.classical.embeddings_topics
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    DATA_PROC, DATA_EMB, MODELS_DIR, RESULTS_DIR, VIZ_DIR,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS,
    LDA_N_TOPICS, LDA_N_WORDS_SHOW, RANDOM_SEED
)


# ── 1. Load cleaned data ─────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATA_PROC / "cleaned.csv"
    df   = pd.read_csv(path)
    df.dropna(subset=["text_clean_classical", "text_clean_deep"], inplace=True)
    return df


def get_token_lists(df: pd.DataFrame) -> list[list[str]]:
    """Convert preprocessed text column to list of token lists for gensim."""
    return [row.split() for row in df["text_clean_classical"]]


# ── 2. Word2Vec ──────────────────────────────────────────────────────────────

def train_word2vec(token_lists: list[list[str]]) -> Word2Vec:
    print("[w2v] Training Word2Vec ...")
    model = Word2Vec(
        sentences=token_lists,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        epochs=W2V_EPOCHS,
        workers=4,
        seed=RANDOM_SEED
    )
    save_path = MODELS_DIR / "word2vec.model"
    model.save(str(save_path))
    print(f"[w2v] Model saved → {save_path}")
    return model


def demo_word2vec(model: Word2Vec):
    """Show nearest neighbours for mental-health-relevant terms."""
    probe_words = ["depression", "anxiety", "hopeless", "therapy", "medication"]
    print("\n[w2v] Nearest neighbours for mental-health terms:")
    for word in probe_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"  {word:15s} → {[w for w, _ in similar]}")
        else:
            print(f"  {word:15s} → (not in vocabulary)")

    # Analogy: depression - sadness + hope ≈ ?
    try:
        result = model.wv.most_similar(
            positive=["depression", "hope"],
            negative=["sadness"], topn=3
        )
        print(f"\n[w2v] Analogy [depression + hope - sadness]: {result}")
    except KeyError:
        pass


# ── 3. VADER Sentiment ───────────────────────────────────────────────────────

def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies VADER to every post and adds compound/label columns.
    VADER is rule-based so it needs the raw (minimally cleaned) text,
    not the aggressively preprocessed version.
    """
    print("\n[vader] Running VADER sentiment analysis ...")
    analyzer = SentimentIntensityAnalyzer()

    def score(text: str) -> dict:
        return analyzer.polarity_scores(str(text))

    tqdm.pandas(desc="[vader] Scoring")
    scores = df["text_clean_deep"].progress_apply(score)

    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])

    # Label: compound >= 0.05 → positive, <= -0.05 → negative, else neutral
    def sentiment_label(c: float) -> str:
        if   c >= 0.05: return "positive"
        elif c <= -0.05: return "negative"
        else:            return "neutral"

    df["vader_sentiment"] = df["vader_compound"].apply(sentiment_label)

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["vader_compound"].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="none")
    axes[0].set_title("VADER Compound Score Distribution")
    axes[0].set_xlabel("Compound Score")

    df["vader_sentiment"].value_counts().plot(kind="bar", ax=axes[1], color=["#e05c5c","#6dbf67","#8fb9d5"])
    axes[1].set_title("Sentiment Label Counts")
    axes[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "vader_distribution.png", dpi=150)
    plt.close()
    print(f"[vader] Distribution → {VIZ_DIR / 'vader_distribution.png'}")

    # Save labelled data
    out = DATA_PROC / "with_vader.csv"
    df.to_csv(out, index=False)
    print(f"[vader] Saved → {out}")
    return df


# ── 4. LDA Topic Modelling ───────────────────────────────────────────────────

def run_lda(token_lists: list[list[str]]) -> LdaModel:
    print(f"\n[lda] Building LDA with {LDA_N_TOPICS} topics ...")
    dictionary  = Dictionary(token_lists)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus      = [dictionary.doc2bow(tokens) for tokens in token_lists]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=LDA_N_TOPICS,
        passes=10,
        random_state=RANDOM_SEED,
        alpha="auto"
    )

    print("\n[lda] Top words per topic:")
    topic_data = {}
    for topic_id in range(LDA_N_TOPICS):
        words = lda.show_topic(topic_id, topn=LDA_N_WORDS_SHOW)
        word_list = [w for w, _ in words]
        topic_data[f"topic_{topic_id}"] = word_list
        print(f"  Topic {topic_id:2d}: {word_list}")

    # Save topic words as JSON
    topics_path = RESULTS_DIR / "lda_topics.json"
    with open(topics_path, "w") as f:
        json.dump(topic_data, f, indent=2)
    print(f"[lda] Topics saved → {topics_path}")

    # Save model
    lda.save(str(MODELS_DIR / "lda.model"))
    return lda


# ── 5. BERTopic (neural topic modelling) ────────────────────────────────────

def run_bertopic(texts: list[str], sample_size: int = 5000) -> BERTopic:
    """
    BERTopic uses SBERT embeddings + UMAP + HDBSCAN.
    Use a sample for speed; increase sample_size for better results.
    """
    print(f"\n[bertopic] Running BERTopic on {sample_size} samples ...")
    sample = texts[:sample_size]

    topic_model = BERTopic(
        language="english",
        calculate_probabilities=False,
        verbose=True,
        nr_topics="auto"
    )
    topics, _   = topic_model.fit_transform(sample)

    info = topic_model.get_topic_info()
    print(f"[bertopic] Found {len(info) - 1} topics (excluding outliers)")
    print(info.head(12).to_string(index=False))

    topic_model.save(str(MODELS_DIR / "bertopic"))
    info.to_csv(RESULTS_DIR / "bertopic_topics.csv", index=False)
    print(f"[bertopic] Saved → {MODELS_DIR / 'bertopic'}")
    return topic_model


# ── 6. Master runner ─────────────────────────────────────────────────────────

def run():
    for d in [MODELS_DIR, RESULTS_DIR, VIZ_DIR, DATA_EMB]:
        d.mkdir(parents=True, exist_ok=True)

    df          = load_data()
    token_lists = get_token_lists(df)
    texts_deep  = df["text_clean_deep"].tolist()

    # Word2Vec
    w2v_model = train_word2vec(token_lists)
    demo_word2vec(w2v_model)

    # VADER
    df = run_vader(df)

    # LDA
    run_lda(token_lists)

    # BERTopic (neural – slower, more powerful)
    run_bertopic(texts_deep)

    print("\n[embeddings_topics] Module 2b complete.")


if __name__ == "__main__":
    run()
