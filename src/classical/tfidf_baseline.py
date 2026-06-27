"""
src/classical/tfidf_baseline.py
Module 2a – Classical NLP Baseline

Covers:
  - Bag of Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
  - Logistic Regression, SVM, Random Forest classifiers
  - Evaluation: accuracy, F1, confusion matrix
  - Saves best model to outputs/models/

Usage:
    python -m src.classical.tfidf_baseline
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    DATA_PROC, MODELS_DIR, RESULTS_DIR, VIZ_DIR,
    TFIDF_MAX_FEATURES, RANDOM_SEED, TEST_SIZE
)


def load_data() -> tuple[list, list]:
    path = DATA_PROC / "cleaned.csv"
    print(f"[tfidf] Loading {path} ...")
    df = pd.read_csv(path)
    df.dropna(subset=["text_clean_classical", "label"], inplace=True)
    X = df["text_clean_classical"].tolist()
    y = df["label"].tolist()
    print(f"[tfidf] {len(X)} samples, {len(set(y))} classes: {sorted(set(y))}")
    return X, y


def build_pipelines() -> dict:
    """
    Returns a dict of {name: sklearn Pipeline}.
    Each pipeline pairs a vectorizer with a classifier.
    """
    return {
        "BoW + LogReg": Pipeline([
            ("vec", CountVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "TF-IDF + LogReg": Pipeline([
            ("vec", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "TF-IDF + SVM": Pipeline([
            ("vec", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000, random_state=RANDOM_SEED))
        ]),
        "TF-IDF + RF": Pipeline([
            ("vec", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1))
        ]),
    }


def evaluate(name: str, pipeline: Pipeline, X_test: list, y_test: list) -> dict:
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    print(f"\n── {name} ──────────────────────────")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 (wtd) : {f1:.4f}")
    print(classification_report(y_test, y_pred))
    return {"name": name, "accuracy": acc, "f1_weighted": f1,
            "y_pred": y_pred, "y_test": y_test}


def plot_confusion_matrix(y_test, y_pred, labels: list, title: str, save_path: Path):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels,
                yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[tfidf] Saved confusion matrix → {save_path}")


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    X, y     = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    pipelines = build_pipelines()
    results   = []
    best_f1   = 0
    best_name = None
    best_pipe = None

    for name, pipe in pipelines.items():
        print(f"\n[tfidf] Training: {name} ...")
        pipe.fit(X_train, y_train)
        res = evaluate(name, pipe, X_test, y_test)
        results.append(res)
        if res["f1_weighted"] > best_f1:
            best_f1   = res["f1_weighted"]
            best_name = name
            best_pipe = pipe

    # ── Save best model ────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "classical_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\n[tfidf] Best model: '{best_name}' (F1={best_f1:.4f}) → {model_path}")

    # ── Confusion matrix for best model ───────────────────────────────────────
    best_res = next(r for r in results if r["name"] == best_name)
    labels   = sorted(set(y_test))
    plot_confusion_matrix(
        best_res["y_test"], best_res["y_pred"], labels,
        title=f"Confusion Matrix – {best_name}",
        save_path=VIZ_DIR / "classical_cm.png"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame([{"Model": r["name"], "Accuracy": r["accuracy"],
                              "F1 (weighted)": r["f1_weighted"]} for r in results])
    summary.sort_values("F1 (weighted)", ascending=False, inplace=True)
    summary.to_csv(RESULTS_DIR / "classical_results.csv", index=False)
    print("\n[tfidf] Results summary:")
    print(summary.to_string(index=False))

    return best_pipe


if __name__ == "__main__":
    run()
