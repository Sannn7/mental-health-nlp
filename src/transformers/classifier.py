"""
src/transformers/classifier.py
Module 3b – Transformer Fine-tuning (DistilBERT)

Covers:
  - Tokenization with Hugging Face tokenizer (WordPiece / BPE)
  - DistilBERT fine-tuning for multi-class classification
  - Evaluation: accuracy, weighted F1, per-class report
  - Saving the fine-tuned model

Usage:
    python -m src.transformers.classifier
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate as hf_evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    DATA_PROC, MODELS_DIR, RESULTS_DIR,
    CLASSIFIER_MODEL, CLASSIFIER_EPOCHS, CLASSIFIER_LR,
    CLASSIFIER_BATCH, MAX_TOKEN_LEN, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE
)


# ── 1. Dataset class ─────────────────────────────────────────────────────────

class RedditDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── 2. Metrics ────────────────────────────────────────────────────────────────

accuracy_metric = hf_evaluate.load("accuracy")
f1_metric       = hf_evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds          = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1_weighted": f1}


# ── 3. Load & prepare data ────────────────────────────────────────────────────

def load_data():
    path = DATA_PROC / "cleaned.csv"
    df   = pd.read_csv(path)
    df.dropna(subset=["text_clean_deep", "label"], inplace=True)

    le     = LabelEncoder()
    labels = le.fit_transform(df["label"].tolist())
    texts  = df["text_clean_deep"].tolist()

    print(f"[classifier] {len(texts)} samples | {len(le.classes_)} classes: {le.classes_.tolist()}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_ratio,
        random_state=RANDOM_SEED, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ── 4. Train ──────────────────────────────────────────────────────────────────

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, le = load_data()
    n_labels = len(le.classes_)

    print(f"[classifier] Loading tokenizer: {CLASSIFIER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)

    train_ds = RedditDataset(X_train, y_train, tokenizer, MAX_TOKEN_LEN)
    val_ds   = RedditDataset(X_val,   y_val,   tokenizer, MAX_TOKEN_LEN)
    test_ds  = RedditDataset(X_test,  y_test,  tokenizer, MAX_TOKEN_LEN)

    print(f"[classifier] Loading model: {CLASSIFIER_MODEL} ({n_labels} labels)")
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL, num_labels=n_labels
    )

    output_dir = str(MODELS_DIR / "distilbert_classifier")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CLASSIFIER_EPOCHS,
        per_device_train_batch_size=CLASSIFIER_BATCH,
        per_device_eval_batch_size=CLASSIFIER_BATCH * 2,
        learning_rate=CLASSIFIER_LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to="none"    # set to "wandb" if you have wandb set up
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\n[classifier] Starting training ...")
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[classifier] Evaluating on test set ...")
    preds_output = trainer.predict(test_ds)
    y_pred       = np.argmax(preds_output.predictions, axis=-1)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(RESULTS_DIR / "classifier_report.csv")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save final model + tokenizer ─────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[classifier] Model saved → {output_dir}")

    # ── Save label mapping ────────────────────────────────────────────────────
    import json
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(RESULTS_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    return trainer, le


# ── 5. Inference helper ────────────────────────────────────────────────────────

def predict(text: str, model_dir: str = None, label_map: dict = None) -> str:
    """
    Single-post inference.
    Example:
        predict("I can't stop crying and I don't know why")
    """
    import json
    model_dir = model_dir or str(MODELS_DIR / "distilbert_classifier")

    if label_map is None:
        with open(RESULTS_DIR / "label_map.json") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    enc = tokenizer(
        text, max_length=MAX_TOKEN_LEN,
        truncation=True, padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().tolist()

    ranked = sorted(
        [(label_map[i], round(p, 4)) for i, p in enumerate(probs)],
        key=lambda x: -x[1]
    )
    print(f"\n[classifier] Input: \"{text[:80]}...\"")
    print("[classifier] Predictions:")
    for label, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"  {label:25s} {prob:.4f}  {bar}")

    return ranked[0][0]


if __name__ == "__main__":
    train()
