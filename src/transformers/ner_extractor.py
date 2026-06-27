"""
src/transformers/ner_extractor.py
Module 3a – Named Entity Recognition (NER)

Covers:
  - spaCy pre-trained NER  (general entities: ORG, PERSON, DATE, GPE)
  - HuggingFace BERT-NER  (fine-grained token classification)
  - Custom pattern matching (medications, diagnosis terms, temporal markers)
    using spaCy's EntityRuler

Output: DataFrame with extracted entities per post saved to processed/

Usage:
    python -m src.transformers.ner_extractor
"""

import sys
import re
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from spacy.language import Language

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import DATA_PROC, RESULTS_DIR


# ── Common medication and diagnosis vocabulary ────────────────────────────────
# (expand this list as you explore the data)

MEDICATIONS = [
    "prozac", "fluoxetine", "zoloft", "sertraline", "lexapro", "escitalopram",
    "wellbutrin", "bupropion", "effexor", "venlafaxine", "celexa", "citalopram",
    "paxil", "paroxetine", "cymbalta", "duloxetine", "adderall", "ritalin",
    "xanax", "alprazolam", "klonopin", "clonazepam", "ativan", "lorazepam",
    "lithium", "lamictal", "lamotrigine", "seroquel", "quetiapine", "abilify",
    "aripiprazole", "risperdal", "risperidone", "ambien", "zolpidem"
]

DIAGNOSES = [
    "depression", "major depressive disorder", "mdd",
    "anxiety", "generalized anxiety disorder", "gad",
    "bipolar", "bipolar disorder",
    "ptsd", "post traumatic stress",
    "ocd", "obsessive compulsive",
    "adhd", "attention deficit",
    "bpd", "borderline personality",
    "schizophrenia", "psychosis",
    "eating disorder", "anorexia", "bulimia",
    "panic disorder", "panic attack",
    "social anxiety", "agoraphobia", "insomnia"
]

TEMPORAL_PATTERNS = [
    r"\d+\s+(?:years?|months?|weeks?|days?)\s+(?:ago|of\s+(?:this|my))",
    r"since\s+(?:\w+\s+)?\d{4}",
    r"for\s+(?:the\s+(?:last|past)\s+)?\d+\s+(?:years?|months?|weeks?)"
]


# ── 1. spaCy EntityRuler with custom patterns ────────────────────────────────

def build_spacy_pipeline() -> Language:
    """Loads en_core_web_sm and adds an EntityRuler for custom terms."""
    nlp = spacy.load("en_core_web_sm")

    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})

    patterns = []
    for med in MEDICATIONS:
        patterns.append({"label": "MEDICATION", "pattern": med})
        patterns.append({"label": "MEDICATION", "pattern": med.capitalize()})

    for dx in DIAGNOSES:
        patterns.append({"label": "DIAGNOSIS", "pattern": dx})
        patterns.append({"label": "DIAGNOSIS", "pattern": dx.upper()})

    ruler.add_patterns(patterns)
    return nlp


def extract_spacy_entities(text: str, nlp: Language) -> dict:
    """Returns dict of entity_type → list of entity strings."""
    doc    = nlp(str(text)[:1000])  # cap at 1000 chars for speed
    result = {}
    for ent in doc.ents:
        result.setdefault(ent.label_, []).append(ent.text)

    # Also extract temporal patterns via regex
    temporals = []
    for pat in TEMPORAL_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        temporals.extend(matches)
    if temporals:
        result["TEMPORAL"] = temporals

    return result


# ── 2. HuggingFace BERT-NER pipeline ─────────────────────────────────────────

def load_bert_ner():
    """
    dslim/bert-base-NER: fine-tuned on CoNLL-2003.
    Recognises: PER, ORG, LOC, MISC.
    Useful for extracting person mentions (therapist, doctor references).
    """
    print("[ner] Loading BERT-NER pipeline ...")
    ner_pipe = pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1  # use CPU; change to 0 for GPU
    )
    return ner_pipe


def extract_bert_entities(text: str, ner_pipe) -> dict:
    result    = {}
    truncated = str(text)[:512]
    try:
        preds = ner_pipe(truncated)
        for item in preds:
            label = item["entity_group"]
            word  = item["word"]
            result.setdefault(label, []).append(word)
    except Exception:
        pass
    return result


# ── 3. Run NER on full dataset ────────────────────────────────────────────────

def run_ner_pipeline(sample_size: int = None):
    """
    Runs both spaCy (custom vocab) and BERT-NER on the cleaned dataset.
    sample_size: set to e.g. 1000 for quick testing, None for full run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROC / "cleaned.csv")
    df.dropna(subset=["text_clean_deep"], inplace=True)

    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"[ner] Running on {sample_size} sample rows")
    else:
        print(f"[ner] Running on full dataset ({len(df)} rows)")

    # spaCy pipeline
    nlp = build_spacy_pipeline()

    tqdm.pandas(desc="[ner] spaCy extraction")
    df["spacy_entities"] = df["text_clean_deep"].progress_apply(
        lambda t: json.dumps(extract_spacy_entities(t, nlp))
    )

    # BERT-NER pipeline (slower — use sample)
    ner_pipe = load_bert_ner()
    tqdm.pandas(desc="[ner] BERT-NER extraction")
    df["bert_entities"] = df["text_clean_deep"].progress_apply(
        lambda t: json.dumps(extract_bert_entities(t, ner_pipe))
    )

    # ── Aggregate: most common medications and diagnoses ─────────────────────
    all_meds  = []
    all_diags = []
    for ents_json in df["spacy_entities"]:
        ents = json.loads(ents_json)
        all_meds.extend(ents.get("MEDICATION", []))
        all_diags.extend(ents.get("DIAGNOSIS", []))

    from collections import Counter
    med_counts  = Counter(m.lower() for m in all_meds)
    diag_counts = Counter(d.lower() for d in all_diags)

    print("\n[ner] Top 15 medications mentioned:")
    for med, cnt in med_counts.most_common(15):
        print(f"  {med:25s} {cnt}")

    print("\n[ner] Top 15 diagnoses mentioned:")
    for dx, cnt in diag_counts.most_common(15):
        print(f"  {dx:30s} {cnt}")

    # Save enriched dataframe
    out = DATA_PROC / "with_ner.csv"
    df.to_csv(out, index=False)
    print(f"\n[ner] Saved → {out}")

    # Save frequency tables
    pd.DataFrame(med_counts.most_common(), columns=["medication", "count"]).to_csv(
        RESULTS_DIR / "medication_frequencies.csv", index=False
    )
    pd.DataFrame(diag_counts.most_common(), columns=["diagnosis", "count"]).to_csv(
        RESULTS_DIR / "diagnosis_frequencies.csv", index=False
    )
    return df


if __name__ == "__main__":
    # Quick test: run on 500 rows
    run_ner_pipeline(sample_size=500)
