"""Prepare XNLI data as a non-conflict semantic baseline for probe transfer experiments.

Downloads XNLI entailment/contradiction pairs for EN and ZH, formats them
to match collect_activations.py's expected JSONL schema (context, question, label).

Input:  HuggingFace XNLI parquet (all_languages validation split)
Output: data/xnli/{lang}_xnli_{split}.jsonl  (train/test per language)

Label mapping:
  entailment (xnli=0)    -> "no_conflict" -> probe label 0
  contradiction (xnli=2) -> "conflict"    -> probe label 1
  (entailment=consistent=no_conflict, contradiction=inconsistent=conflict)
"""

import json
import os
import random
import subprocess
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLES_PER_LANG = 1000
TRAIN_RATIO = 0.8

# XNLI label int -> collect_activations.py label string
XNLI_TO_LABEL = {0: "no_conflict", 2: "conflict"}

PARQUET_URL = "https://hf-mirror.com/datasets/xnli/resolve/main/all_languages/validation-00000-of-00001.parquet"
PARQUET_LOCAL = Path("/tmp/xnli_val.parquet")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "xnli"


def download_parquet():
    """Download XNLI validation parquet if not already present."""
    if PARQUET_LOCAL.exists() and PARQUET_LOCAL.stat().st_size > 1_000_000:
        print(f"Using cached parquet: {PARQUET_LOCAL}")
        return
    print(f"Downloading XNLI parquet from {PARQUET_URL}...")
    subprocess.run(
        ["wget", "-q", "--timeout=120", PARQUET_URL, "-O", str(PARQUET_LOCAL)],
        check=True,
    )


def extract_lang_samples(df: pd.DataFrame, lang: str) -> list[dict]:
    """Extract premise/hypothesis for a specific language, filter to entailment+contradiction."""
    lang_list = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    lang_idx = lang_list.index(lang)

    samples = []
    for _, row in df.iterrows():
        label_int = row["label"]
        if label_int not in XNLI_TO_LABEL:
            continue

        premise = row["premise"][lang]
        hyp_translations = row["hypothesis"]["translation"]
        hypothesis = hyp_translations[lang_idx]

        samples.append({
            "context": premise,
            "question": hypothesis,
            "label": XNLI_TO_LABEL[label_int],
            "language": lang,
            "xnli_original_label": int(label_int),
        })
    return samples


def sample_and_split(samples: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Subsample to n (or all if fewer), then 80/20 train/test split."""
    random.shuffle(samples)
    samples = samples[:n]
    split_idx = int(len(samples) * TRAIN_RATIO)
    return samples[:split_idx], samples[split_idx:]


def write_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    download_parquet()
    df = pd.read_parquet(PARQUET_LOCAL)
    print(f"Loaded parquet: {len(df)} rows")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    for lang in ["en", "zh"]:
        print(f"\n[{lang}] Extracting samples...")
        samples = extract_lang_samples(df, lang)
        print(f"[{lang}] Filtered {len(samples)} entailment+contradiction samples")

        train, test = sample_and_split(samples, SAMPLES_PER_LANG)

        train_path = OUTPUT_DIR / f"{lang}_xnli_train.jsonl"
        test_path = OUTPUT_DIR / f"{lang}_xnli_test.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test, test_path)

        train_labels = [s["label"] for s in train]
        test_labels = [s["label"] for s in test]
        stats[lang] = {
            "train": len(train),
            "test": len(test),
            "train_entailment": train_labels.count("no_conflict"),
            "train_contradiction": train_labels.count("conflict"),
            "test_entailment": test_labels.count("no_conflict"),
            "test_contradiction": test_labels.count("conflict"),
        }
        print(f"[{lang}] Wrote {len(train)} train, {len(test)} test -> {OUTPUT_DIR}/")

    print("\n=== Statistics ===")
    for lang, s in stats.items():
        print(f"  {lang}: train={s['train']} (ent={s['train_entailment']}, cont={s['train_contradiction']}), "
              f"test={s['test']} (ent={s['test_entailment']}, cont={s['test_contradiction']})")


if __name__ == "__main__":
    main()
