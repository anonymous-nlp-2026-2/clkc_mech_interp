"""EN-DE token overlap analysis: per-sample overlap with Llama-3.1-8B-Instruct tokenizer.
Phase 1 (overlap_only): Compute per-sample Jaccard statistics (no probe needed).
Phase 2 (full): Add correlation with EN->DE probe accuracy (requires DE bootstrap).
"""
import json
import os
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from scipy import stats

os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_DIR = Path("./data/llama31")
TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_text(row):
    return row["question"] + " " + row["context"]


def compute_per_sample_overlap(tokenizer, en_texts, de_texts):
    results = []
    for en_text, de_text in zip(en_texts, de_texts):
        en_ids = set(tokenizer.encode(en_text, add_special_tokens=False))
        de_ids = set(tokenizer.encode(de_text, add_special_tokens=False))
        intersection = en_ids & de_ids
        union = en_ids | de_ids
        results.append({
            "jaccard": len(intersection) / len(union) if union else 0.0,
            "overlap_en": len(intersection) / len(en_ids) if en_ids else 0.0,
            "overlap_de": len(intersection) / len(de_ids) if de_ids else 0.0,
            "n_shared": len(intersection),
            "n_en": len(en_ids),
            "n_de": len(de_ids),
        })
    return results


def summarize(values):
    v = np.array(values)
    hist_counts, hist_edges = np.histogram(v, bins=10)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": [float(e) for e in hist_edges],
    }


def compute_correlations(overlap_vals, correctness):
    pearson_r, pearson_p = stats.pearsonr(overlap_vals, correctness)
    spearman_r, spearman_p = stats.spearmanr(overlap_vals, correctness)
    pb_r, pb_p = stats.pointbiserialr(correctness, overlap_vals)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pointbiserial_r": float(pb_r),
        "pointbiserial_p": float(pb_p),
    }


def tercile_analysis(overlap_vals, correctness):
    sorted_idx = np.argsort(overlap_vals)
    n = len(sorted_idx)
    t1, t2 = n // 3, 2 * n // 3
    groups = {"low": sorted_idx[:t1], "medium": sorted_idx[t1:t2], "high": sorted_idx[t2:]}
    result = {}
    for name, idx in groups.items():
        result[name] = {
            "n_samples": int(len(idx)),
            "accuracy": float(np.mean(correctness[idx])),
            "mean_overlap": float(np.mean(overlap_vals[idx])),
        }
    result["high_minus_low"] = float(result["high"]["accuracy"] - result["low"]["accuracy"])
    return result


def phase1_overlap():
    print("=" * 60)
    print("Phase 1: EN-DE Token Overlap Analysis (Llama-3.1-8B-Instruct)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    en_train = load_jsonl(DATA_DIR / "en_conflict_train.jsonl")
    en_test = load_jsonl(DATA_DIR / "en_conflict_test.jsonl")
    de_train = load_jsonl(DATA_DIR / "de_conflict_train.jsonl")
    de_test = load_jsonl(DATA_DIR / "de_conflict_test.jsonl")
    print(f"Data: EN train={len(en_train)}, EN test={len(en_test)}, DE train={len(de_train)}, DE test={len(de_test)}")

    en_train_texts = [make_text(r) for r in en_train]
    en_test_texts = [make_text(r) for r in en_test]
    de_train_texts = [make_text(r) for r in de_train]
    de_test_texts = [make_text(r) for r in de_test]

    # Type-level overlap
    en_all_ids = set()
    de_all_ids = set()
    for t in en_train_texts + en_test_texts:
        en_all_ids.update(tokenizer.encode(t, add_special_tokens=False))
    for t in de_train_texts + de_test_texts:
        de_all_ids.update(tokenizer.encode(t, add_special_tokens=False))
    inter = en_all_ids & de_all_ids
    union = en_all_ids | de_all_ids
    tl = {
        "unique_en": len(en_all_ids), "unique_de": len(de_all_ids),
        "intersection": len(inter), "union": len(union),
        "jaccard": len(inter) / len(union),
        "overlap_en": len(inter) / len(en_all_ids),
        "overlap_de": len(inter) / len(de_all_ids),
    }
    print(f"\nType-level: Jaccard={tl['jaccard']:.4f}, Overlap/EN={tl['overlap_en']:.4f}, Overlap/DE={tl['overlap_de']:.4f}")
    print(f"  unique_en={tl['unique_en']}, unique_de={tl['unique_de']}, intersection={tl['intersection']}")

    # Per-sample overlap
    print("\nComputing per-sample overlap (all)...")
    all_overlaps = compute_per_sample_overlap(tokenizer, en_train_texts + en_test_texts, de_train_texts + de_test_texts)
    print("Computing per-sample overlap (test)...")
    test_overlaps = compute_per_sample_overlap(tokenizer, en_test_texts, de_test_texts)

    output = {
        "model": "Llama-3.1-8B-Instruct",
        "language_pair": "EN-DE",
        "type_level": tl,
        "per_sample": {
            "all": {
                "jaccard": summarize([o["jaccard"] for o in all_overlaps]),
                "overlap_en": summarize([o["overlap_en"] for o in all_overlaps]),
                "overlap_de": summarize([o["overlap_de"] for o in all_overlaps]),
                "n_samples": len(all_overlaps),
            },
            "test": {
                "jaccard": summarize([o["jaccard"] for o in test_overlaps]),
                "overlap_en": summarize([o["overlap_en"] for o in test_overlaps]),
                "overlap_de": summarize([o["overlap_de"] for o in test_overlaps]),
                "n_samples": len(test_overlaps),
            },
        },
    }

    ps = output["per_sample"]
    print(f"\nPer-sample Jaccard (all {ps['all']['n_samples']} samples):")
    print(f"  mean={ps['all']['jaccard']['mean']:.4f}, median={ps['all']['jaccard']['median']:.4f}, std={ps['all']['jaccard']['std']:.4f}")
    print(f"  range: [{ps['all']['jaccard']['min']:.4f}, {ps['all']['jaccard']['max']:.4f}]")
    print(f"Per-sample Jaccard (test {ps['test']['n_samples']} samples):")
    print(f"  mean={ps['test']['jaccard']['mean']:.4f}, median={ps['test']['jaccard']['median']:.4f}, std={ps['test']['jaccard']['std']:.4f}")
    print(f"  range: [{ps['test']['jaccard']['min']:.4f}, {ps['test']['jaccard']['max']:.4f}]")
    print(f"\nPer-sample Overlap/EN (all): mean={ps['all']['overlap_en']['mean']:.4f}")
    print(f"Per-sample Overlap/DE (all): mean={ps['all']['overlap_de']['mean']:.4f}")

    # Comparison note
    print("\n" + "=" * 60)
    print("Comparison with EN-ZH (from prior analysis):")
    print("  EN-ZH per-sample Jaccard ~ 3-4%")
    print(f"  EN-DE per-sample Jaccard = {ps['all']['jaccard']['mean']*100:.1f}%")
    print(f"  Ratio EN-DE/EN-ZH ~ {ps['all']['jaccard']['mean']/0.035:.1f}x")
    print("=" * 60)

    out_path = DATA_DIR / "token_overlap_en_de.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Phase 1 results saved to {out_path}")
    return output


def phase2_correlation():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    result_path = DATA_DIR / "token_overlap_en_de.json"
    if not result_path.exists():
        print("Phase 1 results not found. Run --phase overlap_only first.")
        return
    with open(result_path) as f:
        output = json.load(f)

    probe_path = DATA_DIR / "probe_results_bootstrap.json"
    if not probe_path.exists():
        print(f"Probe results not found at {probe_path}. DE bootstrap not yet complete.")
        return

    with open(probe_path) as f:
        pr = json.load(f)

    if "EN→DE" not in pr.get("directions", {}):
        print("EN→DE direction not found in probe results. DE bootstrap not yet complete.")
        return

    peak_layers = pr["directions"]["EN→DE"]["peak_layers"]
    best_layer = Counter(peak_layers).most_common(1)[0][0]
    print(f"Best layer (EN→DE): {best_layer}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    en_test = load_jsonl(DATA_DIR / "en_conflict_test.jsonl")
    de_test = load_jsonl(DATA_DIR / "de_conflict_test.jsonl")
    en_test_texts = [make_text(r) for r in en_test]
    de_test_texts = [make_text(r) for r in de_test]
    test_overlaps = compute_per_sample_overlap(tokenizer, en_test_texts, de_test_texts)

    en_acts = torch.load(DATA_DIR / "en_activations.pt", map_location="cpu")
    de_acts = torch.load(DATA_DIR / "de_activations.pt", map_location="cpu")
    n_train = en_acts["split_sizes"]["train"]

    X_train = en_acts["activations"][:n_train, best_layer, :].float().numpy()
    y_train = en_acts["labels"][:n_train].numpy()
    X_test_de = de_acts["activations"][n_train:, best_layer, :].float().numpy()
    y_test_de = de_acts["labels"][n_train:].numpy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test_de)
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)

    correctness = (preds == y_test_de).astype(int)
    ba = balanced_accuracy_score(y_test_de, preds)
    print(f"EN→DE balanced_accuracy (layer {best_layer}): {ba:.4f}")

    test_jaccard = np.array([o["jaccard"] for o in test_overlaps])
    corr = compute_correlations(test_jaccard, correctness)
    output["correlation"] = {
        "best_layer": best_layer,
        "balanced_accuracy": float(ba),
        **corr,
    }
    print(f"Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4f})")
    print(f"Spearman r={corr['spearman_r']:.4f} (p={corr['spearman_p']:.4f})")
    print(f"Point-biserial r={corr['pointbiserial_r']:.4f} (p={corr['pointbiserial_p']:.4f})")

    terc = tercile_analysis(test_jaccard, correctness)
    output["control_tercile"] = terc
    print(f"\nTercile analysis:")
    for g in ["low", "medium", "high"]:
        print(f"  {g}: n={terc[g]['n_samples']}, acc={terc[g]['accuracy']:.4f}, mean_overlap={terc[g]['mean_overlap']:.4f}")
    print(f"  Δ(high-low) = {terc['high_minus_low']:+.4f}")

    with open(DATA_DIR / "token_overlap_en_de.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Phase 2 results saved to {DATA_DIR / 'token_overlap_en_de.json'}")


def main():
    parser = argparse.ArgumentParser(description="EN-DE token overlap analysis")
    parser.add_argument("--phase", choices=["overlap_only", "full"], default="overlap_only")
    args = parser.parse_args()

    if args.phase == "overlap_only":
        phase1_overlap()
    else:
        phase1_overlap()
        phase2_correlation()


if __name__ == "__main__":
    main()
