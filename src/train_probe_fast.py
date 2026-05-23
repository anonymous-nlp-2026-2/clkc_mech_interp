"""Parallelized bootstrap probe with joblib.

Runs n_seeds bootstrap iterations across 4 cross-lingual directions,
parallelized over seeds. Computes per-layer balanced_accuracy stats
and paired asymmetry CI (zh_en - en_zh).
"""

import argparse
import json
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def load_activations(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    n_train = data["split_sizes"]["train"]
    acts = data["activations"]
    if acts.dtype == torch.bfloat16:
        acts = acts.float()
    acts_np = acts.numpy()
    labels_np = data["labels"].numpy()
    return {
        "train_X": acts_np[:n_train],
        "test_X": acts_np[n_train:],
        "train_y": labels_np[:n_train],
        "test_y": labels_np[n_train:],
    }


def single_bootstrap(seed, train_X_all, train_y, test_X_all, test_y, n_layers):
    rng = np.random.RandomState(seed)
    n = len(train_y)
    idx = rng.choice(n, size=n, replace=True)
    y_boot = train_y[idx]

    layer_accs = np.empty(n_layers)
    for layer in range(n_layers):
        X_tr = train_X_all[:, layer, :]
        X_tr_boot = X_tr[idx]
        X_te = test_X_all[:, layer, :]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_boot)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_tr_s, y_boot)
        pred = clf.predict(X_te_s)
        layer_accs[layer] = balanced_accuracy_score(test_y, pred)

    return seed, layer_accs


def run_bootstrap(en_path, zh_path, output_path, n_seeds=2000, n_jobs=32):
    en = load_activations(en_path)
    zh = load_activations(zh_path)
    n_layers = en["train_X"].shape[1]

    directions = {
        "en_en": (en["train_X"], en["train_y"], en["test_X"], en["test_y"]),
        "zh_zh": (zh["train_X"], zh["train_y"], zh["test_X"], zh["test_y"]),
        "en_zh": (en["train_X"], en["train_y"], zh["test_X"], zh["test_y"]),
        "zh_en": (zh["train_X"], zh["train_y"], en["test_X"], en["test_y"]),
    }

    all_results = {}
    raw_accs = {}

    for dir_name, (tr_X, tr_y, te_X, te_y) in directions.items():
        print(f"\n=== {dir_name} (train={len(tr_y)}, test={len(te_y)}, layers={n_layers}) ===")
        t0 = time.time()

        results_list = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(single_bootstrap)(
                seed, tr_X, tr_y, te_X, te_y, n_layers
            )
            for seed in range(n_seeds)
        )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        accs_matrix = np.array([r[1] for r in results_list])  # [n_seeds, n_layers]
        raw_accs[dir_name] = accs_matrix

        layer_stats = {}
        for layer in range(n_layers):
            col = accs_matrix[:, layer]
            layer_stats[f"layer_{layer}"] = {
                "mean": round(float(np.mean(col)), 4),
                "std": round(float(np.std(col)), 4),
                "ci_lower": round(float(np.percentile(col, 2.5)), 4),
                "ci_upper": round(float(np.percentile(col, 97.5)), 4),
            }
        all_results[dir_name] = layer_stats

    # Paired asymmetry: per-seed peak(zh_en) - peak(en_zh)
    zh_en_peaks = np.max(raw_accs["zh_en"], axis=1)
    en_zh_peaks = np.max(raw_accs["en_zh"], axis=1)
    diffs = zh_en_peaks - en_zh_peaks
    asymmetry = {
        "mean_diff": round(float(np.mean(diffs)), 4),
        "std_diff": round(float(np.std(diffs)), 4),
        "ci_lower": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value_two_sided": round(float(2 * min(np.mean(diffs > 0), np.mean(diffs < 0))), 4),
        "n_seeds": n_seeds,
    }

    # Per-layer paired asymmetry
    layer_asymmetry = {}
    for layer in range(n_layers):
        d = raw_accs["zh_en"][:, layer] - raw_accs["en_zh"][:, layer]
        layer_asymmetry[f"layer_{layer}"] = {
            "mean_diff": round(float(np.mean(d)), 4),
            "ci_lower": round(float(np.percentile(d, 2.5)), 4),
            "ci_upper": round(float(np.percentile(d, 97.5)), 4),
        }

    output = {
        "per_layer": all_results,
        "asymmetry_peak": asymmetry,
        "asymmetry_per_layer": layer_asymmetry,
        "config": {"n_seeds": n_seeds, "n_jobs": n_jobs},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"Asymmetry (zh_en - en_zh peak): {asymmetry['mean_diff']:.4f} "
          f"[{asymmetry['ci_lower']:.4f}, {asymmetry['ci_upper']:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", required=True)
    parser.add_argument("--zh_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_seeds", type=int, default=2000)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    run_bootstrap(args.en_path, args.zh_path, args.output, args.n_seeds, args.n_jobs)
