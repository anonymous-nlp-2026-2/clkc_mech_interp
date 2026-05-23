"""Compute layer-wise Linear CKA and SVCCA between EN vs ZH activations.

v2: Fixed pairing confound — runs all 4 combinations (2 tasks x 2 alignment modes).
Uses test set instead of train set.
"""

import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

DATA_ROOT = "./data"
OUT_JSON = os.path.join(DATA_ROOT, "cka_svcca_results_v2.json")
OUT_PLOT = os.path.join(DATA_ROOT, "cka_svcca_plot_v2.png")

SEED = 42

DATA_PATHS = {
    "qwen3": {
        "conflict": {"en": "qwen3/en_activations.pt", "zh": "qwen3/zh_activations.pt"},
        "xnli": {"en": "xnli/qwen3/en_activations.pt", "zh": "xnli/qwen3/zh_activations.pt"},
    },
    "llama31": {
        "conflict": {"en": "llama31/en_activations.pt", "zh": "llama31/zh_activations.pt"},
        "xnli": {"en": "xnli/llama31/en_activations.pt", "zh": "xnli/llama31/zh_activations.pt"},
    },
}

MODES = ["aligned", "shuffled"]


def load_test_data(path):
    d = torch.load(os.path.join(DATA_ROOT, path), map_location="cpu")
    n_train = d["split_sizes"]["train"]
    act = d["activations"][n_train:].float().numpy()
    labels = d["labels"][n_train:].numpy()
    return act, labels


def shuffle_activations(act, rng):
    perm = rng.permutation(act.shape[0])
    return act[perm]


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def fast_truncated_svd(M, threshold=0.99):
    n, d = M.shape
    if n <= d:
        G = M @ M.T
        eigvals, eigvecs = eigh(G)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        eigvals = np.maximum(eigvals, 0)
        s = np.sqrt(eigvals)
        cumvar = np.cumsum(eigvals)
        total = cumvar[-1]
        if total < 1e-12:
            return eigvecs[:, :1] * s[:1]
        k = np.searchsorted(cumvar / total, threshold) + 1
        k = min(k, len(s))
        return eigvecs[:, :k] * s[:k]
    else:
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        cumvar = np.cumsum(s ** 2)
        total = cumvar[-1]
        if total < 1e-12:
            return U[:, :1] * s[:1]
        k = np.searchsorted(cumvar / total, threshold) + 1
        k = min(k, len(s))
        return U[:, :k] * s[:k]


def svcca(X, Y, variance_threshold=0.99):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Xr = fast_truncated_svd(X, variance_threshold)
    Yr = fast_truncated_svd(Y, variance_threshold)
    Qx, _ = np.linalg.qr(Xr)
    Qy, _ = np.linalg.qr(Yr)
    d = min(Qx.shape[1], Qy.shape[1])
    _, s, _ = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    correlations = np.clip(s[:d], 0, 1)
    return float(np.mean(correlations))


def compute_layerwise(en_act, zh_act, n_layers):
    layer_results = {}
    for l in range(n_layers):
        tl = time.time()
        X = en_act[:, l, :]
        Y = zh_act[:, l, :]
        cka_val = linear_cka(X, Y)
        svcca_val = svcca(X, Y)
        layer_results[l] = {"cka": round(cka_val, 6), "svcca": round(svcca_val, 6)}
        dt = time.time() - tl
        if l % 4 == 0 or l == n_layers - 1:
            print(f"    Layer {l:2d}: CKA={cka_val:.4f}, SVCCA={svcca_val:.4f}  ({dt:.1f}s)", flush=True)
    return layer_results


def main():
    rng = np.random.RandomState(SEED)
    results = {}
    t0 = time.time()

    for model in ["qwen3", "llama31"]:
        results[model] = {}
        for task in ["conflict", "xnli"]:
            paths = DATA_PATHS[model][task]
            print(f"\n=== {model} / {task} ===", flush=True)
            en_act, en_labels = load_test_data(paths["en"])
            zh_act, zh_labels = load_test_data(paths["zh"])

            n = min(en_act.shape[0], zh_act.shape[0])
            en_act = en_act[:n]
            zh_act = zh_act[:n]
            en_labels = en_labels[:n]
            zh_labels = zh_labels[:n]

            n_samples, n_layers, hidden = en_act.shape
            print(f"  Test samples={n_samples}, Layers={n_layers}, Hidden={hidden}", flush=True)

            # Aligned: index-paired (same premise/question in EN and ZH)
            print(f"\n  --- {task}_aligned ---", flush=True)
            results[model][f"{task}_aligned"] = compute_layerwise(en_act, zh_act, n_layers)

            # Shuffled: break index pairing by permuting ZH
            print(f"\n  --- {task}_shuffled ---", flush=True)
            zh_act_shuffled = shuffle_activations(zh_act, rng)
            results[model][f"{task}_shuffled"] = compute_layerwise(en_act, zh_act_shuffled, n_layers)

    # Compute diffs for the two fair comparisons
    for model in ["qwen3", "llama31"]:
        for mode in MODES:
            conflict_key = f"conflict_{mode}"
            xnli_key = f"xnli_{mode}"
            diff_key = f"conflict_minus_xnli_{mode}"
            conflict = results[model][conflict_key]
            xnli = results[model][xnli_key]
            diff = {}
            for l in conflict:
                if l in xnli:
                    diff[l] = {
                        "cka_diff": round(conflict[l]["cka"] - xnli[l]["cka"], 6),
                        "svcca_diff": round(conflict[l]["svcca"] - xnli[l]["svcca"], 6),
                    }
            results[model][diff_key] = diff

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "conflict_aligned": "#d62728",
        "conflict_shuffled": "#ff7f0e",
        "xnli_aligned": "#1f77b4",
        "xnli_shuffled": "#2ca02c",
    }
    markers = {
        "conflict_aligned": "o",
        "conflict_shuffled": "^",
        "xnli_aligned": "s",
        "xnli_shuffled": "D",
    }

    for col, model in enumerate(["qwen3", "llama31"]):
        for metric_idx, metric in enumerate(["cka", "svcca"]):
            ax = axes[metric_idx, col]
            for key in ["conflict_aligned", "conflict_shuffled", "xnli_aligned", "xnli_shuffled"]:
                data = results[model][key]
                layers = sorted(data.keys())
                vals = [data[l][metric] for l in layers]
                ax.plot(layers, vals, f"{markers[key]}-", label=key.replace('_', ' ').title(),
                        color=colors[key], markersize=3, linewidth=1.5)
            title_metric = "Linear CKA" if metric == "cka" else "SVCCA"
            ax.set_title(f"{model.upper()} - {title_metric} (EN vs ZH)", fontsize=12)
            ax.set_xlabel("Layer")
            ax.set_ylabel(title_metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_PLOT}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FAIR COMPARISONS")
    print("=" * 70)

    for model in ["qwen3", "llama31"]:
        print(f"\n{'='*50}")
        print(f"  {model.upper()}")
        print(f"{'='*50}")

        for mode in MODES:
            conflict_key = f"conflict_{mode}"
            xnli_key = f"xnli_{mode}"
            diff_key = f"conflict_minus_xnli_{mode}"

            conflict = results[model][conflict_key]
            xnli = results[model][xnli_key]
            diff = results[model][diff_key]
            layers = sorted(conflict.keys())

            cka_c = [conflict[l]["cka"] for l in layers]
            cka_x = [xnli[l]["cka"] for l in layers]
            cka_d = [diff[l]["cka_diff"] for l in layers]

            print(f"\n  --- {mode.upper()} comparison ---")
            print(f"  Conflict {mode} CKA: mean={np.mean(cka_c):.4f}, range=[{np.min(cka_c):.4f}, {np.max(cka_c):.4f}]")
            print(f"  XNLI     {mode} CKA: mean={np.mean(cka_x):.4f}, range=[{np.min(cka_x):.4f}, {np.max(cka_x):.4f}]")
            print(f"  CKA diff (conflict-xnli): mean={np.mean(cka_d):.4f}, range=[{np.min(cka_d):.4f}, {np.max(cka_d):.4f}]")

            top5 = np.argsort(np.abs(cka_d))[-5:][::-1]
            vals = [(int(layers[i]), round(cka_d[i], 4)) for i in top5]
            print(f"  Top-5 |CKA diff| layers: {vals}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: Does CKA difference survive after controlling pairing?")
    print("=" * 70)
    for model in ["qwen3", "llama31"]:
        aligned_diff = results[model]["conflict_minus_xnli_aligned"]
        shuffled_diff = results[model]["conflict_minus_xnli_shuffled"]
        layers = sorted(aligned_diff.keys())
        aligned_cka = [aligned_diff[l]["cka_diff"] for l in layers]
        shuffled_cka = [shuffled_diff[l]["cka_diff"] for l in layers]
        print(f"\n  {model.upper()}:")
        print(f"    Aligned comparison:  mean CKA diff = {np.mean(aligned_cka):.4f}")
        print(f"    Shuffled comparison: mean CKA diff = {np.mean(shuffled_cka):.4f}")
        if abs(np.mean(aligned_cka)) < 0.05 and abs(np.mean(shuffled_cka)) < 0.05:
            print(f"    -> CKA difference is SMALL in both modes. C3 CKA evidence WEAK.")
        elif abs(np.mean(aligned_cka)) > 0.05 and abs(np.mean(shuffled_cka)) > 0.05:
            print(f"    -> CKA difference PERSISTS after controlling pairing. C3 CKA evidence SUPPORTED.")
        else:
            print(f"    -> Mixed signal. Inspect per-layer results.")


if __name__ == "__main__":
    main()
