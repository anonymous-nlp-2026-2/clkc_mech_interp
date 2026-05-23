"""Parallel bootstrap probe training for EN-DE cross-lingual transfer.

Runs n=2000 bootstrap resamples across 4 directions (EN->EN, DE->DE, EN->DE, DE->EN)
using multiprocessing for speed. Outputs per-direction accuracy stats and
asymmetry (DE->EN - EN->DE) with 95% CI and p-value.

Input: {en,de}_activations.pt from collect_activations.py
Output: de_bootstrap_n2000_results.json
"""

import json
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def load_activations(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    n_train = data["split_sizes"]["train"]
    acts = data["activations"]
    if acts.dtype == torch.bfloat16:
        acts = acts.float()
    return {
        "train_X": acts[:n_train].numpy(),
        "train_y": data["labels"][:n_train].numpy(),
        "test_X": acts[n_train:].numpy(),
        "test_y": data["labels"][n_train:].numpy(),
    }


def bootstrap_one_seed(seed, train_X, train_y, test_X, test_y, num_layers):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_X), len(train_X), replace=True)
    X_boot = train_X[idx]
    y_boot = train_y[idx]

    best_bal_acc = 0.0
    best_result = None

    for layer in range(num_layers):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_boot[:, layer, :])
        X_te = scaler.transform(test_X[:, layer, :])
        clf = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='lbfgs', random_state=seed
        )
        clf.fit(X_tr, y_boot)
        y_pred = clf.predict(X_te)
        bal_acc = balanced_accuracy_score(test_y, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_result = {
                'balanced_accuracy': round(float(bal_acc), 4),
                'accuracy': round(float(np.mean(y_pred == test_y)), 4),
                'f1': round(float(f1_score(test_y, y_pred)), 4),
                'peak_layer': layer,
            }

    return best_result


def run_direction_parallel(train_data, test_data, num_layers, direction_name,
                           n_bootstrap=2000, n_workers=64):
    print(f"\n[{direction_name}] Bootstrap ({n_bootstrap} seeds, {n_workers} workers)...",
          flush=True)

    func = partial(
        bootstrap_one_seed,
        train_X=train_data["train_X"],
        train_y=train_data["train_y"],
        test_X=test_data["test_X"],
        test_y=test_data["test_y"],
        num_layers=num_layers,
    )

    with Pool(n_workers) as pool:
        results = pool.map(func, range(n_bootstrap))

    bal_accs = [r['balanced_accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    peak_layers = [r['peak_layer'] for r in results]

    mean_ba = float(np.mean(bal_accs))
    std_ba = float(np.std(bal_accs))
    ci_lo = float(np.percentile(bal_accs, 2.5))
    ci_hi = float(np.percentile(bal_accs, 97.5))

    print(f"  {direction_name}: mean={mean_ba:.4f} +/- {std_ba:.4f}, "
          f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    return {
        'direction': direction_name,
        'n_bootstrap': n_bootstrap,
        'balanced_accuracy_mean': round(mean_ba, 4),
        'balanced_accuracy_std': round(std_ba, 4),
        'balanced_accuracy_ci_lo': round(ci_lo, 4),
        'balanced_accuracy_ci_hi': round(ci_hi, 4),
        'f1_mean': round(float(np.mean(f1s)), 4),
        'f1_std': round(float(np.std(f1s)), 4),
        'peak_layers': peak_layers,
        'per_bootstrap': results,
    }


def compute_asymmetry(results, src_tgt_key, tgt_src_key, n_bootstrap):
    src_tgt = results[src_tgt_key]['per_bootstrap']
    tgt_src = results[tgt_src_key]['per_bootstrap']

    diffs = []
    for i in range(n_bootstrap):
        diff = tgt_src[i]['balanced_accuracy'] - src_tgt[i]['balanced_accuracy']
        diffs.append(diff)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    p_value = float(np.mean(diffs <= 0))

    return {
        'mean_diff_pp': round(mean_diff * 100, 2),
        'ci_lo_pp': round(ci_lo * 100, 2),
        'ci_hi_pp': round(ci_hi * 100, 2),
        'p_value': round(p_value, 4),
        'n_bootstrap': n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_dir', type=str, default='data/llama31/')
    parser.add_argument('--output_dir', type=str, default='data/llama31/')
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--n_workers', type=int, default=64)
    parser.add_argument('--output_file', type=str,
                        default='de_bootstrap_n2000_results.json')
    args = parser.parse_args()

    act_dir = Path(args.activation_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    en_data = load_activations(str(act_dir / 'en_activations.pt'))
    de_data = load_activations(str(act_dir / 'de_activations.pt'))
    num_layers = en_data['train_X'].shape[1]

    print(f"EN: train={en_data['train_X'].shape[0]}, test={en_data['test_X'].shape[0]}")
    print(f"DE: train={de_data['train_X'].shape[0]}, test={de_data['test_X'].shape[0]}")
    print(f"Layers: {num_layers}")
    print(f"Bootstrap: {args.n_bootstrap} seeds, {args.n_workers} workers")

    directions = {
        'EN->EN': (en_data, en_data),
        'DE->DE': (de_data, de_data),
        'EN->DE': (en_data, de_data),
        'DE->EN': (de_data, en_data),
    }

    results = {}
    for name, (train_d, test_d) in directions.items():
        results[name] = run_direction_parallel(
            train_d, test_d, num_layers, name,
            n_bootstrap=args.n_bootstrap, n_workers=args.n_workers
        )

    asym = compute_asymmetry(results, 'EN->DE', 'DE->EN', args.n_bootstrap)
    print(f"\nAsymmetry (DE->EN - EN->DE): {asym['mean_diff_pp']:+.2f}pp "
          f"[{asym['ci_lo_pp']:.2f}, {asym['ci_hi_pp']:.2f}], "
          f"p={asym['p_value']:.4f}", flush=True)

    output = {
        'mode': 'parallel_bootstrap',
        'n_bootstrap': args.n_bootstrap,
        'n_workers': args.n_workers,
        'num_layers': num_layers,
        'langs': ['en', 'de'],
        'directions': {k: {kk: vv for kk, vv in v.items() if kk != 'per_bootstrap'}
                       for k, v in results.items()},
        'directions_full': results,
        'asymmetry': asym,
    }

    out_path = out_dir / args.output_file
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
