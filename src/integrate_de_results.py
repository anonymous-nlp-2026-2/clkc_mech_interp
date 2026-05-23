#!/usr/bin/env python3
"""Integrate DE bootstrap n=2000 results: 4-direction table, Δ_asym, cross-language comparison with EN-ZH.

Usage:
    python src/integrate_de_results.py --results_file data/llama31/de_bootstrap_n2000_results.json
"""

import json
import argparse
from collections import Counter
from pathlib import Path
import numpy as np


# EN-ZH reference (Llama-3.1, probe_results_bootstrap_n2000.json asymmetry_peak)
EN_ZH_REF = {
    "delta_pp": 7.27,       # mean_diff * 100
    "ci_lower_pp": 0.37,    # ci_lower * 100
    "ci_upper_pp": 13.97,   # ci_upper * 100
    "p_value": 0.038,
    "token_overlap_pct": 3.4,
}

EN_DE_TOKEN_OVERLAP_PCT = 14.77  # per_sample.all.jaccard.mean * 100


def load_results(path):
    with open(path) as f:
        return json.load(f)


def peak_layer_mode(peak_layers):
    return Counter(peak_layers).most_common(1)[0][0]


def print_direction_table(results):
    directions = results["directions"]
    print("\n" + "=" * 72)
    print("  4-Direction Bootstrap Summary (Llama-3.1, EN-DE, n=2000)")
    print("=" * 72)
    print(f"  {'Direction':<12} {'Peak Layer':>10} {'Bal Acc (mean)':>15} {'95% CI':>22}")
    print("  " + "-" * 68)
    for name in ["EN->EN", "DE->DE", "EN->DE", "DE->EN"]:
        d = directions[name]
        pl = peak_layer_mode(d["peak_layers"])
        m = d["balanced_accuracy_mean"]
        lo = d["balanced_accuracy_ci_lo"]
        hi = d["balanced_accuracy_ci_hi"]
        print(f"  {name:<12} {pl:>10} {m:>15.4f} [{lo:.4f}, {hi:.4f}]")
    print("=" * 72)


def print_asymmetry(results):
    a = results["asymmetry"]
    sig = "SIGNIFICANT" if a["p_value"] < 0.05 else "NOT significant"
    print(f"\n  Δ_DE = (DE→EN) - (EN→DE) = {a['mean_diff_pp']:+.2f}pp  "
          f"95% CI [{a['ci_lo_pp']:.2f}, {a['ci_hi_pp']:.2f}]  "
          f"p = {a['p_value']:.4f}  ({sig})")


def print_comparison(results):
    a = results["asymmetry"]
    r = EN_ZH_REF
    print("\n" + "=" * 72)
    print("  Cross-Language Pair Comparison")
    print("=" * 72)
    print(f"  {'Metric':<22} {'EN-ZH':>24} {'EN-DE':>24}")
    print("  " + "-" * 68)
    zh_str = f"+{r['delta_pp']:.1f} [{r['ci_lower_pp']:.1f}, {r['ci_upper_pp']:.1f}] p={r['p_value']:.3f}"
    de_str = f"{a['mean_diff_pp']:+.2f} [{a['ci_lo_pp']:.2f}, {a['ci_hi_pp']:.2f}] p={a['p_value']:.4f}"
    print(f"  {'Δ_asym (pp)':<22} {zh_str:>24} {de_str:>24}")
    print(f"  {'Token overlap (%)':<22} {r['token_overlap_pct']:>23.1f}% {EN_DE_TOKEN_OVERLAP_PCT:>23.2f}%")
    print("=" * 72)
    if a["p_value"] >= 0.05:
        print("\n  → Scenario A: DE asymmetry NOT significant despite 14.77% token overlap")
        print("    Token overlap does not drive EN-ZH asymmetry")
    else:
        print("\n  → Scenario B: Asymmetry replicates on second language pair")
        print("    Cross-lingual knowledge asymmetry is a general phenomenon")


def save_summary(results, out_path):
    a = results["asymmetry"]
    summary = {
        "language_pair": "EN-DE",
        "n_bootstrap": results["n_bootstrap"],
        "directions": {},
        "asymmetry_de": a,
        "asymmetry_zh_ref": EN_ZH_REF,
        "token_overlap_pct": {"en_zh": EN_ZH_REF["token_overlap_pct"], "en_de": EN_DE_TOKEN_OVERLAP_PCT},
        "scenario": "A" if a["p_value"] >= 0.05 else "B",
    }
    for name in ["EN->EN", "DE->DE", "EN->DE", "DE->EN"]:
        d = results["directions"][name]
        summary["directions"][name] = {
            "peak_layer": peak_layer_mode(d["peak_layers"]),
            "balanced_accuracy_mean": d["balanced_accuracy_mean"],
            "balanced_accuracy_ci": [d["balanced_accuracy_ci_lo"], d["balanced_accuracy_ci_hi"]],
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate DE bootstrap results and compare with EN-ZH")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to de_bootstrap_n2000_results.json")
    parser.add_argument("--summary_output", type=str, default=None,
                        help="Summary JSON path (default: alongside results)")
    args = parser.parse_args()

    results = load_results(args.results_file)
    print_direction_table(results)
    print_asymmetry(results)
    print_comparison(results)

    out = args.summary_output or str(Path(args.results_file).parent / "de_integration_summary.json")
    save_summary(results, out)


if __name__ == "__main__":
    main()
