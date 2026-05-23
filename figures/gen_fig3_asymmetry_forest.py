#!/usr/bin/env python3
"""Generate Fig 3: Asymmetry Forest Plot.

Δ_asym = (L2→L1) - (L1→L2) with 95% bootstrap CI for each model × language pair.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("./figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, delta_pp, ci_lo_pp, ci_hi_pp, category)
data = [
    # --- EN-ZH pairs ---
    ("Llama-3.1 (ZH)", 7.27, 0.37, 13.97, "zh"),
    ("Gemma-2 (ZH)",   6.62, 1.90, 11.08, "zh"),

    # --- EN-DE pair ---
    # DE data — uncomment when de_bootstrap_n2000_results.json is ready
    # ("Llama-3.1 (DE)", DELTA_DE, CI_LO_DE, CI_HI_DE, "de"),
]

COLORS = {"zh": "#e6550d", "de": "#2171b5"}
SECTION_HEADERS = {
    0: "EN-ZH",
    # When DE row is uncommented, add:  2: "EN-DE"
}

n = len(data)
fig, ax = plt.subplots(figsize=(5.5, 0.7 * n + 1.2))

for i, (label, delta, ci_lo, ci_hi, cat) in enumerate(data):
    color = COLORS[cat]
    y = n - 1 - i
    ax.errorbar(delta, y, xerr=[[delta - ci_lo], [ci_hi - delta]],
                fmt='o', color=color, capsize=4, capthick=1.5,
                markersize=7, markeredgecolor='k', markeredgewidth=0.5,
                linewidth=1.5, zorder=5)
    ax.text(ci_hi + 0.6, y, f"{delta:+.1f}pp", va='center', fontsize=9,
            color=color, fontweight='bold')

ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='--', zorder=1)

# Section separator and headers (for multi-section layout)
# When DE is added: ax.axhline(y=0.5, color='#cccccc', linewidth=0.6, linestyle='-')

ax.set_yticks(list(range(n)))
ax.set_yticklabels([d[0] for d in reversed(data)], fontsize=10)
ax.set_xlabel(r"$\Delta_{\mathrm{asym}}$ (L2$\rightarrow$L1 $-$ L1$\rightarrow$L2, pp)",
              fontsize=10.5)
ax.set_title("Cross-Lingual Probe Asymmetry", fontsize=12, fontweight='bold', pad=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_xlim(-5, 20)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_asymmetry_forest.pdf", bbox_inches='tight', dpi=300)
fig.savefig(OUT_DIR / "fig3_asymmetry_forest.png", bbox_inches='tight', dpi=300)
print(f"Saved to {OUT_DIR / 'fig3_asymmetry_forest.pdf'}")
print(f"Saved to {OUT_DIR / 'fig3_asymmetry_forest.png'}")
plt.close()
