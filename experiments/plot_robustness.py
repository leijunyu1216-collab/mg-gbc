"""Regenerate the paper robustness figure from the matched-policy CSV results.

The figure is drawn from results/robustness_matched.csv, the file written by
experiments/run_robust.py under the matched graph policy (size-ratio guard
disabled in both methods) that every comparison in the manuscript uses. The
unmatched results/robustness.csv is left in place as a record and is not
plotted.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "results" / "robustness_matched.csv")
noise = data[data["kind"].eq("noise")]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
colors = {"ARP-GB": "#5b6770", "MG-GBC": "#c83e4d"}
markers = {"ARP-GB": "o", "MG-GBC": "s"}

for ax, dataset in zip(axes, ["Digits-Rare", "Synthetic-SplitRare"]):
    subset = noise[noise["dataset"].eq(dataset)]
    summary = (
        subset.groupby(["sigma", "method"], as_index=False)["ARI"]
        .agg(["mean", "std"])
        .reset_index()
    )
    for method in ["ARP-GB", "MG-GBC"]:
        rows = summary[summary["method"].eq(method)]
        ax.errorbar(
            rows["sigma"],
            rows["mean"],
            yerr=rows["std"],
            label=method,
            color=colors[method],
            marker=markers[method],
            linewidth=1.5,
            capsize=2,
        )
    ax.set_title(dataset)
    ax.set_xlabel(r"noise level $\sigma$")
    ax.grid(alpha=0.25, linewidth=0.5)

axes[0].set_ylabel("ARI (mean ± SD, five seeds)")
axes[1].legend(frameon=False, loc="lower right")
fig.tight_layout()
for suffix in (".png", ".pdf"):
    fig.savefig(ROOT / "results" / f"robustness_matched{suffix}", dpi=220,
                bbox_inches="tight")
plt.close(fig)
