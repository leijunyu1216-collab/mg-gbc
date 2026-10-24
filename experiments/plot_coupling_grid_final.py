"""Figure ``fig:joint``: the joint grid under the shipped fragment rule.

Reads ``results/coupling_grid_guard_raw.csv`` -- the 315-cell sweep evaluated
through ``MGGBC`` as shipped, i.e. with the admissibility condition of
Definition (signed edit) in force -- and draws the response surface over all
31 units (20 UNSW-NB15 v2 replication blocks, ten development sets,
Covertype-Rare).

Top row: fragment edges summed over the 31 units.  Bottom row: units whose
partition differs from their default partition.  The holdout blocks are not
in the file; they are never evaluated off the default cell.

Outputs
-------
results/coupling_grid_heatmap.png
results/coupling_grid_heatmap.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "coupling_grid_guard_raw.csv"
OUT = ROOT / "results"

D0, E0, B0 = 0.25, 1.5, 0.5


def load():
    rows = list(csv.DictReader(open(SRC)))
    for r in rows:
        r["delta"] = float(r["delta"])
        r["eta"] = float(r["eta"])
        r["beta"] = float(r["beta"])
        r["edges"] = int(r["edges"])
        r["same"] = r["same"] == "True"
    deltas = sorted({r["delta"] for r in rows})
    etas = sorted({r["eta"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    return rows, deltas, etas, betas


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, deltas, etas, betas = load()
    levels = {"delta": deltas, "eta": etas, "beta": betas}

    def matrix(xvar, yvar, fixed_var, fixed_value, fn):
        xv, yv = levels[xvar], levels[yvar]
        out = np.zeros((len(yv), len(xv)))
        sub = [r for r in rows if abs(r[fixed_var] - fixed_value) < 1e-9]
        for r in sub:
            out[yv.index(r[yvar]), xv.index(r[xvar])] += fn(r)
        return out

    def edges(r):
        return float(r["edges"])

    def changed(r):
        return 0.0 if r["same"] else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4))
    panels = [
        (axes[0, 0], "delta", "eta", "beta", B0, edges,
         r"(a) $\beta=0.5$: fragment edges"),
        (axes[0, 1], "delta", "beta", "eta", E0, edges,
         r"(b) $\eta=1.5$: fragment edges"),
        (axes[0, 2], "beta", "eta", "delta", D0, edges,
         r"(c) $\delta=0.25$: fragment edges"),
        (axes[1, 0], "delta", "eta", "beta", B0, changed,
         r"(d) $\beta=0.5$: units changed"),
        (axes[1, 1], "delta", "beta", "eta", E0, changed,
         r"(e) $\eta=1.5$: units changed"),
        (axes[1, 2], "beta", "eta", "delta", D0, changed,
         r"(f) $\delta=0.25$: units changed"),
    ]
    symbols = {"delta": r"$\delta$", "eta": r"$\eta$", "beta": r"$\beta$"}
    for ax, xvar, yvar, fvar, fval, fn, title in panels:
        grid = matrix(xvar, yvar, fvar, fval, fn)
        vmax = max(1.0, float(grid.max()))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues",
                       vmin=0, vmax=vmax)
        xv, yv = levels[xvar], levels[yvar]
        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels([f"{v:g}" for v in xv], fontsize=5.5)
        ax.set_yticks(range(len(yv)))
        ax.set_yticklabels([f"{v:g}" for v in yv], fontsize=5.5)
        ax.set_xlabel(symbols[xvar], fontsize=7)
        ax.set_ylabel(symbols[yvar], fontsize=7)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=4.5,
                        color="white" if v > 0.55 * vmax else "black")
        fx = {"delta": D0, "eta": E0, "beta": B0}[xvar]
        fy = {"delta": D0, "eta": E0, "beta": B0}[yvar]
        ax.add_patch(plt.Rectangle(
            (xv.index(fx) - 0.5, yv.index(fy) - 0.5), 1, 1,
            fill=False, ec="crimson", lw=1.3))
        ax.set_title(title, fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(OUT / "coupling_grid_heatmap.png", dpi=200)
    fig.savefig(OUT / "coupling_grid_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()