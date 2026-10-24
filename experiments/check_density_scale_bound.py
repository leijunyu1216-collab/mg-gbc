"""Numerical check of the density bound behind the retention predicate.

The predicate's density term is ARP-GB Eq. (2): for a point i,

    rho_i = 1 / mean_{j in kNN(i)} ( d_ij / d_ik )^2 ,

normalised by the distance to the k-th neighbour, so it is scale free.  For a
point sitting in a compact group of m points that is otherwise surrounded by a
sparse background, the m - 1 companions contribute ratios near zero while the
remaining k - m + 1 neighbours are background points sitting at roughly d_ik,
so

    rho_group -> k / (k - m + 1)          (m <= k, group radius -> 0).

For m >= k every neighbour is a companion and the estimate diverges.  The bound
is the reason a small rare group cannot outscore the background, and therefore
the reason the predicate is silent below a size that scales with k.

This script checks that expression directly.  It is a property of the host's
estimator, not a claim about any dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import knn_density  # noqa: E402

OUT = ROOT / "results" / "density_scale_bound.json"


def trial(m: int, k: int, d: int, n_bg: int, radius: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bg = rng.random((n_bg, d))
    centre = rng.random(d)
    group = centre + rng.normal(scale=radius, size=(m, d))
    X = np.vstack([bg, group])
    rho, _ = knn_density(X, k)
    rho_bg = float(np.median(rho[:n_bg]))
    rho_group = float(np.median(rho[n_bg:]))
    return {
        "m": m,
        "k": k,
        "d": d,
        "rho_background": rho_bg,
        "rho_group": rho_group,
        "ratio": rho_group / rho_bg,
        "predicted_bound": float(k / (k - m + 1)) if m <= k else float("inf"),
    }


def main() -> None:
    rows = []
    for d in (8, 40):
        for m in range(2, 15):
            for seed in range(5):
                rows.append(trial(m, 10, d, 3000, 1e-3, seed))
    import pandas as pd
    frame = pd.DataFrame(rows)
    table = frame.groupby(["d", "m"]).agg(
        ratio=("ratio", "median"),
        rho_group=("rho_group", "median"),
        rho_background=("rho_background", "median"),
        predicted=("predicted_bound", "median"),
    ).reset_index()

    out = {
        "k": 10,
        "seeds": 5,
        "n_background": 3000,
        "group_radius": 1e-3,
        "bound": "rho_group -> k / (k - m + 1) for m <= k, and diverges for m >= k",
        "rows": table.to_dict(orient="records"),
        "reading": (
            "The bound is a ceiling on how far a compact group of m points can lift "
            "the estimator relative to a background whose own density is set by its "
            "tightest structures. Where the background carries duplicates, its "
            "density is far above the uniform value, and a group of m < k points "
            "cannot reach it at any radius."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()