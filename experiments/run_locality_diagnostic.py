"""Locality diagnostic for the vertex intervention (Section 4.2).

Corollary 2 (partition invariance under local redistribution) holds exactly
when every point dissolved from a component C of the inherited overlap graph
is reassigned to a ball of C.  That condition is not a theorem, so we measure
it: for every dropped (small, non-rare) ball we compare the component of the
ball that absorbs its points with the component of the nearest retained ball
to the source ball.

The measurement is taken on the retained ball set *before* radius reduction,
so the components are those of the inherited graph on a fixed ball set.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import (
    generate_granular_balls,
    knn_density,
    minmax_scale,
    pairwise_overlap_mask,
    select_representatives,
)
from src.mg_gbc import ball_mean_density, rare_core_mask
from run_rare_real import EVAL_LOADERS, _load


def components(overlap: np.ndarray) -> np.ndarray:
    m = len(overlap)
    if m == 0:
        return np.zeros(0, dtype=int)
    graph = csr_matrix(overlap.astype(np.int8))
    _, comp = connected_components(graph, directed=False)
    return comp


def diagnose(X: np.ndarray, k: int, **kw) -> dict:
    Xs = minmax_scale(np.asarray(X, dtype=float))
    n = len(Xs)
    rho, _ = knn_density(Xs, k)
    reps = select_representatives(Xs, rho)
    balls = generate_granular_balls(Xs, reps, dissolve_small=False)
    sizes = np.array([b.size for b in balls], dtype=float)
    gamma = ball_mean_density(rho, balls)
    centers = np.stack([b.center for b in balls])
    rare = rare_core_mask(
        sizes, gamma, centers, n,
        kw.get("rare_density_z", 0.5),
        kw.get("rare_isolation_z", 0.0),
        kw.get("rare_min_size", 3),
    )
    small = sizes < np.sqrt(n)
    drop = small & (~rare)
    kept_ids = np.flatnonzero(~drop)
    if len(kept_ids) == 0 or not drop.any():
        return {"dropped_points": 0, "cross_component": 0, "fraction": 0.0, "n_components": 0}

    kept = [balls[i] for i in kept_ids]
    kept_centers = np.stack([b.center for b in kept])
    overlap = pairwise_overlap_mask(kept)
    comp = components(overlap)

    # origin: nearest retained ball to the dropped ball's centre
    total = 0
    crossed = 0
    for i in np.flatnonzero(drop):
        pts = Xs[balls[i].indices]
        target = np.argmin(
            np.linalg.norm(pts[:, None, :] - kept_centers[None, :, :], axis=2),
            axis=1,
        )
        origin = int(
            np.argmin(np.linalg.norm(kept_centers - balls[i].center, axis=1))
        )
        total += len(target)
        crossed += int((comp[target] != comp[origin]).sum())
    return {
        "dropped_points": total,
        "cross_component": crossed,
        "fraction": crossed / total if total else 0.0,
        "n_components": int(comp.max()) + 1,
    }


def main() -> None:
    rows = []
    for spec in EVAL_LOADERS:
        loaded = _load(spec)
        if loaded is None:
            continue
        X, y, name, k = loaded
        k = int(np.clip(k, 2, max(2, len(X) - 1)))
        r = diagnose(X, k)
        r.update({"dataset": name, "n": len(X), "k": k})
        rows.append(r)
        print(
            f"{name:<22} dropped={r['dropped_points']:6d} "
            f"cross={r['cross_component']:5d} frac={r['fraction']:.3f} "
            f"components={r['n_components']}"
        )
    out = ROOT / "results" / "locality_diagnostic.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "n", "k", "dropped_points", "cross_component", "fraction", "n_components"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()