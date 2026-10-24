from __future__ import annotations

import numpy as np

from .core import (
    assign_boundary_points,
    generate_granular_balls,
    knn_density,
    minmax_scale,
    pairwise_overlap_mask,
    propagate_gb_labels,
    reduce_granular_balls,
    select_representatives,
)


class ARPGB:
    """Adaptive representative point-driven granular-ball clustering (Wang & Zhu, 2026)."""

    def __init__(
        self,
        k: int = 10,
        scale: bool = True,
        size_gate: bool = True,
        size_cut_factor: float = 1.0,
    ):
        self.k = int(k)
        self.scale = scale
        # Exposed so that comparisons can hold the graph policy fixed.
        self.size_gate = bool(size_gate)
        # The published cut is 1.0 * sqrt(n); other values instantiate the same
        # pipeline under a tightened or loosened truncation.
        self.size_cut_factor = float(size_cut_factor)
        self.labels_ = None
        self.n_clusters_ = None
        self.n_balls_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scale:
            X = minmax_scale(X)
        rho, _ = knn_density(X, self.k)
        reps = select_representatives(X, rho)
        balls = generate_granular_balls(
            X, reps, dissolve_small=True, size_cut_factor=self.size_cut_factor
        )
        balls = reduce_granular_balls(X, balls)
        overlap = pairwise_overlap_mask(balls)
        gb_labels = propagate_gb_labels(X, balls, overlap, size_gate=self.size_gate)
        labels = assign_boundary_points(X, balls, gb_labels)
        self.labels_ = labels
        self.n_clusters_ = int(len(np.unique(labels)))
        self.n_balls_ = len(balls)
        return labels
