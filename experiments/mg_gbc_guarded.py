"""Candidate repairs for the UNSW-NB15 rare-core false-positive problem.

Diagnosis behind these variants (see diagnose_unsw_b12*.py):

`rare_core_mask` scores density with a MAD-normalised z-score.  That statistic
is scale-free, so on a block whose granular balls are almost equally dense --
the normal situation for high-dimensional network-flow features, where every
ball sits near rho ~ 1.3 -- the MAD is tiny and a trivial density difference
still clears `dens_z > 0.5`.  Dense *majority* fragments are then retained as
"rare cores", they absorb redistributed mass, and the majority ball graph
loses the propagation edges that used to chain it into one cluster.

Two label-free repairs are implemented:

`MGGBCGuarded`
    Adds an absolute density-contrast requirement on top of the z-test: a
    rare core must be at least (1 + tau) times as dense as the median ball.
    Scale-free z-scores and absolute contrast have to agree.

`MGGBCOverlay`
    Keeps MG-GBC's rare cores but takes the majority partition from the
    ARP-GB ball geometry, so retaining a rare core can no longer fragment
    the majority.  Rare-core members override the majority assignment.

Neither variant consults labels.  src/ is left untouched because the protocol
freezes its hashes.
"""

from __future__ import annotations

import numpy as np

from src.core import (
    assign_boundary_points,
    generate_granular_balls,
    knn_density,
    minmax_scale,
    pairwise_overlap_mask,
    propagate_gb_labels,
    reduce_granular_balls,
    select_representatives,
)
from src.mg_gbc import (
    _redistribute_dropped,
    ball_mean_density,
    isolation_from_background,
    protect_rare_cores,
    rare_core_mask,
    rare_fragment_edges,
)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) + 1e-12


def guarded_rare_core_mask(
    sizes: np.ndarray,
    gamma: np.ndarray,
    centers: np.ndarray,
    n_points: int,
    density_z_threshold: float = 0.5,
    isolation_z_threshold: float = 0.0,
    min_size: int = 3,
    contrast_tau: float = 0.25,
) -> np.ndarray:
    """rare_core_mask plus an absolute density-contrast floor."""
    dens_z = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
    size_cut = np.sqrt(n_points)
    med = float(np.median(gamma))
    candidate = (
        (sizes >= int(min_size))
        & (sizes < size_cut)
        & (dens_z > float(density_z_threshold))
        & (gamma >= (1.0 + float(contrast_tau)) * med)
    )
    if int(candidate.sum()) == 0:
        return candidate
    iso = isolation_from_background(centers, candidate)
    iso_z = 0.6745 * (iso - np.median(iso)) / _mad(iso)
    return candidate & (iso_z > float(isolation_z_threshold))


class MGGBCGuarded:
    """MG-GBC with an absolute density-contrast guard on rare-core detection."""

    def __init__(self, k: int = 10, scale: bool = True, contrast_tau: float = 0.25,
                 rare_density_z: float = 0.5, rare_isolation_z: float = 0.0,
                 rare_min_size: int = 3):
        self.k = int(k)
        self.scale = scale
        self.contrast_tau = float(contrast_tau)
        self.rare_density_z = float(rare_density_z)
        self.rare_isolation_z = float(rare_isolation_z)
        self.rare_min_size = int(rare_min_size)
        self.labels_ = None
        self.n_clusters_ = None
        self.n_balls_ = None
        self.n_rare_cores_ = None
        self.n_fragment_edges_ = None

    def _mask(self, sizes, gamma, centers, n):
        return guarded_rare_core_mask(
            sizes, gamma, centers, n,
            self.rare_density_z, self.rare_isolation_z,
            self.rare_min_size, self.contrast_tau,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scale:
            X = minmax_scale(X)
        n = len(X)
        rho, _ = knn_density(X, self.k)
        reps = select_representatives(X, rho)
        balls = generate_granular_balls(X, reps, dissolve_small=False)
        sizes = np.array([b.size for b in balls], dtype=float)
        gamma = ball_mean_density(rho, balls)
        centers = np.stack([b.center for b in balls])
        rare = self._mask(sizes, gamma, centers, n)
        small = sizes < np.sqrt(n)
        balls = _redistribute_dropped(X, balls, (small & ~rare) if rare.any() else small)

        balls = reduce_granular_balls(X, balls)
        overlap = pairwise_overlap_mask(balls)
        gamma = ball_mean_density(rho, balls)
        sizes = np.array([b.size for b in balls], dtype=float)
        centers = np.stack([b.center for b in balls])
        radii = np.array([b.radius for b in balls], dtype=float)
        rare = self._mask(sizes, gamma, centers, n)
        fragments = rare_fragment_edges(centers, radii, gamma, rare)
        if rare.any():
            allow = np.logical_or(overlap.copy(), fragments)
            allow = protect_rare_cores(allow, rare, gamma)
            allow = np.logical_or(allow, fragments)
        else:
            allow = None
        gb_labels = propagate_gb_labels(
            X, balls, overlap, allow_merge=allow, extra_edges=fragments, size_gate=False
        )
        labels = assign_boundary_points(X, balls, gb_labels)
        self.labels_ = labels
        self.n_clusters_ = int(len(np.unique(labels)))
        self.n_balls_ = len(balls)
        self.n_rare_cores_ = int(rare.sum())
        self.n_fragment_edges_ = int(fragments.sum() // 2)
        return labels


class MGGBCOverlay:
    """Majority partition from ARP-GB geometry, rare cores overlaid on top.

    Retaining a rare core can no longer break the majority ball chain, because
    the majority partition is computed on the ball set ARP-GB would have built.
    """

    def __init__(self, k: int = 10, scale: bool = True, contrast_tau: float = 0.0,
                 rare_density_z: float = 0.5, rare_isolation_z: float = 0.0,
                 rare_min_size: int = 3):
        self.k = int(k)
        self.scale = scale
        self.contrast_tau = float(contrast_tau)
        self.rare_density_z = float(rare_density_z)
        self.rare_isolation_z = float(rare_isolation_z)
        self.rare_min_size = int(rare_min_size)
        self.labels_ = None
        self.n_clusters_ = None
        self.n_balls_ = None
        self.n_rare_cores_ = None
        self.n_fragment_edges_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scale:
            X = minmax_scale(X)
        n = len(X)
        rho, _ = knn_density(X, self.k)
        reps = select_representatives(X, rho)
        raw = generate_granular_balls(X, reps, dissolve_small=False)
        sizes = np.array([b.size for b in raw], dtype=float)
        gamma = ball_mean_density(rho, raw)
        centers = np.stack([b.center for b in raw])
        if self.contrast_tau > 0:
            rare = guarded_rare_core_mask(
                sizes, gamma, centers, n, self.rare_density_z,
                self.rare_isolation_z, self.rare_min_size, self.contrast_tau,
            )
        else:
            rare = rare_core_mask(
                sizes, gamma, centers, n, self.rare_density_z,
                self.rare_isolation_z, self.rare_min_size,
            )
        small = sizes < np.sqrt(n)
        retained = np.flatnonzero(small & rare)

        # majority layer: exactly the ARP-GB ball set and propagation
        maj_balls = reduce_granular_balls(X, _redistribute_dropped(X, raw, small))
        ov = pairwise_overlap_mask(maj_balls)
        maj_gb = propagate_gb_labels(X, maj_balls, ov, size_gate=False)
        labels = assign_boundary_points(X, maj_balls, maj_gb)

        # rare layer: each retained core becomes its own cluster
        next_label = int(labels.max()) + 1
        for i in retained:
            labels[raw[i].core_indices] = next_label
            next_label += 1

        self.labels_ = labels
        self.n_clusters_ = int(len(np.unique(labels)))
        self.n_balls_ = len(maj_balls) + len(retained)
        self.n_rare_cores_ = int(len(retained))
        self.n_fragment_edges_ = 0
        return labels
