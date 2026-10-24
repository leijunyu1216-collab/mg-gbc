"""Multi-granularity granular-ball clustering (J1 / MG-GBC).

ARP-GB fails in two complementary ways on imbalanced, varying-density data:
  * majority classes are over-split (Shuttle: 7 -> 26)
  * minority classes are absorbed (Zoo/Glass/Ecoli)

The mechanisms are the sqrt(N) dissolve rule and overlap-only label
propagation. MG-GBC keeps ARP-GB's representative points and ball geometry,
then adds a second granularity layer on the balls themselves:

    1. Do not dissolve small *dense* balls (rare cores).
    2. Dissolve other small balls as noise or majority fragments.
    3. Merge rare-core fragments that are nearby and density-similar
       even if boundary reduction has broken geometric overlap.
    4. Never absorb a rare core into a lower-density majority ball.
"""

from __future__ import annotations

import numpy as np

from .core import (
    GranularBall,
    _ball_from_points,
    assign_boundary_points,
    generate_granular_balls,
    knn_density,
    minmax_scale,
    pairwise_overlap_mask,
    propagate_gb_labels,
    reduce_granular_balls,
    select_representatives,
    should_propagate,
)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) + 1e-12


def ball_mean_density(density: np.ndarray, balls: list[GranularBall]) -> np.ndarray:
    return np.array(
        [float(density[b.core_indices].mean()) if len(b.core_indices) else 0.0 for b in balls],
        dtype=float,
    )


def isolation_delta(centers: np.ndarray) -> np.ndarray:
    """Distance from each ball to the nearest other ball."""
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    return dmat.min(axis=1)


def isolation_from_background(
    centers: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Distance from each ball to the nearest non-candidate (background) ball.

    Rare-cluster fragments sit close to each other; isolating from *any* neighbor
    would miss them. Isolation from the background majority is the right test.
    """
    m = len(centers)
    iso = np.zeros(m, dtype=float)
    bg = np.flatnonzero(~candidate)
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    if len(bg) == 0:
        return dmat.min(axis=1)
    for i in range(m):
        iso[i] = float(dmat[i, bg].min())
    return iso


def rare_core_mask(
    sizes: np.ndarray,
    gamma: np.ndarray,
    centers: np.ndarray,
    n_points: int,
    density_z_threshold: float = 0.5,
    isolation_z_threshold: float = 0.0,
    min_size: int = 3,
) -> np.ndarray:
    """A rare core is small, dense, and isolated from the majority background."""
    dens_z = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
    size_cut = np.sqrt(n_points)
    candidate = (
        (sizes >= int(min_size))
        & (sizes < size_cut)
        & (dens_z > float(density_z_threshold))
    )
    if int(candidate.sum()) == 0:
        return candidate
    iso = isolation_from_background(centers, candidate)
    iso_z = 0.6745 * (iso - np.median(iso)) / _mad(iso)
    return candidate & (iso_z > float(isolation_z_threshold))


def noise_mask(sizes: np.ndarray, gamma: np.ndarray, n_points: int) -> np.ndarray:
    size_cut = np.sqrt(n_points)
    dens_z = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
    return (sizes < size_cut) & (dens_z <= 0.0)


def _redistribute_dropped(
    X: np.ndarray, balls: list[GranularBall], drop: np.ndarray
) -> list[GranularBall]:
    if not drop.any() or int((~drop).sum()) < 1:
        return balls
    leftover = np.concatenate([balls[i].indices for i in np.flatnonzero(drop)])
    kept = [balls[i] for i in np.flatnonzero(~drop)]
    centers = np.stack([b.center for b in kept])
    pts = X[leftover]
    nn = np.argmin(np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2), axis=1)
    buckets: dict[int, list[int]] = {i: [] for i in range(len(kept))}
    for idx, b_id in zip(leftover, nn):
        buckets[int(b_id)].append(int(idx))
    rebuilt = []
    for i, gb in enumerate(kept):
        extra = buckets[i]
        merged = (
            np.unique(np.concatenate([gb.indices, np.asarray(extra, dtype=int)]))
            if extra
            else gb.indices
        )
        new_gb = _ball_from_points(X, merged)
        if new_gb is not None:
            rebuilt.append(new_gb)
    return rebuilt


def density_delta(centers: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """DPC-style delta: distance to the nearest higher-density ball."""
    m = len(gamma)
    delta = np.zeros(m, dtype=float)
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(-gamma)
    for rank, i in enumerate(order):
        higher = order[:rank]
        if len(higher) == 0:
            delta[i] = float(dmat[i].max()) if m > 1 else 1.0
        else:
            delta[i] = float(dmat[i, higher].min())
    return delta


def similar_density_mask(gamma: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Default to ARP-GB (allow overlapping pairs). Block only high-contrast overlaps.

    Majority fragments have similar density and should still merge. Rare-core vs
    background pairs are contrast outliers and must not share a label.
    """
    allow = overlap.copy()
    contrasts = []
    pairs = []
    m = len(gamma)
    for i in range(m):
        for j in range(i + 1, m):
            if not overlap[i, j]:
                continue
            c = abs(gamma[i] - gamma[j]) / max(gamma[i], gamma[j], 1e-12)
            contrasts.append(c)
            pairs.append((i, j, c))
    if len(contrasts) < 2:
        return allow
    arr = np.asarray(contrasts)
    thr = float(np.median(arr) + 0.6745 * _mad(arr) * 2.0)
    for i, j, c in pairs:
        if c > thr:
            allow[i, j] = allow[j, i] = False
    return allow


def protect_rare_cores(allow: np.ndarray, rare: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Block absorption of a rare core into a sparser non-rare ball.

    Rare-rare pairs stay mergeable so fragments of one minority cluster can join.
    """
    allow = allow.copy()
    rare_ids = np.flatnonzero(rare)
    non_rare = np.flatnonzero(~rare)
    for i in rare_ids:
        for j in non_rare:
            if not allow[i, j]:
                continue
            if gamma[j] < gamma[i]:
                allow[i, j] = allow[j, i] = False
    return allow


def rare_fragment_edges(
    centers: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    rare: np.ndarray,
    density_contrast_threshold: float = 0.25,
    near_radius_factor: float = 1.5,
    background_closeness_factor: float = 0.5,
    corridor_clearance_factor: float = 0.5,
) -> np.ndarray:
    """Link rare cores that are fragments of one minority cluster.

    Boundary reduction shrinks radii, so true fragments often no longer overlap.
    They remain similar in density, and either still geometrically local
    after radius shrinkage, or connected by a corridor that stays farther
    from majority balls than from the two cores themselves.

    The rule is a description of fragmentation, not a second merger test: it
    restores contact that trimming removed, so it applies only to pairs that
    are not in the overlap graph.  The overlap graph is built by the caller,
    which applies that condition when ``fragment_guard`` is on.
    """
    m = len(gamma)
    edges = np.zeros((m, m), dtype=bool)
    ids = np.flatnonzero(rare)
    if len(ids) < 2:
        return edges

    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    non_rare = np.flatnonzero(~rare)
    if len(non_rare) == 0:
        iso = dmat.min(axis=1)
    else:
        iso = dmat[:, non_rare].min(axis=1)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = int(ids[a]), int(ids[b])
            d = float(dmat[i, j])
            rsum = float(radii[i] + radii[j])
            contrast = abs(gamma[i] - gamma[j]) / max(gamma[i], gamma[j], 1e-12)
            if contrast > float(density_contrast_threshold):
                continue
            near = d <= float(near_radius_factor) * max(rsum, 1e-12)
            closer = d < float(background_closeness_factor) * min(float(iso[i]), float(iso[j]))
            mid = 0.5 * (centers[i] + centers[j])
            if len(non_rare) == 0:
                corridor = True
            else:
                d_bg_mid = float(np.linalg.norm(centers[non_rare] - mid, axis=1).min())
                corridor = d_bg_mid > float(corridor_clearance_factor) * d
            # Geometric proximity alone is not evidence that two rare balls
            # belong to the same cluster.  In particular, two different rare
            # classes can have overlapping radii after boundary assignment.
            # Require the pair to be mutually closer than either ball is to
            # the background, then use either radius proximity or a clear
            # background-free corridor as supporting evidence.
            if closer and (near or corridor):
                edges[i, j] = edges[j, i] = True
    return edges


class MGGBC:
    """Multi-granularity granular-ball clustering for imbalanced / multi-density data."""

    def __init__(
        self,
        k: int = 10,
        scale: bool = True,
        keep_rare: bool = True,
        protect: bool = True,
        fragments: bool = True,
        size_gate: bool = False,
        rare_density_z: float = 0.5,
        rare_isolation_z: float = 0.0,
        rare_min_size: int = 3,
        fragment_density_contrast: float = 0.25,
        fragment_near_factor: float = 1.5,
        fragment_closeness_factor: float = 0.5,
        fragment_corridor_factor: float = 0.5,
        fragment_guard: bool = True,
    ):
        self.k = int(k)
        self.scale = scale
        self.keep_rare = bool(keep_rare)
        self.protect = bool(protect)
        self.use_fragments = bool(fragments)
        # MG-GBC uses one fixed graph policy for every ablation.  The original
        # reconstruction's size-ratio gate can be restored explicitly, but it
        # must never switch on or off as a side effect of rare-core detection.
        self.size_gate = bool(size_gate)
        self.rare_density_z = float(rare_density_z)
        self.rare_isolation_z = float(rare_isolation_z)
        self.rare_min_size = int(rare_min_size)
        self.fragment_density_contrast = float(fragment_density_contrast)
        self.fragment_near_factor = float(fragment_near_factor)
        self.fragment_closeness_factor = float(fragment_closeness_factor)
        self.fragment_corridor_factor = float(fragment_corridor_factor)
        # A fragment edge may restore contact that radius reduction destroyed,
        # but it may not overturn a merge the overlap test refused.  The guard
        # can be switched off to ablate the admissibility condition; with it
        # off, a fragment edge may also join a pair that still overlaps.
        self.fragment_guard = bool(fragment_guard)
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
        balls = generate_granular_balls(X, reps, dissolve_small=False)
        sizes = np.array([b.size for b in balls], dtype=float)
        gamma = ball_mean_density(rho, balls)
        centers = np.stack([b.center for b in balls])
        rare = rare_core_mask(
            sizes,
            gamma,
            centers,
            n,
            self.rare_density_z,
            self.rare_isolation_z,
            self.rare_min_size,
        )
        if not self.keep_rare:
            rare = np.zeros_like(rare)
        # Bookkeeping only: the branch is chosen by the first-pass predicate,
        # and a core can disappear after redistribution and trimming.
        self.n_first_pass_rare_ = int(rare.sum())

        if rare.any():
            small = sizes < np.sqrt(n)
            drop = small & (~rare)
            balls = _redistribute_dropped(X, balls, drop)
        else:
            small = sizes < np.sqrt(n)
            balls = _redistribute_dropped(X, balls, small)

        balls = reduce_granular_balls(X, balls)
        overlap = pairwise_overlap_mask(balls)
        gamma = ball_mean_density(rho, balls)
        sizes = np.array([b.size for b in balls], dtype=float)
        centers = np.stack([b.center for b in balls])
        radii = np.array([b.radius for b in balls], dtype=float)
        rare = rare_core_mask(
            sizes,
            gamma,
            centers,
            n,
            self.rare_density_z,
            self.rare_isolation_z,
            self.rare_min_size,
        )
        if not self.keep_rare:
            rare = np.zeros_like(rare)
        fragments = (
            rare_fragment_edges(
                centers,
                radii,
                gamma,
                rare,
                self.fragment_density_contrast,
                self.fragment_near_factor,
                self.fragment_closeness_factor,
                self.fragment_corridor_factor,
            )
            if self.use_fragments
            else np.zeros((len(balls), len(balls)), dtype=bool)
        )
        if fragments.any() and overlap.any() and self.fragment_guard:
            # A fragment edge exists to restore contact that radius reduction
            # destroyed.  It must not overrule a merge the overlap test already
            # refused: there the balls do overlap and should_propagate has
            # decided, so a fragment edge would harden a rejected link into a
            # merge.  Edges between pairs that already overlap and merge are
            # redundant and are kept (they change no label).
            for i, j in zip(*np.nonzero(np.triu(fragments & overlap))):
                if not should_propagate(X, balls, overlap, int(i), int(j), n,
                                        size_gate=self.size_gate):
                    fragments[i, j] = fragments[j, i] = False

        if rare.any():
            allow = overlap.copy()
            allow = np.logical_or(allow, fragments)
            if self.protect:
                allow = protect_rare_cores(allow, rare, gamma)
            allow = np.logical_or(allow, fragments)
        else:
            allow = None

        gb_labels = propagate_gb_labels(
            X,
            balls,
            overlap,
            allow_merge=allow,
            extra_edges=fragments,
            size_gate=self.size_gate,
        )
        labels = assign_boundary_points(X, balls, gb_labels)
        self.labels_ = labels
        self.n_clusters_ = int(len(np.unique(labels)))
        self.n_balls_ = len(balls)
        self.n_rare_cores_ = int(rare.sum())
        self.n_fragment_edges_ = int(fragments.sum() // 2)
        return labels
