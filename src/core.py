from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors


def minmax_scale(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (X - lo) / span


def knn_density(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """ARP-GB Eq. (2). Returns (density, neighbor_indices)."""
    n = len(X)
    k_use = int(np.clip(k, 1, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm="auto")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    dik = np.maximum(dist[:, -1], 1e-12)
    ratio = dist / dik[:, None]
    mean_sq = np.mean(ratio**2, axis=1)
    rho = 1.0 / np.maximum(mean_sq, 1e-12)
    rho = np.nan_to_num(rho, nan=0.0, posinf=1.0 / 1e-12, neginf=0.0)
    return rho, idx


def _window_size(
    rank: int,
    B: int,
    gaps: np.ndarray,
    gap_median: float,
    rho_sorted: np.ndarray,
    rho_median: float,
) -> float:
    n = len(rho_sorted)
    end = min(n, rank + B + 1)
    denom = float(np.max(gaps[rank:end])) if end > rank else 0.0
    s = 1.0 if denom < 1e-15 else gap_median / denom
    rho_i = max(float(rho_sorted[rank]), 1e-12)
    adapt = min(1.0, s + rho_median / rho_i)
    return B * (1.0 + adapt)


def select_representatives(X: np.ndarray, density: np.ndarray) -> np.ndarray:
    """ARP-GB Algorithm 1.

    Walk the density-sorted sequence in adaptive windows of length R(i).
    Advancing the start by R after each pick keeps m ~ sqrt(n), matching
    the complexity argument in the original paper.
    """
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)
    B = int(np.ceil(np.sqrt(n)))
    order = np.argsort(-density)
    rho_sorted = density[order]
    rho_median = float(np.median(density[np.isfinite(density)]))
    gaps = np.append(np.abs(np.diff(rho_sorted)), 0.0)
    gaps = np.nan_to_num(gaps, nan=0.0)
    gap_median = float(np.median(gaps))

    selected: list[int] = [int(order[0])]
    selected_set = {int(order[0])}
    rank = 0

    while True:
        if rank >= n:
            break
        R = _window_size(rank, B, gaps, gap_median, rho_sorted, rho_median)
        R = max(2.0, float(R))
        if n - len(selected) < R:
            break
        end = min(n, rank + int(np.ceil(R)))
        window = [int(i) for i in order[rank:end] if int(i) not in selected_set]
        if not window:
            rank = end if end > rank else rank + 1
            if rank >= n:
                break
            continue
        reps = X[np.array(selected)]
        cand = X[np.array(window)]
        dmin = np.min(np.linalg.norm(cand[:, None, :] - reps[None, :, :], axis=2), axis=1)
        pick = window[int(np.argmax(dmin))]
        selected.append(pick)
        selected_set.add(pick)
        rank = end
        if len(selected) >= max(2, B * 2):
            break
        if len(selected) >= n:
            break
    return np.asarray(selected, dtype=int)


@dataclass
class GranularBall:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    core_indices: np.ndarray
    boundary_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def size(self) -> int:
        return int(len(self.indices))


def _ball_from_points(X: np.ndarray, indices: np.ndarray) -> GranularBall | None:
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return None
    pts = X[indices]
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    radius = float(dist.max()) if len(dist) else 0.0
    return GranularBall(
        indices=indices,
        center=center,
        radius=max(radius, 1e-12),
        core_indices=indices.copy(),
    )


def generate_granular_balls(
    X: np.ndarray,
    representative_idx: np.ndarray,
    dissolve_small: bool = True,
    size_cut_factor: float = 1.0,
) -> list[GranularBall]:
    """ARP-GB Algorithm 2. If dissolve_small, drop balls with size < sqrt(N).

    ``size_cut_factor`` scales the cut and defaults to the published value
    (1.0 times sqrt(n)).  It exists so that the host's one design knob can be
    swept; MG-GBC itself never changes it, because its claim is about what
    happens to a ball below the line, not about where the line is drawn.
    """
    n, d = X.shape
    reps = X[representative_idx]
    dist = np.linalg.norm(X[:, None, :] - reps[None, :, :], axis=2)
    assign = np.argmin(dist, axis=1)

    groups: dict[int, list[int]] = {i: [] for i in range(len(representative_idx))}
    for p, a in enumerate(assign):
        groups[int(a)].append(p)

    balls: list[GranularBall] = []
    leftover: list[int] = []
    size_cut = float(size_cut_factor) * np.sqrt(n)
    for members in groups.values():
        gb = _ball_from_points(X, np.array(members, dtype=int))
        if gb is None:
            continue
        if dissolve_small and gb.size < size_cut and len(representative_idx) > 1:
            leftover.extend(members)
        else:
            balls.append(gb)

    if not balls:
        return [
            _ball_from_points(X, np.arange(n, dtype=int))  # type: ignore[list-item]
        ]

    if leftover:
        centers = np.stack([b.center for b in balls])
        pts = X[np.array(leftover)]
        nn = np.argmin(
            np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2), axis=1
        )
        buckets: dict[int, list[int]] = {i: [] for i in range(len(balls))}
        for idx, b_id in zip(leftover, nn):
            buckets[int(b_id)].append(idx)
        rebuilt: list[GranularBall] = []
        for i, gb in enumerate(balls):
            merged = np.concatenate(
                [gb.indices, np.asarray(buckets[i], dtype=int)]
            ) if buckets[i] else gb.indices
            new_gb = _ball_from_points(X, np.unique(merged))
            if new_gb is not None:
                rebuilt.append(new_gb)
        balls = rebuilt
    return [b for b in balls if b is not None]


def _quality(X: np.ndarray, gb: GranularBall) -> float:
    pts = X[gb.core_indices]
    if len(pts) == 0:
        return 0.0
    dist = np.linalg.norm(pts - gb.center, axis=1)
    return float(len(pts) / max(dist.sum(), 1e-12))


def reduce_granular_balls(X: np.ndarray, balls: list[GranularBall]) -> list[GranularBall]:
    """ARP-GB Algorithm 3: at most two core-radius trims per ball."""
    dim = max(X.shape[1], 1)
    dim_factor = 1.0 + np.log(dim) / 2.0
    reduced: list[GranularBall] = []

    for gb in balls:
        current = gb
        for _ in range(2):
            core_idx = current.core_indices
            if len(core_idx) < 3:
                break
            pts = X[core_idx]
            dist = np.linalg.norm(pts - current.center, axis=1)
            r = float(max(dist.max(), 1e-12))
            r_avg = float(max(np.mean(dist), np.median(dist)))
            # Eq. (7)
            r_core = r_avg * r / r + (1.0 - r_avg / r) * r_avg
            keep = dist <= r_core + 1e-15
            if keep.sum() < 2 or keep.all():
                break
            q_orig = _quality(X, current)
            n_orig = len(core_idx)
            new_core = core_idx[keep]
            boundary = core_idx[~keep]
            candidate = _ball_from_points(X, new_core)
            if candidate is None:
                break
            candidate.boundary_indices = np.unique(
                np.concatenate([current.boundary_indices, boundary])
            )
            candidate.core_indices = new_core
            # full membership = core + boundary (for later assignment)
            candidate.indices = np.unique(
                np.concatenate([new_core, candidate.boundary_indices])
            )
            n_new = len(new_core)
            q_new = _quality(X, candidate)
            if q_new * (n_new / max(n_orig, 1)) * dim_factor >= q_orig:
                current = candidate
            else:
                break
        reduced.append(current)
    return reduced


def pairwise_overlap_mask(balls: list[GranularBall]) -> np.ndarray:
    m = len(balls)
    overlap = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(balls[i].center - balls[j].center))
            if d < balls[i].radius + balls[j].radius:
                overlap[i, j] = overlap[j, i] = True
    return overlap


def true_points(
    X: np.ndarray,
    balls: list[GranularBall],
    overlap: np.ndarray,
    i: int,
    partner: int | None = None,
) -> np.ndarray:
    members = [balls[i].indices]
    if partner is None:
        nbrs = np.flatnonzero(overlap[i])
    else:
        nbrs = np.array([partner], dtype=int)
    for j in nbrs:
        members.append(balls[j].indices)
    pool = np.unique(np.concatenate(members))
    dist = np.linalg.norm(X[pool] - balls[i].center, axis=1)
    return pool[dist < balls[i].radius]


def overlap_stats(
    X: np.ndarray, balls: list[GranularBall], overlap: np.ndarray, i: int, j: int
) -> tuple[float, float]:
    t1 = true_points(X, balls, overlap, i, partner=j)
    t2 = true_points(X, balls, overlap, j, partner=i)
    if len(t1) == 0 or len(t2) == 0:
        return 0.0, 0.0
    inter = np.intersect1d(t1, t2, assume_unique=False)
    union = np.union1d(t1, t2)
    oratio = float(len(inter) / min(len(t1), len(t2)))
    jacc = float(len(inter) / max(len(union), 1))
    return oratio, jacc


def overlap_ratio(
    X: np.ndarray, balls: list[GranularBall], overlap: np.ndarray, i: int, j: int
) -> float:
    oratio, _ = overlap_stats(X, balls, overlap, i, j)
    return oratio


def should_propagate(
    X: np.ndarray,
    balls: list[GranularBall],
    overlap: np.ndarray,
    i: int,
    j: int,
    n: int,
    size_gate: bool = True,
) -> bool:
    if not overlap[i, j]:
        return False
    oratio, jacc = overlap_stats(X, balls, overlap, i, j)
    r1, r2 = balls[i].radius, balls[j].radius
    d = float(np.linalg.norm(balls[i].center - balls[j].center))
    geom = (r1 + r2 - d) / max(r1 + r2, 1e-12)
    s1, s2 = balls[i].size, balls[j].size
    size_ratio = max(s1, s2) / max(min(s1, s2), 1)
    # A large ball covering a much smaller one inflates OR via min(|T1|,|T2|).
    # Callers choose this graph policy explicitly.  MG-GBC keeps it fixed
    # across all keep/no-keep ablations; ARP-GB uses the guarded default.
    if size_gate and size_ratio > 3.0 and jacc < 0.25:
        return False
    # Tiny geometric lenses make the original OR>=geom test vacuous.
    cond1 = oratio >= max(geom, 0.10)
    bp = np.union1d(balls[i].boundary_indices, balls[j].boundary_indices)
    t2 = len(bp) / max(n, 1)
    # cond2 is only for shallow leftover overlap; do not rescue pairs that
    # already failed a meaningful cond1 (deep overlap, insufficient OR).
    cond2 = (
        len(bp) > 0
        and geom <= 0.32
        and oratio >= max(t2, 0.10)
        and jacc >= 0.05
    )
    return bool(cond1 or cond2)


def propagate_gb_labels(
    X: np.ndarray,
    balls: list[GranularBall],
    overlap: np.ndarray,
    allow_merge: np.ndarray | None = None,
    extra_edges: np.ndarray | None = None,
    size_gate: bool = True,
) -> np.ndarray:
    """BFS label propagation on overlapping GBs, plus optional fragment edges."""
    m = len(balls)
    n = len(X)
    labels = -np.ones(m, dtype=int)
    current = 0
    extra = extra_edges if extra_edges is not None else np.zeros((m, m), dtype=bool)
    for seed in range(m):
        if labels[seed] >= 0:
            continue
        labels[seed] = current
        stack = [seed]
        while stack:
            i = stack.pop()
            nbrs = np.unique(np.concatenate([np.flatnonzero(overlap[i]), np.flatnonzero(extra[i])]))
            for j in nbrs:
                if labels[j] >= 0:
                    continue
                if allow_merge is not None and not allow_merge[i, j]:
                    continue
                if extra[i, j] or should_propagate(
                    X, balls, overlap, i, j, n, size_gate=size_gate
                ):
                    labels[j] = current
                    stack.append(j)
        current += 1
    return labels


def assign_boundary_points(
    X: np.ndarray,
    balls: list[GranularBall],
    gb_labels: np.ndarray,
) -> np.ndarray:
    """ARP-GB multi-distance, Eq. (14)."""
    n = len(X)
    labels = -np.ones(n, dtype=int)
    core_assigned = np.zeros(n, dtype=bool)
    for gb, lab in zip(balls, gb_labels):
        labels[gb.core_indices] = lab
        core_assigned[gb.core_indices] = True

    unlabeled = np.flatnonzero(~core_assigned)
    if len(unlabeled) == 0:
        return labels

    centers = np.stack([b.center for b in balls])
    radii = np.array([b.radius for b in balls])
    pts = X[unlabeled]
    dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    mdis = 2.0 * dist - radii[None, :]
    nearest = np.argmin(mdis, axis=1)
    labels[unlabeled] = gb_labels[nearest]
    return labels
