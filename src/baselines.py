from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from .core import minmax_scale


class KMeansBaseline:
    def __init__(self, n_clusters: int):
        self.n_clusters = int(n_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = minmax_scale(X)
        return KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=0
        ).fit_predict(X)


class DBSCANBaseline:
    def __init__(self, eps: float, min_samples: int):
        self.eps = float(eps)
        self.min_samples = int(min_samples)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = minmax_scale(X)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(X)
        # paper treats noise as its own label
        return labels


class DPCBaseline:
    """Rodriguez-Laio DPC. dc is a percentile of pairwise distances (paper's %)."""

    def __init__(self, percent: float, n_clusters: int):
        self.percent = float(percent)
        self.n_clusters = int(n_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = minmax_scale(np.asarray(X, dtype=float))
        n = len(X)
        if n > 4500:
            return self._knn_dpc(X)
        dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        tri = dmat[np.triu_indices(n, 1)]
        dc = float(np.percentile(tri, 100.0 * self.percent))
        dc = max(dc, 1e-12)
        rho = np.sum(dmat < dc, axis=1) - 1.0
        return self._assign(dmat, rho)

    def _knn_dpc(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        k = min(50, max(5, int(np.sqrt(n))))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X)
        rho = 1.0 / np.maximum(dist[:, 1:].mean(axis=1), 1e-12)
        dmat_approx = np.full((n, n), np.inf)
        for i in range(n):
            dmat_approx[i, idx[i]] = dist[i]
            dmat_approx[idx[i], i] = dist[i]
        np.fill_diagonal(dmat_approx, 0.0)
        return self._assign(dmat_approx, rho)

    def _assign(self, dmat: np.ndarray, rho: np.ndarray) -> np.ndarray:
        n = len(rho)
        order = np.argsort(-rho)
        delta = np.zeros(n)
        nnh = np.zeros(n, dtype=int)
        for t, i in enumerate(order):
            if t == 0:
                delta[i] = np.nanmax(dmat[i])
                nnh[i] = i
            else:
                prev = order[:t]
                row = dmat[i, prev]
                j = int(np.argmin(row))
                delta[i] = float(row[j])
                nnh[i] = int(prev[j])
        score = rho * delta
        centers = np.argsort(-score)[: self.n_clusters]
        labels = -np.ones(n, dtype=int)
        labels[centers] = np.arange(self.n_clusters)
        for i in order:
            if labels[i] < 0:
                labels[i] = labels[nnh[i]]
        # leftover
        if (labels < 0).any():
            unlabeled = np.flatnonzero(labels < 0)
            d = dmat[np.ix_(unlabeled, centers)]
            labels[unlabeled] = np.argmin(d, axis=1)
        return labels


class SNN_DPCBaseline:
    """Shared-nearest-neighbor DPC (Liu et al., 2018), compact reconstruction."""

    def __init__(self, k: int, n_clusters: int):
        self.k = int(k)
        self.n_clusters = int(n_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = minmax_scale(np.asarray(X, dtype=float))
        n = len(X)
        k = int(np.clip(self.k, 2, n - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        sets = [set(row.tolist()) for row in idx]
        rho = np.zeros(n)
        for i in range(n):
            rho[i] = sum(len(sets[i] & sets[j]) for j in idx[i])
        dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) if n <= 4500 else None
        if dmat is None:
            dpc = DPCBaseline(0.02, self.n_clusters)
            return dpc._knn_dpc(X)  # type: ignore[attr-defined]
        return DPCBaseline(0.02, self.n_clusters)._assign(dmat, rho)


class RNN_DBSCANBaseline:
    """RNN-DBSCAN-style clustering (Bryant & Cios, 2018), compact reconstruction."""

    def __init__(self, k: int):
        self.k = int(k)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = minmax_scale(np.asarray(X, dtype=float))
        n = len(X)
        k = int(np.clip(self.k, 1, n - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        rnn_count = np.zeros(n, dtype=int)
        for i in range(n):
            rnn_count[idx[i]] += 1
        is_core = rnn_count >= k
        labels = -np.ones(n, dtype=int)
        current = 0
        for i in range(n):
            if not is_core[i] or labels[i] >= 0:
                continue
            stack = [i]
            labels[i] = current
            while stack:
                p = stack.pop()
                for q in idx[p]:
                    if labels[q] >= 0:
                        continue
                    labels[q] = current
                    if is_core[q]:
                        stack.append(int(q))
            current += 1
        # leftover noise stays -1, matching DBSCAN convention
        return labels
