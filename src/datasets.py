from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, load_wine


PAPER_K = {
    "Zoo": 4,
    "Iris": 47,
    "Glass": 38,
    "Ecoli": 6,
    "Shuttle": 34,
    "Wine": 10,
}

PAPER_ARPGB = {
    # Table 7
    "Zoo": {"ACC": 0.8911, "NMI": 0.8785, "ARI": 0.9184, "k_hat": 5, "k_true": 7},
    "Iris": {"ACC": 0.9667, "NMI": 0.8851, "ARI": 0.9038, "k_hat": 3, "k_true": 3},
    "Glass": {"ACC": 0.5234, "NMI": 0.4209, "ARI": 0.2530, "k_hat": 4, "k_true": 6},
    "Ecoli": {"ACC": 0.8125, "NMI": 0.6968, "ARI": 0.7436, "k_hat": 4, "k_true": 8},
    "Shuttle": {"ACC": 0.4619, "NMI": 0.3551, "ARI": 0.1556, "k_hat": 26, "k_true": 7},
}

# Table 6 (real-world) / Table 2 (synthetic) comparison hyperparameters
PAPER_BASELINE_PARAMS = {
    "Zoo": {"kmeans": 7, "dbscan": (1.5, 3), "dpc": (0.024, 7), "rnn_k": 7, "snn_k": 11},
    "Iris": {"kmeans": 3, "dbscan": (0.4, 10), "dpc": (0.028, 3), "rnn_k": 7, "snn_k": 15},
    "Glass": {"kmeans": 6, "dbscan": (0.32, 11), "dpc": (0.005, 6), "rnn_k": 4, "snn_k": 20},
    "Ecoli": {"kmeans": 8, "dbscan": (0.18, 9), "dpc": (0.002, 8), "rnn_k": 3, "snn_k": 6},
    "Shuttle": {"kmeans": 7, "dbscan": (0.08, 25), "dpc": (0.021, 7), "rnn_k": 47, "snn_k": 15},
}


def synthetic_imbalance(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.35, size=(800, 2))
    c2 = rng.normal(loc=[3.2, 0.3], scale=0.22, size=(120, 2))
    c3 = rng.normal(loc=[1.5, 3.8], scale=0.10, size=(25, 2))
    X = np.vstack([c1, c2, c3])
    y = np.array([0] * 800 + [1] * 120 + [2] * 25)
    return X, y, "Synthetic-Imbalance"


def synthetic_split_rare(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    majority = rng.normal(loc=[0.0, 0.0], scale=0.45, size=(700, 2))
    frag_a = rng.normal(loc=[5.0, 5.0], scale=0.05, size=(22, 2))
    frag_b = rng.normal(loc=[5.0, 6.15], scale=0.05, size=(22, 2))
    X = np.vstack([majority, frag_a, frag_b])
    y = np.array([0] * 700 + [1] * 22 + [1] * 22)
    return X, y, "Synthetic-SplitRare"


def default_k(n: int) -> int:
    return int(np.clip(np.round(np.sqrt(n)), 2, max(2, n - 1)))


def load_olivetti_pca(n_components: int = 30) -> tuple[np.ndarray, np.ndarray, str]:
    """40 people, 10 images each. PCA keeps compact identity clusters.

    Each class has size 10, below sqrt(400)=20, so a dense identity ball can
    survive as a rare core instead of being dissolved.
    """
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA

    faces = fetch_olivetti_faces(shuffle=False)
    X = PCA(n_components=n_components, random_state=0).fit_transform(faces.data)
    return np.asarray(X, dtype=float), np.asarray(faces.target, dtype=int), "Olivetti-PCA"


def load_digits_rare(seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """Semi-real: real digit pixels, controlled imbalance.

    160 samples of digit 0 plus 12 of digit 1 and 12 of digit 4.
    Both minorities are compact and smaller than sqrt(n)≈13.6.
    """
    from sklearn.datasets import load_digits

    data = load_digits()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    rng = np.random.default_rng(seed)
    n0 = int((y == 0).sum())
    maj = rng.choice(np.flatnonzero(y == 0), size=min(160, n0), replace=False)
    r1 = rng.choice(np.flatnonzero(y == 1), size=12, replace=False)
    r4 = rng.choice(np.flatnonzero(y == 4), size=12, replace=False)
    idx = np.concatenate([maj, r1, r4])
    yr = np.array([0] * len(maj) + [1] * 12 + [2] * 12)
    return X[idx], yr, "Digits-Rare"


def load_covtype_rare(seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """Semi-real forest-cover subset: large Lodgepole majority, two small types.

    Class 2 (Lodgepole pine) subsampled to 600; class 4 (Cottonwood/Willow) and
    class 5 (Aspen) kept at 20 points nearest their class centroid so the rare
    groups stay compact. Remote-sensing features, n≈640, sqrt(n)≈25.
    """
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)
    rng = np.random.default_rng(seed)

    def _take_maj(label: int, n_take: int) -> np.ndarray:
        idx = np.flatnonzero(y == label)
        return rng.choice(idx, size=min(n_take, len(idx)), replace=False)

    def _take_compact(label: int, n_take: int) -> np.ndarray:
        idx = np.flatnonzero(y == label)
        pts = X[idx]
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        keep = np.argsort(dist)[:n_take]
        return idx[keep]

    parts = [_take_maj(2, 600), _take_compact(4, 20), _take_compact(5, 20)]
    idx = np.concatenate(parts)
    yr = np.array([0] * len(parts[0]) + [1] * len(parts[1]) + [2] * len(parts[2]))
    return X[idx], yr, "Covertype-Rare"


# Real OpenML / UCI sets whose smallest class is often below sqrt(n).
RARE_OPENML = [
    ("Yeast", "yeast", 181),
    ("Pageblocks", "page-blocks", 30),
    ("Lymphography", "lymph", 10),
    ("Dermatology", "dermatology", 35),
    ("Thyroid", "thyroid", 40497),
    ("Wilt", "wilt", 40983),
    ("Vehicle", "vehicle", 54),
    ("Vowel", "vowel", 307),
    ("Segment", "segment", 36),
    ("Satimage", "satimage", 182),
]


def experiment_k(name: str, n: int) -> int:
    """Table 6 k when it exists; otherwise clip sqrt(n) into [5, 10]."""
    if name in PAPER_K:
        return int(PAPER_K[name])
    return int(np.clip(default_k(n), 5, 10))


def load_rare_openml(label: str, name: str, data_id: int) -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        X, y = _from_openml(name, data_id=data_id)
        return X, y, label
    except Exception as exc:
        try:
            X, y = _from_openml(name)
            return X, y, label
        except Exception as exc2:
            print(f"load {label} failed: {exc} | {exc2}")
            return None


def load_rare_candidates() -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Real and semi-real sets for rare-core screening. Shuttle is not included."""
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    try:
        out.append(load_olivetti_pca())
    except Exception as exc:
        print(f"load Olivetti-PCA failed: {exc}")
    try:
        out.append(load_digits_rare())
    except Exception as exc:
        print(f"load Digits-Rare failed: {exc}")
    try:
        out.append(load_covtype_rare())
    except Exception as exc:
        print(f"load Covertype-Rare failed: {exc}")
    for item in RARE_OPENML:
        loaded = load_rare_openml(*item)
        if loaded is not None:
            out.append(loaded)
    for name in ("ecoli", "zoo", "glass", "wine", "iris"):
        loaded = load_uci_named(name)
        if loaded is not None:
            out.append(loaded)
    return out


def load_sklearn_toy() -> list[tuple[np.ndarray, np.ndarray, str]]:
    iris = load_iris()
    wine = load_wine()
    return [
        (iris.data, iris.target, "Iris"),
        (wine.data, wine.target, "Wine"),
    ]


def _encode_frame(frame) -> np.ndarray:
    import pandas as pd

    cols = []
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(pd.to_numeric(s, errors="coerce"))
            continue
        lowered = s.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()) <= {"true", "false", "0", "1", "yes", "no"}:
            cols.append(lowered.map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}))
        else:
            cols.append(pd.factorize(s)[0].astype(float))
    X = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    return X


def _from_openml(name: str, data_id: int | None = None, target: str | None = None):
    from sklearn.datasets import fetch_openml

    kwargs = {"as_frame": True, "parser": "auto"}
    if data_id is not None:
        bunch = fetch_openml(data_id=data_id, **kwargs)
    else:
        bunch = fetch_openml(name, version=1, **kwargs)
    X_df = bunch.data
    y_s = bunch.target
    if hasattr(X_df, "columns"):
        if target is not None and target in X_df.columns:
            y_s = X_df[target]
            X_df = X_df.drop(columns=[target])
        X = _encode_frame(X_df)
    else:
        X = np.asarray(X_df, dtype=float)
    _, y = np.unique(np.asarray(y_s).astype(str), return_inverse=True)
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


def load_uci_named(name: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    key = name.lower()
    try:
        if key == "iris":
            data = load_iris()
            return np.asarray(data.data, dtype=float), np.asarray(data.target), "Iris"
        if key == "wine":
            data = load_wine()
            return np.asarray(data.data, dtype=float), np.asarray(data.target), "Wine"
        if key == "zoo":
            X, y = _from_openml("zoo", data_id=62)
            return X, y, "Zoo"
        if key == "glass":
            X, y = _from_openml("glass", data_id=41)
            return X, y, "Glass"
        if key == "ecoli":
            X, y = _from_openml("ecoli", data_id=39)
            return X, y, "Ecoli"
        if key == "shuttle":
            last_err = None
            for data_id in (40685, 258, 354, 152):
                try:
                    X, y = _from_openml("shuttle", data_id=data_id)
                    return X, y, "Shuttle"
                except Exception as exc:
                    last_err = exc
                    continue
            try:
                X, y = _from_openml("shuttle")
                return X, y, "Shuttle"
            except Exception as exc:
                last_err = exc
            raise RuntimeError(last_err)
    except Exception as exc:
        print(f"load {name} failed: {exc}")
        return None
    return None


def load_paper_sets() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    sets: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    syn_x, syn_y, syn_n = synthetic_imbalance()
    sets.append((syn_x, syn_y, syn_n, 12))
    split_x, split_y, split_n = synthetic_split_rare()
    sets.append((split_x, split_y, split_n, 12))
    for name, k in [("zoo", 4), ("glass", 38), ("ecoli", 6)]:
        loaded = load_uci_named(name)
        if loaded is not None:
            X, y, n = loaded
            sets.append((X, y, n, k))
    return sets


def load_alignment_sets(include_shuttle: bool = False):
    names = [("zoo", 4), ("glass", 38), ("ecoli", 6), ("iris", 47)]
    if include_shuttle:
        names.append(("shuttle", 34))
    out = []
    for name, k in names:
        loaded = load_uci_named(name)
        if loaded is not None:
            X, y, n = loaded
            out.append((X, y, n, k))
    return out
