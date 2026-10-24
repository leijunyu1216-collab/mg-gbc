"""Run the frozen recent baselines (NGBC, EKM) on the paper's benchmark sets.

Matched preprocessing: every method sees MinMax-scaled features, the scaling
used by ARP-GB and MG-GBC in the paper.  ARP-GB and MG-GBC rows are recomputed
here so that all four methods share one input matrix.

NGBC is the authors' implementation (n-by-n distance matrix); sets with
n > 2000 are marked resource-ineligible rather than replaced.  EKM receives
the true number of classes, as its API requires; this is disclosed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "EKM" / "python"))
sys.path.insert(0, str(ROOT / "third_party" / "NGBC" / "NGBC-code"))

from ekm_sklearn import EKM
import Naturalneighbor_Split as ngbc_split
from NaturalBall.NaturalBallClustering_synthetic import form_nb_cluster_by_neighbor

from src.arp_gb import ARPGB
from src.datasets import (
    RARE_OPENML,
    PAPER_K,
    experiment_k,
    load_rare_openml,
    load_uci_named,
)
from src.metrics import evaluate
from src.mg_gbc import MGGBC


# Disable visualization in the authors' NGBC split routine; logic is unchanged.
ngbc_split.plot_dot = lambda data: None
ngbc_split.draw_ball = lambda balls: None

DATASETS = ["Zoo", "Iris", "Glass", "Ecoli", "Yeast", "Pageblocks", "Segment"]
OUTPUT = ROOT / "results" / "recent_paper_baselines.csv"
REPEATS = 3


def load_dataset(name: str):
    for label, openml_name, data_id in RARE_OPENML:
        if label == name:
            loaded = load_rare_openml(label, openml_name, data_id)
            if loaded is not None:
                return loaded
    loaded = load_uci_named(name)
    return loaded


def ngbc_predict(X: np.ndarray) -> np.ndarray:
    distance = squareform(pdist(X))
    tool = ngbc_split.NNSearch(distance)
    _, nn, rnn, dis_index = tool.natural_search()
    groups = tool.get_nb_group(nn, rnn, X)
    clusters = form_nb_cluster_by_neighbor(X, dis_index, groups)
    labels = np.full(len(X), -1, dtype=int)
    for cluster_id, cluster in enumerate(clusters.values()):
        for index in cluster.dataIndex:
            labels[int(index)] = cluster_id
    if np.any(labels < 0):
        next_label = int(labels.max() + 1)
        labels[labels < 0] = next_label
    return labels


def ekm_predict(X: np.ndarray, n_clusters: int) -> np.ndarray:
    return EKM(
        n_clusters=n_clusters,
        metric="euclidean",
        alpha="dvariance",
        scale=2.0,
        max_iter=500,
        tol=1e-3,
        n_init=30,
        init="plus",
        random_state=0,
        use_numba=False,
    ).fit_predict(X)


def timed(callable_):
    callable_()
    durations = []
    labels = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        labels = callable_()
        durations.append(time.perf_counter() - start)
    return labels, float(np.median(durations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    wanted = [s.strip() for s in args.datasets.split(",") if s.strip()]

    rows = []
    for name in wanted:
        loaded = load_dataset(name)
        if loaded is None:
            print(f"skip {name}: load failed", flush=True)
            continue
        X, y, label = loaded
        X = MinMaxScaler().fit_transform(np.asarray(X, dtype=float))
        y = np.asarray(y)
        n = len(X)
        k = int(PAPER_K.get(name, experiment_k(name, n)))
        print(f"\n{label} n={n} d={X.shape[1]} k={k}", flush=True)

        jobs = [
            ("ARP-GB", lambda: ARPGB(k=k, scale=False, size_gate=False).fit_predict(X)),
            ("MG-GBC", lambda: MGGBC(k=k, scale=False).fit_predict(X)),
            ("EKM", lambda: ekm_predict(X, len(np.unique(y)))),
        ]
        if n <= 2000:
            jobs.insert(2, ("NGBC", lambda: ngbc_predict(X)))
        else:
            print("NGBC: n > 2000, resource-ineligible", flush=True)

        for method, call in jobs:
            labels, runtime = timed(call)
            score = evaluate(y, labels)
            rows.append(
                {
                    "dataset": label,
                    "n": n,
                    "d": X.shape[1],
                    "k": k,
                    "method": method,
                    "runtime_median_s": runtime,
                    **score,
                }
            )
            print(
                f"{method:<8} ACC={score['ACC']:.3f} NMI={score['NMI']:.3f} "
                f"ARI={score['ARI']:.3f} R-rec={score['rare_recall']:.3f} "
                f"k_hat={int(score['n_clusters'])} t={runtime:.3f}s",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"\nwrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()