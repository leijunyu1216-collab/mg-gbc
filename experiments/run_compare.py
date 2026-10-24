from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.baselines import (
    DBSCANBaseline,
    DPCBaseline,
    KMeansBaseline,
    RNN_DBSCANBaseline,
    SNN_DPCBaseline,
)
from src.datasets import (
    PAPER_ARPGB,
    PAPER_BASELINE_PARAMS,
    PAPER_K,
    load_uci_named,
)
from src.metrics import evaluate
from src.mg_gbc import MGGBC


def _run(model, X):
    labels = model.fit_predict(X)
    return np.asarray(labels)


def compare_dataset(X, y, name: str, include_heavy: bool = True) -> None:
    params = PAPER_BASELINE_PARAMS[name]
    k_arp = PAPER_K[name]
    paper = PAPER_ARPGB.get(name)
    jobs = [
        ("K-means", KMeansBaseline(params["kmeans"])),
        ("DBSCAN", DBSCANBaseline(*params["dbscan"])),
        ("RNN", RNN_DBSCANBaseline(params["rnn_k"])),
        ("ARP-GB", ARPGB(k=k_arp)),
        ("MG-GBC", MGGBC(k=k_arp)),
    ]
    n = len(X)
    if include_heavy and n <= 4500:
        jobs.insert(2, ("DPC", DPCBaseline(params["dpc"][0], params["dpc"][1])))
        jobs.insert(3, ("SNN-DPC", SNN_DPCBaseline(params["snn_k"], params["kmeans"])))

    print(f"\n{name}  n={n} d={X.shape[1]}  (ARP-GB k={k_arp})")
    print(f"{'method':<10} {'ACC':>6} {'NMI':>6} {'ARI':>6} {'k_hat':>6}")
    print("-" * 40)
    for tag, model in jobs:
        pred = _run(model, X)
        s = evaluate(y, pred)
        print(
            f"{tag:<10} {s['ACC']:.3f} {s['NMI']:.3f} {s['ARI']:.3f} "
            f"{int(s['n_clusters']):6d}"
        )
    if paper:
        print(
            f"{'paper ARP':<10} {paper['ACC']:.3f} {paper['NMI']:.3f} {paper['ARI']:.3f} "
            f"{paper['k_hat']:6d}"
        )


def main():
    names = ["Zoo", "Iris", "Glass", "Ecoli"]
    shuttle = "--shuttle" in sys.argv
    if shuttle:
        names.append("Shuttle")
    print("IJMLC-style comparison (minmax, ACC/NMI/ARI)")
    for name in names:
        loaded = load_uci_named(name)
        if loaded is None:
            print(f"\n{name}: load failed")
            continue
        X, y, n = loaded
        compare_dataset(X, y, n, include_heavy=True)


if __name__ == "__main__":
    main()
