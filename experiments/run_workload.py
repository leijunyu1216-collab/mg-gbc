from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.baselines import DBSCANBaseline, DPCBaseline, KMeansBaseline
from src.core import minmax_scale
from src.datasets import (
    experiment_k,
    load_digits_rare,
    load_rare_openml,
    load_uci_named,
    synthetic_imbalance,
    synthetic_split_rare,
)
from src.metrics import evaluate
from src.mg_gbc import MGGBC


def _dbscan_auto(X: np.ndarray, min_samples: int = 5) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    Xs = minmax_scale(np.asarray(X, dtype=float))
    k = min(min_samples, max(2, len(Xs) - 1))
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
    eps = float(np.percentile(dist[:, -1], 90))
    eps = max(eps, 1e-6)
    return DBSCANBaseline(eps, min_samples).fit_predict(X)


def load_win_sets() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    jobs = []
    X, y, n = synthetic_imbalance()
    jobs.append((X, y, n, 12))
    X, y, n = synthetic_split_rare()
    jobs.append((X, y, n, 12))
    X, y, name = load_digits_rare()
    jobs.append((X, y, name, experiment_k(name, len(X))))
    for spec in [
        ("Yeast", "yeast", 181),
        ("Pageblocks", "page-blocks", 30),
        ("Segment", "segment", 36),
    ]:
        item = load_rare_openml(*spec)
        if item is not None:
            X, y, name = item
            jobs.append((X, y, name, experiment_k(name, len(X))))
    item = load_uci_named("ecoli")
    if item is not None:
        X, y, name = item
        jobs.append((X, y, name, 6))
    return jobs


def ablation() -> list[dict]:
    variants = [
        ("MG-GBC", dict(keep_rare=True, protect=True, fragments=True)),
        ("w/o keep", dict(keep_rare=False, protect=True, fragments=True)),
        ("w/o protect", dict(keep_rare=True, protect=False, fragments=True)),
        ("w/o fragment", dict(keep_rare=True, protect=True, fragments=False)),
    ]
    rows = []
    print("\n=== Ablation of MG-GBC rules ===")
    print(f"{'dataset':<22} {'variant':<14} {'ACC':>6} {'ARI':>6} {'Rrec':>6} {'k_hat':>5} rare")
    for X, y, name, k in load_win_sets():
        print()
        for tag, kw in variants:
            model = MGGBC(k=k, **kw)
            pred = model.fit_predict(X)
            s = evaluate(y, pred)
            rec = s["rare_recall"]
            recs = f"{rec:6.3f}" if rec == rec else "   nan"
            print(
                f"{name:<22} {tag:<14} {s['ACC']:6.3f} {s['ARI']:6.3f} {recs} "
                f"{int(s['n_clusters']):5d} {int(model.n_rare_cores_)}"
            )
            rows.append(
                {
                    "exp": "ablation",
                    "dataset": name,
                    "variant": tag,
                    "k": k,
                    "ACC": s["ACC"],
                    "ARI": s["ARI"],
                    "NMI": s["NMI"],
                    "rare_recall": rec,
                    "k_hat": int(s["n_clusters"]),
                    "rare": int(model.n_rare_cores_),
                    "frag": int(model.n_fragment_edges_),
                }
            )
    return rows


def vs_baselines() -> list[dict]:
    rows = []
    print("\n=== Baselines on rare-core sets ===")
    print(f"{'dataset':<22} {'method':<10} {'ACC':>6} {'NMI':>6} {'ARI':>6} {'Rrec':>6} {'k_hat':>5}")
    for X, y, name, k in load_win_sets():
        n_true = int(len(np.unique(y)))
        jobs = [
            ("K-means", KMeansBaseline(n_true)),
            ("DPC", DPCBaseline(0.02, n_true)),
            ("ARP-GB", ARPGB(k=k)),
            ("MG-GBC", MGGBC(k=k)),
        ]
        print()
        for tag, model in jobs:
            pred = model.fit_predict(X)
            s = evaluate(y, pred)
            rec = s["rare_recall"]
            recs = f"{rec:6.3f}" if rec == rec else "   nan"
            print(
                f"{name:<22} {tag:<10} {s['ACC']:6.3f} {s['NMI']:6.3f} {s['ARI']:6.3f} "
                f"{recs} {int(s['n_clusters']):5d}"
            )
            rows.append(
                {
                    "exp": "baseline",
                    "dataset": name,
                    "variant": tag,
                    "k": k,
                    "ACC": s["ACC"],
                    "ARI": s["ARI"],
                    "NMI": s["NMI"],
                    "rare_recall": rec,
                    "k_hat": int(s["n_clusters"]),
                    "rare": int(getattr(model, "n_rare_cores_", 0) or 0),
                    "frag": int(getattr(model, "n_fragment_edges_", 0) or 0),
                }
            )
        pred = _dbscan_auto(X)
        s = evaluate(y, pred)
        rec = s["rare_recall"]
        recs = f"{rec:6.3f}" if rec == rec else "   nan"
        print(
            f"{name:<22} {'DBSCAN':<10} {s['ACC']:6.3f} {s['NMI']:6.3f} {s['ARI']:6.3f} "
            f"{recs} {int(s['n_clusters']):5d}"
        )
        rows.append(
            {
                "exp": "baseline",
                "dataset": name,
                "variant": "DBSCAN",
                "k": k,
                "ACC": s["ACC"],
                "ARI": s["ARI"],
                "NMI": s["NMI"],
                "rare_recall": rec,
                "k_hat": int(s["n_clusters"]),
                "rare": 0,
                "frag": 0,
            }
        )
    return rows


def multi_seed() -> list[dict]:
    rows = []
    print("\n=== Multi-seed (10 draws) ===")
    print(f"{'dataset':<22} {'method':<8} {'ACC':>6} {'ARI':>6} {'Rrec':>6}  (mean±std)")
    specs = [
        ("Synthetic-Imbalance", synthetic_imbalance, 12),
        ("Synthetic-SplitRare", synthetic_split_rare, 12),
        ("Digits-Rare", load_digits_rare, 10),
    ]
    for name, loader, k in specs:
        for tag, factory in (("ARP-GB", lambda kk: ARPGB(k=kk)), ("MG-GBC", lambda kk: MGGBC(k=kk))):
            accs, aris, recs = [], [], []
            for seed in range(10):
                X, y, _ = loader(seed=seed)
                pred = factory(k).fit_predict(X)
                s = evaluate(y, pred)
                accs.append(s["ACC"])
                aris.append(s["ARI"])
                recs.append(s["rare_recall"] if s["rare_recall"] == s["rare_recall"] else 0.0)
            print(
                f"{name:<22} {tag:<8} {np.mean(accs):6.3f}±{np.std(accs):.3f} "
                f"{np.mean(aris):6.3f}±{np.std(aris):.3f} "
                f"{np.mean(recs):6.3f}±{np.std(recs):.3f}"
            )
            rows.append(
                {
                    "exp": "seed",
                    "dataset": name,
                    "variant": tag,
                    "k": k,
                    "ACC": float(np.mean(accs)),
                    "ARI": float(np.mean(aris)),
                    "NMI": float("nan"),
                    "rare_recall": float(np.mean(recs)),
                    "k_hat": float("nan"),
                    "rare": float("nan"),
                    "frag": float(np.std(aris)),
                }
            )
    return rows


def runtime() -> list[dict]:
    rows = []
    print("\n=== Runtime (s, mean of 3) ===")
    print(f"{'dataset':<22} {'n':>5} {'ARP':>7} {'MG':>7}")
    for X, y, name, k in load_win_sets():
        times = {"ARP-GB": [], "MG-GBC": []}
        for _ in range(3):
            t0 = time.perf_counter()
            ARPGB(k=k).fit_predict(X)
            times["ARP-GB"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            MGGBC(k=k).fit_predict(X)
            times["MG-GBC"].append(time.perf_counter() - t0)
        a = float(np.mean(times["ARP-GB"]))
        m = float(np.mean(times["MG-GBC"]))
        print(f"{name:<22} {len(X):5d} {a:7.3f} {m:7.3f}")
        rows.append(
            {
                "exp": "time",
                "dataset": name,
                "variant": "both",
                "k": k,
                "ACC": a,
                "ARI": m,
                "NMI": float(len(X)),
                "rare_recall": float("nan"),
                "k_hat": 0,
                "rare": 0,
                "frag": 0,
            }
        )
    return rows


def main() -> None:
    rows = []
    rows.extend(ablation())
    rows.extend(vs_baselines())
    rows.extend(multi_seed())
    rows.extend(runtime())
    out = ROOT / "results" / "workload_extra.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
