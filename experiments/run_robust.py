from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.datasets import load_digits_rare, synthetic_split_rare
from src.metrics import evaluate
from src.mg_gbc import MGGBC

# Matched graph policy: MG-GBC runs with the size-ratio guard disabled, so the
# reference is run the same way for the robustness comparison.
ARP_KW = dict(size_gate=False)


def add_gaussian(X: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    span = np.ptp(X, axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    return X + rng.normal(0.0, sigma, size=X.shape) * span


def add_outliers(X: np.ndarray, y: np.ndarray, n_out: int, seed: int):
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    extra = rng.uniform(lo, hi, size=(n_out, X.shape[1]))
    y_out = np.concatenate([y, np.full(n_out, -1)])
    return np.vstack([X, extra]), y_out


def main() -> None:
    rows = []
    print("=== Gaussian noise (feature-span units) ===")
    print(f"{'set':<22} {'sigma':>5} {'method':<8} {'ARI':>6} {'Rrec':>6} rare")
    for loader, name, k, n_out in (
        (load_digits_rare, "Digits-Rare", 10, 15),
        (synthetic_split_rare, "Synthetic-SplitRare", 12, 30),
    ):
        X0, y0, _ = loader(seed=0)
        for sigma in (0.0, 0.02, 0.05, 0.10, 0.20):
            for seed in range(5):
                X = add_gaussian(X0, sigma, seed)
                for tag, factory in (("ARP-GB", ARPGB), ("MG-GBC", MGGBC)):
                    model = factory(k=k, **ARP_KW) if tag == "ARP-GB" else factory(k=k)
                    pred = model.fit_predict(X)
                    s = evaluate(y0, pred)
                    rec = s["rare_recall"]
                    rows.append(
                        {
                            "kind": "noise",
                            "dataset": name,
                            "sigma": sigma,
                            "seed": seed,
                            "method": tag,
                            "ARI": s["ARI"],
                            "rare_recall": rec if rec == rec else 0.0,
                            "rare": int(getattr(model, "n_rare_cores_", 0) or 0),
                        }
                    )
            sub = [r for r in rows if r["dataset"] == name and r["sigma"] == sigma]
            for tag in ("ARP-GB", "MG-GBC"):
                ss = [r for r in sub if r["method"] == tag]
                print(
                    f"{name:<22} {sigma:5.2f} {tag:<8} "
                    f"{np.mean([r['ARI'] for r in ss]):6.3f} "
                    f"{np.mean([r['rare_recall'] for r in ss]):6.3f} "
                    f"{np.mean([r['rare'] for r in ss]):4.1f}"
                )

    print("\n=== Uniform outliers (y=-1 ignored in metrics via y0) ===")
    print(f"{'set':<22} {'nout':>4} {'method':<8} {'ARI':>6} {'Rrec':>6} rare")
    for loader, name, k, n_out in (
        (load_digits_rare, "Digits-Rare", 10, 15),
        (synthetic_split_rare, "Synthetic-SplitRare", 12, 30),
    ):
        X0, y0, _ = loader(seed=0)
        for seed in range(5):
            X, y_full = add_outliers(X0, y0, n_out, seed)
            for tag, factory in (("ARP-GB", ARPGB), ("MG-GBC", MGGBC)):
                model = factory(k=k, **ARP_KW) if tag == "ARP-GB" else factory(k=k)
                pred = model.fit_predict(X)
                s = evaluate(y0, pred[: len(y0)])
                rec = s["rare_recall"]
                rows.append(
                    {
                        "kind": "outlier",
                        "dataset": name,
                        "sigma": n_out,
                        "seed": seed,
                        "method": tag,
                        "ARI": s["ARI"],
                        "rare_recall": rec if rec == rec else 0.0,
                        "rare": int(getattr(model, "n_rare_cores_", 0) or 0),
                    }
                )
        sub = [r for r in rows if r["kind"] == "outlier" and r["dataset"] == name]
        for tag in ("ARP-GB", "MG-GBC"):
            ss = [r for r in sub if r["method"] == tag]
            print(
                f"{name:<22} {n_out:4d} {tag:<8} "
                f"{np.mean([r['ARI'] for r in ss]):6.3f} "
                f"{np.mean([r['rare_recall'] for r in ss]):6.3f} "
                f"{np.mean([r['rare'] for r in ss]):4.1f}"
            )

    out = ROOT / "results" / "robustness_matched.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
