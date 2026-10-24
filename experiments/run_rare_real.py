from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.datasets import (
    experiment_k,
    load_covtype_rare,
    load_digits_rare,
    load_olivetti_pca,
    load_rare_openml,
    load_uci_named,
    synthetic_imbalance,
    synthetic_split_rare,
)
from src.metrics import evaluate
from src.mg_gbc import MGGBC


# Pre-specified: synthetics, Ecoli (Table 6), then sets where a small dense
# ball can exist. k = Table 6 or clip(sqrt(n), 5, 10). Shuttle is omitted.
EVAL_LOADERS = [
    ("Synthetic-Imbalance", lambda: synthetic_imbalance(), 12),
    ("Synthetic-SplitRare", lambda: synthetic_split_rare(), 12),
    ("Digits-Rare", load_digits_rare, None),
    ("Covertype-Rare", load_covtype_rare, None),
    ("Olivetti-PCA", load_olivetti_pca, None),
    ("Yeast", lambda: load_rare_openml("Yeast", "yeast", 181), None),
    ("Pageblocks", lambda: load_rare_openml("Pageblocks", "page-blocks", 30), None),
    ("Segment", lambda: load_rare_openml("Segment", "segment", 36), None),
    ("Lymphography", lambda: load_rare_openml("Lymphography", "lymph", 10), None),
    ("Dermatology", lambda: load_rare_openml("Dermatology", "dermatology", 35), None),
    ("Ecoli", lambda: load_uci_named("ecoli"), 6),
]


def _load(spec):
    name, fn, k_fixed = spec
    item = fn()
    if item is None:
        return None
    if len(item) == 3:
        X, y, loaded_name = item
    else:
        X, y, loaded_name = item[0], item[1], item[2]
    n = len(X)
    k = int(k_fixed) if k_fixed is not None else experiment_k(loaded_name, n)
    return np.asarray(X, dtype=float), np.asarray(y), loaded_name, k


def run() -> list[dict]:
    rows = []
    print("MG-GBC on real / semi-real rare-core candidates")
    print("k = Table 6 if present, else clip(sqrt(n), 5, 10). Shuttle omitted.\n")
    print(
        f"{'dataset':<22} {'n':>5} {'k':>4} {'method':<8} "
        f"{'ACC':>6} {'NMI':>6} {'ARI':>6} {'Rrec':>6} {'k_hat':>5} rare"
    )
    print("-" * 92)
    for spec in EVAL_LOADERS:
        loaded = _load(spec)
        if loaded is None:
            print(f"{spec[0]:<22} load failed")
            continue
        X, y, name, k = loaded
        k = int(np.clip(k, 2, max(2, len(X) - 1)))
        for tag, model in (("ARP-GB", ARPGB(k=k)), ("MG-GBC", MGGBC(k=k))):
            pred = model.fit_predict(X)
            s = evaluate(y, pred)
            rare = getattr(model, "n_rare_cores_", None)
            frag = getattr(model, "n_fragment_edges_", None)
            rec = s["rare_recall"]
            rec_s = f"{rec:6.3f}" if rec == rec else "   nan"
            extra = "" if rare is None else f"{int(rare):4d}"
            if frag:
                extra += f" f={frag}"
            print(
                f"{name:<22} {len(X):5d} {k:4d} {tag:<8} "
                f"{s['ACC']:6.3f} {s['NMI']:6.3f} {s['ARI']:6.3f} {rec_s} "
                f"{int(s['n_clusters']):5d} {extra}"
            )
            rows.append(
                {
                    "dataset": name,
                    "n": len(X),
                    "d": X.shape[1],
                    "k": k,
                    "method": tag,
                    "ACC": s["ACC"],
                    "NMI": s["NMI"],
                    "ARI": s["ARI"],
                    "rare_recall": s["rare_recall"],
                    "k_hat": int(s["n_clusters"]),
                    "k_true": int(s["n_true"]),
                    "rare": int(rare) if rare is not None else 0,
                    "frag": int(frag or 0),
                }
            )
        print()
    return rows


def save_pca_figure(path: Path, X, y, k: int, title: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    Xs = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X, dtype=float))
    pred_a = ARPGB(k=k).fit_predict(X)
    pred_m = MGGBC(k=k).fit_predict(X)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, lab, head in zip(
        axes,
        [y, pred_a, pred_m],
        ["Ground truth", "ARP-GB", "MG-GBC"],
    ):
        ax.scatter(Xs[:, 0], Xs[:, 1], c=lab, s=8, cmap="tab10", linewidths=0)
        ax.set_title(head)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    rows = run()
    out = ROOT / "results" / "rare_real_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    for loader, fname, title in (
        (load_digits_rare, "digits_rare.png", "Digits-Rare (PCA-2 of 64-D pixels)"),
        (load_covtype_rare, "covertype_rare.png", "Covertype-Rare (PCA-2 of 54-D)"),
    ):
        X, y, name = loader()
        k = experiment_k(name, len(X))
        save_pca_figure(ROOT / "results" / fname, X, y, k, title)


if __name__ == "__main__":
    main()
