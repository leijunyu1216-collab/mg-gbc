"""Does the UNSW-motivated rare-core guard damage the paper's headline results?

Any repair proposed from block 12 has to be checked against the datasets the
main claim rests on, especially Synthetic-SplitRare, Digits-Rare and Ecoli,
where MG-GBC's advantage comes from exactly the rare cores the guard prunes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_DATA", str(Path(__file__).resolve().parents[1] / "tmp"))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mg_gbc_guarded import MGGBCGuarded, MGGBCOverlay  # noqa: E402
from src.arp_gb import ARPGB  # noqa: E402
from src.datasets import (  # noqa: E402
    PAPER_K,
    experiment_k,
    load_digits_rare,
    load_rare_openml,
    load_uci_named,
    synthetic_imbalance,
    synthetic_split_rare,
)
from src.metrics import evaluate  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

OUT = ROOT / "results"


def paper_datasets():
    out = []
    for name in ("Zoo", "Iris", "Glass", "Ecoli"):
        loaded = load_uci_named(name)
        if loaded:
            X, y, canonical = loaded
            out.append((canonical, X, y, PAPER_K[canonical]))
    for loader in (synthetic_imbalance, synthetic_split_rare):
        try:
            X, y, name = loader(seed=0)
        except TypeError:
            X, y, name = loader()
        out.append((name, X, y, 12))
    try:
        X, y, name = load_digits_rare(seed=0)
        out.append((name, X, y, experiment_k(name, len(X))))
    except Exception as exc:
        print(f"digits-rare failed: {exc}")
    for label, openml_name, data_id in [("Yeast", "yeast", 181), ("Pageblocks", "page-blocks", 30), ("Segment", "segment", 36)]:
        try:
            loaded = load_rare_openml(label, openml_name, data_id)
        except Exception as exc:
            print(f"{label} failed: {exc}")
            continue
        if loaded:
            X, y, canonical = loaded
            out.append((canonical, X, y, experiment_k(canonical, len(X))))
    return out


def main() -> None:
    variants = {
        "ARP-GB": lambda k: ARPGB(k=k),
        "MG-GBC": lambda k: MGGBC(k=k),
        "MG z=1.5": lambda k: MGGBC(k=k, rare_density_z=1.5),
        "MG z=2.0": lambda k: MGGBC(k=k, rare_density_z=2.0),
        "MG guard tau=0.10": lambda k: MGGBCGuarded(k=k, contrast_tau=0.10),
        "MG guard tau=0.15": lambda k: MGGBCGuarded(k=k, contrast_tau=0.15),
        "MG guard tau=0.20": lambda k: MGGBCGuarded(k=k, contrast_tau=0.20),
        "MG guard tau=0.25": lambda k: MGGBCGuarded(k=k, contrast_tau=0.25),
        "MG overlay": lambda k: MGGBCOverlay(k=k),
    }
    rows = []
    for name, X, y, k in paper_datasets():
        for vname, factory in variants.items():
            model = factory(k)
            try:
                pred = model.fit_predict(np.asarray(X, dtype=float))
                met = evaluate(np.asarray(y), pred)
                met["n_rare_cores"] = int(getattr(model, "n_rare_cores_", 0) or 0)
            except Exception as exc:
                met = {"ACC": np.nan, "NMI": np.nan, "ARI": np.nan, "error": str(exc)}
            rows.append({"dataset": name, "n": len(X), "k": k, "variant": vname, **met})
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "guard_paper_regression.csv", index=False)

    pivot = df.pivot_table(index="dataset", columns="variant", values="ARI")
    order = [v for v in variants if v in pivot.columns]
    print("\n=== ARI by dataset ===")
    print(pivot[order].to_string())

    print("\n=== ARI delta vs MG-GBC (negative = the guard costs the paper a win) ===")
    delta = pivot[order].subtract(pivot["MG-GBC"], axis=0).drop(columns=["MG-GBC"])
    print(delta.to_string())

    print("\n=== ACC by dataset ===")
    print(df.pivot_table(index="dataset", columns="variant", values="ACC")[order].to_string())

    (OUT / "guard_paper_regression.json").write_text(
        json.dumps(df.to_dict("records"), indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'guard_paper_regression.csv'}")


if __name__ == "__main__":
    main()
