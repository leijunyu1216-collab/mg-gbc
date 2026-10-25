"""One-at-a-time sensitivity of the registered constants on the paper suite.

Every row changes exactly one constant of MGGBC; the remaining five keep
their defaults.  The comparison is against the default configuration on the
same data, so a flat row means the reported behaviour does not depend on
that constant over the stated range.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.check_guard_paper_regression import paper_datasets  # noqa: E402
from src.arp_gb import ARPGB  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

OUT = ROOT / "results"

# name -> (parameter, levels, default)
SWEEPS = {
    "s_min": ("rare_min_size", [3, 5, 10], 3),
    "tau_gamma": ("rare_density_z", [0.0, 0.5, 1.0], 0.5),
    "tau_iso": ("rare_isolation_z", [-10.0, 0.0, 0.5], 0.0),
    "delta": ("fragment_density_contrast", [0.15, 0.25, 0.35], 0.25),
    "eta": ("fragment_near_factor", [1.25, 1.5, 2.0], 1.5),
    "beta": ("fragment_closeness_factor", [0.4, 0.5, 0.6], 0.5),
}
BETA_PAIRED = "fragment_corridor_factor"


def fit(name, X, y, k, **kwargs) -> dict:
    model = MGGBC(k=k, **kwargs)
    met = evaluate(y, model.fit_predict(X))
    return {
        "dataset": name,
        "ACC": met["ACC"],
        "NMI": met["NMI"],
        "ARI": met["ARI"],
        "rare_recall": met["rare_recall"],
        "n_clusters": met["n_clusters"],
        "n_rare_cores": int(getattr(model, "n_rare_cores_", 0) or 0),
    }


def main() -> None:
    data = [(n, np.asarray(X, dtype=float), np.asarray(y), k) for n, X, y, k in paper_datasets()]
    rows = []
    for name, X, y, k in data:
        arp = evaluate(y, ARPGB(k=k).fit_predict(X))
        rows.append({
            "sweep": "baseline", "level": "ARP-matched", "dataset": name,
            "ACC": arp["ACC"], "NMI": arp["NMI"], "ARI": arp["ARI"],
            "rare_recall": arp["rare_recall"], "n_clusters": arp["n_clusters"],
            "n_rare_cores": 0,
        })
        rows.append({"sweep": "baseline", "level": "default",
                     **fit(name, X, y, k, scale=True)})
        for sweep, (param, levels, default) in SWEEPS.items():
            for level in levels:
                if float(level) == float(default):
                    continue
                kwargs = {param: level}
                if sweep == "beta":
                    kwargs[BETA_PAIRED] = level
                rows.append({"sweep": sweep, "level": str(level),
                             **fit(name, X, y, k, scale=True, **kwargs)})
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity_paper_v2.csv", index=False, float_format="%.12g")

    base = df.loc[df["sweep"].eq("baseline") & df["level"].eq("default")].set_index("dataset")
    summary = []
    for (sweep, level), grp in df.loc[~df["sweep"].eq("baseline")].groupby(["sweep", "level"], sort=False):
        g = grp.set_index("dataset")
        d_ari = (g["ARI"] - base["ARI"]).to_numpy(float)
        d_rare = (g["rare_recall"] - base["rare_recall"]).to_numpy(float)
        finite = np.isfinite(d_rare)
        summary.append({
            "sweep": sweep,
            "level": level,
            "median_dARI_vs_default": float(np.median(d_ari)),
            "worst_dARI_vs_default": float(d_ari.min()),
            "median_dRare_vs_default": float(np.median(d_rare[finite])) if finite.any() else None,
            "worst_dRare_vs_default": float(d_rare[finite].min()) if finite.any() else None,
            "datasets_changed": int((np.abs(d_ari) > 1e-12).sum()),
            "datasets_losing_recall": int((d_rare < -1e-12).sum()),
            "datasets_gaining_recall": int((d_rare > 1e-12).sum()),
            "total_cores": int(g["n_rare_cores"].sum()),
        })
    (OUT / "sensitivity_paper_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()