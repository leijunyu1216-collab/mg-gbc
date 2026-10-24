"""Sweep the host pipeline's single design knob: the size cut.

ARP-GB dissolves every ball smaller than c*sqrt(n) with c=1, and that cut is
the only place in the pipeline where the size of a group is compared with a
global scale.  If the truncation were a tuning problem, some c would recover
the rare class without damaging the majority.  This script instantiates the
host at nine values of c, on the ten data sets the paper scores, and records
what each value buys and what it costs.  MG-GBC is evaluated at the published
cut only: its claim is about what happens to a ball below the line, not about
where the line is drawn.

Output: results/host_sizecut_sweep.csv
"""

from __future__ import annotations

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

FACTORS = (0.25, 0.50, 0.625, 0.75, 0.875, 1.00, 1.25, 1.50, 2.00)
DEFAULT = 1.00
OUT = ROOT / "results"


def main() -> None:
    rows = []
    for name, X, y, k in paper_datasets():
        X = np.asarray(X, dtype=float)
        for c in FACTORS:
            model = ARPGB(k=k, size_gate=False, size_cut_factor=c)
            met = evaluate(y, model.fit_predict(X))
            rows.append(dict(dataset=name, method="ARP-GB", cut=c,
                             ACC=met["ACC"], ARI=met["ARI"],
                             Rrec=met["rare_recall"], k=int(met["n_clusters"]),
                             balls=int(model.n_balls_)))
        model = MGGBC(k=k)  # published cut, correction only
        met = evaluate(y, model.fit_predict(X))
        rows.append(dict(dataset=name, method="MG-GBC", cut=DEFAULT,
                         ACC=met["ACC"], ARI=met["ARI"],
                         Rrec=met["rare_recall"], k=int(met["n_clusters"]),
                         balls=int(model.n_balls_)))
        print(f"{name}: n={len(X)} k={k} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "host_sizecut_sweep.csv", index=False)
    print(f"{len(df)} rows -> results/host_sizecut_sweep.csv")


if __name__ == "__main__":
    main()