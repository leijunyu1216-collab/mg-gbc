"""Wall-clock cost of the inherited stages and the added layer.

Five repetitions per method; the reported value is the median.  Both
methods receive identical inputs and parameters.  ARP-GB uses the matched
graph policy, so the difference isolates the added ball-level layer.
"""

from __future__ import annotations

import os
import sys
import time
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
from src.mg_gbc import MGGBC  # noqa: E402

OUT = ROOT / "results"
DATASETS = ["Digits-Rare", "Ecoli", "Yeast", "Segment", "Pageblocks"]
REPS = 5


def main() -> None:
    data = {n: (np.asarray(X, float), k) for n, X, y, k in paper_datasets()}
    rows = []
    for name in DATASETS:
        X, k = data[name]
        for method, factory in (
            ("ARP-GB", lambda: ARPGB(k=k, size_gate=False)),
            ("MG-GBC", lambda: MGGBC(k=k)),
        ):
            times, balls = [], []
            for _ in range(REPS):
                model = factory()
                t0 = time.perf_counter()
                model.fit_predict(X)
                times.append(time.perf_counter() - t0)
                balls.append(int(model.n_balls_))
            rows.append({
                "dataset": name, "n": int(X.shape[0]), "d": int(X.shape[1]),
                "k": k, "method": method, "median_s": float(np.median(times)),
                "iqr_s": float(np.percentile(times, 75) - np.percentile(times, 25)),
                "n_balls": int(np.median(balls)),
            })
        print(f"{name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runtime_v2.csv", index=False, float_format="%.6g")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()