"""The 315-cell joint grid evaluated through the shipped pipeline.

Produces the per-cell table behind Fig. (joint).  Every cell of the
delta x eta x beta grid is evaluated on the 31 units the paper scores: the
20 UNSW-NB15 v2 replication blocks (43 features), the ten development sets
and Covertype-Rare.  The fragment rule is the one defined in Definition
(signed edit), i.e. with the admissibility condition in force; the condition
can be left out with ``run_tail(..., guard=False)``, which is the ablation of
that condition.

The untouched v1 holdout is never evaluated at a non-default cell and does
not appear in the output.

Output: results/coupling_grid_guard_raw.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_coupling_grid import (  # noqa: E402
    BETAS,
    D0,
    DELTAS,
    E0,
    ETAS,
    B0,
    OUT,
    build_state,
    label_hash,
    load_units,
    run_tail,
)
from src.metrics import evaluate  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

CHECK_CELLS = ((D0, E0, B0), (0.45, 2.50, 0.70))


def main() -> None:
    units = load_units()
    t0 = time.perf_counter()
    print(f"{len(units)} units", flush=True)

    rows: list[dict] = []
    for ui, unit in enumerate(units, 1):
        X = np.asarray(unit.X, dtype=float)
        state = build_state(X, unit.k, 0.5)
        base, _ = run_tail(state, D0, E0, B0, guard=True)
        h0 = label_hash(base)
        mb = evaluate(unit.y, base)

        for cell in CHECK_CELLS:
            lab, _ = run_tail(state, *cell, guard=True)
            ref = MGGBC(k=unit.k,
                        fragment_density_contrast=cell[0],
                        fragment_near_factor=cell[1],
                        fragment_closeness_factor=cell[2],
                        fragment_corridor_factor=cell[2]).fit_predict(X)
            if label_hash(ref) != label_hash(lab):
                raise SystemExit(
                    f"cached-state path disagrees with MGGBC on {unit.name} "
                    f"at {cell}")

        for d in DELTAS:
            for e in ETAS:
                for b in BETAS:
                    lab, ne = run_tail(state, d, e, b, guard=True)
                    met = evaluate(unit.y, lab)
                    rows.append(dict(
                        cohort=unit.cohort, unit=unit.name, delta=d, eta=e,
                        beta=b, edges=ne, same=label_hash(lab) == h0,
                        dARI=met["ARI"] - mb["ARI"],
                        dR=met["rare_recall"] - mb["rare_recall"],
                        k=int(met["n_clusters"]), k0=int(mb["n_clusters"])))
        print(f"[{ui}/{len(units)}] {unit.cohort}/{unit.name} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coupling_grid_guard_raw.csv", index=False)
    print(f"{len(df)} (cell, unit) pairs -> "
          f"results/coupling_grid_guard_raw.csv "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()