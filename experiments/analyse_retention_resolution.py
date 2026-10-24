"""Resolution of the retention predicate as a function of rare-class size.

The CIC-IDS2018 diagnosis leaves one mechanism standing: a pure rare ball does
form and does sit below the size cut, but it is not selected, because its
density z-score is negative.  The density is a k-nearest-neighbour estimate
with k = 10, so a class of m points can only look dense when m is comparable
to k: below that, every neighbour the estimate averages over is a background
point and the class inherits the background density.

This script puts both real sources on the same axis.  For every (block, true
class below sqrt(n)) pair it records m and the density z-score of the purest
ball containing that class, then bins by m.  If the predicate has a resolution
floor near m ~ k it appears as the density z-score crossing the threshold
between the lowest bins.

The UNSW-NB15 blocks are the frozen ones.  The CIC-IDS2018 rows are read from
results/cicids2018_mechanism_diagnosis.json, computed by
experiments/diagnose_cicids2018_mechanism.py against the same blocks the
pre-registered run evaluated.  Nothing is re-sampled here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_unsw_nb15_temporal_compare as R  # noqa: E402
from src.core import (  # noqa: E402
    generate_granular_balls,
    knn_density,
    select_representatives,
)
from src.mg_gbc import _mad, ball_mean_density, rare_core_mask  # noqa: E402

PROTOCOL = ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json"
CIC_ROWS = ROOT / "results" / "cicids2018_mechanism_diagnosis.json"
OUT = ROOT / "results" / "retention_resolution_by_class_size.json"
TABLE = ROOT / "results" / "retention_resolution_by_class_size.csv"

BINS = [3, 6, 10, 15, 25, 50, np.inf]


def unsw_pairs() -> tuple[list[dict], dict]:
    protocol, _ = R.load_protocol(PROTOCOL)
    starts = protocol["sampling"]["selected_block_starts_unix"]
    source_rows = R.load_selected_rows(protocol)
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    k = int(protocol["models"]["k"])

    rows, balls_total, masked_total = [], 0, 0
    for block_id, start in enumerate(starts, start=1):
        sample = R.sample_block(source_rows, int(start), n, base_seed + block_id)
        X = R.prepare_matrix(sample)
        lab, classes, counts = R.normalized_labels(sample)
        rare = [i for i, c in enumerate(counts) if 3 <= c < np.sqrt(len(X))]
        if not rare:
            continue
        rho, _ = knn_density(X, k)
        balls = generate_granular_balls(X, select_representatives(X, rho), dissolve_small=False)
        gamma = ball_mean_density(rho, balls)
        centers = np.stack([b.center for b in balls])
        dens_z = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
        masked = rare_core_mask(np.array([b.size for b in balls], float), gamma, centers,
                                len(X), 0.5, 0.0, 3)
        balls_total += len(balls)
        masked_total += int(masked.sum())
        for cls in rare:
            member = lab == cls
            purity = np.array([float(np.mean(lab[b.indices] == cls)) for b in balls])
            best = int(np.argmax(purity))
            rows.append({
                "source": "UNSW-NB15",
                "block": block_id,
                "family": str(classes[cls]),
                "rare_points": int(counts[cls]),
                "purest_ball_size": int(balls[best].size),
                "purest_ball_purity": float(purity[best]),
                "purest_ball_density_z": float(dens_z[best]),
                "purest_ball_masked": bool(masked[best]),
                "purest_ball_below_cut": bool(balls[best].size < np.sqrt(len(X))),
                "balls_in_block": int(len(balls)),
                "masked_balls_in_block": int(masked.sum()),
            })
        print(f"UNSW block {block_id:02d} done", flush=True)
    return rows, {"balls": balls_total, "masked": masked_total}


def cic_pairs() -> list[dict]:
    payload = json.loads(CIC_ROWS.read_text(encoding="utf-8"))
    rows = []
    for row in payload["rows"]:
        rows.append({
            "source": "CIC-IDS2018",
            "block": row["start_utc"],
            "family": row["family"],
            "rare_points": row["rare_points"],
            "purest_ball_size": row["purest_ball_size"],
            "purest_ball_purity": row["purest_ball_purity"],
            "purest_ball_density_z": row["purest_ball_density_z"],
            "purest_ball_masked": row["purest_ball_masked"],
            "purest_ball_below_cut": row["purest_ball_below_cut"],
            "balls_in_block": np.nan,
            "masked_balls_in_block": np.nan,
        })
    return rows


def main() -> None:
    unsw, ball_stats = unsw_pairs()
    rows = unsw + cic_pairs()
    frame = pd.DataFrame(rows)
    frame["size_bin"] = pd.cut(frame["rare_points"], bins=BINS, right=False,
                               labels=["3-5", "6-9", "10-14", "15-24", "25-49", "50+"])
    frame.to_csv(TABLE, index=False, float_format="%.12g")

    grouped = frame.groupby(["source", "size_bin"], observed=True).agg(
        pairs=("rare_points", "size"),
        median_rare_points=("rare_points", "median"),
        median_density_z=("purest_ball_density_z", "median"),
        share_masked=("purest_ball_masked", "mean"),
        median_purity=("purest_ball_purity", "median"),
        share_below_cut=("purest_ball_below_cut", "mean"),
    ).reset_index()

    pooled = frame.groupby("size_bin", observed=True).agg(
        pairs=("rare_points", "size"),
        median_density_z=("purest_ball_density_z", "median"),
        share_masked=("purest_ball_masked", "mean"),
    ).reset_index()

    out = {
        "k": 10,
        "threshold_density_z": 0.5,
        "unsw_predicate": ball_stats,
        "unsw_pairs": int(len(unsw)),
        "cic_pairs": int(len(frame) - len(unsw)),
        "by_source_and_size": grouped.to_dict(orient="records"),
        "pooled_by_size": pooled.to_dict(orient="records"),
        "reading": (
            "share_masked is the fraction of (block, class) pairs whose purest ball is "
            "selected by the retention predicate. The predicate should be unselective "
            "below m ~ k because the k-nearest-neighbour density of a point in a class "
            "smaller than k is dominated by background neighbours."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")

    print("\nby source and size bin")
    print(grouped.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\npooled by size bin")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}\nwrote {TABLE}")


if __name__ == "__main__":
    main()