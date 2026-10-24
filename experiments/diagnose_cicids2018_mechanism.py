"""Why the CIC-IDS2018 blocked comparison has no rare-recall signal.

The pre-registered experiment (experiments/cicids2018_temporal_protocol_v1.json)
returned ties on almost every window that carries a class below sqrt(n).  This
script tests the three candidate explanations against the frozen blocks, using
only the block construction and the labels that the frozen run already opened:

  dispersion   the rare class is split across several balls, so no single ball
               holds it and the size cut cannot be what removes it
  absorption   the ball that holds the rare points is large, so the class is
               swallowed before any size cut applies
  resolution   the smallest pure rare ball scores below the retention threshold
               on the density z-score, whose resolution is set by k

For every window of the replication day it records, per rare class, the number
of balls holding it, the largest share in one ball, the size of that ball, and
the density z-score of the purest rare ball.  Nothing here changes a reported
number; it explains one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import (  # noqa: E402
    generate_granular_balls,
    knn_density,
    minmax_scale,
    select_representatives,
)
from src.mg_gbc import _mad, ball_mean_density, rare_core_mask  # noqa: E402

DATA_DIR = ROOT / "data" / "cicids2018"
OUT = ROOT / "results" / "cicids2018_mechanism_diagnosis.json"
PROTOCOL = ROOT / "experiments" / "cicids2018_temporal_protocol_v1.json"


def load_day(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    keep = ts.notna() & df["Label"].astype(str).str.strip().ne("Label")
    df = df.loc[keep].copy()
    df["_ts"] = ts[keep]
    cols = [c for c in df.columns if c not in ("Timestamp", "Label", "_ts")]
    df = df.loc[~df.duplicated(subset=cols, keep="first")]
    return df


def diagnose(blocks: pd.DataFrame, day_frame: pd.DataFrame, family_pool: dict) -> list[dict]:
    rows = []
    for _, blk in blocks.iterrows():
        start = pd.Timestamp(blk["start_utc"])
        window = day_frame.loc[day_frame["_ts"].dt.floor("300s").eq(start.tz_convert(None) if start.tzinfo else start)]
        if not len(window):
            continue
        seed = int(blk["sample_seed"])
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(window), size=5000, replace=False)
        sample = window.iloc[np.sort(pick)].reset_index(drop=True)
        cols = [c for c in sample.columns if c not in ("Timestamp", "Label", "_ts")]
        X = sample[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        X = minmax_scale(X)
        lab = sample["Label"].astype(str).str.strip().to_numpy()
        counts = pd.Series(lab).value_counts()
        rare_labels = [k for k, v in counts.items() if 3 <= v < np.sqrt(len(X))]
        if not rare_labels:
            continue

        rho, _ = knn_density(X, 10)
        reps = select_representatives(X, rho)
        balls = generate_granular_balls(X, reps, dissolve_small=False)
        owner = np.full(len(X), -1, dtype=int)
        for b_index, ball in enumerate(balls):
            owner[ball.indices] = b_index
        sizes = np.array([b.size for b in balls], dtype=float)
        gamma = ball_mean_density(rho, balls)
        centers = np.stack([b.center for b in balls])
        dens_z = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
        masked = rare_core_mask(sizes, gamma, centers, len(X), 0.5, 0.0, 3)

        for label in rare_labels:
            member = lab == label
            total = int(member.sum())
            in_ball = member & (owner >= 0)
            shares = pd.Series(owner[in_ball]).value_counts()
            largest_ball = int(shares.index[0]) if len(shares) else -1
            purity = []
            for b_index, ball in enumerate(balls):
                vals = lab[ball.indices]
                purity.append(float(np.mean(vals == label)))
            purest = int(np.argmax(purity))
            rows.append({
                "start_utc": str(start),
                "day": str(blk["day"]),
                "family": label,
                "rare_points": total,
                "balls_holding": int(len(shares)),
                "largest_share_in_one_ball": int(shares.iloc[0]) if len(shares) else 0,
                "largest_ball_size": int(balls[largest_ball].size) if largest_ball >= 0 else 0,
                "purest_ball_size": int(balls[purest].size),
                "purest_ball_purity": purity[purest],
                "purest_ball_density_z": float(dens_z[purest]),
                "purest_ball_masked": bool(masked[purest]),
                "purest_ball_below_cut": bool(balls[purest].size < np.sqrt(len(X))),
            })
    return rows


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blocks = pd.read_csv(ROOT / "results" / "cicids2018_temporal_blocks_pooled.csv")
    blocks = blocks.loc[blocks["method"].eq("ARP-GB")].copy()
    blocks = blocks.loc[blocks["rare_true_classes"].gt(0)]

    out = {"source": protocol["source"]["name"], "blocks_analysed": int(len(blocks)),
           "window_width_s": protocol["sampling"]["block_width_seconds"],
           "rows_per_block": protocol["sampling"]["rows_per_block"], "rows": []}
    for name in sorted({b["name"] for b in protocol["source"]["files"]}):
        day_frame = load_day(name)
        day = str(day_frame["_ts"].dt.date.astype(str).mode().iloc[0])
        day_blocks = blocks.loc[blocks["day"].eq(day)]
        if not len(day_blocks):
            continue
        found = diagnose(day_blocks, day_frame, {})
        out["rows"].extend(found)
        print(f"{day}: {len(found)} rare (window, family) pairs", flush=True)
        del day_frame

    frame = pd.DataFrame(out["rows"])
    if len(frame):
        contained = frame["largest_share_in_one_ball"] / frame["rare_points"]
        out["summary"] = {
            "pairs": int(len(frame)),
            "median_rare_points": float(frame["rare_points"].median()),
            "median_share_of_class_in_one_ball": float(contained.median()),
            "share_of_pairs_fully_in_one_ball": float((contained >= 0.999).mean()),
            "median_largest_ball_size": float(frame["largest_ball_size"].median()),
            "median_purest_ball_purity": float(frame["purest_ball_purity"].median()),
            "median_purest_ball_density_z": float(frame["purest_ball_density_z"].median()),
            "share_purest_ball_masked": float(frame["purest_ball_masked"].mean()),
            "share_purest_ball_below_cut": float(frame["purest_ball_below_cut"].mean()),
            "reading": (
                "A class is contained when share_of_class_in_one_ball is 1. The correction "
                "can only act on a class that is contained in a ball below the cut: if the "
                "class is dispersed across balls, or absorbed into a ball above the cut, the "
                "size truncation is not the step that removes it and the predicate has "
                "nothing to retain."
            ),
        }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out.get("summary", {}), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()