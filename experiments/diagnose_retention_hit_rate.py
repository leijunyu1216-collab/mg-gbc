"""How many rare points the retention predicate actually rescues, per source.

The correction can only change a block's partition through the balls it keeps
that ARP-GB would have dissolved.  So the natural quantity to measure is the
share of rare-class points that sit inside a retained ball.  If that share is
near zero the edit is empty and the two methods coincide, whatever the rest of
the geometry looks like.

This script reports it for both real sources against the same frozen blocks
the reported comparisons used.  It is a diagnostic over stored blocks; it
changes no reported number.
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
    minmax_scale,
    select_representatives,
)
from src.mg_gbc import _mad, ball_mean_density, rare_core_mask  # noqa: E402

PROTOCOL = ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json"
CIC_PROTOCOL = ROOT / "experiments" / "cicids2018_temporal_protocol_v1.json"
CIC_DATA = ROOT / "data" / "cicids2018"
OUT = ROOT / "results" / "retention_hit_rate.json"
TABLE = ROOT / "results" / "retention_hit_rate.csv"

K = 10


def rescues(X: np.ndarray, lab: np.ndarray, rare: np.ndarray) -> dict:
    rho, _ = knn_density(X, K)
    balls = generate_granular_balls(X, select_representatives(X, rho), dissolve_small=False)
    sizes = np.array([b.size for b in balls], dtype=float)
    gamma = ball_mean_density(rho, balls)
    centers = np.stack([b.center for b in balls])
    dz = 0.6745 * (gamma - np.median(gamma)) / _mad(gamma)
    masked = rare_core_mask(sizes, gamma, centers, len(X), 0.5, 0.0, 3)

    in_masked = np.zeros(len(X), dtype=bool)
    for flag, ball in zip(masked, balls):
        if flag:
            in_masked[ball.indices] = True
    rare_points = int(rare.sum())
    return {
        "n_balls": int(len(balls)),
        "n_masked_balls": int(masked.sum()),
        "n_masked_points": int(in_masked.sum()),
        "rare_points": rare_points,
        "rare_points_in_masked": int((in_masked & rare).sum()),
        "rare_recall_of_rescued": float((in_masked & rare).sum() / rare_points) if rare_points else np.nan,
        "purity_of_masked": float(rare[in_masked].mean()) if in_masked.any() else np.nan,
        "median_density_z_masked": float(np.median(dz[masked])) if masked.any() else np.nan,
        "median_density_z_all": float(np.median(dz)),
    }


def unsw_rows() -> list[dict]:
    protocol, _ = R.load_protocol(PROTOCOL)
    starts = protocol["sampling"]["selected_block_starts_unix"]
    source_rows = R.load_selected_rows(protocol)
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    rows = []
    for block_id, start in enumerate(starts, start=1):
        sample = R.sample_block(source_rows, int(start), n, base_seed + block_id)
        X = R.prepare_matrix(sample)
        lab, classes, counts = R.normalized_labels(sample)
        rare = np.isin(lab, [i for i, c in enumerate(counts) if 0 < c < np.sqrt(len(X))])
        if not rare.any():
            continue
        rows.append({"source": "UNSW-NB15", "block": f"{block_id:02d}", **rescues(X, lab, rare)})
        print(f"UNSW block {block_id:02d}", flush=True)
    return rows


def cic_rows() -> list[dict]:
    protocol = json.loads(CIC_PROTOCOL.read_text(encoding="utf-8"))
    width = int(protocol["sampling"]["block_width_seconds"])
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    rows = []
    for spec in protocol["source"]["files"]:
        df = pd.read_csv(CIC_DATA / spec["name"], low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        keep = ts.notna() & df["Label"].astype(str).str.strip().ne("Label")
        df = df.loc[keep].copy()
        df["_ts"] = ts[keep]
        cols = [c for c in df.columns if c not in ("Timestamp", "Label", "_ts")]
        df = df.loc[~df.duplicated(subset=cols, keep="first")]
        bucket = (df["_ts"].astype("int64") // 10**9 // width) * width
        for bucket_start, window in df.groupby(bucket):
            if len(window) < n:
                continue
            rng = np.random.default_rng(base_seed + int(bucket_start))
            pick = rng.choice(len(window), size=n, replace=False)
            sample = window.iloc[np.sort(pick)].reset_index(drop=True)
            X = sample[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            X[~np.isfinite(X)] = np.nan
            X = minmax_scale(np.where(np.isnan(X), np.nanmedian(X, axis=0), X))
            lab, counts = pd.factorize(sample["Label"].astype(str).str.strip())
            rare = np.isin(lab, [i for i, c in enumerate(np.bincount(lab, minlength=len(counts)))
                                 if 0 < c < np.sqrt(len(X))])
            if not rare.any():
                continue
            stamp = pd.Timestamp(int(bucket_start), unit="s", tz="UTC").isoformat()
            rows.append({"source": "CIC-IDS2018", "block": stamp, **rescues(X, lab, rare)})
        print(f"{spec['name']}: cumulative {len(rows)} rare-bearing windows", flush=True)
        del df
    return rows


def main() -> None:
    rows = unsw_rows() + cic_rows()
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE, index=False, float_format="%.12g")
    summary = {}
    for source, group in frame.groupby("source"):
        summary[source] = {
            "blocks_with_a_rare_class": int(len(group)),
            "median_rare_points": float(group["rare_points"].median()),
            "median_masked_balls": float(group["n_masked_balls"].median()),
            "share_blocks_with_no_masked_ball": float((group["n_masked_balls"] == 0).mean()),
            "median_rare_recall_of_rescued": float(group["rare_recall_of_rescued"].median()),
            "share_blocks_rescuing_any_rare_point": float((group["rare_points_in_masked"] > 0).mean()),
            "median_purity_of_masked": float(np.nanmedian(group["purity_of_masked"])),
            "median_density_z_masked": float(np.nanmedian(group["median_density_z_masked"])),
            "median_density_z_all": float(np.nanmedian(group["median_density_z_all"])),
        }
    out = {"k": K, "by_source": summary,
           "reading": (
               "rare_recall_of_rescued is the share of a block's rare-class points that sit "
               "inside a retained ball. It bounds the recall change the correction can "
               "produce on that block: the edit cannot move points it does not keep."
           )}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT}\nwrote {TABLE}")


if __name__ == "__main__":
    main()