"""Pool the matched-policy UNSW v2 and holdout blocks (40 blocks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TOL = 1e-12


def sign_p(d: np.ndarray) -> float:
    nz = d[np.abs(d) > TOL]
    return float(binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue) if nz.size else 1.0


def boot_median_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(d, size=d.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    v2 = pd.read_csv(OUT / "unsw_nb15_temporal_paired_v2_matched.csv")
    hold = pd.read_csv(OUT / "unsw_holdout_blocks_matched.csv")

    d = np.concatenate([
        v2["delta_ARI"].to_numpy(float),
        hold["dARI_z050"].to_numpy(float),
    ])
    d_acc = np.concatenate([
        v2["delta_ACC"].to_numpy(float),
        hold["dACC_z050"].to_numpy(float),
    ])
    d_m2o = np.concatenate([
        v2["delta_many_to_one_ACC"].to_numpy(float),
        hold["dmany_to_one_ACC_z050"].to_numpy(float),
    ])
    d_rare = np.concatenate([
        v2["delta_rare_recall"].to_numpy(float),
        hold["drare_recall_z050"].to_numpy(float),
    ])
    ok = np.isfinite(d) & np.isfinite(d_rare)
    d, d_acc, d_m2o, d_rare = d[ok], d_acc[ok], d_m2o[ok], d_rare[ok]
    lo, hi = boot_median_ci(d)
    res = {
        "blocks": int(d.size),
        "median_dARI": float(np.median(d)),
        "ARI_CI95": [lo, hi],
        "ARI_wins": int((d > TOL).sum()),
        "ARI_losses": int((d < -TOL).sum()),
        "median_dACC": float(np.median(d_acc)),
        "median_dManyToOneACC": float(np.median(d_m2o)),
        "median_dRareRecall": float(np.median(d_rare)),
        "rare_wins": int((d_rare > TOL).sum()),
        "rare_ties": int((np.abs(d_rare) <= TOL).sum()),
        "rare_losses": int((d_rare < -TOL).sum()),
        "rare_sign_p": sign_p(d_rare),
        "ARI_sign_p": sign_p(d),
        "rare_wilcoxon_p": float(wilcoxon(d_rare, alternative="two-sided", method="exact").pvalue),
        "ARI_wilcoxon_p": float(wilcoxon(d, alternative="two-sided", method="exact").pvalue),
    }
    (OUT / "unsw_pooled_matched.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()