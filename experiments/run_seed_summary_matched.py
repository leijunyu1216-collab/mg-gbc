"""Ten-draw summary with the matched graph policy in both methods."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.datasets import load_digits_rare, synthetic_imbalance, synthetic_split_rare
from src.metrics import evaluate
from src.mg_gbc import MGGBC


def run(name, loader, k):
    arp_ari, mg_ari, arp_rec, mg_rec = [], [], [], []
    for seed in range(10):
        X, y, _ = loader(seed=seed)
        for model, ari, rec in (
            (ARPGB(k=k, size_gate=False), arp_ari, arp_rec),
            (MGGBC(k=k), mg_ari, mg_rec),
        ):
            s = evaluate(y, model.fit_predict(X))
            ari.append(float(s["ARI"]))
            r = s["rare_recall"]
            rec.append(float(r) if r == r else float("nan"))
    d = np.asarray(mg_ari) - np.asarray(arp_ari)
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz, alternative="greater", method="exact").pvalue) if len(nz) else 1.0
    return {
        "dataset": name,
        "ARP_mean_ari": float(np.mean(arp_ari)),
        "ARP_sd_ari": float(np.std(arp_ari)),
        "MG_mean_ari": float(np.mean(mg_ari)),
        "MG_sd_ari": float(np.std(mg_ari)),
        "ARP_mean_rrec": float(np.nanmean(arp_rec)),
        "ARP_sd_rrec": float(np.nanstd(arp_rec)),
        "MG_mean_rrec": float(np.nanmean(mg_rec)),
        "MG_sd_rrec": float(np.nanstd(mg_rec)),
        "wins": int((d > 1e-12).sum()),
        "ties": int((np.abs(d) <= 1e-12).sum()),
        "losses": int((d < -1e-12).sum()),
        "median_delta_ari": float(np.median(d)),
        "wilcoxon_p": p,
    }


def main() -> None:
    out = [
        run("Synthetic-Imbalance", synthetic_imbalance, 12),
        run("Synthetic-SplitRare", synthetic_split_rare, 12),
        run("Digits-Rare", load_digits_rare, 10),
    ]
    path = ROOT / "results" / "seed_summary_matched.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in out:
        print(
            f"{r['dataset']:<22} ARP {r['ARP_mean_ari']:.3f}+-{r['ARP_sd_ari']:.3f} "
            f"MG {r['MG_mean_ari']:.3f}+-{r['MG_sd_ari']:.3f} "
            f"W/T/L {r['wins']}/{r['ties']}/{r['losses']} p={r['wilcoxon_p']:.5f} "
            f"Rrec {r['ARP_mean_rrec']:.3f} -> {r['MG_mean_rrec']:.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()