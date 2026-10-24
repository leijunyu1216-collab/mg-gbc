"""ARP-GB re-run under the identical graph policy used by MG-GBC.

ARPGB's guarded default (size_gate=True) drops merge candidates that MG-GBC
never applies, so a default-config ARP-GB row differs from MG-GBC in two
factors at once.  This script holds the policy fixed (size_gate=False) so the
only remaining difference is the rare-core layer.

Outputs results/matched_arp.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.metrics import evaluate
from run_rare_real import EVAL_LOADERS, _load


def main() -> None:
    rows = []
    print(f"{'dataset':<22} {'ARP-def':>8} {'ARP-matched':>12} {'delta':>7}")
    for spec in EVAL_LOADERS:
        loaded = _load(spec)
        if loaded is None:
            print(f"{spec[0]:<22} load failed")
            continue
        X, y, name, k = loaded
        k = int(np.clip(k, 2, max(2, len(X) - 1)))
        out = {}
        for tag, gate in (("ARP-default", True), ("ARP-matched", False)):
            pred = ARPGB(k=k, size_gate=gate).fit_predict(X)
            s = evaluate(y, pred)
            out[tag] = s
            rows.append(
                {
                    "dataset": name,
                    "n": len(X),
                    "d": X.shape[1],
                    "k": k,
                    "policy": tag,
                    "ACC": s["ACC"],
                    "NMI": s["NMI"],
                    "ARI": s["ARI"],
                    "rare_recall": s["rare_recall"],
                    "k_hat": int(s["n_clusters"]),
                    "k_true": int(s["n_true"]),
                }
            )
        d = out["ARP-matched"]["ARI"] - out["ARP-default"]["ARI"]
        print(
            f"{name:<22} {out['ARP-default']['ARI']:8.3f} "
            f"{out['ARP-matched']['ARI']:12.3f} {d:+7.3f}"
        )

    path = ROOT / "results" / "matched_arp.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()