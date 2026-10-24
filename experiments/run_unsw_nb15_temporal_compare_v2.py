"""Run the UNSW-NB15 temporal benchmark under protocol v2.

Differences from the v1 runner, all declared in the v2 protocol:

  * blocks are drawn from buckets satisfying the label-free rule n_services >= 8
  * MG-GBC takes rare_density_z from the protocol instead of the 0.5 default
  * every block is scored with many-to-one ACC and the Hungarian bijection
    penalty in addition to the frozen metric bundle
  * the primary research-scope ARI/NMI summary uses multi-class blocks; the
    finite sklearn values on single-class blocks remain in the raw block output

The v1 runner and the v1 protocol are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_unsw_nb15_temporal_compare import (  # noqa: E402
    exact_duplicate_stats,
    load_protocol,
    normalized_labels,
    prediction_hash,
    prepare_matrix,
    sample_block,
    sample_identity_hash,
    sha256_bytes,
    verify_frozen_inputs,
)
from experiments.unsw_block_cache import get_source_rows  # noqa: E402
from src.arp_gb import ARPGB  # noqa: E402
from src.metrics_ext import EXTENDED_KEYS, evaluate_extended  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

RESULT_COLUMNS = [
    "protocol_id", "protocol_sha256", "block_id", "block_start_unix",
    "block_end_unix", "block_start_utc", "available_rows", "sample_seed",
    "sample_n", "feature_count", "sample_identity_sha256",
    "input_matrix_sha256", "prediction_sha256", "duplicate_extra_rows",
    "duplicate_rate", "n_true", "min_true_class_count", "rare_true_classes",
    "method", "runtime_s", "ACC", "NMI", "ARI", "rare_recall",
    "n_clusters", "n_balls", "n_rare_cores", "n_fragment_edges",
] + EXTENDED_KEYS + ["status", "error"]

TOL = 1e-12


def finite_median(series: pd.Series) -> float | None:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def sign_p(deltas: np.ndarray) -> float | None:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > TOL]
    return float(binomtest(int(np.sum(nz > 0)), nz.size, 0.5).pvalue) if nz.size else None


def build_summary(results: pd.DataFrame, protocol: dict, protocol_hash: str):
    ok = results.loc[results["status"].eq("ok")].copy()
    metric_cols = ["ARI", "NMI", "ACC", "rare_recall", "runtime_s", "n_clusters", "n_balls",
                   "many_to_one_ACC", "unmatched_cluster_share"]
    left = ok.loc[ok["method"].eq("ARP-GB"), ["block_id", "n_true"] + metric_cols].set_index("block_id")
    right = ok.loc[ok["method"].eq("MG-GBC"),
                   ["block_id"] + metric_cols + ["n_rare_cores", "n_fragment_edges"]].set_index("block_id")
    paired = left.join(right, how="inner", lsuffix="_arp", rsuffix="_mg").reset_index()
    for m in ["ARI", "NMI", "ACC", "rare_recall", "many_to_one_ACC", "unmatched_cluster_share"]:
        paired[f"delta_{m}"] = paired[f"{m}_mg"] - paired[f"{m}_arp"]

    informative = pd.to_numeric(paired["n_true"], errors="coerce").gt(1)
    inf_rows = paired.loc[informative]
    d_ari = inf_rows["delta_ARI"].to_numpy(dtype=float)
    d_rare = pd.to_numeric(inf_rows["delta_rare_recall"], errors="coerce").to_numpy(dtype=float)

    methods = {}
    for method, group in ok.groupby("method", sort=True):
        inf_group = group.loc[pd.to_numeric(group["n_true"], errors="coerce").gt(1)]
        methods[method] = {
            "blocks": int(len(group)),
            "informative_blocks": int(len(inf_group)),
            "median_ARI_informative": finite_median(inf_group["ARI"]),
            "median_NMI_informative": finite_median(inf_group["NMI"]),
            "median_ACC_all_blocks": finite_median(group["ACC"]),
            "median_many_to_one_ACC_all_blocks": finite_median(group["many_to_one_ACC"]),
            "median_rare_recall": finite_median(group["rare_recall"]),
            "median_runtime_s": finite_median(group["runtime_s"]),
            "median_n_clusters": finite_median(group["n_clusters"]),
            "median_n_balls": finite_median(group["n_balls"]),
        }

    primary = {
        "definition": (
            "ARI and NMI medians are taken over blocks with at least two true classes, "
            "because single-class blocks do not test between-class recovery. Their raw "
            "sklearn metric values remain in the block-level output."
        ),
        "informative_blocks": int(informative.sum()),
        "single_class_blocks": int((~informative).sum()),
        "median_delta_ARI": finite_median(inf_rows["delta_ARI"]),
        "median_delta_NMI": finite_median(inf_rows["delta_NMI"]),
        "median_delta_ACC": finite_median(inf_rows["delta_ACC"]),
        "median_delta_many_to_one_ACC": finite_median(inf_rows["delta_many_to_one_ACC"]),
        "median_delta_rare_recall": finite_median(inf_rows["delta_rare_recall"]),
        "ARI_wins": int(np.sum(d_ari > TOL)),
        "ARI_ties": int(np.sum(np.abs(d_ari) <= TOL)),
        "ARI_losses": int(np.sum(d_ari < -TOL)),
        "ARI_worst_delta": float(d_ari.min()) if d_ari.size else None,
        "ARI_noninferior_at_minus_0_005": int(np.sum(d_ari >= -0.005)),
        "ARI_noninferior_fraction": float(np.mean(d_ari >= -0.005)) if d_ari.size else None,
        "ARI_two_sided_exact_sign_p": sign_p(d_ari),
        "rare_recall_two_sided_exact_sign_p": sign_p(d_rare),
    }
    acc_note = {
        "explanation": "Hungarian ACC cannot score more clusters than there are classes; unmatched_cluster_share is the mass the bijection discards.",
        "median_delta_unmatched_cluster_share": finite_median(paired["delta_unmatched_cluster_share"]),
        "median_delta_ACC_net_of_bijection": finite_median(
            paired["delta_ACC"] + paired["delta_unmatched_cluster_share"]),
        "blocks_where_bijection_explains_full_ACC_gap": int(
            np.sum((paired["delta_ACC"] + paired["delta_unmatched_cluster_share"]) >= -1e-9)),
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mg_gbc_params": protocol["models"].get("mg_gbc_params", {}),
        "methods": methods,
        "primary_paired_comparison_mg_minus_arp": primary,
        "acc_bijection_accounting": acc_note,
        "relationship_to_v1": protocol["relationship_to_v1"],
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_blocks_v2.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_summary_v2.json")
    parser.add_argument("--paired", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_paired_v2.csv")
    parser.add_argument("--limit-blocks", type=int, default=None)
    parser.add_argument("--arp-size-gate", default="true",
                        help="size-ratio guard in the ARP-GB overlap test")
    parser.add_argument("--skip-freeze-check", action="store_true",
                        help="run a declared sensitivity despite changed source hashes")
    args = parser.parse_args()

    protocol, protocol_hash = load_protocol(args.protocol)
    if args.skip_freeze_check:
        print("WARNING: frozen-source check skipped; this run is a sensitivity analysis.", flush=True)
    else:
        verify_frozen_inputs(protocol)
    print(f"protocol verified: {protocol['protocol_id']} {protocol_hash}", flush=True)

    starts = protocol["sampling"]["selected_block_starts_unix"]
    if args.limit_blocks:
        starts = starts[: args.limit_blocks]
    source_rows = get_source_rows(protocol)

    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    k = int(protocol["models"]["k"])
    rare_z = float(protocol["models"]["mg_gbc_params"]["rare_density_z"])
    stime_all = pd.to_numeric(source_rows["stime"], errors="coerce")

    records = []
    for block_id, start in enumerate(starts, start=1):
        seed = base_seed + block_id
        sample = sample_block(source_rows, int(start), n, seed)
        available = int(((stime_all >= start) & (stime_all < start + 300)).sum())
        X = prepare_matrix(sample)
        y_true, classes, counts = normalized_labels(sample)
        dup_extra, dup_rate = exact_duplicate_stats(sample)
        common = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_hash,
            "block_id": block_id,
            "block_start_unix": int(start),
            "block_end_unix": int(start + 300),
            "block_start_utc": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
            "available_rows": available,
            "sample_seed": seed,
            "sample_n": n,
            "feature_count": int(X.shape[1]),
            "sample_identity_sha256": sample_identity_hash(sample),
            "input_matrix_sha256": sha256_bytes(np.ascontiguousarray(X).tobytes()),
            "duplicate_extra_rows": dup_extra,
            "duplicate_rate": dup_rate,
            "n_true": int(len(classes)),
            "min_true_class_count": int(counts.min()),
            "rare_true_classes": int(np.sum(counts < np.sqrt(n))),
        }
        arp_gate = args.arp_size_gate.lower() in ("1", "true", "yes")
        methods = [
            ("ARP-GB", ARPGB(k=k, scale=False, size_gate=arp_gate)),
            ("MG-GBC", MGGBC(k=k, scale=False, rare_density_z=rare_z)),
        ]
        for method_name, model in methods:
            started = time.perf_counter()
            row = dict(common, method=method_name)
            try:
                y_pred = model.fit_predict(X)
                runtime = time.perf_counter() - started
                row.update(evaluate_extended(y_true, y_pred))
                row.update(
                    runtime_s=runtime,
                    prediction_sha256=prediction_hash(y_pred),
                    n_balls=int(model.n_balls_),
                    n_rare_cores=int(getattr(model, "n_rare_cores_", 0) or 0),
                    n_fragment_edges=int(getattr(model, "n_fragment_edges_", 0) or 0),
                    status="ok", error="",
                )
                print(f"block {block_id:02d}/{len(starts)} {method_name}: "
                      f"ARI={row['ARI']:.6f} NMI={row['NMI']:.6f} ACC={row['ACC']:.6f} "
                      f"m2oACC={row['many_to_one_ACC']:.6f} rare={row['rare_recall']:.6f} "
                      f"clusters={int(row['n_clusters'])} time={runtime:.2f}s", flush=True)
            except Exception as exc:
                row.update(runtime_s=time.perf_counter() - started, status="error",
                           error=f"{type(exc).__name__}: {exc}")
                print(f"block {block_id:02d} {method_name}: ERROR {row['error']}", flush=True)
            records.append(row)
            pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(
                args.output, index=False, float_format="%.12g")

    results = pd.DataFrame(records, columns=RESULT_COLUMNS)
    summary, paired = build_summary(results, protocol, protocol_hash)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired.to_csv(args.paired, index=False, float_format="%.12g")
    print("\n" + json.dumps(summary["primary_paired_comparison_mg_minus_arp"], indent=2), flush=True)
    print("\n" + json.dumps(summary["acc_bijection_accounting"], indent=2), flush=True)


if __name__ == "__main__":
    main()
