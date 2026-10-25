"""Score the frozen held-out OpenML cohort once, at the frozen defaults.

The cohort, the preprocessing, the constants and the four decision criteria are
fixed in experiments/openml_heldout_protocol_v1.json before this script runs.
The script verifies the protocol's model hashes, refuses to overwrite an
existing summary, fits ARP-GB and MG-GBC on every cohort data set, and evaluates
the pre-registered criteria exactly as written.  All four are reported whether or
not they are met; none is redefined here.

Usage:
    python -I experiments/run_openml_heldout.py
    python -I experiments/run_openml_heldout.py --resume   # continue an interrupted run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB  # noqa: E402
from src.datasets import _from_openml  # noqa: E402
from src.metrics_ext import evaluate_extended  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

PROTOCOL = ROOT / "experiments" / "openml_heldout_protocol_v1.json"
OUT_CSV = ROOT / "results" / "openml_heldout_sets.csv"
OUT_JSON = ROOT / "results" / "openml_heldout_summary.json"
TOL = 1e-12

RESULT_COLUMNS = [
    "protocol_id", "protocol_sha256", "subset", "data_id", "name", "version", "source",
    "n", "C", "d", "minority_class_size", "sqrt_n", "md5_checksum_protocol", "k", "method",
    "ACC", "NMI", "ARI", "rare_recall", "many_to_one_ACC", "n_clusters",
    "n_clusters_matched", "n_clusters_unmatched", "unmatched_cluster_share",
    "n_balls", "n_rare_cores", "n_fragment_edges", "runtime_s",
    "prediction_sha256", "partition_agreement_ari", "partition_identical",
    "n_loaded", "d_loaded", "shape_matches_protocol", "status", "error",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_protocol(protocol: dict) -> None:
    for rel, expected in protocol["models"]["model_hashes"].items():
        got = sha256_file(ROOT / rel)
        if got != str(expected).upper():
            raise RuntimeError(
                f"{rel} does not match the frozen protocol ({got[:12]} != "
                f"{str(expected)[:12]}). The cohort was registered against the "
                "source files recorded in it; do not score it with a "
                "modified model."
            )


def boot_median_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(d, size=d.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def sign_stats(d: np.ndarray) -> dict:
    nz = d[np.abs(d) > TOL]
    return {
        "wins": int((d > TOL).sum()),
        "ties": int((np.abs(d) <= TOL).sum()),
        "losses": int((d < -TOL).sum()),
        "two_sided_exact_sign_p": (
            float(binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue) if nz.size else None
        ),
        "two_sided_exact_wilcoxon_p": (
            float(wilcoxon(nz, alternative="two-sided", method="exact").pvalue) if nz.size else None
        ),
    }


def paired_stats(paired: pd.DataFrame, metric: str) -> dict:
    d = pd.to_numeric(paired[f"delta_{metric}"], errors="coerce").to_numpy(float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"pairs": 0}
    lo, hi = boot_median_ci(d)
    out = {"pairs": int(len(d)), "median_delta": float(np.median(d)),
           "mean_delta": float(np.mean(d)), "bootstrap95": [lo, hi],
           "min_delta": float(np.min(d)), "max_delta": float(np.max(d))}
    out.update(sign_stats(d))
    return out


def score_one(entry: dict, protocol: dict, common: dict) -> list[dict]:
    name, data_id = entry["name"], int(entry["data_id"])
    subset = entry["subset"]
    n_proto = int(entry["n"])
    k = int(np.clip(round(np.sqrt(n_proto)), 5, 10))
    X, y = _from_openml(name, data_id=data_id)
    n, d = int(len(X)), int(np.asarray(X).shape[1])
    shape_ok = (n == n_proto and d == int(entry["d"]))
    params = {key: value for key, value in protocol["models"]["mg_gbc_params"].items()
              if key not in ("rationale", "reason")}
    methods = [
        ("ARP-GB", ARPGB(k=k, scale=True, size_gate=False)),
        ("MG-GBC", MGGBC(k=k, **params)),
    ]
    rows, preds = [], {}
    for method_name, model in methods:
        row = dict(common, subset=subset, data_id=data_id, name=name,
                   version=entry.get("version"), source=entry.get("source"),
                   n=n_proto, C=int(entry["C"]),
                   d=int(entry["d"]), minority_class_size=int(entry["minority_class_size"]),
                   sqrt_n=float(entry["sqrt_n"]),
                   md5_checksum_protocol=entry.get("md5_checksum"),
                   k=k, method=method_name,
                   n_loaded=n, d_loaded=d, shape_matches_protocol=bool(shape_ok))
        started = time.perf_counter()
        try:
            labels = model.fit_predict(X)
            row["runtime_s"] = time.perf_counter() - started
            row.update(evaluate_extended(y, labels))
            row.update(
                prediction_sha256=sha256_bytes(np.asarray(labels, dtype=np.int64).tobytes(order="C")),
                n_balls=int(model.n_balls_),
                n_rare_cores=int(getattr(model, "n_rare_cores_", 0) or 0),
                n_fragment_edges=int(getattr(model, "n_fragment_edges_", 0) or 0),
                status="ok", error="",
            )
            preds[method_name] = np.asarray(labels)
        except Exception as exc:  # noqa: BLE001
            row.update(runtime_s=time.perf_counter() - started, status="error",
                       error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    if len(preds) == 2:
        agree = float(adjusted_rand_score(preds["ARP-GB"], preds["MG-GBC"]))
        for row in rows:
            row["partition_agreement_ari"] = agree
            row["partition_identical"] = bool(agree >= 1.0 - TOL)
    else:
        for row in rows:
            row["partition_agreement_ari"] = np.nan
            row["partition_identical"] = False
    print(f"  {subset:<8} id={data_id:>5} {name:<28} n={n:>6} d={d:>3} k={k:>2} "
          f"cores={rows[-1].get('n_rare_cores')} frag={rows[-1].get('n_fragment_edges')} "
          f"ARI {rows[0].get('ARI', float('nan')):.4f} -> {rows[-1].get('ARI', float('nan')):.4f} "
          f"rare {rows[0].get('rare_recall', float('nan')):.3f} -> "
          f"{rows[-1].get('rare_recall', float('nan')):.3f}", flush=True)
    return rows


def build_paired(results: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    ok = results.loc[results["status"].eq("ok")]
    if subset is not None:
        ok = ok.loc[ok["subset"].eq(subset)]
    metrics = ["ACC", "NMI", "ARI", "rare_recall", "many_to_one_ACC", "n_clusters", "runtime_s"]
    left = ok.loc[ok["method"].eq("ARP-GB")].set_index("data_id")
    right = ok.loc[ok["method"].eq("MG-GBC")].set_index("data_id")
    paired = left[["name", "subset", "source", "n", "C", "d", "k",
                   "shape_matches_protocol"] + metrics].join(
        right[metrics + ["n_balls", "n_rare_cores", "n_fragment_edges",
                         "partition_agreement_ari", "partition_identical"]],
        how="inner", lsuffix="_arp", rsuffix="_mg",
    ).reset_index()
    for metric in ["ACC", "NMI", "ARI", "rare_recall", "many_to_one_ACC"]:
        paired[f"delta_{metric}"] = paired[f"{metric}_mg"] - paired[f"{metric}_arp"]
    paired["runtime_ratio_mg_over_arp"] = paired["runtime_s_mg"] / paired["runtime_s_arp"]
    return paired


def subset_stats(paired: pd.DataFrame) -> dict:
    with_rare = paired.loc[
        pd.to_numeric(paired["rare_recall_arp"], errors="coerce").notna()
        & pd.to_numeric(paired["rare_recall_mg"], errors="coerce").notna()
    ]
    return {
        "sets": int(len(paired)),
        "sets_with_defined_rare_recall": int(len(with_rare)),
        "rare_recall": paired_stats(with_rare, "rare_recall") if len(with_rare) else {},
        "ARI": paired_stats(paired, "ARI") if len(paired) else {},
        "ACC": paired_stats(paired, "ACC") if len(paired) else {},
        "NMI": paired_stats(paired, "NMI") if len(paired) else {},
        "many_to_one_ACC": (paired_stats(paired, "many_to_one_ACC") if len(paired) else {}),
        "identical_partitions": int(paired["partition_identical"].sum()) if len(paired) else 0,
        "identical_partition_share": (
            float(paired["partition_identical"].mean()) if len(paired) else None
        ),
        "sets_with_rare_core": int((paired["n_rare_cores"] > 0).sum()) if len(paired) else 0,
        "sets_with_fragment_edge": (
            int((paired["n_fragment_edges"] > 0).sum()) if len(paired) else 0
        ),
        "median_runtime_ratio": (
            float(np.nanmedian(paired["runtime_ratio_mg_over_arp"])) if len(paired) else None
        ),
        "shape_matches_protocol": (
            int(paired["shape_matches_protocol"].sum()) if len(paired) else 0
        ),
    }


def summarise(results: pd.DataFrame, protocol: dict, protocol_hash: str) -> dict:
    rare = build_paired(results, "rare")
    control = build_paired(results, "control")
    whole = build_paired(results, None)

    stats = {
        "rare_subset": subset_stats(rare),
        "locality_control": subset_stats(control),
        "whole_cohort": subset_stats(whole),
        "excluded": {
            "rows_by_status": {str(k): int(v) for k, v in
                               results["status"].value_counts().items()},
            "data_ids_not_scored": sorted(
                {int(e["data_id"]) for e in protocol["cohort"]}
                - set(int(v) for v in whole["data_id"])
            ),
            "note": ("A data set leaves the paired analysis only when a load or a fit raised; the "
                     "reason is in the error column of the result CSV. No data set is dropped "
                     "because of the value of any metric."),
        },
    }

    crit = protocol["preregistered_criteria"]
    rr = stats["rare_subset"]["rare_recall"]
    ari = stats["rare_subset"]["ARI"]
    worst_ari = ari.get("min_delta")
    identical_share = stats["locality_control"]["identical_partition_share"]
    loss_sets = rare.loc[pd.to_numeric(rare["delta_ARI"], errors="coerce") < -0.05]
    h4_violations = loss_sets.loc[pd.to_numeric(loss_sets["n_rare_cores"], errors="coerce") < 1]

    verdicts = {
        "H1_rare_recall": {
            "criterion": crit["H1_rare_recall"],
            "evaluated_on": "rare subset",
            "observed": {"median_delta_rare_recall": rr.get("median_delta"),
                         "bootstrap95": rr.get("bootstrap95"),
                         "two_sided_exact_sign_p": rr.get("two_sided_exact_sign_p"),
                         "two_sided_exact_wilcoxon_p": rr.get("two_sided_exact_wilcoxon_p"),
                         "wins": rr.get("wins"), "ties": rr.get("ties"), "losses": rr.get("losses"),
                         "pairs": rr.get("pairs")},
            "met": bool(rr.get("pairs") and (rr.get("median_delta") or 0) > 0
                        and rr.get("two_sided_exact_sign_p") is not None
                        and rr["two_sided_exact_sign_p"] < 0.05),
        },
        "H2_noninferiority": {
            "criterion": crit["H2_noninferiority"],
            "evaluated_on": "rare subset, with the whole cohort reported beside it",
            "observed": {"median_delta_ARI_rare": ari.get("median_delta"),
                         "worst_delta_ARI_rare": worst_ari,
                         "bootstrap95_rare": ari.get("bootstrap95"),
                         "median_delta_ARI_whole_cohort":
                             stats["whole_cohort"]["ARI"].get("median_delta"),
                         "worst_delta_ARI_whole_cohort":
                             stats["whole_cohort"]["ARI"].get("min_delta")},
            "met": bool(ari.get("pairs") and (ari.get("median_delta") or -1) >= -0.05
                        and worst_ari is not None and worst_ari >= -0.15),
        },
        "H3_locality": {
            "criterion": crit["H3_locality"],
            "evaluated_on": "locality control",
            "observed": {"control_sets": stats["locality_control"]["sets"],
                         "identical_partitions": stats["locality_control"]["identical_partitions"],
                         "identical_partition_share": identical_share,
                         "control_sets_with_rare_core":
                             stats["locality_control"]["sets_with_rare_core"]},
            "met": bool(identical_share is not None and identical_share >= 0.90),
        },
        "H4_loss_mechanism": {
            "criterion": crit["H4_loss_mechanism"],
            "evaluated_on": "rare subset",
            "observed": {"sets_below_margin": int(len(loss_sets)),
                         "sets_below_margin_without_rare_core": int(len(h4_violations)),
                         "violating_data_ids": [int(v) for v in h4_violations["data_id"]]},
            "met": bool(len(h4_violations) == 0),
        },
        "reporting_rule": crit["reporting_rule"],
    }

    members = protocol["analysis_subsets"]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort_sizes": {
            "cohort": len(protocol["cohort"]),
            "rare_subset_frozen": members["rare"]["members"],
            "locality_control_frozen": members["control"]["members"],
            "pairs_scored": {"rare": int(len(rare)), "control": int(len(control)),
                             "whole": int(len(whole))},
        },
        "statistics": stats,
        "preregistered_verdicts": verdicts,
        "criteria_met": {k: v["met"] for k, v in verdicts.items() if isinstance(v, dict)
                         and "met" in v},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run instead of refusing")
    ap.add_argument("--no-summary", action="store_true",
                    help="score only; do not write the summary (for an interrupted run)")
    args = ap.parse_args()

    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    protocol_hash = sha256_bytes(raw)
    verify_protocol(protocol)
    print(f"protocol verified: {protocol['protocol_id']} {protocol_hash[:16]}", flush=True)

    if OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.name} already exists: this cohort is scored once. "
                         "The pre-registered run is complete and is not repeated.")

    common = {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash}
    existing = pd.DataFrame(columns=RESULT_COLUMNS)
    if OUT_CSV.exists():
        if not args.resume:
            raise SystemExit(f"{OUT_CSV.name} exists from an interrupted run. "
                             "Re-run with --resume to finish it; the summary is written once.")
        existing = pd.read_csv(OUT_CSV)
        print(f"resuming: {len(existing)} rows already scored", flush=True)
    done = set(existing["data_id"].astype(int)) if len(existing) else set()

    records = existing.to_dict(orient="records")
    cohort = protocol["cohort"]
    subsets = {s: int(sum(1 for e in cohort if e["subset"] == s))
               for s in ("rare", "control")}
    print(f"\n=== cohort: {len(cohort)} data sets "
          f"(rare {subsets['rare']}, locality control {subsets['control']}) ===", flush=True)
    for position, entry in enumerate(cohort, start=1):
        if int(entry["data_id"]) in done:
            continue
        if position % 20 == 1:
            print(f"  [{position}/{len(cohort)}]", flush=True)
        try:
            rows = score_one(entry, protocol, common)
        except Exception as exc:  # noqa: BLE001
            rows = [dict(common, subset=entry["subset"], data_id=int(entry["data_id"]),
                         name=entry["name"], version=entry.get("version"),
                         source=entry.get("source"),
                         n=int(entry["n"]), C=int(entry["C"]), d=int(entry["d"]),
                         minority_class_size=int(entry["minority_class_size"]),
                         sqrt_n=float(entry["sqrt_n"]), method=m, status="load_error",
                         error=f"{type(exc).__name__}: {exc}")
                    for m in ("ARP-GB", "MG-GBC")]
            print(f"  {entry['subset']} id={entry['data_id']} {entry['name']}: "
                  f"LOAD ERROR {rows[0]['error']}", flush=True)
        records.extend(rows)
        pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(
            OUT_CSV, index=False, float_format="%.12g")

    results = pd.DataFrame(records, columns=RESULT_COLUMNS)
    print(f"\nscored {results['data_id'].nunique()} data sets "
          f"({int(results['status'].eq('ok').sum())} method runs ok, "
          f"{int((~results['status'].eq('ok')).sum())} failed)", flush=True)

    if args.no_summary:
        print(f"wrote {OUT_CSV.relative_to(ROOT)} (summary withheld)")
        return

    summary = summarise(results, protocol, protocol_hash)
    OUT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("\n=== pre-registered criteria ===")
    for name, block in summary["preregistered_verdicts"].items():
        if not isinstance(block, dict) or "met" not in block:
            continue
        print(f"{name:<20} {'MET' if block['met'] else 'NOT MET'}   {json.dumps(block['observed'])}")
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
