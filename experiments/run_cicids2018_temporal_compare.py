"""Paired temporal-block comparison of MG-GBC and ARP-GB on CIC-IDS2018.

Mirrors experiments/run_unsw_nb15_temporal_compare.py: the same block width,
the same rows per block, the same k, the same matched graph policy, and the
same honesty rules -- labels are read only after a block has been drawn and its
feature matrix frozen, and every eligible window is evaluated.

    python -I experiments/run_cicids2018_temporal_compare.py --days 2018-02-22 \
        --tag replication

The protocol is frozen in experiments/cicids2018_temporal_protocol_v1.json and
its SHA-256 is written into every output row, so a result cannot be attached to
a protocol it was not produced under.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB  # noqa: E402
from src.core import minmax_scale  # noqa: E402
from src.metrics_ext import evaluate_extended  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

TIMESTAMP = "Timestamp"
LABEL = "Label"
DATA_DIR = ROOT / "data" / "cicids2018"

RESULT_COLUMNS = [
    "protocol_id", "protocol_sha256", "window_id", "day", "start_utc", "window_rows",
    "sample_seed", "feature_count", "sample_identity_sha256", "input_matrix_sha256",
    "prediction_sha256", "n_true", "min_true_class_count", "rare_true_classes",
    "method", "runtime_s", "ACC", "NMI", "ARI", "rare_recall", "many_to_one_ACC",
    "unmatched_cluster_share", "n_clusters", "n_balls", "n_rare_cores",
    "n_fragment_edges", "partition_agreement_ari", "partition_identical",
    "partition_identical_unrelabelled", "status", "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def verify_frozen_inputs(protocol: dict) -> None:
    mismatches = []
    for item in protocol["source"]["files"]:
        path = DATA_DIR / item["name"]
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != item["sha256"]:
            mismatches.append(f"{path}: expected {item['sha256']}, got {actual}")
    for rel, expected in protocol["models"]["model_hashes"].items():
        path = ROOT / rel
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError("Frozen-input verification failed:\n" + "\n".join(mismatches))


def iter_day_windows(protocol: dict, days: set[str] | None):
    """Yield (day, {start: frame}) one source file at a time.

    Windows are never held for more than one day, because the eligible windows
    of the whole collection would not fit in memory at 78 float64 columns per
    row.  The window id is derived from the bucket start rather than from a
    counter, so a subset run and the full run draw the same rows for the same
    window.
    """
    width = int(protocol["sampling"]["block_width_seconds"])
    n = int(protocol["sampling"]["rows_per_block"])
    for file_index, item in enumerate(protocol["source"]["files"]):
        path = DATA_DIR / item["name"]
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        ts = pd.to_datetime(df[TIMESTAMP], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        keep = ts.notna() & df[LABEL].astype(str).str.strip().ne("Label")
        df = df.loc[keep].copy()
        df["_ts"] = ts[keep]
        if not len(df):
            print(f"{item['name']}: no parsable rows, skipped", flush=True)
            continue
        day = str(df["_ts"].dt.date.astype(str).mode().iloc[0])
        if days is not None and day not in days:
            print(f"{item['name']}: day {day} not requested, skipped", flush=True)
            continue
        feature_cols = [c for c in df.columns if c not in (TIMESTAMP, LABEL, "_ts")]
        df = df.loc[~df.duplicated(subset=feature_cols, keep="first")]
        df["_bucket"] = df["_ts"].dt.floor(f"{width}s")
        df["_row"] = np.arange(len(df), dtype=np.int64) + file_index * 10_000_000
        sizes = df.groupby("_bucket").size()
        eligible = set(sizes[sizes >= n].index)
        df = df.loc[df["_bucket"].isin(eligible)]
        windows = {start: group.reset_index(drop=True) for start, group in df.groupby("_bucket")}
        print(f"{item['name']}: {day}, eligible windows {len(windows)}", flush=True)
        yield day, dict(sorted(windows.items()))
        del df, windows


def sample_block(window: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(window), size=n, replace=False)
    return window.iloc[np.sort(positions)].reset_index(drop=True)


def prepare_matrix(sample: pd.DataFrame) -> np.ndarray:
    cols = [c for c in sample.columns if c not in (TIMESTAMP, LABEL, "_ts", "_bucket", "_row")]
    parts = []
    for col in cols:
        values = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        median = float(np.median(values[finite])) if bool(finite.any()) else 0.0
        values[~finite] = median
        parts.append(values[:, None])
    raw = np.hstack(parts).astype(np.float64, copy=False)
    if raw.shape[1] != 78:
        raise RuntimeError(f"Expected 78 features, got {raw.shape[1]}")
    return minmax_scale(raw)


def normalized_labels(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = sample[LABEL].astype(str).str.strip()
    classes, encoded, counts = np.unique(labels.to_numpy(), return_inverse=True, return_counts=True)
    return encoded, classes, counts


def sample_identity_hash(sample: pd.DataFrame) -> str:
    text = "\n".join(str(int(r)) for r in sample["_row"])
    return sha256_bytes(text.encode("utf-8"))


def prediction_hash(labels: np.ndarray) -> str:
    return sha256_bytes(np.asarray(labels, dtype=np.int64).tobytes(order="C"))


def load_existing(path: Path, protocol_id: str, protocol_hash: str, force: bool) -> list[dict]:
    if force or not path.exists():
        return []
    existing = pd.read_csv(path)
    if len(existing):
        if set(existing["protocol_id"].astype(str)) != {protocol_id}:
            raise RuntimeError("Existing checkpoint belongs to a different protocol; use --force.")
        if set(existing["protocol_sha256"].astype(str)) != {protocol_hash}:
            raise RuntimeError("Existing checkpoint was produced under a different protocol file; use --force.")
    return existing.to_dict("records")


def finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def sign_test(values: np.ndarray, tol: float) -> dict:
    non_ties = values[np.abs(values) > tol]
    return {
        "wins": int(np.sum(values > tol)),
        "ties": int(np.sum(np.abs(values) <= tol)),
        "losses": int(np.sum(values < -tol)),
        "two_sided_exact_sign_p": (
            float(binomtest(int(np.sum(non_ties > 0)), len(non_ties), 0.5).pvalue)
            if len(non_ties) else None
        ),
    }


def build_summary(results: pd.DataFrame, protocol: dict, protocol_hash: str) -> tuple[dict, pd.DataFrame]:
    ok = results.loc[results["status"].eq("ok")].copy()
    methods = {}
    for method, group in ok.groupby("method", sort=True):
        methods[method] = {
            "windows": int(len(group)),
            "median_ARI_informative": finite_median(group.loc[group["n_true"] >= 2, "ARI"]),
            "median_rare_recall": finite_median(group["rare_recall"]),
            "windows_with_defined_rare_recall": int(pd.to_numeric(group["rare_recall"], errors="coerce").notna().sum()),
            "median_ACC": finite_median(group["ACC"]),
            "median_many_to_one_ACC": finite_median(group["many_to_one_ACC"]),
            "median_runtime_s": finite_median(group["runtime_s"]),
            "median_n_clusters": finite_median(group["n_clusters"]),
            "median_n_balls": finite_median(group["n_balls"]),
        }

    index = ["window_id", "day"]
    left = ok.loc[ok["method"].eq("ARP-GB")].set_index(index)
    right = ok.loc[ok["method"].eq("MG-GBC")].set_index(index)
    metrics = ["ARI", "NMI", "ACC", "rare_recall", "many_to_one_ACC", "runtime_s", "n_clusters", "n_balls"]
    paired = left[["n_true"] + metrics].join(
        right[metrics + ["n_rare_cores", "n_fragment_edges", "partition_agreement_ari",
                            "partition_identical"]], how="inner",
        lsuffix="_arp", rsuffix="_mg",
    ).reset_index()
    for metric in ["ARI", "NMI", "ACC", "rare_recall", "many_to_one_ACC"]:
        paired[f"delta_{metric}"] = paired[f"{metric}_mg"] - paired[f"{metric}_arp"]
    paired["runtime_ratio_mg_over_arp"] = paired["runtime_s_mg"] / paired["runtime_s_arp"]
    paired["identical_partition"] = paired["partition_identical"].astype(bool)
    paired["has_rare_class"] = pd.to_numeric(paired["rare_recall_arp"], errors="coerce").notna() | \
        pd.to_numeric(paired["rare_recall_mg"], errors="coerce").notna()

    tol = 1e-12
    rare = paired.loc[paired["has_rare_class"]]
    informative = paired.loc[paired["n_true"] >= 2]
    control = paired.loc[~paired["has_rare_class"]]

    rare_delta = pd.to_numeric(rare["delta_rare_recall"], errors="coerce").to_numpy(dtype=float)
    rare_delta = rare_delta[np.isfinite(rare_delta)]
    ari_delta = pd.to_numeric(informative["delta_ARI"], errors="coerce").to_numpy(dtype=float)

    per_day = {}
    for day, group in rare.groupby("day"):
        values = pd.to_numeric(group["delta_rare_recall"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        per_day[day] = {
            "windows_with_defined_rare_recall": int(len(values)),
            "median_delta_rare_recall": float(np.median(values)) if len(values) else None,
            **sign_test(values, tol),
        }

    agreement_rate = float(control["identical_partition"].mean()) if len(control) else None

    rare_sign = sign_test(rare_delta, tol)
    criteria = {
        "H1_mechanism": {
            "statistic": "median delta rare_recall over windows with a defined rare-recall",
            "value": float(np.median(rare_delta)) if len(rare_delta) else None,
            "sign_test": rare_sign,
            "met": bool(len(rare_delta) and np.median(rare_delta) > 0
                        and rare_sign["two_sided_exact_sign_p"] is not None
                        and rare_sign["two_sided_exact_sign_p"] < 0.05),
        },
        "H2_replication": {
            "statistic": "sign of the median delta rare_recall on each rare-bearing day",
            "per_day": per_day,
            "met": bool(len(per_day) >= 2 and all(
                (v["median_delta_rare_recall"] is not None and v["median_delta_rare_recall"] > 0)
                for v in per_day.values())),
        },
        "H3_locality": {
            "statistic": "share of control windows with an identical partition",
            "value": agreement_rate,
            "control_windows": int(len(control)),
            "threshold": 0.90,
            "met": bool(agreement_rate is not None and agreement_rate >= 0.90),
        },
        "H4_noninferiority": {
            "statistic": "median delta ARI over blocks with at least two true classes",
            "value": float(np.median(ari_delta)) if len(ari_delta) else None,
            "threshold": -0.005,
            "met": bool(len(ari_delta) and np.median(ari_delta) >= -0.005),
        },
    }

    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows_evaluated": int(ok["window_id"].nunique()),
        "methods": methods,
        "paired": {
            "windows": int(len(paired)),
            "windows_with_at_least_two_classes": int(len(informative)),
            "windows_with_defined_rare_recall": int(len(rare)),
            "control_windows_without_rare_class": int(len(control)),
            "median_delta_ARI_informative": finite_median(informative["delta_ARI"]),
            "median_delta_rare_recall": float(np.median(rare_delta)) if len(rare_delta) else None,
            "median_delta_many_to_one_ACC": finite_median(paired["delta_many_to_one_ACC"]),
            "median_runtime_ratio_mg_over_arp": finite_median(paired["runtime_ratio_mg_over_arp"]),
            "rare_core_activation_rate": float(pd.to_numeric(paired["n_rare_cores"], errors="coerce").fillna(0).gt(0).mean()),
            "identical_partition_rate_all_windows": float(paired["identical_partition"].mean()),
        },
        "criteria": criteria,
        "preregistration_caveat": protocol["pre_registration_caveat"],
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments" / "cicids2018_temporal_protocol_v1.json")
    parser.add_argument("--days", type=str, default=None,
                        help="comma-separated ISO days to include, e.g. 2018-02-22; default all nine")
    parser.add_argument("--tag", type=str, default="all")
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol, protocol_hash = load_protocol(args.protocol)
    verify_frozen_inputs(protocol)
    print(f"protocol verified: {protocol['protocol_id']} {protocol_hash}", flush=True)

    days = set(d.strip() for d in args.days.split(",")) if args.days else None
    out = ROOT / "results" / f"cicids2018_temporal_blocks_{args.tag}.csv"
    summary_path = ROOT / "results" / f"cicids2018_temporal_summary_{args.tag}.json"
    paired_path = ROOT / "results" / f"cicids2018_temporal_paired_{args.tag}.csv"
    records = load_existing(out, protocol["protocol_id"], protocol_hash, args.force)
    completed = {(int(r["window_id"]), str(r["method"])) for r in records}

    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    width = int(protocol["sampling"]["block_width_seconds"])
    k = int(protocol["models"]["k"])
    params = {k: v for k, v in protocol["models"]["mg_gbc_params"].items()
              if k != "rationale" and k != "reason"}
    seen = 0

    for day, windows in iter_day_windows(protocol, days):
        for start, window in windows.items():
            if args.limit_windows and seen >= args.limit_windows:
                break
            seen += 1
            window_id = int(start.value // 10**9) // width
            seed = base_seed + window_id
            sample = sample_block(window, n, seed)
            X = prepare_matrix(sample)
            y_true, classes, counts = normalized_labels(sample)
            common = {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_hash,
                "window_id": window_id,
                "day": day,
                "start_utc": start.tz_localize("UTC").isoformat() if start.tzinfo is None else start.isoformat(),
                "window_rows": int(len(window)),
                "sample_seed": seed,
                "feature_count": int(X.shape[1]),
                "sample_identity_sha256": sample_identity_hash(sample),
                "input_matrix_sha256": sha256_bytes(np.ascontiguousarray(X).tobytes()),
                "n_true": int(len(classes)),
                "min_true_class_count": int(counts.min()),
                "rare_true_classes": int(np.sum(counts < np.sqrt(n))),
            }
            methods = [
                ("ARP-GB", ARPGB(k=k, scale=False, size_gate=False)),
                ("MG-GBC", MGGBC(k=k, scale=False, size_gate=False, **params)),
            ]
            if all((window_id, name) in completed for name, _ in methods):
                continue
            # A window is scored as a pair: the partition comparison needs both
            # predictions, so a half-written window is recomputed rather than
            # resumed.
            records = [r for r in records if int(r["window_id"]) != window_id]
            window_rows = []
            predictions = {}
            for name, model in methods:
                started = time.perf_counter()
                row = dict(common, method=name)
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
                        status="ok",
                        error="",
                    )
                    predictions[name] = np.asarray(y_pred)
                    print(f"{day} w{window_id} {name}: "
                          f"ARI={row['ARI']:.4f} rare={row['rare_recall']:.4f} "
                          f"clusters={int(row['n_clusters'])} {runtime:.2f}s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    row.update(runtime_s=time.perf_counter() - started, status="error",
                               error=f"{type(exc).__name__}: {exc}")
                    print(f"{day} w{window_id} {name}: ERROR {row['error']}", flush=True)
                window_rows.append(row)
            if len(predictions) == 2:
                pair_ari = float(adjusted_rand_score(predictions["ARP-GB"], predictions["MG-GBC"]))
                cand = [r for r in window_rows if r["status"] == "ok"]
                agree = int(np.sum([np.array_equal(predictions["ARP-GB"], predictions["MG-GBC"])]))
                for row in cand:
                    row["partition_agreement_ari"] = pair_ari
                    row["partition_identical"] = bool(pair_ari >= 1.0 - 1e-12)
                    row["partition_identical_unrelabelled"] = bool(agree)
            else:
                for row in window_rows:
                    row["partition_agreement_ari"] = np.nan
                    row["partition_identical"] = False
                    row["partition_identical_unrelabelled"] = False
            records.extend(window_rows)
            pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(out, index=False, float_format="%.12g")

    results = pd.DataFrame(records, columns=RESULT_COLUMNS)
    summary, paired = build_summary(results, protocol, protocol_hash)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired.to_csv(paired_path, index=False, float_format="%.12g")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}\nwrote {summary_path}\nwrote {paired_path}")


if __name__ == "__main__":
    main()