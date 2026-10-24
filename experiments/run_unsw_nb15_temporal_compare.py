from __future__ import annotations

import argparse
import hashlib
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
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arp_gb import ARPGB
from src.core import minmax_scale
from src.metrics import evaluate
from src.mg_gbc import MGGBC


COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smean", "dmean", "trans_depth", "response_body_len",
    "sjit", "djit", "stime", "ltime", "sinpkt", "dinpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]
CATEGORICAL = ["proto", "state", "service"]
EXCLUDED = ["srcip", "dstip", "stime", "ltime", "attack_cat", "label"]
FEATURES = [c for c in COLUMNS if c not in EXCLUDED]
RESULT_COLUMNS = [
    "protocol_id", "protocol_sha256", "block_id", "block_start_unix",
    "block_end_unix", "block_start_utc", "available_rows", "sample_seed",
    "sample_n", "feature_count", "sample_identity_sha256",
    "input_matrix_sha256", "prediction_sha256", "duplicate_extra_rows",
    "duplicate_rate", "n_true", "min_true_class_count", "rare_true_classes",
    "method", "runtime_s", "ACC", "NMI", "ARI", "rare_recall",
    "n_clusters", "n_balls", "n_rare_cores", "n_fragment_edges",
    "status", "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def verify_frozen_inputs(protocol: dict) -> None:
    mismatches: list[str] = []
    for item in protocol["source"]["files"]:
        path = ROOT / "data" / "unsw_nb15" / item["name"]
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


def load_selected_rows(protocol: dict) -> pd.DataFrame:
    starts = set(protocol["sampling"]["selected_block_starts_unix"])
    width = int(protocol["sampling"]["block_width_seconds"])
    frames: list[pd.DataFrame] = []
    data_dir = ROOT / "data" / "unsw_nb15"
    shard_names = [x["name"] for x in protocol["source"]["files"] if x["name"].startswith("UNSW-NB15_")]
    for shard_name in shard_names:
        path = data_dir / shard_name
        offset = 0
        kept = 0
        for chunk in pd.read_csv(
            path,
            header=None,
            names=COLUMNS,
            chunksize=100_000,
            low_memory=False,
        ):
            stime = pd.to_numeric(chunk["stime"], errors="coerce")
            bucket = (stime // width * width).astype("Int64")
            mask = bucket.isin(starts)
            if bool(mask.any()):
                selected = chunk.loc[mask].copy()
                selected["source_file"] = shard_name
                selected["source_row"] = np.arange(offset, offset + len(chunk), dtype=np.int64)[mask.to_numpy()]
                frames.append(selected)
                kept += len(selected)
            offset += len(chunk)
        print(f"loaded {shard_name}: rows={offset:,}, selected={kept:,}", flush=True)
    if not frames:
        raise RuntimeError("No rows matched the frozen time blocks.")
    return pd.concat(frames, ignore_index=True)


def prepare_matrix(sample: pd.DataFrame) -> np.ndarray:
    parts: list[np.ndarray] = []
    for col in FEATURES:
        if col in CATEGORICAL:
            values = sample[[col]].fillna("<missing>").astype(str).apply(lambda s: s.str.strip())
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            parts.append(encoder.fit_transform(values))
        else:
            values = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            median = float(np.median(values[finite])) if bool(finite.any()) else 0.0
            values[~finite] = median
            parts.append(values[:, None])
    raw = np.hstack(parts).astype(np.float64, copy=False)
    if raw.shape[1] != 43:
        raise RuntimeError(f"Expected 43 features, got {raw.shape[1]}")
    return minmax_scale(raw)


def normalized_labels(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = sample["attack_cat"].fillna("Normal").astype(str).str.strip()
    labels = labels.mask(labels.eq(""), "Normal")
    classes, encoded, counts = np.unique(labels.to_numpy(), return_inverse=True, return_counts=True)
    return encoded, classes, counts


def sample_block(rows: pd.DataFrame, start: int, n: int, seed: int) -> pd.DataFrame:
    stime = pd.to_numeric(rows["stime"], errors="coerce")
    eligible = rows.loc[(stime >= start) & (stime < start + 300)]
    if len(eligible) < n:
        raise RuntimeError(f"Block {start} has {len(eligible)} rows, fewer than {n}.")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(eligible), size=n, replace=False)
    return eligible.iloc[positions].reset_index(drop=True)


def sample_identity_hash(sample: pd.DataFrame) -> str:
    text = "\n".join(f"{f}:{int(r)}" for f, r in zip(sample["source_file"], sample["source_row"]))
    return sha256_bytes(text.encode("utf-8"))


def prediction_hash(labels: np.ndarray) -> str:
    return sha256_bytes(np.asarray(labels, dtype=np.int64).tobytes(order="C"))


def exact_duplicate_stats(sample: pd.DataFrame) -> tuple[int, float]:
    extra = int(sample[COLUMNS].duplicated(keep="first").sum())
    return extra, float(extra / len(sample))


def load_existing(path: Path, protocol_id: str, protocol_hash: str, force: bool) -> pd.DataFrame:
    if force or not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    existing = pd.read_csv(path)
    if len(existing):
        ids = set(existing["protocol_id"].astype(str))
        hashes = set(existing["protocol_sha256"].astype(str))
        if ids != {protocol_id} or hashes != {protocol_hash}:
            raise RuntimeError("Existing checkpoint belongs to a different protocol; use --force to replace it.")
    return existing


def save_checkpoint(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def build_summary(results: pd.DataFrame, protocol: dict, protocol_hash: str) -> tuple[dict, pd.DataFrame]:
    ok = results.loc[results["status"].eq("ok")].copy()
    methods = {}
    for method, group in ok.groupby("method", sort=True):
        methods[method] = {
            "blocks": int(len(group)),
            "median_ARI": finite_median(group["ARI"]),
            "median_NMI": finite_median(group["NMI"]),
            "median_ACC": finite_median(group["ACC"]),
            "median_rare_recall": finite_median(group["rare_recall"]),
            "median_runtime_s": finite_median(group["runtime_s"]),
            "median_n_clusters": finite_median(group["n_clusters"]),
            "median_n_balls": finite_median(group["n_balls"]),
        }
    metric_cols = ["ARI", "NMI", "ACC", "rare_recall", "runtime_s", "n_clusters", "n_balls"]
    left = ok.loc[ok["method"].eq("ARP-GB"), ["block_id", "n_true"] + metric_cols].set_index("block_id")
    right_cols = metric_cols + ["n_rare_cores", "n_fragment_edges"]
    right = ok.loc[ok["method"].eq("MG-GBC"), ["block_id"] + right_cols].set_index("block_id")
    paired = left.join(right, how="inner", lsuffix="_arp", rsuffix="_mg").reset_index()
    for metric in ["ARI", "NMI", "ACC", "rare_recall"]:
        paired[f"delta_{metric}"] = paired[f"{metric}_mg"] - paired[f"{metric}_arp"]
    paired["runtime_ratio_mg_over_arp"] = paired["runtime_s_mg"] / paired["runtime_s_arp"]
    tol = float(protocol["labels_and_metrics"]["tie_tolerance"])
    ari_delta = paired["delta_ARI"].to_numpy(dtype=float)
    active = pd.to_numeric(paired["n_rare_cores"], errors="coerce").fillna(0).gt(0)
    active_rare_delta = paired.loc[active, "delta_rare_recall"]
    informative = pd.to_numeric(paired["n_true"], errors="coerce").gt(1)
    informative_rows = paired.loc[informative]
    informative_ari = informative_rows["delta_ARI"].to_numpy(dtype=float)
    informative_rare = pd.to_numeric(informative_rows["delta_rare_recall"], errors="coerce").to_numpy(dtype=float)
    informative_rare = informative_rare[np.isfinite(informative_rare)]
    ari_non_ties = informative_ari[np.abs(informative_ari) > tol]
    rare_non_ties = informative_rare[np.abs(informative_rare) > tol]
    comparison = {
        "paired_blocks": int(len(paired)),
        "median_delta_ARI": finite_median(paired["delta_ARI"]),
        "median_delta_NMI": finite_median(paired["delta_NMI"]),
        "median_delta_ACC": finite_median(paired["delta_ACC"]),
        "median_delta_rare_recall": finite_median(paired["delta_rare_recall"]),
        "ARI_wins": int(np.sum(ari_delta > tol)),
        "ARI_ties": int(np.sum(np.abs(ari_delta) <= tol)),
        "ARI_losses": int(np.sum(ari_delta < -tol)),
        "median_runtime_ratio_mg_over_arp": finite_median(paired["runtime_ratio_mg_over_arp"]),
        "rare_core_activation_blocks": int(active.sum()),
        "rare_core_activation_rate": float(active.mean()) if len(active) else None,
        "median_delta_rare_recall_when_rare_core_active": finite_median(active_rare_delta),
    }
    sensitivity = {
        "definition": "Blocks containing at least two true classes; labels are used only after frozen sampling, so this is a labeled sensitivity analysis rather than a replacement sampling rule.",
        "single_class_blocks": int((~informative).sum()),
        "multi_class_blocks": int(informative.sum()),
        "median_delta_ARI": finite_median(informative_rows["delta_ARI"]),
        "median_delta_NMI": finite_median(informative_rows["delta_NMI"]),
        "median_delta_ACC": finite_median(informative_rows["delta_ACC"]),
        "median_delta_rare_recall": finite_median(informative_rows["delta_rare_recall"]),
        "ARI_wins": int(np.sum(informative_ari > tol)),
        "ARI_ties": int(np.sum(np.abs(informative_ari) <= tol)),
        "ARI_losses": int(np.sum(informative_ari < -tol)),
        "ARI_two_sided_exact_sign_p": float(binomtest(int(np.sum(ari_non_ties > 0)), len(ari_non_ties), 0.5).pvalue) if len(ari_non_ties) else None,
        "ARI_noninferior_blocks_at_minus_0_005": int(np.sum(informative_ari >= -0.005)),
        "ARI_noninferior_fraction_at_minus_0_005": float(np.mean(informative_ari >= -0.005)) if len(informative_ari) else None,
        "ARI_losses_below_minus_0_05": int(np.sum(informative_ari < -0.05)),
        "rare_recall_wins": int(np.sum(informative_rare > tol)),
        "rare_recall_ties": int(np.sum(np.abs(informative_rare) <= tol)),
        "rare_recall_losses": int(np.sum(informative_rare < -tol)),
        "rare_recall_two_sided_exact_sign_p": float(binomtest(int(np.sum(rare_non_ties > 0)), len(rare_non_ties), 0.5).pvalue) if len(rare_non_ties) else None,
    }
    conclusion = {
        "scope": "One UNSW-NB15 real-world benchmark result based on 20 paired temporal blocks.",
        "cross_source_claim": "Not evaluated; blocks are repeated measures, not independent data sources.",
        "direction": (
            "MG-GBC has positive median paired ARI on UNSW-NB15."
            if comparison["median_delta_ARI"] is not None and comparison["median_delta_ARI"] > 0
            else "MG-GBC does not have positive median paired ARI on UNSW-NB15."
        ),
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "paired_comparison_mg_minus_arp": comparison,
        "multi_class_block_sensitivity": sensitivity,
        "interpretation": conclusion,
        "data_quality": protocol["observed_source_profile_before_modeling"],
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments" / "unsw_nb15_temporal_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_blocks.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_summary.json")
    parser.add_argument("--paired", type=Path, default=ROOT / "results" / "unsw_nb15_temporal_paired.csv")
    parser.add_argument("--limit-blocks", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol, protocol_hash = load_protocol(args.protocol)
    verify_frozen_inputs(protocol)
    print(f"protocol verified: {protocol['protocol_id']} {protocol_hash}", flush=True)
    all_selected = protocol["sampling"]["selected_block_starts_unix"]
    selected_starts = all_selected[: args.limit_blocks] if args.limit_blocks else all_selected
    source_rows = load_selected_rows(protocol)

    existing = load_existing(args.output, protocol["protocol_id"], protocol_hash, args.force)
    records = existing.to_dict("records")
    completed = {(int(r["block_id"]), str(r["method"])) for r in records}
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    k = int(protocol["models"]["k"])

    for block_id, start in enumerate(selected_starts, start=1):
        seed = base_seed + block_id
        sample = sample_block(source_rows, int(start), n, seed)
        available = int(((pd.to_numeric(source_rows["stime"], errors="coerce") >= start) & (pd.to_numeric(source_rows["stime"], errors="coerce") < start + 300)).sum())
        X = prepare_matrix(sample)
        # Labels are deliberately accessed only after sampling and feature freezing.
        y_true, classes, counts = normalized_labels(sample)
        duplicate_extra, duplicate_rate = exact_duplicate_stats(sample)
        identity_hash = sample_identity_hash(sample)
        matrix_hash = sha256_bytes(np.ascontiguousarray(X).tobytes())
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
            "sample_identity_sha256": identity_hash,
            "input_matrix_sha256": matrix_hash,
            "duplicate_extra_rows": duplicate_extra,
            "duplicate_rate": duplicate_rate,
            "n_true": int(len(classes)),
            "min_true_class_count": int(counts.min()),
            "rare_true_classes": int(np.sum(counts < np.sqrt(n))),
        }
        methods = [("ARP-GB", ARPGB(k=k, scale=False)), ("MG-GBC", MGGBC(k=k, scale=False))]
        for method_name, model in methods:
            if (block_id, method_name) in completed:
                print(f"block {block_id:02d}/20 {method_name}: checkpoint exists", flush=True)
                continue
            started = time.perf_counter()
            row = dict(common)
            row["method"] = method_name
            try:
                y_pred = model.fit_predict(X)
                runtime = time.perf_counter() - started
                row.update(evaluate(y_true, y_pred))
                row.update(
                    runtime_s=runtime,
                    prediction_sha256=prediction_hash(y_pred),
                    n_balls=int(model.n_balls_),
                    n_rare_cores=int(getattr(model, "n_rare_cores_", 0) or 0),
                    n_fragment_edges=int(getattr(model, "n_fragment_edges_", 0) or 0),
                    status="ok",
                    error="",
                )
                print(
                    f"block {block_id:02d}/20 {method_name}: "
                    f"ARI={row['ARI']:.6f} NMI={row['NMI']:.6f} "
                    f"ACC={row['ACC']:.6f} rare={row['rare_recall']:.6f} "
                    f"clusters={int(row['n_clusters'])} time={runtime:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                row.update(
                    runtime_s=time.perf_counter() - started,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                print(f"block {block_id:02d}/20 {method_name}: ERROR {row['error']}", flush=True)
            records.append(row)
            save_checkpoint(records, args.output)

    results = pd.DataFrame(records, columns=RESULT_COLUMNS)
    summary, paired = build_summary(results, protocol, protocol_hash)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired.to_csv(args.paired, index=False, float_format="%.12g")
    print(json.dumps(summary["paired_comparison_mg_minus_arp"], indent=2), flush=True)


if __name__ == "__main__":
    main()
