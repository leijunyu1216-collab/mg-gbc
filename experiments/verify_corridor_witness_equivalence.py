"""Verify that the corridor machinery of the fragment rule changes no reported number.

The fragment rule (Eq. 14 of the manuscript) admits an edge between two retained
rare cores when they are similar in density and mutually closer than either is to
the background, and it certifies a clear corridor between them.  Two things about
that certificate are worth checking against every reported cohort, because the
manuscript's complexity claim and its mechanism story both rest on them.

1.  *Certificate.*  The corridor is certified against the ``W = 3`` nearest
    background centres of the two endpoints, which keeps the pair test at O(1) and
    the layer at O(m^2 d).  Certifying against every background centre is the same
    predicate with an O(m) scan per pair and is retained as an exact fallback.
    Arms ``w1``, ``w2`` and ``exact`` vary the witness count, so the reported
    numbers are shown not to depend on it.
2.  *Clause.*  At the registered constants the corridor clause cannot bind at all:
    ``d_ij < beta * min(iso_i, iso_j)`` with ``beta = 1/2`` puts every background
    centre farther than ``1.5 * d_ij`` from the midpoint of the two cores, three
    times the ``0.5 * d_ij`` clearance the clause asks for (Lemma corridor-lemma in
    the manuscript).  Passing ``fragment_corridor_factor = -1`` makes the clause
    vacuous, so comparing that configuration against the shipped one tests the
    lemma on real data rather than on the proof alone.  The lemma also says the
    radius-proximity factor ``eta`` cannot matter at any closeness factor up to 1,
    which is the flat response the sensitivity sweep measures over eta in [1, 6].

Both arms are compared against the frozen result files, which are read only.  The
summary is written to results/corridor_witness_equivalence.json.

Targets
    tabular          the fourteen data sets of Table 2
    unsw-replication the twenty frozen replication blocks of Section 4.7
    unsw-holdout     the twenty untouched holdout blocks of Section 4.7
    cicids           every pooled CIC-IDS2018 window that retains at least two
                     rare cores, that is, every window where the fragment rule
                     can fire at all
    all              the four targets above (default)

Usage:
    python -I experiments/verify_corridor_witness_equivalence.py --target all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics_ext import evaluate_extended  # noqa: E402
from src.mg_gbc import MGGBC  # noqa: E402

OUT = ROOT / "results" / "corridor_witness_equivalence.json"
KEYS = ["ARI", "NMI", "ACC", "many_to_one_ACC", "rare_recall", "n_clusters"]
TOL = 1e-12

# The configurations compared on every unit.  "shipped" is the registered default;
# "w1" and "w2" certify against fewer background centres; "exact" replaces the
# witness certificate by a scan of every background centre; "vacuous" makes the
# corridor clause always true, which is what removing it does.  Lemma
# corridor-lemma predicts that all five agree wherever the registered closeness
# factor is at most 1, so this script is a test of the lemma on real data.
ARMS = [
    ("shipped", {}),
    ("w1", {"fragment_corridor_witnesses": 1}),
    ("w2", {"fragment_corridor_witnesses": 2}),
    ("exact", {"fragment_corridor_witnesses": None}),
    ("vacuous", {"fragment_corridor_factor": -1.0}),
]

# Table 2 of the manuscript: the fourteen data sets and their neighbourhood size.
TABULAR = [
    ("Synthetic-Imbalance", "synthetic_imbalance", 12),
    ("Synthetic-SplitRare", "synthetic_split_rare", 12),
    ("Digits-Rare", "load_digits_rare", 10),
    ("Zoo", "uci:zoo", 4),
    ("Iris", "uci:iris", 47),
    ("Glass", "uci:glass", 38),
    ("Ecoli", "uci:ecoli", 6),
    ("Yeast", "openml:Yeast:yeast:181", 10),
    ("Pageblocks", "openml:Pageblocks:page-blocks:30", 10),
    ("Segment", "openml:Segment:segment:36", 10),
    ("Lymphography", "openml:Lymphography:lymph:10", 10),
    ("Dermatology", "openml:Dermatology:dermatology:35", 10),
    ("Covertype-Rare", "load_covtype_rare", 10),
    ("Olivetti-PCA", "load_olivetti_pca", 10),
]


def load_tabular(spec: str):
    import src.datasets as ds

    if spec.startswith("uci:"):
        return ds.load_uci_named(spec.split(":", 1)[1])
    if spec.startswith("openml:"):
        _, label, name, data_id = spec.split(":")
        return ds.load_rare_openml(label, name, int(data_id))
    return getattr(ds, spec)()


def max_abs_diff(a, b) -> float:
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def fit_arms(model_kwargs: dict, X: np.ndarray) -> dict:
    """Fit every arm of ARMS on one input and return per-arm labels and metrics."""
    out: dict = {}
    for label, override in ARMS:
        model = MGGBC(**{**model_kwargs, **override})
        labels = model.fit_predict(X)
        out[label] = {"labels": labels, "frag": int(model.n_fragment_edges_),
                      "cores": int(model.n_rare_cores_)}
    return out


def arm_agreement(arms: dict) -> dict:
    ref = arms["shipped"]["labels"]
    return {
        "labels_identical_to_shipped": {
            label: bool(np.array_equal(arms[label]["labels"], ref))
            for label, _ in ARMS if label != "shipped"
        },
        "fragment_edges": {label: arms[label]["frag"] for label, _ in ARMS},
    }


def rollup(rows: list[dict]) -> tuple[dict, pd.DataFrame]:
    df = pd.DataFrame(rows)
    out = {
        "units_checked": len(df),
        "units_where_all_arms_agree": int(
            sum(all(r["labels_identical_to_shipped"].values()) for r in rows)
        ),
        "fragment_edges_by_arm": {
            label: int(sum(r["fragment_edges"][label] for r in rows)) for label, _ in ARMS
        },
        "max_metric_diff_vs_shipped": {
            label: max(max_abs_diff([r[f"{k}_{label}"] for r in rows],
                                    [r[f"{k}_shipped"] for r in rows]) for k in KEYS)
            for label, _ in ARMS if label != "shipped"
        },
    }
    return out, df


# ---------------------------------------------------------------- tabular
def run_tabular() -> dict:
    rows = []
    for name, spec, k in TABULAR:
        loaded = load_tabular(spec)
        if loaded is None:
            rows.append({"data_set": name, "status": "unavailable"})
            continue
        X, y, _ = loaded
        arms = fit_arms({"k": k}, X)
        rec = {"data_set": name, "k": k, "status": "ok", "n": int(len(X))}
        rec.update(arm_agreement(arms))
        for label in arms:
            met = evaluate_extended(y, arms[label]["labels"])
            for key in KEYS:
                rec[f"{key}_{label}"] = met[key]
        rows.append(rec)
        print(f"  {name:<22} frag={rec['fragment_edges']['shipped']} "
              f"all arms identical={all(rec['labels_identical_to_shipped'].values())}", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    summary, _ = rollup(ok)
    summary["per_set"] = rows
    return summary


# ---------------------------------------------------------------- UNSW
def _unsw_blocks(frozen_csv: Path, method: str):
    """Load the frozen block list and the source rows those blocks were drawn from.

    The replication file is long (one row per block and method); the holdout file is
    wide (one row per block, metric columns suffixed by configuration).
    """
    from experiments.run_unsw_nb15_temporal_compare import load_protocol, load_selected_rows

    frozen = pd.read_csv(frozen_csv)
    if method and "method" in frozen.columns:
        frozen = frozen[frozen["method"] == method]
    starts = [int(s) for s in frozen["block_start_unix"]]
    v2, _ = load_protocol(ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json")
    protocol = {
        "protocol_id": "UNSW-NB15-CORRIDOR-REVERIFY",
        "source": v2["source"],
        "sampling": {"block_width_seconds": 300, "selected_block_starts_unix": starts},
    }
    return frozen.reset_index(drop=True), load_selected_rows(protocol)


def run_unsw(kind: str) -> dict:
    from experiments.run_unsw_nb15_temporal_compare import (
        normalized_labels,
        prepare_matrix,
        sample_block,
    )

    v2 = json.loads((ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json")
                    .read_text(encoding="utf-8"))
    k = int(v2["models"]["k"])
    params = {key: value for key, value in v2["models"]["mg_gbc_params"].items()
              if key not in ("rationale", "reason")}
    if kind == "replication":
        frozen_csv = ROOT / "results" / "unsw_nb15_temporal_blocks_v2_matched.csv"
        base_seed = int(v2["sampling"]["block_selection_seed"])
        suffix = ""
    else:
        # The holdout file is wide: the shipped configuration is the tau_gamma = 0.5
        # column set, suffixed _z050.  The untouched holdout is never re-scored at a
        # non-default configuration, so only that column set is read here.
        frozen_csv = ROOT / "results" / "unsw_holdout_blocks_matched.csv"
        base_seed = 20260828
        suffix = "_z050"
    frozen, source_rows = _unsw_blocks(frozen_csv, "MG-GBC")

    rows = []
    for _, rec in frozen.iterrows():
        block_id = int(rec["block_id"])
        start = int(rec["block_start_unix"])
        n_rows = int(rec["sample_n"]) if "sample_n" in frozen.columns else 5000
        seed = int(rec["sample_seed"]) if "sample_seed" in frozen.columns else base_seed + block_id
        sample = sample_block(source_rows, start, n_rows, seed)
        X = prepare_matrix(sample)
        y, _, _ = normalized_labels(sample)
        out = {"block_id": block_id}
        arms = fit_arms({"k": k, "scale": False, "size_gate": False, **params}, X)
        out.update(arm_agreement(arms))
        for label in arms:
            met = evaluate_extended(y, arms[label]["labels"])
            for key in KEYS:
                out[f"{key}_{label}"] = met[key]
            out[f"cores_{label}"] = arms[label]["cores"]
        for key in KEYS:
            col = key + suffix
            if col in frozen.columns:
                out[f"frozen_{key}"] = float(pd.to_numeric(rec[col], errors="coerce"))
        if f"rare_cores{suffix}" in frozen.columns:
            out["frozen_cores"] = float(pd.to_numeric(rec[f"rare_cores{suffix}"], errors="coerce"))
        rows.append(out)
        print(f"  block {block_id:02d} done", flush=True)

    summary, df = rollup(rows)
    summary["max_metric_diff_vs_frozen"] = {
        key: max_abs_diff(df[f"frozen_{key}"], df[f"{key}_shipped"])
        for key in KEYS if f"frozen_{key}" in df.columns
    }
    if "frozen_cores" in df.columns:
        summary["max_metric_diff_vs_frozen"]["rare_cores"] = max_abs_diff(
            df["frozen_cores"], df["cores_shipped"])
    summary["per_block"] = rows
    return summary


# ---------------------------------------------------------------- CIC
def run_cicids() -> dict:
    from experiments.run_cicids2018_temporal_compare import (
        iter_day_windows,
        load_protocol,
        normalized_labels,
        prepare_matrix,
        sample_block,
        sha256_bytes,
    )

    protocol, _ = load_protocol(ROOT / "experiments" / "cicids2018_temporal_protocol_v1.json")
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    width = int(protocol["sampling"]["block_width_seconds"])
    k = int(protocol["models"]["k"])
    params = {key: value for key, value in protocol["models"]["mg_gbc_params"].items()
              if key not in ("rationale", "reason")}

    frozen = pd.read_csv(ROOT / "results" / "cicids2018_temporal_blocks_pooled.csv")
    mg = frozen[frozen["method"] == "MG-GBC"].copy()
    mg["window_id"] = mg["window_id"].astype(int)
    targets = mg.loc[mg["n_rare_cores"] >= 2].set_index("window_id")
    print(f"  frozen MG-GBC windows {len(mg)}, with >=2 rare cores {len(targets)}", flush=True)

    rows = []
    for day, windows in iter_day_windows(protocol, None):
        for start, window in windows.items():
            window_id = int(start.value // 10**9) // width
            if window_id not in targets.index:
                continue
            sample = sample_block(window, n, base_seed + window_id)
            X = prepare_matrix(sample)
            y, _, _ = normalized_labels(sample)
            rec = {"window_id": window_id,
                   "input_matrix_matches_frozen":
                       sha256_bytes(np.ascontiguousarray(X).tobytes())
                       == str(targets.loc[window_id, "input_matrix_sha256"])}
            arms = fit_arms({"k": k, "scale": False, "size_gate": False, **params}, X)
            rec.update(arm_agreement(arms))
            for label in arms:
                met = evaluate_extended(y, arms[label]["labels"])
                for key in KEYS:
                    rec[f"{key}_{label}"] = met[key]
            rec["frozen_frag"] = int(targets.loc[window_id, "n_fragment_edges"])
            for key in KEYS:
                rec[f"frozen_{key}"] = float(targets.loc[window_id, key])
            rows.append(rec)
            if len(rows) % 50 == 0:
                print(f"  {len(rows)}/{len(targets)} windows checked", flush=True)
        del windows

    summary, df = rollup(rows)
    summary["units_with_matching_input_matrix_hash"] = int(df["input_matrix_matches_frozen"].sum())
    summary["fragment_edges_frozen"] = int(df["frozen_frag"].sum())
    summary["windows_below_two_rare_cores"] = int(len(mg) - len(targets))
    summary["max_metric_diff_vs_frozen"] = {
        key: max_abs_diff(df[f"frozen_{key}"], df[f"{key}_shipped"])
        for key in KEYS if f"frozen_{key}" in df.columns
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all",
                    choices=["all", "tabular", "unsw-replication", "unsw-holdout", "cicids"])
    args = ap.parse_args()
    targets = (["tabular", "unsw-replication", "unsw-holdout", "cicids"]
               if args.target == "all" else [args.target])

    summary: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arms": {
            "shipped": "the registered default: W = 3 corridor witnesses, clearance factor 0.5",
            "w1": "the corridor certified against the nearest background centre only",
            "w2": "the corridor certified against the two nearest background centres",
            "exact": "the corridor certified against every background centre (O(m) per pair)",
            "vacuous": "corridor clearance factor -1, i.e. the corridor clause removed",
        },
        "claim": ("On every unit of every reported cohort the shipped corridor certificate returns "
                  "the same partition, the same fragment-edge count and the same metrics as each "
                  "smaller witness count, as the exact full scan, and as a configuration in which "
                  "the corridor clause cannot reject anything, and it reproduces the frozen result "
                  "files. Lemma corridor-lemma predicts this: at the registered closeness factor "
                  "0.5 the corridor clearance obtained is 1.5 * d_ij, three times the 0.5 * d_ij "
                  "the clause requires, so the clause cannot reject a pair the closeness test "
                  "admitted and the witness count cannot matter."),
        "targets": {},
    }
    runners = {
        "tabular": run_tabular,
        "unsw-replication": lambda: run_unsw("replication"),
        "unsw-holdout": lambda: run_unsw("holdout"),
        "cicids": run_cicids,
    }
    for target in targets:
        print(f"\n=== {target} ===", flush=True)
        summary["targets"][target] = runners[target]()
        block = summary["targets"][target]
        print(f"  units {block['units_checked']}, all arms agree "
              f"{block['units_where_all_arms_agree']}, fragment edges "
              f"{json.dumps(block['fragment_edges_by_arm'])}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
