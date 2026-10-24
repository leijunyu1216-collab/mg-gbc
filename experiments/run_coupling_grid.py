"""Joint sensitivity of the fragment constants (delta, eta, beta).

Major comment 2 of the review asks whether the three fragment constants
catastrophically fail when they interact, on high-dimensional real data.  The
grid is pre-registered in ``experiments/coupling_grid_protocol.json``; this
script only evaluates it.

Design in one paragraph.  The three constants enter ``rare_fragment_edges``
and nothing else, so for a fixed data unit the ball state (balls, overlap,
rare cores, densities, centres, radii) is computed once and every grid cell
re-runs only the fragment rule, the absorption guard, label propagation and
boundary assignment.  A deterministic subset of cells is re-run through the
shipped ``MGGBC.fit_predict``; the label sha256 must agree or the run aborts.
Cohorts: the 20 validation blocks of the UNSW-NB15 v2 replication protocol
(43 features), the ten development sets, and Covertype-Rare (54 features).
The untouched holdout blocks are never evaluated.

Outputs
-------
results/coupling_grid_raw.csv        long table, one row per (cell, unit)
results/coupling_grid_summary.json   aggregates, hypotheses, margins
results/coupling_grid_table.csv      the paper table
results/coupling_margins.csv         per-pair decision margins
results/coupling_grid_heatmap.png/pdf
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.check_guard_paper_regression import paper_datasets  # noqa: E402
from experiments.unsw_block_cache import (  # noqa: E402
    get_block,
    get_protocol,
    get_source_rows,
)
from src.core import (  # noqa: E402
    assign_boundary_points,
    generate_granular_balls,
    knn_density,
    minmax_scale,
    pairwise_overlap_mask,
    propagate_gb_labels,
    reduce_granular_balls,
    select_representatives,
)
from src.datasets import experiment_k, load_covtype_rare  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.mg_gbc import (  # noqa: E402
    MGGBC,
    _redistribute_dropped,
    ball_mean_density,
    protect_rare_cores,
    rare_core_mask,
    rare_fragment_edges,
    should_propagate,
)

OUT = ROOT / "results"
PROTOCOL = ROOT / "experiments" / "coupling_grid_protocol.json"
UNSW_V2 = ROOT / "experiments" / "unsw_nb15_temporal_protocol_v2.json"

DELTAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
ETAS = (1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50)
BETAS = (0.30, 0.40, 0.50, 0.60, 0.70)
D0, E0, B0 = 0.25, 1.50, 0.50

# Secondary slice: the density screen changes the core set, hence the fragment
# opportunities.  eta and beta stay at their defaults.
TAU_GAMMAS = (0.25, 0.50, 0.75, 1.00, 1.25)
TAU_ISO = 0.0
S_MIN = 3


@dataclass
class Unit:
    cohort: str
    name: str
    X: np.ndarray
    y: np.ndarray
    k: int


@dataclass
class State:
    X: np.ndarray
    balls: list
    overlap: np.ndarray
    gamma: np.ndarray
    rare: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    tau_gamma: float


def label_hash(labels: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_state(X: np.ndarray, k: int, tau_gamma: float) -> State:
    """The whole pipeline up to the point where the fragment rule starts.

    This is line-for-line the prefix of ``MGGBC.fit_predict`` with the shipped
    defaults for everything except ``tau_gamma``.
    """
    X = minmax_scale(np.asarray(X, dtype=float))
    n = len(X)
    rho, _ = knn_density(X, k)
    reps = select_representatives(X, rho)
    balls = generate_granular_balls(X, reps, dissolve_small=False)
    sizes = np.array([b.size for b in balls], dtype=float)
    gamma = ball_mean_density(rho, balls)
    centers = np.stack([b.center for b in balls])
    rare = rare_core_mask(sizes, gamma, centers, n, tau_gamma, TAU_ISO, S_MIN)

    small = sizes < np.sqrt(n)
    drop = (small & ~rare) if rare.any() else small
    balls = _redistribute_dropped(X, balls, drop)
    balls = reduce_granular_balls(X, balls)

    overlap = pairwise_overlap_mask(balls)
    gamma = ball_mean_density(rho, balls)
    sizes = np.array([b.size for b in balls], dtype=float)
    centers = np.stack([b.center for b in balls])
    radii = np.array([b.radius for b in balls], dtype=float)
    rare = rare_core_mask(sizes, gamma, centers, n, tau_gamma, TAU_ISO, S_MIN)
    return State(X, balls, overlap, gamma, rare, centers, radii, tau_gamma)


def run_tail(state: State, delta: float, eta: float, beta: float,
             guard: bool = False):
    """The suffix of ``MGGBC.fit_predict`` for one (delta, eta, beta) cell.

    ``guard`` applies the admissibility condition of the signed edit: a
    fragment edge may restore contact that radius trimming removed, but it may
    not overturn a merge the overlap test refused.  With ``guard=False`` the
    condition is left out, which is the ablation of that condition.
    """
    if state.rare.any():
        fragments = rare_fragment_edges(
            state.centers,
            state.radii,
            state.gamma,
            state.rare,
            delta,
            eta,
            beta,
            beta,
        )
        if guard:
            n = len(state.X)
            for i, j in zip(*np.nonzero(np.triu(fragments & state.overlap))):
                if not should_propagate(state.X, state.balls, state.overlap,
                                        int(i), int(j), n, size_gate=False):
                    fragments[i, j] = fragments[j, i] = False
        allow = state.overlap | fragments
        allow = protect_rare_cores(allow, state.rare, state.gamma)
        allow = allow | fragments
    else:
        fragments = np.zeros((len(state.gamma), len(state.gamma)), dtype=bool)
        allow = None
    gb = propagate_gb_labels(
        state.X,
        state.balls,
        state.overlap,
        allow_merge=allow,
        extra_edges=fragments,
        size_gate=False,
    )
    labels = assign_boundary_points(state.X, state.balls, gb)
    return labels, int(fragments.sum() // 2)


def load_units() -> list[Unit]:
    units: list[Unit] = []

    protocol, _ = get_protocol(UNSW_V2)
    rows = get_source_rows(protocol)
    for b in range(1, int(protocol["sampling"]["selected_blocks"]) + 1):
        _, X, y, _, _ = get_block(protocol, rows, b)
        units.append(Unit("unsw_v2", f"block-{b:02d}", X, y, 10))

    for name, X, y, k in paper_datasets():
        units.append(Unit("dev", name, np.asarray(X, float), np.asarray(y), int(k)))

    X, y, name = load_covtype_rare(seed=0)
    units.append(Unit("covtype", name, np.asarray(X, float), np.asarray(y),
                      int(experiment_k(name, len(X)))))
    return units


def main_grid_cells():
    for d in DELTAS:
        for e in ETAS:
            for b in BETAS:
                yield d, e, b


def metrics(y: np.ndarray, labels: np.ndarray) -> dict:
    met = evaluate(y, labels)
    return {
        "ARI": float(met["ARI"]),
        "ACC": float(met["ACC"]),
        "NMI": float(met["NMI"]),
        "rare_recall": float(met["rare_recall"]),
        "n_clusters": int(met["n_clusters"]),
    }


# --------------------------------------------------------------------------
# decision margins
# --------------------------------------------------------------------------

def pair_coordinates(state: State) -> np.ndarray:
    """Dimensionless coordinates of every rare pair: (c, n, q, z)."""
    ids = np.flatnonzero(state.rare)
    if len(ids) < 2:
        return np.zeros((0, 4))
    dmat = np.linalg.norm(
        state.centers[:, None, :] - state.centers[None, :, :], axis=2
    )
    np.fill_diagonal(dmat, np.inf)
    non_rare = np.flatnonzero(~state.rare)
    iso = (dmat.min(axis=1) if len(non_rare) == 0
           else dmat[:, non_rare].min(axis=1))
    rsum = state.radii[ids][:, None] + state.radii[ids][None, :]
    d = dmat[np.ix_(ids, ids)]
    c = np.abs(state.gamma[ids][:, None] - state.gamma[ids][None, :]) / np.maximum(
        np.maximum(state.gamma[ids][:, None], state.gamma[ids][None, :]), 1e-12
    )
    q = d / np.maximum(np.minimum(iso[ids][:, None], iso[ids][None, :]), 1e-12)
    n = d / np.maximum(rsum, 1e-12)
    if len(non_rare) == 0:
        z = np.full_like(d, np.inf)
    else:
        mids = 0.5 * (state.centers[ids][:, None, :] + state.centers[ids][None, :, :])
        d_bg = np.linalg.norm(
            mids[:, :, None, :] - state.centers[non_rare][None, None, :, :], axis=3
        ).min(axis=2)
        z = d_bg / np.maximum(d, 1e-12)
    iu = np.triu_indices(len(ids), 1)
    return np.column_stack([c[iu], n[iu], q[iu], z[iu]])


def linked(coords: np.ndarray, delta, eta, beta) -> np.ndarray:
    """Pair x parameter-grid truth table; scalars broadcast as columns."""
    c, n, q, z = (coords[:, i:i + 1] for i in range(4))
    with np.errstate(invalid="ignore"):
        return (c <= delta) & (q < beta) & ((n <= eta) | (z > beta))


def margin_scan(coords: np.ndarray, axis: str) -> tuple[float, float] | None:
    """Distance from the defaults to the nearest membership flip on one axis.

    The other two coordinates stay at their defaults.  Returns (distance,
    value at the flip) or None when no flip occurs inside the scan range.
    """
    if len(coords) == 0:
        return None
    if axis == "delta":
        grid = np.linspace(0.0, 1.0, 2001)
        base = linked(coords, D0, E0, B0)
        hit = linked(coords, grid[None, :], E0, B0) != base
    elif axis == "eta":
        grid = np.linspace(1.0, 6.0, 2001)
        base = linked(coords, D0, E0, B0)
        hit = linked(coords, D0, grid[None, :], B0) != base
    else:
        grid = np.linspace(0.0, 1.0, 2001)
        base = linked(coords, D0, E0, B0)
        hit = linked(coords, D0, E0, grid[None, :]) != base
    if not hit.any():
        return None
    rows, cols = np.nonzero(hit)
    values = grid[cols]
    distance = np.abs(values - {"delta": D0, "eta": E0, "beta": B0}[axis])
    j = int(np.argmin(distance))
    return float(distance[j]), float(values[j])


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["grid"]["delta"] == list(DELTAS)
    assert protocol["grid"]["eta"] == list(ETAS)
    assert protocol["grid"]["beta"] == list(BETAS)

    t_start = time.perf_counter()
    units = load_units()
    print(f"{len(units)} units: "
          f"{sum(u.cohort == 'unsw_v2' for u in units)} unsw, "
          f"{sum(u.cohort == 'dev' for u in units)} dev, "
          f"{sum(u.cohort == 'covtype' for u in units)} covtype", flush=True)

    cells = list(main_grid_cells())
    rng = np.random.default_rng(20261023)
    check_idx = sorted(set(
        [DELTAS.index(D0) * len(ETAS) * len(BETAS) + ETAS.index(E0) * len(BETAS) + BETAS.index(B0)]
        + list(rng.choice(len(cells), size=32, replace=False))
    ))
    print(f"{len(cells)} cells; equivalence check on {len(check_idx)} cells per unit",
          flush=True)

    rows: list[dict] = []
    margins: list[dict] = []
    equivalence_failures = 0

    for ui, unit in enumerate(units):
        state = build_state(unit.X, unit.k, 0.5)
        default_labels, default_edges = run_tail(state, D0, E0, B0)
        default_met = metrics(unit.y, default_labels)
        default_hash = label_hash(default_labels)

        coords = pair_coordinates(state)
        for axis in ("delta", "eta", "beta"):
            hit = margin_scan(coords, axis)
            margins.append({
                "cohort": unit.cohort, "unit": unit.name, "axis": axis,
                "n_rare_pairs": int(len(coords)),
                "distance": -1.0 if hit is None else hit[0],
                "flip_value": np.nan if hit is None else hit[1],
            })

        for ci, (d, e, b) in enumerate(cells):
            labels, edges = run_tail(state, d, e, b)
            met = metrics(unit.y, labels)
            rhash = label_hash(labels)

            if ci in check_idx:
                params = dict(
                    fragment_density_contrast=d,
                    fragment_near_factor=e,
                    fragment_closeness_factor=b,
                    fragment_corridor_factor=b,
                )
                # The cached tail above leaves out the admissibility
                # condition, so the reference is the pipeline with that
                # condition switched off.  run_coupling_grid_final.py runs
                # the same grid through the shipped pipeline.
                ref = MGGBC(k=unit.k, fragment_guard=False,
                            **params).fit_predict(unit.X)
                if label_hash(ref) != rhash:
                    equivalence_failures += 1
                    print(f"MISMATCH {unit.name} delta={d} eta={e} beta={b}",
                          flush=True)

            rows.append({
                "cohort": unit.cohort, "unit": unit.name, "n": len(unit.X),
                "k": unit.k, "slice": "main", "tau_gamma": 0.5,
                "delta": d, "eta": e, "beta": b,
                **met, "n_rare_cores": int(state.rare.sum()),
                "n_fragment_edges": edges, "label_hash": rhash,
                "default_hash": default_hash,
                "is_default_cell": bool(d == D0 and e == E0 and b == B0),
                "is_default_partition": bool(rhash == default_hash),
                "default_ARI": default_met["ARI"],
                "default_rare_recall": default_met["rare_recall"],
                "default_n_clusters": default_met["n_clusters"],
                "default_edges": default_edges,
            })

        # Secondary slice: tau_gamma x delta, eta and beta at their defaults.
        for tau in TAU_GAMMAS:
            state_tau = state if tau == 0.5 else build_state(unit.X, unit.k, tau)
            t_labels, t_edges = run_tail(state_tau, D0, E0, B0)
            t_met = metrics(unit.y, t_labels)
            for d in DELTAS:
                labels, edges = run_tail(state_tau, d, E0, B0)
                met = metrics(unit.y, labels)
                rows.append({
                    "cohort": unit.cohort, "unit": unit.name, "n": len(unit.X),
                    "k": unit.k, "slice": "tau_gamma", "tau_gamma": tau,
                    "delta": d, "eta": E0, "beta": B0,
                    **met, "n_rare_cores": int(state_tau.rare.sum()),
                    "n_fragment_edges": edges,
                    "label_hash": label_hash(labels),
                    "default_hash": label_hash(t_labels),
                    "is_default_cell": bool(tau == 0.5 and d == D0),
                    "is_default_partition": bool(label_hash(labels) == label_hash(t_labels)),
                    "default_ARI": t_met["ARI"],
                    "default_rare_recall": t_met["rare_recall"],
                    "default_n_clusters": t_met["n_clusters"],
                    "default_edges": t_edges,
                })
        print(f"[{ui + 1}/{len(units)}] {unit.cohort}/{unit.name} "
              f"cores={int(state.rare.sum())} default_edges={default_edges} "
              f"({time.perf_counter() - t_start:.1f}s)", flush=True)

    if equivalence_failures:
        raise SystemExit(
            f"cached-state path disagrees with MGGBC.fit_predict on "
            f"{equivalence_failures} checks; aborting"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coupling_grid_raw.csv", index=False, float_format="%.12g")
    pd.DataFrame(margins).to_csv(OUT / "coupling_margins.csv", index=False,
                                 float_format="%.12g")

    summary = summarise(df, margins, protocol)
    (OUT / "coupling_grid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["cohort_table"]).to_csv(
        OUT / "coupling_grid_table.csv", index=False, float_format="%.12g")
    print(json.dumps(summary["hypotheses"], indent=2))
    print(f"done in {time.perf_counter() - t_start:.1f}s")

    try:
        make_figure(df)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"figure failed: {exc}")


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------

def _cell_agg(group: pd.DataFrame) -> dict:
    d_ari = group["ARI"] - group["default_ARI"]
    finite = np.isfinite(group["rare_recall"]) & np.isfinite(group["default_rare_recall"])
    d_rec = (group["rare_recall"] - group["default_rare_recall"])[finite]
    return {
        "cells": int(group.groupby(["delta", "eta", "beta"]).ngroups),
        "units": int(group["unit"].nunique()),
        "changed_units": int((~group["is_default_partition"]).sum()),
        "worst_dARI": float(d_ari.min()),
        "worst_dRrec": float(d_rec.min()) if len(d_rec) else float("nan"),
        "median_dARI": float(d_ari.median()),
        "median_dRrec": float(d_rec.median()) if len(d_rec) else float("nan"),
        "edge_min": int(group["n_fragment_edges"].min()),
        "edge_max": int(group["n_fragment_edges"].max()),
    }


def _ss_decomposition(table: np.ndarray) -> dict:
    """Balanced three-way sum-of-squares decomposition of a 9x7x5 table."""
    a, b, c = table.shape
    g = table.mean()
    ma = table.mean(axis=(1, 2))[:, None, None]
    mb = table.mean(axis=(0, 2))[None, :, None]
    mc = table.mean(axis=(0, 1))[None, None, :]
    mab = table.mean(axis=2)[:, :, None]
    mac = table.mean(axis=1)[:, None, :]
    mbc = table.mean(axis=0)[None, :, :]
    ss_total = float(((table - g) ** 2).sum())
    ss_a = float(b * c * ((ma - g) ** 2).sum())
    ss_b = float(a * c * ((mb - g) ** 2).sum())
    ss_c = float(a * b * ((mc - g) ** 2).sum())
    ss_ab = float(c * ((mab - ma - mb + g) ** 2).sum())
    ss_ac = float(b * ((mac - ma - mc + g) ** 2).sum())
    ss_bc = float(a * ((mbc - mb - mc + g) ** 2).sum())
    ss_abc = float(
        ((table - mab - mac - mbc + ma + mb + mc - g) ** 2).sum()
    )
    denom = ss_total if ss_total > 0 else 1.0
    return {
        "ss_total": ss_total,
        "eta2": {
            "delta": ss_a / denom, "eta": ss_b / denom, "beta": ss_c / denom,
            "delta_x_eta": ss_ab / denom, "delta_x_beta": ss_ac / denom,
            "eta_x_beta": ss_bc / denom, "three_way": ss_abc / denom,
        },
        "main_effects_sum": (ss_a + ss_b + ss_c) / denom,
    }


def summarise(df: pd.DataFrame, margins: list[dict], protocol: dict) -> dict:
    main = df[df["slice"] == "main"]
    cohorts = {}
    for cohort, grp in main.groupby("cohort"):
        cohorts[cohort] = _cell_agg(grp)

    table_rows = []
    for cohort, agg in cohorts.items():
        sub = main[main["cohort"] == cohort]
        per_cell = [
            g for _, g in sub.groupby(["delta", "eta", "beta"])
        ]
        table_rows.append({"cohort": cohort, **agg})

    # H1 on the UNSW validation blocks.
    unsw = main[main["cohort"] == "unsw_v2"].copy()
    unsw["dARI"] = unsw["ARI"] - unsw["default_ARI"]
    unsw["dRrec"] = unsw["rare_recall"] - unsw["default_rare_recall"]
    cell_stats = (
        unsw.groupby(["delta", "eta", "beta"])
        .agg(worst_dARI=("dARI", "min"),
             worst_dRrec=("dRrec", "min"),
             median_dRrec=("dRrec", "median"))
        .reset_index()
    )
    h1_viol = cell_stats[
        (cell_stats["worst_dRrec"] < -0.10)
        | (cell_stats["worst_dARI"] < -0.05)
        | (cell_stats["median_dRrec"] < -0.01)
    ]
    h1 = {
        "cells": int(len(cell_stats)),
        "violations": int(len(h1_viol)),
        "worst_dRrec": float(cell_stats["worst_dRrec"].min()),
        "worst_dARI": float(cell_stats["worst_dARI"].min()),
        "min_median_dRrec": float(cell_stats["median_dRrec"].min()),
        "violating_cells": h1_viol.to_dict(orient="records")[:20],
    }

    # H2: SS decomposition of the total-edge response over UNSW blocks.
    edge_table = np.zeros((len(DELTAS), len(ETAS), len(BETAS)))
    for (d, e, b), g in unsw.groupby(["delta", "eta", "beta"]):
        edge_table[DELTAS.index(d), ETAS.index(e), BETAS.index(b)] = float(
            g["n_fragment_edges"].sum())
    ss_edges = _ss_decomposition(edge_table)

    # H2 also on the pooled median rare-recall response, when it is not constant.
    rec_table = np.full_like(edge_table, np.nan)
    for (d, e, b), g in unsw.groupby(["delta", "eta", "beta"]):
        rec_table[DELTAS.index(d), ETAS.index(e), BETAS.index(b)] = float(
            (g["rare_recall"] - g["default_rare_recall"]).median())
    ss_rec = _ss_decomposition(rec_table) if np.nanstd(rec_table) > 0 else None

    # H3: plateau.
    per_cell_changed = (
        main.groupby(["delta", "eta", "beta"])["is_default_partition"]
        .all().reset_index()
    )
    h3_cells = int(per_cell_changed["is_default_partition"].sum())

    # H4: Digits-Rare.
    digits = main[main["unit"] == "Digits-Rare"]
    d_ari = digits["ARI"] - digits["default_ARI"]
    d_rec = digits["rare_recall"] - digits["default_rare_recall"]
    h4_viol = (d_ari < -0.001) | (d_rec < -0.001)
    h4 = {
        "cells": int(len(digits)),
        "violations": int(h4_viol.sum()),
        "worst_dARI": float(d_ari.min()),
        "worst_dRrec": float(d_rec.min()),
    }

    # Margins per cohort: nearest flip on each axis.
    mar = pd.DataFrame(margins)
    margin_summary = {}
    for cohort, grp in mar.groupby("cohort"):
        row = {}
        for axis, sub in grp.groupby("axis"):
            finite = sub[sub["distance"] >= 0]
            row[axis] = {
                "nearest_distance": float(finite["distance"].min()) if len(finite) else None,
                "flip_value": float(finite.loc[finite["distance"].idxmin(), "flip_value"])
                if len(finite) else None,
                "unit": str(finite.loc[finite["distance"].idxmin(), "unit"])
                if len(finite) else None,
            }
        margin_summary[cohort] = row

    # tau_gamma slice summary.
    tau = df[df["slice"] == "tau_gamma"]
    tau_summary = []
    for (cohort, tau_value, d), g in tau.groupby(["cohort", "tau_gamma", "delta"]):
        tau_summary.append({
            "cohort": cohort, "tau_gamma": float(tau_value), "delta": float(d),
            "changed_units": int((~g["is_default_partition"]).sum()),
            "edges": int(g["n_fragment_edges"].sum()),
            "worst_dRrec": float((g["rare_recall"] - g["default_rare_recall"]).min()),
        })

    return {
        "protocol_id": protocol["protocol_id"],
        "cells": len(DELTAS) * len(ETAS) * len(BETAS),
        "cohort_table": table_rows,
        "hypotheses": {"H1": h1, "H2_edges": ss_edges,
                       "H2_rare_recall": ss_rec,
                       "H3": {"cells_identical_on_all_units": h3_cells},
                       "H4": h4},
        "margins": margin_summary,
        "tau_gamma_slice": tau_summary,
        "edge_table": edge_table.tolist(),
    }


def make_figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = df[(df["slice"] == "main") & (df["cohort"] == "unsw_v2")]
    levels = {"delta": DELTAS, "eta": ETAS, "beta": BETAS}

    def matrix(xvar, yvar, fixed_var, fixed_value, fn):
        xv, yv = levels[xvar], levels[yvar]
        out = np.full((len(yv), len(xv)), 0.0)
        sub = main[np.isclose(main[fixed_var], fixed_value)]
        for (x, y), g in sub.groupby([xvar, yvar]):
            out[yv.index(y), xv.index(x)] = fn(g)
        return out

    def edges(g):
        return float(g["n_fragment_edges"].sum())

    def changed(g):
        return float((~g["is_default_partition"]).sum())

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.5))
    panels = [
        (axes[0, 0], "delta", "eta", "beta", 0.50, edges,
         r"(a) $\beta=0.5$: fragment edges"),
        (axes[0, 1], "delta", "beta", "eta", 1.50, edges,
         r"(b) $\eta=1.5$: fragment edges"),
        (axes[0, 2], "beta", "eta", "delta", 0.25, edges,
         r"(c) $\delta=0.25$: fragment edges"),
        (axes[1, 0], "delta", "eta", "beta", 0.50, changed,
         r"(d) $\beta=0.5$: blocks changed"),
        (axes[1, 1], "delta", "beta", "eta", 1.50, changed,
         r"(e) $\eta=1.5$: blocks changed"),
        (axes[1, 2], "beta", "eta", "delta", 0.25, changed,
         r"(f) $\delta=0.25$: blocks changed"),
    ]
    for ax, xvar, yvar, fvar, fval, fn, title in panels:
        grid = matrix(xvar, yvar, fvar, fval, fn)
        vmax = max(1.0, float(np.nanmax(grid)))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues",
                       vmin=0, vmax=vmax)
        xv, yv = levels[xvar], levels[yvar]
        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels([f"{v:g}" for v in xv], fontsize=5.5)
        ax.set_yticks(range(len(yv)))
        ax.set_yticklabels([f"{v:g}" for v in yv], fontsize=5.5)
        symbols = {"delta": r"$\delta$", "eta": r"$\eta$", "beta": r"$\beta$"}
        ax.set_xlabel(symbols[xvar], fontsize=7)
        ax.set_ylabel(symbols[yvar], fontsize=7)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if v > 0.55 * vmax else "black")
        fx, fy = {"delta": D0, "eta": E0, "beta": B0}[xvar], \
                 {"delta": D0, "eta": E0, "beta": B0}[yvar]
        ax.add_patch(plt.Rectangle((xv.index(fx) - 0.5, yv.index(fy) - 0.5), 1, 1,
                                   fill=False, ec="crimson", lw=1.3))
        ax.set_title(title, fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(OUT / "coupling_grid_heatmap.png", dpi=200)
    fig.savefig(OUT / "coupling_grid_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()