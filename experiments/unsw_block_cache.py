"""Cache the frozen UNSW-NB15 temporal-block rows so diagnostics avoid rescanning 2.5M rows."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_unsw_nb15_temporal_compare import (  # noqa: E402
    load_protocol,
    load_selected_rows,
    normalized_labels,
    prepare_matrix,
    sample_block,
)

CACHE_DIR = ROOT / "results" / "cache"
CACHE = CACHE_DIR / "unsw_selected_rows.pkl"
PROTOCOL = ROOT / "experiments" / "unsw_nb15_temporal_protocol.json"


def get_protocol(path: Path | None = None) -> tuple[dict, str]:
    return load_protocol(path or PROTOCOL)


def _cache_path(protocol: dict) -> Path:
    """One cache per protocol id; v1 keeps its original filename."""
    pid = str(protocol.get("protocol_id", "unknown"))
    if pid.endswith("-v1"):
        return CACHE
    return CACHE_DIR / f"unsw_selected_rows_{pid}.pkl"


def get_source_rows(protocol: dict) -> pd.DataFrame:
    cache = _cache_path(protocol)
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)
    rows = load_selected_rows(protocol)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def get_block(protocol: dict, source_rows: pd.DataFrame, block_id: int):
    """Reproduce one frozen block exactly as the main experiment does."""
    starts = protocol["sampling"]["selected_block_starts_unix"]
    n = int(protocol["sampling"]["rows_per_block"])
    base_seed = int(protocol["sampling"]["block_selection_seed"])
    start = int(starts[block_id - 1])
    sample = sample_block(source_rows, start, n, base_seed + block_id)
    X = prepare_matrix(sample)
    y_true, classes, counts = normalized_labels(sample)
    return sample, X, y_true, classes, counts


if __name__ == "__main__":
    protocol, _ = get_protocol()
    rows = get_source_rows(protocol)
    print(f"cached {len(rows):,} rows -> {CACHE}")
