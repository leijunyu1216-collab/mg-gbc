"""Merge the two CIC-IDS2018 executions into the pre-declared pooled sample.

The protocol evaluates the replication day and the remaining eight days in two
separate executions so that the ordering is on record.  This script only
concatenates the two block tables and rebuilds the summary with the same
function the runner used, so the pooled statistics cannot drift from the
per-execution ones.  It refuses to merge rows that came from a different
protocol file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_cicids2018_temporal_compare as R  # noqa: E402

PROTOCOL = ROOT / "experiments" / "cicids2018_temporal_protocol_v1.json"
PARTS = ["replication", "holdout"]


def main() -> None:
    protocol, protocol_hash = R.load_protocol(PROTOCOL)
    frames = []
    for part in PARTS:
        path = ROOT / "results" / f"cicids2018_temporal_blocks_{part}.csv"
        frame = pd.read_csv(path)
        if set(frame["protocol_sha256"].astype(str)) != {protocol_hash}:
            raise RuntimeError(f"{path} was produced under a different protocol file")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(subset=["window_id", "method"]).any():
        raise RuntimeError("A window appears in both executions; the day sets are not disjoint.")

    out = ROOT / "results" / "cicids2018_temporal_blocks_pooled.csv"
    summary_path = ROOT / "results" / "cicids2018_temporal_summary_pooled.json"
    paired_path = ROOT / "results" / "cicids2018_temporal_paired_pooled.csv"
    combined.to_csv(out, index=False, float_format="%.12g")
    summary, paired = R.build_summary(combined, protocol, protocol_hash)
    summary["execution_order"] = {
        "replication": "2018-02-22",
        "holdout_and_control": "the remaining eight days",
        "note": "Two executions, replication first; file modification times record the order.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired.to_csv(paired_path, index=False, float_format="%.12g")
    print(json.dumps(summary["paired"], indent=1))
    print(json.dumps(summary["criteria"], indent=1))
    print(f"wrote {out}\nwrote {summary_path}\nwrote {paired_path}")


if __name__ == "__main__":
    main()