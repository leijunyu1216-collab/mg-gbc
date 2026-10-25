"""Register the held-out OpenML benchmark cohort before any model is fitted.

The manuscript's constants were fixed on development data, so a reviewer can ask
whether the reported gains survive a cohort that played no part in any decision.
This script builds such a cohort and freezes it into a protocol document.  It
reads OpenML metadata only.  It never fits a model and never writes a score, so
nothing in the protocol can depend on how MG-GBC performs.

Membership reads no class-rarity quantity.  A data set enters the cohort when its
metadata alone shows that it sits in the domain the method declares: an active ARFF
classification task with exactly one nominal target, between 300 and 20000 rows, at
least two classes (a task-validity bound: one class has no partition to recover),
between 3 and 100 usable features, and no missing values.  The universe is the fixed
block of OpenML data ids 1 to ID_MAX and there is no cap: the cohort is every
admissible data set in that block.  The block is permuted with the seed frozen below,
and the permutation decides only which member of a variant family represents its
source.  Data sets whose names share a first token are one source measured several
ways (``volcanoes-a2`` and ``volcanoes-b1``, ``fri_c3_1000_25`` and
``fri_c4_500_50``, ``mfeat-fourier`` and ``mfeat-zernike``), so only the first drawn
of each source is admitted; without that rule a single uploader's variant family
fills the cohort.  Ids and names that belong to the development history are
excluded, and the exclusion lists are written into the protocol so they can be
audited.

The two analysis subsets are defined *after* membership, by the rare band the
manuscript already uses: a data set is in the rare subset when its smallest class
holds fewer than sqrt(n) points, and in the locality control otherwise.  No lower
floor is imposed on that class: a set whose smallest class is below s_min = 3 is
admitted, falls in the rare subset, and counts against the method, which cannot
retain a core there.  The subset a data set falls in is a property of its labels,
not a choice made here, and the cohort is the whole admissible population of the
id block, so its size is a property of the repository rather than a parameter.

Usage:
    python -I experiments/register_openml_heldout.py --screen-only   # inspect
    python -I experiments/register_openml_heldout.py                 # freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "https://api.openml.org/api/v1/json"
HEADERS = {"User-Agent": "mg-gbc-heldout-registration/1.0 (research replication)"}
META_CACHE = ROOT / "scratch" / "openml_meta"
DESC_CACHE = ROOT / "scratch" / "openml_desc"
PROTOCOL = ROOT / "experiments" / "openml_heldout_protocol_v1.json"
SCREEN_LOG = ROOT / "results" / "openml_heldout_screen_log.json"

ID_MAX = 10000
SEED = 20261024
BATCH = 40
WORKERS = 6
# No cap: the cohort is every admissible data set in the id block, so there is no
# stopping rule to tune and no set that a larger cohort would have changed.
COHORT_SIZE = None

N_MIN, N_MAX = 300, 20000
C_MIN = 2
D_MIN, D_MAX = 3, 100
MIN_RARE_SIZE = 3

# Freezing needs both subsets to be able to carry their criterion.  The two-sided
# exact sign test behind H1 cannot reach p < 0.05 on fewer than six non-tied pairs
# (its smallest attainable value on six pairs is 2/2^6 = 0.031), so six rare-subset
# data sets is the shortest cohort on which the registered criterion is decidable at
# all; the locality control is held to a larger floor because its criterion is a
# 90% share and needs the resolution to distinguish 9/10 from 8/10.
MIN_RARE = 6
MIN_CONTROL = 20

# Every data set that took part in choosing a constant, in any version of the
# development history.  Excluded by id and, independently, by name.
DEVELOPMENT_IDS = sorted({
    10, 30, 35, 36, 39, 41, 53, 54, 62, 152, 181, 182, 187, 258, 307,
    40497, 40983, 40685,
})
DEVELOPMENT_NAMES = [
    "iris", "glass", "ecoli", "zoo", "wine", "shuttle", "yeast", "page-blocks",
    "lymph", "dermatology", "vehicle", "segment", "satimage", "vowel", "wilt",
    "thyroid", "digits", "olivetti", "covertype", "balance-scale",
]

MODEL_FILES = ["src/arp_gb.py", "src/mg_gbc.py", "src/core.py", "src/metrics.py",
               "src/metrics_ext.py"]


def fetch_json(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=HEADERS), timeout=45
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410, 412):
                return {"__http_error__": exc.code}
            time.sleep(1.0 + attempt)
        except Exception:  # noqa: BLE001
            time.sleep(1.0 + attempt)
    return None


def _as_list(payload):
    """OpenML returns a bare object instead of a one-element list."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _read_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _qualities(data_id: int) -> dict:
    cached = _read_cache(META_CACHE / f"{data_id}.json")
    if "qualities" in cached:
        return cached["qualities"]
    doc = fetch_json(f"{API}/data/qualities/{data_id}")
    if not doc or "data_qualities" not in doc:
        return {}
    out = {q["name"]: q.get("value")
           for q in _as_list(doc["data_qualities"].get("quality")) if q.get("name")}
    cached.update(data_id=data_id, qualities=out)
    META_CACHE.mkdir(parents=True, exist_ok=True)
    (META_CACHE / f"{data_id}.json").write_text(json.dumps(cached), encoding="utf-8")
    return out


def _features(data_id: int) -> list[dict]:
    cached = _read_cache(META_CACHE / f"{data_id}.json")
    if cached.get("features"):
        return cached["features"]
    doc = fetch_json(f"{API}/data/features/{data_id}")
    out = []
    if doc and "data_features" in doc:
        for f in _as_list(doc["data_features"].get("feature")):
            out.append({
                "name": f.get("name"),
                "data_type": f.get("data_type"),
                "is_target": str(f.get("is_target", "")).lower() == "true",
                "is_ignore": str(f.get("is_ignore", "")).lower() == "true",
                "is_row_identifier": str(f.get("is_row_identifier", "")).lower() == "true",
                "missing": int(f.get("number_of_missing_values") or 0),
            })
    cached.update(data_id=data_id, features=out)
    META_CACHE.mkdir(parents=True, exist_ok=True)
    (META_CACHE / f"{data_id}.json").write_text(json.dumps(cached), encoding="utf-8")
    return out


def _description(data_id: int) -> dict:
    DESC_CACHE.mkdir(parents=True, exist_ok=True)
    path = DESC_CACHE / f"{data_id}.json"
    cached = _read_cache(path)
    if cached:
        return cached
    doc = fetch_json(f"{API}/data/{data_id}")
    if not doc or "data_set_description" not in doc:
        return {}
    d = doc["data_set_description"]
    out = {
        "name": str(d.get("name", "")),
        "version": str(d.get("version", "")),
        "status": str(d.get("status", "")),
        "format": str(d.get("format", "")),
        "md5_checksum": str(d.get("md5_checksum", "") or ""),
        "target": str(d.get("default_target_attribute", "") or ""),
        "upload_date": str(d.get("upload_date", "") or ""),
    }
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def source_key(name: str) -> str:
    """The source a data set name belongs to: its first token, digits removed.

    One uploader's variant families share that token: ``volcanoes-a2`` and
    ``volcanoes-b1``, ``fri_c3_1000_25`` and ``fri_c4_500_50``,
    ``chscase_census2`` and ``chscase_census6``, ``mfeat-fourier`` and
    ``mfeat-zernike``, ``one-hundred-plants-margin`` and
    ``one-hundred-plants-shape``.  The rule is deliberately coarse: it can only
    merge data sets, never split a family, so it can only make the cohort more
    diverse.  Where it merges too much (``analcatdata_authorship`` with
    ``analcatdata_dmft``, ``kc1`` with nothing) the loss is a set the cohort could
    have carried, not a duplicate it silently kept.
    """
    n = str(name).strip().lower()
    first = n.replace("-", "_").split("_")[0]
    key = "".join(ch for ch in first if not ch.isdigit())
    return key or n


def qualities_gate(data_id: int) -> tuple[str, dict]:
    """The label-free part of the screen that qualities alone can decide."""
    rec: dict = {"data_id": data_id, "n": None, "C": None, "d": None, "minority": None}
    q = _qualities(data_id)
    if not q:
        return "skip:metadata-unavailable", rec
    n = fnum(q.get("NumberOfInstances"))
    c = fnum(q.get("NumberOfClasses"))
    minority = fnum(q.get("MinorityClassSize"))
    missing = fnum(q.get("NumberOfMissingValues"))
    rec.update(n=n, C=c, minority=minority)
    if n is None or c is None or minority is None:
        return "skip:incomplete-qualities", rec
    if not (N_MIN <= n <= N_MAX):
        return f"skip:n={int(n)}", rec
    if c < C_MIN:
        return f"skip:C={int(c)}", rec
    if (missing or 0) > 0:
        return "skip:missing-values", rec
    # MinorityClassSize must be reported, because it is read once after the cohort is
    # fixed to place each member in a subset.  Its value is not a membership condition:
    # a set whose smallest class is below s_min = 3 is admitted, lands in the rare
    # subset, and counts against the method, which cannot retain a core there.
    return "pass", rec


def structural(data_id: int) -> tuple[str, dict]:
    """Full label-free screen. Verdict: eligible / skip:<reason>."""
    verdict, rec = qualities_gate(data_id)
    if verdict != "pass":
        return verdict, rec
    feats = _features(data_id)
    if not feats:
        return "skip:features-unavailable", rec
    targets = [f for f in feats if f["is_target"]]
    if len(targets) != 1:
        return f"skip:targets={len(targets)}", rec
    if targets[0]["data_type"] != "nominal":
        return f"skip:target-type={targets[0]['data_type']}", rec
    usable = [f for f in feats if not f["is_target"] and not f["is_ignore"]
              and not f["is_row_identifier"]]
    if any(f["missing"] > 0 for f in usable):
        return "skip:missing-values", rec
    d = len(usable)
    rec["d"] = d
    rec["target"] = targets[0]["name"]
    rec["nominal_features"] = sum(1 for f in usable if f["data_type"] == "nominal")
    if not (D_MIN <= d <= D_MAX):
        return f"skip:d={d}", rec
    rec["sqrt_n"] = float(rec["n"] ** 0.5)
    rec["rare_band"] = bool(rec["minority"] < rec["sqrt_n"])
    return "eligible", rec


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-only", action="store_true",
                    help="screen and print the cohort without writing the protocol")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing protocol")
    ap.add_argument("--from-log", metavar="PATH",
                    help="freeze from a complete screen log instead of re-walking the id space; "
                         "the log's draw order is re-derived from the seed and must match")
    args = ap.parse_args()

    if PROTOCOL.exists() and not (args.screen_only or args.force):
        raise SystemExit(f"{PROTOCOL.name} already exists. A frozen protocol is not "
                         "rewritten; use --screen-only to inspect, or --force to supersede.")

    order = np.random.default_rng(SEED).permutation(np.arange(1, ID_MAX + 1)).tolist()

    if args.from_log:
        # Freezing from the log rather than from a second walk removes network
        # nondeterminism: a transient fetch failure on the second walk would reject
        # a data set the screen admitted.  The cohort is whatever the log records,
        # and the log's own draw order is checked against the seed, so nothing here
        # is a choice.
        path = Path(args.from_log)
        log = json.loads(path.read_text(encoding="utf-8"))
        walked = [int(rec["data_id"]) for rec in log]
        if walked != order:
            raise SystemExit(f"{path.name} does not record the seeded permutation of ids "
                             f"1-{ID_MAX} ({len(walked)} records, first divergence at "
                             f"{next((i for i, (a, b) in enumerate(zip(walked, order)) if a != b), None)}). "
                             "Only a complete log of this draw can be frozen.")
        cohort = [rec for rec in log if rec["verdict"] == "eligible"]
        admitted_sources = {rec["source"] for rec in cohort}
        drawn = len(log)
        last_drawn_id = walked[-1]
        print(f"read {path.name}: {drawn} drawn ids, cohort {len(cohort)} "
              f"(rare {sum(1 for r in cohort if r['rare_band'])})", flush=True)
    else:
        log: list[dict] = []
        cohort: list[dict] = []
        admitted_sources: set[str] = set()
        drawn = 0
        last_drawn_id = None

        print(f"universe ids 1-{ID_MAX}, seed {SEED}: first drawn ids "
              f"{order[:8]} ...", flush=True)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for start in range(0, len(order), BATCH):
                ids = order[start:start + BATCH]
                for data_id, (verdict, rec) in zip(ids, pool.map(structural, ids)):
                    drawn += 1
                    last_drawn_id = data_id
                    if verdict == "eligible":
                        desc = _description(data_id)
                        rec.update({k: desc.get(k) for k in
                                    ("name", "version", "status", "format", "md5_checksum",
                                     "upload_date")})
                        name = str(desc.get("name", "")).lower()
                        source = source_key(desc.get("name", ""))
                        rec["source"] = source
                        if desc.get("status") != "active":
                            verdict = f"skip:status={desc.get('status')}"
                        elif desc.get("format") != "ARFF":
                            verdict = f"skip:format={desc.get('format')}"
                        elif data_id in DEVELOPMENT_IDS:
                            verdict = "skip:development-id"
                        elif any(token in name for token in DEVELOPMENT_NAMES):
                            verdict = "skip:development-name"
                        elif source in admitted_sources:
                            verdict = f"skip:source={source}"
                    rec["verdict"] = verdict
                    rec["n"] = None if rec.get("n") is None else int(rec["n"])
                    rec["C"] = None if rec.get("C") is None else int(rec["C"])
                    rec["minority"] = None if rec.get("minority") is None else int(rec["minority"])
                    log.append(rec)
                    if verdict == "eligible":
                        admitted_sources.add(rec["source"])
                        cohort.append(rec)
                print(f"  drawn {drawn} ids (last {last_drawn_id}): cohort {len(cohort)} "
                      f"(rare {sum(1 for r in cohort if r['rare_band'])})", flush=True)
                if start // BATCH % 25 == 24:
                    # The metadata cache is the expensive part and is written per id; this
                    # checkpoint only keeps the screen log itself from being lost.
                    SCREEN_LOG.parent.mkdir(parents=True, exist_ok=True)
                    SCREEN_LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")

    cohort.sort(key=lambda r: r["data_id"])

    SCREEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    SCREEN_LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
    rare = [r for r in cohort if r["rare_band"]]
    control = [r for r in cohort if not r["rare_band"]]
    print(f"\ndrew all {drawn} ids of the seeded permutation of 1-{ID_MAX}; "
          f"wrote {SCREEN_LOG.relative_to(ROOT)}")
    print(f"cohort {len(cohort)}: rare subset {len(rare)}, locality control {len(control)}, "
          f"{len(admitted_sources)} distinct sources")
    print(f"\nrare subset ({len(rare)}):")
    for r in rare:
        print(f"  id={r['data_id']:>5} {str(r['name'])[:30]:<30} src={str(r['source'])[:14]:<14} "
              f"n={r['n']:>6} C={r['C']:>3} d={r['d']:>3} min={r['minority']:>5} "
              f"sqrt(n)={r['sqrt_n']:.1f}")
    print(f"\nlocality control ({len(control)}), first 15 of {len(control)}:")
    for r in control[:15]:
        print(f"  id={r['data_id']:>5} {str(r['name'])[:30]:<30} src={str(r['source'])[:14]:<14} "
              f"n={r['n']:>6} C={r['C']:>3} d={r['d']:>3} min={r['minority']:>5} "
              f"sqrt(n)={r['sqrt_n']:.1f}")

    if args.screen_only:
        return
    if len(rare) < MIN_RARE:
        raise SystemExit(f"the cohort holds only {len(rare)} rare-subset data sets, below the "
                         f"floor of {MIN_RARE}. A cohort that short cannot support criterion H1, "
                         "whose two-sided exact sign test cannot reach p < 0.05 on fewer than six "
                         "non-tied pairs; widen ID_MAX before freezing rather than freezing a "
                         "subset on which the registered test is undecidable.")
    if len(control) < MIN_CONTROL:
        raise SystemExit(f"the cohort holds only {len(control)} control data sets, below the "
                         f"floor of {MIN_CONTROL}.")

    def entry(r: dict) -> dict:
        return {
            "data_id": r["data_id"], "name": r["name"], "version": r["version"],
            "source": r["source"], "subset": "rare" if r["rare_band"] else "control",
            "n": int(r["n"]), "C": int(r["C"]), "d": int(r["d"]),
            "minority_class_size": int(r["minority"]), "sqrt_n": r["sqrt_n"],
            "nominal_features": r["nominal_features"], "target": r["target"],
            "md5_checksum": r["md5_checksum"],
        }

    protocol = {
        "protocol_id": "OPENML-HELDOUT-20261024-v1",
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "supersedes": None,
        "purpose": (
            "A held-out benchmark cohort that answers the in-sample-fixing objection directly. "
            "Every constant of MG-GBC was chosen on the development data of Sections 4.2-4.8. "
            "This protocol scores a cohort of real public classification data sets that took no "
            "part in any of those decisions, once, at the frozen defaults, against pre-declared "
            "criteria."
        ),
        "evidence_tier": (
            "Held-out cohort. The data sets are independent sources, so the paired statistic is a "
            "cross-source statement, unlike the temporal blocks of Sections 4.7-4.8, which are "
            "repeated measures inside one capture. Cohort membership is decided without labels; "
            "the rare subset inside it is a property of the data, not a selection."
        ),
        "design_principle": (
            f"The cohort is the whole admissible population of a fixed id block, so there is no "
            f"stopping rule to tune. The universe is OpenML data ids 1-{ID_MAX}; "
            f"numpy's default_rng(SEED={SEED}) permutes it; a data set is admissible when it is an "
            f"active ARFF classification task with exactly one nominal target, {N_MIN} <= n <= "
            f"{N_MAX}, C >= {C_MIN}, {D_MIN} <= d <= {D_MAX} usable features, no missing values, not "
            "part of the development history, and the first drawn data set of its source. Every "
            "admissible data set is admitted, in draw order, and the draw order decides only which "
            "member of a variant family represents its source. No class-rarity quantity enters "
            "membership and no model was fitted while this document was being written, so no data "
            "set entered or left the cohort because of how rare its classes are or how either method "
            "performs on it. The two analysis subsets are defined after membership by the rare band "
            "the manuscript already uses, and the pre-registered criteria name which subset each of "
            "them is evaluated on."
        ),
        "source": {
            "name": (f"every admissible OpenML data set in the fixed id block 1-{ID_MAX}; "
                     f"numpy.random.default_rng({SEED}).permutation decides the order in which the "
                     f"block is walked and which member of a source family is admitted "
                     f"({drawn} ids drawn, {len(cohort)} admitted)"),
            "surface": "https://api.openml.org/api/v1/json/data/qualities/{id} and /data/features/{id} for the screen; /data/{id} for admissible candidates",
            "download_surface": "src.datasets._from_openml, i.e. sklearn.datasets.fetch_openml(data_id=..., as_frame=True, parser='auto')",
            "official_reference": "Vanschoren, van Rijn, Bischl and Kotthoff, 2014 (OpenML)",
            "accessed_at": datetime.now(timezone.utc).date().isoformat(),
            "license_note": "Public OpenML data sets; no authentication used.",
            "screen_log": "results/openml_heldout_screen_log.json",
            "superseded_screen_logs": [
                "results/openml_heldout_screen_log_labelfiltered_v0.json: the walk of the first "
                "design, in which the smallest class size was a membership condition",
                "results/openml_heldout_screen_log_ascending_v1.json: the walk of the second "
                "design, the first 120 admissible data sets in ascending id order",
            ],
        },
        "screen": {
            "id_universe": [1, ID_MAX],
            "draw": {
                "generator": "numpy.random.default_rng(SEED).permutation(arange(1, ID_MAX + 1))",
                "seed": SEED,
                "ids_drawn": drawn,
                "last_id_drawn": last_drawn_id,
                "order": ("permutation order; every admissible data set is taken, one per source, so "
                          "the order decides which variant of a family is admitted and nothing else"),
                "cohort_listed_by": "ascending data id, for readability; membership is the draw order",
            },
            "cohort_size": len(cohort),
            "cohort_size_is_a_maximum": False,
            "cohort_size_rule": (
                "There is no cap. The cohort is every admissible data set in the id block, so its "
                "size is a property of the repository and of the declared domain, not a parameter."
            ),
            "source_rule": (
                "The source of a data set is the first separator-delimited token of its name with "
                "digits removed, so volcanoes-a2 and volcanoes-b1, fri_c3_1000_25 and "
                "fri_c4_500_50, chscase_census2 and chscase_census6, mfeat-fourier and "
                "mfeat-zernike are each one source, and only the first drawn is admitted. The rule "
                "is coarse on purpose: it can merge data sets that are in fact distinct "
                "(analcatdata_authorship with analcatdata_dmft, kc1 with kc2), which costs the "
                "cohort a member, but it cannot let one uploader's variant family occupy several "
                "places in it."
            ),
            "min_rare_to_freeze": MIN_RARE,
            "min_control_to_freeze": MIN_CONTROL,
            "structural_conditions": [
                "status == 'active' and format == 'ARFF'",
                "exactly one target feature and its data type is nominal",
                f"{N_MIN} <= n <= {N_MAX}",
                f"C >= {C_MIN}, a task-validity condition: a data set with one class has no "
                "partition to recover",
                f"{D_MIN} <= d <= {D_MAX} usable features",
                "NumberOfMissingValues == 0 and no usable feature has a missing value",
                "NumberOfClasses, NumberOfInstances and MinorityClassSize are all reported, so that "
                "the subsets can be assigned after membership",
                "not in DEVELOPMENT_IDS and its name contains no DEVELOPMENT_NAMES token",
                "its source has not already been admitted",
            ],
            "membership_uses_class_rarity": False,
            "membership_uses_class_count": (
                f"only the bound C >= {C_MIN}. The smallest class size is not bounded: a data set "
                "whose smallest class is below s_min = 3 is admitted, lands in the rare subset, and "
                "counts against the method, which provably cannot retain a core there."
            ),
            "subset_rule": (
                "rare subset: the smallest class holds fewer than sqrt(n) points, the rare band of "
                "the manuscript, with no lower floor. locality control: otherwise, so no class is "
                "below sqrt(n) and the retention predicate provably cannot fire. A smallest class "
                f"below s_min = {MIN_RARE_SIZE} does not remove a data set from the rare subset; it "
                "stays in and counts against the method. The subset is computed after the cohort is "
                "fixed."
            ),
            "development_ids_excluded": DEVELOPMENT_IDS,
            "development_names_excluded": DEVELOPMENT_NAMES,
        },
        "preprocessing": {
            "loader": "src.datasets._from_openml(name, data_id)",
            "encoding": (
                "src.datasets._encode_frame: numeric columns pass through; true/false, yes/no and "
                "0/1 nominal columns become 0/1; every other nominal column becomes its integer "
                "factor code; rows holding a non-finite value are dropped. No scaling, no feature "
                "selection, no imputation, no target encoding."
            ),
            "md5_check": (
                "The recorded md5_checksum of each admitted version is written into the result rows "
                "so a later re-download that changes the file is visible."
            ),
            "k": "clip(round(sqrt(n)), 5, 10), computed from the frozen row count, never from the labels",
            "class_count_note": (
                "C >= 2 rather than C >= 3 because the declared domain already contains a binary "
                "data set (wilt, in the development cohort): the rare-core predicate is defined per "
                "ball and does not need three classes."
            ),
        },
        "models": {
            "graph_policy": (
                "matched: the inherited size-ratio guard is disabled in both methods "
                "(size_gate=False), so the comparison isolates the rare-core retention predicate "
                "and the signed edit rather than a difference in ball construction."
            ),
            "arp_gb_params": {"scale": True, "size_gate": False},
            "mg_gbc_params": {
                "scale": True,
                "size_gate": False,
                "keep_rare": True,
                "rare_density_z": 0.5,
                "rare_isolation_z": 0.0,
                "rare_min_size": 3,
                "fragment_density_contrast": 0.25,
                "fragment_near_factor": 1.5,
                "fragment_closeness_factor": 0.5,
                "fragment_corridor_factor": 0.5,
                "fragment_corridor_witnesses": 3,
                "fragment_guard": True,
            },
            "model_hashes": {rel: sha256_file(ROOT / rel) for rel in MODEL_FILES},
            "tuning_on_this_cohort": "None. Every constant above is the registered default.",
        },
        "cohort": [entry(r) for r in cohort],
        "analysis_subsets": {
            "rare": {"members": len(rare),
                     "role": "the primary analysis set: the mechanism can fire here"},
            "control": {"members": len(control),
                        "role": "the locality control: the mechanism cannot fire here"},
        },
        "endpoints": {
            "primary": "rare_recall, mean Hungarian-matched recall over the true classes below sqrt(n), paired by data set, on the rare subset",
            "secondary": ["ARI", "NMI", "ACC", "many_to_one_ACC", "n_clusters",
                          "n_balls", "n_rare_cores", "n_fragment_edges", "wall time"],
            "paired_delta": "MG-GBC minus ARP-GB on the same data set, same k, same preprocessing.",
            "agreement": "The two partitions are called identical when their adjusted Rand index is 1 to within 1e-12.",
        },
        "preregistered_criteria": {
            "H1_rare_recall": (
                "On the rare subset, the median paired delta rare_recall is greater than 0 and the "
                "two-sided exact sign test over the non-tied pairs gives p < 0.05."
            ),
            "H2_noninferiority": (
                "On the rare subset, where the edit can act, the median paired delta ARI is at least "
                "-0.05 and no single data set loses more than 0.15 ARI. The same two numbers over "
                "the whole cohort are reported beside them; the criterion is stated on the rare "
                "subset because the whole-cohort median is dominated by control data sets on which "
                "the two partitions are identical by construction."
            ),
            "H3_locality": (
                "On the locality control, where no class is below sqrt(n), the two methods return "
                "identical partitions on at least 90% of the data sets. This is the empirical "
                "counterpart of the manuscript's locality-of-the-edge-edit proposition "
                "(Section 3.4): with no rare core to retain, the edit is empty."
            ),
            "H4_loss_mechanism": (
                "Every rare-subset data set with delta ARI below -0.05 retains at least one rare "
                "core, as the locality corollaries of Section 3.4 predict: a loss requires a core "
                "admitted inside a dense majority region."
            ),
            "reporting_rule": (
                "All four criteria are reported whether or not they are met. No criterion, no margin, "
                "no subset rule and no cohort membership is redefined after the run. The run happens "
                "once; the runner refuses to overwrite an existing result file."
            ),
        },
        "pre_registration_caveat": (
            "Cohort membership reads no class-rarity quantity. The class count enters only as "
            f"C >= {C_MIN}, a task-validity bound: a data set with one class has no partition to "
            "recover. The smallest class size is read once, after the cohort is fixed, to place each "
            "member in the rare subset or in the locality control; that placement is the "
            "manuscript's own rare band and not a configuration choice. Three design decisions were "
            "taken before any model was fitted on any candidate data set, and are recorded here. "
            "(i) The first version of this screen made the smallest class a membership condition. A "
            "walk over ids 1-1560 then filled its primary cohort with twelve data sets from three "
            "sources padded by their variant families (leaf, one-hundred-plants-margin/shape/"
            "texture, volcanoes-a2..e5), which is a weaker cross-source claim than an unfiltered "
            "cohort; membership was therefore made label-free. "
            "(ii) The second version took the first 120 admissible data sets in ascending id order. "
            "That cohort filled at id 1069 and held 2 rare-subset members out of 120, because an "
            "ascending prefix of OpenML is its oldest uploads, dominated by balanced sets, rather "
            "than a sample of the repository; membership was therefore made a seeded permutation of "
            "the id block. "
            "(iii) The third version stopped that permutation at 60 admitted data sets, a cap set by "
            "a power calculation on the rare fraction measured in the second walk. The cap was "
            "removed before any candidate was scored: with no cap the cohort is the whole admissible "
            "population of the block, so its size is a property of the repository rather than a "
            "parameter, and the rare subset is whatever the block contains. "
            f"The realised cohort has {len(cohort)} members over {drawn} draws, {len(rare)} in the "
            f"rare subset and {len(control)} in the locality control, over {len(admitted_sources)} "
            f"distinct sources. The floors of {MIN_RARE} rare and {MIN_CONTROL} control members are "
            "capability floors fixed before the draw, not expectations: "
            f"{MIN_RARE} is the shortest subset on which the two-sided exact sign test behind H1 can "
            "reach p < 0.05 at all, its smallest attainable value on six non-tied pairs being "
            "2/2^6 = 0.031, so on a rare subset that short H1 can be met only by an unbroken run of "
            "wins and is reported as it comes out. The full screen log, including every drawn id and "
            "its skip reason, is written to results/openml_heldout_screen_log.json, and the logs of "
            "the first two superseded designs are kept beside it as "
            "results/openml_heldout_screen_log_labelfiltered_v0.json and "
            "results/openml_heldout_screen_log_ascending_v1.json, so all three cohorts can be "
            "reconstructed; the third design was stopped mid-walk when its cap was removed and wrote "
            "no log. Restricting the rare subset to the declared domain is a scope statement, not "
            "evidence of accuracy outside it: the locality control is what tests the outside."
        ),
    }

    PROTOCOL.write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest().upper()
    print(f"\nfroze {PROTOCOL.relative_to(ROOT)}  sha256={digest[:16]}")


if __name__ == "__main__":
    main()
