# MG-GBC

Reference implementation of **multi-granularity granular-ball clustering
(MG-GBC)**, a scale-free correction to the size truncation used by every
granular-ball clustering pipeline.

Granular-ball clustering summarises a data set as a set of adaptive balls and
then reasons about relations among balls. Current realisations dissolve every
ball smaller than a fixed cardinality, typically `sqrt(n)`, before any
structural comparison, and redistribute its points. When a class lies entirely
inside such a ball, no later stage of the pipeline can represent it as a
separate cluster. MG-GBC replaces that verdict with a retention predicate that
is local to the ball — small relative to `sqrt(n)`, denser than the ball
population by a margin calibrated against that population's robust spread, and
isolated from the non-candidate background — and pairs retention with a signed
edit of the ball-overlap graph that deletes absorption edges and adds fragment
edges.

The paper proves that the edit is confined to rare vertices, that a retained
core cannot be absorbed by a sparser ball, and that the added layer costs
`O(m^2) = O(n)` after ball construction. When no ball satisfies the predicate,
the method returns the inherited partition exactly.

## Layout

```
src/core.py        density index, representative selection, ball generation, trims
src/arp_gb.py      ARP-GB: the operator under correction
src/mg_gbc.py      MG-GBC: retention predicate, absorption blocking, fragment edges
src/metrics.py     ACC, NMI, ARI, rare-class recall
src/metrics_ext.py extended metrics used in the paired comparisons
src/datasets.py    UCI/OpenML loaders and the two synthetic probes
src/baselines.py   DBSCAN, DPC, k-means, RNN-DBSCAN, SNN-DPC
experiments/       one script per table or figure; frozen protocol JSONs
results/           the output files reported in the paper
tests/             invariants the paper depends on
```

## Install and check

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The six tests take about a second and need no data. They cover the fixed graph
policy, the admissibility condition on fragment edges, the exact fallback to
the baseline when no rare core is retained, determinism, and the partition
invariant.

## Data

Neither data set is redistributed here; both are public. See
[`data/README.md`](data/README.md) for the download sources and the expected
layout. The frozen protocol files in `experiments/` record the row counts and
SHA-256 digests of the CIC-IDS2018 shards, and the loaders verify them before
running.

## Reproducing the results

Every script is run from the repository root and writes into `results/`.
Scripts with options are shown with the setting used in the paper.

| Output in `results/` | Script | What it holds |
| --- | --- | --- |
| `rare_real_compare.csv` | `run_rare_real.py` | partitions on the real benchmark sets |
| `matched_arp.csv` | `run_matched_arp.py` | baseline rows under the matched graph policy |
| `workload_extra.csv` | `run_workload.py` | splits, imbalance ratios, runtime |
| `seed_summary_matched.json` | `run_seed_summary_matched.py` | ten paired draws on the probe |
| `sensitivity_paper_v2.csv` | `run_sensitivity_paper_v2.py` | sweep over the six constants |
| `host_sizecut_sweep.csv` | `run_host_sizecut_sweep.py` | response of the baseline to its own size cut |
| `runtime_v2.csv` | `run_runtime_v2.py` | timings against the baseline |
| `robustness.csv`, `robustness.png` | `run_robust.py`, `plot_robustness.py` | perturbation curves |
| `recent_paper_baselines.csv` | `run_recent_paper.py` | NGBC and EKM runs |
| `unsw_nb15_temporal_summary_v2.json` | `run_unsw_nb15_temporal_compare_v2.py --arp-size-gate false` | per-block UNSW-NB15 results |
| `unsw_pooled_matched.json` | `pool_unsw_matched.py` | pooled UNSW-NB15 blocks |
| `locality_diagnostic.csv` | `run_locality_diagnostic.py` | cross-component reassignment |
| `coupling_grid_guard_raw.csv`, `coupling_grid_heatmap.pdf` | `run_coupling_grid_final.py`, `plot_coupling_grid_final.py` | joint `delta`/`eta` grid |
| `cicids2018_temporal_summary_pooled.json` | `run_cicids2018_temporal_compare.py` then `combine_cicids2018_sets.py` | pooled CIC-IDS2018 blocks |
| `cicids2018_mechanism_diagnosis.json` | `diagnose_cicids2018_mechanism.py` | per-block ball geometry |
| `retention_hit_rate.json` | `diagnose_retention_hit_rate.py` | share of a block's rare points inside a retained ball |
| `retention_resolution_by_class_size.csv` | `analyse_retention_resolution.py` | rare-class size against density z |
| `density_scale_bound.json` | `check_density_scale_bound.py` | the `k/(k-m+1)` ceiling of the density index |

`run_compare.py` prints the tabular comparison and writes no file. It needs the
small UCI/OpenML sets, which `src/datasets.py` downloads through their public
interfaces.

The CIC-IDS2018 scripts must be run for both days before pooling:

```bash
python experiments/run_cicids2018_temporal_compare.py --days 2018-02-22 --tag 0222
python experiments/run_cicids2018_temporal_compare.py --days 2018-02-23 --tag 0223
python experiments/combine_cicids2018_sets.py
```

## Constants and protocol

The registered setting is `k = 10`, `tau_gamma = 0.5`, `tau_iso = 0.0`,
`s_min = 3`, `delta = 0.25`, `eta = 1.5`, `beta = 0.5`, with the inherited
size-ratio guard disabled in both methods so that the two share one graph
policy. The thresholds are fixed once and are not tuned per data set. Labels
are not used at fit time, and `k` is never set to the ground-truth cluster
count.

The UNSW-NB15 holdout is a disjoint set of twenty blocks that was drawn before
the replication results were inspected and was scored once, at the default
configuration. The CIC-IDS2018 protocol `CIC-IDS2018-TEMPORAL-20261024-v1` was
frozen before any number from that source was computed; every eligible window
is evaluated, and the four pre-declared criteria are reported as they came out.

## Citation

If you use this code, cite the paper (see `CITATION.cff`) and, for the boolean
baseline, Wang and Zhu's ARP-GB, which `src/arp_gb.py` reconstructs.

## Licence

MIT. See `LICENSE`.