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
`O(m^2 d + m^2)` time and `O(m^2)` storage on the existing ball graph, with
`m <= 2*ceil(sqrt(n))`. When no ball satisfies the predicate, the method returns
the inherited partition exactly.

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

The nine tests take about a second and need no data. They cover the fixed graph
policy, the admissibility condition on fragment edges, the exact fallback to
the baseline when no rare core is retained, determinism, the partition
invariant, and the two consequences of the corridor-vacuity lemma: at the
registered constants the witness count cannot change the edge set, and the
corridor clause is implied by the closeness clause. A third corridor test
confirms the lemma's hypothesis is sharp, by showing that at `beta_c = 2` the
clause does reject a pair.

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
| `robustness_matched.csv`, `robustness_matched.png` | `run_robust.py`, `plot_robustness.py` | perturbation curves, matched graph policy |
| `recent_paper_baselines.csv` | `run_recent_paper.py` | NGBC and EKM runs |
| `unsw_nb15_temporal_blocks_v2_matched.csv`, `unsw_nb15_temporal_summary_v2_matched.json` | `run_unsw_nb15_temporal_compare_v2.py --arp-size-gate false --output results/unsw_nb15_temporal_blocks_v2_matched.csv` | per-block UNSW-NB15 results |
| `unsw_holdout_blocks_matched.csv` | `holdout_validate_z10.py` | the twenty holdout blocks, scored once |
| `unsw_pooled_matched.json` | `pool_unsw_matched.py` | pooled UNSW-NB15 blocks |
| `locality_diagnostic.csv` | `run_locality_diagnostic.py` | cross-component reassignment |
| `coupling_grid_guard_raw.csv`, `coupling_grid_heatmap.pdf` | `run_coupling_grid_final.py`, `plot_coupling_grid_final.py` | joint `delta`/`eta` grid |
| `cicids2018_temporal_blocks_pooled.csv`, `cicids2018_temporal_summary_pooled.json` | `run_cicids2018_temporal_compare.py` then `combine_cicids2018_sets.py` | per-window and pooled CIC-IDS2018 blocks |
| `cicids2018_mechanism_diagnosis.json` | `diagnose_cicids2018_mechanism.py` | per-block ball geometry |
| `retention_hit_rate.json` | `diagnose_retention_hit_rate.py` | share of a block's rare points inside a retained ball |
| `retention_resolution_by_class_size.csv` | `analyse_retention_resolution.py` | rare-class size against density z |
| `density_scale_bound.json` | `check_density_scale_bound.py` | the `k/(k-m+1)` ceiling of the density index |
| `corridor_witness_equivalence.json` | `verify_corridor_witness_equivalence.py` | five corridor configurations on every reported cohort |
| `openml_heldout_screen_log.json` | `register_openml_heldout.py --screen-only` | every drawn id, its skip reason, and the admitted cohort |
| `openml_heldout_sets.csv`, `openml_heldout_summary.json` | `run_openml_heldout.py` | the held-out OpenML cohort, scored once at the frozen defaults |

`verify_corridor_witness_equivalence.py` re-runs the frozen source files over every
reported unit of the two real sources — 20 UNSW-NB15 replication blocks, 20
UNSW-NB15 holdout blocks and the 396 CIC-IDS2018 windows that admit a candidate
pair — under five configurations of the fragment rule: the registered witness count
`W = 3`, `W = 1`, `W = 2`, the exact full scan, and a corridor factor that
disables the clause. It reads the frozen result files without modifying them, and
reports two things. All five configurations agree on every unit, with a metric
difference of exactly `0.0` from the registered one. And the registered sources
reproduce the frozen per-block numbers to within `5e-13` on ARI, NMI and rare
recall and exactly on ACC, many-to-one ACC, cluster count and rare-core count.
That reproduction is the reason the frozen files stay in `results/` even though
the model sources have been refactored since they were written;
`results/UNSW_PROTOCOL_ERRATA.md` records the affected hashes.

`run_compare.py` prints the tabular comparison and writes no file. It needs the
small UCI/OpenML sets, which `src/datasets.py` downloads through their public
interfaces.

The CIC-IDS2018 runner is executed twice, in this order, before pooling: the
first execution is restricted to 2018-02-22 and its outputs are written before
the second one starts, and the second covers the remaining eight days. The
`--tag` values are the ones the frozen result files carry.

```bash
python experiments/run_cicids2018_temporal_compare.py --days 2018-02-22 --tag replication
python experiments/run_cicids2018_temporal_compare.py --days 2018-02-14,2018-02-15,2018-02-16,2018-02-21,2018-02-23,2018-02-28,2018-03-01,2018-03-02 --tag holdout
python experiments/combine_cicids2018_sets.py
```

## Constants and protocol

The registered setting is `k = 10`, `tau_gamma = 0.5`, `tau_iso = 0.0`,
`s_min = 3`, `delta = 0.25`, `eta = 1.5`, `beta = 0.5`, `W = 3`, with the
inherited size-ratio guard disabled in both methods so that the two share one
graph policy. The thresholds are fixed once and are not tuned per data set.
`W` is the number of background witnesses per endpoint in the corridor
certificate of the fragment rule; it is an implementation constant that keeps
the corridor test at `O(d)` per candidate pair, and `W = None` certifies against
every background centre instead. At the registered `beta = 0.5` the corridor
clause is implied by the closeness clause — the midpoint of an admissible pair
is at least `1.5 d` from every background centre, three times the `0.5 d` the
clause requires — so the witness count cannot change the edge set and neither
can the clause itself. `tests/test_mg_gbc_invariants.py` checks both statements,
and `experiments/verify_corridor_witness_equivalence.py` checks them on every
reported cohort. Labels are not used at fit time, and `k` is never set to the
ground-truth cluster count.

The UNSW-NB15 holdout is a disjoint set of twenty blocks that was drawn before
the replication results were inspected and was scored once, at the default
configuration. The CIC-IDS2018 protocol `CIC-IDS2018-TEMPORAL-20261024-v1` was
frozen before any number from that source was computed; every eligible window
is evaluated, and the four pre-declared criteria are reported as they came out.

## Held-out cohort

`experiments/openml_heldout_protocol_v1.json` (`OPENML-HELDOUT-20261024-v1`)
fixes a cohort and four decision criteria before the cohort is scored. Membership
is label-free: every data set in OpenML ids 1-10000 is visited in a seeded
permutation (`seed = 20261024`), and a data set is admitted when it is active, in
ARFF, has 300-20000 rows, at least two classes and 3-100 numeric features, and
comes from a source not already admitted, sources being deduplicated on the first
token of the name with digits stripped. Ids and names from the development
history are excluded by a list written into the protocol. There is no cap, so the
cohort is the whole admissible population of the id block and its size is a
property of the repository rather than a parameter. The smallest class size is not
a membership condition: it is read once, after the cohort is fixed, to place each
member in the rare subset (smallest class below `sqrt(n)`) or in the locality
control (no class below `sqrt(n)`). A set whose smallest class is below
`s_min = 3` stays in the rare subset and counts against the method.

The freeze and the scoring are reproducible from the recorded screen log, without
network access for the freeze:

```bash
python experiments/register_openml_heldout.py --from-log results/openml_heldout_screen_log.json
python experiments/run_openml_heldout.py
```

`--from-log` rebuilds the protocol from the log and checks that the log's draw
order is the seeded permutation, so the cohort cannot depend on which metadata
fetches happened to succeed. `run_openml_heldout.py` verifies the model hashes
recorded in the protocol, refuses to overwrite an existing summary, and reports
all four criteria whether or not they are met.

The cohort was scored once, at the registered defaults: 95 data sets, 188 method
runs, one load failure. Two criteria are met. ARI is non-inferior on the rare
subset, with a median change of `+0.0052` and a worst case of `-0.0139` against
a required margin of `-0.05`, and no rare-subset set falls below that margin.
Rare-class recall is not established: five of the nine rare-subset pairs tie,
and at the four non-tied pairs the smallest attainable two-sided sign-test `p`
is `0.125`, so no outcome of this cohort could have met that criterion. The
locality criterion is not met: 24 of the 85 control sets return an identical
partition, because the predicate reads ball size rather than class size and 49
of the 85 control sets retain a core. `results/openml_heldout_summary.json`
carries the four verdicts with the statistics behind them, and
`results/openml_heldout_sets.csv` the per-set rows.

## Citation

If you use this code, cite the paper (see `CITATION.cff`) and, for the boolean
baseline, Wang and Zhu's ARP-GB, which `src/arp_gb.py` reconstructs.

## Licence

MIT. See `LICENSE`.