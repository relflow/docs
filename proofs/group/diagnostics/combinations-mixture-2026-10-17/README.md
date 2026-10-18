# Frozen normalized-mixture + predictive-reassignment add-on

The additional bundled arm does not solve grouping across all tasks. It passes all fresh gates for all three crossing-curve seeds, one of three six-group seeds, and one of three categorical seeds. Every failed count remains failed. No source or method was changed after outcomes.

| Process | Seed | Validation score | Fresh score | Fresh ARI | Occupied | All gates |
|---|---:|---:|---:|---:|---:|---|
| crossing_four | 7901 | 0.062025 | 0.061689 | 1.000000 | 4 | met |
| separated_six | 7901 | 0.177092 | 0.174837 | 0.831624 | 5 | not met |
| crossing_four | 7902 | 0.064522 | 0.063556 | 1.000000 | 4 | met |
| separated_six | 7902 | 0.172368 | 0.178011 | 0.831624 | 5 | not met |
| crossing_four | 7903 | 0.053772 | 0.054288 | 1.000000 | 4 | met |
| separated_six | 7903 | 0.032780 | 0.032207 | 1.000000 | 6 | met |
| contextual_labels | 8001 | 0.983000 | 0.985600 | 0.854317 | 7 | not met |
| contextual_labels | 8002 | 0.980000 | 0.981200 | 1.000000 | 5 | met |
| contextual_labels | 8003 | 0.983000 | 0.980000 | 0.928882 | 7 | not met |

Scores are source-unit normalized RMSE for numeric cases (lower is better), accuracy for categorical cases (higher is better). All shuffled-ID, known-exposure, finite-score and frozen-parameter/buffer/vocabulary/assignment checks pass. All four failures are exact occupied-count gates.

The method preserves the prior experiment’s exact column-normalized identity likelihood, balanced index-modulo-slot initialization at logit .1, deterministic straight-through routing and standard Sinkhorn balance. Predictive membership search and the hard proposal margin remain canonical. This bundles multiple changes; comparing it with small-normal-init ST does not isolate normalized likelihood alone.

## Matched fresh comparison with the existing four-arm matrices

Cells show number of three seeds passing both quality/ARI gates, followed by number passing the exact occupied count. Numeric fresh split is 10003; categorical fresh seed offset is 503. Original +3 test rows are never loaded.

| Arm | Crossing four quality/count | Separated six quality/count | Categorical quality/count |
|---|---|---|---|
| Native Gumbel + predictive | 3/3 / 2/3 | 2/3 / 0/3 | 2/3 / 2/3 |
| Deterministic ST + predictive | 3/3 / 3/3 | 3/3 / 2/3 | 0/3 / 0/3 |
| ST + small normal init + predictive | 3/3 / 2/3 | 3/3 / 3/3 | 0/3 / 0/3 |
| ST + small normal init + margin1 proposal | 3/3 / 3/3 | 3/3 / 3/3 | 0/3 / 1/3 |
| ST + balanced small init + normalized mixture | 3/3 / 3/3 | 3/3 / 1/3 | 3/3 / 1/3 |

The existing ST/small-normal/margin1 arm is stronger on six-group recovery. The extra mixture arm restores high categorical predictive quality for all seeds but fragments two partitions into seven occupied slots. Native Gumbel predictive reassignment still clears all categorical gates in two seeds. These comparisons are diagnostic; no winner is selected or retrained from fresh holdout outcomes.

## Reproduction and provenance

`protocol.json` and `freeze.json` were written before this add-on’s first outcome. All nine 600-update training/validation fits finished before separate fresh-audit generation. The raw frozen source fingerprint is 4f900097c5f12eb6eef07e88b7abba5c61089f52696d52ee8c08d105894a5b58. The numeric/categorical canonical proof functions are retained as `p079.py` and `p080.py`; original test-running `run()` entrypoints are not called.

`run_balanced_original.py` preserves the old experiment for attribution. The exact `hard`, `mixture`, `init`, and `grow` numerical function ASTs match that source. Existing K1 nonuniform-prior and bf16 writer/loss tests were run **inside the experimental mixture scorer patch** before training. Separate direct analytic calls in `check_mixture_math.py` verify float32 and bf16 inputs, .9/.1 posterior retention, float32 output, NLL/readout agreement and finite gradients; `mixture-path-check.json` retains values. These checks do not change the frozen harness.

Use a matching recorded Python/dependency environment and the raw source snapshot in the separate P079–P081 expansion archive (`experiments.tar.gz`, SHA-256 `adab27f329b5909aa39a2390fd683b7b9d72b287ac871a87c99a3459a6f4c6fe`). This compact archive omits that duplicate snapshot. In a fresh working copy of these files, run `run.py regression`, then `run.py train`, then `run.py audit`. Train/audit refuse existing output paths, so retain the original evidence and use a copied directory with recorded outputs removed from the copy only. Fresh audit validates all nine fit counts and source/harness/protocol/checkpoint hashes before scoring. The separate checks script is optional for reproducing the analytic path evidence.

The local archive excludes checkpoints and caches; reproduction requires retraining. Raw checkpoint hashes remain recorded. No production code, dependencies, canonical proofs, quality thresholds, docs, commits or deployments changed. CPU timings were collected under concurrent load and are not a speed benchmark.

The historical `run.py regression` entrypoint refers to the original local package test path. For a relocated archive, use the separate direct `check_mixture_math.py` check or provide the matching original package tests. Frozen measured files and absolute provenance paths are preserved unchanged.
