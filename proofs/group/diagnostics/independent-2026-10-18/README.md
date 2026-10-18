# Group-owned default refinement: original proof regression panel

Frozen SDK source: `f2c31b6fd12d0eecad3edee0a80079385ac5c7970d3e30c63deed5e543d24e9c`. Original proof scripts, budgets, gates, train/validation/test seeds and native validation cadence are unchanged. Six proof invocations train nine models on CPU with one thread and default zero-worker loaders.

| Proof | Seed | Outcome | Main held-out result |
| --- | ---: | --- | --- |
| P065 | 7651 | not_met | Known accuracy at full/50%/10% coverage: 76.95%/77.34%/77.45% |
| P065 | 7652 | not_met | Known accuracy at full/50%/10% coverage: 75.10%/75.00%/81.37% |
| P065 | 7653 | not_met | Known accuracy at full/50%/10% coverage: 63.67%/63.67%/59.80% |
| P078 | 7801 | not_met | Stable ARI 0.6927; normalized RMSE 0.2466; occupied groups 6 |
| P078 | 7802 | met | Stable ARI 0.8901; normalized RMSE 0.0313; occupied groups 6 |
| P078 | 7803 | not_met | Stable ARI 0.7088; normalized RMSE 0.2279; occupied groups 7 |

All three P065 runs miss the unchanged >95% known-identity reconstruction gates. Coverage/count/OOV, rebatching, and frozen mapping/exposure controls pass.

P078 seed 7802 passes every original gate; seeds 7801 and 7803 miss only ARI >=0.80. All three stable arms pass normalized RMSE <=0.35 and all negative-control/coverage/frozen-assignment gates. Exact group count is not an original P078 gate.

Default Group refinement uses its own identity likelihood exclusively. Its observations and validation acceptance are separate from ordinary supervised gradient updates. The P078 own-likelihood proposals were all rejected; improvements cannot be attributed to accepted Group refinement.

`results.yaml` contains every original report, including all failed gates. `summary.json`, `readout.json`, per-invocation logs, and `refinement-events.json` provide compact evidence. No test observations select updates or follow-up settings. Original split streams have been examined previously, so these are regression results, not a fresh unbiased holdout.

The earlier directory `/private/tmp/relflow-group-independent-20261018` preserves six execution errors from a superseded lifecycle-guard snapshot. It contains no valid learning claims.


## Objective and unchanged protocol

This is the integrated **Group-owned identity NLL** recipe. It does not call
another field's loss or score, inspect supervised target error to choose
memberships, or add an external scoring API. Normal supervised backpropagation
can still train visible Group assignments and embeddings through the ordinary
model graph. This differs from the earlier downstream-selected P078/P079/P080
recipes; their reported recovery rates and 28.60% timing cost do not apply.

P065 uses the original 800 AdamW updates, batch 64, learning rate 0.002 and
8,192 training rows. Each P078 seed trains both original stable and no-regime
arms for 600 updates, batch 128, learning rate 0.003 and 4,096 training rows.
Training, validation, and held-out generators retain their original independent
streams. No checkpoint selection, shortened budget, relaxed gate, or further
training was introduced. The native validation cadence produces P065 refinement
attempts at updates 256/512/768 and P078 attempts at 224/448 in each arm.

Three of nine P065 proposals are accepted. All twelve P078 proposals are
rejected by Group's own validation NLL. The latter task deliberately makes
opaque identity unidentifiable from independent visible x when identity is
masked and y remains hidden. Passing predictive gates therefore cannot be
attributed to accepted own-objective refinement.

Every original failed check remains in `results.yaml`, `summary.json`, and
`readout.json`. There were zero execution errors in this final frozen panel.
Model training elapsed times are reported for diagnosis, without a matched
baseline or a GPU overhead claim.

## Reproduce

`experiments.tar.gz` contains the exact frozen SDK Python source, unchanged
proof scripts and reporter, standalone result registry, runner, raw logs, and
measured evidence. `manifest.json` hashes the archive, its members, and the
review files. No trained checkpoints, caches, or credentials are included.

Use a prepared Python environment matching the package versions recorded in
`results.yaml`, extract the archive, and select the frozen source explicitly:

```bash
tar -xzf experiments.tar.gz
export PYTHONPATH="$PWD/source/src:$PWD/proofs"
python proofs/vocabulary/group_pretraining_coverage.py --seed 7651 --accelerator cpu --threads 1 --output fresh-results.yaml
python proofs/group/regime_recovery.py --seed 7801 --accelerator cpu --threads 1 --output fresh-results.yaml
```

Repeat P065 with seeds 7652/7653 and P078 with seeds 7802/7803. Omit `--steps`
to retain the original full budgets. Do not replace this recorded evidence
with the new output. The canonical documentation `proofs/results.yaml` and the
published website remain unchanged.


## Post-freeze runtime guards

After the frozen `f2c31b6fd12d0eecad3edee0a80079385ac5c7970d3e30c63deed5e543d24e9c`
proof panel, the final SDK workspace added two guards: defer refinement while
an optimizer update has unfinished gradient accumulation, and skip calibration
and refinement when no Group has its own reconstruction objective. All nine
recorded fits used `accumulate_grad_batches=1` and a reconstructing Group, so
neither guard changes this panel's execution. No learning rerun was needed.
Frozen source fingerprints, measurements, and the archived source/log bundle
remain unchanged; this note describes the final workspace's guard-only
differences.
