# Predictive-grouping expansion evidence: P079–P081

This local review archive preserves the measured sources, raw data, stdout/stderr
logs, reporting helpers, protocols and current installed runtime contract tests
for three independent proofs. It contains no trained checkpoints, cached Python,
lock files, later combination experiments, or verification scratch directories.
The package source appears once under `snapshot/`; per-proof duplicate snapshots
are deliberately omitted. No documentation is published by extracting this file.

## Recorded evidence

- P079: `numeric/proofs/group/predictive_regimes.py`, 18 fits across six predefined
  numeric processes and seeds 7901–7903, each 600 updates. Reporter rows are in
  `numeric/p079-results.yaml`; separate raw JSON/logs and `numeric/summary.json`
  retain case metrics and failed gates. All three complete runs remain `not_met`.
- P080: `categorical/proofs/group/predictive_labels.py`, 12 fits across native,
  hardening, predictive and no-regime arms at seeds 8001–8003, each 600 updates.
  `categorical/results.yaml`, `results.json`, and per-seed logs retain every
  measured result. Outcomes are met / not_met / met; no failed seed is removed.
  The separate one-update smoke output is also retained as smoke evidence only.
- P081: `cost/proofs/group/reassignment_cost.py`, six matched CPU fits across
  seeds 7801–7803, plus two excluded two-update warmup fits. Full reporter rows
  are in `cost/proofs/results.yaml`; raw JSON/logs are `matched-cost.json/log`.
  Native and candidate mean wall times are 10.070 and 12.950 seconds; the ratio
  of sums gives **28.60% additional wall time**, including update-600
  reassignment. This is not a FLOP percentage, a GPU result, or a measurement
  of later combinations. See `TIMING_PROTOCOL.md` and `COST_RESULTS.md`.

The shared package Python-source fingerprint is
`4f900097c5f12eb6eef07e88b7abba5c61089f52696d52ee8c08d105894a5b58`.
This is an unreleased 0.3.3 development source snapshot, not a published release
commit. The original directories, timestamps and absolute paths in reporter
provenance/manifests remain historical evidence; paths may differ after unpacking.
The exact source/module and script hashes identify the measured code.

`integration/preserved-registry-prefix.sha256` records SHA-256
`96920095081621e092c922e1a153b297f5a93b6a84c2622a8e76620fc21c455b`
of the 1,853,580-byte pre-expansion canonical results registry. The installed
registry retains that prefix unchanged before the appended P079–P081 sections.
`integration/test_group_reassignment_proofs.py` is copied from the current
installed review docs checkout; it is not an import-time training requirement.
The integration installer/SPEC/page are preserved for provenance and are not
needed to reproduce measured model runs.

## Reproduce deliberately

Unpack into an empty directory. Use Python 3.12.10 and an environment matching
the dependency versions recorded in each reporter row, then install the local
`snapshot` package (including its `pyproject.toml`, `README.md` and `LICENSE`).
The measured primary versions include Torch 2.12.0, Lightning 2.6.4, NumPy 2.4.6,
PyArrow 24.0.0, Polars 1.44.1, TorchMetrics 1.9.0, TensorDict 0.12.4 and
Pydantic 2.13.4. The archived source snapshot is authoritative for RelFlow;
installing a released package named 0.3.3 would not establish source identity.
Results can vary with hardware or other environment differences.

For a fresh P079/P080 reproduction, remove no recorded evidence: point `--output`
to a new temporary path. Set `PYTHONPATH` to the proof's own `proofs` directory
and the single shared `snapshot/src`, then run the corresponding script with
`--seed`, `--accelerator cpu`, `--threads 1`, and that fresh `--output`.
For example, from the unpacked archive root:

```bash
PYTHONPATH="$PWD/numeric/proofs:$PWD/snapshot/src" /path/to/env/bin/python \
  numeric/proofs/group/predictive_regimes.py --seed 7901 --threads 1 \
  --accelerator cpu --output /tmp/new-p079-results.yaml
PYTHONPATH="$PWD/categorical/proofs:$PWD/snapshot/src" /path/to/env/bin/python \
  categorical/proofs/group/predictive_labels.py --seed 8001 --threads 1 \
  --accelerator cpu --output /tmp/new-p080-results.yaml
```

Each fresh reporter output starts from its corresponding minimal registry,
which has no runs. Native/positive/control comparisons use training and
validation data for fitting/acceptance; official test observations are generated
only after the frozen per-fit method completes. The P081 timing proof generates
no test split. No variant is selected by its reported test gates inside these
scripts. Continued research after viewing these results must retain that history.

`run_cost_panel.py` reproduces P081 with `snapshot/src` automatically. Reserve an
otherwise idle CPU window before using `--timing-isolation-confirmed`. It appends
to `cost/proofs/results.yaml`; **work on a copy of the extracted directory** so
original measured evidence is not overwritten. Its JSON output path must not
already exist. The historical absolute paths in `TIMING_PROTOCOL.md` should be
replaced with the unpacked location when invoking the driver.

## Integrity and scope

The sibling publication `manifest.json` records the archive SHA-256 and the
SHA-256, byte size and recorded origin of every member file. The archive also
contains `FILE_SHA256.json`, an internal payload manifest which excludes itself.
Archive member metadata (UID/GID/mode/mtime and gzip timestamp) are normalized;
no symlinks are included. Every member was read back and verified after creation.
Only files inside the compressed archive are Python source artifacts; copying
these three publication files into docs diagnostics adds no loose proof scripts.
