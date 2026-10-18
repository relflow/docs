# Categorical predictive-grouping combination diagnostics

This archive preserves all twelve fixed 600-update fits, their development
records, checkpoint hashes, complete stdout/stderr logs, and the sealed fresh
audit for seeds 8001–8003. Canonical P080 and production source were unchanged.
No failed arm or seed is omitted. None of the three added recipes improves
categorical reliability; native predictive training remains the strongest
recipe here and meets every diagnostic gate in two of three seeds.

## Recorded comparison

Accuracy figures are fractions, rounded to four decimals in these tables.
`SUMMARY.json` retains full precision. Gate counts require every recorded
gate, including ARI≥.80, accuracy≥.90 and exactly five occupied groups.

| Recipe | Development accuracy mean [min, max] | Fresh accuracy mean [min, max] | All gates: development / fresh |
| --- | --- | --- | --- |
| Native predictive | 0.8988 [0.7815, 0.9820] | 0.9004 [0.7832, 0.9852] | 2/3 / 2/3 |
| Deterministic ST | 0.6592 [0.5955, 0.7835] | 0.6580 [0.5928, 0.7848] | 0/3 / 0/3 |
| ST + .1 initialization | 0.6210 [0.5930, 0.6745] | 0.6155 [0.5868, 0.6676] | 0/3 / 0/3 |
| ST + .1 initialization + margin 1 | 0.6722 [0.5950, 0.7860] | 0.6656 [0.5872, 0.7860] | 0/3 / 0/3 |

Seed order is 8001 / 8002 / 8003. ARI and occupancy concern the learned
known-ID mapping and therefore agree between development and fresh audit.

| Recipe | Fresh accuracy per seed | ARI per seed | Occupied groups per seed |
| --- | --- | --- | --- |
| Native predictive | 0.9852 / 0.7832 / 0.9328 | 1.000000 / 0.698647 / 1.000000 | 5 / 6 / 5 |
| Deterministic ST | 0.7848 / 0.5964 / 0.5928 | 0.767773 / 0.595041 / 0.529349 | 4 / 3 / 4 |
| ST + .1 initialization | 0.5868 / 0.6676 / 0.5920 | 0.595041 / 0.767773 / 0.553493 | 3 / 4 / 4 |
| ST + .1 initialization + margin 1 | 0.5872 / 0.7860 / 0.6236 | 0.595041 / 0.767773 / 0.710059 | 3 / 4 / 5 |

Fresh shuffled-ID accuracy ranges from .1944 to .2076; chance is .20.
Every evaluation-resource freeze check passes. Margin-1 proposal row-logmass
drift is at most 2.384185791015625e-7. These are diagnostic gate counts, not
new complete canonical proof passes: no newly fitted no-regime controls were
included for the added recipes. Native/hardening/no-regime controls from the
original P080 panel remain separate historical evidence.

## Frozen method and information boundary

The matrix was fixed in protocol.json before any development outcome was
inspected: native hard Gumbel, deterministic hard ST, ST with .1 normal
assignment initialization, and the latter with margin-1 accepted proposals.
The Group-only runtime patches are explicit in run.py; PATCHES.json maps
each target to its implementation. Shared decoding/writing/loss semantics
remain raw tied identity scoring. No teacher labels or normalized likelihood
mixture is part of this four-arm study.

Small initialization scales the same original normal draws and only new-ID
rows on growth, preserving existing learned rows and the moved sentinel.
Counterfactual scoring stays hard in all arms. Only the last arm's proposed
rows use normalized 0/-1 logits plus their prior row logsumexp; baseline
proposals preserve canonical 0/log(1e-8) behavior and its tiny mass deviation.

These are paired-seed training-recipe comparisons, not perfectly RNG/mask
matched causal routing ablations. Removing Gumbel draws changes subsequent
Torch RNG consumption, including any dropout or mask streams.

No original P080 test stream (seed*100+3) was loaded during this study.
The harness imports P080 generators/helpers but never invokes P080.run.
After all twelve fits and checkpoint files were frozen, a separate phase
required all matching records before generating 2,500 fresh observations
at seed*100+503 and their paired ID permutation. Every arm was then scored
without further fitting or selecting a recipe. This is fresh-observation
generalization for known identities/context cells, not unseen-ID transfer.

## Archive contents and reproduction

Measured raw JSON and complete logs live under measured/. Checkpoints are
intentionally omitted; CHECKPOINT_SHA256.json and each raw record retain
their verified hashes. No caches, locks, symlinks or duplicate full package
snapshots are included. Group and Category baseline modules are included
under runtime-source/ for direct inspection; data/scoring helpers are in
the frozen proof_source.py. Only compressed Python sources are published.

The authoritative full package snapshot already appears once in the
P079–P081 expansion archive. SOURCE_REFERENCE.json records its archive
SHA-256 `adab27f329b5909aa39a2390fd683b7b9d72b287ac871a87c99a3459a6f4c6fe` and package source fingerprint
`4f900097c5f12eb6eef07e88b7abba5c61089f52696d52ee8c08d105894a5b58`. Obtain that archive alongside this one;
extract its snapshot/ directory into this harness's snapshot/ path in a
NEW working copy. Use the Python/dependency versions in ENVIRONMENT.json.
Installing a released package merely named 0.3.3 does not establish source
identity. Historical absolute paths in measured records are preserved.

Do not overwrite the measured/ evidence. In the new working copy, run:

```sh
/path/to/env/bin/python launch.py --seed 8001
/path/to/env/bin/python launch.py --seed 8002
/path/to/env/bin/python launch.py --seed 8003
/path/to/env/bin/python run.py holdout
```

Training writes new records/checkpoints at the harness root, separate from
measured/. Holdout refuses incomplete or changed protocol/harness records.
Wall times came from concurrent experiments and are not CPU/GPU throughput
benchmarks. The archive and every file were read back and SHA-256 verified;
tar ownership/mode/mtime and gzip timestamp are normalized. FILE_SHA256.json
lists internal payload hashes excluding itself; the sibling manifest.json
also verifies that internal manifest and records every member's origin.
