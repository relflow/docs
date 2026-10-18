# Group 0.3.3 local review

## Current Group-owned implementation, 2026-10-18

The current public count option is `n_groups`: an integer fixes group width,
and a `(lower, upper)` pair keeps the existing adaptive range semantics.
`bounds` is no longer accepted and has no compatibility alias. Current
examples and executable proof constructors use `rf.Group(n_groups=...)`. This
API-only rename changes neither training computation nor the learning recipe;
no learning run or compute comparison was repeated for it. Historical examples
below, recorded results, diagnostics, frozen proof sources, and source
fingerprints retain the spelling and provenance of their original runs.

The local SDK now enables normalized identity likelihood, balanced
initialization, deterministic straight-through routing, and automatic
membership refinement using **Group's own identity reconstruction objective**.
Group owns its targets, proposal, and acceptance. No shared score component,
other-extension scoring API, or dependency was introduced. The intermediate
implementation that consulted other fields' supervised scoring hooks was
withdrawn and is not the promoted recipe.

For each nonretired group, identity scores normalize over populated vocabulary
rows. Contextual group probabilities mix these identity distributions; Group
reconstruction and writing use the same likelihood. This column normalization
preserves learned identity-frequency priors, unlike the earlier rejected row
normalization. New identity rows cycle across available groups with score
`0.1` on one group and zero elsewhere. Growth preserves existing rows and the
unavailable row. Training and evaluation route deterministically to the current
argmax group, with straight-through softmax gradients during training.

Refinement requires Group's own reconstructing mask. A plain visible Group
still receives ordinary gradients from downstream supervised objectives;
without its own reconstruction objective, it does not run refinement or update
its reconstruction-driven adaptive balancing state. Other field losses are
not called by reassignment, but can shape Group through the existing training
graph. A model whose only objective is `Group(mask=True)` can refine without
any other supervised field or extension.

The proposal freezes Group's training-context probabilities and evaluates each
identity row's candidate membership with all other identity rows held fixed in
the column-normalized likelihood. It proposes uniquely supported minima, then
recomputes Group's actual validation predictions with the joint proposal
installed. Acceptance requires a strict improvement in the mean NLL of Group's
own known selected identities, with unchanged target coverage. Unobserved or
unknown identities and tied candidate minima provide no new membership evidence.
Accepted replacements preserve identity row log-mass and the unavailable row,
and clear only their overwritten gradient and optimizer-history rows, retaining
scalar steps. Rejection and evaluation failure restore the original mapping.

The automatic callback runs at ordinary fitting validation start after at least
200 optimizer updates since the previous evaluation. The ensuing validation and
checkpoint scores therefore describe the accepted mapping. Standalone
validation, testing, and prediction never run refinement. Calibration stores
canonical Arrow observations after preprocessing, bounded across ranks to
4,096 training rows / 64 MiB and 1,024 first-loader validation rows / 16 MiB.
Re-encoding does not rerun preprocessors or learn vocabulary. Selected Group
coordinates provide evidence independently, including multiple identities per
root row and repeated nested geometry. Frozen assignments, active overrides,
missing own objectives, absent independent validation, or no known target
evidence cause a recorded skip or rejection reason. Decisions coordinate across
DDP ranks; evaluation modes and random state are restored.

Focused runtime tests cover the Group-only objective, unrelated third-party
loss and decoder isolation, ordinary supervised gradients, accepted and rejected
updates, optimizer state, nested coordinates, and actual training/checkpoint
freshness. These tests establish runtime contracts, not useful group discovery.
Fresh original P065/P078 regression runs are now complete. The dated readout
below preserves their original gates and reports every failed seed.

Earlier P078 diagnostics, P079/P080 reassignment results, and the normalized
mixture add-on used downstream target-error acceptance. Their categorical
accuracy and partition scores do **not** establish this Group-owned objective's
quality. The original matched 28.60% CPU overhead also describes a different
callback; no cost or GPU speed claim is made for the integrated recipe. Exact
group count, minority-group recovery, unseen identities, and reconstruction
quality still require separate evidence.

## Fresh original-proof regression panel, 2026-10-18

The final Group-owned implementation was frozen before six original proof
invocations, comprising nine model fits. The unchanged P065 source ran seeds
7651/7652/7653 for its original 800 updates. The unchanged P078 source ran seeds
7801/7802/7803, each training the original stable and no-regime arms for 600
updates. Original data streams, learning rates, batch sizes, native validation
cadence, and every gate were retained. CPU execution used one thread and the
original zero-worker data module defaults. All six invocations completed
without an execution error.

Frozen full-package Python source SHA-256:
`f2c31b6fd12d0eecad3edee0a80079385ac5c7970d3e30c63deed5e543d24e9c`.

After this freeze, the final SDK workspace added two runtime guards: defer
refinement during unfinished optimizer gradient accumulation, and skip
calibration/refinement when no Group has its own reconstruction objective.
All nine fits used `accumulate_grad_batches=1` and a reconstructing Group, so
neither guard changes this panel's execution. The frozen source fingerprint,
archive, and measurements remain unchanged.

| P065 seed | Known accuracy: full coverage | 50% coverage | 10% coverage | All original gates |
| --- | ---: | ---: | ---: | --- |
| 7651 | 76.95% | 77.34% | 77.45% | Fail |
| 7652 | 75.10% | 75.00% | 81.37% | Fail |
| 7653 | 63.67% | 63.67% | 59.80% | Fail |

All three runs miss only the unchanged known-answer accuracy gates above 95%.
Exact known/unavailable counts, coverage, all-answer accounting, undefined
all-OOV conditional scores, rebatching agreement, and frozen vocabulary/exposure
controls pass. Group reconstruction remains unresolved at this budget.

| P078 seed | Stable ARI | Stable normalized RMSE | Occupied groups | All original gates |
| --- | ---: | ---: | ---: | --- |
| 7801 | 0.6927 | 0.2466 | 6 | Fail: ARI below 0.80 |
| 7802 | 0.8901 | 0.0313 | 6 | Pass |
| 7803 | 0.7088 | 0.2279 | 7 | Fail: ARI below 0.80 |

Every stable arm clears the original normalized-RMSE gate at most 0.35.
No-regime ARI is 0.0347 / 0.0055 / -0.0115, with normalized RMSE
1.0022 / 1.0050 / 1.0045. All negative-control, shuffled-ID dependency,
finite-value, full-exposure, and frozen-assignment checks pass. Exact group
count has no original P078 success gate; six or seven occupied groups must not
be described as recovery of the four generating groups.

The integrated callback made nine own-identity-NLL proposals across P065,
accepting three, and twelve across the six P078 arms, accepting none. P065's
native cadence produced attempts at updates 256/512/768; P078's at 224/448.
Its rejected P078 proposals are consistent with the stated observability
limit: an opaque masked identity cannot be inferred from independent visible x
when the behavioral target y remains hidden. The improved supervised
predictions therefore cannot be attributed to accepted own-objective
refinement. Ordinary supervised gradients, normalized identity likelihood,
initialization, and deterministic routing remain bundled in this implementation;
this panel does not isolate their effects.

This is **Group-owned refinement**, distinct from historical reassignment that
selected memberships using another field's supervised MSE or NLL. No external
score API or other tensorfield loss is consulted by the current refinement.
The older downstream-selected categorical results and 28.60% CPU timing figure
do not establish this implementation's quality or cost. These familiar
synthetic split streams provide regression evidence, not a new unbiased
holdout or RelArena benchmark claim. No held-out score selected a new recipe,
budget, or follow-up fit.

The [standalone evidence directory](diagnostics/independent-2026-10-18/README.md)
contains every raw report and failed gate, callback events, protocol, hashes,
and a reproducible source/log archive. Its `results.yaml` is separate from the
canonical `proofs/results.yaml`, which remains unchanged. An earlier lifecycle
guard execution failure is recorded as an excluded attempt and retained in
temporary local storage; it contributes no learning claim. Nothing was
published.

## Historical experimental record

All sections below retain their original measurements, source fingerprints,
and protocol descriptions. Statements about retained raw scores, proof-local
callbacks, or unchanged production defaults describe those historical snapshots
and are superseded for the current SDK by the implementation description above.
The canonical proof sources and recorded `proofs/results.yaml` were not rewritten
by this documentation update. The site remains unpublished.

This is an unpublished development review for `stage/0.3.3`. The API rename is
breaking: `rf.Group(bounds=...)`, `extensions.group`, and the `group` output key
replace their Cluster counterparts, with no runtime aliases or checkpoint shim.
The published site has not been updated.

The learning work is **not complete**. The normalized-affinity prototype was
rejected after worsening P078 performance and passing only one of three P065
seeds. The branch retains raw tied scores and the independently tested runtime
corrections. Its final full panel passes P019 and one of three P065 seeds;
P018, P020, and all three P078 seeds remain unmet. It is not ready to merge as
a fix for the Group learning proofs. No gate or training budget was changed.

## Revisions and evidence

- Package base: `2bb35adb87b6fefe615864b170c196e7509f81fe` (RelFlow 0.3.2).
- Docs base: `8b074e691f77a03d456623af7b9aa5bf7adb5057`.
- Baseline full-package Python source SHA-256:
  `1b3a0087fc8872c96222aa9a330a8e5ed0d2f486921c70525ab1e58fa111ea13`.
- Rejected normalized-affinity prototype full-package Python source SHA-256:
  `c2901368ed8d1eb22f3c1742da6a1e5a146e0be6630001817971277ca3ff3049`.
- Rejected normalized prototype with occupied-column revival,
  bfloat16 writer consistency, and frozen-parameter guards:
  `d02a341b14d9910a613391d621cbba5678a032b6c0ff66c290a431a3efdd56b1`.
- Final retained raw-score source, with the runtime fixes:
  `4f900097c5f12eb6eef07e88b7abba5c61089f52696d52ee8c08d105894a5b58`.

The source digest uses the reporter's sorted `src/relflow/` paths and file bytes.
It identifies the uncommitted source independently of package version. All nine
final raw-score records share the final retained source digest.
The proof revision identifies the docs base; script and syntax fingerprints
record the actual local proof used. Subsequent narrative edits do not rewrite
those recorded fingerprints.

[Canonical results](../results.yaml) retain every earlier run and append five
fresh 0.3.2 baseline runs and eighteen rejected normalized-affinity prototype
runs across two source snapshots, followed by nine final raw-score runs. All are
full CPU runs with one thread, Python 3.12.10,
Torch 2.12.0, and Lightning 2.6.4. Intermediate mechanics-only experiments were
not imported as final proof evidence.

## Original proof outcomes: final raw scores

| Proof | Fresh 0.3.2 baseline | Final retained raw-score candidate |
| --- | --- | --- |
| P018, five regression regimes | Not met: terminal counts all 7; usage perplexity 6.92–7.35 | Not met: terminal counts all 3; perplexity 1.84–1.85 |
| P019, plain-input control | Met: commitment constant; adherence zero | Met: same negative control |
| P020, repeated categorical labels | Met: terminal counts all 5; perplexity 5.08–5.11 | Not met: terminal counts all 3; perplexity 3.06–3.08 |
| P065, reconstruction and coverage | Not met at seed 7651: known accuracy 88.9% / 90.6% / 84.3% | Seed 7652 met: 100% / 100% / 100%; seeds 7651 and 7653 not met |
| P078, held-out regime recovery | Not met at seed 7801: ARI 0.1854; nRMSE 0.7012 | All three seeds not met; ARI 0.1653–0.1918; nRMSE 0.5963–0.6278 |

P065's three accuracy entries correspond to 1,024, 512, and 102 known selected
targets. Candidate seed 7651 scores 75.9% / 77.7% / 67.6%; seed 7653 scores
63.7% / 63.7% / 59.8%. Coverage counts, unavailable-target accounting,
rebatching, all-OOV behavior, and frozen evaluation state pass. One successful
reconstruction seed is not a reliable fix.

P078 retains the ARI ≥ 0.80, nRMSE ≤ 0.35, no-regime, and shuffled-ID gates.
No-regime controls pass in the candidate panel; meaningful stable partition
recovery and the required prediction accuracy do not. Shuffling IDs does increase
error by the required amount in all three final raw-score runs. P018 and P020
measure capacity state and reuse validation observations, so even their older
passes did not establish correct partitions or held-out predictive quality.

## What the model corrections establish

Visible identity assignments can learn from downstream targets without a Group
reconstruction mask. The mask additionally trains the decoder and supplies
evidence for its adaptive balancing state. `committed` identifies that balancing
subset; unretired candidate columns may still route identities.

The retained implementation uses raw learned identity rows for tied
reconstruction and writing scores. It uses finite balancing in log space,
retires merged columns across
embedding and prediction paths, and clears obsolete optimizer history for
replaced slices. Its batch-end callback weights observations globally across
DDP ranks, including ranks with no Group objective. Mutual information separates
variation between predictions from uncertainty in one repeated prediction.
Runtime regressions cover these properties; they do not establish learning
success.

Row normalization was a modeling proposal, not an established API invariance.
Its unit-level translation invariance did not justify shipping the observed
learning regression: post-fix normalized P078 ARI was -0.0058–0.0042 and nRMSE
0.9643–0.9749, worse than raw scores. A one-group model can retain a learned
90/10 prior between two labels even though both route to the same group; row
normalization erases that distinction and forces 50/50. The retained-score
regression checks writer probability and reconstruction NLL for this case.
Both normalized snapshots remain recorded as rejected
development evidence; their scores do not describe the retained raw-score model.

A masked opaque ID cannot be identified from an independent `x` while `y` is
hidden. In P020, the ID can be the only visible field before masking. This makes
the identity reconstruction objective uninformative on those masked rows even
though visible IDs can learn useful behavior. Correcting an uncertainty-based
capacity estimate does not resolve that objective tension or justify lowering
the existing proof gates.

## Validation-only diagnostics

These exploratory single-seed interventions use validation data, not the
canonical held-out proof result. They change routing, initialization, fixed
capacity, or the reconstruction objective; they are not one jointly validated
configuration. P078 diagnostics use 600 updates. P065 diagnostics use 800
updates and 512 / 256 / 51 known validation targets, rather than the canonical
1,024 / 512 / 102. Raw JSON files preserve all recorded values and intervention
descriptions.

| Diagnostic | P078 validation ARI / nRMSE | P065 known accuracy |
| --- | --- | --- |
| Fixed four groups, visible IDs, deterministic straight-through routing, initialization ×0.01 [A] | 1.0000 / 0.0499 | — |
| Adaptive bounds, visible IDs, same routing and scale [B] | 0.9173 / 0.1138 | — |
| Adaptive reconstruction, balanced prior 0.1, straight-through routing, raw affinity [D] | 0.8916 / 0.0365 | 71.5% / 73.0% / 76.5% |
| Normalized affinity, original Gumbel routing and initialization [C] | — | 100% / 100% / 100% |
| Adaptive reconstruction, normalized affinity, straight-through routing, initialization ×0.01 [C] | 0.7155 / 0.2646 | — |
| Normalized affinity plus balanced prior 0.1 and straight-through routing [E] | 0.4740 / 0.4435 | 73.8% / 74.2% / 78.4% |

The strong fixed-capacity result shows that downstream supervision can train
assignments. The raw-affinity balanced-prior variant improves one grouping
diagnostic while failing reconstruction. Combining it with normalized affinity
does not preserve both benefits. These probes motivate further work; they do
not clear the original proofs or support promoting a new default.

- [A: routing ablations](diagnostics/routing-ablations.json)
- [B: routing follow-up](diagnostics/followup-validation.json)
- [C: normalized affinity](diagnostics/normalized-affinity-validation.json)
- [D: balanced initialization](diagnostics/balanced-initialization-validation.json)
- [E: combined interventions](diagnostics/combination-validation.json)

Each JSON's `group_source_sha256` identifies its snapshot module before the
described in-memory interventions. These are weaker provenance records than the
canonical reporter runs and do not identify the final production source. The
temporary intervention scripts are not a supported public API.

## Teacher and compression hypotheses: not promoted

A further proposal makes the Group head predict the current identity's detached
group membership, while downstream objectives learn membership through visible
IDs. This changes the optimized objective and the interpretation of vocabulary
probabilities. It is a research direction, not the implementation in the package.

The initial [teacher panel](diagnostics/teacher-validation.json) looked promising
at one seed. Updating adaptive usage from actual teacher memberships then
[collapsed one variant](diagnostics/actual-teacher-usage-validation.json).
The final [protected-revival hypothesis panel](diagnostics/final-hypothesis-validation.json)
retained the occupied-group revival correction but still failed broader checks:

| Validation-only hypothesis | Recorded result | Why it is not a demonstrated fix |
| --- | --- | --- |
| Detached teacher, balanced prior 0.1, decoder-derived usage | P078 ARI 0.8669 / 0.7427 / 0.8007 across seeds 7801–7803; nRMSE 0.0451 / 0.1283 / 0.0273 | Seed 7802 misses the original ARI ≥ 0.80 requirement; this is not the complete held-out/control proof panel. |
| Actual-membership usage with entropy compression, coefficient 0.01 | P078 seed 7801: ARI 0.9182, nRMSE 0.1800, four occupied groups; P065 seed 7651: 100% known accuracy at all three diagnostic coverage levels | Only one validation run per task; the same design fails the separate count-proof tasks. |
| Same compression design, P018 full 30 epochs | Final committed counts all 3; usage perplexity 1.019–1.046 | Original count and perplexity gates fail; usage collapses toward one group. |
| Same compression design, P020 full 30 epochs | Final committed counts all 3; usage perplexity 1.187–1.346 | Original count and perplexity gates fail again. |

The compression objective scans every populated identity row rather than just
the current batch, adding work proportional to vocabulary size times group width.
The favorable P078 diagnostic has only 64 identities; no large-vocabulary
throughput or memory claim follows. The separate count-proof tasks show
collapsed usage, but they are different tasks, not continuations of that
regime-recovery run. This is a failure to generalize across tasks, not evidence
that additional training caused the successful run to collapse. Neither teacher
training nor compression was promoted into production, and neither was appended
as a canonical production
proof run. A future teacher integration needs its own explicit proposal,
consistent output semantics, and unchanged-gate multi-seed evidence.

## Reproduce the canonical checks

Use an environment containing the candidate RelFlow source; static docs tooling
does not install or import RelFlow. From this docs checkout, run the original
CLIs with no `--steps` override and a separate output file:

```bash
PYTHONPATH=proofs /path/to/relflow-env/bin/python proofs/vocabulary/group_pretraining_coverage.py --seed 7651 --output /tmp/group-review.yaml
PYTHONPATH=proofs /path/to/relflow-env/bin/python proofs/group/regime_recovery.py --seed 7801 --output /tmp/group-review.yaml
```

Repeat P065 with seeds 7652 and 7653, and P078 with 7802 and 7803. P018, P019,
and P020 keep seed 0 and their original epoch budgets. Read the recorded
`checks` and `outcome`: an unmet behavioral gate is recorded as `not_met` while
the CLI still exits successfully. Passing runtime tests is a separate result.

## Possible solutions from primary research, 2026-10-17

Research proposals only. This pass inspected the retained Group implementation,
proof generators, and earlier intervention scripts, and read primary papers and
author implementations. It did not change the model, rerun training, append
canonical measurements, or establish a fix. Nothing in this section authorizes
publication of the review branch.

### What the current evidence identifies

The adaptation statistic measures variation between the masked-ID head's
predictions. It does not measure stable identity membership. In P020, the ID
and the only other field, the target label, are both hidden on identity
reconstruction rows. In P078, the remaining visible x is independent of the ID.
In expectation, a correctly calibrated head can therefore predict the same
distribution for every such row, yielding zero mutual information and pushing
the estimated count toward its lower bound. Visible identities can still learn
from the supervised task. This mismatch survives any choice of routing gradient.

Even using actual hard assignments would not make effective count equal to
cardinality. Two occupied groups at frequencies 90/10 have exp(H) about 1.38.
An entropy estimate cannot generally discover the number of unequal groups.
The current committed mask also only controls the balancing subset: uncommitted,
nonretired columns remain available in the forward pass. A replacement capacity
mechanism must define and control actual assignment support.

### First: isolate the small changes

Compare deterministic hard straight-through routing with the current per-value
Gumbel sampling, changing nothing else. Both existing paths produce one-hot
forward values, but only the current training path resamples an identity's group
on each occurrence. An optional intermediate control samples once per distinct
identity per batch. Track assignment churn and training/evaluation disagreement.
The [Gumbel-Softmax paper](https://arxiv.org/abs/1611.01144) distinguishes hard
samples from the continuous gradient estimator. For a fixed noise draw,
argmax((logits + noise) / temperature) is unchanged by positive temperature;
temperature annealing alone does not remove hard-routing randomness.

Separately compare decoder-derived evidence with actual known-ID membership
evidence, leaving routing, reconstruction loss, balancing, initialization, and
update budget fixed. Record both observation-weighted and identity-weighted
usage; frequent IDs and numerous IDs are different notions of group size.
Any distributed implementation must globally reduce sufficient statistics and
handle ranks without eligible observations. This addresses the evidence source,
not the separate question of selecting cardinality.

Then isolate whether identity-reconstruction gradients into the shared
assignment table help or interfere with downstream supervision. Keep the same
mask schedule and decoder present so this comparison does not change input
availability or shared initialization. Preserve learned identity frequency
priors and the one-group 90/10 regression. Neither row normalization nor uniform
probability within a group preserves those priors.

### Main modeling candidate: group by predictive behavior

[Grouped-effects estimation](https://www.cemfi.es/ftp/wp/1208.pdf), section 3,
alternates estimating model parameters and assigning each entity to the group
that best explains its repeated outcomes. Its section 5.2 treats unknown group
count as a separate model-selection problem. This motivates a supervised Group
objective in which one membership is shared across an identity's observations:

    z_i = argmin_g sum_r loss(y_ir, model(x_ir, group_embedding_g))

The proposal is to accumulate behavioral evidence from training observations,
then update the saved identity-to-group mapping. Prediction continues to use
that mapping without seeing the current target. This is entity grouping, rather
than choosing a different expert for each row. The paper's statistical results
do not establish recovery guarantees for a flexible neural RelFlow model.

A related principle in the [agglomerative information bottleneck](https://proceedings.neurips.cc/paper_files/paper/1999/file/be3e9d3f7d70537357c67bb3f4086846-Paper.pdf)
is to merge items by similarity of predictive distributions. For RelFlow, compare
response behavior over common calibration contexts, rather than using only
prototype or identity-decoder cosine similarity. Preserve conditioning on x:
clustering outcome means alone can miss regimes with equal means and crossing
response curves. A proof-local grouped linear-regression reference is a useful
first check before integrating an expensive counterfactual neural algorithm.
Production integration must use generic extension loss contracts; shared code
must not special-case Number or Category.

### Capacity: explicit predictive cost rather than entropy rounding

A candidate objective is supervised prediction loss plus a cost per occupied
group, constrained by bounds. The [DP-means paper](https://icml.cc/2012/papers/291.pdf)
derives squared distortion plus a cluster-count penalty. That provides a model
for an explicit tradeoff, not a theorem for neural supervised losses. Its penalty
scale must be selected and reported; a default does not remove that choice.

A simpler initial diagnostic is a bounded group-count search or merge path,
selected using validation prediction loss and a declared preference for fewer
groups. Count every candidate evaluation in the compute budget. Merge groups
only when their predictive differences are negligible on calibration data;
consider splits where pooled predictive residuals consistently disagree.
Preserve minimum capacity, frozen-parameter rules, retirement, optimizer-state
reset, and checkpoint invariants. Predictive quality and recovery of the
generating partition remain distinct outcomes.

### Alternative: a task-trained vector quantizer

[VQ-VAE](https://arxiv.org/html/1711.00937v2) uses hard nearest-prototype forwards
with distinct encoder, dictionary, and commitment updates. Its appendix also
describes EMA dictionary updates, implemented in [DeepMind Sonnet](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py).
A supervised adaptation could learn an ID representation through downstream
losses and quantize it to a shared group. This is a larger alternative to
categorical assignment logits, and still needs a capacity-selection mechanism.

Keep its internal dimension small or explicitly account for added storage:
replacing V-by-K assignment rows with V-by-d_model embeddings can substantially
increase memory. If basic VQ is promising but unstable, [Huh et al.](https://arxiv.org/html/2305.08842v1)
identify representation/codebook mismatch and sparse code updates as optimization
problems. Test EMA or alternating updates one at a time before adding further
gradient estimators. Low quantization error or high code utilization alone does
not demonstrate useful grouping.

### What the earlier hypotheses did not isolate

- The actual-teacher-usage probe also disabled balancing. It does not isolate
  changing the evidence source.
- Minimizing positive assignment entropy encourages compression toward one
  group. It is not a group-count estimator; at an exactly uniform histogram its
  assignment-logit gradient can vanish. Later cross-task failures do not prove
  that membership statistics alone are ineffective.
- The detached teacher's P065 control has eight labels and eight slots with
  index-modulo-eight initial assignments. With input IDs unavailable, its
  successful cue classification does not establish learned behavioral grouping.
  Its uniform-within-group writer also changes identity-prior semantics.
- Plain-input and reconstructed constructors consume random numbers differently
  because they instantiate different decoders. A common seed alone does not
  guarantee identical shared initial weights.

### Evaluation order

Start with isolated routing, evidence-source, and reconstruction-gradient
comparisons using copied shared initial weights, identical batches, and the
original per-task budgets. Keep a fixed-known-count control explicitly labeled
as an oracle diagnostic. Only carry a coherent adaptive candidate forward to
the unchanged P065/P078 three-seed panel and P018/P019/P020 checks. Preserve
their gates and canonical histories; add unequal-frequency, crossing-response,
redundant-group, and no-regime diagnostics alongside them. A minimum-one-group
control requires separate bounds that permit one group.

P019 currently asserts dormant adaptation without reconstruction. If a later
design deliberately enables supervised-only adaptation, document that contract
change and introduce its own proof, retaining the original P019 record. Do not
silently edit the dormant-control gate to make a new design pass.

Use validation for development choices; canonical test scores are final checks.
Report actual occupied groups separately from committed count and effective
usage, and report ARI, predictive error, reconstruction accuracy/calibration,
assignment stability, memory, and elapsed work. Additional reassignment or
candidate-model evaluations are additional compute even when the number of
AdamW steps stays fixed. None of these proposals requires a new dependency or
public configuration option merely to run the first diagnostic.

## Iteration results, 2026-10-17

These are local CPU experiments against frozen copies of the retained 0.3.3
working tree, not changes to the package defaults. All retain the original
per-task optimizer budgets: 600 updates for P078 and 800 for P065. Development
uses validation; final held-out checks below use a frozen method and checkpoints.
No documentation was published and no canonical measurements were overwritten.

The frozen package source hash is
`4f900097c5f12eb6eef07e88b7abba5c61089f52696d52ee8c08d105894a5b58`;
the Group module hash is
`80429258fbc252530c049d1b02f31c5270b04dbcb723f60b790a8c57e7a5ae7c`.
The mixture harness uses a differently framed aggregate source digest; the
individual Python files were independently compared and are identical.

### Decisions from the experiments

| Direction | Observed result | Decision |
| --- | --- | --- |
| Membership evidence alone | Changes the committed count, but P078 validation ARI 0.1245 and nRMSE 0.7397 at seed 7801 | Correcting the statistic does not solve membership learning. |
| Deterministic routing and smaller initialization | One of three seeds clears both P078 validation gates; P065 remains unreliable | Do not promote a seed-sensitive default. |
| Routing + membership evidence + detached reconstruction, smaller initialization | P078 nRMSE 0.0599 / 0.2588 / 0.1694, but ARI 0.7512 / 0.5566 / 0.6534; P065 full-known accuracy 0.1387 / 0.1270 / 0.1113 | Prediction and partition recovery differ; removing reconstruction's only assignment-learning path breaks Group-as-target learning. |
| Column-normalized label mixture, balanced initialization and deterministic routing | Stable P078 validation ARI 0.8308 / 0.8507 / 0.8481 and nRMSE 0.1345 / 0.1169 / 0.1202; P065 seed 7651 remains about 71.5% accurate | Useful grouping signal, but not a complete proof panel or general reconstruction fix. |
| Detached teacher or hard membership likelihood with learned priors | P065 seed 7651 reaches 100%, but P078 validation ARI falls to 0.3525 or 0.4229 | Preserving prior semantics does not remove the cross-objective tradeoff. |
| Fixed-simplex gradient/commitment alternatives | Seed 7801 P078 ARI 0.1117–0.1644; nRMSE 0.5860–0.6077 | Screened out at the original budget. |
| Learned EMA codebook | P078 seed 7801 ARI 0.2843, nRMSE 0.4639; eight occupied groups | A genuine VQ fork improves its control somewhat but misses both gates; no multi-seed promotion. |
| Sibling-conditioned Group decoder | P065 full-coverage accuracy 0.8828 / 0.6973 / 0.8887; P078 seed 7801 ARI 0.1934, nRMSE 0.5852 | No proof passes; the extra projection is not justified by this panel. |
| Predictive reassignment between ordinary training updates | All three frozen P078 validation and final held-out runs recover the exact partition and clear prediction gates | Strongest experimental integration; still proof-local. |

Results in different rows have different explicit routing/RNG protocols.
Use the matching controls inside each artifact, rather than attributing every
cross-row difference to one mechanism. The ablation panel checks matching shared
initial weights, input batches, masks, and common RNG streams. Failed variants
and all seeds are retained. The mixture screens did not run the complete
P078 control panel, so their stable-validation successes are not canonical
proof passes.

### The successful predictive reassignment experiment

The same `rf.Group(bounds=(2, 8))` model trains for 600 AdamW updates. At updates
200, 400, and 600, a proof-local routine evaluates each available group on the
training observations and assigns each identity to the group with the smallest
pooled target-prediction error. It keeps the replacement only if validation error
improves. The update at 600 occurs after the last optimizer update. The true
partition and true count are not supplied to training or selection.

Each accepted transition preserves identity-row mass, sets a hard assignment
margin, and clears obsolete gradients and Adam moments on replaced rows. The
method uses the original Group architecture and does not add learned parameters.
Its high-margin assignments also suppress routing noise; that change is part of
the intervention, not an unreported side effect.

| Seed | Validation ARI | Validation nRMSE | Original held-out ARI | Original held-out nRMSE | Occupied groups |
| --- | --- | --- | --- | --- | --- |
| 7801 | 1.0000 | 0.0616 | 1.0000 | 0.0630 | 4 |
| 7802 | 1.0000 | 0.0425 | 1.0000 | 0.0408 | 4 |
| 7803 | 1.0000 | 0.1610 | 1.0000 | 0.1604 | 4 |

All six fresh validation models clear the 30 recorded original numerical and
control checks. Separately trained no-regime models retain validation nRMSE
about 1.00, near-zero ARI, and held-out nRMSE 1.0065 / 1.0054 / 1.0001.
Shuffling stable validation IDs raises nRMSE to approximately 1.41–1.43.
Parameter, buffer, and assignment snapshots remain unchanged during evaluation.

A matched control hardens each identity's existing assignment at the same
updates, preserving row mass and resetting the same optimizer state without
searching prediction residuals. Its three validation ARIs are
0.0262 / 0.0466 / -0.0009 and nRMSEs 0.8671 / 0.7843 / 0.9064. Hardening and
history reset alone do not explain the predictive reassignment result.

The extra work is material: three eight-group sweeps evaluate 98,304 additional
training rows, plus 6,144 validation rows for proposal acceptance. Measured
intervention time on this local machine is approximately 3.1–3.4 seconds for
the stable runs, in addition to ordinary training. Equal optimizer updates
do not mean equal computation. These timings are not a hardware throughput
benchmark.

### Provenance and limits

The immutable runner is `predictive_em_v1.py`, SHA-256
`f7a25003f1b9c76a7c273055b3999725140ec4e4c1aedc916585a1a35703463a`.
Its fresh validation panel refuses cached outputs, records dependency and
checkpoint hashes, and gates held-out evaluation on the complete matching
validation panel. Early exploratory scripts and results are preserved separately.
An early pilot visited seed 7801's original test split; subsequent selection used
validation only. Seeds 7802 and 7803 were first scored on that test split after
the method was frozen. Treat the seed-7801 original test score with that
qualification, rather than claiming the whole development history was test-blind.

A separate independent audit uses new observations generated with
`seed * 100 + 10003` after freezing the same checkpoints. It recovers ARI 1.0000
at all three seeds, nRMSE 0.0623 / 0.0421 / 0.1621, and clean negative controls.
These are additional held-out observations, not replacements for the original
proof measurements.

The intervention currently knows the proof's scalar target and computes its
per-identity squared error. This is an actual RelFlow model experiment, but the
selection routine is not a generic SDK implementation. A production integration
needs an extension-owned loss/scoring contract and bounded calibration data,
plus distributed, frozen-parameter, vocabulary-growth, and checkpoint behavior.
Shared architecture must not acquire a Number-specific branch.

The old committed-balancing metric remains separate from occupied membership;
this experiment does not repair its meaning or establish adaptive capacity for
every distribution. P018/P020's count requirements and P019's dormant-input
contract remain unchanged. P065 reconstruction is not solved by the reassignment
experiment, and new-identity generalization has not been demonstrated. Broader
crossing-response stress runs also remain less consistent than the core process.

The proper-mixture and hard-membership-prior scorers each passed the existing
one-group 90/10-prior and bf16 writer/loss agreement regressions under their
experimental overrides. Those four runtime checks are separate from learning
success.

Full source snapshots, runners, failed outcomes, per-run hashes, and logs are
preserved in [the experiment archive](diagnostics/iteration-2026-10-17/experiments.tar.gz),
with [file hashes](diagnostics/iteration-2026-10-17/manifest.json).
The primary frozen panels are available directly as
[validation](diagnostics/iteration-2026-10-17/predictive-validation.json) and
[held-out results](diagnostics/iteration-2026-10-17/predictive-heldout.json), with
[the independent audit](diagnostics/iteration-2026-10-17/fresh-holdout-audit.json).

To reproduce, unpack the archive into an empty directory and use the existing
RelFlow environment (or an environment matching the recorded dependencies):

```bash
tar -xzf experiments.tar.gz -C /path/to/unpacked
/path/to/relflow-env/bin/python /path/to/unpacked/group-iteration-predictive-20261017/predictive_em_v1.py validation --output /tmp/group-validation.json
/path/to/relflow-env/bin/python /path/to/unpacked/group-iteration-predictive-20261017/predictive_em_v1.py heldout --validation-input /tmp/group-validation.json --output /tmp/group-heldout.json
```

Both output paths must be unused. The validation command trains fresh models;
the held-out command verifies and loads their matching checkpoints.

## Expanded proofs and measured cost, 2026-10-17

Three new standalone proofs retain the experimental callback separately from
ordinary `Model.fit`. Their source was frozen before the recorded runs. All
nine reporter records preserve the raw measurements, gates, and fingerprints;
the first 78 proof entries were preserved without rewriting their bytes.

### P079: different numeric group processes

The panel adds crossing curves with equal response means, unequal group sizes,
and three or six true groups. Every fit retains broad `(2, 8)` bounds and 600
optimizer updates. Training residuals propose reassignment; validation error
accepts or rejects it. Test outcomes never select a proposal.

| Case | Prediction and ARI gates | Exact group count |
| --- | --- | --- |
| Four separated curves | 3/3 | 3/3 |
| Four crossing curves | 3/3 | 2/3 |
| Unequal four groups | 3/3 | 2/3 |
| Three separated groups | 3/3 | 3/3 |
| Six separated groups | 2/3 | 0/3 |

Prediction and partition gates pass 14/15 positive fits; exact group count
passes 10/15. All three six-group fits occupy five groups. Crossing seed 7901
splits a group; unequal seed 7901 merges a small group. All no-regime,
shuffled-ID, frozen-state, group-permutation, and unknown-ID checks pass. Each
complete run remains failed because at least one fixed count gate fails.
The exact-count gate is new to P079 and does not rewrite P078's requirements.

### P080: conditional categorical outcomes

Fifty opaque merchant identities belong to five hidden groups. A visible
channel changes the target according to `(group + channel) modulo 5`, with
2% label corruption. Every identity sees every channel equally, so marginal
label frequencies do not identify its hidden rule. The scorer uses true-label
probabilities from the public Category prediction output.

| Seed | Native accuracy | Hardening alone | Predictive accuracy | Predictive ARI | Groups |
| --- | --- | --- | --- | --- | --- |
| 8001 | 20.40% | 39.56% | 97.80% | 1.0000 | 5 |
| 8002 | 25.04% | 36.56% | 78.00% | 0.6986 | 6 |
| 8003 | 23.12% | 32.52% | 93.24% | 1.0000 | 5 |

Two seeds clear all gates: accuracy at least 90%, ARI at least 0.80, and exactly
five occupied groups. Seed 8002 merges true groups 0 and 4 and fragments others.
It remains a meaningful failure. No-regime and shuffled-ID controls stay near
the 20% chance level. All arms receive 600 updates. Vocabulary, assignment,
and content-counter snapshots remain frozen during evaluation; this proof
does not assert a full parameter-and-buffer snapshot like P079.

### P081: fresh matched CPU comparison

Six isolated timed fits follow two excluded warmups. Alternating native and
candidate order across three seeds, both arms match initial weights, data
order, masks, Torch state through the first intervention, 600 updates, and
18 regular validation events comprising 18,432 rows. Timers exclude result
and checkpoint I/O, and include the terminal update-600 intervention.

| Seed | Native | Predictive | Additional wall time |
| --- | --- | --- | --- |
| 7801 | 10.043 s | 12.926 s | 28.70% |
| 7802 | 10.048 s | 12.909 s | 28.47% |
| 7803 | 10.118 s | 13.015 s | 28.64% |

The ratio of sums is `38.850 / 30.209 - 1 = 28.60%`: means of 10.070 seconds
and 12.950 seconds. The three interventions add 98,304 counterfactual training
forward rows and 6,144 validation-selection forward rows to 76,800 training
rows, without adding backward passes or optimizer updates. All matching,
budget, finite-value, and frozen-evaluation checks pass. Validation ARI is 1.0
for all candidate seeds, reproducing the frozen original assignments exactly.

This is additional elapsed time on the recorded Mac CPU with one Torch thread.
It is neither a FLOP estimate nor a GPU benchmark. It measures the original
predictive method; later combination experiments have a separate runtime
profile. The earlier event-timing estimate is superseded by this matched
baseline comparison, while its original artifacts remain preserved.

### Scope and verification

All split comparisons concern new observations of known identities, not
unseen-identity discovery. Reassignment is proof-local, uses metric-specific
scorers, and adds inference passes. It is not a new default or public knob.
Generic extension-owned losses, nested attribution, distributed coordination,
and identity reconstruction remain separate work.

Eleven focused runtime tests protect the conditional-error boundary,
validation-only acceptance, rejected-state rollback, selective optimizer-state
reset, equal-mean opposite-slope separation, and categorical true-label NLL.
The 63 static documentation tests pass. All 122 pages render locally with
the pinned Quarto and Typst diagrams, and rendered links and assets validate.
No live documentation has been published.

## Combining predictive reassignment with other changes, 2026-10-17

The original successful callback was not assumed to remain successful when
combined with other proposals. A frozen four-recipe matrix trained 48 numeric
fits and 12 categorical fits, each for 600 updates. Every fit was then audited
on fresh held-out observations of the same known identities. The original
proof test streams were never loaded by these diagnostic harnesses.

Each numeric recipe receives three seeds across four cases: crossing curves,
unequal groups, six groups, and a separated-four anchor. Each categorical
recipe receives all three P080 seeds. All failures remain in the results.

| Training recipe | Numeric cases clearing all five gates | Categorical seeds clearing all gates |
| --- | --- | --- |
| Predictive reassignment alone | 7/12 | 2/3 |
| Plus deterministic hard straight-through routing | 8/12 | 0/3 |
| Plus assignment initialization scaled to 0.1 | 9/12 | 0/3 |
| Plus less rigid accepted proposals, logit margin 1 | 10/12 | 0/3 |

These numeric gates include exact occupied count, ARI at least 0.80, nRMSE
at most 0.35, and both shuffled-ID controls. They are stricter than P078's
original quality gates. The new recipes were not separately trained on a
no-regime process in this bounded diagnostic, so these totals are not new
complete canonical proof passes.

### Where combinations help

The six-group case is the clearest numeric gain. Reassignment alone occupies
five groups at every seed. Deterministic routing plus the smaller initialization
recovers all six exactly with ARI 1.0 and fresh nRMSE
0.0415 / 0.0518 / 0.0375. Adding margin 1 retains exact recovery and gives
0.0391 / 0.0327 / 0.0373. The last recipe also passes every crossing-curve and
separated-four seed.

Unequal groups remain difficult. The last recipe passes only one of three
seeds; one failed seed has ARI 0.535 despite acceptable global nRMSE 0.223.
That seed fragments into five groups, while another merges to three. An
aggregate prediction objective can tolerate these errors, so improving mean
error does not establish satisfactory minority-group recovery.

### Where combinations hurt

Fresh categorical accuracy falls despite the numeric gains:

| Recipe | Seed 8001 | Seed 8002 | Seed 8003 |
| --- | --- | --- | --- |
| Predictive alone | 98.52% | 78.32% | 93.28% |
| Plus deterministic routing | 78.48% | 59.64% | 59.28% |
| Plus smaller initialization | 58.68% | 66.76% | 59.20% |
| Plus margin 1 | 58.72% | 78.60% | 62.36% |

Every added categorical recipe fails the combined partition, accuracy, and
count requirements. Shuffled-ID accuracy stays approximately 20%, and the
checked vocabulary and assignment state remains frozen. Smaller initialization
and softer proposals therefore do not constitute a safe general improvement.

### Interpretation and protocol limits

The four arms are complete training recipes, not perfectly isolated causal
ablations. Removing Gumbel sampling also removes Torch random draws, which
changes later dropout and masking streams. The deterministic arms share
their draw protocol with one another; the native baseline exactly reproduces
the canonical assignments and acceptance flags. Initial non-assignment
parameters and seeds are matched, but differences cannot all be attributed
solely to the routing estimator.

Margin 1 changes only the applied proposal, retaining its previous row
log-mass through normalized logits; counterfactual searches still use hard
forcing. Accepted updates reset stale optimizer moments as before. The
measured categorical row-mass drift is at most 2.38e-7. Original high-margin
baseline behavior remains unchanged.

Training and validation choose proposals, and all validation results are
frozen before each separate all-arm fresh audit. No arm is selected or
discarded using the audit. Checkpoint, source, input, and protocol hashes
identify each fit. Numeric evaluation checks full parameter, buffer, and
assignment snapshots; categorical checks cover vocabulary, assignments, and
counters. Checkpoints are excluded from publication archives, while their
hashes are retained.

These results support task-specific further experiments, not stacking every
change into a new Group default. The 28.60% timing comparison applies only
to the original callback; concurrent combination timings are not a matched
performance benchmark. Production source remains untouched by this matrix.

### A fifth recipe: normalized identity likelihood with predictive reassignment

A separately frozen add-on combines the previously explored column-normalized
identity likelihood, deterministic routing, weak balanced index-based
initialization, and the original sharp predictive reassignment. The likelihood
and initialization functions were copied unchanged from the earlier balanced
mixture experiment. Sinkhorn balancing remains enabled.

This is a bundled recipe, not an isolated normalization ablation: its balanced
initialization differs from the scaled-normal initialization above. Nine fits
were declared before execution: the three numeric seeds on crossing curves
and six groups, plus the three categorical seeds. All nine development fits
completed before a separate fresh audit evaluated every frozen checkpoint.
The original test streams were never loaded. No method was revised after
measurement.

| Fresh case | Prediction and ARI gates | Exact group count / full gates |
| --- | --- | --- |
| Four crossing curves | 3/3 | 3/3 |
| Six separated groups | 3/3 | 1/3 |
| Conditional categorical outcomes | 3/3 | 1/3 |

Fresh crossing nRMSE is 0.06169 / 0.06356 / 0.05429, with ARI 1.0 and four
groups in every seed. Six-group nRMSE is 0.17484 / 0.17801 / 0.03221; the first
two seeds recover five groups with ARI 0.83162, while the last recovers all six.
The earlier scaled-normal recipe remains stronger on this six-group case.

The categorical predictive gain is substantial:

| Seed | Fresh accuracy | ARI | Occupied groups |
| --- | --- | --- | --- |
| 8001 | 98.56% | 0.8543 | 7 |
| 8002 | 98.12% | 1.0000 | 5 |
| 8003 | 98.00% | 0.9289 | 7 |

All three categorical fits clear accuracy and ARI gates, compared with two of
three for predictive reassignment alone. Two still fragment the five true
groups into seven occupied codes, so they remain failed on exact count.
This distinction matters: improved predictions are evidence of useful learned
rules, while extra groups still violate the declared discovery objective.

All finite-value, vocabulary-exposure, shuffled-ID, and checked frozen-state
controls pass. No no-regime models were newly trained for this add-on. It is a
small same-identity observation panel, not a universal recovery result or a new
default.

The existing one-group 90/10 learned-prior and bf16 writer/loss regressions
also pass with the experimental scorer patched in. A separate direct-path
test checks its analytical column likelihood, NLL, argmax, gradients, and
float32 output for float32 and bf16 inputs. A one-group 90/10 prior remains
90/10 in float32 and approximately 89.9475/10.0525 in bf16. This avoids the
incorrect row-normalized likelihood that would erase identity priors.

The next research question is consolidation and separation of groups with
similar predictive behavior, especially small groups. These results support
combining mechanisms selectively; they do not support a single stacked recipe
across the full panel. No production implementation was changed or promoted.

### Retained evidence for this expansion

The standalone sources and measured reporter records are registered as P079,
P080, and P081 in `proofs/results.yaml`. Their historical predecessors and
failed measurements remain unchanged. All additional training recipes and
failed seeds remain in the diagnostic archives:

- [Expanded proof sources, raw runs, and matched timing](diagnostics/expansion-2026-10-17/README.md)
- [All 48 numeric combination fits and fresh audits](diagnostics/combinations-numeric-2026-10-17/README.md)
- [All 12 categorical combination fits and fresh audits](diagnostics/combinations-categorical-2026-10-17/README.md)
- [All nine normalized-likelihood combination fits and fresh audits](diagnostics/combinations-mixture-2026-10-17/README.md)

Each directory includes a compressed source/data archive and a verified
per-file hash manifest. These local review artifacts do not publish the site.
