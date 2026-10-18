# RelFlow documentation guide

This repository owns the static RelFlow documentation, its assets, and its
canonical proof experiments. It must build from this checkout alone.

## Scope and build

- Authored website pages, assets, extensions, and `_quarto.yml` live in `docs/`.
  Canonical proof source, runner, and results live in `proofs/`. Preserve page
  URLs, anchors, branding, layout, and mobile overflow behavior.
- Use `make check`, `make check-docs`, `make render`, and `make dev`.
- The build uses pinned Quarto 1.10.18, bundled Typst, and the committed
  extension. PyYAML is the only Python runtime dependency; pytest and Ruff are
  development tools. Never install or import RelFlow, Torch, Lightning, or a
  sibling checkout to render or validate the site.
- Python examples are static illustrations. Use ordinary `python` fences and
  the Markdown engine. Never execute examples, train models, read application
  data, or download model weights during a build.
- Proof execution is a separate, explicit task in an environment containing
  the target RelFlow version. Runtime proof tests live in `tests/proofs/` and
  are excluded from default static validation.

## Teach in order

Start with one payment, one target, and a small model. Introduce repeated
transaction history next, then data preparation, training, and prediction.
Keep architecture internals, self-supervision, extension protocols, and
operational details on their own later pages. Give each concept one home and
link to it.

Keep a progressive transaction-modeling path: one payment and a dispute target;
then a transaction sequence including that payment as its latest item; then
event attributes and account profile context with a merchant-repeat outcome;
then nested purchases, separate withdrawal history, and targets at transaction
and account levels. Introduce reconstruction and
representation reuse after the supervised workflow. Explain the business
decision and what each addition contributes before its implementation. Adapt
research use cases and terminology into native examples; keep paper-specific
attribution and adaptation boundaries in `research/documentation-sources.json`.

Use coherent financial examples without implying that RelFlow is restricted to
finance. The first flat payment has `amount`, `merchant`, and
`disputed=rf.Boolean(mask=True)`. Sequence examples use one `trxns` branch
containing earlier transactions and the latest payment exactly once. Put
`booked_at`, `amount`, `merchant`, and `disputed` inside it; do not duplicate
current-payment fields at the root. Use
`rf.DateParts(dateparts=["day_of_month", "day_of_week", "hour_of_day"])`
for booking-time calendar features. Calendar cycles do not encode exact
elapsed durations. Oldest-to-newest order plus `overflow="tail"` keeps the
latest transactions, with no events after the scoring cutoff.

Use `disputed=rf.Boolean(mask=True)` inside `trxns`: every retained transaction
is a supervised target and all dispute labels are excluded from input.
Training observations have settled Boolean labels for every transaction.
At a common account-review cutoff, each label describes a new dispute during
the following 30 days; every input record is known by that cutoff. Prediction
requests omit `disputed` from the transaction records. Keep selectors,
`rf.Mask` policies, and dynamic-masking discussion out of introductory pages;
teach them only on their later dedicated pages.

## Public examples

Use `import relflow as rf`, `rf.Model(...)` or a size factory, and
`rf.Branch(...)`. Parent keywords name children; definitions have no `name` or
positional arguments. Use bare field classes for defaults. The model root is
anonymous at `rf.Address()`; diagrams may label it `payment` or `account`.
Use one argument per address component, such as
`rf.Address("trxns", "amount")`.

`mask=True` defines a supervised target kept out of the input. `embed=True`
exports context and creates no objective. `ffnx` defaults independently to 8
on each node; tanh GELU and RMSNorm are fixed.

Child branches default to `length=None, overflow=None`, retaining all selected
slots with batch-dependent widths. A positive integer length sets a hard cap;
only bounded branches accept `head`, `tail`, or `error`, with omitted overflow
resolving to `head`. Prediction format 2 writes every repeated
record collection as a variable Arrow list of structs, including bounded
branches; input caps do not add public padding. Fixed-width vectors retain
their feature axis. Explain memory costs and the warning
for public serving; automatic widths do not make padding free.

Prefer application-supplied Parquet file paths (`.parquet`) for ordinary
training and prediction examples. Pass paths to `rf.ArrowDataModule`, including
its `predict` split; the direct `model.predict(...)` input is an in-memory API.
Reserve table-input examples for explaining that API. Returned predictions
remain Arrow tables.

Use `rf.ArrowDataModule(model=model, train=train, validate=validate)` for
canonical training examples. Name application-supplied inputs. Configure AdamW
through `model.fit(data, batch_size=128, learning_rate=1e-4, weight_decay=0.01, max_epochs=30)`;
omitted options use defaults. Each fit binds a fresh optimizer to the current
parameters, with any scheduler supplied for that run. Full training continuation
uses `ckpt_path`. Reserve `rf.adamw(...)` and explicit `model.optimizer` setup
for custom optimization or native `lit.Trainer` examples. Prediction returns
selected `predictions`, `representations`, and retained `inputs`
columns, omitting unselected planes and rejecting an empty public output plan.
Use ordinary child names in the prediction tree, never literal slash keys.
Repeated records always use variable Arrow lists, including bounded branches;
input caps do not add public padding. Fixed-width vectors retain their feature
axis. Extension output list children must be named `element`.
Embeddings live in representation records with `path`, `parents`, `output_index`,
and `embedding`. The pinned contract targets RelFlow 0.3.1; recorded proofs
retain their original provenance even when executable output readers are
migrated. A reader migration does not establish fresh learning results. Show
observation and output examples as one YAML record, nesting repeated values
under their field names.
Keep examples realistic without constructing or executing miniature datasets.
Set batch size on the loop method in ordinary training examples. A positive
`batch_size` on any loop updates the persistent model setting; Arrow data
modules read it when creating loaders, while custom Lightning loaders own their
configuration. Under DDP it is per device. Direct prediction with explicit
`batch_size=N` preprocesses once, chunks processed Arrow rows before encoding,
and postprocesses the combined output once. Omission or `None` preserves direct
whole-input behavior. Constructor batch size remains the initial/checkpoint
setting and supports native Trainer/Tuner workflows.

Verify API claims against pinned upstream source and `contracts/relflow.json`;
existing prose alone is not evidence. Update the contract deliberately when
targeting another package version.

## Visuals and proofs

Use the shared `node` and `tree` Typst functions in
`docs/assets/typst/model-tree.typ`; read `docs/assets/typst/README.md` before
changing structural diagrams. Diagrams explain field roles and repeated context
rather than hyperparameters. Keep nearby code, YAML, labels, captions, and text
alternatives consistent. Preserve bundled extension and font licenses.

`proofs/render.py` publishes the local canonical sources and recorded results
without executing them. Generated proof pages and fragments under `docs/proofs/`
are ignored; never edit them as a second source. Follow `proofs/SPEC.md` when
changing experiments. Preserve results, source fingerprints, and provenance;
never alter measurements to make a documentation check pass. Clean only generated
publication output, retaining all canonical experiments and results.
