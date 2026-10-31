# RelFlow evidence

Controlled synthetic experiments, archived measurements, and pre-merge
behavioral regression checks for [RelFlow](https://github.com/relflow/relflow).
The [protocol](SPEC.md), [catalog](catalog.yaml), and [extraction manifest](extraction.json)
identify the experiments and their original provenance. Existing `P001`-style
identifiers remain stable. The historical archive is retained unchanged in
`archive/results.yaml`; it is not a current-main baseline.

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Running experiments is separate from static tooling. Supply the intended
RelFlow wheel and runtime environment. To run one experiment locally:

```sh
uv run --with /path/to/relflow.whl python experiments/run.py P001 --steps 2
uv run --with /path/to/relflow.whl pytest tests/runtime
```

`--steps 2` is a smoke run and does not establish learning. Local executions
append to `runs/local/results.yaml`, preserving the historical archive.

## Modal and the required merge check

`modal_run.py` runs every registered experiment against both the baseline wheel
and the candidate wheel, on CPU, in the same locked environment. The wheels
must be built from the exact GitHub event revisions. Every baseline-passing
check must also pass on the candidate. Baseline capability misses remain visible;
errors on either side, missing results, smoke runs, and partial suites fail the
required gate. Each experiment keeps its original default seed and budget;
experiments that implement an internal seed panel retain it.

The library integration is provided by `integration/evidence.yml`. It pins this
repository to a full commit SHA, builds wheels without Modal credentials, and
launches the isolated workers from a trusted job. Add dedicated GitHub Actions
secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to the calling repository.
Fork PRs fail closed until an approved trusted execution path is configured.
The workflow includes `merge_group` and an always-evaluated final gate.

## Reports and docs

Every run produces `manifest.json`, per-task JSON records and logs,
`summary.json`, `report.html`, and `checksums.json`. GitHub Actions retains the
bundle for PR review. The publication workflow promotes selected complete runs
to immutable Release assets and creates a static docs export. This requires
only GitHub and Modal; no AWS account is needed.

```sh
uv run python export_docs.py /tmp/evidence-export --bundle /tmp/evidence-docs.tar.gz
```

That command reads archived results without executing experiments. Pass
`--reports runs/<run-id>` to include a complete new candidate run. Docs imports
pin the bundle digest and keep the snapshot in their own checkout. Exporting
historical measurements does not establish new learning results.

Licensed under Apache 2.0; see LICENSE and NOTICE.
