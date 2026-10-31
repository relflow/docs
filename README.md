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

## Lab and the required merge check

The primary backend runs on `lab` over Tailscale SSH. `ssh_run.py` sends the
pinned suite and both wheels to the host; `lab_run.py` runs each experiment in a
separate disposable Docker container. Experiments have no network, no host home
mount, no Docker socket, a read-only root filesystem, and bounded CPU/memory.
The host needs Docker and Python 3.10+ with PyYAML. Runtime packages are installed
in an isolated Python 3.12 image; RelFlow is never installed on the host.

The gate runs every registered experiment against both the baseline wheel and
the candidate wheel in the same locked environment. Every baseline-passing
check must also pass on the candidate. Baseline capability misses remain visible;
errors on either side, missing results, smoke runs, and partial suites fail.
Each experiment retains its declared default seed and budget, including any
internal seed panel. A one-seed experiment remains provisional evidence.

The library integration is provided by `integration/evidence.yml`. It calls a
pinned reusable workflow that checks the caller repository and excludes fork
PR execution. Tailscale workload identity federation ties short-lived access
to that exact workflow. See [the setup guide](integration/lab.md) for the
repository variables and tailnet trust rules. No AWS account, Modal CI token,
or self-hosted GitHub runner is required.

The optional `modal_run.py` backend remains available with `uv sync --extra modal`.
`integration/modal.yml` is its alternative CI template; it is not the primary
workflow. Dedicated Modal service users require a supported Modal plan.

## Reports and docs

Every run produces `manifest.json`, per-task JSON records and logs,
`summary.json`, `report.html`, and `checksums.json`. GitHub Actions retains the
bundle for PR review. The publication workflow promotes selected complete runs
to immutable Release assets and creates a static docs export. It also preserves
both tested wheels, the runtime requirements, and GitHub run/head/base provenance,
with checksums. This requires only GitHub and the existing lab/Tailscale
connection; no AWS account is needed.

Promote a selected library run before its Actions artifacts expire:

```sh
gh workflow run publish.yml --repo relflow/evidence \
  -f tag=<unique-release-tag> -f library_run_id=<library-actions-run-id>
```

```sh
uv run python export_docs.py /tmp/evidence-export --bundle /tmp/evidence-docs.tar.gz
```

That command reads archived results without executing experiments. Pass
`--reports runs/<run-id>` to include a complete new candidate run. Docs imports
pin the bundle digest and keep the snapshot in their own checkout. Exporting
historical measurements does not establish new learning results.

Licensed under Apache 2.0; see LICENSE and NOTICE.
