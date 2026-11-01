# RelFlow evidence

This directory owns the 79 controlled synthetic experiments, their protocol,
lab runner, runtime tests, and current publication results.
The website and evidence share this repository. Stable P identifiers remain
unchanged. results.yaml contains one selected full run per experiment, including
its metrics, checks, seed, environment, and source provenance.

Run from the repository root:

```sh
make check
uv run python evidence/experiments/run.py --list
uv run --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001 --steps 2
uv run --with /path/to/matching-relflow.whl pytest evidence/tests/runtime
```

Supply an actual wheel. A two-step run checks execution only. Each execution
writes a new JSON file under evidence/runs/local; --output-dir selects another
directory. Execution never edits the reviewed publication selection.

## Required lab execution

The library's workflow builds exact baseline and merge-candidate wheels and
calls .github/workflows/evidence-lab.yml at a reviewed full commit SHA. GitHub
joins Tailscale using short-lived OIDC and dispatches one experiment per Docker
container on lab. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail Evidence / required.

See [lab.md](lab.md) for trust and repository variables. No AWS account,
Modal setup, stored Tailscale secret, or self-hosted GitHub runner is required.

## Reports and publication

Routine CI reports are Actions artifacts in relflow/relflow, retained for 90
days; input wheels and requirements are retained for seven days. Before expiry,
promote a selected complete run to an immutable Release in relflow/docs:

```sh
gh workflow run publish-evidence.yml --repo relflow/docs \
  -f tag=<unique-release-tag> -f library_run_id=<library-actions-run-id>
```

The workflow verifies completeness, provenance, wheel digests, and the runtime
lock; preserves reports and inputs with checksums; and generates a validated
publication patch for evidence/results.yaml. Download evidence-publication-update,
review the patch on a branch at its recorded docs base, and open a normal PR.
The site's normal build reads the committed records and canonical sources.
There are no snapshot bundles, importers/exporters, or old Proofs redirects.

Raw report releases retain execution provenance. Updating source or output
readers does not establish a fresh learning result. The checkout keeps the
current selection without archived result snapshots or history readers.
