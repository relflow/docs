# RelFlow evidence

This repository owns the canonical experiments and their recorded evidence.
Preserve stable P identifiers, data generators, seeds, budgets, optimizers,
model capacities, checks, and archived provenance when maintaining the suite.
Read SPEC.md before changing experimental semantics. Never rewrite archived
measurements to make a gate pass.

Run `uv sync --locked`, `uv run ruff check .`, `uv run ruff format --check .`,
and `uv run pytest` for static tooling. Runtime tests under tests/runtime are
explicit and require an independently supplied RelFlow wheel. Documentation
exports require only PyYAML and must never import RelFlow, Torch, or Lightning.

Lab runs use the exact supplied baseline and candidate wheels and record each
experiment and seed separately. Smoke or partial suite runs never satisfy the
required gate. Execution failures, incomplete results, and regressions in checks
that pass on the baseline must block. Keep credentials out of experiment workers. The lab backend isolates each task
in a network-disabled Docker container; no host home or Docker socket may be
mounted into experiment containers. Modal is an optional backend.
Reports use GitHub Actions artifacts and immutable GitHub Release bundles;
there must be no AWS account or S3 dependency.

