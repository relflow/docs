# Modeling proofs

Each experiment asks one modeling question using controlled synthetic data.
[`results.yaml`](results.yaml) registers its stable proof ID, canonical source,
and recorded evidence. [`SPEC.md`](SPEC.md) defines the experimental protocol,
controls, reporting rules, and planned coverage. The website presents the
[proof catalog](../docs/proofs.qmd) without executing experiments.

## Run A Proof

Proofs train with `model.fit(...)` and data modules that bind automatically.
Batch size and optimization belong on the fit call. Fixed-budget runs disable
early stopping and rollback, and specify FP32 precision. Custom optimizer
factories preserve experiments that deliberately use Torch Adam or AdamW.

The documentation build only publishes local source and archived evidence.
Running experiments is an explicit task in an environment with the intended
RelFlow release. For an application-supplied package artifact:

```bash
uv run --with /path/to/relflow.whl python proofs/run.py P001 --steps 2
uv run --with /path/to/relflow.whl pytest tests/proofs
```

A short smoke run checks execution. Full learning claims require the budgets,
controls, and quality gates defined by each experiment and the specification.
