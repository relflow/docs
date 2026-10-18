# Modeling proofs

Each experiment asks one modeling question using controlled synthetic data.
[`results.yaml`](results.yaml) registers its stable proof ID, canonical source,
and recorded evidence. [`SPEC.md`](SPEC.md) defines the experimental protocol,
controls, reporting rules, and planned coverage. The website presents the
[proof catalog](../docs/proofs.qmd) without executing experiments.

## RelFlow 0.3.1 compatibility

Executable readers use RelFlow 0.3.1 prediction output: nested decoded fields under
`predictions`, variable lists for repeated records, and separate
`representations` records for requested embeddings. A reader migration does not
change the experiment's stored measurements or prove that its quality gates
pass on the newer package. Existing results retain their recorded package,
source fingerprints, seeds, and training budgets.

The documentation build only publishes local source and archived evidence.
Running experiments is an explicit task in an environment with the intended
RelFlow release. For an application-supplied package artifact:

```bash
uv run --with /path/to/relflow.whl python proofs/run.py P001 --steps 2
uv run --with /path/to/relflow.whl pytest tests/proofs
```

A short smoke run checks execution. Full learning claims require the budgets,
controls, and quality gates defined by each experiment and the specification.
