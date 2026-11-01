# %% [markdown]
# ---
# title: Can it average prepared squared differences?
# categories:
# - Distribution statistics
# evidence-id: P009
# description: Supply each value’s squared distance from the average, then predict the average of those distances.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Variance nRMSE below 0.20: true
#   duration_seconds: 9.572
#   metrics:
#     measured:
#       baseline_rmse: 0.4800716415268894
#       nrmse: 0.012960816013301747
#       rmse: 0.0062221202190337645
#     steps: 350
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3fbc0578397be881a8502d628e1b6f2d70deef37ff67e88619cd02bd47e86d43
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 58388c7ac30f
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.0
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 9c246da8e84fb3414ba738d8155cf481de50b0b7
#     relflow_source_sha256: 97b7b12d6fdc05c829f00a6e4a36559f0a1ae4bcc5c53cc5406a0b2c88587aca
#     script_sha256: 62ea2b92463f93b1bd7845bbab12b27e5565107f86e65527f3a1c9937400950f
#     source_sha256: 7ab17bdaffd249e4ab3b2997e91a984353b3e5403e812969a535aedab6f4b2d4
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 3500
#   started_at: '2026-11-01T00:05:43.587694+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41254785503-2
#   candidate:
#     revision: 9c246da8e84fb3414ba738d8155cf481de50b0b7
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   baseline:
#     revision: be1e01d4b9540aa3803fbc190fa4a1200629e32d
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#   manifest_sha256: 208987bcb5a67bd796f14c2eb24d922bb056eaca1d2f2bfd2b8a999c28a7128a
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: e1831feaed3b6984565784048b66e8f47ae533b1
#     html_url: https://github.com/relflow/relflow/actions/runs/41254785503
#     run_attempt: 2
#     run_id: 41254785503
# ---
#
# ## Example
#
# ```yaml
# items:
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
# variance: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-supplied-squared-deviations
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, squared_deviation, variance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("squared_deviation", width: 150pt, type: "Number"),
#   )),
#   node("variance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare the prediction with the correct variance on new records.
#
# ## Result
#
# {{< evidence P009 status >}}
#
# This tests the final averaging step. The inputs already contain the difficult parts of calculating variance.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P009 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P009 script >}}
#
# ### How it works
#
# The generator computes `(value - mean(value)) ** 2` before encoding. The
# item branch uses `attention=None` and `rf.Mean()`; the root uses learned
# attention to predict the average of eight supplied squared deviations.
#
# Mean averages encoded tokens, so the decoder still has to learn their numerical
# interpretation. Passing this diagnostic supports reduction and decoding of
# sufficient statistics. It does not prove learned centering or squaring; the
# [raw-value variance evidence](raw-value-variance.html) checks those requirements.
#
# ### Remaining work
#
# Repeat this control across seeds and broaden collection lengths and edge
# cases. Keep population-versus-sample variance semantics explicit, and retain the
# raw-value comparison when claiming that dispersion was learned from inputs.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Supplied squared deviations isolate collection averaging.

Mean learns population variance once squared deviations are supplied.

Run: uv run python evidence/experiments/run.py P009"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P009"
ITEMS = 8


def dispersion_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw rows whose location and spread are statistically independent."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        location = float(rng.uniform(-0.8, 0.8))
        scale = float(rng.uniform(0.12, 1.25))
        values = location + scale * basis
        items = [{"squared_deviation": float(np.square(value - location))} for value in values]
        yield {"items": items, "variance": scale * scale}


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(dispersion_records(rows=768, seed=seed + 1))
    test = list(dispersion_records(rows=384, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, attention=None, reduction=rf.Mean(), squared_deviation=rf.Number),
        variance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: dispersion_records(rows=768, seed=seed + 1),
        validate=lambda: dispersion_records(rows=192, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=350 if steps is None else min(steps, 350),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    measured = score(train=train, test=test, predicted=prediction(model, test))
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Variance nRMSE below 0.20": bool(measured["nrmse"] < 0.2)}
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3500)

# %% [markdown]
# </details>
