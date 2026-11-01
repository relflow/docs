# %% [markdown]
# ---
# title: Can it average prepared contributions to covariance?
# categories:
# - Distribution statistics
# evidence-id: P008
# description: Supply the centered products used to calculate covariance, then ask for their average.
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
#     Covariance nRMSE below 0.20: true
#   duration_seconds: 8.938
#   metrics:
#     measured:
#       baseline_rmse: 0.4641245625126908
#       nrmse: 0.023600511282418087
#       rmse: 0.010953576974028118
#     steps: 350
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c55062466efa917c1541c6467b0a51d57442904cb1bc50839a630bdcb9370833
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 51d9f02a57f6
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
#     script_sha256: d8d4c9f78c01684ca2eeb669de08bb89548e49087b3c79ab55696307e08057ee
#     source_sha256: 757f8c2f3e7b889581a8036294c1c7441d3e018cd8a002b109913805d110162f
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 3513
#   started_at: '2026-11-01T00:05:40.618877+00:00'
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
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 1.4}
# covariance: 0.8
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-supplied-cross-deviations
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, cross_deviation, covariance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("cross_deviation", type: "Number"),
#   )),
#   node("covariance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare the prediction with the correct average on new records.
#
# ## Result
#
# {{< evidence P008 status >}}
#
# The final averaging step works here. Centering and multiplication were already done by the data generator, so they were not learned in this test.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P008 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P008 script >}}
#
# ### How it works
#
# The generator supplies `(x - mean(x)) * (y - mean(y))` for every item.
# The model sees these eight centered products, not raw `x` and `y` values. The
# item branch has `attention=None` and `rf.Mean()`, followed by learned root
# attention and a covariance decoder.
#
# This tests whether an encoded average can be decoded into the population
# cross-moment. It bypasses centering and pairwise multiplication. Compare it with
# [aligned-pair covariance](aligned-pair-covariance.html) to localize a failure in
# sibling interaction rather than in the final reduction.
#
# ### Remaining work
#
# Repeat across seeds and broaden lengths, missing-value behavior, and
# outliers. This sufficient-statistic diagnostic should remain easier than the raw
# pairing task; it cannot establish that the model learned the supplied operation.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Supplied centered products isolate covariance reduction.

Mean learns covariance once centered products are supplied.

Run: uv run python evidence/experiments/run.py P008"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P008"
ITEMS = 8


def covariance_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw paired fields with controlled means, scales, and covariance."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        x_basis = rng.normal(size=ITEMS)
        x_basis -= x_basis.mean()
        x_basis /= np.sqrt(np.mean(np.square(x_basis)))
        orthogonal = rng.normal(size=ITEMS)
        orthogonal -= orthogonal.mean()
        orthogonal -= np.mean(orthogonal * x_basis) * x_basis
        orthogonal /= np.sqrt(np.mean(np.square(orthogonal)))
        correlation = float(rng.uniform(-0.88, 0.88))
        y_basis = correlation * x_basis + np.sqrt(1.0 - correlation * correlation) * orthogonal
        x_scale = float(rng.uniform(0.35, 1.45))
        y_scale = float(rng.uniform(0.35, 1.45))
        x_location = float(rng.uniform(-0.8, 0.8))
        y_location = float(rng.uniform(-0.8, 0.8))
        x_values = x_location + x_scale * x_basis
        y_values = y_location + y_scale * y_basis
        yield {
            "items": [{"x": float(x), "y": float(y)} for x, y in zip(x_values, y_values, strict=True)],
            "covariance": correlation * x_scale * y_scale,
            "correlation": correlation,
        }


def covariance_sufficient_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Expose per-item centered products while retaining the same targets."""
    rows_with_pairs = covariance_records(rows=rows, seed=seed)
    for row in rows_with_pairs:
        x = np.asarray([item["x"] for item in row["items"]], dtype=np.float64)
        y = np.asarray([item["y"] for item in row["items"]], dtype=np.float64)
        cross_deviation = (x - x.mean()) * (y - y.mean())
        yield {
            "items": [{"cross_deviation": float(value)} for value in cross_deviation],
            "covariance": row["covariance"],
        }


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["covariance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["covariance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["covariance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(covariance_sufficient_records(rows=768, seed=seed + 1))
    test = list(covariance_sufficient_records(rows=384, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, attention=None, reduction=rf.Mean(), cross_deviation=rf.Number),
        covariance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: covariance_sufficient_records(rows=768, seed=seed + 1),
        validate=lambda: covariance_sufficient_records(rows=192, seed=seed + 2),
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
    checks = {"Covariance nRMSE below 0.20": bool(measured["nrmse"] < 0.2)}
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3513)

# %% [markdown]
# </details>
