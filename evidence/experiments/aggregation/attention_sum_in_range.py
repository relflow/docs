# %% [markdown]
# ---
# title: Can it add a list of numbers?
# categories:
# - Cardinality generalization
# evidence-id: P001
# description: Predict the total of a list whose length varies within the range used in training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New lists
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Permutation drift below 0.12 target SD: true
#     Seen-length nRMSE below 0.35: true
#   duration_seconds: 42.13
#   metrics:
#     measured:
#       baseline_rmse: 1.0961391615079292
#       nrmse: 0.012362172408772186
#       rmse: 0.013550661298568002
#     permutation_drift: 0.005140111435372105
#     steps: 1100
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 65a5e601f33eb9e0bf0bc6a616c149731ab3dea3b8eba45e398f93a0de2d229b
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: d352a8c9cf8d
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
#     script_sha256: ae067fcebfc33c74fda92a0e1b21f2957cb94ec1f9893ff4df0c3f491a843123
#     source_sha256: f5f89eaea91513831f487412ab8e7aef135751915799b968874860bf3ed3d513
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 3609
#   started_at: '2026-11-01T00:04:55.854750+00:00'
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
#   - {amount: 0.5}
#   - {amount: -0.2}
#   - {amount: 0.9}
# total: 1.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-attention-sum-in-range
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test on fresh lists, then reorder the same items. Reordering should preserve the total.
#
# ## Result
#
# {{< evidence P001 status >}}
#
# The model learns an approximate total in this range. Longer lists need a separate test.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P001 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P001 script >}}
#
# ### How it works
#
# Training, validation, and test records contain one through six independent
# amounts drawn from −1 to 1. Both the repeated item branch and root use learned
# `rf.Attention` reductions. There is no visible count input.
#
# Attention’s reduction retains aggregate evidence and present-token count as
# well as its normalized learned summary. These representations offer a route to
# predicting sums with changing length. Independent values prevent count alone
# from solving the task; a complete-item permutation checks approximate order
# stability. [Unseen-length tests](attention-sum-unseen-lengths.html) separately
# check whether the learned behavior extends beyond interpolation.
#
# ### Remaining work
#
# Repeat across seeds and change capacities, widths, and nested placement.
# This in-range result alone establishes neither arbitrary-length generalization
# nor an exact arithmetic sum; empty and missing-item cases need separate checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Learned Attention can interpolate a sum over seen cardinalities.

Check ordinary in-range accuracy and approximate order invariance.

Run: uv run python evidence/experiments/run.py P001"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P001"
TRAIN_MAX = 6
CAPACITY = 12


def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        yield row


def permute(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1536, seed=seed + 1))
    test = list(random_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(
            length=CAPACITY, attention="mha", n_layers=2, reduction=rf.Attention(n_layers=2), amount=rf.Number
        ),
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=1536, seed=seed + 1),
        validate=lambda: random_records(rows=384, seed=seed + 2),
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
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact_prediction = prediction(model, test)
    measured = score(train=train, test=test, predicted=intact_prediction)
    permuted_prediction = prediction(model, permute(test, seed=seed + 5))
    target_scale = float(np.asarray([row["total"] for row in test], dtype=np.float64).std())
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    metrics = {"measured": measured, "permutation_drift": permutation_drift, "steps": trainer.global_step}
    checks = {
        "Seen-length nRMSE below 0.35": bool(measured["nrmse"] < 0.35),
        "Permutation drift below 0.12 target SD": bool(permutation_drift < 0.12),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3609)

# %% [markdown]
# </details>
