# %% [markdown]
# ---
# title: Does repeating a list preserve its average?
# categories:
# - Cardinality generalization
# evidence-id: P003
# description: Predict the average amount, using a branch that averages its encoded items without item attention;
#   the branch's count lane sees the repetition, the average must not.
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
# - label: Change after repetition
#   metric:
#   - duplication_drift
#   format: number
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Duplication drift below 0.05 target SD: true
#     Mean nRMSE below 0.25: true
#     Permutation drift below 1e-6 target SD: true
#   duration_seconds: 9.522
#   metrics:
#     duplication_drift: 0.0011719380509312303
#     measured:
#       baseline_rmse: 0.4464820375017577
#       nrmse: 0.00950241596273492
#       rmse: 0.004242658040231113
#     permutation_drift: 5.3915851615687914e-08
#     steps: 350
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3d947fcad36c332871d74f477418c0d2a5ca6b1199530463bc08608f9499ec14
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 32f655491308
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
#     script_sha256: b8911e9850d0e5f16d490ecf08ed604165ed2f3dba8c91dcacdbd1ad34fe078d
#     source_sha256: 2668cbc7d230ffef4ad0d6d1983541f5d3254c194805193c68f30bc4b7b12ac6
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 3600
#   started_at: '2026-11-01T00:04:56.129521+00:00'
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
#   - {amount: 0.2}
#   - {amount: 0.8}
# mean_amount: 0.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mean-duplicate-invariance
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, mean_amount. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Reorder the list and repeat every item. Both changes should leave its average unchanged.
#
# ## Result
#
# {{< evidence P003 status >}}
#
# This setup learns the average and keeps it under repetition to within a thousandth of the target spread. The summary itself is no longer identical: Mean carries a count lane, so the same summary can also tell a total apart.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P003 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P003 script >}}
#
# ### How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. It averages encoded
# Number tokens rather than directly averaging raw amounts, and adds a learned
# direction scaled by the number of present items over the branch capacity. A
# learned downstream path must still map that representation to the
# arithmetic-mean target.
#
# Without preceding item attention, permuting tokens leaves the summary
# unchanged exactly, and the test checks that invariance at numerical precision.
# Duplicating the complete bag leaves the averaged content unchanged but doubles
# the count lane, so the summary differs; the mean readout learns to ignore that
# lane, and the test bounds the remaining drift at five percent of the target
# spread. The same lane is what lets a total be recovered from this route, which
# the [contribution count](mean-keeps-contribution-count.html) and
# [cardinality](../structure/mean-keeps-cardinality.html) evidences measure.
#
# ### Remaining work
#
# Repeat across seeds and test missing values, empty collections, and nested
# placement. The invariance applies to this no-item-attention path; it does not
# imply that every model using Mean is insensitive to multiplicity.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Mean keeps its average under repetition with item attention disabled.

Learn the average, retain it exactly under permutation, and keep it within
a small tolerance when the whole bag is duplicated and only the count lane moves.

Run: uv run python evidence/experiments/run.py P003"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P003"
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


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
    return rows


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
    return np.asarray([row["mean_amount"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["mean_amount"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["mean_amount"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=768, seed=seed + 1))
    test = list(random_records(rows=384, seed=seed + 3, maximum=3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=768, seed=seed + 1),
        validate=lambda: random_records(rows=192, seed=seed + 2),
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
    intact_prediction = prediction(model, test)
    measured = score(train=train, test=test, predicted=intact_prediction)
    permutation_prediction = prediction(model, permute(test, seed=seed + 4))
    duplication_prediction = prediction(model, duplicate(test))
    scale = float(np.std(np.asarray([row["mean_amount"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permutation_prediction) / scale
    duplication_drift = rmse(intact_prediction, duplication_prediction) / scale
    metrics = {
        "measured": measured,
        "permutation_drift": permutation_drift,
        "duplication_drift": duplication_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "Mean nRMSE below 0.25": bool(measured["nrmse"] < 0.25),
        "Permutation drift below 1e-6 target SD": bool(permutation_drift < 1e-06),
        "Duplication drift below 0.05 target SD": bool(duplication_drift < 0.05),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3600)

# %% [markdown]
# </details>
