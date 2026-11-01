# %% [markdown]
# ---
# title: Can it add longer lists than it saw in training?
# categories:
# - Cardinality generalization
# evidence-id: P002
# description: Train on short lists, then ask for totals of longer lists.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Familiar lengths
#   metric:
#   - in_range_score
#   - nrmse
#   format: error
# - label: Longer lists
#   metric:
#   - unseen_score
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All measurements finite: true
#     Seen nRMSE < 0.35, unseen nRMSE < 0.30, duplication and cardinality errors < 0.20: true
#   duration_seconds: 40.583
#   metrics:
#     duplication_error: 0.07085658887561218
#     in_range_score:
#       baseline_rmse: 1.0401230906449561
#       nrmse: 0.015460484500793705
#       rmse: 0.01608080692183399
#     probe_error: 0.04737908791419197
#     probe_prediction:
#     - 0.663047156025836
#     - 1.9400064310296836
#     - 3.7542805325034276
#     probe_target:
#     - 0.65
#     - 1.9500000000000002
#     - 3.9000000000000004
#     steps: 1100
#     unseen_score:
#       baseline_rmse: 1.7874220810107455
#       nrmse: 0.07348960132621424
#       rmse: 0.1313569361351519
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: e45d9b8128fb9fd022fbe4f970b0f718529d3a8d6ec47ae8ee7cfa7d97f2b9eb
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 999c851b0d8f
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
#     script_sha256: b1df7e132512f7574f52c49afb6ced0635f1d86fc13127b34811dd1d7a7956df
#     source_sha256: 39752aa0b4f2aa76aaaa992ebceefecf0dbb45484905f875a0e81473d5646813
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 3615
#   started_at: '2026-11-01T00:04:56.243156+00:00'
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
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
# total: 4.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-attention-sum-unseen-lengths
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
# Compare familiar and unfamiliar lengths. Also repeat a whole list: its total should double.
#
# ## Result
#
# {{< evidence P002 status >}}
#
# The learned total extends to the tested longer lists. This does not establish exact addition or unlimited length.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P002 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P002 script >}}
#
# ### How it works
#
# The schema matches the [in-range sum](attention-sum-in-range.html):
# learned `rf.Attention` reductions receive raw amounts without a supplied count.
# Training lengths are one through six, while the unseen test set uses seven
# through ten. Branch capacity is twelve, allowing these inputs without overflow.
#
# A second intervention duplicates complete bags of at most five items and checks
# that predictions approximately double. Equal-value probes at lengths one,
# three, and six isolate sensitivity to multiplicity. The capacity bound permits
# these shapes; the accuracy and intervention gates establish the learned behavior.
#
# ### Remaining work
#
# Repeat the gates across seeds, capacities, and nested branches. Heavy tails,
# high duplication, empty collections, missing values, and overflow semantics need
# separate coverage. The doubling gate allows approximation error; it is not an
# exact-sum contract.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Characterize sum extrapolation beyond every trained collection length.

Require unseen-length accuracy and exact complete-bag scaling together.

Run: uv run python evidence/experiments/run.py P002"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P002"
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


def equal_value_probes(*, value: float, lengths: tuple[int, ...]) -> list[dict]:
    """Create matched bags that differ only in repetition count."""
    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {"items": [{"amount": value}] * length, "mean_amount": value, "total": length * value}
        rows.append(row)
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
    in_range = list(random_records(rows=512, seed=seed + 3))
    unseen = list(random_records(rows=512, seed=seed + 4, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 5, maximum=5))
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
    in_range_score = score(train=train, test=in_range, predicted=prediction(model, in_range))
    unseen_score = score(train=train, test=unseen, predicted=prediction(model, unseen))
    short_prediction = prediction(model, short)
    duplicated_prediction = prediction(model, duplicate(short))
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    probes = equal_value_probes(value=0.65, lengths=(1, 3, 6))
    probe_prediction = prediction(model, probes)
    probe_target = np.asarray([row["total"] for row in probes], dtype=np.float64)
    probe_error = rmse(probe_target, probe_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    metrics = {
        "in_range_score": in_range_score,
        "unseen_score": unseen_score,
        "duplication_error": duplication_error,
        "probe_error": probe_error,
        "probe_target": probe_target.tolist(),
        "probe_prediction": probe_prediction.tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "All measurements finite": bool(
            np.isfinite([in_range_score["nrmse"], unseen_score["nrmse"], duplication_error, probe_error]).all()
        ),
        "Seen nRMSE < 0.35, unseen nRMSE < 0.30, duplication and cardinality errors < 0.20": bool(
            in_range_score["nrmse"] < 0.35
            and unseen_score["nrmse"] < 0.3
            and (duplication_error < 0.2)
            and (probe_error < 0.2)
        ),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3615)

# %% [markdown]
# </details>
