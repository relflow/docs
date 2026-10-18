# %% [markdown]
# ---
# title: How much wall time does predictive reassignment add?
# categories:
# - Grouping
# proof-id: P081
# description: Compare native training and frozen-neural predictive reassignment with matched initialization, observations, masking, and training budgets.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Training wall time increase
#   metric: [wall_time_increase_fraction]
#   format: percent
# - label: Extra prediction rows
#   metric: [extra_forward_rows]
#   format: number
# ---
#
# ## Example
#
# ```yaml
# entity: entity-0017
# x: 0.5
# y: -6.27
# ```
#
# ```{typst}
# //| label: fig-proof-group-reassignment-cost
# //| fig-cap: "Both arms train this same model. Reassignment adds frozen prediction passes at updates 200, 400, and 600."
# //| fig-alt: "An event has entity Group and x Number inputs, with hidden y Number target. The entity has a 10 percent reconstruction mask in both arms."
# #tree(node("event", kind: "root", children: (
#   node("entity", type: "Group", detail: "10% reconstruction mask"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Time 600 optimizer updates in each arm on one CPU thread. Native training
# receives no reassignment callback. The candidate additionally evaluates all
# available groups against training residuals and accepts a proposed assignment
# only when validation mean squared error improves. Both arms have identical
# ordinary validation events and matching audit instrumentation.
#
# ## Result
#
# {{< proof P081 status >}}
#
# This proof measures wall time on the recorded machine, not FLOPs. Prediction
# quality is a diagnostic, not a timing success gate. See the three paired seeds
# and individual timing components before generalizing the percentage.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P081 evidence >}}
#
# ### Run this experiment
#
# {{< proof P081 script >}}
#
# ### Timing protocol
#
# The original P078 generator supplies 4,096 training and 1,024 validation
# observations of the same 64 identities. Four balanced hidden numerical rules
# generate y from x. Bounds are 2 through 8; the true count is not supplied.
# Both arms receive the same initial parameters, ordered observations, masks,
# native Torch random stream through update 200, and 600 AdamW updates of 128
# rows. The native arm retains the package's extension callbacks; it has no
# experimental reassignment callback. The audit callback is identical in both.
#
# The candidate's frozen v1 numerical intervention runs after updates 200,
# 400, and 600. All eight available groups each evaluate the 4,096 training
# rows, adding 98,304 counterfactual forward rows. Before/after validation
# adds 6,144 further forward rows. These 104,448 extra forward rows are
# separate from 76,800 training rows and the measured ordinary validation
# workload. No extra backward passes or optimizer updates are introduced.
#
# Two excluded two-update warmups, one per arm, precede the first measured
# pair in a process. Measured order alternates native/candidate, then
# candidate/native, then native/candidate across seeds 7801, 7802, and 7803.
# No checkpoint or result I/O occurs inside the timers. Construction, fit,
# terminal update-600 reassignment, and readout have separate timers. The
# primary comparison is fit plus terminal reassignment; fit already includes
# reassignment at updates 200 and 400. Event time is also reported as a share
# of candidate wall time and relative to its own non-event time; those are
# not substitutes for the fresh native comparison. Smoke caps are not timing
# evidence. No test split is generated or inspected.
#
# ### Complete experiment code
#

# %%
"""P081: matched CPU wall time for native training versus predictive EM v1."""

import hashlib
import json
import math
import time
from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P081"
BUDGET = 600
BATCH_SIZE = 128
TRAIN_ROWS = 4096
VALIDATION_ROWS = 1024
ENTITY_COUNT = 64
REGIME_COUNT = 4
INTERCEPTS = np.asarray((-6.0, -2.0, 2.0, 6.0))
SLOPES = np.asarray((-0.6, 0.6, -0.6, 0.6))
WARMED = False


def partition(seed: int) -> np.ndarray:
    """Keep the hidden partition fixed across independent observations."""
    return np.random.default_rng(seed).permutation(np.arange(ENTITY_COUNT) % REGIME_COUNT)


def records(*, rows: int, seed: int, partition_seed: int) -> Iterator[dict]:
    """Original stable P078 observations; neither labels nor parameters are inputs."""
    rng = np.random.default_rng(seed)
    entities = np.resize(np.arange(ENTITY_COUNT), rows)
    rng.shuffle(entities)
    groups = partition(partition_seed)[entities]
    x = rng.uniform(-1.0, 1.0, rows)
    y = INTERCEPTS[groups] + SLOPES[groups] * x + rng.normal(0.0, 0.1, rows)
    for entity, value, target in zip(entities, x, y, strict=True):
        yield {"entity": f"entity-{entity:04d}", "x": float(value), "y": float(target)}


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        entity=rf.Group(bounds=(2, 8), p_unavailable=0.0, mask=rf.Mask(rate=0.1, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=0.003)
    return model


def adjusted_rand(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Measure partition agreement without depending on arbitrary group names."""
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)
    joint = float(np.sum(cells * (cells - 1) / 2))
    left_pairs = float(np.sum(cells.sum(axis=1) * (cells.sum(axis=1) - 1) / 2))
    right_pairs = float(np.sum(cells.sum(axis=0) * (cells.sum(axis=0) - 1) / 2))
    expected = left_pairs * right_pairs / (len(actual) * (len(actual) - 1) / 2)
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


def tensor_hash(values) -> str:
    """Fingerprint state without consuming the native random stream."""
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Audit(lit.Callback):
    """Match workload and randomness while adding identical work to both arms."""

    def __init__(self):
        self.training_rows = 0
        self.training_batches = 0
        self.validation_rows = 0
        self.validation_batches = 0
        self.validation_events = []
        self.batches = hashlib.sha256()
        self.masks = hashlib.sha256()
        self.pre_intervention_rng = hashlib.sha256()
        self.first_bound_weights = None
        self.first_bound_step = None

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if self.first_bound_weights is None:
            self.first_bound_weights = tensor_hash(model.named_parameters())
            self.first_bound_step = trainer.global_step
        self.training_rows += len(batch.source)
        self.training_batches += 1
        self.batches.update(json.dumps(batch.source.to_pylist(), sort_keys=True, separators=(",", ":")).encode())
        self.masks.update(batch.tensors["/entity"].trainable.cpu().numpy().tobytes())
        if trainer.global_step < 200:
            self.pre_intervention_rng.update(torch.get_rng_state().numpy().tobytes())

    def on_validation_start(self, trainer, model):
        self.validation_events.append({"step": trainer.global_step, "rows": 0, "batches": 0})

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        count = len(batch.source)
        self.validation_rows += count
        self.validation_batches += 1
        self.validation_events[-1]["rows"] += count
        self.validation_events[-1]["batches"] += 1

    def result(self):
        return {
            "training_rows": self.training_rows,
            "training_batches": self.training_batches,
            "regular_validation_rows": self.validation_rows,
            "regular_validation_batches": self.validation_batches,
            "regular_validation_events": self.validation_events,
            "ordered_batches_sha256": self.batches.hexdigest(),
            "training_masks_sha256": self.masks.hexdigest(),
            "native_rng_before_first_intervention_sha256": self.pre_intervention_rng.hexdigest(),
            "first_bound_weights_sha256": self.first_bound_weights,
            "first_bound_step": self.first_bound_step,
        }


def arrays(rows):
    return (
        np.asarray([int(row["entity"].split("-")[-1]) for row in rows]),
        np.asarray([row["x"] for row in rows]),
        np.asarray([row["y"] for row in rows]),
    )


def neural_predict(model, rows):
    visible = pa.Table.from_pylist([{"entity": row["entity"], "x": row["x"]} for row in rows])
    original_batch_size = model.batch_size
    try:
        output = model.predict(visible, batch_size=256)["predictions"].combine_chunks().field("y").field("content")
    finally:
        model.batch_size = original_batch_size
    return output.to_numpy(zero_copy_only=False)


def intervene(model, train, validate, optimizers, mechanism):
    """Predictive residual search versus a matched margin/history-reset control."""
    started = time.perf_counter()
    embedder = model.nodes["/entity"].embedder
    vocabulary = list(rf.Group.vocabulary(model, "/entity"))
    weight = embedder.embeddings["group"].weight
    before = weight.detach().clone()
    entity_order = np.asarray([int(value.split("-")[-1]) for value in vocabulary])
    available = (~embedder.retired).nonzero(as_tuple=True)[0].tolist()
    validation_target = arrays(validate)[2]
    before_mse = float(np.mean((neural_predict(model, validate) - validation_target) ** 2))
    row_mass = before[: len(vocabulary)].logsumexp(-1, keepdim=True)
    counterfactual_seconds = 0.0
    if mechanism == "predictive":
        train_entity, _, train_target = arrays(train)
        errors = np.full((64, embedder.size), np.inf)
        tick = time.perf_counter()
        try:
            for group_id in available:
                forced = torch.full_like(before[: len(vocabulary)], math.log(1e-8))
                forced[:, group_id] = 0
                with torch.no_grad():
                    weight[: len(vocabulary)].copy_(row_mass + forced)
                predicted = neural_predict(model, train)
                errors[:, group_id] = np.bincount(train_entity, weights=(predicted - train_target) ** 2, minlength=64)
        finally:
            with torch.no_grad():
                weight.copy_(before)
        counterfactual_seconds = time.perf_counter() - tick
        proposed = errors.argmin(-1)
    else:
        proposed = np.empty(64, dtype=np.int64)
        proposed[entity_order] = embedder.support(before[: len(vocabulary)]).argmax(-1).cpu().numpy()
    forced = torch.full_like(before[: len(vocabulary)], math.log(1e-8))
    forced[torch.arange(len(vocabulary)), torch.as_tensor(proposed[entity_order])] = 0
    with torch.no_grad():
        weight[: len(vocabulary)].copy_(row_mass + forced)
    after_mse = float(np.mean((neural_predict(model, validate) - validation_target) ** 2))
    # Hardening deliberately retains existing membership, so it is always applied.
    accepted = mechanism == "harden" or after_mse < before_mse
    if accepted:
        if weight.grad is not None:
            weight.grad[: len(vocabulary)].zero_()
        for optimizer in optimizers:
            for value in optimizer.state.get(weight, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                    value[: len(vocabulary)].zero_()
    else:
        with torch.no_grad():
            weight.copy_(before)
    return {
        "mechanism": mechanism,
        "event_wall_seconds": time.perf_counter() - started,
        "counterfactual_seconds": counterfactual_seconds,
        "counterfactual_training_rows_evaluated": len(train) * len(available) if mechanism == "predictive" else 0,
        "selection_validation_rows_evaluated": 2 * len(validate),
        "candidate_groups": available,
        "before_validation_mse": before_mse,
        "proposed_validation_mse": after_mse,
        "accepted_by_validation": accepted if mechanism == "predictive" else None,
        "proposed_assignments": proposed.tolist(),
    }


class Alternation(lit.Callback):
    """Run the frozen v1 numerical intervention at the original update boundaries."""

    def __init__(self, train, validate):
        self.train_rows = train
        self.validation_rows = validate
        self.events = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step not in (200, 400) or any(event["step"] == step for event in self.events):
            return
        event = intervene(pl_module, self.train_rows, self.validation_rows, trainer.optimizers, "predictive")
        self.events.append({"step": step, **event})


def fit_arm(seed: int, budget: int, accelerator: str, mechanism: str) -> dict:
    lit.seed_everything(seed, workers=True)
    tick = time.perf_counter()
    model = build()
    initial_weights = tensor_hash(model.named_parameters())
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1, partition_seed=seed * 100))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed * 100 + 2, partition_seed=seed * 100))
    data_hash = hashlib.sha256(json.dumps([train, validate], sort_keys=True).encode()).hexdigest()
    data = rf.SyntheticDataModule(model=model, train=partial(iter, train), validate=partial(iter, validate), seed=seed)
    audit = Audit()
    alternation = Alternation(train, validate) if mechanism == "predictive" else None
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_steps=budget,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[audit] if alternation is None else [audit, alternation],
    )
    construction_wall_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    trainer.fit(model, datamodule=data)
    fit_wall_seconds = time.perf_counter() - tick
    model.eval()
    terminal_wall_seconds = 0.0
    if alternation is not None:
        tick = time.perf_counter()
        event = intervene(model, train, validate, trainer.optimizers, "predictive")
        terminal_wall_seconds = time.perf_counter() - tick
        alternation.events.append({"step": trainer.global_step, **event})
    events = [] if alternation is None else alternation.events
    tick = time.perf_counter()
    frozen_before = tensor_hash((*model.named_parameters(), *model.named_buffers()))
    assignment_rows = rf.Group.assignments(model, "/entity")
    assignments = np.asarray([assignment_rows.get(f"entity-{i:04d}", {}).get("group", -1) for i in range(64)])
    predicted = neural_predict(model, validate)
    target = arrays(validate)[2]
    mean = float(arrays(train)[2].mean())
    baseline_rmse = float(np.mean((mean - target) ** 2) ** 0.5)
    rmse = float(np.mean((predicted - target) ** 2) ** 0.5)
    frozen_after = tensor_hash((*model.named_parameters(), *model.named_buffers()))
    frozen_assignments = assignment_rows == rf.Group.assignments(model, "/entity")
    readout_wall_seconds = time.perf_counter() - tick
    return {
        "seed": seed,
        "mechanism": mechanism,
        "optimizer_updates": trainer.global_step,
        "batch_size": BATCH_SIZE,
        "initial_weights_sha256": initial_weights,
        "input_rows_sha256": data_hash,
        "construction_wall_seconds": construction_wall_seconds,
        "fit_wall_seconds": fit_wall_seconds,
        "terminal_reassignment_wall_seconds": terminal_wall_seconds,
        "training_and_reassignment_wall_seconds": fit_wall_seconds + terminal_wall_seconds,
        "readout_wall_seconds": readout_wall_seconds,
        "events": events,
        "event_wall_seconds": sum(event["event_wall_seconds"] for event in events),
        "counterfactual_forward_rows": sum(event["counterfactual_training_rows_evaluated"] for event in events),
        "selection_validation_forward_rows": sum(event["selection_validation_rows_evaluated"] for event in events),
        "evaluation_preserves_parameters_and_buffers": frozen_before == frozen_after,
        "evaluation_preserves_assignments": frozen_assignments,
        "all_parameter_and_buffer_values_finite": all(
            bool(torch.isfinite(value).all()) for value in (*model.parameters(), *model.buffers())
        ),
        "audit": audit.result(),
        "validation": {
            "rmse": rmse,
            "normalized_rmse": rmse / baseline_rmse,
            "adjusted_rand": adjusted_rand(partition(seed * 100), assignments),
            "occupied_groups": len(set(assignments.tolist())),
            "assigned_entities": int((assignments >= 0).sum()),
            "assignments": assignments.tolist(),
        },
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    global WARMED
    if accelerator != "cpu" or torch.get_num_threads() != 1:
        raise ValueError("P081 requires --accelerator cpu --threads 1 for comparable CPU wall time")
    budget = min(BUDGET, steps) if steps is not None else BUDGET
    warmups = []
    if not WARMED:
        for mechanism in ("native", "predictive"):
            result = fit_arm(7799, 2, accelerator, mechanism)
            warmups.append({"mechanism": mechanism, "optimizer_updates": result["optimizer_updates"], "excluded": True})
        WARMED = True
    order = ("native", "predictive") if seed % 2 else ("predictive", "native")
    measured = {mechanism: fit_arm(seed, budget, accelerator, mechanism) for mechanism in order}
    native, candidate = measured["native"], measured["predictive"]
    base_wall = native["training_and_reassignment_wall_seconds"]
    candidate_wall = candidate["training_and_reassignment_wall_seconds"]
    event_wall = candidate["event_wall_seconds"]
    metrics = {
        "run_order": list(order),
        "excluded_warmups": warmups,
        "budget_per_arm": budget,
        "torch_threads": torch.get_num_threads(),
        "native": native,
        "predictive": candidate,
        "wall_time_increase_fraction": candidate_wall / base_wall - 1,
        "wall_time_increase_percent": (candidate_wall / base_wall - 1) * 100,
        "event_fraction_of_candidate_wall_time": event_wall / candidate_wall,
        "event_fraction_of_candidate_non_event_wall_time": event_wall / (candidate_wall - event_wall),
        "extra_forward_rows": candidate["counterfactual_forward_rows"] + candidate["selection_validation_forward_rows"],
        "not_a_flops_measurement": True,
    }
    checks = {}
    for mechanism, result in measured.items():
        checks[f"{mechanism}: exactly the configured optimizer updates"] = result["optimizer_updates"] == budget
        checks[f"{mechanism}: exactly the configured training rows"] = (
            result["audit"]["training_rows"] == budget * BATCH_SIZE
        )
        checks[f"{mechanism}: frozen evaluation state"] = (
            result["evaluation_preserves_parameters_and_buffers"] and result["evaluation_preserves_assignments"]
        )
        checks[f"{mechanism}: finite parameters and diagnostic metrics"] = result[
            "all_parameter_and_buffer_values_finite"
        ] and bool(
            np.isfinite(
                [
                    result["validation"]["rmse"],
                    result["validation"]["normalized_rmse"],
                    result["validation"]["adjusted_rand"],
                ]
            ).all()
        )
        checks[f"{mechanism}: finite positive measured wall time"] = (
            math.isfinite(result["training_and_reassignment_wall_seconds"])
            and result["training_and_reassignment_wall_seconds"] > 0
        )
    for key in ("initial_weights_sha256", "input_rows_sha256"):
        checks[f"Matched {key}"] = native[key] == candidate[key]
    for key in (
        "ordered_batches_sha256",
        "training_masks_sha256",
        "native_rng_before_first_intervention_sha256",
        "first_bound_weights_sha256",
        "regular_validation_events",
        "regular_validation_rows",
        "regular_validation_batches",
    ):
        checks[f"Matched {key}"] = native["audit"][key] == candidate["audit"][key]
    checks["Native arm has no reassignment events"] = not native["events"]
    expected_steps = [step for step in (200, 400) if step <= budget] + [budget]
    checks["Candidate uses the original intervention update schedule"] = [
        event["step"] for event in candidate["events"]
    ] == expected_steps
    checks["Extra forward row accounting matches evaluated groups"] = candidate["counterfactual_forward_rows"] == sum(
        TRAIN_ROWS * len(event["candidate_groups"]) for event in candidate["events"]
    ) and candidate["selection_validation_forward_rows"] == 2 * VALIDATION_ROWS * len(candidate["events"])
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7801)

# %% [markdown]
# </details>
