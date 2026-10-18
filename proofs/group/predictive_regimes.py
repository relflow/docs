# %% [markdown]
# ---
# title: Can predictive reassignment recover different kinds of groups?
# categories:
# - Grouping
# proof-id: P079
# description: Test an experimental training callback on crossing curves, unequal groups, and different hidden group counts.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Four separated curves
#   metric: [separated_four, normalized_rmse]
#   format: error
# - label: Four crossing curves
#   metric: [crossing_four, normalized_rmse]
#   format: error
# - label: Six hidden groups
#   metric: [separated_six, normalized_rmse]
#   format: error
# ---
#
# ## Example
#
# ```yaml
# entity: entity-0021
# x: 0.5
# y: -6.27
# ```
#
# Think of an entity as a merchant with repeated observations. The numerical
# input changes between observations; its hidden response rule stays the same.
# The answer `y` is shown here for explanation and hidden when predicting.
#
# ```{typst}
# //| label: fig-proof-predictive-regimes
# //| fig-cap: "The entity and x are inputs; y is the hidden answer. An experimental callback revisits training errors."
# //| fig-alt: "A root record with an entity Group input, numerical x input, and hidden numerical y target."
# #tree(node("observation", kind: "root", children: (
#   node("entity", type: "Group", detail: "Repeated opaque identity"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Train the same model on separated curves, crossing curves with equal means,
# unequal group sizes, and three or six hidden groups. A shuffled-identity
# control removes the stable link between an entity and its response.
#
# ## Result
#
# {{< proof P079 status >}}
#
# This is a proof-local training experiment, not the behavior of ordinary
# `Model.fit`. Recovering the right number of groups is checked separately
# from prediction error: one can succeed while the other fails. Recorded
# failures remain failures; the callback is not a general grouping guarantee.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P079 evidence >}}
#
# ### Run this experiment
#
# {{< proof P079 script >}}
#
# ### Process and split
#
# Each process has 64 opaque identities. All splits contain the same identities
# but independent numerical inputs and noise. Generate
# `y = intercept[group] + slope[group] * x + Normal(0, 0.1)`, with
# `x ~ Uniform(-1, 1)`. Group membership is fixed per identity and hidden from
# the model. Crossing curves have zero intercepts, so outcome means alone do
# not identify their slopes. The unequal case has 40, 12, 8, and 4 identities
# in its four groups. The other positive cases have three, four, or six groups.
# The model always receives the broad bounds `(2, 8)`, never the true count.
#
# The split unit is a new observation of a known identity. This does not claim
# group discovery for an unseen opaque identity without behavioral history.
# The no-regime process preserves every numerical pair and identity frequency,
# then permutes identity labels independently in each split.
#
# ### Experimental training method
#
# Every arm receives 4,096 training rows, 1,024 validation rows, and 2,048 test
# rows. Fit the same `xs` model for 600 AdamW updates, batch 128, learning rate
# 0.003. The ordinary Group reconstruction mask still hides 10% of training
# identity inputs. At updates 200, 400, and 600, a proof-local callback tries
# each available group on the training rows, preserving their actual x values.
# It sums squared prediction errors per identity and proposes the best group.
# Accept the complete proposal only when validation MSE improves. Continue
# training after updates 200 and 400; no updates follow the final proposal.
# Replaced rows use the existing hard-assignment margin, retain their previous
# row log-mass as an offset, and clear stale Adam moments while retaining its
# scalar step. No test outcomes or generating group labels guide these choices.
#
# Extra forward passes are real compute. Record counterfactual rows, validation
# rows, event wall time, fit wall time, and the fraction of measured fit-plus-
# final-proposal time spent inside the intervention. Those wall fractions are
# descriptive; they are not hardware-independent FLOP percentages.
#
# ### Fixed gates and controls
#
# Every positive case requires ARI at least 0.80, source-unit normalized RMSE at
# most 0.35, and exactly the generating number of occupied groups. The count
# gate is an additional challenge in this new proof; it does not change P078.
# Per-regime errors are recorded without adding a new worst-regime threshold.
# Shuffling test identities must give nRMSE at least 0.85 and raise it by 0.40.
# The separately trained no-regime control requires ARI at most 0.15 and nRMSE
# at least 0.85; its group count is diagnostic because the lower bound is two.
#
# Evaluation must preserve vocabulary, parameters, buffers, and assignments.
# Permuting every group coordinate consistently must preserve predictions and
# merely relabel assignments. Unseen identities must not grow the vocabulary;
# their prediction error is diagnostic, with no cold-start accuracy claim.
# All six cases and gates are declared before running seeds 7901, 7902, 7903.
# This bounded panel can expose failure; it does not establish universal
# recovery, generic per-example losses, nested identity attribution, or DDP.
#
# ### Complete experiment code
#

# %%
"""P079: test predictive membership refinement across distinct numeric group processes."""

import math
import time
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P079"
BUDGET = 600
BATCH_SIZE = 128
ENTITY_COUNT = 64
TRAIN_ROWS = 4096
VALIDATION_ROWS = 1024
TEST_ROWS = 2048
CASES = {
    "separated_four": {
        "populations": (16, 16, 16, 16),
        "intercepts": (-6.0, -2.0, 2.0, 6.0),
        "slopes": (-0.6, 0.6, -0.6, 0.6),
    },
    "crossing_four": {
        "populations": (16, 16, 16, 16),
        "intercepts": (0.0, 0.0, 0.0, 0.0),
        "slopes": (-6.0, -2.0, 2.0, 6.0),
    },
    "imbalanced_four": {
        "populations": (40, 12, 8, 4),
        "intercepts": (-6.0, -2.0, 2.0, 6.0),
        "slopes": (-0.6, 0.6, -0.6, 0.6),
    },
    "separated_three": {
        "populations": (22, 21, 21),
        "intercepts": (-6.0, 0.0, 6.0),
        "slopes": (-0.6, 0.0, 0.6),
    },
    "separated_six": {
        "populations": (11, 11, 11, 11, 10, 10),
        "intercepts": (-6.0, -3.6, -1.2, 1.2, 3.6, 6.0),
        "slopes": (-0.6, 0.6, -0.6, 0.6, -0.6, 0.6),
    },
    "no_regime": {
        "populations": (16, 16, 16, 16),
        "intercepts": (-6.0, -2.0, 2.0, 6.0),
        "slopes": (-0.6, 0.6, -0.6, 0.6),
    },
}


def partition(seed: int, case: str) -> np.ndarray:
    populations = CASES[case]["populations"]
    groups = np.repeat(np.arange(len(populations)), populations)
    return np.random.default_rng(seed * 100).permutation(groups)


def generate(seed: int, case: str, split: int, rows: int, *, shuffle: bool = False) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed * 100 + split)
    entities = np.resize(np.arange(ENTITY_COUNT), rows)
    rng.shuffle(entities)
    groups = partition(seed, case)[entities]
    x = rng.uniform(-1, 1, rows)
    process = CASES[case]
    y = np.asarray(process["intercepts"])[groups] + np.asarray(process["slopes"])[groups] * x
    y += rng.normal(0, 0.1, rows)
    if shuffle or case == "no_regime":
        entities = entities[rng.permutation(rows)]
    observations = [
        {"entity": f"entity-{identity:04d}", "x": float(value), "y": float(target)}
        for identity, value, target in zip(entities, x, y, strict=True)
    ]
    return observations, groups


def records(observations):
    yield from observations


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        entity=rf.Group(n_groups=(2, 8), p_unavailable=0, mask=rf.Mask(rate=0.1, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=0.003)
    return model


def predict(model, rows) -> np.ndarray:
    inputs = pa.Table.from_pylist([{"entity": row["entity"], "x": row["x"]} for row in rows])
    batch_size = model.batch_size
    try:
        output = model.predict(inputs, batch_size=256)["predictions"].combine_chunks().field("y").field("content")
    finally:
        model.batch_size = batch_size
    return output.to_numpy(zero_copy_only=False)


def reassignment(model, train, validation, optimizers) -> dict:
    """Choose membership from conditional training errors, never generating labels."""
    started = time.perf_counter()
    embedder = model.nodes["/entity"].embedder
    vocabulary = rf.Group.vocabulary(model, "/entity")
    index = {token: ordinal for ordinal, token in enumerate(vocabulary)}
    identities = np.asarray([index[row["entity"]] for row in train])
    targets = np.asarray([row["y"] for row in train])
    validation_targets = np.asarray([row["y"] for row in validation])
    weight = embedder.embeddings["group"].weight
    saved = weight.detach().clone()
    row_mass = saved[: len(vocabulary)].logsumexp(-1, keepdim=True)
    groups = (~embedder.retired).nonzero(as_tuple=True)[0].tolist()
    before = float(np.mean((predict(model, validation) - validation_targets) ** 2))
    errors = np.full((len(vocabulary), embedder.size), np.inf)
    tick = time.perf_counter()
    try:
        for group_id in groups:
            forced = torch.full_like(saved[: len(vocabulary)], math.log(1e-8))
            forced[:, group_id] = 0
            with torch.no_grad():
                weight[: len(vocabulary)].copy_(row_mass + forced)
            residual = (predict(model, train) - targets) ** 2
            errors[:, group_id] = np.bincount(identities, weights=residual, minlength=len(vocabulary))
    finally:
        with torch.no_grad():
            weight.copy_(saved)
    forward_seconds = time.perf_counter() - tick
    selected = errors.argmin(-1)
    forced = torch.full_like(saved[: len(vocabulary)], math.log(1e-8))
    forced[torch.arange(len(vocabulary), device=weight.device), torch.as_tensor(selected, device=weight.device)] = 0
    with torch.no_grad():
        weight[: len(vocabulary)].copy_(row_mass + forced)
    after = float(np.mean((predict(model, validation) - validation_targets) ** 2))
    accepted = after < before
    if accepted:
        if weight.grad is not None:
            weight.grad[: len(vocabulary)].zero_()
        for optimizer in optimizers:
            for value in optimizer.state.get(weight, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                    value[: len(vocabulary)].zero_()
    else:
        with torch.no_grad():
            weight.copy_(saved)
    return {
        "accepted": accepted,
        "before_validation_mse": before,
        "proposed_validation_mse": after,
        "candidate_groups": groups,
        "training_rows_evaluated": len(train) * len(groups),
        "validation_rows_evaluated": 2 * len(validation),
        "counterfactual_forward_seconds": forward_seconds,
        "event_wall_seconds": time.perf_counter() - started,
    }


class PredictiveGrouping(lit.Callback):
    """Proof-local experimental callback; ordinary Model.fit does not do this."""

    def __init__(self, train, validation):
        self.train_rows, self.validation_rows = train, validation
        self.events = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step not in (200, 400) or any(event["step"] == step for event in self.events):
            return
        self.events.append(
            {"step": step, **reassignment(pl_module, self.train_rows, self.validation_rows, trainer.optimizers)}
        )


def adjusted_rand(actual, predicted) -> float:
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)
    joint = float(np.sum(cells * (cells - 1) / 2))
    left_pairs = float(np.sum(cells.sum(1) * (cells.sum(1) - 1) / 2))
    right_pairs = float(np.sum(cells.sum(0) * (cells.sum(0) - 1) / 2))
    pairs = len(actual) * (len(actual) - 1) / 2
    expected = left_pairs * right_pairs / pairs
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


def measure(predicted, rows, training_mean) -> dict:
    target = np.asarray([row["y"] for row in rows])
    rmse = float(np.mean((target - predicted) ** 2) ** 0.5)
    baseline = float(np.mean((target - training_mean) ** 2) ** 0.5)
    return {"rmse": rmse, "baseline_rmse": baseline, "normalized_rmse": rmse / baseline}


def relabel(model, rows, seed) -> dict:
    """Group coordinates are arbitrary names when every corresponding tensor moves together."""
    node = model.nodes["/entity"]
    embedder = node.embedder
    tensors = (
        (embedder.embeddings["group"].weight, 1),
        (embedder.embeddings["content"].weight, 1),
        (node.decoder.linears["group"].weight, 0),
        (embedder.usage_ema, 0),
        (embedder.committed, 0),
        (embedder.retired, 0),
        (embedder.statistics[:-2], 0),
    )
    saved = [value.detach().clone() for value, _ in tensors]
    permutation = np.random.default_rng(seed + 991).permutation(embedder.size)
    inverse = np.argsort(permutation)
    before = predict(model, rows)
    assignments = rf.Group.assignments(model, "/entity")
    try:
        with torch.no_grad():
            for (value, axis), original in zip(tensors, saved, strict=True):
                value.copy_(original.index_select(axis, torch.as_tensor(permutation, device=value.device)))
        after = predict(model, rows)
        moved = rf.Group.assignments(model, "/entity")
        consistent = all(moved[token]["group"] == int(inverse[entry["group"]]) for token, entry in assignments.items())
    finally:
        with torch.no_grad():
            for (value, _), original in zip(tensors, saved, strict=True):
                value.copy_(original)
    return {
        "max_prediction_delta": float(np.max(np.abs(after - before))),
        "assignment_labels_move_consistently": consistent,
    }


def run(seed, steps, accelerator):
    metrics, checks = {}, {}
    budget = BUDGET if steps is None else min(steps, BUDGET)
    for case in CASES:
        lit.seed_everything(seed, workers=True)
        train, _ = generate(seed, case, 1, TRAIN_ROWS)
        validation, _ = generate(seed, case, 2, VALIDATION_ROWS)
        model = build()
        data = rf.SyntheticDataModule(
            model=model, train=partial(records, train), validate=partial(records, validation), seed=seed
        )
        callback = PredictiveGrouping(train, validation)
        trainer = lit.Trainer(
            accelerator=accelerator,
            devices=1,
            max_steps=budget,
            max_epochs=-1,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        started = time.perf_counter()
        trainer.fit(model, datamodule=data)
        fit_seconds = time.perf_counter() - started
        model.eval()
        final_event_seconds = 0.0
        if trainer.global_step == BUDGET:
            callback.events.append(
                {"step": trainer.global_step, **reassignment(model, train, validation, trainer.optimizers)}
            )
            final_event_seconds = callback.events[-1]["event_wall_seconds"]
        intervention_seconds = sum(event["event_wall_seconds"] for event in callback.events)
        assignments = rf.Group.assignments(model, "/entity")
        membership = np.asarray(
            [assignments.get(f"entity-{identity:04d}", {"group": -1})["group"] for identity in range(ENTITY_COUNT)]
        )
        vocabulary = rf.Group.vocabulary(model, "/entity")
        frozen = {name: value.detach().clone() for name, value in (*model.named_parameters(), *model.named_buffers())}
        test, regimes = generate(seed, case, 3, TEST_ROWS)
        mean = float(np.mean([row["y"] for row in train]))
        prediction = predict(model, test)
        measured = measure(prediction, test, mean)
        expected = partition(seed, case)
        measured.update(
            adjusted_rand=adjusted_rand(expected, membership),
            occupied_groups=int(len(np.unique(membership[membership >= 0]))),
            generating_groups=len(CASES[case]["populations"]),
            assignments=membership.tolist(),
            status=rf.Group.status(model, "/entity"),
            optimizer_updates=trainer.global_step,
            noise_rmse=0.1,
            per_regime={
                str(group_id): {
                    "test_rows": int((regimes == group_id).sum()),
                    **measure(
                        prediction[regimes == group_id],
                        [row for row, keep in zip(test, regimes == group_id, strict=True) if keep],
                        mean,
                    ),
                }
                for group_id in range(len(CASES[case]["populations"]))
            },
            events=callback.events,
            compute={
                "fit_wall_seconds_including_200_400_events": fit_seconds,
                "final_event_wall_seconds": final_event_seconds,
                "all_intervention_wall_seconds": intervention_seconds,
                "measured_fit_plus_final_event_seconds": fit_seconds + final_event_seconds,
                "intervention_wall_percent": 100 * intervention_seconds / (fit_seconds + final_event_seconds),
                "extra_training_forward_rows": sum(event["training_rows_evaluated"] for event in callback.events),
                "extra_validation_forward_rows": sum(event["validation_rows_evaluated"] for event in callback.events),
                "training_forward_rows": trainer.global_step * BATCH_SIZE,
            },
        )
        if case != "no_regime":
            corrupted, _ = generate(seed, case, 3, TEST_ROWS, shuffle=True)
            measured["shuffled"] = measure(predict(model, corrupted), corrupted, mean)
            checks[f"{case}: ARI at least 0.80"] = measured["adjusted_rand"] >= 0.80
            checks[f"{case}: normalized RMSE at most 0.35"] = measured["normalized_rmse"] <= 0.35
            checks[f"{case}: occupied count equals generating count"] = (
                measured["occupied_groups"] == measured["generating_groups"]
            )
            checks[f"{case}: shuffled normalized RMSE at least 0.85"] = measured["shuffled"]["normalized_rmse"] >= 0.85
            checks[f"{case}: shuffling raises normalized RMSE by at least 0.40"] = (
                measured["shuffled"]["normalized_rmse"] - measured["normalized_rmse"] >= 0.40
            )
        else:
            checks["no_regime: ARI at most 0.15"] = measured["adjusted_rand"] <= 0.15
            checks["no_regime: normalized RMSE at least 0.85"] = measured["normalized_rmse"] >= 0.85
        measured["group_relabeling"] = relabel(model, test[:256], seed)
        unseen = [{**row, "entity": f"unseen-{index:04d}"} for index, row in enumerate(test[:256])]
        unseen_prediction = predict(model, unseen)
        measured["unseen_identity"] = measure(unseen_prediction, unseen, mean)
        frozen_preserved = all(
            torch.equal(value, frozen[name]) for name, value in (*model.named_parameters(), *model.named_buffers())
        )
        checks[f"{case}: finite prediction and partition metrics"] = bool(
            np.isfinite([measured["rmse"], measured["normalized_rmse"], measured["adjusted_rand"]]).all()
        )
        checks[f"{case}: all known identities covered"] = len(assignments) == ENTITY_COUNT
        checks[f"{case}: evaluation preserves parameters and buffers"] = frozen_preserved
        checks[f"{case}: evaluation preserves vocabulary and assignments"] = vocabulary == rf.Group.vocabulary(
            model, "/entity"
        ) and assignments == rf.Group.assignments(model, "/entity")
        checks[f"{case}: group relabeling preserves predictions"] = (
            measured["group_relabeling"]["max_prediction_delta"] <= 1e-5 * measured["baseline_rmse"]
        )
        checks[f"{case}: group relabeling moves assignment labels consistently"] = measured["group_relabeling"][
            "assignment_labels_move_consistently"
        ]
        checks[f"{case}: unseen-identity predictions finite"] = bool(np.isfinite(unseen_prediction).all())
        checks[f"{case}: declared optimizer budget respected"] = trainer.global_step == budget
        metrics[case] = measured
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7901)

# %% [markdown]
# </details>
