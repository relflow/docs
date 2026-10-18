# %% [markdown]
# ---
# title: Can shared identities learn context-dependent labels?
# categories:
# - Grouping
# proof-id: P080
# description: Test predictive reassignment on categorical outcomes whose label marginals do not reveal the hidden groups.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Predictive reassignment
#   metric: [predictive, test, accuracy]
#   format: number
# - label: No persistent groups
#   metric: [no_regime, test, accuracy]
#   format: number
# ---
#
# ## Example
#
# ```yaml
# merchant_id: merchant-0017
# channel: channel-2
# amount: 27.30
# outcome: class-4
# ```
#
# The outcome is hidden. A merchant's hidden group changes the rule connecting
# channel to outcome; the amount is an independent nuisance feature.
#
# ```{typst}
# //| label: fig-proof-group-predictive-labels
# //| fig-cap: "Merchant identity and visible context predict a hidden categorical outcome."
# //| fig-alt: "Payment tree with merchant_id as Group, channel as Category, amount as Number, and outcome as a hidden Category target. Merchant identities are masked for reconstruction on ten percent of training rows."
# #tree(node("payment", kind: "root", children: (
#   node("merchant_id", type: "Group", detail: "10% training reconstruction mask"),
#   node("channel", type: "Category"),
#   node("amount", type: "Number"),
#   node("outcome", type: "Category", kind: "target"),
# )))
# ```
#
# ## Comparison
#
# Compare native training, hardening existing assignments, and predictive
# reassignment. Shuffle test identities and train a separate no-regime control.
#
# ## Result
#
# {{< proof P080 status >}}
#
# This provisional experiment tests an explicit experimental callback, not the
# default behavior of Model.fit. A passing prediction score alone does not
# establish meaningful groups: partition recovery must pass independently.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P080 evidence >}}
#
# ### Run this experiment
#
# {{< proof P080 script >}}
#
# ### Process and information boundary
#
# Fifty opaque identities are randomly divided into five equally sized groups.
# For each independent observation, the clean outcome is
# `(hidden_group + channel) % 5`. Each identity sees all five channels equally;
# every clean label therefore has the same marginal frequency for every ID.
# An independent 2% flip chooses a different label, giving Bayes accuracy 98%.
# Amount is independent lognormal noise. Hidden groups are never schema fields.
#
# Train, validation, and test contain 5,000, 2,000, and 2,500 observations from
# independent random streams. The same known identities and channel values
# recur. This tests prediction on new observations of known identities, not
# unseen-ID or unseen-channel generalization. Repeated identity/channel cells
# are deliberate; fresh amounts and label noise differ across splits.
#
# The no-regime control permutes identities within each split, preserving every
# other field and the exact identity-frequency distribution. Its outcomes have
# no persistent identity-conditioned rule. A second control shuffles only the
# stable model's test IDs. Both have chance accuracy 20% in expectation.
#
# ### Frozen method and gates
#
# Each of four paired arms receives the same xs schema, 600 AdamW updates,
# batch 128, and learning rate .003. Group bounds are `(2, 8)`; the true count
# five is not supplied. At updates 200 and 400, and after update 600, the
# predictive callback evaluates each available group on training observations.
# Each ID chooses the group minimizing its summed true-label negative log
# probability. Apply the proposal only when independent validation NLL falls.
# No test labels or generating partitions participate in this decision.
#
# The hardening control keeps each ID's existing group at the same three
# events. Both interventions replace rows with the same sharp logits and clear
# replaced-row Adam moments while retaining the scalar optimizer step. Native
# training has no intervention. A hardening event is unconditional; a rejected
# predictive proposal leaves the preceding rows intact. Extra inference and
# validation work is recorded separately from optimizer updates.
#
# The categorical scorer uses all observed class probabilities exported by
# `Category(topk=[8])`; the ceiling is clipped to the five discovered labels.
# It scores the actual target probability, not confidence in a wrong top class.
# True labels are supplied only to the scorer, never to prediction inputs.
#
# The predictive arm must achieve test ARI at least .80 and accuracy at least
# .90, and occupy exactly five groups. The no-regime arm must have ARI at most
# .15 and accuracy at most .30. Capacity is a separate gate: useful prediction
# or high ARI does not establish the correct number of representations.
# Shuffling stable test IDs must give accuracy at most .30 and reduce accuracy
# by at least .50. Require all IDs/classes exposed, finite scores, and frozen
# evaluation resources. Native/hardening differences are diagnostics: if they
# also pass, this does not establish a specific benefit from reassignment.
# Three fresh seeds are 8001–8003. These gates are fixed before the panel; failed
# gates remain failures. Ten-seed calibration and general multi-target runtime
# integration are separate work. This proof does not repair Group-only masked
# identity reconstruction or establish P065's coverage-quality claim.
#
# ### Complete experiment code

# %%
"""P080: evaluate categorical predictive reassignment with uniform ID label marginals."""

import math
import time
from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P080"
BUDGET = 600
BATCH_SIZE = 128
ENTITY_COUNT = 50
CLASS_COUNT = 5
TRAIN_ROWS = 5000
VALIDATION_ROWS = 2000
TEST_ROWS = 2500


def partition(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.arange(ENTITY_COUNT) % CLASS_COUNT)


def records(*, rows: int, seed: int, partition_seed: int, no_regime: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    cells = np.resize(np.arange(ENTITY_COUNT * CLASS_COUNT), rows)
    rng.shuffle(cells)
    entities, channels = cells // CLASS_COUNT, cells % CLASS_COUNT
    labels = (partition(partition_seed)[entities] + channels) % CLASS_COUNT
    labels = (labels + (rng.random(rows) < 0.02) * rng.integers(1, CLASS_COUNT, rows)) % CLASS_COUNT
    amounts = rng.lognormal(3.0, 0.5, rows)
    if no_regime:
        entities = entities[rng.permutation(rows)]
    for entity, channel, amount, label in zip(entities, channels, amounts, labels, strict=True):
        yield {
            "merchant_id": f"merchant-{entity:04d}",
            "channel": f"channel-{channel}",
            "amount": float(amount),
            "outcome": f"class-{label}",
        }


def adjusted_rand(actual: np.ndarray, predicted: np.ndarray) -> float:
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)

    def pairs(values):
        return float(np.sum(values * (values - 1) / 2))

    joint = pairs(cells)
    left_pairs, right_pairs = pairs(cells.sum(1)), pairs(cells.sum(0))
    expected = left_pairs * right_pairs / (len(actual) * (len(actual) - 1) / 2)
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        merchant_id=rf.Group(n_groups=(2, 8), p_unavailable=0.0, mask=rf.Mask(rate=0.1, reconstruct=True)),
        channel=rf.Category(p_unavailable=0.0),
        amount=rf.Number,
        outcome=rf.Category(mask=True, topk=[8]),
    )
    model.optimizer = rf.adamw(learning_rate=0.003)
    return model


def losses(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Read the complete public categorical distribution; omit every target input."""
    visible = pa.Table.from_pylist([{key: row[key] for key in ("merchant_id", "channel", "amount")} for row in rows])
    previous_size = model.batch_size
    try:
        output = model.predict(visible, batch_size=256)["predictions"].to_pylist()
    finally:
        model.batch_size = previous_size
    vocabulary = set(rf.Category.vocabulary(model, "/outcome"))
    values, correct = [], []
    for row, prediction in zip(rows, output, strict=True):
        content = prediction["outcome"]["content"]
        probabilities = {item["value"]: item["probability"] for item in content["topk"]}
        if set(probabilities) != vocabulary or row["outcome"] not in probabilities:
            raise ValueError("P080 scorer requires all observed target classes in the public distribution")
        values.append(-math.log(max(probabilities[row["outcome"]], 1e-30)))
        correct.append(content["value"] == row["outcome"])
    return np.asarray(values), np.asarray(correct, dtype=bool)


def reset_rows(weight: torch.Tensor, size: int, optimizers: list) -> None:
    if weight.grad is not None:
        weight.grad[:size].zero_()
    for optimizer in optimizers:
        for value in optimizer.state.get(weight, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                value[:size].zero_()


def intervene(model: rf.Model, train: list[dict], validate: list[dict], optimizers: list, *, search: bool) -> dict:
    started = time.perf_counter()
    embedder = model.nodes["/merchant_id"].embedder
    vocabulary = list(rf.Group.vocabulary(model, "/merchant_id"))
    weight = embedder.embeddings["group"].weight
    original = weight.detach().clone()
    indices = np.asarray([int(token.rsplit("-", 1)[1]) for token in vocabulary])
    current = embedder.support(original[: len(vocabulary)]).argmax(-1).cpu().numpy()
    row_mass = original[: len(vocabulary)].logsumexp(-1, keepdim=True)
    available = (~embedder.retired).nonzero(as_tuple=True)[0].tolist()
    before = float(losses(model, validate)[0].mean())
    scoring_seconds = 0.0

    def replace(groups: np.ndarray) -> None:
        forced = torch.full_like(original[: len(vocabulary)], math.log(1e-8))
        forced[torch.arange(len(vocabulary), device=weight.device), torch.as_tensor(groups, device=weight.device)] = 0
        with torch.no_grad():
            weight[: len(vocabulary)].copy_(row_mass + forced)

    if search:
        errors = np.full((ENTITY_COUNT, embedder.size), np.inf)
        entities = np.asarray([int(row["merchant_id"].rsplit("-", 1)[1]) for row in train])
        tick = time.perf_counter()
        try:
            for group_id in available:
                replace(np.full(len(vocabulary), group_id))
                nll, _ = losses(model, train)
                errors[:, group_id] = np.bincount(entities, weights=nll, minlength=ENTITY_COUNT)
        finally:
            with torch.no_grad():
                weight.copy_(original)
        scoring_seconds = time.perf_counter() - tick
        proposed = errors.argmin(-1)[indices]
    else:
        proposed = current
    replace(proposed)
    after = float(losses(model, validate)[0].mean())
    accepted = after < before if search else True
    if accepted:
        reset_rows(weight, len(vocabulary), optimizers)
    else:
        with torch.no_grad():
            weight.copy_(original)
    return {
        "search": search,
        "accepted": accepted,
        "before_validation_nll": before,
        "proposed_validation_nll": after,
        "changed_ids": int(np.sum(proposed != current)),
        "candidate_groups": available,
        "train_rows_scored": len(train) * len(available) if search else 0,
        "validation_rows_scored": 2 * len(validate),
        "scoring_seconds": scoring_seconds,
        "event_seconds": time.perf_counter() - started,
    }


class Reassignment(lit.Callback):
    def __init__(self, train: list[dict], validate: list[dict], *, search: bool):
        self.train = train
        self.validate = validate
        self.search = search
        self.events = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in (200, 400) and not any(event["step"] == step for event in self.events):
            event = intervene(pl_module, self.train, self.validate, trainer.optimizers, search=self.search)
            self.events.append({"step": step, **event})


def measure(model: rf.Model, rows: list[dict], expected: np.ndarray, majority: str) -> dict:
    nll, correct = losses(model, rows)
    mapping = rf.Group.assignments(model, "/merchant_id")
    assigned = np.asarray(
        [mapping.get(f"merchant-{entity:04d}", {}).get("group", -1) for entity in range(ENTITY_COUNT)]
    )
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "majority_accuracy": float(np.mean([row["outcome"] == majority for row in rows])),
        "adjusted_rand": adjusted_rand(expected, assigned),
        "assigned_ids": int(np.sum(assigned >= 0)),
        "occupied_groups": int(len(np.unique(assigned[assigned >= 0]))),
        "assignments": assigned.tolist(),
        "status": rf.Group.status(model, "/merchant_id"),
        "known_target_classes": len(rf.Category.vocabulary(model, "/outcome")),
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    budget = BUDGET if steps is None else min(steps, BUDGET)
    expected = partition(seed * 100)
    for arm in ("native", "harden_existing", "predictive", "no_regime"):
        lit.seed_everything(seed, workers=True)
        model = build()
        options = {"partition_seed": seed * 100, "no_regime": arm == "no_regime"}
        train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1, **options))
        validate = list(records(rows=VALIDATION_ROWS, seed=seed * 100 + 2, **options))
        data = rf.SyntheticDataModule(
            model=model, train=lambda: iter(train), validate=lambda: iter(validate), seed=seed
        )
        intervention = Reassignment(train, validate, search=arm != "harden_existing") if arm != "native" else None
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
            callbacks=[intervention] if intervention is not None else [],
        )
        started = time.perf_counter()
        trainer.fit(model, datamodule=data)
        model.eval()
        if intervention is not None and trainer.global_step == BUDGET:
            event = intervene(model, train, validate, trainer.optimizers, search=intervention.search)
            intervention.events.append({"step": BUDGET, **event})
        events = intervention.events if intervention is not None else []
        majority = max(
            (f"class-{label}" for label in range(CLASS_COUNT)),
            key=lambda label: sum(row["outcome"] == label for row in train),
        )
        before = rf.Group.assignments(model, "/merchant_id")
        vocabulary = rf.Group.vocabulary(model, "/merchant_id")
        exposures = model.nodes["/merchant_id"].embedder.counters["content"].counts.clone()
        target_vocabulary = rf.Category.vocabulary(model, "/outcome")
        test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, **options))
        result = {
            "steps": trainer.global_step,
            "events": events,
            "test": measure(model, test, expected, majority),
            "wall_seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "counterfactual_train_rows_scored": sum(event["train_rows_scored"] for event in events),
            "selection_validation_rows_scored": sum(event["validation_rows_scored"] for event in events),
        }
        if arm == "predictive":
            shuffled = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, partition_seed=seed * 100, no_regime=True))
            result["shuffled"] = measure(model, shuffled, expected, majority)
        metrics[arm] = result
        checks[f"{arm}: finite metrics"] = bool(
            np.isfinite([result["test"]["accuracy"], result["test"]["nll"], result["test"]["adjusted_rand"]]).all()
        )
        checks[f"{arm}: all 50 identities and five target classes exposed"] = (
            result["test"]["assigned_ids"] == ENTITY_COUNT and result["test"]["known_target_classes"] == CLASS_COUNT
        )
        checks[f"{arm}: evaluation resources remain frozen"] = (
            before == rf.Group.assignments(model, "/merchant_id")
            and vocabulary == rf.Group.vocabulary(model, "/merchant_id")
            and torch.equal(exposures, model.nodes["/merchant_id"].embedder.counters["content"].counts)
            and target_vocabulary == rf.Category.vocabulary(model, "/outcome")
        )
    stable, control = metrics["predictive"]["test"], metrics["no_regime"]["test"]
    shuffled = metrics["predictive"]["shuffled"]
    checks.update(
        {
            "Predictive groups recover ARI at least 0.80": stable["adjusted_rand"] >= 0.80,
            "Predictive groups achieve accuracy at least 0.90": stable["accuracy"] >= 0.90,
            "Predictive groups occupy exactly five groups": stable["occupied_groups"] == CLASS_COUNT,
            "No-regime control ARI is at most 0.15": control["adjusted_rand"] <= 0.15,
            "No-regime control accuracy is at most 0.30": control["accuracy"] <= 0.30,
            "Shuffled-ID accuracy is at most 0.30": shuffled["accuracy"] <= 0.30,
            "Shuffling reduces accuracy by at least 0.50": stable["accuracy"] - shuffled["accuracy"] >= 0.50,
        }
    )
    metrics["protocol"] = {
        "budgets_per_arm": BUDGET,
        "observations": [TRAIN_ROWS, VALIDATION_ROWS, TEST_ROWS],
        "noise_probability": 0.02,
        "chance_accuracy": 0.20,
        "bayes_accuracy": 0.98,
        "expected_partition": expected.tolist(),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=8001)

# %% [markdown]
# </details>
