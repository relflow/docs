# %% [markdown]
# ---
# title: Can one number representation serve additive and multiplicative structure?
# categories:
# - Representation
# proof-id: P079
# description: Compare raw, log, rank, and combined input lanes with raw, log, and rank target spaces on
#   an additive task and multiplicative tasks with moderate and very heavy tails at a fixed update budget.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Heavy tails, raw lane only
#   metric:
#   - heavy
#   - raw_log
#   - normalized_mae
#   format: error
# - label: Heavy tails, both lanes
#   metric:
#   - heavy
#   - both_log
#   - normalized_mae
#   format: error
# - label: Additive, both lanes
#   metric:
#   - additive
#   - both_raw
#   - normalized_mae
#   format: error
# ---
#
# ## Example
#
# ```yaml
# price: 18.4
# rate: 0.62
# ltv: 15.1
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-number-lanes
# //| fig-cap: "Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# //| fig-alt: "Model tree with customer, price, rate, ltv. Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# #tree(node("customer", kind: "root", children: (
#   node("price", type: "Number", detail: "raw, log, or both lanes"),
#   node("rate", type: "Number", detail: "raw, log, or both lanes"),
#   node("ltv", kind: "target", type: "Number / Quantile", detail: "raw, log, or rank space"),
# )))
# ```
#
# ## Comparison
#
# Hold the data, model width, optimizer, and update budget fixed. Change only how
# numbers are represented: which standardized lanes the inputs embed, and which
# space the target trains in. Score every arm in source units against the known
# conditional median, on the bulk of the distribution and on its tails, for an
# additive task and for multiplicative tasks with moderate and very heavy tails.
#
# ## Result
#
# {{< proof P079 status >}}
#
# A number field that embeds both a raw lane and a log lane is meant to stay
# close to the better single lane on every task without a per-field decision.
# With moderate tails the lanes are interchangeable: Fourier features over a
# standardized value resolve lognormal inputs whose maximum is a few hundred
# times the median. Only when the maximum reaches ten thousand times the median
# does the raw lane degrade, because standardization is then set by a handful of
# extreme rows. The target space matters far more than the input lane: a
# raw-unit target must grow a readout that spans the tail and lags a log-unit
# target even with three times the budget. A monotone target transform leaves
# the median-optimal MAE answer unchanged, so these are claims about
# optimization at a budget, not about the estimand. A Quantile target with the
# same objective fails on heavy tails because its inverse mapping continues
# linearly past the learned support, so a slightly high rank becomes an
# enormous source value.
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
# ### Synthetic processes and oracles
#
# The multiplicative tasks draw `price = exp(3 + 1.2 z1)` and
# `rate = exp(s z2)` from independent standard normals, then
# `ltv = price * sqrt(rate) * exp(0.3 z3)`. Prices have a median near 20 and a
# long tail; rates have a median of 1. The heavy task uses `s = 2.5`, where the
# largest of 8,192 training rates is about ten thousand times the median, the
# regime of RelBench's study-adverse count. The moderate task uses `s = 1.5`,
# where that ratio is a few hundred. The noise is symmetric in log units, so
# the conditional median is exactly `price * sqrt(rate)`, and the evaluator
# computes it from the visible inputs without any latent variable. The additive
# task draws `left` and `right` uniformly on [-50, 50] and sets
# `total = left + right + 2 z`; its conditional median is `left + right`.
#
# Oracle MAE is the test-set error of the conditional median. Baseline MAE is
# the error of the training-median constant. The multiplicative bulk is the set
# of test rows whose price and rate both fall between the 10th and 90th training
# percentiles; the tail is any row above the 95th percentile of either input.
# Every arm also records how much of its absolute error comes from predictions
# outside the training target range.
#
# ### Arms and budget
#
# Inputs embed as `Number(lanes=("raw",))`, `Number(lanes=("log",))`,
# `Number` with both lanes, or `Quantile` ranks. Targets train as a raw-unit
# `Number`, a `Number(log=True)`, or a `Quantile` with the MAE objective. The
# heavy task compares the four input choices with a log target, then the three
# target spaces with both input lanes; two extra heavy arms give the raw-lane
# inputs and the raw-unit target three times the budget. The moderate task
# compares the four input choices with a log target. The additive task compares
# the four input choices with a raw target.
#
# Every arm uses the `xs` preset, 8,192 training rows, 1,024 validation rows,
# batch size 128, learning rate 0.002, and 500 AdamW updates (1,500 for the
# long arms), scored once on 4,096 fresh rows. A shuffled-input control pairs
# each test target with another row's inputs. Lane weight norms of the first
# projection are recorded for the combined arms as a diagnostic.
#
# ### Gates
#
# The first full run used `s = 1.5` only and gated a bulk blur of the raw lane
# that did not occur: every input lane scored within a few percent of the
# others, and the raw lane with three times the budget was the best arm. A
# recorded sweep over `s` in 1.5, 2.0, 2.5, and 3.0 located the degradation at
# `s >= 2.5`, so the proof now carries both regimes and gates the contract the
# data support. That run remains in the evidence with its own code fingerprint.
#
# Gates remain provisional until ten-seed calibration. The combined lanes must
# reach at most 1.5 times the oracle MAE on the heavy and additive tasks and
# stay within 10 percent of the best single input choice on all three tasks. On
# the heavy task the raw lane must score at least 25 percent above the log lane,
# the raw-unit target must lag the log target by at least 25 percent, and the
# raw-unit target with three times the budget must still trail the log target.
# Every arm must produce finite predictions and lose the relationship under
# shuffled inputs. The long raw-lane arm and the beyond-support error shares are
# diagnostics without gates.
#
# ### Remaining work
#
# The tasks are two-input and noise-free apart from symmetric log or additive
# noise. They do not cover bounded or bimodal bulks, where a rank lane may add
# information beyond the log lane, mixed-sign heavy tails, or bf16 precision.
# The combined lanes trail the log lane by a few percent on the heavy task, so
# the first projection does not isolate the informative lane completely at this
# budget; whether a longer budget closes that gap is not measured here.
#
# ### Complete experiment code
#

# %%
"""P079: compare numeric input lanes and target spaces on additive and multiplicative structure."""

from collections.abc import Callable, Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P079"
BUDGET = 500
LONG_BUDGET = 1500
BATCH_SIZE = 128
TRAIN_ROWS = 8192
TEST_ROWS = 4096
PRICE_SCALE = 1.2
HEAVY_SCALE = 2.5
MODERATE_SCALE = 1.5
NOISE = 0.3
SPAN = 50.0
ADDITIVE_NOISE = 2.0
INPUTS = {"heavy": ("price", "rate"), "moderate": ("price", "rate"), "additive": ("left", "right")}
TARGETS = {"heavy": "ltv", "moderate": "ltv", "additive": "total"}
ARMS = (
    ("heavy", "raw", "log", BUDGET),
    ("heavy", "log", "log", BUDGET),
    ("heavy", "both", "log", BUDGET),
    ("heavy", "rank", "log", BUDGET),
    ("heavy", "both", "raw", BUDGET),
    ("heavy", "both", "rank", BUDGET),
    ("heavy", "raw", "log", LONG_BUDGET),
    ("heavy", "both", "raw", LONG_BUDGET),
    ("moderate", "raw", "log", BUDGET),
    ("moderate", "log", "log", BUDGET),
    ("moderate", "both", "log", BUDGET),
    ("moderate", "rank", "log", BUDGET),
    ("additive", "raw", "raw", BUDGET),
    ("additive", "log", "raw", BUDGET),
    ("additive", "both", "raw", BUDGET),
    ("additive", "rank", "raw", BUDGET),
)


def multiplicative(*, rows: int, seed: int, scale: float = HEAVY_SCALE) -> Iterator[dict]:
    """Lognormal prices and rates multiply under symmetric log noise; ``scale`` sets the rate tail."""
    rng = np.random.default_rng(seed)
    price = np.exp(3.0 + PRICE_SCALE * rng.standard_normal(rows))
    rate = np.exp(scale * rng.standard_normal(rows))
    ltv = price * np.sqrt(rate) * np.exp(NOISE * rng.standard_normal(rows))
    for index in range(rows):
        yield {"price": float(price[index]), "rate": float(rate[index]), "ltv": float(ltv[index])}


def additive(*, rows: int, seed: int) -> Iterator[dict]:
    """Two uniform terms add under symmetric noise; nothing is heavy-tailed."""
    rng = np.random.default_rng(seed)
    left = rng.uniform(-SPAN, SPAN, rows)
    right = rng.uniform(-SPAN, SPAN, rows)
    total = left + right + ADDITIVE_NOISE * rng.standard_normal(rows)
    for index in range(rows):
        yield {"left": float(left[index]), "right": float(right[index]), "total": float(total[index])}


RECORDS: dict[str, Callable[..., Iterator[dict]]] = {
    "heavy": multiplicative,
    "moderate": partial(multiplicative, scale=MODERATE_SCALE),
    "additive": additive,
}


def median(task: str, rows: list[dict]) -> np.ndarray:
    """Conditional median of the target given the visible inputs."""
    if task == "additive":
        return np.asarray([row["left"] + row["right"] for row in rows])
    return np.asarray([row["price"] * np.sqrt(row["rate"]) for row in rows])


def build(task: str, inputs: str, target: str) -> rf.Model:
    if inputs == "rank":
        fields: dict[str, object] = {name: rf.Quantile for name in INPUTS[task]}
    else:
        lanes = ("raw", "log") if inputs == "both" else (inputs,)
        fields = {name: rf.Number(lanes=lanes) for name in INPUTS[task]}
    if target == "rank":
        fields[TARGETS[task]] = rf.Quantile(objective="mae", mask=True)
    else:
        fields[TARGETS[task]] = rf.Number(log=target == "log", mask=True)
    return rf.Model.xs(fields=fields)


def fit(model: rf.Model, train, validate, seed: int, steps: int | None, accelerator: str, budget: int) -> int:
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=BATCH_SIZE,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget if steps is None else min(steps, budget),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return model.trainer.global_step


def predict(model: rf.Model, task: str, rows: list[dict]) -> np.ndarray:
    visible = [{name: row[name] for name in INPUTS[task]} for row in rows]
    predictions = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row[TARGETS[task]]["content"] for row in predictions], dtype=np.float64)


def lane_norms(model: rf.Model, task: str) -> dict[str, float]:
    """Weight norms of the first projection per lane: which lane the model reads."""
    norms = {}
    for name in INPUTS[task]:
        embedder = model.nodes[rf.Address(f"/{name}")].embedder
        bands = 2 * embedder.weights.numel()
        fourier, scalar = embedder.linear.weight.detach(), embedder.scalar.weight.detach()
        for index, lane in enumerate(embedder.lanes):
            norms[f"{name}_{lane}_fourier"] = float(fourier[:, index * bands : (index + 1) * bands].norm())
            norms[f"{name}_{lane}_scalar"] = float(scalar[:, index].norm())
    return norms


def evaluate(output: np.ndarray, control: np.ndarray, reference: dict, bulk: np.ndarray | None) -> dict:
    target, oracle = reference["target"], reference["oracle"]
    error = np.abs(output - target)
    mae = float(error.mean())
    beyond = (output < reference["support"][0]) | (output > reference["support"][1])
    result = {
        "mae": mae,
        "oracle_ratio": mae / reference["oracle_mae"],
        "normalized_mae": mae / reference["baseline_mae"],
        "median_distance": float(np.mean(np.abs(output - oracle))),
        "shuffled_normalized_mae": float(np.mean(np.abs(control - target))) / reference["baseline_mae"],
        "beyond_support_rows": int(beyond.sum()),
        "beyond_support_error_share": float(error[beyond].sum() / error.sum()) if error.sum() else 0.0,
        "finite": bool(np.isfinite(output).all() and np.isfinite(control).all()),
    }
    if bulk is not None:
        tail = reference["tail"]
        irreducible = np.abs(target - oracle)
        result["bulk_oracle_ratio"] = float(error[bulk].mean() / irreducible[bulk].mean())
        result["tail_oracle_ratio"] = float(error[tail].mean() / irreducible[tail].mean())
    return result


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    stream = seed * 100
    metrics: dict = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "budget": BUDGET, "long_budget": LONG_BUDGET}
    for offset, task in ((0, "heavy"), (20, "moderate"), (10, "additive")):
        records, name = RECORDS[task], TARGETS[task]
        training = list(records(rows=TRAIN_ROWS, seed=stream + offset + 1))
        test = list(records(rows=TEST_ROWS, seed=stream + offset + 3))
        target = np.asarray([row[name] for row in test])
        oracle = median(task, test)
        observed = np.asarray([row[name] for row in training])
        constant = float(np.median(observed))
        reference = {
            "target": target,
            "oracle": oracle,
            "oracle_mae": float(np.mean(np.abs(target - oracle))),
            "baseline_mae": float(np.mean(np.abs(target - constant))),
            "support": (float(observed.min()), float(observed.max())),
        }
        bulk = None
        if task != "additive":
            columns = {key: np.asarray([row[key] for row in training]) for key in INPUTS[task]}
            low = {key: np.percentile(values, 10) for key, values in columns.items()}
            high = {key: np.percentile(values, 90) for key, values in columns.items()}
            extreme = {key: np.percentile(values, 95) for key, values in columns.items()}
            bulk = np.all([[low[key] <= row[key] <= high[key] for key in INPUTS[task]] for row in test], axis=1)
            reference["tail"] = np.any([[row[key] > extreme[key] for key in INPUTS[task]] for row in test], axis=1)
        permutation = np.random.default_rng(stream + offset + 4).permutation(TEST_ROWS)
        shuffled = [{**test[index], name: row[name]} for row, index in zip(test, permutation, strict=True)]
        metrics[task] = {
            "oracle_mae": reference["oracle_mae"],
            "baseline_mae": reference["baseline_mae"],
            "training_median": constant,
            "training_support": list(reference["support"]),
            "bulk_rows": None if bulk is None else int(bulk.sum()),
            "tail_rows": None if bulk is None else int(reference["tail"].sum()),
        }
        for arm_task, inputs, space, budget in ARMS:
            if arm_task != task:
                continue
            lit.seed_everything(seed, workers=True)
            model = build(task, inputs, space)
            updates = fit(
                model,
                partial(records, rows=TRAIN_ROWS, seed=stream + offset + 1),
                partial(records, rows=1024, seed=stream + offset + 2),
                seed,
                steps,
                accelerator,
                budget,
            )
            result = evaluate(predict(model, task, test), predict(model, task, shuffled), reference, bulk)
            result["steps"] = updates
            if inputs != "rank":
                result["lane_norms"] = lane_norms(model, task)
            metrics[task][f"{inputs}_{space}" + ("_long" if budget != BUDGET else "")] = result
    heavy, moderate, total = metrics["heavy"], metrics["moderate"], metrics["additive"]
    arms = [value for task in TARGETS for value in metrics[task].values() if isinstance(value, dict)]
    single = ("raw", "log", "rank")
    checks = {
        "Heavy tails: both lanes with a log target reach 1.5 times the oracle MAE": heavy["both_log"]["oracle_ratio"]
        <= 1.5,
        "Additive: both lanes reach 1.5 times the oracle MAE": total["both_raw"]["oracle_ratio"] <= 1.5,
        "Heavy tails: both lanes stay within 10 percent of the best single input choice": (
            heavy["both_log"]["mae"] <= 1.10 * min(heavy[f"{lane}_log"]["mae"] for lane in single)
        ),
        "Moderate tails: both lanes stay within 10 percent of the best single input choice": (
            moderate["both_log"]["mae"] <= 1.10 * min(moderate[f"{lane}_log"]["mae"] for lane in single)
        ),
        "Additive: both lanes stay within 10 percent of the best single input choice": (
            total["both_raw"]["mae"] <= 1.10 * min(total[f"{lane}_raw"]["mae"] for lane in single)
        ),
        "Heavy tails: the raw lane scores at least 25 percent above the log lane": (
            heavy["raw_log"]["mae"] >= 1.25 * heavy["log_log"]["mae"]
        ),
        "Heavy tails: a raw-unit target lags a log target by at least 25 percent": (
            heavy["both_raw"]["mae"] >= 1.25 * heavy["both_log"]["mae"]
        ),
        "Heavy tails: three times the budget does not close the raw-target gap": (
            heavy["both_raw_long"]["mae"] > heavy["both_log"]["mae"]
        ),
        "Shuffled inputs lose the relationship in every arm": all(
            arm["shuffled_normalized_mae"] >= 0.90 for arm in arms
        ),
        "Every arm produces finite predictions": all(arm["finite"] for arm in arms),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7901)

# %% [markdown]
# </details>
