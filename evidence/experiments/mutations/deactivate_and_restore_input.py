# %% [markdown]
# ---
# title: Can an input be disabled and restored?
# categories:
# - Schema mutation
# evidence-id: P049
# description: Train on y = a + b, temporarily disable b, then enable it again.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original model
#   metric:
#   - source
#   - nrmse
#   format: error
# - label: Best prediction using only a
#   metric:
#   - controls
#   - only_a_oracle
#   - nrmse
#   format: error
# evidence-group: Mutations
# evidence-result:
#   accelerator: cpu
#   checks:
#     Inactive checkpoint preserves predictions: true
#     Inactive checkpoint preserves schema and state: true
#     Loaded inactive input can restore its trained function: true
#     Shuffling b destroys useful signal: true
#     Source hidden target values cannot affect predictions: true
#     Source learns both-input relationship below 0.25 nRMSE: true
#     Unchanged checkpoint preserves predictions: true
#     'cycle 1 deactivate: all learned state survives': true
#     'cycle 1 deactivate: b values cannot affect predictions': true
#     'cycle 1 deactivate: input is inactive': true
#     'cycle 1 deactivate: removing b loses information': true
#     'cycle 1 reactivate: all learned state survives': true
#     'cycle 1 reactivate: original schema is restored': true
#     'cycle 1 reactivate: trained predictions are restored': true
#     'cycle 2 deactivate: all learned state survives': true
#     'cycle 2 deactivate: b values cannot affect predictions': true
#     'cycle 2 deactivate: input is inactive': true
#     'cycle 2 deactivate: removing b loses information': true
#     'cycle 2 reactivate: all learned state survives': true
#     'cycle 2 reactivate: original schema is restored': true
#     'cycle 2 reactivate: trained predictions are restored': true
#     'cycle 3 deactivate: all learned state survives': true
#     'cycle 3 deactivate: b values cannot affect predictions': true
#     'cycle 3 deactivate: input is inactive': true
#     'cycle 3 deactivate: removing b loses information': true
#     'cycle 3 reactivate: all learned state survives': true
#     'cycle 3 reactivate: original schema is restored': true
#     'cycle 3 reactivate: trained predictions are restored': true
#     'exceptional override exit: all learned state survives': true
#     'exceptional override exit: original schema is restored': true
#     'exceptional override exit: trained predictions are restored': true
#     'normal override exit: all learned state survives': true
#     'normal override exit: original schema is restored': true
#     'normal override exit: trained predictions are restored': true
#     'override before exception: all learned state survives': true
#     'override before exception: b values cannot affect predictions': true
#     'override before exception: input is inactive': true
#     'override before exception: removing b loses information': true
#     'temporary override: all learned state survives': true
#     'temporary override: b values cannot affect predictions': true
#     'temporary override: input is inactive': true
#     'temporary override: removing b loses information': true
#   duration_seconds: 14.199
#   metrics:
#     controls:
#       only_a_oracle:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.7119104014052033
#         rmse: 0.5857724467092484
#       shuffled_b:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 1.0059826079783953
#         rmse: 0.82774024997993
#       zero_filled_b:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.71187244478633
#         rmse: 0.5857412153331337
#     downstream_interpretable: true
#     edits:
#     - edit: cycle 1 deactivate
#       inactive: true
#       learned_state_preserved: true
#       max_source_prediction_drift: 1.97914895632392
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.9582986834971589
#         rmse: 0.7885050750801961
#     - edit: cycle 1 reactivate
#       inactive: false
#       learned_state_preserved: true
#       max_source_prediction_drift: 0.0
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.013208015100434373
#         rmse: 0.010867788005741579
#     - edit: cycle 2 deactivate
#       inactive: true
#       learned_state_preserved: true
#       max_source_prediction_drift: 1.97914895632392
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.9582986834971589
#         rmse: 0.7885050750801961
#     - edit: cycle 2 reactivate
#       inactive: false
#       learned_state_preserved: true
#       max_source_prediction_drift: 0.0
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.013208015100434373
#         rmse: 0.010867788005741579
#     - edit: cycle 3 deactivate
#       inactive: true
#       learned_state_preserved: true
#       max_source_prediction_drift: 1.97914895632392
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.9582986834971589
#         rmse: 0.7885050750801961
#     - edit: cycle 3 reactivate
#       inactive: false
#       learned_state_preserved: true
#       max_source_prediction_drift: 0.0
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.013208015100434373
#         rmse: 0.010867788005741579
#     - edit: temporary override
#       inactive: true
#       learned_state_preserved: true
#       max_source_prediction_drift: 1.97914895632392
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.9582986834971589
#         rmse: 0.7885050750801961
#     - edit: normal override exit
#       inactive: false
#       learned_state_preserved: true
#       max_source_prediction_drift: 0.0
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.013208015100434373
#         rmse: 0.010867788005741579
#     - edit: override before exception
#       inactive: true
#       learned_state_preserved: true
#       max_source_prediction_drift: 1.97914895632392
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.9582986834971589
#         rmse: 0.7885050750801961
#     - edit: exceptional override exit
#       inactive: false
#       learned_state_preserved: true
#       max_source_prediction_drift: 0.0
#       scores:
#         baseline_rmse: 0.8228176545152625
#         nrmse: 0.013208015100434373
#         rmse: 0.010867788005741579
#     mutation: update /b active=False/True three times; override with normal and exceptional exits
#     optimizer_policy: Fresh AdamW for source; no fitting inside updates or overrides
#     source:
#       baseline_rmse: 0.8228176545152625
#       nrmse: 0.013208015100434373
#       rmse: 0.010867788005741579
#     source_prerequisite_met: true
#     source_steps: 512
#     test_rows: 2048
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c465a35305a939cafa70f21d6093df873085f849b21f97e928d824dde994b5c0
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 8743a4dc08a6
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
#     script_sha256: 27e4d67edf94a9287a669a6ed47da17c7689e53c433b8b41abc1d62e27674431
#     source_sha256: 3ddcb2aae2469e0474161654be249da7d3b5e62ebbac89e5fb94ae350e80ecc2
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 4901
#   started_at: '2026-11-01T00:19:37.701059+00:00'
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
# a: 0.5
# b: 0.75
# y: 1.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mutations-deactivate-input
# //| fig-alt: "Model tree with record, a, b, y. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number", body: [Always active]),
#   node("b", type: "Number", body: [Active → inactive → active]),
#   node("y", type: "Number", kind: "target", body: [Always hidden]),
# )))
# ```
#
# ## Comparison
#
# Check repeated edits, temporary overrides, and a saved inactive model. No training occurs while b is disabled.
#
# ## Result
#
# {{< evidence P049 status >}}
#
# Disabling b removes useful information; enabling it restores the trained predictions. This tests an input field, rather than disabling a learned output decoder.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P049 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P049 script >}}
#
# ### Remaining work
#
# Three GPU seeds support this case. Calibrate numerical gates on ten independent
# seeds. Deactivating output heads and branches that produce context requires
# separate experiments. Training inside an override does not have checkpoint
# rollback semantics and is not covered here.
#
# ### Complete experiment code
#

# %%
"""P049: remove and restore an informative input without further training."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P049"


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "y": a + b}


def prediction(model: rf.Model, rows: list[dict], *, include_b: bool = True, corrupt: bool = False) -> np.ndarray:
    inputs = [{"a": row["a"], **({"b": row["b"]} if include_b else {})} for row in rows]
    if corrupt:
        for row in inputs:
            row["y"] = 1000.0
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["y"]["content"] for row in output], dtype=np.float64)


def errors(actual: np.ndarray, predicted: np.ndarray, baseline: float) -> dict:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}


def equal(first, second) -> bool:
    """Compare tensors and extension-owned state, including normalization."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[name]) for name, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    budget = 512 if steps is None else min(steps, 512)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["y"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["y"] for row in train])) ** 2)))
    tolerance = 1e-6 * float(np.std([row["y"] for row in train]))
    oracle = errors(actual, np.asarray([row["a"] for row in test]), baseline)
    rng = np.random.default_rng(seed + 4)
    order = rng.permutation(len(test))
    shuffled = [{**row, "b": test[index]["b"]} for row, index in zip(test, order, strict=True)]
    changed = [{**row, "b": -row["b"]} for row in test]
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        a=rf.Number,
        b=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.003,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    reference = prediction(model, test)
    source = errors(actual, reference, baseline)
    corruption = errors(actual, prediction(model, shuffled), baseline)
    zero_filled = errors(actual, prediction(model, [{**row, "b": 0.0} for row in test]), baseline)
    schema = deepcopy(model.schema.model_dump())
    state = deepcopy(model.state_dict())
    selected = rf.where("address") == "/b"
    learned = source["nrmse"] < 0.25
    checks = {
        "Source learns both-input relationship below 0.25 nRMSE": learned,
        "Shuffling b destroys useful signal": corruption["nrmse"] > source["nrmse"] + 0.35
        and corruption["nrmse"] >= 0.9 * oracle["nrmse"],
        "Source hidden target values cannot affect predictions": bool(
            np.allclose(reference, prediction(model, test, corrupt=True), rtol=1e-5, atol=tolerance)
        ),
    }
    edits = []

    def observe(label: str, *, inactive: bool) -> np.ndarray:
        """Evaluate the appropriate information-loss or restoration invariant."""
        values = prediction(model, test)
        measured = errors(actual, values, baseline)
        same_state = equal(state, model.state_dict())
        checks[f"{label}: all learned state survives"] = same_state
        if inactive:
            omitted = prediction(model, test, include_b=False)
            flipped = prediction(model, changed)
            checks[f"{label}: input is inactive"] = not model.schema.requests["/b"].active
            checks[f"{label}: b values cannot affect predictions"] = bool(
                np.allclose(values, omitted, rtol=1e-5, atol=tolerance)
                and np.allclose(values, flipped, rtol=1e-5, atol=tolerance)
            )
            checks[f"{label}: removing b loses information"] = (
                measured["nrmse"] > source["nrmse"] + 0.35 and measured["nrmse"] >= 0.9 * oracle["nrmse"]
            )
        else:
            checks[f"{label}: original schema is restored"] = model.schema.model_dump() == schema
            checks[f"{label}: trained predictions are restored"] = bool(
                np.allclose(reference, values, rtol=1e-5, atol=tolerance)
            )
        edits.append(
            {
                "edit": label,
                "inactive": inactive,
                "scores": measured,
                "max_source_prediction_drift": float(np.max(np.abs(values - reference))),
                "learned_state_preserved": same_state,
            }
        )
        return values

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        model.save(checkpoint)
        unchanged = rf.Model.load(checkpoint).to(model.device).eval()
        checks["Unchanged checkpoint preserves predictions"] = bool(
            np.allclose(reference, prediction(unchanged, test), rtol=1e-5, atol=tolerance)
        )
        for cycle in range(3):
            model.update(selected, active=False)
            inactive_values = observe(f"cycle {cycle + 1} deactivate", inactive=True)
            if cycle == 0:
                path = Path(directory) / "inactive.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                checks["Inactive checkpoint preserves schema and state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump() and equal(state, loaded.state_dict())
                )
                checks["Inactive checkpoint preserves predictions"] = bool(
                    np.allclose(inactive_values, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )
                loaded.update(selected, active=True)
                checks["Loaded inactive input can restore its trained function"] = bool(
                    loaded.schema.model_dump() == schema
                    and equal(state, loaded.state_dict())
                    and np.allclose(reference, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )
            model.update(selected, active=True)
            observe(f"cycle {cycle + 1} reactivate", inactive=False)
        with model.override(selected, active=False):
            observe("temporary override", inactive=True)
        observe("normal override exit", inactive=False)
        marker = RuntimeError("intentional override exit")
        try:
            with model.override(selected, active=False):
                observe("override before exception", inactive=True)
                raise marker
        except RuntimeError as error:
            if error is not marker:
                raise
        observe("exceptional override exit", inactive=False)
    return {
        "source": source,
        "source_steps": trainer.global_step,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "controls": {"shuffled_b": corruption, "zero_filled_b": zero_filled, "only_a_oracle": oracle},
        "edits": edits,
        "test_rows": len(test),
        "optimizer_policy": "Fresh AdamW for source; no fitting inside updates or overrides",
        "mutation": "update /b active=False/True three times; override with normal and exceptional exits",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4901)

# %% [markdown]
# </details>
