# %% [markdown]
# ---
# title: Can it learn the month from the day number?
# categories:
# - Calendar reasoning
# evidence-id: P044
# description: Learn month boundaries from day-of-year in non-leap years.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original dates
#   metric:
#   - periodic_accuracy
#   format: percent
# - label: Shuffled dates
#   metric:
#   - permuted_accuracy
#   format: percent
# evidence-group: Temporal
# evidence-result:
#   accelerator: cpu
#   checks:
#     Original dates exceed permutation by at least 0.65 accuracy: true
#     Timestamp permutation accuracy is at most 0.20: true
#     Unseen-date month accuracy reaches 0.90: true
#   duration_seconds: 12.293
#   metrics:
#     accuracy_gap: 0.8519552871584892
#     periodic_accuracy: 0.9441340565681458
#     permuted_accuracy: 0.09217876940965652
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: b44b9c5bbe327eae6295bb6caf5f91ca58af34aa65784efcff81fec72d478193
#     cuda: null
#     evidence_revision: 86f0472d5dd7b331d5e12844768f0b38e133fa9e
#     gpu: null
#     host: 686490660a52
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
#     script_sha256: 6b77de1ee3925ec385171cbf61ecd7cd4e7287c97057a88087116beb86da16fb
#     source_sha256: 06ea2838810c53ba2c1ed43449bd3c41d3fddfa008eddb8865b7c2cfcf08fcc3
#     wheel_sha256: 460e2b28cbe9617f3900ae0a1341f38bb38d25bb476830ba582ca189ac89416b
#   seed: 41
#   started_at: '2026-11-01T00:19:03.638969+00:00'
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
# observed_at: 2025-02-04
# target: February
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-month-from-day-of-year
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calendar, observed_at, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt,),
#   node("target", kind: "target", type: "Category", detail: "month",),
# )))
# ```
#
# ## Comparison
#
# Train on odd days of each month, test even days in unseen years, and shuffle timestamps while keeping month answers fixed.
#
# ## Result
#
# {{< evidence P044 status >}}
#
# The model learns the restricted calendar relationship. Exact month extraction is simpler in preprocessing, and leap-year ambiguity needs additional information.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P044 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P044 script >}}
#
# ### How it works
#
# In a non-leap calendar, day-of-year determines month. The model must learn
# the intervals in that cyclic representation. Training uses odd-numbered
# days of each month in 2017–2019; validation uses odd days in 2021. Test dates
# are even-numbered days in 2023 and 2025. All these years are non-leap.
#
# After 70 deterministic epochs, the test evaluates both the original
# held-out data and a version with timestamps permuted while labels stay put.
# Permutation preserves the marginals but destroys the useful relationship.
#
# ### Remaining work
#
# Repeat the gates over three core seeds and at least ten calibration seeds.
# Extend the family to week and multi-coordinate targets. Characterize timezone,
# daylight-saving, and leap-day preprocessing policies separately rather than
# assuming that DateParts adds those semantics.
#
# ### Complete experiment code
#

# %%
"""P044: learn month from day-of-year on unseen dates and years.

Only non-leap years are used. Training dates are odd days of the month;
held-out dates are even days in new years. Timestamp permutation retains the
labels and date marginals while removing their relationship.
"""

import calendar
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P044"


def records(years: Sequence[int], *, parity: int) -> Iterator[dict]:
    """Yield non-leap dates selected by day-of-month parity."""
    for year in years:
        if calendar.isleap(year):
            raise ValueError(f"month periodicity requires non-leap years, got {year}")
        current = date(year, 1, 1)
        while current.year == year:
            if current.day % 2 == parity:
                yield {"observed_at": current, "target": calendar.month_name[current.month]}
            current += timedelta(days=1)


def fit(
    *,
    dateparts: Sequence[str],
    train: Callable[[], Iterator[dict]],
    validate: Callable[[], Iterator[dict]],
    epochs: int,
    seed: int,
    steps: int | None,
    accelerator: str,
) -> rf.Model:
    """Train on the selected calendar coordinates with the remaining schema fixed."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=3,
        n_heads=4,
        observed_at=rf.DateParts(dateparts=list(dateparts)),
        target=rf.Category(mask=True, p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=epochs,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    return model


def accuracy(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> float:
    """Evaluate held-out labels without updating the Category vocabulary."""
    data = rf.SyntheticDataModule(test=records)
    metrics = model.test(
        data,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        verbose=False,
        batch_size=128,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
    )[0]
    return float(metrics[".target/test.accuracy.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = partial(records, (2017, 2018, 2019), parity=1)
    validate = partial(records, (2021,), parity=1)
    test = list(records((2023, 2025), parity=0))
    model = fit(
        dateparts=("day_of_year",),
        train=train,
        validate=validate,
        epochs=70,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    periodic_accuracy = accuracy(model, lambda: iter(test), accelerator)
    order = np.random.default_rng(seed + 1).permutation(len(test))
    permuted = [{**row, "observed_at": test[index]["observed_at"]} for row, index in zip(test, order, strict=True)]
    permuted_accuracy = accuracy(model, lambda: iter(permuted), accelerator)
    gap = periodic_accuracy - permuted_accuracy
    return {
        "periodic_accuracy": periodic_accuracy,
        "permuted_accuracy": permuted_accuracy,
        "accuracy_gap": gap,
    }, {
        "Unseen-date month accuracy reaches 0.90": periodic_accuracy >= 0.90,
        "Timestamp permutation accuracy is at most 0.20": permuted_accuracy <= 0.20,
        "Original dates exceed permutation by at least 0.65 accuracy": gap >= 0.65,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=41)

# %% [markdown]
# </details>
