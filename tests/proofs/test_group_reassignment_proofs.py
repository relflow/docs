"""Protect the information boundary and scorer used by experimental grouping proofs."""

import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import relflow as rf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "proofs"))


def load(name):
    spec = importlib.util.spec_from_file_location(f"proof_{name}", ROOT / "proofs/group" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case", ["separated_four", "crossing_four", "imbalanced_four", "separated_three", "separated_six"]
)
def test_numeric_control_preserves_targets_and_identity_frequencies(case):
    proof = load("predictive_regimes")
    stable, groups = proof.generate(19, case, 1, 4096)
    control, _ = proof.generate(19, case, 1, 4096, shuffle=True)
    fresh, _ = proof.generate(19, case, 2, 4096)

    assert stable == proof.generate(19, case, 1, 4096)[0]
    assert stable != fresh
    assert [(row["x"], row["y"]) for row in stable] == [(row["x"], row["y"]) for row in control]
    assert Counter(row["entity"] for row in stable) == Counter(row["entity"] for row in control)
    process = proof.CASES[case]
    expected_counts = {index: count for index, count in enumerate(process["populations"])}
    assert Counter(proof.partition(19, case)) == expected_counts
    x = np.asarray([row["x"] for row in stable])
    y = np.asarray([row["y"] for row in stable])
    noise = y - np.asarray(process["intercepts"])[groups] - np.asarray(process["slopes"])[groups] * x
    assert 0.09 < noise.std() < 0.11


@pytest.mark.parametrize("accept", [True, False])
def test_numeric_reassignment_uses_training_error_and_validation_acceptance(monkeypatch, accept):
    proof = load("predictive_regimes")
    model = rf.Model.xs(d_model=8, entity=rf.Group(n_groups=(2, 3), mask=True), y=rf.Number(mask=True))
    for token in ("a", "b"):
        rf.Group.assign(model, "/entity", token, 0)
    embedder = model.nodes["/entity"].embedder
    weight = embedder.embeddings["group"].weight
    before = weight.detach().clone()
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.AdamW([weight])
    optimizer.state[weight] = {
        "step": torch.tensor(5.0),
        "exp_avg": torch.ones_like(weight),
        "exp_avg_sq": torch.ones_like(weight),
    }

    def predict(configured, rows):
        choices = rf.Group.assignments(configured, "/entity")
        return np.asarray([(-1.0, 1.0, 3.0)[choices[row["entity"]]["group"]] for row in rows])

    def forbidden(*args, **kwargs):
        raise AssertionError("The reassignment routine must not consult the generating partition")

    monkeypatch.setattr(proof, "predict", predict)
    monkeypatch.setattr(proof, "partition", forbidden)
    train = [{"entity": "a", "x": 0.0, "y": 3.0}, {"entity": "b", "x": 0.0, "y": 1.0}]
    validation = train if accept else [{**row, "y": -1.0} for row in train]
    event = proof.reassignment(model, train, validation, [optimizer])

    assert event["accepted"] is accept
    assert optimizer.state[weight]["step"].item() == 5
    if accept:
        assert [item["group"] for item in rf.Group.assignments(model, "/entity").values()] == [2, 1]
        assert torch.allclose(weight[:2].logsumexp(-1), before[:2].logsumexp(-1), atol=1e-6)
        assert weight.grad[:2].eq(0).all()
        assert optimizer.state[weight]["exp_avg"][:2].eq(0).all()
        assert optimizer.state[weight]["exp_avg_sq"][:2].eq(0).all()
    else:
        assert torch.equal(weight, before)
        assert weight.grad.eq(1).all()
        assert optimizer.state[weight]["exp_avg"].eq(1).all()
    assert torch.equal(weight[2:], before[2:])
    assert optimizer.state[weight]["exp_avg"][2:].eq(1).all()


def test_numeric_reassignment_distinguishes_opposite_slopes_with_equal_means(monkeypatch):
    proof = load("predictive_regimes")
    model = rf.Model.xs(d_model=8, entity=rf.Group(n_groups=(2, 2), mask=True), y=rf.Number(mask=True))
    for token in ("a", "b"):
        rf.Group.assign(model, "/entity", token, 0)
    weight = model.nodes["/entity"].embedder.embeddings["group"].weight
    optimizer = torch.optim.AdamW([weight])

    def predict(configured, rows):
        choices = rf.Group.assignments(configured, "/entity")
        return np.asarray([(1.0, -1.0)[choices[row["entity"]]["group"]] * row["x"] for row in rows])

    monkeypatch.setattr(proof, "predict", predict)
    train = [
        {"entity": entity, "x": x, "y": slope * x} for entity, slope in (("a", 1.0), ("b", -1.0)) for x in (-1.0, 1.0)
    ]
    assert all(np.mean([row["y"] for row in train if row["entity"] == token]) == 0 for token in ("a", "b"))
    validation = [{**row, "x": row["x"] / 2, "y": row["y"] / 2} for row in train]
    event = proof.reassignment(model, train, validation, [optimizer])
    assert event["accepted"]
    assert event["proposed_validation_mse"] == 0.0
    assert [item["group"] for item in rf.Group.assignments(model, "/entity").values()] == [0, 1]


def test_categorical_control_breaks_identity_links_without_changing_observations():
    proof = load("predictive_labels")
    options = {"rows": 5000, "seed": 19, "partition_seed": 17}
    stable = list(proof.records(**options))
    control = list(proof.records(**options, no_regime=True))
    fresh = list(proof.records(**{**options, "seed": 20}))

    assert stable == list(proof.records(**options))
    assert stable != fresh
    assert Counter(row["merchant_id"] for row in stable) == Counter(row["merchant_id"] for row in control)
    assert [(row["channel"], row["amount"], row["outcome"]) for row in stable] == [
        (row["channel"], row["amount"], row["outcome"]) for row in control
    ]
    expected = proof.partition(17)
    stable_accuracy = np.mean(
        [
            row["outcome"]
            == f"class-{(expected[int(row['merchant_id'].rsplit('-', 1)[1])] + int(row['channel'].rsplit('-', 1)[1])) % 5}"
            for row in stable
        ]
    )
    control_accuracy = np.mean(
        [
            row["outcome"]
            == f"class-{(expected[int(row['merchant_id'].rsplit('-', 1)[1])] + int(row['channel'].rsplit('-', 1)[1])) % 5}"
            for row in control
        ]
    )
    assert stable_accuracy > 0.95
    assert 0.15 < control_accuracy < 0.25


def test_categorical_scorer_uses_true_label_probability_and_excludes_the_answer(monkeypatch):
    proof = load("predictive_labels")
    monkeypatch.setattr(proof.rf.Category, "vocabulary", classmethod(lambda cls, *args: ("class-0", "class-1")))
    observed = []

    def predict(inputs, **options):
        observed.append((inputs.column_names, options))
        return pa.Table.from_pylist(
            [
                {
                    "predictions": {
                        "outcome": {
                            "content": {
                                "value": "class-0",
                                "probability": 0.99,
                                "topk": [
                                    {"value": "class-0", "probability": 0.99},
                                    {"value": "class-1", "probability": 0.01},
                                ],
                            }
                        }
                    },
                }
            ]
        )

    model = SimpleNamespace(batch_size=128, predict=predict)
    rows = [{"merchant_id": "merchant-0000", "channel": "channel-0", "amount": 4.0, "outcome": "class-1"}]
    nll, correct = proof.losses(model, rows)

    assert nll.tolist() == pytest.approx([-math.log(0.01)])
    assert correct.tolist() == [False]
    assert observed == [(["merchant_id", "channel", "amount"], {"batch_size": 256})]
    assert model.batch_size == 128


def test_categorical_scorer_rejects_an_incomplete_distribution(monkeypatch):
    proof = load("predictive_labels")
    monkeypatch.setattr(proof.rf.Category, "vocabulary", classmethod(lambda cls, *args: ("class-0", "class-1")))
    output = pa.Table.from_pylist(
        [
            {
                "predictions": {
                    "outcome": {
                        "content": {
                            "value": "class-0",
                            "probability": 0.99,
                            "topk": [{"value": "class-0", "probability": 0.99}],
                        }
                    }
                }
            }
        ]
    )
    model = SimpleNamespace(batch_size=128, predict=lambda *args, **options: output)
    rows = [{"merchant_id": "merchant-0000", "channel": "channel-0", "amount": 4.0, "outcome": "class-0"}]

    with pytest.raises(ValueError, match="requires all observed target classes"):
        proof.losses(model, rows)
