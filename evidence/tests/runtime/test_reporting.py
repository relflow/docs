"""Evidence survives repeated runs and drives static proof documentation."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(path: Path) -> ModuleType:
    """Load the standalone reporting tools without importing experiment scripts."""
    spec = importlib.util.spec_from_file_location(f"proof_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(*, seed: int = 11, outcome: str = "met", mode: str = "full") -> dict:
    """Make a recorded experiment with enough provenance for its public report."""
    return {
        "started_at": "2026-09-14T12:00:00+00:00",
        "mode": mode,
        "seed": seed,
        "steps_override": 2 if mode == "smoke" else None,
        "accelerator": "cpu",
        "provenance": {
            "source_sha256": f"source-{seed}",
            "python": "3.12.0",
            "packages": {"torch": "2.8.0"},
        },
        "metrics": {"held_out_rmse": 0.125},
        "checks": {"Held-out RMSE is below 0.2": outcome == "met"},
        "outcome": outcome,
        "duration_seconds": 0.01,
    }


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, Path]:
    reporting = load(ROOT / "experiments/reporting.py")
    renderer = load(ROOT / "experiments/render.py")
    scripts = tmp_path / "experiments"
    pages = tmp_path / "docs/proofs/family"
    scripts.mkdir()
    pages.mkdir(parents=True)
    entries = {}
    for identifier, slug, state in (("P001", "signal", "Passing"), ("P002", "noise", "Limited")):
        script = scripts / f"{slug}.py"
        metadata = {
            "proof-id": identifier,
            "title": slug.title(),
            "categories": ["Family"],
            "description": slug,
            "execute": {"enabled": False, "eval": False},
        }
        header = "---\n" + yaml.safe_dump(metadata) + "---\n\n# Static experiment\n"
        script.write_text(
            "# %% [markdown]\n" + "\n".join("# " + line for line in header.splitlines()) + "\n\n# %%\n"
            '"""Literal Markdown example: ```python and ````.\n"""\n'
            "from pathlib import Path\n"
            f"Path({str(tmp_path / 'executed')!r}).write_text('executed')\n"
            "raise RuntimeError('a documentation build executed a proof')\n"
        )
        page = pages / f"{slug}.py"
        entries[identifier] = {
            "script": script.relative_to(tmp_path).as_posix(),
            "page": page.relative_to(tmp_path).as_posix(),
            "historical": {
                "status": state,
                "summary": f"Historical summary of {slug}.",
                "insights": f"The insight for {slug}.",
                "evidence": f"Historical measurements of {slug}.",
                "sources": [f"https://example.com/{slug}"],
            },
            "runs": [],
        }
    source = scripts / "results.yaml"
    source.write_text(yaml.safe_dump({"schema": 1, "experiments": entries}, sort_keys=False))
    monkeypatch.setattr(reporting, "RESULTS", source)
    monkeypatch.setattr(reporting, "provenance", lambda script: deepcopy(result()["provenance"]))
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    return reporting, renderer, tmp_path


def test_cli_records_actual_metrics_failed_checks_and_options(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reporting, _, root = evidence
    output = root / "runs.yaml"
    calls = []

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        calls.append((seed, steps, accelerator))
        return {"rmse": np.float64(0.375), "predictions": np.array([1.0, 2.0])}, {
            "Control remains at chance": np.bool_(True),
            "RMSE is below 0.2": np.bool_(False),
        }

    monkeypatch.setattr(
        sys, "argv", ["proof.py", "--seed", "23", "--accelerator", "auto", "--threads", "2", "--output", str(output)]
    )
    reporting.report("P001", experiment, seed=11)
    recorded = yaml.safe_load(output.read_text())["experiments"]["P001"]["runs"][-1]
    assert calls == [(23, None, "auto")]
    assert recorded["mode"] == "full"
    assert recorded["seed"] == 23
    assert recorded["threads"] == 2
    assert recorded["accelerator"] == "auto"
    assert recorded["metrics"] == {"rmse": 0.375, "predictions": [1.0, 2.0]}
    assert recorded["checks"] == {"Control remains at chance": True, "RMSE is below 0.2": False}
    assert recorded["outcome"] == "not_met"
    assert recorded["duration_seconds"] >= 0
    assert recorded["provenance"]["source_sha256"] == "source-11"
    assert "Recorded P001" in capsys.readouterr().out


def test_cli_records_execution_errors_before_exiting(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    output = root / "experiments/results.yaml"
    reporting.save("P001", result(), output)

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        raise RuntimeError("synthetic generation failed")

    monkeypatch.setattr(sys, "argv", ["proof.py", "--output", str(output)])
    with pytest.raises(SystemExit) as caught:
        reporting.report("P001", experiment, seed=19)
    assert caught.value.code == 1
    entry = yaml.safe_load(output.read_text())["experiments"]["P001"]
    assert len(entry["runs"]) == 2
    assert entry["runs"][-1]["outcome"] == "error"
    assert entry["runs"][-1]["seed"] == 19
    assert "RuntimeError: synthetic generation failed" in entry["runs"][-1]["error"]
    renderer.render(root)
    assert "Error" in (root / "docs/proofs/_generated/P001-status.md").read_text()
    assert "synthetic generation failed" in (root / "docs/proofs/_generated/P001-evidence.md").read_text()


def test_smoke_runs_do_not_replace_full_evidence_or_status(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    output = root / "experiments/results.yaml"
    full = result(seed=7, outcome="not_met")
    reporting.save("P001", full, output)

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        assert (seed, steps, accelerator) == (11, 2, "cpu")
        return {"smoke_only_measurement": 999}, {"Execution completed": True}

    monkeypatch.setattr(sys, "argv", ["proof.py", "--steps", "2", "--output", str(output)])
    reporting.report("P001", experiment, seed=11)
    entry = yaml.safe_load(output.read_text())["experiments"]["P001"]
    assert entry["runs"][-1]["mode"] == "smoke"
    assert entry["runs"][-1]["steps_override"] == 2
    assert renderer.latest(entry) == full
    renderer.render(root)
    generated = root / "docs/proofs/_generated"
    assert "Failing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **7**" in text
    assert "| held_out_rmse | 0.125 |" in text
    assert "smoke_only_measurement" not in text
    assert "2-step cap" in text
    assert "does not establish learning capability or change the proof status" in text


def test_provenance_failure_is_recorded_and_docs_remain_renderable(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    source = root / "experiments/results.yaml"
    reporting.save("P001", result(seed=7), source)
    calls = []

    def unavailable(script: Path) -> dict:
        raise OSError("cannot read installed package source")

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        calls.append(seed)
        return {"unexpected": 1}, {"unexpected": True}

    monkeypatch.setattr(reporting, "provenance", unavailable)
    monkeypatch.setattr(sys, "argv", ["proof.py", "--output", str(source)])
    with pytest.raises(SystemExit) as caught:
        reporting.report("P001", experiment, seed=31)
    assert caught.value.code == 1
    assert calls == []
    runs = yaml.safe_load(source.read_text())["experiments"]["P001"]["runs"]
    assert len(runs) == 2 and runs[0]["seed"] == 7
    assert runs[-1]["outcome"] == "error"
    assert runs[-1]["seed"] == 31
    assert not runs[-1].get("provenance")
    assert "OSError: cannot read installed package source" in runs[-1]["error"]
    assert "metrics" not in runs[-1]

    renderer.render(root)
    generated = root / "docs/proofs/_generated"
    assert "Error" in (generated / "P001-status.md").read_text()
    assert "cannot read installed package source" in (generated / "P001-evidence.md").read_text()
    catalog = yaml.safe_load((root / "docs/proofs/catalog.yaml").read_text())
    assert next(row for row in catalog if row["id"] == "P001")["proof-status"] == "Error"


def test_provenance_distinguishes_proof_revision_from_installed_library(monkeypatch):
    reporting = load(ROOT / "experiments/reporting.py")
    calls = []
    original_run = subprocess.run

    def revision(command, **options):
        if command != ["git", "rev-parse", "HEAD"]:
            return original_run(command, **options)
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="proof-commit\n", stderr="")

    monkeypatch.setattr(reporting.subprocess, "run", revision)
    measured = reporting.provenance(ROOT / "experiments/aggregation/raw_value_weight_sum.py")
    package = Path(measured["relflow_path"])
    expected = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        expected.update(("src/relflow/" + path.relative_to(package).as_posix()).encode())
        expected.update(path.read_bytes())
    assert measured["proof_revision"] == "proof-commit"
    assert "revision" not in measured
    assert measured["relflow_source_sha256"] == expected.hexdigest()
    assert len(measured["source_sha256"]) == 64
    assert calls[0][1]["cwd"] == ROOT
