"""Evidence survives repeated runs and drives static proof documentation."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
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
    renderer = None
    scripts = tmp_path / "proofs"
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
    source.write_text(yaml.safe_dump({"schema": 1, "proofs": entries}, sort_keys=False))
    monkeypatch.setattr(reporting, "RESULTS", source)
    monkeypatch.setattr(reporting, "provenance", lambda script: deepcopy(result()["provenance"]))
    return reporting, renderer, tmp_path


def test_concurrent_appends_preserve_history_and_other_proofs(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    reporting, _, root = evidence
    output = root / "collected/results.yaml"
    historical = result(seed=1)
    reporting.save("P001", historical, output)
    before = yaml.safe_load(output.read_text())
    barrier = Barrier(8)

    def append(index: int) -> None:
        barrier.wait(timeout=10)
        reporting.save("P001" if index % 2 else "P002", result(seed=index + 10), output)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(8)))

    after = yaml.safe_load(output.read_text())
    assert after["schema"] == 1
    for identifier in ("P001", "P002"):
        original = before["proofs"][identifier]
        recorded = after["proofs"][identifier]
        assert {key: value for key, value in recorded.items() if key != "runs"} == {
            key: value for key, value in original.items() if key != "runs"
        }
    assert after["proofs"]["P001"]["runs"][0] == historical
    assert sorted(run["seed"] for run in after["proofs"]["P001"]["runs"]) == [1, 11, 13, 15, 17]
    assert sorted(run["seed"] for run in after["proofs"]["P002"]["runs"]) == [10, 12, 14, 16]
    assert yaml.safe_load(reporting.RESULTS.read_text())["proofs"]["P001"]["runs"] == []


def test_interrupted_replacement_keeps_existing_evidence(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, _, root = evidence
    output = root / "collected/results.yaml"
    reporting.save("P001", result(seed=1), output)
    before = output.read_bytes()

    def interrupt(source: Path, destination: Path) -> None:
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted publication"):
        reporting.save("P001", result(seed=2), output)
    assert output.read_bytes() == before
    assert {path.name for path in output.parent.iterdir()} == {"results.yaml", "results.yaml.lock"}
