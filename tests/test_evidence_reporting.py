"""Evidence survives repeated runs and drives static evidence documentation."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evidence"))


def load(path: Path) -> ModuleType:
    """Load the standalone reporting tools without importing experiment scripts."""
    spec = importlib.util.spec_from_file_location(f"evidence_{path.stem}", path)
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


def append_result(identifier: str, recorded: dict, output: Path) -> None:
    document = yaml.safe_load(output.read_text())
    document["evidence"][identifier]["runs"].append(recorded)
    output.write_text(yaml.safe_dump(document, sort_keys=False))


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, Path]:
    reporting = None
    renderer = load(ROOT / "scripts/render_evidence.py")
    scripts = tmp_path / "evidence"
    pages = tmp_path / "docs/evidence/family"
    scripts.mkdir(parents=True)
    pages.mkdir(parents=True)
    entries = {}
    for identifier, slug, state in (("P001", "signal", "Passing"), ("P002", "noise", "Limited")):
        script = scripts / f"{slug}.py"
        metadata = {
            "evidence-id": identifier,
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
            "raise RuntimeError('a documentation build executed a evidence')\n"
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
    source.write_text(yaml.safe_dump({"schema": 2, "evidence": entries}, sort_keys=False))
    return reporting, renderer, tmp_path


@pytest.mark.parametrize(
    ("historical", "outcome", "expected", "kind"),
    [
        ("Passing", "met", "Passing", "note"),
        ("Limited", "met", "Limited", "warning"),
        ("Partial", "met", "Partial", "note"),
        ("Passing", "not_met", "Failing", "warning"),
        ("Passing", "error", "Error", "warning"),
        ("Passing", None, "Missing", "note"),
    ],
)
def test_statuses_match_catalog_callouts_legend_and_colors(
    evidence: tuple[ModuleType, ModuleType, Path], historical: str, outcome: str | None, expected: str, kind: str
) -> None:
    _, renderer, root = evidence
    source = root / "evidence/results.yaml"
    document = yaml.safe_load(source.read_text())
    entry = document["evidence"]["P001"]
    entry["historical"]["status"] = historical
    entry["runs"] = [result(outcome=outcome)] if outcome is not None else []
    source.write_text(yaml.safe_dump(document))

    renderer.render(root)

    catalog = yaml.safe_load((root / "docs/evidence/catalog.yaml").read_text())
    assert next(row for row in catalog if row["id"] == "P001")["evidence-status"] == expected
    notice = (root / "docs/evidence/_generated/P001-status.md").read_text()
    title = f"Latest full run · {expected}" if outcome is not None else expected
    assert f'::: {{.evidence-result .evidence-result-{kind} role="note"}}' in notice
    assert f"**P001 · {title}**" in notice
    legend = (ROOT / "docs/evidence.qmd").read_text()
    assert f'[{expected}]{{.evidence-status data-evidence-status="{expected}"}}' in legend
    styles = (ROOT / "docs/assets/stylesheets/evidence.css").read_text()
    assert f'\n.evidence-status[data-evidence-status="{expected}"] {{' in styles
    assert f'\nbody.quarto-dark .evidence-status[data-evidence-status="{expected}"] {{' in styles


def test_docs_refresh_from_yaml_and_embed_scripts_without_executing(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    source = root / "evidence/results.yaml"
    generated = root / "docs/evidence/_generated"
    renderer.render(root)
    notice = (generated / "P001-status.md").read_text()
    assert "Missing" in notice and "No full experiment has been recorded." in notice
    assert "Historical summary" not in notice
    assert "Historical measurements" not in (generated / "P001-evidence.md").read_text()

    failed = result(outcome="not_met")
    failed["metrics"] = {"held_out_rmse": 0.875}
    append_result("P001", failed, source)
    renderer.render(root)
    catalog = yaml.safe_load((root / "docs/evidence/catalog.yaml").read_text())
    assert {row["id"]: row["evidence-status"] for row in catalog} == {
        "P001": "Failing",
        "P002": "Missing",
    }
    assert "0 of 1 test checks met" in (generated / "P001-status.md").read_text()
    assert "| held_out_rmse | 0.875 |" in (generated / "P001-evidence.md").read_text()

    append_result("P001", result(seed=29), source)
    renderer.render(root)
    assert "Passing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **29**" in text and "| held_out_rmse | 0.125 |" in text
    assert "| held_out_rmse | 0.875 |" not in text
    assert "<details>\n<summary>Behavioral checks</summary>\n\n" in text
    assert "| Held-out RMSE is below 0.2 | Met |\n\n</details>" in text
    assert "Historical" not in text and "https://example.com/signal" not in text
    assert (generated / "results.yaml").read_bytes() == source.read_bytes()
    assert len(yaml.safe_load((generated / "results.yaml").read_text())["evidence"]["P001"]["runs"]) == 2
    script = (root / "evidence/signal.py").read_text()
    listing = (generated / "P001-script.md").read_text()
    assert (root / "docs/evidence/family/signal.py").read_text() == script
    assert (generated / "P001.py").read_text() == script
    assert "Download the complete evidence" in listing
    assert "experiments/run.py P001" in listing
    assert "{python}" not in listing
    assert not (root / "executed").exists()


def test_render_preserves_unchanged_input_timestamps(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    renderer.render(root)
    generated = root / "docs/evidence"
    files = {path for path in generated.rglob("*") if path.is_file()}
    timestamp = 1_700_000_000_000_000_000
    for path in files:
        os.utime(path, ns=(timestamp, timestamp))

    renderer.render(root)
    assert all(path.stat().st_mtime_ns == timestamp for path in files)

    script = root / "evidence/signal.py"
    script.write_text(script.read_text().replace("# title: Signal", "# title: Updated signal"))
    renderer.render(root)
    changed = {path.relative_to(generated).as_posix() for path in files if path.stat().st_mtime_ns != timestamp}
    assert changed == {"family/signal.py", "_generated/P001.py", "catalog.yaml"}
    assert (generated / "family/signal.py").read_text() == script.read_text()
    assert (generated / "_generated/P001.py").read_text() == script.read_text()


def test_docs_reject_enabled_experiments_before_staging(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    script = root / "evidence/signal.py"
    script.write_text(script.read_text().replace("enabled: false", "enabled: true"))
    with pytest.raises(ValueError, match="requires execute.enabled: false"):
        renderer.render(root)
    assert not (root / "docs/evidence/family/signal.py").exists()
    assert not (root / "executed").exists()


def test_docs_cleanup_preserves_authored_sources_results_and_published_files(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    docs = root / "docs/evidence"
    catalog = root / "docs/evidence.qmd"
    catalog.write_text("# Evidences\n")
    notes = docs / "notes"
    notes.mkdir()
    (notes / "draft.md").write_text("Authored notes\n")
    (docs / "retired-family").mkdir()
    published = root / "docs/site/evidence/_generated"
    published.mkdir(parents=True)
    (published / "results.yaml").write_text("Published evidence\n")
    source = root / "evidence/results.yaml"
    before = source.read_bytes()
    script = (root / "evidence/signal.py").read_bytes()
    renderer.render(root)
    (docs / "_generated/P001-insights.md").write_text("Obsolete fragment\n")
    (docs / "family/signal.quarto_ipynb").write_text("Temporary notebook\n")
    (docs / "family/signal.quarto_ipynb_1").write_text("Temporary notebook cell\n")
    (docs / "family/signal.quarto_ipynb_2").write_text("Temporary notebook cell\n")

    renderer.clean(root)
    renderer.clean(root)

    assert {path.name for path in docs.iterdir()} == {"notes"}
    assert catalog.read_text() == "# Evidences\n"
    assert (notes / "draft.md").read_text() == "Authored notes\n"
    assert (published / "results.yaml").read_text() == "Published evidence\n"
    assert source.read_bytes() == before
    assert (root / "evidence/signal.py").read_bytes() == script
    (notes / "draft.md").unlink()
    renderer.clean(root)
    renderer.clean(root)
    assert not docs.exists()


def test_docs_cleanup_rejects_paths_outside_generated_families(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    source = root / "evidence/results.yaml"
    document = yaml.safe_load(source.read_text())
    document["evidence"]["P001"]["page"] = "evidence/signal.py"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="Cannot clean evidence page outside"):
        renderer.clean(root)
    assert (root / "evidence/signal.py").is_file()


def test_changed_script_warns_without_rewriting_recorded_results(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    source = root / "evidence/results.yaml"
    script = root / "evidence/signal.py"
    measured = result(seed=43)
    measured["provenance"]["code_sha256"] = renderer.fingerprint(script.read_text())
    append_result("P001", measured, source)
    before = source.read_bytes()
    generated = root / "docs/evidence/_generated"

    renderer.render(root)
    assert "code has changed" not in (generated / "P001-status.md").read_text()
    script.write_text(script.read_text().replace("# Static experiment", "# Clearer explanation"))
    renderer.render(root)
    assert "code has changed" not in (generated / "P001-status.md").read_text()
    script.write_text(script.read_text() + "\nexperiment_size = 2048\n")
    renderer.render(root)

    status = (generated / "P001-status.md").read_text()
    assert "code has changed" in status
    assert "earlier version" in status
    assert "Passing" in status
    evidence_text = (generated / "P001-evidence.md").read_text()
    assert "Seed **43**" in evidence_text
    assert "| held_out_rmse | 0.125 |" in evidence_text
    assert "experiment_size = 2048" in (generated / "P001.py").read_text()
    assert source.read_bytes() == before
    assert not (root / "executed").exists()


def test_readout_uses_explicit_paths_and_explains_scores(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, _ = evidence
    header = {
        "evidence-id": "P001",
        "evidence-readout": [
            {"label": "Prediction error", "metric": ["arm", "nrmse"], "format": "error"},
            {"label": "Correct answers", "metric": ["accuracy"], "format": "percent"},
            {"label": "Matching IDs", "metric": ["auc"], "format": "auc"},
            {"label": "Last count", "metric": ["counts", -1], "format": "number"},
        ],
    }
    run = result()
    run["metrics"] = {"arm": {"nrmse": 0.025}, "accuracy": 0.75, "auc": 0.9998, "counts": [2, 5]}
    text = renderer.readout(header, run)
    assert "| Prediction error | 2.5% of baseline error |" in text
    assert "| Correct answers | 75.0% |" in text
    assert "| Matching IDs | 0.9998 |" in text
    assert "| Last count | 5 |" in text
    assert "100% matches that guess; lower is better" in text
    assert "not percentages of correct answers" in text
    assert renderer.readout(header, None) == ""
    run["metrics"] = {"auc": float("nan")}
    text = renderer.readout(header, run)
    assert "| Prediction error | Not recorded in this run |" in text
    assert "| Matching IDs | Not available |" in text


@pytest.mark.parametrize("measured", [True, "0.125", {"nrmse": 0.125}])
def test_readout_rejects_non_numerical_selections(
    evidence: tuple[ModuleType, ModuleType, Path], measured: object
) -> None:
    _, renderer, _ = evidence
    header = {"evidence-id": "P001", "evidence-readout": [{"label": "Error", "metric": ["score"], "format": "error"}]}
    with pytest.raises(ValueError, match="P001: readout 'Error' requires a numerical metric"):
        renderer.readout(header, {"metrics": {"score": measured}})


def test_overview_retains_mixed_history_and_ignores_smoke_metrics(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    source = root / "evidence/results.yaml"
    script = root / "evidence/signal.py"
    script.write_text(
        script.read_text().replace(
            "# description: signal",
            "# evidence-readout:\n# - label: Prediction error\n#   metric: [held_out_rmse]\n#   format: number\n# description: signal",
        )
    )
    append_result("P001", result(seed=7, outcome="not_met"), source)
    append_result("P001", result(seed=11), source)
    smoke = result(seed=11, mode="smoke")
    smoke["metrics"]["held_out_rmse"] = 999
    append_result("P001", smoke, source)
    before = source.read_bytes()
    renderer.render(root)
    generated = root / "docs/evidence/_generated"
    text = (generated / "P001-status.md").read_text()
    assert "History: 1 of 2 full runs met every check, across 2 distinct seeds" in text
    assert "| Prediction error | 0.125 |" in text and "999" not in text
    history = (generated / "P001-evidence.md").read_text()
    assert "| 7 | not met |" in history and "| 11 | met |" in history
    assert "Repeating a seed is not an independent seed check" in history
    assert source.read_bytes() == before


def test_static_evidence_publication_imports_no_model_runtime(evidence, monkeypatch):
    _, _, root = evidence
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"relflow", "torch", "lightning", "numpy"}, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    renderer = load(ROOT / "scripts/render_evidence.py")
    append_result("P001", result(), root / "evidence/results.yaml")
    renderer.render(root)
    assert (root / "docs/evidence/_generated/P001-status.md").is_file()
    assert not (root / "executed").exists()
    listing = (root / "docs/evidence/_generated/P001-script.md").read_text()
    assert "separate from building this site" in listing
    assert "matching-relflow.whl" in listing
    assert "relflow/docs repository" in listing
