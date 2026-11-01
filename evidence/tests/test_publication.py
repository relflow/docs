"""Publication must not disguise incomplete execution with historical results."""

import json

import pytest
import yaml
from evidence_ci import plan, task_key, write_json
from record import select


@pytest.fixture
def reports(tmp_path):
    root = tmp_path / "reports"
    manifest = plan(
        run_id="publication-test",
        candidate={"revision": "candidate", "wheel_sha256": "candidate-wheel"},
        baseline={"revision": "base", "wheel_sha256": "base-wheel"},
        evidence_sha="tested-suite",
    )
    write_json(root / "manifest.json", manifest)
    for task in manifest["tasks"]:
        result = {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": task["accelerator"],
            "outcome": "met",
            "checks": {"held-out signal": True},
            "provenance": {
                "evidence_revision": manifest["evidence_sha"],
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        write_json(root / "results" / f"{task_key(task)}.json", {"task": task, "error": None, "result": result})
    # An old aggregate must not override what the individual records prove.
    write_json(root / "summary.json", {"complete": True, "passed": True, "errors": []})
    return root


def test_selection_preserves_candidate_and_github_provenance(reports, tmp_path):
    github_run = tmp_path / "github-run.json"
    github_run.write_text(json.dumps({"run_id": 123, "run_attempt": 2, "head_sha": "pr-head"}))
    output = tmp_path / "publication"
    select(reports, output, github_run=github_run)
    selection = yaml.safe_load(output.read_text())["selections"][-1]
    assert selection["evidence_revision"] == "tested-suite"
    assert selection["github_run"]["head_sha"] == "pr-head"
    results = yaml.safe_load(output.read_text())["evidence"]
    assert len(results) == 79
    assert all(entry["runs"][-1]["provenance"]["relflow_revision"] == "candidate" for entry in results.values())


@pytest.mark.parametrize("fault", ["missing", "error", "provenance", "smoke", "partial"])
def test_selection_rejects_invalid_records_even_with_successful_summary(reports, tmp_path, fault):
    path = next((reports / "results/candidate").rglob("*.json"))
    if fault == "missing":
        path.unlink()
    elif fault in {"error", "provenance"}:
        record = json.loads(path.read_text())
        if fault == "error":
            record["error"] = "timeout"
        else:
            record["result"]["provenance"]["wheel_sha256"] = "wrong-wheel"
        path.write_text(json.dumps(record))
    else:
        path = reports / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["mode" if fault == "smoke" else "full_catalog"] = "smoke" if fault == "smoke" else False
        path.write_text(json.dumps(manifest))
    output = tmp_path / "publication"
    with pytest.raises(ValueError, match="complete full-catalog"):
        select(reports, output)
    assert not output.exists()
