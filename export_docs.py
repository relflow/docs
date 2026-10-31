"""Export static sources and recorded measurements without importing RelFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from evidence_ci import bundle, digest, evaluate, registry, write_json

ROOT = Path(__file__).resolve().parent


def export(output: Path, *, reports: Path | None = None, github_run: Path | None = None) -> None:
    selected = None
    records = []
    if github_run and not reports:
        raise ValueError("GitHub run provenance requires a selected report")
    if reports:
        manifest = json.loads((reports / "manifest.json").read_text())
        records = [json.loads(path.read_text()) for path in sorted((reports / "results").rglob("*.json"))]
        summary = evaluate(manifest, records)
        if not summary["complete"] or summary["errors"]:
            raise ValueError("Publication requires a complete full-catalog run without execution or provenance errors")
        selected = {
            "run_id": manifest["run_id"],
            "candidate": manifest["candidate"],
            "manifest_sha256": digest(reports / "manifest.json"),
            "passed": summary["passed"],
            "evidence_revision": manifest["evidence_sha"],
        }
        if github_run:
            selected["github_run"] = json.loads(github_run.read_text())
    output.mkdir(parents=True, exist_ok=False)
    document = yaml.safe_load((ROOT / "archive/results.yaml").read_text())
    entries = registry()
    for identifier, entry in document["proofs"].items():
        entry["script"] = "vendor/evidence/" + entries[identifier]["script"]
        entry["page"] = entry["page"].replace("docs/proofs/", "docs/evidence/")
        script = ROOT / entries[identifier]["script"]
        target = output / entries[identifier]["script"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(script.read_bytes())
    for record in records:
        if record["task"]["role"] == "candidate":
            document["proofs"][record["task"]["id"]]["runs"].append(record["result"])
    document["schema"] = 2
    document["evidence"] = document.pop("proofs")
    (output / "results.yaml").write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    shutil.copy2(ROOT / "NOTICE", output / "NOTICE")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    contract = {
        "schema": 1,
        "repository": "https://github.com/relflow/evidence",
        "commit": revision,
        "archive_sha256": digest(ROOT / "archive/results.yaml"),
        "selection": selected,
        "files": {
            path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    write_json(output / "manifest.json", contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--github-run", type=Path)
    parser.add_argument("--bundle", type=Path)
    options = parser.parse_args()
    export(options.output, reports=options.reports, github_run=options.github_run)
    if options.bundle:
        bundle(options.output, options.bundle)
        options.bundle.with_suffix(options.bundle.suffix + ".sha256").write_text(
            hashlib.sha256(options.bundle.read_bytes()).hexdigest() + "  " + options.bundle.name + "\n"
        )


if __name__ == "__main__":
    main()
