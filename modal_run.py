"""Run the complete paired evidence suite on Modal and collect immutable reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import modal

from evidence_ci import ROOT, collect, digest, finish, plan, write_json


def run(options: argparse.Namespace) -> None:
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = {"revision": options.candidate_sha, "wheel_sha256": digest(options.candidate_wheel)}
    baseline = {"revision": options.baseline_sha, "wheel_sha256": digest(options.baseline_wheel)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = plan(
        run_id=output.name,
        candidate=candidate,
        baseline=baseline,
        evidence_sha=revision,
        identifiers=options.ids,
        steps=options.steps,
    )
    manifest["requirements_sha256"] = digest(options.requirements)
    write_json(output / "manifest.json", manifest)
    app = modal.App("relflow-evidence")
    # Preinstall CPU Torch so the exact public version in the lock uses its CPU build.
    torch_version = None
    for line in options.requirements.read_text().splitlines():
        if line.startswith("torch=="):
            torch_version = line.split("==", 1)[1].split(";", 1)[0].split(" ", 1)[0].rstrip("\\")
            break
    if not torch_version:
        raise ValueError("Locked requirements must pin torch")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(f"torch=={torch_version}+cpu", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install_from_requirements(str(options.requirements.resolve()))
        .pip_install("pyyaml==6.0.3")
        .add_local_dir(str(ROOT / "experiments"), "/evidence/experiments", copy=True)
        .add_local_file(str(ROOT / "evidence_ci.py"), "/evidence/evidence_ci.py", copy=True)
        .add_local_file(
            str(options.candidate_wheel.resolve()), "/wheels/candidate/" + options.candidate_wheel.name, copy=True
        )
        .add_local_file(
            str(options.baseline_wheel.resolve()), "/wheels/baseline/" + options.baseline_wheel.name, copy=True
        )
    )
    wheels = {
        "candidate": "/wheels/candidate/" + options.candidate_wheel.name,
        "baseline": "/wheels/baseline/" + options.baseline_wheel.name,
    }
    identities = {"candidate": candidate, "baseline": baseline}

    @app.function(
        image=image,
        cpu=2,
        memory=4096,
        max_containers=options.workers,
        timeout=1800,
        retries=0,
        serialized=True,
        include_source=False,
        single_use_containers=True,
        block_network=True,
        restrict_modal_access=True,
        _experimental_restrict_output=True,
    )
    def worker(task: dict) -> dict:
        import os
        import sys

        sys.path.insert(0, "/evidence")
        from evidence_ci import execute

        role = task["role"]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheels[role]],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        os.environ["EVIDENCE_REVISION"] = revision
        os.environ["EVIDENCE_RELFLOW_REVISION"] = identities[role]["revision"]
        os.environ["EVIDENCE_WHEEL_SHA256"] = identities[role]["wheel_sha256"]
        import relflow

        if not Path(relflow.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("RelFlow must be imported from the installed wheel")
        record = execute(task, root=Path("/evidence"), timeout=1700)
        record["modal_function_call_id"] = modal.current_function_call_id()
        # Restricted output transports plain values, never a candidate-produced pickle.
        json.dumps(record, allow_nan=False)
        return record

    try:
        with modal.enable_output(), app.run():
            print(f"Modal evidence dashboard: {app.get_dashboard_url()}", flush=True)
            manifest_path = output / "modal.json"
            write_json(manifest_path, {"app_id": app.app_id, "url": app.get_dashboard_url()})
            for record in worker.map(manifest["tasks"], order_outputs=False, return_exceptions=True):
                if isinstance(record, BaseException):
                    print(f"Modal task error: {record}", flush=True)
                    continue
                collect(output, record)
                print(
                    f"Collected {record['task']['role']} {record['task']['id']}: "
                    f"{(record.get('result') or {}).get('outcome', 'error')}",
                    flush=True,
                )
    finally:
        summary = finish(output)
        print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-wheel", required=True, type=Path)
    parser.add_argument("--baseline-wheel", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--ids", nargs="+")
    options = parser.parse_args()
    if options.workers < 1 or options.workers > 16:
        parser.error("workers must be between 1 and 16")
    if options.steps is not None and options.steps < 1:
        parser.error("steps must be positive")
    run(options)


if __name__ == "__main__":
    main()
