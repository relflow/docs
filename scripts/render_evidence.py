"""Render recorded evidence and source listings without running experiments."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from evidence.experiments.source import fingerprint, metadata


def clean(root: Path) -> None:
    """Remove temporary evidence inputs after Quarto has copied the rendered output."""
    docs = root / "docs/evidence"
    if not docs.exists():
        return
    entries = yaml.safe_load((root / "evidence/results.yaml").read_text())["evidence"]
    pages = [(root / entry["page"]).resolve() for entry in entries.values()]
    for page in pages:
        if page.parent.parent != docs.resolve() or page.suffix != ".py":
            raise ValueError(f"Cannot clean evidence page outside docs/evidence/<family>/: {page}")
    for page in pages:
        page.unlink(missing_ok=True)
        page.with_suffix(".quarto_ipynb").unlink(missing_ok=True)
        for temporary in page.parent.glob(f"{page.stem}.quarto_ipynb_*"):
            temporary.unlink()
    for name in ("_generated", "__pycache__"):
        path = docs / name
        if path.exists():
            shutil.rmtree(path)
    (docs / "catalog.yaml").unlink(missing_ok=True)
    for path in docs.iterdir():
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if not any(docs.iterdir()):
        docs.rmdir()


def latest(entry: dict[str, Any]) -> dict[str, Any] | None:
    return next((run for run in reversed(entry["runs"]) if run["mode"] == "full"), None)


def status(entry: dict[str, Any]) -> str:
    run = latest(entry)
    if run is None:
        return "Missing"
    if run["outcome"] == "error":
        return "Error"
    if run["outcome"] == "not_met":
        return "Failing"
    return entry["historical"]["status"]


def value(item: Any) -> str:
    if isinstance(item, float):
        return f"{item:.6g}"
    if isinstance(item, list):
        prefix = f"{len(item)} values; final 8: " if len(item) > 8 else ""
        return prefix + ", ".join(value(part) for part in item[-8:])
    if isinstance(item, dict):
        return "; ".join(f"{key}: {value(part)}" for key, part in item.items())
    return str(item).replace("|", "\\|").replace("\n", " ")


def readout(header: dict[str, Any], run: dict[str, Any] | None) -> str:
    """Present a few authored comparisons directly from the recorded metrics."""
    rows = []
    formats = set()
    for selection in header.get("evidence-readout", []):
        label = selection["label"]
        path = selection["metric"]
        style = selection["format"]
        if not isinstance(path, list) or not path or style not in {"number", "percent", "error", "auc"}:
            raise ValueError(f"{header['evidence-id']}: readout {label!r} requires a metric path and supported format")
        measured = run.get("metrics", {}) if run else {}
        try:
            for part in path:
                measured = measured[part]
        except (KeyError, IndexError, TypeError):
            displayed = "Not recorded in this run"
        else:
            if isinstance(measured, bool) or not isinstance(measured, (int, float)):
                raise ValueError(
                    f"{header['evidence-id']}: readout {label!r} requires a numerical metric, got {measured!r}"
                )
            if not math.isfinite(measured):
                displayed = "Not available"
            elif style == "error":
                displayed = f"{measured * 100:.1f}% of baseline error"
            elif style == "percent":
                displayed = f"{measured * 100:.1f}%"
            elif style == "auc":
                displayed = f"{measured:.4f}"
            else:
                displayed = f"{measured:.4g}"
        rows.append(f"| {value(label)} | {displayed} |")
        formats.add(style)
    if not rows or run is None:
        return ""
    sections = ["| Comparison | Latest measurement |\n| --- | --- |\n" + "\n".join(rows)]
    if "error" in formats:
        sections.append(
            "Baseline error compares prediction error with a constant guess. "
            "100% matches that guess; lower is better. Details specify the baseline and root-mean-square error calculation."
        )
    if "auc" in formats:
        sections.append(
            "These are AUC ranking scores: 0.5 is chance and 1 is perfect separation. They are not percentages of correct answers."
        )
    return "\n\n".join(sections) + "\n"


def evidence(entry: dict[str, Any]) -> str:
    sections = []
    runs = entry["runs"]
    run = latest(entry)
    if run is not None:
        sections.extend(
            [
                "### Latest full run",
                f"Seed **{run['seed']}**, `{run['accelerator']}`, recorded {run['started_at']}. "
                f"Outcome: **{run['outcome'].replace('_', ' ')}**.",
            ]
        )
        origin = run.get("provenance", {})
        if origin:
            sections.append(
                f"Source fingerprint: `{origin['source_sha256']}`. "
                f"Python {origin['python']}; Torch {origin['packages']['torch']}."
            )
        if run.get("error"):
            sections.append("```text\n" + run["error"] + "\n```")
        if run.get("metrics"):
            sections.append(
                "| Measurement | Value |\n| --- | --- |\n"
                + "\n".join(f"| {value(name)} | {value(measured)} |" for name, measured in run["metrics"].items())
            )
        if run.get("checks"):
            sections.append(
                "<details>\n<summary>Behavioral checks</summary>\n\n"
                "| Behavioral check | Outcome |\n| --- | --- |\n"
                + "\n".join(f"| {value(name)} | {'Met' if met else 'Not met'} |" for name, met in run["checks"].items())
                + "\n\n</details>"
            )
    if runs and runs[-1]["mode"] == "smoke":
        smoke = runs[-1]
        sections.append(
            f"The latest smoke run used seed {smoke['seed']} and a {smoke['steps_override']}-step cap "
            f"per training stage. Execution {'failed' if smoke['outcome'] == 'error' else 'completed'}; "
            "a shortened run does not establish learning capability or change the evidence status."
        )
    full = [record for record in runs if record["mode"] == "full"]
    if len(full) > 1:
        sections.append(
            "### Full run history\n\nRuns may include earlier source versions. Repeating a seed is not an independent seed check.\n\n"
            "| Seed | Outcome | Recorded | Code fingerprint |\n| --- | --- | --- | --- |\n"
            + "\n".join(
                f"| {record['seed']} | {record['outcome'].replace('_', ' ')} | {record['started_at']} | "
                f"{record.get('provenance', {}).get('code_sha256', 'Not recorded')} |"
                for record in full
            )
        )
    sections.append("[Recorded results](../_generated/results.yaml).")
    return "\n\n".join(sections) + "\n"


def write(path: Path, content: str) -> None:
    """Preserve unchanged input timestamps so preview rendering can settle."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


GROUPS = {
    "aggregation": "Aggregation",
    "structure": "Structure",
    "state": "State",
    "identity": "Identity",
    "relational": "Relational",
    "representation": "Representation",
    "quantile": "Representation",
    "enum": "Representation",
    "cluster": "Representation",
    "vocabulary": "Vocabulary",
    "masking": "Masking",
    "mutations": "Mutations",
    "temporal": "Temporal",
    "composition": "Composition",
    "calibration": "Calibration",
    "anomaly": "Anomaly",
}
GROUP_ORDER = list(dict.fromkeys(GROUPS.values()))


def group(script: Path) -> str:
    """Catalog group for an evidence experiment, from its folder; unknown folders keep their own name."""
    return GROUPS.get(script.parent.name, script.parent.name.capitalize())


def render(root: Path) -> None:
    docs = root / "docs/evidence"
    source = root / "evidence/results.yaml"
    recorded = source.read_text()
    document = yaml.safe_load(recorded)
    if document.get("schema") != 2:
        raise ValueError(f"{source}: expected evidence schema 2")
    entries = document["evidence"]
    output = docs / "_generated"
    output.mkdir(parents=True, exist_ok=True)
    catalog = []
    pages = set()
    for identifier, entry in entries.items():
        script = root / entry["script"]
        code = script.read_text()
        header = metadata(code)
        if header["evidence-id"] != identifier:
            raise ValueError(f"{script}: evidence ID {header['evidence-id']} does not match {identifier}")
        if header.get("execute", {}).get("enabled") is not False or header.get("execute", {}).get("eval") is not False:
            raise ValueError(
                f"{script}: evidence documentation requires execute.enabled: false and execute.eval: false"
            )
        page = root / entry["page"]
        if page.parent.parent != docs or page.suffix != ".py":
            raise ValueError(f"{identifier}: expected a generated script page under docs/evidence/<family>/")
        if page in pages:
            raise ValueError(f"{identifier}: another experiment already uses page {page}")
        pages.add(page)
        page.parent.mkdir(exist_ok=True)
        write(page, code)
        run = latest(entry)
        state = status(entry)
        if run is None:
            title = state
            summary = "No full experiment has been recorded."
        else:
            title = f"Latest full run · {state}"
            met = sum(run.get("checks", {}).values())
            total = len(run.get("checks", {}))
            summary = f"Seed {run['seed']} on `{run['accelerator']}`: {met} of {total} test checks met."
            if run["outcome"] == "error":
                summary = "The latest full experiment encountered an execution error. See the recorded error below."
            elif run["outcome"] == "not_met":
                summary += " The experiment did not meet every criterion."
            elif state == "Limited":
                summary += " This experiment demonstrates a limitation."
            elif state == "Partial":
                summary += " The broader claim remains incomplete."
            full = [record for record in entry["runs"] if record["mode"] == "full"]
            if len(full) > 1:
                passed = sum(record["outcome"] == "met" for record in full)
                seeds = len({record["seed"] for record in full})
                summary += (
                    f" History: {passed} of {len(full)} full runs met every check, across {seeds} distinct seeds. "
                    "History can include earlier source versions."
                )
            recorded_hash = run.get("provenance", {}).get("code_sha256")
            if recorded_hash and recorded_hash != fingerprint(code):
                summary += (
                    " The experiment code has changed since this run; these measurements describe its earlier version."
                )
        kind = "warning" if state in {"Failing", "Error", "Limited"} else "note"
        write(
            output / f"{identifier}-status.md",
            f'::: {{.evidence-result .evidence-result-{kind} role="note"}}\n'
            f"**{identifier} · {title}**\n\n{summary}\n:::\n\n" + readout(header, run),
        )
        write(output / f"{identifier}-evidence.md", evidence(entry))
        write(
            output / f"{identifier}-script.md",
            "Evidence execution is optional and separate from building this site. "
            "Use a RelFlow build matching the recorded provenance; replace the wheel path below with that artifact. "
            "Run from the relflow/docs repository's root:\n\n"
            f"```bash\nuv run --with /path/to/matching-relflow.whl python evidence/experiments/run.py {identifier}\n```\n\n"
            "Or, from a prepared evidence environment, run the self-contained script directly:\n\n"
            f"```bash\nPYTHONPATH=evidence/experiments python {entry['script']}\n```\n\n"
            f"Add `--accelerator gpu` for CUDA or `--seed 42` for another seeded experiment. "
            "`--steps 2` checks execution with a short training budget; it is recorded as a smoke run.\n\n"
            f"[Download the complete evidence](../_generated/{identifier}.py).\n",
        )
        write(output / f"{identifier}.py", code)
        catalog.append(
            {
                "id": identifier,
                "title": header["title"],
                "group": group(script),
                "categories": header["categories"],
                "path": page.relative_to(root / "docs").with_suffix(".html").as_posix(),
                "evidence-status": state,
                "description": header["description"],
            }
        )
    for page in docs.glob("*/*.py"):
        if page.parent != output and page not in pages:
            page.unlink()

    def position(row: dict[str, Any]) -> tuple[int, str]:
        order = GROUP_ORDER.index(row["group"]) if row["group"] in GROUP_ORDER else len(GROUP_ORDER)
        return (order, row["id"])

    catalog.sort(key=position)
    write(docs / "catalog.yaml", yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True))
    write(output / "results.yaml", recorded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove generated evidence inputs after rendering.")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if options.clean:
        clean(root)
    else:
        render(root)
