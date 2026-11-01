"""Extraction preserves historical bytes and every experiment's executable syntax."""

import ast
import hashlib
import json

from evidence_ci import ROOT


def test_historical_archive_and_executable_experiments_are_preserved():
    manifest = json.loads((ROOT / "extraction.json").read_text())
    assert hashlib.sha256((ROOT / "archive/results.yaml").read_bytes()).hexdigest() == manifest["archive_sha256"]
    assert len(manifest["experiments"]) == 79
    for identifier, entry in manifest["experiments"].items():
        source = (ROOT / entry["destination"]).read_text()
        current = hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()
        assert current == entry["original_ast_sha256"], identifier


def test_canonical_publication_preserves_all_archived_measurements():
    import yaml

    original = yaml.safe_load((ROOT / "archive/results.yaml").read_text())["proofs"]
    current = yaml.safe_load((ROOT / "results.yaml").read_text())["evidence"]
    assert set(original) == set(current)
    for identifier, entry in original.items():
        assert json.dumps(current[identifier]["runs"][: len(entry["runs"])], sort_keys=True) == json.dumps(
            entry["runs"], sort_keys=True
        )
        assert current[identifier]["historical"] == entry["historical"]
