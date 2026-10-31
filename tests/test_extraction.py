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
