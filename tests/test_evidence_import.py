"""Verify imported releases before changing the docs publication snapshot."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts.import_evidence import import_bundle, sha256, verify_snapshot


def archive(tmp_path: Path, *, malicious: str | None = None, script: str | None = None) -> Path:
    records = {
        "schema": 2,
        "evidence": {
            "P001": {
                "script": script or "vendor/evidence/experiments/family/signal.py",
                "page": "docs/evidence/family/signal.py",
            }
        },
    }
    files = {
        "results.yaml": yaml.safe_dump(records).encode(),
        "experiments/family/signal.py": b"raise RuntimeError('must never execute')\n",
    }
    manifest = {
        "schema": 1,
        "commit": "a" * 40,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["manifest.json"] = json.dumps(manifest).encode()
    output = tmp_path / "export.tar.gz"
    with tarfile.open(output, "w:gz") as handle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
        if malicious:
            member = tarfile.TarInfo(malicious)
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
    return output


def test_pinned_bundle_imports_without_executing_its_sources(tmp_path):
    bundle = archive(tmp_path)
    import_bundle(bundle, sha256(bundle), root=tmp_path)
    assert verify_snapshot(tmp_path)["commit"] == "a" * 40
    (tmp_path / "vendor/evidence/experiments/family/signal.py").write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_snapshot(tmp_path)


@pytest.mark.parametrize("fault", ["digest", "traversal", "absolute", "duplicate", "source-escape"])
def test_failed_import_preserves_the_previous_snapshot(tmp_path, fault):
    original = archive(tmp_path)
    import_bundle(original, sha256(original), root=tmp_path)
    before = (tmp_path / "contracts/evidence.json").read_bytes()
    malicious = {"traversal": "../outside", "absolute": "/outside", "duplicate": "manifest.json"}.get(fault)
    replacement = archive(tmp_path, malicious=malicious, script="/etc/passwd" if fault == "source-escape" else None)
    with pytest.raises(ValueError):
        import_bundle(replacement, "0" * 64 if fault == "digest" else sha256(replacement), root=tmp_path)
    assert (tmp_path / "contracts/evidence.json").read_bytes() == before
    verify_snapshot(tmp_path)
