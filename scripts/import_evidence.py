"""Import a checksum-pinned GitHub Release export; rendering only reads local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(root: Path = ROOT) -> dict:
    snapshot = root / "vendor/evidence"
    contract = json.loads((root / "contracts/evidence.json").read_text())
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if contract["schema"] != 1 or manifest["schema"] != 1 or contract["commit"] != manifest["commit"]:
        raise ValueError("Evidence snapshot schema or revision differs from its contract")
    if contract["files"] != manifest["files"]:
        raise ValueError("Evidence snapshot file plan differs from its contract")
    actual = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Evidence snapshot contains missing or unregistered files")
    for name, expected in manifest["files"].items():
        path = snapshot / name
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Evidence snapshot digest mismatch: {name}")
    records = yaml.safe_load((snapshot / "results.yaml").read_text())
    if records.get("schema") != 2:
        raise ValueError("Expected publication schema 2")
    for entry in records["evidence"].values():
        source = root / entry["script"]
        page = Path(entry["page"])
        if not source.resolve().is_relative_to(snapshot.resolve()) or not source.is_file():
            raise ValueError("Evidence source must belong to the verified snapshot")
        if page.parent.parent != Path("docs/evidence") or page.suffix != ".py":
            raise ValueError("Evidence pages must belong to docs/evidence/<family>/")
    return contract


def import_bundle(
    bundle: Path, expected: str, *, root: Path = ROOT, repository: str = "relflow/evidence", release: str | None = None
) -> None:
    if len(expected) != 64 or sha256(bundle) != expected:
        raise ValueError("Evidence bundle digest differs from the pinned SHA-256")
    vendor = root / "vendor"
    vendor.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=vendor, prefix=".evidence-import-") as temporary:
        stage = Path(temporary) / "snapshot"
        stage.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if sum(member.size for member in members) > 256 * 1024 * 1024:
                raise ValueError("Evidence export exceeds the 256 MiB import bound")
            names = set()
            for member in members:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts or member.name in names:
                    raise ValueError(f"Unsafe or duplicate evidence archive member: {member.name}")
                names.add(member.name)
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        manifest = json.loads((stage / "manifest.json").read_text())
        if manifest["schema"] != 1 or names != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Evidence bundle manifest does not match its files")
        for name, checksum in manifest["files"].items():
            if sha256(stage / name) != checksum:
                raise ValueError(f"Evidence bundle file digest mismatch: {name}")
        # Validate the publication paths before replacing a working snapshot.
        records = yaml.safe_load((stage / "results.yaml").read_text())
        if records.get("schema") != 2:
            raise ValueError("Expected publication schema 2")
        for entry in records["evidence"].values():
            source = PurePosixPath(entry["script"])
            page = PurePosixPath(entry["page"])
            if source.parts[:2] != ("vendor", "evidence") or ".." in source.parts:
                raise ValueError("Evidence source escapes the publication snapshot")
            if source.relative_to("vendor/evidence").as_posix() not in manifest["files"]:
                raise ValueError("Evidence source is absent from the verified file plan")
            if page.parent.parent != PurePosixPath("docs/evidence") or page.suffix != ".py":
                raise ValueError("Evidence page escapes the generated page directory")
        contract = dict(manifest, bundle_sha256=expected, release=release, release_repository=repository)
        old = vendor / "evidence"
        backup = Path(temporary) / "previous"
        if old.exists():
            old.replace(backup)
        try:
            stage.replace(old)
            (root / "contracts").mkdir(exist_ok=True)
            (root / "contracts/evidence.json").write_text(json.dumps(contract, indent=2) + "\n")
        except BaseException:
            if old.exists():
                shutil.rmtree(old)
            if backup.exists():
                backup.replace(old)
            raise
    verify_snapshot(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--release")
    parser.add_argument("--repository", default="relflow/evidence")
    parser.add_argument("--sha256", required=True)
    options = parser.parse_args()
    if options.bundle:
        import_bundle(options.bundle, options.sha256)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    options.release,
                    "--repo",
                    options.repository,
                    "--pattern",
                    "evidence-docs.tar.gz",
                    "--dir",
                    temporary,
                ],
                check=True,
            )
            import_bundle(
                Path(temporary) / "evidence-docs.tar.gz",
                options.sha256,
                repository=options.repository,
                release=options.release,
            )


if __name__ == "__main__":
    main()
