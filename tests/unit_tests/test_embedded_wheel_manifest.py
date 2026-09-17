"""The CI source wheel must not silently fall back to a released binary."""

import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "wheel_manifest", Path(__file__).parents[2] / ".github/scripts/embedded_wheel_manifest.py"
)
wheel_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_manifest)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    pin = {"seekdb_sha": "a" * 40, "bindings_sha": "b" * 40, "fix_commit": "c" * 40}
    pin_path = tmp_path / "source.json"
    pin_path.write_text(json.dumps(pin))
    binary = tmp_path / "seekdb"
    binary.write_bytes(b"the fixed binary")
    wheel = tmp_path / "pylibseekdb-0.1.0-cp311-cp311-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pylibseekdb/seekdb", binary.read_bytes())
    (tmp_path / "seekdb-version.txt").write_text(f"REVISION: {pin['seekdb_sha']}\n")
    monkeypatch.setattr(wheel_manifest.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wheel_manifest.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        wheel_manifest,
        "sys",
        SimpleNamespace(version_info=(3, 11), argv=["manifest", "create", str(pin_path), str(tmp_path), str(binary)]),
    )
    wheel_manifest.main()
    return pin, tmp_path, binary, wheel


def test_exact_wheel_and_revision_pass(bundle):
    pin, directory, binary, _ = bundle
    assert wheel_manifest.verify(pin, directory)["binary_sha256"] == wheel_manifest.digest(binary)


def test_wrong_source_cannot_reuse_cache(bundle):
    pin, directory, _, _ = bundle
    with pytest.raises(ValueError, match="source identity"):
        wheel_manifest.verify({**pin, "seekdb_sha": "d" * 40}, directory)


def test_corrupt_wheel_is_rejected(bundle):
    pin, directory, _, wheel = bundle
    with wheel.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        wheel_manifest.verify(pin, directory)


def test_replaced_binary_is_rejected_even_with_updated_wheel_hash(bundle):
    pin, directory, _, wheel = bundle
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pylibseekdb/seekdb", b"old released binary")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["wheel_sha256"] = wheel_manifest.digest(wheel)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different SeekDB"):
        wheel_manifest.verify(pin, directory)


def test_short_revision_is_not_enough(bundle):
    pin, directory, _, _ = bundle
    (directory / "seekdb-version.txt").write_text(f"REVISION: {pin['seekdb_sha'][:8]}")
    with pytest.raises(ValueError, match="full REVISION"):
        wheel_manifest.verify(pin, directory)


def test_dependency_resync_is_detected(bundle, monkeypatch):
    _, directory, binary, _ = bundle
    binary.write_bytes(b"old released binary installed by uv sync")
    monkeypatch.setattr(
        wheel_manifest.importlib.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: binary)
    )
    wheel_manifest.sys.argv = ["manifest", "installed", str(directory / "source.json"), str(directory)]
    with pytest.raises(ValueError, match="dependency resync"):
        wheel_manifest.main()


def test_incompatible_host_is_rejected(bundle, monkeypatch):
    pin, directory, _, _ = bundle
    monkeypatch.setattr(wheel_manifest.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="Linux x86_64"):
        wheel_manifest.verify(pin, directory)
