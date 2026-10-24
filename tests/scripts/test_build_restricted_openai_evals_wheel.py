from __future__ import annotations

import hashlib
import importlib.util
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/security/build_restricted_openai_evals_wheel.py"


@pytest.mark.parametrize("from_checkout", [False, True])
def test_executable_cli_rejects_unauthenticated_input(
    tmp_path: Path, from_checkout: bool
) -> None:
    source = tmp_path / "untrusted.whl"
    source.write_bytes(b"not the pinned upstream wheel")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build-wheel",
            "--input",
            str(source),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT if from_checkout else tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "SHA-256 changed" in result.stderr
    assert not output.exists()


def _load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("restricted_evals", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(module, path: Path) -> dict[str, bytes]:
    from build_cache_free_lm_eval_wheel import _record

    files = {
        "evals/elsuite/basic/match.py": b"class Match: pass\n",
        "evals/record.py": b"class Recorder: pass\n",
        "evals/unsupported_task.py": b"import nltk\n",
        f"{module.UPSTREAM_DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.4\nName: evals\nVersion: 3.0.1.post1\n"
            b"Requires-Dist: nltk\nRequires-Dist: snowflake-connector-python[pandas]\n"
            b"Requires-Dist: openai\n\n"
        ),
        f"{module.UPSTREAM_DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    record = f"{module.UPSTREAM_DIST_INFO}/RECORD"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(record, _record(files, record))
    return files


def test_derivation_authenticates_source_and_preserves_all_upstream_code(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(monkeypatch)
    source = tmp_path / "upstream.whl"
    files = _source(module, source)
    with pytest.raises(module.DerivationError, match="SHA-256"):
        module.build_wheel(source, tmp_path / "rejected")
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    first = module.build_wheel(source, tmp_path / "one")
    second = module.build_wheel(source, tmp_path / "two")
    assert first.name == "evals-3.0.1.post1+invarlock.match.2-py3-none-any.whl"
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        module.validate_wheel_record(archive)
        for name, payload in files.items():
            if name.endswith(".py"):
                assert archive.read(name) == payload
        metadata = archive.read(f"{module.DERIVED_DIST_INFO}/METADATA")
        assert b"Requires-Dist: nltk" not in metadata
        assert b"Requires-Dist: snowflake-connector-python" not in metadata
        assert b"Requires-Dist: openai\n" in metadata
        assert b"Version: 3.0.1.post1+invarlock.match.2\n" in metadata


def test_openai_lock_removes_only_replaced_wheel_and_rejects_unsupported_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(monkeypatch)
    source = tmp_path / "full.txt"
    source.write_text(f"evals=={module.DERIVED_VERSION}\nopenai==2.40.1\n")
    output = tmp_path / "filtered.txt"
    assert (
        module.main(["filter-lock", "--input", str(source), "--output", str(output)])
        == 0
    )
    assert output.read_text() == "openai==2.40.1\n"
    source.write_text("openai==2.40.1\n")
    assert (
        module.main(["filter-lock", "--input", str(source), "--output", str(output)])
        == 2
    )
    for unsupported in ("nltk", "snowflake-connector-python"):
        source.write_text(f"evals=={module.DERIVED_VERSION}\n{unsupported}==1.0\n")
        with pytest.raises(module.DerivationError, match="unsupported dependency"):
            module.filter_lock(source, output)
    assert output.read_text() == "openai==2.40.1\n"


@pytest.mark.parametrize("download", (False, True))
def test_bootstrap_authenticates_source_and_existing_derived_bytes(
    tmp_path, monkeypatch, download
):
    module = _load(monkeypatch)
    source = tmp_path / "upstream.whl"
    _source(module, source)
    payload = source.read_bytes()
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    output = tmp_path / "wheels"
    kwargs = {} if download else {"source": source}
    arguments = ["bootstrap", "--output-directory", str(output)]
    if not download:
        arguments.extend(["--input", str(source)])
    assert module.main(arguments) == 0
    wheel = output / module.DERIVED_WHEEL_NAME
    assert wheel.is_file()
    assert module.bootstrap(output, **kwargs) == wheel
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(module.DerivationError, match="bytes changed"):
        module.bootstrap(output, **kwargs)
    assert not list(output.glob(".evals-build-*"))


@pytest.mark.parametrize("destination_kind", ("symlink", "directory"))
def test_bootstrap_rejects_non_regular_destination_without_overwriting(
    tmp_path, monkeypatch, destination_kind
):
    module = _load(monkeypatch)
    source = tmp_path / "upstream.whl"
    _source(module, source)
    payload = source.read_bytes()
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(payload).hexdigest()
    )
    output = tmp_path / "wheels"
    output.mkdir()
    destination = output / module.DERIVED_WHEEL_NAME
    if destination_kind == "symlink":
        destination.symlink_to(source)
    else:
        destination.mkdir()
    with pytest.raises(module.DerivationError, match="regular file"):
        module.bootstrap(output, source=source)
    assert source.read_bytes() == payload
    assert destination.is_symlink() or destination.is_dir()
    assert not list(output.glob(".evals-build-*"))


def test_bootstrap_rejects_bad_download_and_symlink_cache(tmp_path, monkeypatch):
    module = _load(monkeypatch)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"untrusted")
    )
    with pytest.raises(module.DerivationError, match="downloaded"):
        module.bootstrap(tmp_path)
    source = tmp_path / ".upstream" / module.UPSTREAM_WHEEL_NAME
    assert not source.exists()
    source.symlink_to(tmp_path / "missing")
    with pytest.raises(module.DerivationError, match="symlink"):
        module.bootstrap(tmp_path)


def test_openai_metadata_drift_and_cli_build_errors_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(monkeypatch)
    with pytest.raises(module.DerivationError, match="dependency"):
        module._patch_metadata(b"Version: 3.0.1.post1\nRequires-Dist: nltk>=3\n")
    assert (
        module.main(
            [
                "build-wheel",
                "--input",
                str(tmp_path / "absent"),
                "--output-directory",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )
    built = tmp_path / "derived.whl"
    monkeypatch.setattr(module, "build_wheel", lambda source, output: built)
    assert (
        module.main(
            ["build-wheel", "--input", "source", "--output-directory", "output"]
        )
        == 0
    )
