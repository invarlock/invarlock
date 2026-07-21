from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "security" / "build_cache_free_lm_eval_wheel.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_cache_free_lm_eval_wheel", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(files: dict[str, bytes], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, payload in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
        writer.writerow((name, f"sha256={digest.rstrip('=')}", len(payload)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode()


def _source_wheel(path: Path, *, metadata: bytes | None = None) -> bytes:
    root = "lm_eval-0.4.12.dist-info"
    record_name = f"{root}/RECORD"
    files = {
        "lm_eval/__init__.py": b"",
        "lm_eval/api/model.py": (
            b"from typing import TYPE_CHECKING, Any\n\n"
            b"if TYPE_CHECKING:\n"
            b"    from sqlitedict import SqliteDict\n\n"
            b"    from lm_eval.api.instance import Instance\n\n"
            b"class CacheHook:\n"
            b"    def __init__(self):\n"
            b"        self.dbdict: SqliteDict | None = None\n\n"
            b"class CachingLM:\n"
            b"    def __init__(self, lm: LM, cache_db: str) -> None:\n"
            b'        """LM wrapper that returns cached results when available, falling back to the underlying model.\n\n'
            b"        Args:\n"
            b"            lm: The underlying language model to wrap.\n"
            b"            cache_db: Path to the SQLite cache database.\n"
            b'        """\n'
            b"        from sqlitedict import SqliteDict\n\n"
            b"        self.lm: LM = lm\n"
            b"        self.cache_db: str = cache_db\n"
            b"        if os.path.dirname(cache_db):\n"
            b"            os.makedirs(os.path.dirname(cache_db), exist_ok=True)\n"
            b"        self.dbdict = SqliteDict(cache_db, autocommit=True)\n\n"
            b"        # add hook to lm\n"
            b"        lm.set_cache_hook(self.get_cache_hook())\n"
        ),
        f"{root}/METADATA": metadata
        or (
            b"Metadata-Version: 2.4\n"
            b"Name: lm_eval\n"
            b"Version: 0.4.12\n"
            b"Requires-Dist: datasets>=2.16.0\n"
            b"Requires-Dist: sqlitedict\n"
            b'Requires-Dist: torch>=1.8; extra == "hf"\n\n'
        ),
        f"{root}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    files[record_name] = _record(files, record_name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path.read_bytes()


def _write_wheel(
    path: Path,
    files: dict[str, bytes],
    *,
    record_name: str = "package-1.0.dist-info/RECORD",
    record: bytes | None = None,
) -> Path:
    contents = dict(files)
    contents[record_name] = (
        record if record is not None else _record(files, record_name)
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
    return path


def test_builds_deterministic_cache_free_wheel(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    source = tmp_path / "lm_eval-0.4.12-py3-none-any.whl"
    payload = _source_wheel(source)
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(payload).hexdigest()
    )

    first = module.build_wheel(source, tmp_path / "one")
    second = module.build_wheel(source, tmp_path / "two")

    assert first.name == "lm_eval-0.4.12+invarlock.nocache.1-py3-none-any.whl"
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        metadata = archive.read("lm_eval-0.4.12+invarlock.nocache.1.dist-info/METADATA")
        model = archive.read("lm_eval/api/model.py")
        module.validate_wheel_record(archive)
    assert names[-1].endswith(".dist-info/RECORD")
    assert b"Version: 0.4.12+invarlock.nocache.1\n" in metadata
    assert b"sqlitedict" not in metadata
    assert b"sqlitedict" not in model.lower()
    assert b"response caching is unavailable" in model


def test_rejects_wrong_upstream_digest(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "lm_eval-0.4.12-py3-none-any.whl"
    _source_wheel(source)

    with pytest.raises(module.DerivationError, match="SHA-256"):
        module.build_wheel(source, tmp_path / "output")


def test_rejects_changed_upstream_patch_surface(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    source = tmp_path / "lm_eval-0.4.12-py3-none-any.whl"
    payload = _source_wheel(
        source,
        metadata=b"Metadata-Version: 2.4\nName: lm_eval\nVersion: 0.4.12\n\n",
    )
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(payload).hexdigest()
    )

    with pytest.raises(module.DerivationError, match="metadata dependency"):
        module.build_wheel(source, tmp_path / "output")


def test_record_validation_rejects_unsafe_and_malformed_wheels(
    tmp_path: Path,
) -> None:
    module = _load_module()
    unsafe = _write_wheel(tmp_path / "unsafe.whl", {"../escape": b"x"})
    missing = tmp_path / "missing.whl"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("package.py", b"x")
    unreadable = _write_wheel(
        tmp_path / "unreadable.whl", {"package.py": b"x"}, record=b"\xff"
    )
    bad_row = _write_wheel(
        tmp_path / "bad-row.whl", {"package.py": b"x"}, record=b"only,two\n"
    )
    incomplete = _write_wheel(
        tmp_path / "incomplete.whl",
        {"package.py": b"x"},
        record=b"package-1.0.dist-info/RECORD,,\n",
    )
    file_digest = base64.urlsafe_b64encode(hashlib.sha256(b"x").digest()).decode()
    self_hashed = _write_wheel(
        tmp_path / "self-hashed.whl",
        {"package.py": b"x"},
        record=(
            f"package.py,sha256={file_digest.rstrip('=')},1\n"
            "package-1.0.dist-info/RECORD,sha256=invalid,1\n"
        ).encode(),
    )
    mismatched = _write_wheel(
        tmp_path / "mismatched.whl",
        {"package.py": b"x"},
        record=(b"package.py,sha256=invalid,99\npackage-1.0.dist-info/RECORD,,\n"),
    )

    cases = (
        (unsafe, "duplicated or unsafe"),
        (missing, "exactly one RECORD"),
        (unreadable, "RECORD is unreadable"),
        (bad_row, "invalid row"),
        (incomplete, "does not cover"),
        (self_hashed, "must not hash itself"),
        (mismatched, "does not match"),
    )
    for wheel, message in cases:
        with zipfile.ZipFile(wheel) as archive:
            with pytest.raises(module.DerivationError, match=message):
                module.validate_wheel_record(archive)


def test_patch_surfaces_fail_closed_on_upstream_drift() -> None:
    module = _load_module()
    metadata = b"Version: 9.9.9\nRequires-Dist: sqlitedict\n"
    with pytest.raises(module.DerivationError, match="metadata version"):
        module._patch_metadata(metadata)

    model = (
        b"if TYPE_CHECKING:\n"
        b"    from sqlitedict import SqliteDict\n\n"
        b"    from lm_eval.api.instance import Instance\n"
        b"class CachingLM:\n"
        b"    def __init__(self, lm: LM, cache_db: str) -> None:\n"
        b"        self.dbdict: SqliteDict | None = None\n"
        b"        lm.set_cache_hook(self.get_cache_hook())\n"
    )
    with pytest.raises(module.DerivationError, match="type surface"):
        module._patch_model(model.replace(b"SqliteDict | None", b"object"))
    with pytest.raises(module.DerivationError, match="implementation changed"):
        module._patch_model(model.replace(b"lm.set_cache_hook", b"lm.other_hook"))
    with pytest.raises(module.DerivationError, match="dependency remained"):
        module._patch_model(model + b"# sqlitedict must not remain\n")


def test_build_rejects_unreadable_invalid_incomplete_and_existing_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    with pytest.raises(module.DerivationError, match="unreadable"):
        module.build_wheel(tmp_path / "missing.whl", tmp_path / "output")

    bad_zip = tmp_path / "bad.whl"
    bad_zip.write_bytes(b"not a zip")
    monkeypatch.setattr(
        module,
        "UPSTREAM_WHEEL_SHA256",
        hashlib.sha256(bad_zip.read_bytes()).hexdigest(),
    )
    with pytest.raises(module.DerivationError, match="readable ZIP"):
        module.build_wheel(bad_zip, tmp_path / "bad-output")

    incomplete = _write_wheel(
        tmp_path / "incomplete-source.whl", {"lm_eval/__init__.py": b""}
    )
    monkeypatch.setattr(
        module,
        "UPSTREAM_WHEEL_SHA256",
        hashlib.sha256(incomplete.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(module, "UPSTREAM_DIST_INFO", "package-1.0.dist-info")
    with pytest.raises(module.DerivationError, match="patch surface"):
        module.build_wheel(incomplete, tmp_path / "incomplete-output")

    source = tmp_path / "source.whl"
    payload = _source_wheel(source)
    monkeypatch.setattr(module, "UPSTREAM_DIST_INFO", "lm_eval-0.4.12.dist-info")
    monkeypatch.setattr(
        module, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(payload).hexdigest()
    )
    destination = module.build_wheel(source, tmp_path / "complete-output")
    assert destination.is_file()
    with pytest.raises(module.DerivationError, match="already exists"):
        module.build_wheel(source, tmp_path / "complete-output")


def test_filter_lock_removes_only_upstream_and_sqlitedict(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "compiled.txt"
    output = tmp_path / "filtered.txt"
    source.write_text(
        "# generated\n"
        "alpha==1 \\\n+    --hash=sha256:aaa\n"
        "lm-eval==0.4.12 \\\n+    --hash=sha256:bbb\n"
        "    # via input\n"
        "sqlitedict==2.1.0 \\\n+    --hash=sha256:ccc\n"
        "    # via lm-eval\n"
        "torch==2 \\\n+    --hash=sha256:ddd\n",
        encoding="utf-8",
    )

    module.filter_lock(source, output)

    assert output.read_text(encoding="utf-8") == (
        "# generated\n"
        "alpha==1 \\\n+    --hash=sha256:aaa\n"
        "torch==2 \\\n+    --hash=sha256:ddd\n"
    )


@pytest.mark.parametrize(
    "content, message",
    [
        ("lm-eval==0.4.12\nalpha==1\n", "sqlitedict"),
        ("sqlitedict==2.1.0\nalpha==1\n", "lm-eval"),
        (
            "lm-eval==0.4.12\nsqlitedict==2.1.0\nsqlitedict==2.1.0\n",
            "exactly once",
        ),
    ],
)
def test_filter_lock_requires_exact_expected_packages(
    tmp_path: Path, content: str, message: str
) -> None:
    module = _load_module()
    source = tmp_path / "compiled.txt"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(module.DerivationError, match=message):
        module.filter_lock(source, tmp_path / "filtered.txt")


def test_filter_lock_rejects_residual_and_unsafe_outputs(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "compiled.txt"
    source.write_text(
        "# sqlitedict must not survive\nlm-eval==0.4.12\nsqlitedict==2.1.0\nalpha==1\n",
        encoding="utf-8",
    )
    with pytest.raises(module.DerivationError, match="remained"):
        module.filter_lock(source, tmp_path / "residual.txt")

    source.write_text(
        "lm-eval==0.4.12\nsqlitedict==2.1.0\nalpha==1\n", encoding="utf-8"
    )
    directory_output = tmp_path / "directory-output"
    directory_output.mkdir()
    with pytest.raises(module.DerivationError, match="regular file"):
        module.filter_lock(source, directory_output)

    temporary_output = tmp_path / "filtered.txt"
    temporary_output.with_name(".filtered.txt.tmp").write_text("occupied")
    with pytest.raises(module.DerivationError, match="temporary output"):
        module.filter_lock(source, temporary_output)


def test_main_dispatches_both_commands_and_reports_failures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_module()
    built = tmp_path / "derived.whl"
    built.write_bytes(b"wheel")
    calls: list[tuple[str, Path, Path]] = []

    def fake_build(source: Path, output: Path) -> Path:
        calls.append(("build", source, output))
        return built

    def fake_filter(source: Path, output: Path) -> None:
        calls.append(("filter", source, output))

    monkeypatch.setattr(module, "build_wheel", fake_build)
    monkeypatch.setattr(module, "filter_lock", fake_filter)
    assert (
        module.main(
            [
                "build-wheel",
                "--input",
                str(tmp_path / "upstream.whl"),
                "--output-directory",
                str(tmp_path / "wheelhouse"),
            ]
        )
        == 0
    )
    assert str(built) in capsys.readouterr().out
    assert (
        module.main(
            [
                "filter-lock",
                "--input",
                str(tmp_path / "compiled.txt"),
                "--output",
                str(tmp_path / "filtered.txt"),
            ]
        )
        == 0
    )
    assert [call[0] for call in calls] == ["build", "filter"]

    def fail_build(_source: Path, _output: Path) -> Path:
        raise module.DerivationError("rejected")

    monkeypatch.setattr(module, "build_wheel", fail_build)
    assert (
        module.main(
            [
                "build-wheel",
                "--input",
                str(tmp_path / "upstream.whl"),
                "--output-directory",
                str(tmp_path / "wheelhouse"),
            ]
        )
        == 2
    )
    assert "ERROR: rejected" in capsys.readouterr().err
