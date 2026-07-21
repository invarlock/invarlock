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
