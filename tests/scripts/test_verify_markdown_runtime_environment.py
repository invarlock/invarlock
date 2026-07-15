from __future__ import annotations

import json
from pathlib import Path

from tests.scripts._support_verify_markdown_bash_blocks import _load_script_module


def test_default_env_pins_staged_runtime_fixture_digest(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    fixture = (
        tmp_path / "tests" / "fixtures" / "runtime_provenance" / "runtime.manifest.json"
    )
    fixture.parent.mkdir(parents=True)
    digest = "sha256:" + "a" * 64
    fixture.write_text(
        json.dumps({"runtime": {"image_digest": digest}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("EXPECTED_RUNTIME_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("TRUSTED_RUNTIME_IMAGE_DIGEST", raising=False)

    env = module._default_env(tmp_path)

    assert env["EXPECTED_RUNTIME_IMAGE_DIGEST"] == digest
    assert env["TRUSTED_RUNTIME_IMAGE_DIGEST"] == digest


def test_default_env_preserves_independently_supplied_runtime_digest(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = (
        tmp_path / "tests" / "fixtures" / "runtime_provenance" / "runtime.manifest.json"
    )
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps({"runtime": {"image_digest": "sha256:" + "a" * 64}}),
        encoding="utf-8",
    )
    supplied = "sha256:" + "b" * 64
    monkeypatch.setenv("EXPECTED_RUNTIME_IMAGE_DIGEST", supplied)
    module = _load_script_module()

    env = module._default_env(tmp_path)

    assert env["EXPECTED_RUNTIME_IMAGE_DIGEST"] == supplied
