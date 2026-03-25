from __future__ import annotations

from pathlib import Path


def test_runtime_dockerfile_installs_hf_stack() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install -e /opt/invarlock[hf]" in text
