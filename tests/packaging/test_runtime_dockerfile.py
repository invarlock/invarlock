from __future__ import annotations

from pathlib import Path


def test_runtime_dockerfile_installs_hf_stack() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    cpu_torch_install = (
        "pip install --index-url https://download.pytorch.org/whl/cpu torch"
    )
    assert "pip install -e /opt/invarlock[hf]" in text
    assert cpu_torch_install in text
    assert text.index(cpu_torch_install) < text.index("pip install -e /opt/invarlock[hf]")
