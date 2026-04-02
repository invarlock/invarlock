from __future__ import annotations

from pathlib import Path


def test_runtime_dockerfile_installs_hf_stack() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    cpu_torch_install = (
        "pip install --index-url https://download.pytorch.org/whl/cpu torch"
    )
    assert "sentencepiece>=0.2.1" in text
    assert "tiktoken>=0.9.0" in text
    assert "datasets>=3.0" in text
    assert "python -m pip install --no-deps -e /opt/invarlock" in text
    assert cpu_torch_install in text
    assert text.index(cpu_torch_install) < text.index(
        "python -m pip install --no-deps -e /opt/invarlock"
    )
