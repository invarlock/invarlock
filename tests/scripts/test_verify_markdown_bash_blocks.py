from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "verify_markdown_bash_blocks.py"
    spec = importlib.util.spec_from_file_location(
        "tests_verify_markdown_bash_blocks", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_bash_blocks_only_keeps_invarlock_blocks(tmp_path: Path) -> None:
    module = _load_script_module()
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "```bash",
                "echo hello",
                "```",
                "```bash",
                "invarlock version",
                "```",
                "```python",
                "print('ignore')",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    blocks = module.extract_bash_blocks([doc])

    assert len(blocks) == 1
    assert blocks[0].file == str(doc)
    assert "invarlock version" in blocks[0].text


def test_sanitize_script_skips_pip_installs() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            'pip install "invarlock[hf]"\n'
            "INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \\\n"
            "  --baseline gpt2 \\\n"
            "python -m pip install foo\n"
        ),
    )

    rendered = module._sanitize_script(block)

    assert "[skip] pip install" in rendered
    assert " -m invarlock evaluate \\" in rendered
    assert "--baseline gpt2 \\" in rendered
    assert "\\ \\" not in rendered
    assert "[skip] python -m pip install foo" in rendered


def test_sanitize_script_host_mode_injects_host_bypass_and_verify_override() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "invarlock evaluate --baseline gpt2 --subject gpt2\n"
            "invarlock verify reports/eval/evaluation.report.json\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" in rendered
    assert "--allow-unattested-artifacts" in rendered


def test_sanitize_script_container_mode_strips_host_bypass_flags() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "INVARLOCK_ALLOW_HOST_EXECUTION=1 invarlock run -c config.yaml\n"
            "invarlock verify --allow-unattested-artifacts reports/eval/evaluation.report.json\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="container")

    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in rendered
    assert "--allow-unattested-artifacts" not in rendered


def test_run_blocks_writes_results(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# test\n", encoding="utf-8")
    (repo_root / "src").mkdir()
    module.ROOT = repo_root
    module.TMP = repo_root / "tmp"

    block = module.BashBlock(
        file=str(repo_root / "README.md"),
        line=1,
        block_index=1,
        text="invarlock version",
    )

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    out_root = tmp_path / "out"
    assert module.run_blocks([block], output_root=out_root) == 0

    records = [
        json.loads(line)
        for line in (out_root / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0]["execution_mode"] == "container"
    assert records[0]["status"] == "ok"
