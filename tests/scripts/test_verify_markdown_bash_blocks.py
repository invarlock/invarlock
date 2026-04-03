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


def test_iter_markdown_files_excludes_top_level_hidden_docs(tmp_path: Path) -> None:
    module = _load_script_module()
    (tmp_path / ".private" / "plans").mkdir(parents=True)
    (tmp_path / ".private" / "plans" / "internal.md").write_text(
        "```bash\ninvarlock version\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "public.md").write_text(
        "```bash\ninvarlock version\n```\n",
        encoding="utf-8",
    )

    files = module.iter_markdown_files(tmp_path)

    assert files == [tmp_path / "docs" / "public.md"]


def test_sanitize_script_skips_pip_installs() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            'pip install "invarlock[hf]"\n'
            "invarlock evaluate --allow-network \\\n"
            "  --baseline gpt2 \\\n"
            "python -m pip install foo\n"
        ),
    )

    rendered = module._sanitize_script(block)

    assert "[skip] pip install" in rendered
    assert " -m invarlock evaluate --allow-network \\" in rendered
    assert "--baseline gpt2 \\" in rendered
    assert "\\ \\" not in rendered
    assert "[skip] python -m pip install foo" in rendered


def test_sanitize_script_host_mode_injects_trusted_local_assurance() -> None:
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

    assert "--assurance trusted-local" in rendered
    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in rendered
    assert rendered.count("--assurance trusted-local") == 2


def test_sanitize_script_host_mode_marks_advanced_calibrate_for_host_execution() -> (
    None
):
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/calibration.md",
        line=1,
        block_index=1,
        text=(
            "invarlock advanced calibrate null-sweep \\\n"
            "  --config configs/calibration/null_sweep_ci.yaml\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" in rendered
    assert "-m invarlock advanced calibrate null-sweep" in rendered


def test_sanitize_script_host_mode_skips_container_only_lines() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "make runtime-image\n"
            "test -f reports/eval/runtime.manifest.json\n"
            "docker ps\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "[skip-host] make runtime-image" in rendered
    assert "[skip-host] test -f reports/eval/runtime.manifest.json" in rendered
    assert "[skip-host] docker ps" in rendered


def test_sanitize_script_container_mode_strips_host_bypass_flags() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "INVARLOCK_ALLOW_HOST_EXECUTION=1 invarlock run -c config.yaml\n"
            "invarlock verify --assurance trusted-local reports/eval/evaluation.report.json\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="container")

    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in rendered
    assert "--assurance trusted-local" not in rendered


def test_seed_demo_inputs_writes_expected_fixture_files(tmp_path: Path) -> None:
    module = _load_script_module()
    fixture_root = tmp_path / "repo"
    (fixture_root / "tests" / "artifacts" / "golden_runs" / "gpt2").mkdir(parents=True)
    (fixture_root / "tests" / "fixtures" / "runtime_attestation").mkdir(parents=True)
    (
        fixture_root
        / "tests"
        / "artifacts"
        / "golden_runs"
        / "gpt2"
        / "evaluation.report.json"
    ).write_text(
        '{"schema_version":"v1"}\n',
        encoding="utf-8",
    )
    (
        fixture_root
        / "tests"
        / "fixtures"
        / "runtime_attestation"
        / "runtime.manifest.json"
    ).write_text(
        '{"format_version":"runtime-manifest-v1"}\n',
        encoding="utf-8",
    )
    module.ROOT = fixture_root
    module.DEMO_EVALUATION_REPORT_FIXTURE = (
        fixture_root
        / "tests"
        / "artifacts"
        / "golden_runs"
        / "gpt2"
        / "evaluation.report.json"
    )
    module.DEMO_RUNTIME_MANIFEST_FIXTURE = (
        fixture_root
        / "tests"
        / "fixtures"
        / "runtime_attestation"
        / "runtime.manifest.json"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    module._seed_demo_inputs(workspace)

    assert (workspace / "reports" / "eval" / "evaluation.report.json").is_file()
    assert (workspace / "report_bundle" / "evaluation.report.json").is_file()
    assert (workspace / "reports" / "eval" / "runtime.manifest.json").is_file()
    assert (workspace / "runs" / "subject" / "report.json").is_file()
    assert (workspace / "resolved_policy.json").is_file()
    assert (workspace / "compatibility.json").is_file()


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
