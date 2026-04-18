from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


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


def test_sanitize_script_host_mode_injects_execution_and_verify_modes() -> None:
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

    assert "--execution-mode host" in rendered
    assert "--runtime-provenance host" in rendered
    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in rendered
    assert rendered.count("--execution-mode host") == 1
    assert rendered.count("--runtime-provenance host") == 1


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
            "make runtime-image-podman\n"
            "make runtime-smoke-podman\n"
            "test -f reports/eval/runtime.manifest.json\n"
            "docker ps\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "[skip-host] make runtime-image" in rendered
    assert "[skip-host] make runtime-image-podman" in rendered
    assert "[skip-host] make runtime-smoke-podman" in rendered
    assert "[skip-host] test -f reports/eval/runtime.manifest.json" in rendered
    assert "[skip-host] docker ps" in rendered


def test_sanitize_script_host_mode_skips_mps_commands_when_unavailable(
    monkeypatch,
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_host_supports_mps", lambda: False)
    block = module.BashBlock(
        file="docs/reference/device-drift-bands.md",
        line=1,
        block_index=1,
        text="invarlock evaluate --baseline gpt2 --subject gpt2 --device mps\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "[skip-host]" in rendered
    assert "--device mps" in rendered


def test_sanitize_script_host_mode_keeps_mps_commands_when_available(
    monkeypatch,
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_host_supports_mps", lambda: True)
    block = module.BashBlock(
        file="docs/reference/device-drift-bands.md",
        line=1,
        block_index=1,
        text="invarlock evaluate --baseline gpt2 --subject gpt2 --device mps\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "[skip-host]" not in rendered
    assert "--device mps" in rendered


def test_sanitize_script_host_mode_skips_full_multiline_mps_command(
    monkeypatch,
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_host_supports_mps", lambda: False)
    block = module.BashBlock(
        file="docs/reference/device-drift-bands.md",
        line=1,
        block_index=1,
        text=(
            "invarlock evaluate --baseline gpt2 --subject gpt2 \\\n"
            "  --device mps \\\n"
            "  --profile dev\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "[skip-host]" in rendered
    assert "--profile dev" not in rendered


def test_sanitize_script_rewrites_python_script_invocations_to_selected_python() -> (
    None
):
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/device-drift-bands.md",
        line=1,
        block_index=1,
        text="python scripts/check_device_drift.py reports/a.json reports/b.json\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert rendered.startswith(str(module.ROOT / ".venv" / "bin" / "python"))
    assert "scripts/check_device_drift.py reports/a.json reports/b.json" in rendered


def test_sanitize_script_adds_force_to_report_html() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/cli.md",
        line=1,
        block_index=1,
        text="invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "-m invarlock report html --force" in rendered


def test_sanitize_script_container_mode_strips_host_bypass_flags() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "INVARLOCK_ALLOW_HOST_EXECUTION=1 invarlock run -c config.yaml\n"
            "invarlock verify --runtime-provenance host reports/eval/evaluation.report.json\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="container")

    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in rendered
    assert "--execution-mode host" not in rendered


def test_sanitize_script_skip_model_loading_skips_full_multiline_command() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "invarlock evaluate --allow-network \\\n"
            "  --baseline gpt2 \\\n"
            "  --subject distilgpt2\n"
            "invarlock verify reports/eval/evaluation.report.json\n"
        ),
    )

    rendered = module._sanitize_script(block, skip_model_loading=True)

    assert "[skip-model-loading] invarlock evaluate --allow-network \\" in rendered
    assert "--baseline gpt2" not in rendered
    assert "--subject distilgpt2" not in rendered
    assert "-m invarlock verify reports/eval/evaluation.report.json" in rendered


def test_sanitize_script_host_mode_rewrites_heavy_evaluate_inputs_to_smoke_assets() -> (
    None
):
    module = _load_script_module()
    block = module.BashBlock(
        file="README.md",
        line=1,
        block_index=1,
        text=(
            "invarlock evaluate --allow-network \\\n"
            "  --baseline gpt2 \\\n"
            "  --subject distilgpt2 \\\n"
            "  --preset configs/presets/causal_lm/wikitext2_512.yaml\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "--baseline sshleifer/tiny-gpt2 \\" in rendered
    assert "--subject sshleifer/tiny-gpt2 \\" in rendered
    assert "configs/presets/causal_lm/gpt2_smoke_128.yaml" in rendered
    assert "--baseline gpt2" not in rendered
    assert "--subject distilgpt2" not in rendered


def test_sanitize_script_host_mode_injects_smoke_preset_when_missing() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/env-vars.md",
        line=1,
        block_index=1,
        text="invarlock evaluate --baseline gpt2 --subject gpt2\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "--profile dev" in rendered
    assert "--preset configs/presets/causal_lm/gpt2_smoke_128.yaml" in rendered
    assert "--baseline sshleifer/tiny-gpt2" in rendered
    assert "--subject sshleifer/tiny-gpt2" in rendered


def test_sanitize_script_host_mode_rewrites_calibration_configs_to_smoke_variants() -> (
    None
):
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/calibration.md",
        line=1,
        block_index=1,
        text=(
            "invarlock advanced calibrate null-sweep \\\n"
            "  --config configs/calibration/null_sweep_ci.yaml \\\n"
            "  --out reports/calibration/null_sweep\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "configs/calibration/null_sweep_smoke.yaml" in rendered
    assert "configs/calibration/null_sweep_ci.yaml" not in rendered


def test_sanitize_script_host_mode_rewrites_profiles_and_seed_counts_for_smoke() -> (
    None
):
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/reference/calibration.md",
        line=1,
        block_index=1,
        text=(
            "invarlock advanced calibrate null-sweep \\\n"
            "  --config configs/calibration/null_sweep_ci.yaml \\\n"
            "  --profile release \\\n"
            "  --n-seeds 10 \\\n"
            "  --out reports/calibration/null_sweep\n"
        ),
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "--profile dev" in rendered
    assert "--n-seeds 1" in rendered
    assert "--profile release" not in rendered
    assert "--n-seeds 10" not in rendered


def test_sanitize_script_help_commands_do_not_receive_smoke_profile_injection() -> None:
    module = _load_script_module()
    block = module.BashBlock(
        file="docs/README.md",
        line=1,
        block_index=1,
        text="invarlock advanced calibrate --help\n",
    )

    rendered = module._sanitize_script(block, execution_mode="host")

    assert "--profile dev" not in rendered


def test_seed_demo_inputs_writes_expected_fixture_files(tmp_path: Path) -> None:
    module = _load_script_module()
    fixture_root = tmp_path / "repo"
    (fixture_root / "tests" / "artifacts" / "golden_runs" / "gpt2").mkdir(parents=True)
    (fixture_root / "tests" / "fixtures" / "runtime_provenance").mkdir(parents=True)
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
        / "runtime_provenance"
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
        / "runtime_provenance"
        / "runtime.manifest.json"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    module._seed_demo_inputs(workspace)

    assert (workspace / "reports" / "eval" / "evaluation.report.json").is_file()
    assert (workspace / "report_bundle" / "evaluation.report.json").is_file()
    assert (
        workspace / "reports" / "baseline_calib" / "evaluation.report.json"
    ).is_file()
    assert (workspace / "reports" / "baseline_cpu" / "evaluation.report.json").is_file()
    assert (workspace / "reports" / "baseline_mps" / "evaluation.report.json").is_file()
    assert (workspace / "reports" / "eval" / "runtime.manifest.json").is_file()
    assert (workspace / "runs" / "baseline" / "report.json").is_file()
    assert (workspace / "runs" / "subject" / "report.json").is_file()
    assert (
        workspace / "runs" / "baseline_calib" / "source" / "demo" / "report.json"
    ).is_file()
    assert (workspace / "resolved_policy.json").is_file()
    assert (workspace / "compatibility.json").is_file()


def test_seed_demo_inputs_writes_self_consistent_demo_report(tmp_path: Path) -> None:
    module = _load_script_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    module._seed_demo_inputs(workspace)

    report = json.loads(
        (workspace / "reports" / "eval" / "evaluation.report.json").read_text(
            encoding="utf-8"
        )
    )
    final_logs = report["evaluation_windows"]["final"]["logloss"]
    final_counts = report["evaluation_windows"]["final"]["token_counts"]
    weighted_final = sum(
        float(logloss) * int(count)
        for logloss, count in zip(final_logs, final_counts, strict=False)
    ) / sum(int(count) for count in final_counts)

    assert report["primary_metric"]["final"] == pytest.approx(math.exp(weighted_final))
    assert report["baseline_ref"]["primary_metric"]["final"] == pytest.approx(
        math.exp(2.30)
    )
    assert report["primary_metric"]["ratio_vs_baseline"] == pytest.approx(1.0)


def test_prepare_workspace_stages_lightweight_repo_view(tmp_path: Path) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for dirname in (".github", "src", "scripts", "configs", "runtime", "tests"):
        (repo_root / dirname).mkdir()
        (repo_root / dirname / "marker.txt").write_text(dirname, encoding="utf-8")
    (repo_root / "README.md").write_text("# repo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='demo'\n", encoding="utf-8"
    )
    (repo_root / "tmp").mkdir()
    (repo_root / "tmp" / "generated.txt").write_text("generated\n", encoding="utf-8")
    (repo_root / "data").mkdir()
    (repo_root / "data" / "large.bin").write_text("payload\n", encoding="utf-8")
    (repo_root / ".git").mkdir()
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    module.ROOT = repo_root

    workspace = tmp_path / "workspace"
    module._prepare_workspace(workspace)

    assert (workspace / ".github").is_symlink()
    assert (workspace / "src").is_symlink()
    assert (workspace / "scripts").is_symlink()
    assert (workspace / "configs").is_symlink()
    assert (workspace / "runtime").is_symlink()
    assert (workspace / "tests").is_symlink()
    assert (workspace / "README.md").is_symlink()
    assert (workspace / "pyproject.toml").is_symlink()
    assert not (workspace / "tmp").exists()
    assert not (workspace / "data").exists()
    assert not (workspace / ".git").exists()
    assert (workspace / "src" / "marker.txt").read_text(encoding="utf-8") == "src"


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

    def _fake_run_logged_script(**kwargs):
        log_path = kwargs["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return 0, "ok\n"

    monkeypatch.setattr(module, "_run_logged_script", _fake_run_logged_script)

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
    assert records[0]["stdout"] == "ok\n"
    assert records[0]["stderr"] == ""


def test_run_blocks_clears_stale_output_root(tmp_path: Path, monkeypatch) -> None:
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

    def _fake_run_logged_script(**kwargs):
        log_path = kwargs["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return 0, "ok\n"

    monkeypatch.setattr(module, "_run_logged_script", _fake_run_logged_script)

    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "results.jsonl").write_text("garbage\n", encoding="utf-8")
    (out_root / "logs").mkdir()
    (out_root / "logs" / "stale.log").write_text("stale\n", encoding="utf-8")

    assert module.run_blocks([block], output_root=out_root) == 0

    records = [
        json.loads(line)
        for line in (out_root / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0]["id"] == "001-01"
    assert not (out_root / "logs" / "stale.log").exists()


def test_remove_tree_retries_transient_directory_not_empty(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    target = tmp_path / "workspaces"
    target.mkdir()
    (target / "stale.txt").write_text("x\n", encoding="utf-8")
    original_rmtree = module.shutil.rmtree
    calls = {"count": 0}

    def _flaky_rmtree(path: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(66, "Directory not empty")
        original_rmtree(path)

    monkeypatch.setattr(module.shutil, "rmtree", _flaky_rmtree)

    module._remove_tree(target)

    assert calls["count"] == 2
    assert not target.exists()


def test_run_logged_script_streams_output_to_log_and_console(
    tmp_path: Path, capsys
) -> None:
    module = _load_script_module()
    script = tmp_path / "emit.sh"
    script.write_text("printf 'alpha\\n'; printf 'beta\\n'\n", encoding="utf-8")
    log_path = tmp_path / "logs" / "emit.log"

    returncode, output_tail = module._run_logged_script(
        cmd=["bash", str(script)],
        cwd=tmp_path,
        env=module._default_env(tmp_path),
        log_path=log_path,
        label="demo-block",
    )

    assert returncode == 0
    assert output_tail == "alpha\nbeta\n"
    assert log_path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    captured = capsys.readouterr()
    assert "[markdown-live] Running demo-block" in captured.out
    assert "alpha" in captured.out
    assert "beta" in captured.out
    assert "[markdown-live] Finished demo-block rc=0" in captured.out
