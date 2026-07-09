import json
import os
import subprocess
from pathlib import Path
from typing import Any

from invarlock.core.config_loader import load_config

EXPECTED_CONFIGS = [
    ("presets/causal_lm", "gpt2_smoke_128.yaml", None),
    ("presets/causal_lm", "wikitext2_512.yaml", None),
    ("calibration", "null_sweep_smoke.yaml", None),
    ("calibration", "rmt_ve_sweep_smoke.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_attn.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_full.yaml", None),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_shell_syntax(path: Path) -> None:
    subprocess.run(["bash", "-n", str(path)], check=True, text=True)


def _assert_smoke_script_ready(script_name: str) -> Path:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "smoke" / script_name
    assert script_path.exists(), f"Expected scripts/smoke/{script_name} to exist"
    assert os.access(script_path, os.X_OK), f"{script_name} should be executable"
    _assert_shell_syntax(script_path)
    return script_path


def _run_smoke_plan(
    script_name: str,
    tmp_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    script_path = _assert_smoke_script_ready(script_name)
    common_path = repo_root / "scripts" / "smoke" / "lib" / "smoke_common.sh"
    _assert_shell_syntax(common_path)

    env = {
        **os.environ,
        "INVARLOCK_SMOKE_PLAN": "1",
        "INVARLOCK_ALLOW_NETWORK": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["bash", str(script_path), str(tmp_path / script_path.stem)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert output_lines, "smoke plan did not emit JSON"
    return json.loads(output_lines[-1])


def test_small_workflow_configs_present() -> None:
    repo_root = _repo_root()
    configs_dir = repo_root / "configs"

    for subdir, primary, fallback in EXPECTED_CONFIGS:
        primary_path = configs_dir / subdir / primary
        cfg_path = primary_path
        if not primary_path.exists() and fallback:
            fb_path = configs_dir / subdir / fallback
            if fb_path.exists():
                cfg_path = fb_path
        assert cfg_path.exists(), f"Expected config {primary} (or fallback) to exist"

        config = load_config(str(cfg_path))
        if "model" in config:
            assert isinstance(config.model.id, str) and len(config.model.id) > 0
        if "dataset" in config:
            assert config.dataset.provider == "wikitext2"
        # For edit configs, verify an edit is specified; task presets may omit edit.
        if subdir.startswith("edits") and "edit" in config:
            assert config.edit.name
        # Presets carry tier context via profile; auto tier may not be set at top-level


def test_gpt2_user_journey_smoke_script_declares_execution_plan(
    tmp_path: Path,
) -> None:
    plan = _run_smoke_plan("run_gpt2_user_journey_smoke.sh", tmp_path)

    assert plan["script"] == "run_gpt2_user_journey_smoke"
    assert plan["mode"] == "all"
    assert plan["profile"] == "dev"
    assert plan["assurance"] == "off"
    assert plan["cache"]["offline_when_cache_complete"] is True
    assert plan["child_suites"] == [
        {"suite": "local", "mode": "local", "assurance": "off"},
        {"suite": "container", "mode": "container", "assurance": "off"},
    ]
    assert {"noop", "quantized", "negative"} <= set(plan["journeys"])
    assert set(plan["commands"]) == {
        "evaluate",
        "verify",
        "report validate",
        "report html",
        "report explain",
        "advanced evidence-pack keygen",
        "advanced evidence-pack build",
        "advanced evidence-pack inspect",
        "advanced evidence-pack verify",
    }
    assert set(plan["helper_contracts"]) == {
        "run_evidence_pack_journey",
        "run_eval_journey",
        "run_negative_journey",
        "write_strict_bundle_fixture",
        "run_strict_bundle_journey",
        "append_child_results",
        "run_child_suite",
        "run_all_mode_journeys",
        "verify-rejects",
    }


def test_tiny_container_smoke_script_declares_runtime_plan(tmp_path: Path) -> None:
    plan = _run_smoke_plan("run_tiny_container_smoke.sh", tmp_path)

    assert plan["script"] == "run_tiny_container_smoke"
    assert plan["model_id"] == "sshleifer/tiny-gpt2"
    assert plan["model_cache_name"] == "models--sshleifer--tiny-gpt2"
    assert plan["mode"] == "container"
    assert plan["runtime_provenance"] == "container"
    assert plan["runtime_image"] == {
        "seed_digest": True,
        "seed_local_image": True,
    }
    assert plan["dataset"] == {
        "provider": {"kind": "local_jsonl"},
        "seq_len": 16,
        "preview_n": 2,
        "final_n": 2,
        "tiny_relax": True,
    }
    assert set(plan["commands"]) == {
        "evaluate",
        "verify",
        "report validate",
        "report html",
        "report explain",
        "advanced evidence-pack keygen",
        "advanced evidence-pack build",
        "advanced evidence-pack inspect",
        "advanced evidence-pack verify",
    }


def test_cli_smoke_fast_declares_repo_python_and_command_groups(
    tmp_path: Path,
) -> None:
    plan = _run_smoke_plan("cli_smoke_fast.sh", tmp_path)

    assert plan["script"] == "cli_smoke_fast"
    assert plan["cli"][1:] == ["-m", "invarlock"]
    assert plan["model_id"] == "sshleifer/tiny-gpt2"
    assert plan["preset"] == "configs/presets/causal_lm/wikitext2_512.yaml"
    assert plan["calibration_configs"] == [
        "configs/calibration/null_sweep_smoke.yaml",
        "configs/calibration/rmt_ve_sweep_smoke.yaml",
    ]
    assert set(plan["command_groups"]) == {
        "help",
        "plugins",
        "fixture_report",
        "report_generation",
        "evidence_pack",
        "policy",
        "offline_evaluate",
        "network_evaluate",
        "calibration",
        "container_local_parity",
    }
    assert set(plan["fixture_contracts"]) == {
        "invarlock.reporting.verify_contract",
        "runtime manifest",
        "evaluation report",
        "run reports",
    }
    assert plan["removed_public_commands"] == ["run"]
    assert set(plan["forbidden_command_surfaces"]) == {
        "report verify --help",
        "invarlock run --help",
        "--source sshleifer/tiny-gpt2",
        "--edited sshleifer/tiny-gpt2",
    }


def test_cli_smoke_negative_declares_failure_category_matrix(
    tmp_path: Path,
) -> None:
    plan = _run_smoke_plan("cli_smoke_negative.sh", tmp_path)

    assert plan["script"] == "cli_smoke_negative"
    cases = {case["label"]: case for case in plan["failure_cases"]}
    assert cases["invarlock run (removed public command)"]["expected_exit"] == "2"
    assert cases["invarlock run (removed public command)"]["expected_fragment"] == (
        "No such command 'run'"
    )
    assert cases["invarlock verify --json (malformed fixture)"] == {
        "label": "invarlock verify --json (malformed fixture)",
        "expected_exit": "2",
        "expected_reason": "malformed",
        "expected_fragment": '"code": "E601"',
    }
    policy_fail_labels = {
        "invarlock verify --json (primary metric policy fail)",
        "invarlock verify --json (invariants policy fail)",
        "invarlock verify --json (spectral policy fail)",
        "invarlock verify --json (rmt policy fail)",
    }
    assert {
        label
        for label in policy_fail_labels
        if cases[label]["expected_reason"] == "policy_fail"
        and cases[label]["expected_exit"] == "3"
    } == policy_fail_labels
    assert (
        cases["invarlock report generate (failed subject run report)"][
            "expected_fragment"
        ]
        == "subject run report with status"
    )
    assert (
        cases["invarlock advanced calibrate null-sweep (missing config)"][
            "expected_fragment"
        ]
        == "Invalid value for '--config'"
    )


def test_cli_smoke_matrix_dispatches_lane_matrix(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = _assert_smoke_script_ready("cli_smoke_matrix.sh")
    work_root = tmp_path / "matrix-unknown-lane-contract"
    env = {**os.environ, "INVARLOCK_SMOKE_LANES": "unknown-lane"}
    result = subprocess.run(
        ["bash", str(script_path), str(work_root)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "unknown smoke lane: unknown-lane" in output
    assert "lanes=unknown-lane" in output
    assert "[summary]" not in output


def test_run_cpu_telemetry_declares_container_cpu_report_plan(
    tmp_path: Path,
) -> None:
    plan = _run_smoke_plan("run_cpu_telemetry.sh", tmp_path)

    assert plan["script"] == "run_cpu_telemetry"
    assert plan["model_id"] == "sshleifer/tiny-gpt2"
    assert plan["profile"] == "ci_cpu"
    assert plan["tier"] == "balanced"
    assert plan["preset"] == "configs/presets/causal_lm/wikitext2_512.yaml"
    assert plan["edit_config"] == "configs/overlays/edits/quant_rtn/8bit_attn.yaml"
    assert plan["runtime_image"] == {"mode": "container", "device": "cpu"}
    assert plan["evaluate"] == {
        "allow_network": True,
        "assurance": "off",
        "device": "cpu",
        "baseline_adapter": "auto",
        "subject_adapter": "auto",
    }
    assert plan["post_checks"] == ["report validate"]
