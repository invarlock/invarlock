import os
from pathlib import Path

from invarlock.core.config_loader import load_config

EXPECTED_CONFIGS = [
    ("presets/causal_lm", "gpt2_smoke_128.yaml", None),
    ("presets/causal_lm", "wikitext2_512.yaml", None),
    ("calibration", "null_sweep_smoke.yaml", None),
    ("calibration", "rmt_ve_sweep_smoke.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_attn.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_full.yaml", None),
]


def test_small_workflow_configs_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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


def test_gpt2_user_journey_smoke_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "run_gpt2_user_journey_smoke.sh"
    assert script_path.exists(), (
        "Expected scripts/smoke/run_gpt2_user_journey_smoke.sh to exist"
    )
    assert os.access(script_path, os.X_OK), (
        "run_gpt2_user_journey_smoke.sh should be executable"
    )
    helper_path = repo_root / "scripts" / "smoke" / "gpt2_journey_helpers.py"
    common_path = repo_root / "scripts" / "smoke" / "lib" / "smoke_common.sh"
    contents = script_path.read_text(encoding="utf-8")
    helper_contents = helper_path.read_text(encoding="utf-8")
    common_contents = common_path.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in contents
    assert 'GPT2_HELPER="$SCRIPT_DIR/gpt2_journey_helpers.py"' in contents
    assert 'smoke_select_python "$REPO_ROOT"' in contents
    assert 'smoke_setup_pythonpath "$REPO_ROOT"' in contents
    assert "smoke_ensure_writable_hf_cache" in contents
    assert "INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT" in contents
    assert 'CLI=("$PYTHON_BIN" -m invarlock)' in contents
    assert "command -v invarlock" not in contents
    assert "INVARLOCK_SMOKE_CACHE_COMPLETE" in contents
    assert 'MODE="${INVARLOCK_SMOKE_MODE:-all}"' in contents
    assert 'PROFILE="dev"' in contents
    assert 'ASSURANCE="${INVARLOCK_SMOKE_ASSURANCE:-}"' in contents
    assert 'EDIT_CONFIG="${INVARLOCK_SMOKE_EDIT_CONFIG:-}"' in contents
    assert (
        'QUANT_EDIT_CONFIG="${INVARLOCK_SMOKE_QUANT_EDIT_CONFIG:-${EDIT_CONFIG:-$DEFAULT_QUANT_EDIT_CONFIG}}"'
        in contents
    )
    assert (
        'CUSTOM_EDIT_CONFIG="${INVARLOCK_SMOKE_CUSTOM_EDIT_CONFIG:-${EDIT_CONFIG:-$DEFAULT_QUANT_EDIT_CONFIG}}"'
        in contents
    )
    assert 'SMOKE_DEVICE="${INVARLOCK_SMOKE_DEVICE:-auto}"' in contents
    assert 'JOURNEYS_RAW="${INVARLOCK_SMOKE_JOURNEYS:-$DEFAULT_JOURNEYS}"' in contents
    assert "INVARLOCK_SMOKE_QUANTIZED" in contents
    assert 'DEFAULT_JOURNEYS="strict-bundle,noop,quantized,edited,negative"' in contents
    assert "write_strict_bundle_fixture" in contents
    assert "run_strict_bundle_journey" in contents
    assert "write_strict_bundle_fixture" in helper_contents
    assert (
        '"${CLI[@]}" verify "$eval_report" --assurance strict --profile ci --json'
        in contents
    )
    assert "run_all_mode_journeys" in contents
    assert "append_child_results" in contents
    assert (
        'if ! append_child_results "$suite" "$child_root"; then\n    rc=1\n  fi'
        in contents
    )
    assert 'run_child_suite "container" "container"' in contents
    assert "INVARLOCK_SMOKE_CONTAINER_PROFILE" in contents
    assert "INVARLOCK_SMOKE_CONTAINER_ASSURANCE" in contents
    assert "INVARLOCK_SMOKE_CONTAINER_JOURNEYS" in contents
    assert 'INVARLOCK_SMOKE_DEVICE="$SMOKE_DEVICE"' in contents
    assert "assurance=$ASSURANCE" in contents
    assert "device=$SMOKE_DEVICE" in contents
    assert '--device "$SMOKE_DEVICE"' in contents
    assert '--assurance "$ASSURANCE"' in contents
    assert 'record_result "$journey/verify-rejects"' in contents
    assert "GPT-2 User Journey Smoke Results" in helper_contents
    assert "journey-results.tsv" in contents
    assert "prefetch_hf_assets_on_host" in contents
    assert "smoke_ensure_current_runtime_image" in contents
    assert 'echo "[smoke] refreshing local container runtime image"' in common_contents
    assert (
        'echo "[smoke] refreshing local CUDA container runtime image"'
        in common_contents
    )
    assert "make runtime-image" in common_contents
    assert "make runtime-image-cuda" in common_contents
    assert (
        'export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"'
        in common_contents
    )
    assert "prefetching GPT-2 + WikiText-2 into host HF cache" in contents
    assert "run_evidence_pack_journey" in contents
    assert "verify rejects mutated report" in contents
    assert (
        '"${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force'
        in contents
    )


def test_tiny_container_smoke_campaign_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "run_tiny_container_smoke.sh"
    assert script_path.exists(), (
        "Expected scripts/smoke/run_tiny_container_smoke.sh to exist"
    )
    assert os.access(script_path, os.X_OK), (
        "run_tiny_container_smoke.sh should be executable"
    )
    common_path = repo_root / "scripts" / "smoke" / "lib" / "smoke_common.sh"
    contents = script_path.read_text(encoding="utf-8")
    common_contents = common_path.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in contents
    assert 'smoke_select_python "$REPO_ROOT"' in contents
    assert 'smoke_setup_pythonpath "$REPO_ROOT"' in contents
    assert "kind: local_jsonl" in contents
    assert "sshleifer/tiny-gpt2" in contents
    assert "tiny_relax: true" in contents
    assert "prefetch_tiny_model_on_host" in contents
    assert "smoke_ensure_current_runtime_image" in contents
    assert 'echo "[smoke] refreshing local container runtime image"' in common_contents
    assert (
        'echo "[smoke] refreshing local CUDA container runtime image"'
        in common_contents
    )
    assert "make runtime-image" in common_contents
    assert "make runtime-image-cuda" in common_contents
    assert "INVARLOCK_RUNTIME_IMAGE_DIGEST" in contents
    assert 'MODEL_CACHE_NAME="models--${MODEL_ID//\\//--}"' in contents
    assert 'MODEL_ID="$MODEL_ID"' in contents
    assert (
        'export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"'
        in common_contents
    )
    assert 'SMOKE_DEVICE="${INVARLOCK_SMOKE_DEVICE:-auto}"' in contents
    assert 'echo "[smoke] device=$SMOKE_DEVICE"' in contents
    assert "runtime_verify_diagnostics" in contents
    assert "assert_semantic_pass" in contents
    assert "resolve_tiny_relax_from_report" in contents
    assert 'VERIFY_RC=0\n"${CLI[@]}" verify' in contents
    assert (
        'EVIDENCE_PACK_VERIFY_RC=0\n"${CLI[@]}" advanced evidence-pack verify'
        in contents
    )
    assert '--profile "$PROFILE" --assurance off --json' in contents
    assert '--device "$SMOKE_DEVICE"' in contents
    assert 'mkdir -p "$SMOKE_EXPORT_DIR"' in contents
    assert (
        '"${CLI[@]}" report html -i "$EVAL_REPORT" -o "$SMOKE_EXPORT_DIR/evaluation.html"'
        in contents
    )
    assert 'advanced evidence-pack keygen "$EVIDENCE_PACK_SIGNING_KEY"' in contents
    assert '--signing-key "$EVIDENCE_PACK_SIGNING_KEY"' in contents
    assert "evaluation report verification failed" in contents
    assert "evidence-pack verification failed" in contents


def test_cli_smoke_fast_uses_repo_selected_python() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "cli_smoke_fast.sh"
    assert script_path.exists(), "Expected scripts/smoke/cli_smoke_fast.sh to exist"
    assert os.access(script_path, os.X_OK), "cli_smoke_fast.sh should be executable"

    contents = script_path.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in contents
    assert 'smoke_select_python "$ROOT"' in contents
    assert 'smoke_setup_pythonpath "$ROOT"' in contents
    assert 'smoke_expected_exit_match "$@"' in contents
    assert (
        'WORK_ROOT="${1:-$(mktemp -d -t invarlock_cli_fast_smoke.XXXXXX.dir)}"'
        in contents
    )
    assert 'TMP_DIR="$WORK_ROOT/tmp"' in contents
    assert "printf -v CLI '%q ' \"$PYTHON_BIN\" -m invarlock" in contents
    assert "\"$PYTHON_BIN\" - <<'PY'" in contents
    assert 'run "invarlock evaluate --help"' in contents
    assert "invarlock run --help" not in contents
    assert (
        'SMOKE_CALIBRATE_NULL_CONFIG="${INVARLOCK_SMOKE_CALIBRATE_NULL_CONFIG:-configs/calibration/null_sweep_smoke.yaml}"'
        in contents
    )
    assert (
        'SMOKE_CALIBRATE_VE_CONFIG="${INVARLOCK_SMOKE_CALIBRATE_VE_CONFIG:-configs/calibration/rmt_ve_sweep_smoke.yaml}"'
        in contents
    )
    assert (
        'run "invarlock report generate --help" "$CLI report generate --help"'
        in contents
    )
    assert 'run "invarlock advanced evidence-pack keygen --help"' in contents
    assert 'run "invarlock doctor --json"' in contents
    assert (
        '--baseline \\"$SMOKE_MODEL_ID\\" --subject \\"$SMOKE_MODEL_ID\\"' in contents
    )
    assert 'run "invarlock report generate (demo run reports)"' in contents
    assert "assert_tiny_eval_parity" in contents
    assert "run_tiny_eval_parity" in contents
    assert contents.count("--profile dev --assurance off --preset") == 3
    assert "unexpected_failures=${UNEXPECTED_FAILURES}" in contents
    assert "else echo '[error] report missing'; exit 1; fi" in contents
    assert "echo '[skip] report missing'" not in contents
    assert "report verify --help" not in contents
    assert "command -v invarlock" not in contents
    assert "evaluate/run" not in contents
    assert "--source sshleifer/tiny-gpt2" not in contents
    assert "--edited sshleifer/tiny-gpt2" not in contents


def test_cli_smoke_negative_exercises_failure_categories() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "cli_smoke_negative.sh"
    assert script_path.exists(), "Expected scripts/smoke/cli_smoke_negative.sh to exist"
    assert os.access(script_path, os.X_OK), "cli_smoke_negative.sh should be executable"

    contents = script_path.read_text(encoding="utf-8")
    assert "tests/artifacts/golden_runs/gpt2/evaluation.report.json" in contents
    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in contents
    assert 'smoke_select_python "$ROOT"' in contents
    assert 'smoke_setup_pythonpath "$ROOT"' in contents
    assert 'smoke_expected_exit_match "$@"' in contents
    assert "invarlock verify --json (malformed fixture)" in contents
    assert "invarlock verify --json (primary metric policy fail)" in contents
    assert "invarlock verify --json (invariants policy fail)" in contents
    assert "invarlock verify --json (spectral policy fail)" in contents
    assert "invarlock verify --json (rmt policy fail)" in contents
    assert "invarlock run (removed public command)" in contents
    assert "invarlock report generate (failed subject run report)" in contents
    assert "invarlock advanced calibrate null-sweep (missing config)" in contents
    assert "summary.reason" in contents
    assert "Invalid value for '--config'" in contents
    assert (
        'assert_contains "$VERIFY_OUT/malformed.out" "\\"code\\": \\"E601\\""'
        in contents
    )


def test_cli_smoke_matrix_dispatches_lane_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "cli_smoke_matrix.sh"
    assert script_path.exists(), "Expected scripts/smoke/cli_smoke_matrix.sh to exist"
    assert os.access(script_path, os.X_OK), "cli_smoke_matrix.sh should be executable"

    contents = script_path.read_text(encoding="utf-8")
    assert 'LANES_RAW="${INVARLOCK_SMOKE_LANES:-fast,negative,realistic}"' in contents
    assert 'script_path="$REPO_ROOT/scripts/smoke/cli_smoke_fast.sh"' in contents
    assert 'script_path="$REPO_ROOT/scripts/smoke/cli_smoke_negative.sh"' in contents
    assert (
        'script_path="$REPO_ROOT/scripts/smoke/run_gpt2_user_journey_smoke.sh"'
        in contents
    )
    assert 'mode="${INVARLOCK_REALISTIC_SMOKE_MODE:-local}"' in contents
    assert 'journeys="${INVARLOCK_REALISTIC_SMOKE_JOURNEYS:-noop,negative}"' in contents
    assert (
        'INVARLOCK_SMOKE_MODE="$mode" INVARLOCK_SMOKE_JOURNEYS="$journeys"' in contents
    )
    assert "lane=realistic exit_code=$rc" in contents
    assert "unknown smoke lane" in contents
    assert "failed=${FAILED_LANES}" in contents


def test_run_cpu_telemetry_uses_repo_selected_python() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "run_cpu_telemetry.sh"
    assert script_path.exists(), "Expected scripts/smoke/run_cpu_telemetry.sh to exist"
    assert os.access(script_path, os.X_OK), "run_cpu_telemetry.sh should be executable"

    contents = script_path.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in contents
    assert 'smoke_select_python "$ROOT"' in contents
    assert 'smoke_setup_pythonpath "$ROOT"' in contents
    assert 'CLI=("$PYTHON_BIN" -m invarlock)' in contents
    assert (
        'PRESET="${PRESET:-configs/presets/causal_lm/wikitext2_512.yaml}"' in contents
    )
    assert 'INVARLOCK_ALLOW_NETWORK=1 "${CLI[@]}" evaluate' in contents
    assert "--assurance off" in contents
    assert 'smoke_ensure_current_runtime_image "container" "cpu"' in contents
    assert "--device cpu" in contents
    assert "EVAL_RC=$?" in contents
    assert 'exit "${EVAL_RC}"' in contents
    assert "using emitted report artifacts" not in contents
    assert '"${CLI[@]}" report validate' in contents
    assert '"${CLI[@]}" verify' not in contents
    assert "command -v invarlock" not in contents


def test_cli_smoke_fast_uses_reporting_verify_contract_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke" / "cli_smoke_fast.sh"
    contents = script_path.read_text(encoding="utf-8")

    assert "from invarlock.reporting import verify_contract as verify_mod" in contents
    assert "from invarlock.cli.commands import verify as verify_mod" not in contents
