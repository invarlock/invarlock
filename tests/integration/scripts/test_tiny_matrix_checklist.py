import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_tiny_all_matrix_script_is_executable() -> None:
    script = Path("scripts/smoke/run_tiny_all_matrix.sh")

    assert os.access(script, os.X_OK)


def test_tiny_gpt2_matrix_dry_run(tmp_path: Path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["GPT2_ID"] = "sshleifer/tiny-gpt2"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    # The script should complete without executing any commands and write a checklist
    subprocess.check_call(["bash", "scripts/smoke/run_tiny_all_matrix.sh"], env=env)
    checklist = Path(env["TMP_DIR"]) / "checklist.md"
    assert checklist.exists()
    text = checklist.read_text()
    assert "Evaluation Matrix" in text
    # Basic sanity: contains at least one evaluate command
    assert (
        "evaluate" in text
        and "--baseline-adapter hf_causal --subject-adapter hf_causal" in text
    )
    assert "INVARLOCK_ALLOW_NETWORK=1" not in text
    assert "INVARLOCK_ALLOW_NETWORK=0" in text
    assert "HF_DATASETS_OFFLINE=1" in text


def test_offline_matrix_overrides_inherited_network_allowance(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["INVARLOCK_ALLOW_NETWORK"] = "1"
    env["TMP_DIR"] = str(tmp_path / "tmp")

    subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"], env=env, check=True
    )
    text = (Path(env["TMP_DIR"]) / "checklist.md").read_text(encoding="utf-8")

    assert "INVARLOCK_ALLOW_NETWORK=0" in text
    assert "INVARLOCK_ALLOW_NETWORK=1" not in text


def test_checklist_records_network_allowance_only_when_net_enabled(tmp_path: Path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "1"
    env["TMP_DIR"] = str(tmp_path / "tmp")

    subprocess.check_call(["bash", "scripts/smoke/run_tiny_all_matrix.sh"], env=env)
    checklist = Path(env["TMP_DIR"]) / "checklist.md"
    text = checklist.read_text(encoding="utf-8")

    assert "INVARLOCK_ALLOW_NETWORK=1" in text
    assert "HF_DATASETS_OFFLINE=0" in text


def test_networked_dry_run_never_bootstraps_python_dependencies(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'echo "$@" >> {log_path}',
                'if [ "${1:-}" = "-c" ]; then exit 0; fi',
                'if [ "${1:-}" = "-" ]; then exit 97; fi',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "RUN": "0",
            "NET": "1",
            "PYTHON_BIN": str(fake_python),
            "TMP_DIR": str(tmp_path / "tmp"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "-c import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    ]


def _read_profile_from_checklist(path: str) -> str:
    txt = Path(path).read_text()
    for line in txt.splitlines():
        if "evaluate" in line and "--profile" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "--profile" and i + 1 < len(parts):
                    return parts[i + 1]
    return ""


def test_checklist_uses_dev_profile_when_tiny_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["INVARLOCK_TINY_RELAX"] = "1"
    subprocess_result = subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: dev" in subprocess_result.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    prof = _read_profile_from_checklist(str(checklists[-1]))
    assert prof == "dev"


def test_checklist_defaults_to_ci_when_no_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env.pop("INVARLOCK_TINY_RELAX", None)
    subprocess_result = subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: ci" in subprocess_result.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    prof = _read_profile_from_checklist(str(checklists[-1]))
    assert prof == "ci"


def test_explicit_profile_overrides_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["PROFILE"] = "ci"
    env["INVARLOCK_TINY_RELAX"] = "1"  # should NOT override explicit PROFILE
    res = subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: ci" in res.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    assert _read_profile_from_checklist(str(checklists[-1])) == "ci"


def test_checklist_no_longer_advertises_distilbert_classification(tmp_path: Path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    checklist = Path(env["TMP_DIR"]) / "checklist.md"
    text = checklist.read_text(encoding="utf-8")
    assert "DistilBERT" not in text
    assert "classification" not in text.lower()


def test_net_bootstrap_prefers_cpu_torch_before_hf_extra() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")
    cpu_torch_install = 'os.environ["TORCH_CPU_INDEX_URL"]'
    hf_install = '".[hf]"'

    assert "google.protobuf" in text
    assert "import sentencepiece" in text
    assert "import tiktoken" in text
    assert "TORCH_CPU_INDEX_URL" in text
    assert "export TORCH_CPU_INDEX_URL" in text
    assert cpu_torch_install in text
    assert hf_install in text
    assert text.index(cpu_torch_install) < text.index(hf_install)
    assert "\"$PYTHON_BIN\" - << 'PY' || true" not in text
    assert 'HF_HOME="${HF_HOME:-$TMP_DIR/.hf}"' in text
    assert 'mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"' in text


def test_hf_extras_include_sentencepiece_for_runtime_tokenizer_support() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"sentencepiece>=0.2.1"' in text
    assert '"tiktoken>=0.9.0"' in text


def test_matrix_uses_repo_python_selector_and_py312_floor() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")

    assert 'source "$SCRIPT_DIR/lib/smoke_common.sh"' in text
    assert 'smoke_select_python "$REPO_ROOT"' in text
    assert 'smoke_setup_pythonpath "$REPO_ROOT"' in text
    assert "requires Python 3.12+" in text
    assert 'CLI=("$PYTHON_BIN" -m invarlock.cli)' in text


def test_verify_exports_selected_python_to_nested_smokes() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "verify: export PYTHON_BIN := $(PYTHON)" in makefile


def test_quant_demo_uses_dev_profile_by_default() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")

    assert 'QUANT_PROFILE="${QUANT_PROFILE:-dev}"' in text
    assert '--profile "$QUANT_PROFILE"' in text
    assert '--edit-config "$QCFG" --assurance off' in text
    assert 'append "gpt2_eval_quant8_${QUANT_PROFILE}"' in text


def test_relaxed_profile_evaluate_commands_disable_strict_assurance() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")

    assert "append_relaxed_assurance_args()" in text
    assert "cmd+=(--assurance off)" in text
    assert text.count("append_relaxed_assurance_args") == 3


def test_encoder_mlm_smoke_uses_stable_tiny_model() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")

    assert 'BERT_ID=${BERT_ID:-"sshleifer/tiny-distilroberta-base"}' in text
    assert 'echo "## Encoder MLM" >> "$TMP_DIR/checklist.md"' in text


def test_matrix_prefers_local_runtime_image_when_available() -> None:
    text = Path("scripts/smoke/run_tiny_all_matrix.sh").read_text(encoding="utf-8")
    common_text = Path("scripts/smoke/lib/smoke_common.sh").read_text(encoding="utf-8")

    assert 'smoke_seed_local_runtime_image "auto"' in text
    assert 'smoke_ensure_current_runtime_image "container" "auto"' in text
    assert "docker image inspect invarlock-runtime:cuda-local" in common_text
    assert (
        'export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"' in common_text
    )
    assert "docker image inspect invarlock-runtime:local" in common_text
    assert 'export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"' in common_text
    assert 'echo "[smoke] refreshing local CUDA container runtime image"' in common_text
    assert 'echo "[smoke] refreshing local container runtime image"' in common_text
    assert "make runtime-image-cuda" in common_text
    assert "make runtime-image" in common_text


def test_run_mode_falls_back_to_python_module_when_console_script_missing(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'echo "$@" >> {log_path}',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["RUN"] = "1"
    env["NET"] = "0"
    env["INVARLOCK_TINY_RELAX"] = "1"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    env["PYTHON_BIN"] = str(fake_python)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    installed_cli = shutil.which("invarlock")
    if installed_cli:
        cli_dir = str(Path(installed_cli).resolve().parent)
        path_parts = [
            part for part in path_parts if Path(part).resolve() != Path(cli_dir)
        ]
    env["PATH"] = os.pathsep.join([str(bin_dir), *path_parts])

    subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8")
    assert "-m invarlock.cli evaluate" in calls


def test_run_mode_reports_failures_after_completing_matrix(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'echo "$@" >> {log_path}',
                'case "$*" in',
                '  *"--edit-config"*) exit 7 ;;',
                "esac",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["RUN"] = "1"
    env["NET"] = "0"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    env["PYTHON_BIN"] = str(fake_python)

    result = subprocess.run(
        ["bash", "scripts/smoke/run_tiny_all_matrix.sh"],
        env=env,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "ERROR: 1 matrix command(s) failed." in result.stderr
    assert "--edit-config" in calls
    assert "hf_mlm" in calls


pytestmark = pytest.mark.integration
