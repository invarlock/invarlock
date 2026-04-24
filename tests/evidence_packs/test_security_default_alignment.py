from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.evidence_packs.python import preset_generator, runtime_tools


def test_runtime_tools_require_remote_code_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)
    with pytest.raises(RuntimeError, match="INVARLOCK_ALLOW_REMOTE_CODE=1"):
        runtime_tools.require_remote_code_opt_in("demo-script.py")

    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    assert runtime_tools.require_remote_code_opt_in("demo-script.py") is True


def test_preset_generator_requires_allow_for_remote_code_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_HF_TRUST_REMOTE_CODE", "true")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_NAME", "demo/dataset")
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)

    with pytest.raises(ValueError, match="INVARLOCK_ALLOW_REMOTE_CODE=1"):
        preset_generator._resolve_dataset_provider_spec("hf_text")

    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    provider = preset_generator._resolve_dataset_provider_spec("hf_text")
    assert isinstance(provider, dict)
    assert provider["trust_remote_code"] is True


def test_config_generator_remote_code_defaults_false() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    command = (
        "set -e\n"
        "source scripts/evidence_packs/lib/config_generator.sh\n"
        'generate_invarlock_config "model" "/dev/stdout" "noop" 42 10 20 100 128 64 1\n'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    assert "trust_remote_code: false" in result.stdout


def test_config_generator_remote_code_true_requires_explicit_allow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    command = (
        "set -e\n"
        "export INVARLOCK_ALLOW_REMOTE_CODE=1\n"
        "source scripts/evidence_packs/lib/config_generator.sh\n"
        'generate_invarlock_config "model" "/dev/stdout" "noop" 42 10 20 100 128 64 1\n'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    assert "trust_remote_code: true" in result.stdout


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/evidence_packs/python/rmt_cross_model_probe.py",
        "scripts/evidence_packs/python/ve_cross_model_probe.py",
    ],
)
def test_probe_scripts_gate_remote_code_explicitly(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / relative_path).read_text(encoding="utf-8")
    assert "require_remote_code_opt_in" in text
    assert "default=False" in text
    assert "INVARLOCK_ALLOW_REMOTE_CODE=1" in text


def test_task_functions_forward_probe_remote_code_opt_in() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "scripts/evidence_packs/lib/task_functions.sh").read_text(
        encoding="utf-8"
    )
    assert "probe_args+=(--trust-remote-code)" in text
    assert "ve_probe_args+=(--trust-remote-code)" in text


def test_evidence_pack_shell_flows_keep_provenance_enforced() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_pack = (repo_root / "scripts/evidence_packs/run_pack.sh").read_text(
        encoding="utf-8"
    )
    verify_pack = (repo_root / "scripts/evidence_packs/verify_pack.sh").read_text(
        encoding="utf-8"
    )
    task_functions = (
        repo_root / "scripts/evidence_packs/lib/task_functions.sh"
    ).read_text(encoding="utf-8")
    config_generator = (
        repo_root / "scripts/evidence_packs/lib/config_generator.sh"
    ).read_text(encoding="utf-8")
    runtime_sh = (repo_root / "scripts/evidence_packs/lib/runtime.sh").read_text(
        encoding="utf-8"
    )

    assert "--allow-unverified-provenance" not in run_pack
    assert "--allow-unverified-provenance" not in verify_pack
    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in task_functions
    assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in config_generator
    assert "invarlock run" not in task_functions
    assert "invarlock run" not in config_generator
    assert "invarlock _run" not in runtime_sh
    assert "run_from_config.py" in runtime_sh
    assert "invarlock advanced evidence-pack verify" in run_pack
    assert "invarlock evidence-pack verify" not in run_pack
