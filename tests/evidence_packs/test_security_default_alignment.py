from __future__ import annotations

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


def test_evidence_pack_shell_verifier_keeps_provenance_enforced() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    verify_pack = (repo_root / "scripts/evidence_packs/verify_pack.sh").read_text(
        encoding="utf-8"
    )
    runtime_sh = (repo_root / "scripts/evidence_packs/lib/core/runtime.sh").read_text(
        encoding="utf-8"
    )

    assert "--allow-unverified-provenance" not in verify_pack
    assert "invarlock _run" not in runtime_sh
    assert "run_from_config.py" in runtime_sh
