from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.cli.config import (
    InvarLockConfig,
    apply_profile,
    load_tiers,
    resolve_edit_kind,
)


def test_load_tiers_from_runtime_override(tmp_path: Path, monkeypatch):
    rt = tmp_path / "runtime"
    rt.mkdir()
    tiers = rt / "tiers.yaml"
    tiers.write_text(
        "balanced: {metrics: {pm_ratio: {min_tokens: 123}}}", encoding="utf-8"
    )
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(tmp_path))
    cfg = load_tiers()
    assert cfg["balanced"]["metrics"]["pm_ratio"]["min_tokens"] == 123


def test_apply_profile_unknown_raises():
    with pytest.raises(ValueError):
        apply_profile(InvarLockConfig(dataset={"provider": "wikitext2"}), "unknown")


def test_resolve_edit_kind_positive():
    assert resolve_edit_kind("prune") == "quant_rtn"
    assert resolve_edit_kind("quant") == "quant_rtn"
    assert resolve_edit_kind("mixed") == "orchestrator"


def test_apply_profile_runtime_profile_success(tmp_path: Path, monkeypatch):
    rt = tmp_path / "runtime" / "profiles"
    rt.mkdir(parents=True)
    prof = rt / "dev.yaml"
    prof.write_text("dataset: {preview_n: 5, final_n: 7}", encoding="utf-8")
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(tmp_path))
    cfg = InvarLockConfig(dataset={"provider": "wikitext2"})
    out = apply_profile(cfg, "dev")
    d = out.data.get("dataset", {})
    assert d.get("preview_n") == 5 and d.get("final_n") == 7


def test_load_tiers_not_found_raises(monkeypatch):
    import invarlock.cli.config as config_mod

    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", "")
    monkeypatch.setattr(config_mod, "_load_runtime_yaml", lambda *a, **k: None)

    class _NoRes:
        def files(self, *a, **k):
            raise FileNotFoundError

    # Replace _ires with object whose files() raises
    monkeypatch.setattr(config_mod, "_ires", _NoRes())
    with pytest.raises(FileNotFoundError):
        _ = load_tiers()
