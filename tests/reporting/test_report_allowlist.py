import json
import math
from pathlib import Path

import pytest

from invarlock.reporting import report_builder as cert


def test_load_validation_allowlist_prefers_contract_file(tmp_path, monkeypatch):
    root = Path(cert.__file__).resolve().parents[3]
    contracts_dir = root / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    path = contracts_dir / "validation_keys.json"
    original_text = path.read_text(encoding="utf-8") if path.exists() else None

    try:
        path.write_text(json.dumps(["a", "b"]))
        keys = cert._load_validation_allowlist()
        assert "a" in keys and "b" in keys

        path.write_text(json.dumps({"bad": True}))
        keys2 = cert._load_validation_allowlist()
        # Fallback to default allowlist when file content is invalid
        assert cert._VALIDATION_ALLOWLIST_DEFAULT.issubset(keys2)
    finally:
        if original_text is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            path.write_text(original_text, encoding="utf-8")
        # Clean up empty contracts dir if we created it
        try:
            if contracts_dir.exists() and not any(contracts_dir.iterdir()):
                contracts_dir.rmdir()
        except Exception:
            pass


def test_load_validation_allowlist_with_source_reports_fallback() -> None:
    keys, source = cert._load_validation_allowlist_with_source()
    assert isinstance(keys, set)
    assert source == "contracts" or source.startswith("fallback:")


def test_apply_validation_allowlist_schema_fails_closed() -> None:
    original = cert.REPORT_JSON_SCHEMA.get("properties")
    try:
        cert.REPORT_JSON_SCHEMA["properties"] = None
        with pytest.raises(RuntimeError, match="properties must be a mapping"):
            cert._apply_validation_allowlist_schema({"primary_metric_acceptable"})
    finally:
        cert.REPORT_JSON_SCHEMA["properties"] = original


def test_compute_edit_digest_branches():
    quant_digest = cert._compute_edit_digest(
        {"edit": {"name": "quant_rtn", "plan": {}}}
    )
    assert quant_digest["family"] == "quantization"

    noop_digest = cert._compute_edit_digest({"edit": {"name": "noop", "plan": {}}})
    assert noop_digest["family"] == "cert_only"


def test_is_ppl_kind_variants():
    assert cert._is_ppl_kind("ppl_causal")
    assert cert._is_ppl_kind("ppl_seq2seq")
    assert not cert._is_ppl_kind("accuracy")


def test_fallback_paired_windows():
    cov = {"preview": {"used": 7}}
    assert cert._fallback_paired_windows(0, cov) == 7
    assert cert._fallback_paired_windows(5, cov) == 5
    assert cert._fallback_paired_windows(0, {}) == 0


def test_enforce_drift_ratio_identity_and_alignment():
    # Matching ratio should return computed ratio
    ratio = cert._enforce_drift_ratio_identity(
        paired_windows=4,
        delta_mean=math.log(1.1),
        drift_ratio=1.1,
        window_plan_profile="ci",
    )
    assert pytest.approx(ratio, rel=1e-3) == 1.1

    # Mismatch in CI profile should raise
    with pytest.raises(ValueError):
        cert._enforce_drift_ratio_identity(
            paired_windows=4,
            delta_mean=0.5,
            drift_ratio=1.1,
            window_plan_profile="ci",
        )

    # Ratio CI alignment: paired baseline enforces exp(logloss_delta_ci)
    with pytest.raises(ValueError):
        cert._enforce_ratio_ci_alignment("paired_baseline", (1.0, 1.1), (-0.2, -0.1))

    # Matching ratios should pass quietly
    cert._enforce_ratio_ci_alignment(
        "paired_baseline",
        (math.exp(-0.2), math.exp(-0.1)),
        (-0.2, -0.1),
    )
