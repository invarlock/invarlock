import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_cert(cert_path: Path) -> dict:
    assert cert_path.is_file(), (
        f"required evaluation report fixture missing: {cert_path}"
    )
    return json.loads(cert_path.read_text(encoding="utf-8"))


def test_variance_enabled_for_balanced_evidence_pack():
    cert = _load_cert(
        REPO_ROOT / "tests/artifacts/evidencepack/svd95_balanced_release_cert.json"
    )

    variance = cert["variance"]
    assert variance["enabled"] is True

    gate = variance["predictive_gate"]
    assert gate["evaluated"] is True
    assert gate["passed"] is True
    hi = max(gate["delta_ci"])
    assert hi < 0, "Balanced predictive CI upper bound should be negative"


def test_variance_disabled_for_conservative_evidence_pack():
    cert = _load_cert(
        REPO_ROOT / "tests/artifacts/evidencepack/svd95_conservative_release_cert.json"
    )

    variance = cert["variance"]
    assert variance["enabled"] is False

    gate = variance["predictive_gate"]
    assert gate["evaluated"] is True
    assert gate["passed"] is False
    assert gate["reason"] in {"ci_contains_zero", "below_min_effect"}


pytestmark = pytest.mark.integration
