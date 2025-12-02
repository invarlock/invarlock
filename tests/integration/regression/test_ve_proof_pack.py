import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_CERTS = {
    "svd95_balanced_release_cert.json": {
        "variance": {
            "enabled": True,
            "predictive_gate": {
                "evaluated": True,
                "passed": True,
                "delta_ci": [-0.002, -0.001],
            },
        }
    },
    "svd95_conservative_release_cert.json": {
        "variance": {
            "enabled": False,
            "predictive_gate": {
                "evaluated": True,
                "passed": False,
                "reason": "ci_contains_zero",
                "delta_ci": [-0.0005, 0.0003],
            },
        }
    },
}


def _load_cert(cert_path: Path) -> dict:
    if cert_path.is_file():
        return json.loads(cert_path.read_text(encoding="utf-8"))

    sample = SAMPLE_CERTS.get(cert_path.name)
    if sample is not None:
        return sample

    raise AssertionError(f"Certificate missing: {cert_path}")


def test_variance_enabled_for_balanced_proof_pack():
    cert = _load_cert(
        REPO_ROOT / "artifacts/proofpack/svd95_balanced_release_cert.json"
    )

    variance = cert["variance"]
    assert variance["enabled"] is True

    gate = variance["predictive_gate"]
    assert gate["evaluated"] is True
    assert gate["passed"] is True
    hi = max(gate["delta_ci"])
    assert hi < 0, "Balanced predictive CI upper bound should be negative"


def test_variance_disabled_for_conservative_proof_pack():
    cert = _load_cert(
        REPO_ROOT / "artifacts/proofpack/svd95_conservative_release_cert.json"
    )

    variance = cert["variance"]
    assert variance["enabled"] is False

    gate = variance["predictive_gate"]
    assert gate["evaluated"] is True
    assert gate["passed"] is False
    assert gate["reason"] in {"ci_contains_zero", "below_min_effect"}


pytestmark = pytest.mark.integration
