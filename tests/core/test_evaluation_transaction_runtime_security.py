from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.evaluation_transaction as transaction
from invarlock.evaluation_transaction import EvaluationTransactionError
from invarlock.evidence_pack_contract import EvidencePackError


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ("INVARLOCK_ALLOW_NETWORK", "network access"),
        ("INVARLOCK_ALLOW_REMOTE_CODE", "remote code"),
        (
            "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
            "third-party provider discovery",
        ),
    ],
)
def test_evaluate_rejects_runtime_opt_ins_before_provider_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: str,
    message: str,
) -> None:
    discovered = False

    def registry() -> object:
        nonlocal discovered
        discovered = True
        raise AssertionError("provider discovery must not run")

    monkeypatch.setattr(transaction, "CoreRegistry", registry)
    monkeypatch.setenv(environment, "1")

    with pytest.raises(EvaluationTransactionError, match=message):
        transaction.evaluate_request_file(
            tmp_path / "request.yaml",
            signing_key_path=tmp_path / "evidence.pem",
        )

    assert discovered is False


def test_evaluate_loads_signing_key_before_request_or_provider_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovered = False

    def registry() -> object:
        nonlocal discovered
        discovered = True
        raise AssertionError("provider discovery must not run")

    key = tmp_path / "invalid.pem"
    key.write_text("not a private key", encoding="utf-8")
    monkeypatch.setattr(transaction, "CoreRegistry", registry)

    with pytest.raises(EvaluationTransactionError, match="could not load signing key"):
        transaction.evaluate_request_file(
            tmp_path / "request.yaml",
            signing_key_path=key,
        )

    assert discovered is False


@pytest.mark.parametrize(
    ("metric", "policy", "message"),
    [
        (
            "exact_match",
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 101}}}},
            "between -100 and 100",
        ),
        (
            "normalized_nll_per_utf8_byte",
            {
                "resolved_policy": {
                    "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 0}}
                }
            },
            "must be positive",
        ),
    ],
)
def test_metric_policy_is_fully_validated_by_preflight(
    metric: str, policy: dict[str, object], message: str
) -> None:
    with pytest.raises(EvidencePackError, match=message):
        transaction._preflight_policy(
            policy,
            metric=metric,
            policy_digest="sha256:" + "a" * 64,
        )


def test_policy_preflight_accepts_each_supported_metric() -> None:
    policies = {
        "exact_match": {
            "resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0}}}
        },
        "normalized_nll_per_utf8_byte": {
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.1}}
            }
        },
    }
    for metric, policy in policies.items():
        transaction._preflight_policy(
            policy,
            metric=metric,
            policy_digest="sha256:" + "a" * 64,
        )
