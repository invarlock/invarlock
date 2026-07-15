from __future__ import annotations

import pytest

from invarlock.reporting.verify_dataset_identity import (
    append_strict_dataset_identity_errors,
)
from invarlock.reporting.verify_policy import (
    append_strict_policy_authorization_errors,
)
from tests.cli.verify._support_runtime_provenance import (
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
)
from tests.reporting.validation._support_strict_verifier_branch_contracts import (
    _baseline_errors,
)


def _hosted_pair() -> tuple[dict, dict]:
    revision = "a" * 40
    subject = {
        "dataset": {
            "provider": "hf_text",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": revision,
        }
    }
    baseline = {
        "data": {
            "dataset": "hf_text",
            "provider": "hf_text",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": revision,
        }
    }
    return subject, baseline


def test_strict_hosted_dataset_identity_accepts_complete_exact_pair() -> None:
    subject, baseline = _hosted_pair()
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert errors == []


@pytest.mark.parametrize("side", ["subject", "baseline"])
@pytest.mark.parametrize(
    "revision",
    [None, "main", "A" * 40, "a" * 39, f" {'a' * 40} "],
)
def test_strict_hosted_dataset_identity_rejects_missing_or_mutable_revision(
    side: str,
    revision: str | None,
) -> None:
    subject, baseline = _hosted_pair()
    target = subject["dataset"] if side == "subject" else baseline["data"]
    if revision is None:
        target.pop("revision")
    else:
        target["revision"] = revision
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any(
        f"Strict {side} hosted dataset revision must be 40-64 lowercase" in error
        for error in errors
    )


@pytest.mark.parametrize("side", ["subject", "baseline"])
def test_strict_hosted_dataset_identity_rejects_missing_configuration(
    side: str,
) -> None:
    subject, baseline = _hosted_pair()
    target = subject["dataset"] if side == "subject" else baseline["data"]
    target.pop("config_name")
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any(
        f"Strict {side} hosted dataset configuration requires non-empty" in error
        for error in errors
    )


def test_strict_hosted_dataset_identity_rejects_legacy_only_configuration_aliases() -> (
    None
):
    subject, baseline = _hosted_pair()
    subject["dataset"]["config"] = subject["dataset"].pop("config_name")
    baseline["data"]["config"] = baseline["data"].pop("config_name")
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any(
        "Strict subject hosted dataset configuration requires non-empty" in error
        for error in errors
    )
    assert any(
        "Strict baseline hosted dataset configuration requires non-empty" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("dataset_name", "other/dataset", "dataset name mismatch"),
        ("config_name", "other-config", "dataset configuration mismatch"),
        ("revision", "b" * 40, "dataset revision mismatch"),
    ],
)
def test_strict_hosted_dataset_identity_rejects_baseline_subject_mismatch(
    field: str,
    replacement: str,
    message: str,
) -> None:
    subject, baseline = _hosted_pair()
    baseline["data"][field] = replacement
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any(message in error for error in errors)


def test_strict_dataset_identity_rejects_baseline_provider_alias_fork() -> None:
    subject, baseline = _hosted_pair()
    baseline["data"]["dataset"] = "wikitext2"
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any("provider fork" in error for error in errors)


def test_strict_dataset_identity_rejects_subject_baseline_provider_mismatch() -> None:
    subject, baseline = _hosted_pair()
    baseline["data"]["provider"] = "wikitext2"
    baseline["data"]["dataset"] = "wikitext2"
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any("dataset provider mismatch" in error for error in errors)


def test_strict_dataset_identity_handles_non_mapping_dataset_section() -> None:
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject={"dataset": "invalid"},
        baseline={"data": {"dataset": "local_jsonl"}},
    )

    assert errors == []


def test_strict_non_hosted_dataset_does_not_require_remote_revision() -> None:
    errors: list[str] = []

    append_strict_dataset_identity_errors(
        errors,
        subject={"dataset": {"provider": "local_jsonl"}},
        baseline={"data": {"dataset": "local_jsonl"}},
    )

    assert errors == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_name", "local-corpus", "dataset name parity requires"),
        ("config_name", "default", "dataset configuration parity requires"),
        ("revision", "local-v1", "dataset revision parity requires"),
    ],
)
def test_strict_non_hosted_dataset_rejects_one_sided_optional_identity(
    field: str,
    value: str,
    message: str,
) -> None:
    errors: list[str] = []
    subject = {"dataset": {"provider": "local_jsonl", field: value}}
    baseline = {"data": {"dataset": "local_jsonl"}}

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )

    assert any(message in error for error in errors)


def test_acceptance_policy_rejects_coordinated_hosted_dataset_relabel() -> None:
    report = _strict_provenance_gate_cert()
    report["dataset"].update(
        {
            "provider": "hf_text",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": "a" * 40,
            "split": "validation",
        }
    )
    policy_pack = _matching_strict_policy_pack(report)
    baseline = _matching_strict_ppl_baseline(report)

    report["dataset"] = {"provider": "local_jsonl", "split": "validation"}
    baseline["data"] = {
        "provider": "local_jsonl",
        "dataset": "local_jsonl",
        "split": "validation",
    }
    parity_errors: list[str] = []
    append_strict_dataset_identity_errors(
        parity_errors,
        subject=report,
        baseline=baseline,
    )
    policy_errors: list[str] = []
    append_strict_policy_authorization_errors(
        policy_errors,
        report=report,
        policy_pack=policy_pack,
    )

    assert parity_errors == []
    assert any(
        "does not match the acceptance policy pack for provider" in error
        for error in policy_errors
    )


def test_strict_baseline_contract_invokes_hosted_identity_gate() -> None:
    report = _strict_provenance_gate_cert()
    report["dataset"]["provider"] = "hf_text"
    supplied = _matching_strict_ppl_baseline(report)

    errors = _baseline_errors(report, supplied)

    assert any("hosted dataset name" in error for error in errors)
    assert any("hosted dataset revision" in error for error in errors)
