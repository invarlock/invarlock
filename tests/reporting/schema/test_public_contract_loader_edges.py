from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock import public_contracts
from invarlock.public_contracts import ContractLoadError


def test_all_shipped_contract_loaders_return_independent_json_objects() -> None:
    cases = (
        (
            public_contracts.load_evaluation_request_schema,
            "invarlock/evaluation-request-v1",
        ),
        (public_contracts.load_evidence_pack_schema, None),
        (
            public_contracts.load_evidence_observation_schema,
            "invarlock/evidence-observation-v1",
        ),
        (
            public_contracts.load_trust_inputs_schema,
            "invarlock/trust-inputs-v1",
        ),
        (public_contracts.load_runtime_manifest_schema, None),
        (public_contracts.load_runtime_provider_capabilities_schema, None),
        (public_contracts.load_model_artifact_identity_schema, None),
        (public_contracts.load_runtime_provider_receipt_schema, None),
        (public_contracts.load_runtime_scoring_observation_schema, None),
        (public_contracts.load_runtime_behavioral_schedule_schema, None),
    )

    for loader, expected_format in cases:
        first = loader()
        second = loader()
        assert isinstance(first, dict) and first
        assert first == second
        assert first is not second
        if expected_format is not None:
            encoded = json.dumps(first, sort_keys=True)
            assert expected_format in encoded


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "No such file"),
        ("{", "Expecting"),
        ("[]", "expected JSON object, got list"),
    ],
)
def test_contract_loader_fails_closed_for_missing_malformed_or_nonobject_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
    message: str,
) -> None:
    monkeypatch.setattr(public_contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path)
    if contents is not None:
        (tmp_path / "contract.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ContractLoadError, match=message) as caught:
        public_contracts._load_object_contract("contract.json")

    assert caught.value.filename == "contract.json"
    assert caught.value.reason
