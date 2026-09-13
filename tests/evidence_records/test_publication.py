"""Publication must not expose partial results or mutate caller-owned evidence."""

import pytest

from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.templates import example_project
from tests._evaluation_support import build_pack, pack_json


def test_evidence_is_a_snapshot_of_caller_owned_inputs():
    baseline, candidate, policy = example_project("classification")
    evidence = build_pack(baseline, candidate, policy)
    candidate["records"][0]["output"] = "changed later"
    policy["metrics"][0]["maximum_regression"] = 1
    assert pack_json(evidence, "subject")["records"][0]["output"] != "changed later"
    assert pack_json(evidence, "policy")["metrics"][0]["maximum_regression"] != 1


def test_contract_resource_limits_apply_to_sdk_inputs(monkeypatch):
    baseline, _, _ = example_project("classification")
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 16)
    with pytest.raises(EvaluationRecordsError, match="byte limit"):
        contracts.validate(baseline, "run")
