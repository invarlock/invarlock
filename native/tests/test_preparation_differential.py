"""Native request and evaluation preparation against the production reference."""

import base64
import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from invarlock import captured_evaluation, evaluation_transaction
from invarlock.core import evaluation_request
from invarlock.evaluation_records.templates import example_project
from tests.cli.test_evaluation_preflight import _materialize_run_request
from tests.cli.test_import_journey import _materialize_request
from tests.core.test_captured_evaluation import _key, _request

ROOT = Path(__file__).resolve().parents[2]


def native(operation, config):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "native/target"))
    process = subprocess.run(
        [str(target / "debug/invarlock-native-verify"), operation, "--request", "-"],
        input=json.dumps(config),
        text=True,
        capture_output=True,
    )
    value = json.loads(process.stdout)
    return {"error": value["error"]} if "error" in value else {"ok": value}


@pytest.mark.parametrize(
    "mode",
    [
        "pass",
        "signed",
        "unlimited",
        "budget",
        "pin",
        "cases",
        "pair",
        "policy",
        "unsigned-type",
        "missing",
        "collision",
    ],
)
def test_captured_preflight(mode, tmp_path):
    root = tmp_path.resolve()
    baseline, subject, policy = example_project("classification")
    if mode == "pair":
        subject["records"][0]["input"] = "changed"
    if mode == "policy":
        policy["metrics"][0]["configuration"] = {"unexpected": 1}
    if mode == "cases":
        policy["expected_case_set_digest"] = "sha256:" + "f" * 64
    if mode == "budget":
        policy["metrics"][0].update(kind="token_f1")
    if mode == "missing":
        subject["records"][0].update(output=None, error="missing")
    request = _request(root, baseline, subject, policy)
    authored = yaml.safe_load((root / "request.yaml").read_text())
    if mode == "pin":
        authored["comparison"]["baseline"]["expected_run_digest"] = "sha256:" + "f" * 64
        (root / "request.yaml").write_text(yaml.safe_dump(authored))
        request = evaluation_request.load_evaluation_request(root / "request.yaml")
    if mode == "collision":
        request.evidence.mkdir()
    key = _key(root) if mode == "signed" else None
    unsigned = False if key else 1 if mode == "unsigned-type" else True
    budget = None if mode == "unlimited" else 0 if mode == "budget" else 102_400_000
    try:
        b, s, p, n, preflight = captured_evaluation._prepare_captured_request(
            request, signing_key_path=key, unsigned=unsigned, max_bootstrap_draws=budget
        )
        expected = {
            "ok": {
                "baseline": b,
                "subject": s,
                "policy": p,
                "normalized_request": n,
                "preflight": json.loads(preflight.as_json()),
            }
        }
    except ValueError as error:
        expected = {"error": str(error)}
    actual = native(
        "prepare-captured",
        {
            "root": str(root),
            "authored_request": authored,
            "signing_key_path": str(key) if key else None,
            "unsigned": unsigned,
            "max_bootstrap_draws": budget,
        },
    )
    assert actual == expected


@pytest.mark.parametrize(
    "mode", ["run", "import", "schedule", "policy", "minimum", "batch", "dataset"]
)
def test_runtime_preparation_and_normalization(mode, tmp_path):
    root = tmp_path.resolve()
    if mode == "import":
        material = _materialize_request(root)
        path = Path(material["request"])
        key = _key(root)
    else:
        path, key = _materialize_run_request(root)
    authored = yaml.safe_load(path.read_text())
    if mode == "policy":
        (root / authored["comparison"]["policy"]).write_text("{}")
    if mode == "minimum":
        (root / authored["comparison"]["policy"]).write_text(
            json.dumps(
                {
                    "resolved_policy": {
                        "metrics": {
                            "exact_match": {
                                "delta_min_pp": 0.0,
                                "minimum_record_count": 2,
                                "maximum_interval_width_pp": 100.0,
                            }
                        }
                    }
                }
            )
        )
    if mode == "batch":
        authored["comparison"]["baseline"]["runtime"]["settings"]["batch_size"] = 2
    if mode == "dataset":
        (root / authored["comparison"]["dataset"]["path"]).write_text("invalid")
    if mode == "schedule":
        authored["comparison"]["dataset"]["limit"] = 2
    path.write_text(yaml.safe_dump(authored, sort_keys=False))
    try:
        loaded = evaluation_request.load_evaluation_request(path)
        prepared = evaluation_transaction._prepare_evaluation_inputs(
            loaded, signing_key_path=key, scorer_registry=None
        )
        expected = {
            "ok": {
                "schedule": prepared.schedule.to_payload(),
                "canonical_schedule": prepared.canonical_schedule.decode(),
                "schedule_digest": prepared.schedule.schedule_sha256,
                "policy": json.loads(prepared.policy_bytes),
                "policy_base64": base64.b64encode(prepared.policy_bytes).decode(),
                "policy_digest": prepared.policy_digest,
                "selected_metric": prepared.selected_metric,
                "sample_requirements": prepared.sample_requirements,
                "observations": [
                    {
                        "id": o.observation_id,
                        "kind": o.kind,
                        "scope": o.scope,
                        "payload_base64": base64.b64encode(o.payload).decode(),
                        "payload_digest": "sha256:"
                        + __import__("hashlib").sha256(o.payload).hexdigest(),
                    }
                    for o in prepared.observations
                ],
                "judge": None,
            }
        }
    except ValueError as error:
        expected = {"error": str(error)}
    actual = native(
        "prepare-runtime-inputs", {"root": str(root), "authored_request": authored}
    )
    assert actual == expected
    if "ok" in expected:
        expected_normalized = evaluation_transaction._normalized_request(
            loaded, prepared.schedule, prepared.observations
        )
        assert native(
            "normalize-runtime-request",
            {
                "authored_request": authored,
                "schedule": prepared.schedule.to_payload(),
                "observations": expected["ok"]["observations"],
            },
        ) == {"ok": expected_normalized}


def test_request_schema_context_diagnostics(tmp_path):
    path, _ = _materialize_run_request(tmp_path.resolve())
    baseline = yaml.safe_load(path.read_text())
    paths = [
        [],
        ["format_version"],
        ["comparison"],
        ["comparison", "dataset"],
        ["comparison", "metric"],
        ["comparison", "baseline"],
        ["comparison", "baseline", "artifact"],
        ["comparison", "baseline", "runtime"],
        ["execution"],
        ["execution", "mode"],
        ["output"],
    ]
    cases = []
    for parts in paths:
        for replacement in [None, False, 0, 1.0, "", {}, [], "unknown"]:
            value = copy.deepcopy(baseline)
            if not parts:
                value = replacement
            else:
                node = value
                for key in parts[:-1]:
                    node = node[key]
                node[parts[-1]] = replacement
            cases.append(value)
    for value in cases:
        try:
            evaluation_request._reject_include_directives(value)
            if not isinstance(value, dict):
                raise ValueError("request must be a YAML object")
            version = value.get("format_version")
            if version == "invarlock/evaluation-request-v1":
                evaluation_request._validate_schema(value)
            elif version == "invarlock/evaluation-request-v2":
                evaluation_request._validate_captured_schema(value)
            else:
                raise ValueError(f"unsupported evaluation request format {version!r}")
            expected = {"ok": value}
        except ValueError as error:
            expected = {"error": str(error)}
        assert native("validate-request", {"value": value}) == expected, value
