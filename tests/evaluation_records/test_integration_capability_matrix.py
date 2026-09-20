"""All maintained adapters preserve reference availability and subject identity.

These are contract tests using authored native records, not live qualification.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.engine import (
    EVALUATORS,
    evaluator_input_capabilities,
    export_evaluator_result,
    load_run,
)
from tests.evaluation_records.test_hosted_service_identity import identity

SHAPES_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/integrations/evaluator-parity/native_shapes.py"
)
SPEC = importlib.util.spec_from_file_location("capability_shapes", SHAPES_PATH)
assert SPEC and SPEC.loader
SHAPES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHAPES)


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("has_reference", [True, False])
@pytest.mark.parametrize("hosted", [True, False])
def test_every_adapter_preserves_capabilities_and_identity_across_routes(
    tmp_path, evaluator, has_reference, hosted
):
    rows = [
        {
            "id": "case-one",
            "input": "Explain why the sky is blue.",
            "output": "Shorter wavelengths scatter more strongly.",
            "expected": "Rayleigh scattering" if has_reference else None,
            "metadata": {"slice": "science"},
        }
    ]
    native = SHAPES.payload(evaluator, rows, "test", run_id="contract")
    projection = {
        "lm-evaluation-harness": {
            "kind": "json-pointer",
            "pointer": "/context/arguments/0/0",
        },
        "promptfoo": {"kind": "json-pointer", "pointer": "/context/prompt"},
    }.get(evaluator)
    options = {
        "run_id": "contract",
        "artifact_digest": None if hosted else "sha256:" + "a" * 64,
        "input_projection": projection,
    }
    if hosted:
        options["service_identity"] = identity()
    envelope = tmp_path / "envelope.json"
    exported = export_evaluator_result(
        evaluator,
        native,
        envelope,
        expected_ids=["case-one"],
        source_version="test",
        **options,
    )
    raw = tmp_path / "native.json"
    raw.write_text(json.dumps(native["result"] if evaluator == "langfuse" else native))
    for path, adapter in (
        (envelope, "evaluator-json"),
        (raw, "evaluator-native-json"),
    ):
        run = load_run(
            path,
            adapter=adapter,
            source={"name": evaluator, "version": "test"},
            **options,
        )
        assert (
            run["source_digest"]
            == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert {k: v for k, v in run.items() if k != "source_digest"} == {
            k: v for k, v in exported.items() if k != "source_digest"
        }
        record = run["records"][0]
        assert record["id"] == rows[0]["id"]
        assert record["output"] == rows[0]["output"]
        assert record["expected"] == rows[0]["expected"]
        assert record["metadata"]["slice"] == "science"
        assert run["artifact_digest"] == options["artifact_digest"]
        if hosted:
            assert run["service_identity"] == options["service_identity"]
        else:
            assert "service_identity" not in run
        capabilities = evaluator_input_capabilities(run)
        assert capabilities["judge"]["usable_count"] == 1
        assert capabilities["exact_match"]["usable_count"] == int(has_reference)
        assert capabilities["normalized_nll_per_utf8_byte"]["usable_count"] == 0
