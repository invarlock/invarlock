"""Connect actual SDK serialization to a separately installed core recipient.

Retained 400-case EM/NLL measurements are serialized, never remeasured. Judge
measurements remain explicitly synthetic contract fixtures. No SDK name or
successful handoff promotes these replays to a new model qualification.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/integrations/evaluator-parity/run.py"
SPEC = importlib.util.spec_from_file_location("sdk_recipient_parity", SCRIPT)
assert SPEC and SPEC.loader
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def test_actual_sdk_exports_reach_installed_recipient(tmp_path, monkeypatch):
    evaluator = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK")
    python = os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON")
    if not evaluator:
        pytest.skip(
            "dedicated SDK gate supplies the selected SDK and installed recipient"
        )
    if evaluator not in PARITY.profiles():
        pytest.fail("selected SDK has no maintained profile")
    if not python:
        pytest.fail("selected SDK gate requires an installed recipient")
    # SDK-side utilities import only the optional evaluator selected by this job.
    if evaluator in {
        "deepeval",
        "ragas",
        "lighteval",
        "hugging-face-evaluate",
        "autoevals",
        "openevals",
        "arize-phoenix-evals",
        "opik",
    }:
        from tests.evaluation_records.sdk_scalar_roundtrip import roundtrip
    elif evaluator in {
        "pydantic-evals",
        "azure-ai-evaluation",
        "evidently",
        "mlflow",
        "garak",
        "openai-evals",
        "trulens",
    }:
        from tests.evaluation_records.sdk_batch_roundtrip import roundtrip
    else:
        from tests.evaluation_records.sdk_original_roundtrip import roundtrip

    def forbid_network(*args, **kwargs):
        raise AssertionError(
            "SDK serialization cannot access a model or network service"
        )

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    recipient = PARITY.installed_identity(python)
    assert len(recipient["sdk_modules_absent"]) == 18
    version = PARITY.profiles()[evaluator]
    if evaluator != "promptfoo":
        package = {
            "lm-evaluation-harness": "lm-eval",
            "hugging-face-evaluate": "evaluate",
            "openai-evals": "evals",
        }.get(evaluator, evaluator)
        assert importlib.metadata.version(package) == version
        if evaluator == "openai-evals":
            revision = (
                (ROOT / "examples/evaluator-qualification/locks/openai-evals.txt")
                .read_text()
                .strip()
                .rsplit("@", 1)[-1]
            )
            direct = importlib.metadata.distribution(package).read_text(
                "direct_url.json"
            )
            assert direct and json.loads(direct)["vcs_info"]["commit_id"] == revision
    results = []
    for scorer in PARITY.SCORERS:
        originals, _, _ = PARITY.retained(scorer)
        captures = tmp_path / scorer / "sdk"
        captures.mkdir(parents=True)
        files = {}
        for side, original in zip(("baseline", "subject"), originals, strict=True):
            native = PARITY.SHAPES.payload(
                evaluator,
                original["records"],
                version,
                run_id=f"parity-{scorer}-{side}",
            )
            if evaluator == "openevals":
                from openevals.exact import exact_match

                # TypedDict construction alone is not an upstream execution.
                # Replace this harness's authored auxiliary score with the
                # actual offline SDK result; retained model facts stay intact.
                # Nullable JSON fields preserve absent NLL answers without
                # passing forbidden top-level None or fabricating generation.
                for entry in native:
                    auxiliary = exact_match(
                        outputs={"retained_output": entry["outputs"]},
                        reference_outputs={
                            "retained_output": entry["reference_outputs"]
                        },
                    )
                    assert float(auxiliary["score"]) == entry["metric_result"]["score"]
                    entry["metric_result"] = auxiliary
            serialized = roundtrip(
                evaluator, native, tmp_path=captures / f"work-{side}"
            )
            raw = json.dumps(serialized, allow_nan=False, ensure_ascii=False).encode(
                "utf-8"
            )
            (captures / f"{side}.json").write_bytes(raw)
            files[side] = {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
        (captures / "origin.json").write_text(
            json.dumps(
                {
                    "format": "invarlock/evaluator-native-captures-v1",
                    "evaluator": evaluator,
                    "source_version": version,
                    "scorer": scorer,
                    "files": files,
                    "provenance": {
                        "capture": "Actual pinned SDK serialization of retained records",
                        **(
                            {
                                "input_projection": {
                                    "kind": "json-pointer",
                                    "pointer": "/context/arguments/gen_args_0/arg_0",
                                }
                            }
                            if evaluator == "lm-evaluation-harness"
                            else {}
                        ),
                        "model_execution": "No new model execution",
                        "measurement_origin": "Synthetic judge fixture"
                        if scorer == "judge"
                        else "Retained 400-case Mistral 7B measurements",
                        "auxiliary_fields": (
                            "OpenEvals computes offline equality of nullable retained_output JSON fields; original case text is unchanged and the auxiliary grade is not new model, likelihood or judge measurement"
                            if evaluator == "openevals"
                            else "SDK-required wrapper fields and unused metric context are serialization fixtures, not additional model measurements"
                        ),
                    },
                },
                allow_nan=False,
            )
        )
        for input_format in PARITY.INPUT_FORMATS:
            output = tmp_path / scorer / input_format
            completed = subprocess.run(
                [
                    python,
                    "-I",
                    str(SCRIPT),
                    "--evaluator",
                    evaluator,
                    "--scorer",
                    scorer,
                    "--input-format",
                    input_format,
                    "--native-captures",
                    str(captures),
                    "--recipient-python",
                    python,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                cwd=tmp_path,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            result = json.loads(completed.stdout)
            assert result["tamper_rejected"]
            assert result["decision"] == (
                "regression" if scorer == "normalized_nll" else "pass"
            )
            results.append(result)
    assert len(results) == 6


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ("unknown-evaluator", "no maintained profile"),
        ("ragas", "requires an installed recipient"),
    ],
)
def test_selected_sdk_gate_cannot_silently_skip(
    tmp_path, monkeypatch, selected, message
):
    monkeypatch.setenv("INVARLOCK_REQUIRE_EVALUATOR_SDK", selected)
    monkeypatch.delenv("INVARLOCK_EVALUATOR_PARITY_PYTHON", raising=False)
    with pytest.raises(pytest.fail.Exception, match=message):
        test_actual_sdk_exports_reach_installed_recipient(tmp_path, monkeypatch)
