from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEMANTICS = ROOT / "examples/evaluator-qualification/maintained/inspect_semantics.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "inspect_literal_semantics", SEMANTICS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("output", "reference"),
    [
        (" answer", " answer"),
        ("answer\n", "answer\n"),
        ("!answer!", "!answer!"),
        ("answer", "different"),
        ("a b", "a  b"),
        ("a,b", "ab"),
        ("A", "a"),
        ("é", "e\u0301"),
        ("“yes”", "yes"),
        ("", ""),
        ("", "no"),
    ],
)
def test_literal_pair_domain_keeps_equal_whitespace_and_distinct_mismatches(
    output: str, reference: str
) -> None:
    module = _module()
    case = {"record_id": "one", "output": output, "reference": reference}
    module.validate_cases([case])
    result = SimpleNamespace(
        value="C" if output == reference else "I",
        answer=output.strip(),
        explanation=output,
    )
    score, detail = module.project_result(case, result)
    assert score == float(output == reference)
    assert detail["answer"] == output.strip()


@pytest.mark.parametrize(
    ("output", "reference"),
    [
        (" answer", "answer"),
        ("answer\n", "answer"),
        ("answer!", "answer"),
        ("\u00a0answer", "answer"),
        ("", "  "),
        ("!", ""),
        ("answer", ["answer", "other"]),
        ("answer", ["answer"]),
    ],
)
def test_unsupported_normalization_collisions_and_multi_target_fail_explicitly(
    output: object, reference: object
) -> None:
    module = _module()
    with pytest.raises(ValueError, match="unsupported Inspect literal pair"):
        module.validate_cases(
            [{"record_id": "one", "output": output, "reference": reference}]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"value": "I"},
        {"value": "P"},
        {"value": 1.0},
        {"answer": "other"},
        {"explanation": "other"},
    ],
)
def test_native_contradictions_are_not_projected_as_literal_scores(changes) -> None:
    module = _module()
    result = {"value": "C", "answer": "answer", "explanation": "answer"} | changes
    with pytest.raises(ValueError, match="Inspect native"):
        module.project_result(
            {"record_id": "one", "output": "answer", "reference": "answer"},
            SimpleNamespace(**result),
        )


def test_duplicate_record_ids_are_rejected_before_upstream_execution() -> None:
    module = _module()
    case = {"record_id": "same", "output": "a", "reference": "a"}
    with pytest.raises(ValueError, match="unique nonempty record IDs"):
        module.validate_cases([case, case])
