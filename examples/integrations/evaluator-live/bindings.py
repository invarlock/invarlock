"""Explicit capture bindings around unchanged local runtime observations.

The SDK records measurements made by the separate model worker. Its source and
input wrapper identify the capture, not the implementation that computed logits.
The original response remains in the transport ledger and in record metadata.
"""

from __future__ import annotations

from copy import deepcopy

import common


def native_input(evaluator, case):
    if evaluator == "lm-evaluation-harness":
        return deepcopy(case)
    if evaluator == "promptfoo":
        return {"prompt": case["input"], "case_id": case["id"]}
    return case["input"]


def projection(evaluator):
    pointer = {
        "lm-evaluation-harness": "/input/input",
        "promptfoo": "/input/prompt",
    }.get(evaluator)
    return {"kind": "json-pointer", "pointer": pointer} if pointer else None


def bind_result(result, case, evaluator, version):
    """Add a declared capture binding without modifying the worker response."""
    value = deepcopy(result)
    metadata = value["metadata"]
    if {"invarlock_capture_binding", "invarlock_task_outcome"} & metadata.keys():
        raise ValueError("runtime response cannot supply a capture binding")
    metadata["invarlock_task_outcome"] = {
        "output": result["output"],
        "error": result.get("error"),
    }
    original = metadata.get("invarlock_likelihood")
    if original is None:
        return value
    execution = metadata.get("invarlock_model_execution", {})
    if (
        original["input_digest"] != common.digest(case["input"])
        or original["reference_digest"] != common.digest(case["expected"])
        or original["utf8_byte_count"] != len(case["expected"].encode("utf-8"))
        or original["source"] != execution.get("source")
        or original["artifact_digest"]
        != execution.get("model", {}).get("artifact_digest")
        or original["configuration_digest"]
        != common.digest(execution.get("configuration"))
        or original["tokenizer_digest"]
        != execution.get("model", {}).get("tokenizer_digest")
    ):
        raise ValueError("runtime likelihood differs from its task or execution")
    captured_input = native_input(evaluator, case)
    facts = {
        **original,
        "input_digest": common.digest(captured_input),
        "source": {"name": evaluator, "version": version},
    }
    metadata["invarlock_capture_binding"] = {
        "format": "invarlock/live-likelihood-capture-v1",
        "original_likelihood": deepcopy(original),
        "native_input": captured_input,
        "input_projection": projection(evaluator),
        "role": "SDK capture of independently retained model-worker measurements",
    }
    metadata["invarlock_likelihood"] = facts
    return value


def rebind_harness_metadata(metadata, case, cases, document):
    """Bind only the SDK's exact, schedule-derived nullable metadata expansion."""
    planned = common.cases(cases)
    if sum(common.encoded(row) == common.encoded(case) for row in planned) != 1:
        raise ValueError("Harness document is outside the frozen case schedule")
    value = deepcopy(metadata)
    if "invarlock_serialization_binding" in value:
        raise ValueError("Harness serialization binding already exists")
    if common.encoded(document) == common.encoded(case):
        return value
    keys = sorted({key for row in planned for key in row["metadata"]})
    expected = {
        **deepcopy(case),
        "metadata": {key: case["metadata"].get(key) for key in keys},
    }
    if common.encoded(document) != common.encoded(expected):
        raise ValueError(
            "Harness input projection document differs from the exact nullable metadata expansion"
        )
    facts = value.get("invarlock_likelihood")
    if facts is not None:
        if (
            not isinstance(facts, dict)
            or facts.get("input_digest") != common.digest(case)
            or facts.get("reference_digest") != common.digest(case["expected"])
            or facts.get("source")
            != {"name": "lm-evaluation-harness", "version": "0.4.12"}
        ):
            raise ValueError(
                "Harness likelihood lacks its original frozen input binding"
            )
        value["invarlock_likelihood"] = {
            **facts,
            "input_digest": common.digest(document),
        }
    binding = value.get("invarlock_capture_binding")
    if binding is not None:
        if (
            not isinstance(binding, dict)
            or common.encoded(binding.get("native_input")) != common.encoded(case)
            or binding.get("input_projection") != projection("lm-evaluation-harness")
        ):
            raise ValueError(
                "Harness capture binding differs from its original document"
            )
        value["invarlock_capture_binding"] = {
            **binding,
            "native_input": deepcopy(document),
        }
    value["invarlock_serialization_binding"] = {
        "format": "invarlock/harness-nullable-metadata-v1",
        "original_input": deepcopy(case),
        "serialized_input": deepcopy(document),
        "nullable_metadata_fields": sorted(set(keys) - case["metadata"].keys()),
        "original_likelihood": deepcopy(facts),
    }
    return value


def check_record(
    record, result, case, evaluator, version, *, cases=None, service_identity=None
):
    """Recipient-side check of normalized facts against the raw task ledger."""
    bound = bind_result(result, case, evaluator, version)
    retained = record["context"].get("input_projection")
    expected_input = native_input(evaluator, case)
    if evaluator == "lm-evaluation-harness" and retained:
        document = retained["source"]["input"]
        bound["metadata"] = rebind_harness_metadata(
            bound["metadata"], case, cases if cases is not None else [case], document
        )
        expected_input = document
    if service_identity is not None and "invarlock_likelihood" in bound["metadata"]:
        bound["metadata"]["invarlock_likelihood"] = common.module(
            "http_service"
        ).hosted_facts(bound["metadata"]["invarlock_likelihood"], service_identity)
    if (
        record["id"] != case["id"]
        or record["input"] != case["input"]
        or record["expected"] != case["expected"]
        or record["output"] != result["output"]
        or common.encoded(record["metadata"])
        != common.encoded(
            {
                key: value
                for key, value in case["metadata"].items()
                if isinstance(value, str)
            }
        )
        or bool(record.get("error")) != bool(result.get("error"))
        or record.get("likelihood") != bound["metadata"].get("invarlock_likelihood")
    ):
        raise ValueError("normalized record differs from the original model task")
    if projection(evaluator):
        if not retained or common.encoded(
            retained["source"]["input"]
        ) != common.encoded(expected_input):
            raise ValueError(
                "native input wrapper differs from the declared projection"
            )
    elif retained:
        raise ValueError("unexpected task input transformation")
    return bound
