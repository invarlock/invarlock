"""Installed parsers for existing exports; no evaluator SDK or inference required."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.pipeline.comparison import make_run
from invarlock.pipeline.contracts import MAX_INPUT_BYTES, PipelineError, validate

ADAPTERS = ("invarlock", "jsonl", "inspect-json", "lm-eval-samples", "promptfoo-jsonl")


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise PipelineError(f"{label} requires a non-empty array of record objects")
    return value


def _scalar_output(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise PipelineError(
            "ambiguous or non-text native completion; select one response in your evaluator"
        )
    return value


def _scores(values: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, str) and value in ("C", "I"):
            value = 1.0 if value == "C" else 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PipelineError(
                f"score {key!r} is not numeric; provide an explicit SDK mapping"
            )
        result[key] = float(value)
    return result


def _tags(row: dict[str, Any]) -> dict[str, str]:
    # Only string metadata is exposed as slice tags; the original file digest
    # binds other upstream fields, which are not decision inputs.
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise PipelineError("native metadata must be an object")
    return {k: v for k, v in metadata.items() if isinstance(v, str)}


def _inspect(value: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] not in (1, 2)
    ):
        raise PipelineError(
            "inspect-json requires a version 1 or 2 JSON EvalLog (export .eval logs as JSON)"
        )
    if value.get("status") not in ("success", "error", "cancelled"):
        raise PipelineError("Inspect log lacks a recognized completion status")
    if value["status"] != "success":
        raise PipelineError(
            "Inspect run did not complete successfully; supply a complete run"
        )
    result = []
    for row in _rows(value.get("samples"), "Inspect samples"):
        output = row.get("output") or {}
        choices = output.get("choices", [])
        if len(choices) != 1 and row.get("error") is None:
            raise PipelineError("Inspect import requires one completion per sample")
        content = choices[0]["message"]["content"] if choices else ""
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise PipelineError(
                    "Inspect completion contains unsupported non-text content"
                )
            content = "".join(part["text"] for part in content)
        target = row.get("target")
        if isinstance(target, list):
            if len(target) != 1:
                raise PipelineError(
                    "multiple Inspect targets require an explicit scoring adapter"
                )
            target = target[0]
        record_id = _identifier(row["id"])
        if row.get("epoch", 1) != 1:
            raise PipelineError(
                "multiple Inspect epochs require an explicit paired trial mapping"
            )
        result.append(
            {
                "id": record_id,
                "input": row["input"],
                "expected": target,
                "output": content,
                "scores": _scores(row.get("scores") or {}),
                "metadata": _tags(row),
                "error": "upstream_error" if row.get("error") else None,
            }
        )
    return result


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PipelineError("native record IDs must be strings or integers")
    return str(value)


def _harness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if any(
            key not in row for key in ("doc", "target", "arguments", "filtered_resps")
        ):
            raise PipelineError(
                "LM Eval requires --log_samples generation records with doc, target, arguments and filtered_resps"
            )
        scores = {
            key: value
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in ("doc_id",)
        }
        result.append(
            {
                "id": _identifier(row["doc_id"]),
                "input": row["doc"],
                "context": {"arguments": row["arguments"]},
                "expected": row["target"],
                "output": _scalar_output(row["filtered_resps"]),
                "scores": scores,
                "metadata": _tags(row),
                "error": None,
            }
        )
    return result


def _promptfoo_error(row: dict[str, Any]) -> str | None:
    """Distinguish Promptfoo's ASSERT failure from provider or grading ERROR."""
    response = row.get("response") or {}
    error = row.get("error") or response.get("error")
    if "failureReason" not in row:
        # Older exports without typed reasons retain conservative error handling.
        return "upstream_error" if error else None
    reason = row["failureReason"]
    if type(reason) is not int or reason not in (0, 1, 2):
        raise PipelineError("Promptfoo failureReason is unsupported or ambiguous")
    if reason == 2:
        if row.get("success") is not False:
            raise PipelineError("Promptfoo runtime failure contradicts success")
        return "upstream_error"
    grading = row.get("gradingResult")
    if reason == 0:
        if (
            error
            or row.get("success") is False
            or (isinstance(grading, dict) and grading.get("pass") is False)
        ):
            raise PipelineError("Promptfoo failureReason contradicts failure fields")
        return None
    if (
        row.get("success") is not False
        or not isinstance(grading, dict)
        or grading.get("pass") is not False
        or "output" not in response
        or response.get("error")
        or type(row.get("score")) not in (int, float)
        or type(grading.get("score")) not in (int, float)
        or row["score"] != grading["score"]
        or not isinstance(grading.get("reason"), str)
        or row.get("error") != grading["reason"]
    ):
        raise PipelineError("Promptfoo assertion failure fields are inconsistent")
    # Native applyGradingResult uses row.error for an ordinary wrong answer.
    # Preserve that answer so deterministic metrics can still evaluate it.
    return None


def _promptfoo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        case = row.get("testCase")
        if not isinstance(case, dict) or "vars" not in case or "prompt" not in row:
            raise PipelineError(
                "Promptfoo rows need testCase and prompt for pairing; use the full export or SDK capture"
            )
        # Never infer a reference answer from an assertion or successful judgment.
        expected = case.get("metadata", {}).get("invarlock_expected")
        prompt = row["prompt"]
        prompt = prompt.get("raw") if isinstance(prompt, dict) else prompt
        if not isinstance(prompt, str):
            raise PipelineError("Promptfoo row requires the actual rendered prompt")
        result.append(
            {
                "id": _identifier(row["testIdx"]) + ":" + _identifier(row["promptIdx"]),
                "input": case["vars"],
                "context": {"prompt": prompt},
                "expected": expected,
                "output": (row.get("response") or {}).get("output"),
                "scores": _scores(
                    {
                        k: row[k]
                        for k in ("score", "latencyMs", "cost")
                        if row.get(k) is not None
                    }
                ),
                "metadata": _tags(case),
                "error": _promptfoo_error(row),
            }
        )
    return result


def load_run(
    path: str | Path,
    *,
    adapter: str = "invarlock",
    source: dict[str, str] | None = None,
    run_id: str | None = None,
    artifact_digest: str | None = None,
    score_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Import a native export or a canonical run with explicit source identities."""
    if adapter not in ADAPTERS:
        raise PipelineError(
            f"unsupported adapter {adapter!r}; choose {', '.join(ADAPTERS)}"
        )
    try:
        raw = read_regular_file_bytes(
            Path(path), label="evaluation export", max_bytes=MAX_INPUT_BYTES
        )
        if adapter in ("jsonl", "lm-eval-samples", "promptfoo-jsonl"):
            rows = _rows(
                [
                    parse_json_bytes(line, label=f"export line {i}")
                    for i, line in enumerate(raw.splitlines(), 1)
                    if line.strip()
                ],
                adapter,
            )
            records = (
                _harness(rows)
                if adapter == "lm-eval-samples"
                else _promptfoo(rows)
                if adapter == "promptfoo-jsonl"
                else rows
            )
        else:
            value = parse_json_bytes(raw, label="evaluation export")
            if adapter == "invarlock":
                validate(value, "run")
                if any(
                    v is not None
                    for v in (source, run_id, artifact_digest, score_provenance)
                ):
                    raise PipelineError(
                        "canonical run identities cannot be overridden at import"
                    )
                return cast(dict[str, Any], value)
            records = _inspect(value)
        if source is None or run_id is None or artifact_digest is None:
            raise PipelineError(
                "native import requires source name/version, run_id and artifact_digest from your pipeline"
            )
        return make_run(
            records,
            source=source,
            run_id=run_id,
            artifact_digest=artifact_digest,
            score_provenance=score_provenance,
            source_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OSError,
        OverflowError,
        RecursionError,
    ) as exc:
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(f"cannot import {adapter}: {exc}") from exc
