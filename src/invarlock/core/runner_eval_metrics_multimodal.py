from __future__ import annotations

import json
import math
import re
import time
from inspect import getattr_static
from typing import Any

from invarlock.core.metric_kind_contract import normalize_metric_kind


def _model_kwargs(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in prepared.items()
        if isinstance(key, str) and not key.startswith("_")
    }


def _resolve_adapter_hook(adapter: Any, name: str) -> Any | None:
    if adapter is None:
        return None
    try:
        getattr_static(adapter, name)
    except AttributeError:
        return None
    hook = getattr(adapter, name, None)
    return hook if callable(hook) else None


def _normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _decode_prediction_text(decoded: Any) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, (list, tuple)):
        return str(decoded[0] if decoded else "").strip()
    return str(decoded).strip() if decoded is not None else ""


_JSON_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"', re.DOTALL)


def _prediction_answer_text(prediction: Any) -> str:
    text = _decode_prediction_text(prediction)
    if not text:
        return ""
    candidates = [text]
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("` \n")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str):
                return answer.strip()
    match = _JSON_ANSWER_RE.search(text)
    if match:
        raw_answer = match.group("answer")
        try:
            return str(json.loads(f'"{raw_answer}"')).strip()
        except json.JSONDecodeError:
            return raw_answer.strip()
    return text


def _normalize_reference_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []
    return [str(item).strip() for item in candidates if str(item).strip()]


def _replay_input_record(batch: dict[str, Any], *, example_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {"id": example_id, "example_id": example_id}
    for key in (
        "image_path",
        "prompt",
        "answer",
        "answers",
        "image_sha256",
        "prompt_sha256",
        "answer_sha256",
        "source_file",
        "source_line",
    ):
        value = batch.get(key)
        if value is not None:
            record[key] = value
    return record


def _is_multimodal_batch(batch: Any) -> bool:
    return isinstance(batch, dict) and (
        "image_path" in batch or "example_id" in batch or "answers" in batch
    )


def _resolve_metric_kind(config: Any, *, fallback: str) -> str:
    if not config or not isinstance(getattr(config, "context", None), dict):
        return fallback
    eval_section = config.context.get("eval")
    if not isinstance(eval_section, dict):
        return fallback
    metric_section = eval_section.get("metric")
    if isinstance(metric_section, dict):
        kind = normalize_metric_kind(metric_section.get("kind"), allow_auto=True)
        if kind:
            return kind
    return fallback


def _evaluate_vision_text_arm(
    model: Any,
    batches: list[dict[str, Any]],
    *,
    adapter: Any,
    device: Any,
) -> tuple[dict[str, Any], float]:
    import torch

    prepare_model_inputs = _resolve_adapter_hook(adapter, "prepare_model_inputs")
    if not callable(prepare_model_inputs):
        raise RuntimeError("Multimodal adapter missing prepare_model_inputs()")
    prepare_generation_inputs = _resolve_adapter_hook(
        adapter, "prepare_generation_inputs"
    )
    if not callable(prepare_generation_inputs):
        raise RuntimeError("Multimodal adapter missing prepare_generation_inputs()")
    decode_generated = _resolve_adapter_hook(adapter, "decode_generated")
    if not callable(decode_generated):
        raise RuntimeError("Multimodal adapter missing decode_generated()")

    total_weighted_log_loss = 0.0
    total_answer_tokens = 0
    log_losses: list[float] = []
    token_counts: list[int] = []
    example_correct: list[int] = []
    records: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    example_ids: list[str] = []
    processor_sha: str | None = None
    latency_ms = 0.0

    for batch in batches:
        prepared = prepare_model_inputs(batch, device, include_labels=True)
        prepared_kwargs = _model_kwargs(prepared)
        with torch.no_grad():
            outputs = model(**prepared_kwargs)
        loss_val = (
            float(outputs.loss.item())
            if hasattr(outputs, "loss") and hasattr(outputs.loss, "item")
            else float("nan")
        )
        answer_token_count = int(prepared.get("_answer_token_count", 0) or 0)
        if math.isfinite(loss_val) and answer_token_count > 0:
            total_weighted_log_loss += float(loss_val) * answer_token_count
            total_answer_tokens += answer_token_count
            log_losses.append(float(loss_val))
            token_counts.append(answer_token_count)

        generation_inputs = prepare_generation_inputs(batch, device)
        generation_kwargs = _model_kwargs(generation_inputs)
        max_new_tokens = int(generation_inputs.get("_max_new_tokens", 32) or 32)
        generation_kwargs.setdefault("max_new_tokens", max_new_tokens)
        generation_kwargs.setdefault("do_sample", False)
        generation_kwargs.setdefault("use_cache", True)
        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(**generation_kwargs)
        latency_ms += (time.perf_counter() - start) * 1000.0

        decoded = decode_generated(generated_ids, generation_inputs)
        prediction = _decode_prediction_text(decoded)
        prediction_answer = _prediction_answer_text(decoded)
        references = _normalize_reference_answers(
            generation_inputs.get("_reference_answers", [])
        )
        normalized_prediction = _normalize_answer_text(prediction_answer)
        correct = int(
            any(
                normalized_prediction == _normalize_answer_text(reference)
                for reference in references
            )
        )
        example_correct.append(correct)
        example_id = str(
            generation_inputs.get("_example_id")
            or batch.get("example_id")
            or batch.get("id")
            or ""
        )
        example_ids.append(example_id)
        input_records.append(_replay_input_record(batch, example_id=example_id))
        if processor_sha is None:
            candidate = generation_inputs.get("_processor_sha256")
            if isinstance(candidate, str) and candidate:
                processor_sha = candidate
        records.append(
            {
                "id": example_id,
                "prediction": prediction,
                "prediction_answer": prediction_answer,
                "references": references,
                "correct": bool(correct),
                "image_sha256": batch.get("image_sha256"),
                "prompt_sha256": batch.get("prompt_sha256"),
                "answer_sha256": batch.get("answer_sha256"),
            }
        )

    total = len(example_correct)
    correct_total = int(sum(example_correct))
    accuracy = float(correct_total / total) if total > 0 else float("nan")
    mean_logloss = (
        float(total_weighted_log_loss / total_answer_tokens)
        if total_answer_tokens > 0
        else float("nan")
    )
    payload: dict[str, Any] = {
        "correct_total": correct_total,
        "total": total,
        "example_correct": example_correct,
        "accuracy": accuracy,
        "records": records,
        "example_ids": example_ids,
        "logloss": log_losses,
        "token_counts": token_counts,
        "total_tokens": total_answer_tokens,
        "mean_logloss": mean_logloss,
    }
    if input_records:
        payload["input_records"] = input_records
    if processor_sha:
        payload["processor_sha256"] = processor_sha
    return payload, latency_ms


def _build_multimodal_eval_result(
    model: Any,
    preview_data: list[dict[str, Any]],
    final_data: list[dict[str, Any]],
    *,
    adapter: Any,
    device: Any,
    config: Any,
    process: Any,
    initial_memory: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preview_payload, preview_latency_ms = _evaluate_vision_text_arm(
        model,
        list(preview_data),
        adapter=adapter,
        device=device,
    )
    final_payload, final_latency_ms = _evaluate_vision_text_arm(
        model,
        list(final_data),
        adapter=adapter,
        device=device,
    )
    current_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = max(initial_memory, current_memory)
    total_tokens = int(preview_payload["total_tokens"]) + int(
        final_payload["total_tokens"]
    )
    latency_ms_per_tok = (
        float((preview_latency_ms + final_latency_ms) / total_tokens)
        if total_tokens > 0
        else 0.0
    )
    metric_kind = _resolve_metric_kind(config, fallback="accuracy")
    preview_accuracy = float(preview_payload.get("accuracy", float("nan")))
    final_accuracy = float(final_payload.get("accuracy", float("nan")))
    preview_total = int(preview_payload.get("total", 0))
    final_total = int(final_payload.get("total", 0))
    paired_windows = min(preview_total, final_total)
    pairing_reason = None if paired_windows > 0 else "no_pairs"
    primary_metric = {
        "kind": metric_kind,
        "preview": preview_accuracy if math.isfinite(preview_accuracy) else None,
        "final": final_accuracy if math.isfinite(final_accuracy) else None,
        "invalid": not (
            math.isfinite(preview_accuracy) and math.isfinite(final_accuracy)
        ),
        "degraded": False,
        "counts_source": "measured",
        "estimated": False,
        "n_preview": preview_total,
        "n_final": final_total,
    }
    metrics = {
        "primary_metric": primary_metric,
        "classification": {
            "preview": {
                "correct_total": int(preview_payload["correct_total"]),
                "total": int(preview_payload["total"]),
                "example_correct": list(preview_payload["example_correct"]),
            },
            "final": {
                "correct_total": int(final_payload["correct_total"]),
                "total": int(final_payload["total"]),
                "example_correct": list(final_payload["example_correct"]),
            },
            "n_correct": int(final_payload["correct_total"]),
            "n_total": int(final_payload["total"]),
            "counts_source": "measured",
            "estimated": False,
        },
        "accuracy": final_accuracy if math.isfinite(final_accuracy) else None,
        "logloss_preview": float(preview_payload.get("mean_logloss", float("nan"))),
        "logloss_final": float(final_payload.get("mean_logloss", float("nan"))),
        "logloss_delta": (
            float(final_payload.get("mean_logloss", float("nan")))
            - float(preview_payload.get("mean_logloss", float("nan")))
        ),
        "latency_ms_per_tok": latency_ms_per_tok,
        "memory_mb_peak": peak_memory,
        "eval_samples": int(preview_payload.get("total", 0))
        + int(final_payload.get("total", 0)),
        "total_tokens": total_tokens,
        "preview_total_tokens": int(preview_payload["total_tokens"]),
        "final_total_tokens": int(final_payload["total_tokens"]),
        "window_overlap_fraction": 0.0,
        "window_match_fraction": 1.0,
        "window_pairing_reason": pairing_reason,
        "window_pairing_preview": {
            "matched": preview_total,
            "expected": preview_total,
            "reason": pairing_reason,
        },
        "window_pairing_final": {
            "matched": final_total,
            "expected": final_total,
            "reason": pairing_reason,
        },
        "paired_windows": paired_windows,
    }
    eval_windows = {
        "preview": {
            "example_ids": list(preview_payload["example_ids"]),
            "records": list(preview_payload["records"]),
            "input_records": list(preview_payload.get("input_records", [])),
            "logloss": list(preview_payload["logloss"]),
            "token_counts": list(preview_payload["token_counts"]),
            "processor_sha256": preview_payload.get("processor_sha256"),
        },
        "final": {
            "example_ids": list(final_payload["example_ids"]),
            "records": list(final_payload["records"]),
            "input_records": list(final_payload.get("input_records", [])),
            "logloss": list(final_payload["logloss"]),
            "token_counts": list(final_payload["token_counts"]),
            "processor_sha256": final_payload.get("processor_sha256")
            or preview_payload.get("processor_sha256"),
        },
    }
    return metrics, eval_windows


__all__ = [
    "_build_multimodal_eval_result",
    "_decode_prediction_text",
    "_evaluate_vision_text_arm",
    "_is_multimodal_batch",
    "_model_kwargs",
    "_normalize_answer_text",
    "_normalize_reference_answers",
    "_prediction_answer_text",
    "_resolve_adapter_hook",
    "_resolve_metric_kind",
]
