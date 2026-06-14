#!/usr/bin/env python3
"""Audit named model-evidence candidates before GPU execution.

The check is intentionally offline. It verifies repo contracts, presets, adapter
auto-routing, materialization metadata, and resource hints without downloading
models or datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_EVIDENCE_DIR = REPO_ROOT / "scripts" / "model_evidence"
SRC_DIR = REPO_ROOT / "src"

for path in (MODEL_EVIDENCE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_evidence_lanes import (  # noqa: E402
    CATALOG_PRESET_OVERRIDES,
    MODEL_CATALOG_GPU_SUITE,
    MODEL_FAMILY_CATALOG_PATH,
    SUITES,
    SUPPORT_MATRIX_BACKLOG_GPU_SUITE,
    SUPPORT_MATRIX_PATH,
    EvidenceLane,
    lane_resource_estimate,
)

from invarlock.adapters.auto import resolve_auto_adapter  # noqa: E402

ALLOWED_ADAPTERS = {"auto", "hf_causal", "hf_mlm", "hf_multimodal", "hf_seq2seq"}
EXPECTED_PROVIDER_KINDS = {
    "hf_causal": {"hf_text", "local_jsonl", "text", "wikitext2"},
    "hf_mlm": {"hf_text", "local_jsonl", "text", "wikitext2"},
    "hf_multimodal": {"vision_text"},
    "hf_seq2seq": {"hf_seq2seq", "seq2seq"},
}
EXPECTED_LOSS_TYPES = {
    "hf_causal": {"causal"},
    "hf_mlm": {"mlm", "masked_lm"},
    "hf_multimodal": {"classification", "vision_text"},
    "hf_seq2seq": {"seq2seq"},
}


@dataclass(frozen=True)
class Finding:
    severity: str
    scope: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "scope": self.scope, "message": self.message}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _provider_kind(provider: Any) -> str | None:
    if isinstance(provider, str):
        return provider
    if isinstance(provider, Mapping):
        kind = provider.get("kind")
        return str(kind) if kind is not None else None
    return None


def _all_lanes() -> list[tuple[str, EvidenceLane]]:
    seen: set[tuple[str, str]] = set()
    lanes: list[tuple[str, EvidenceLane]] = []
    for suite_name, suite_lanes in SUITES.items():
        for lane in suite_lanes:
            key = (lane.slug, lane.lane_id)
            if key in seen:
                continue
            seen.add(key)
            lanes.append((suite_name, lane))
    return lanes


def _model_ids_from_catalog() -> set[str]:
    payload = _load_json(MODEL_FAMILY_CATALOG_PATH)
    model_ids: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"representative_model", "model_id"} and isinstance(
                    item, str
                ):
                    model_ids.add(item)
                elif key == "representative_models" and isinstance(item, list):
                    model_ids.update(model for model in item if isinstance(model, str))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    return model_ids


def _support_matrix_rows() -> dict[str, dict[str, Any]]:
    payload = _load_json(SUPPORT_MATRIX_PATH)
    return {
        str(lane["lane_id"]): lane
        for lane in payload.get("lanes", [])
        if isinstance(lane, dict) and isinstance(lane.get("lane_id"), str)
    }


def _requires_large_model_loading_defaults(
    model_id: str,
    estimate: Mapping[str, Any] | None,
) -> bool:
    lower = model_id.lower()
    if not estimate:
        return any(hint in lower for hint in ("34b", "72b"))
    try:
        recommended_gpus = float(estimate.get("recommended_min_gpus_80gb", 1))
        estimated_weight_gb = float(estimate.get("estimated_weight_gb_bf16", 0))
    except (TypeError, ValueError):
        return False
    return recommended_gpus > 1 or estimated_weight_gb >= 40


def _is_public_materialized_lane(suite_name: str, lane: EvidenceLane) -> bool:
    return bool(
        suite_name == SUPPORT_MATRIX_BACKLOG_GPU_SUITE
        and lane.adapter == "hf_multimodal"
        and "public_vqav2" in lane.preset_relpath
    )


def _check_lane(
    suite_name: str,
    lane: EvidenceLane,
    support_rows: Mapping[str, Mapping[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    scope = f"{suite_name}:{lane.slug}"
    if lane.adapter not in ALLOWED_ADAPTERS:
        findings.append(Finding("error", scope, f"unknown adapter {lane.adapter!r}"))

    preset = REPO_ROOT / lane.preset_relpath
    if not preset.is_file():
        findings.append(
            Finding("error", scope, f"missing preset {lane.preset_relpath}")
        )
        return findings

    data = _load_yaml(preset)
    preset_adapter = _nested(data, "model", "adapter")
    effective_adapter = lane.adapter
    if lane.adapter == "auto" and preset_adapter is not None:
        effective_adapter = str(preset_adapter)

    expected_auto = resolve_auto_adapter(lane.model_id)
    if effective_adapter != "auto" and expected_auto != effective_adapter:
        findings.append(
            Finding(
                "error",
                scope,
                f"adapter:auto resolves {lane.model_id!r} to {expected_auto!r}, "
                f"but the lane uses {effective_adapter!r}",
            )
        )

    if (
        preset_adapter is not None
        and lane.adapter != "auto"
        and str(preset_adapter) != lane.adapter
    ):
        findings.append(
            Finding(
                "error",
                scope,
                f"preset adapter {preset_adapter!r} does not match lane adapter "
                f"{lane.adapter!r}",
            )
        )

    provider = _nested(data, "dataset", "provider")
    provider_kind = _provider_kind(provider)
    expected_kinds = EXPECTED_PROVIDER_KINDS.get(effective_adapter)
    if expected_kinds and provider_kind not in expected_kinds:
        findings.append(
            Finding(
                "error",
                scope,
                f"dataset provider kind {provider_kind!r} is not valid for "
                f"{effective_adapter}; expected one of {sorted(expected_kinds)}",
            )
        )

    loss_type = _nested(data, "eval", "loss", "type")
    expected_loss_types = EXPECTED_LOSS_TYPES.get(effective_adapter)
    if (
        expected_loss_types
        and loss_type is not None
        and str(loss_type) not in expected_loss_types
    ):
        findings.append(
            Finding(
                "error",
                scope,
                f"loss type {loss_type!r} is not valid for {effective_adapter}; "
                f"expected one of {sorted(expected_loss_types)}",
            )
        )

    materialization = lane.vision_text_materialization
    if _is_public_materialized_lane(suite_name, lane):
        if not materialization:
            findings.append(
                Finding(
                    "error",
                    scope,
                    "public vision-text lane lacks materialization metadata",
                )
            )
        elif (
            materialization.get("dataset")
            != "Multimodal-Fatima/VQAv2_sample_validation"
        ):
            findings.append(
                Finding(
                    "error",
                    scope,
                    "public vision-text lane must materialize the pinned VQAv2 sample",
                )
            )

    estimate = lane_resource_estimate(lane.model_id)
    preset_model_id = _nested(data, "model", "id")
    model_specific_preset = preset_model_id in {None, lane.model_id}
    if (
        suite_name != MODEL_CATALOG_GPU_SUITE
        and model_specific_preset
        and _requires_large_model_loading_defaults(lane.model_id, estimate)
    ):
        model_cfg = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
        if model_cfg.get("device_map") != "auto":
            findings.append(
                Finding(
                    "error", scope, "large/MoE lane should set model.device_map=auto"
                )
            )
        if model_cfg.get("low_cpu_mem_usage") is not True:
            findings.append(
                Finding(
                    "error", scope, "large/MoE lane should set low_cpu_mem_usage=true"
                )
            )
        if str(model_cfg.get("dtype", "")).lower() not in {
            "auto",
            "bfloat16",
            "float16",
        }:
            findings.append(
                Finding(
                    "error",
                    scope,
                    "large/MoE lane should pin auto/bfloat16/float16 dtype",
                )
            )
        if model_cfg.get("collect_loading_info") is not False:
            findings.append(
                Finding(
                    "error",
                    scope,
                    "large/MoE lane should disable optional loading-info collection",
                )
            )

    if lane.lane_id in support_rows:
        matrix_adapter = support_rows[lane.lane_id].get("adapter")
        if matrix_adapter != lane.adapter:
            findings.append(
                Finding(
                    "error",
                    scope,
                    f"support matrix adapter {matrix_adapter!r} disagrees with lane",
                )
            )

    return findings


def audit() -> list[Finding]:
    findings: list[Finding] = []
    support_rows = _support_matrix_rows()
    lane_model_ids: set[str] = set()

    for suite_name, lane in _all_lanes():
        lane_model_ids.add(lane.model_id)
        findings.extend(_check_lane(suite_name, lane, support_rows))

    override_model_ids = set(CATALOG_PRESET_OVERRIDES)
    catalog_model_ids = _model_ids_from_catalog()
    missing_catalog_routes = sorted(
        catalog_model_ids - lane_model_ids - override_model_ids
    )
    for model_id in missing_catalog_routes:
        findings.append(
            Finding(
                "error",
                "model_family_catalog",
                f"representative model {model_id!r} lacks a lane or preset override",
            )
        )

    return sorted(findings, key=lambda item: (item.severity, item.scope, item.message))


def _print_text(findings: Sequence[Finding]) -> None:
    if not findings:
        print("Model candidate compatibility OK.")
        return
    print("Model candidate compatibility failures:", file=sys.stderr)
    for finding in findings:
        print(
            f"  {finding.severity.upper()} {finding.scope}: {finding.message}",
            file=sys.stderr,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    findings = audit()
    payload = {
        "schema": "invarlock/model-candidate-compatibility-audit-v1",
        "ok": not findings,
        "finding_count": len(findings),
        "findings": [finding.as_dict() for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(findings)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
