#!/usr/bin/env python3
"""Audit the static public evidence catalog against model contracts.

The check is intentionally offline. It verifies catalog entries, presets, adapter
auto-routing, and dataset-provider contracts without scheduling or placement data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
EVIDENCE_CATALOG_PATH = REPO_ROOT / "contracts" / "evidence_catalog_v1.json"
MODEL_FAMILY_CATALOG_PATH = REPO_ROOT / "contracts" / "model_family_catalog.json"
SUPPORT_MATRIX_PATH = REPO_ROOT / "contracts" / "support_matrix.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from invarlock.adapters.auto import resolve_auto_adapter  # noqa: E402
from invarlock.model_family_registry import ModelFamilyRouteIndex  # noqa: E402

CATALOG_FORMAT = "invarlock/evidence-catalog-v1"
ALLOWED_ADAPTERS = {"hf_causal", "hf_mlm", "hf_multimodal", "hf_seq2seq"}
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


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


def _catalog_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = payload.get("entries")
    return (
        [entry for entry in entries if isinstance(entry, Mapping)]
        if isinstance(entries, list)
        else []
    )


def _model_ids_from_catalog(payload: Mapping[str, Any]) -> set[str]:
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


def _support_matrix_rows(
    payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        return {}
    return {
        str(lane["lane_id"]): lane
        for lane in lanes
        if isinstance(lane, Mapping) and isinstance(lane.get("lane_id"), str)
    }


def _allows_explicit_task_adapter(
    *,
    model_id: str,
    multi_adapter_model_ids: set[str],
    effective_adapter: str,
    preset_adapter: Any,
    provider_kind: str | None,
    loss_type: Any,
) -> bool:
    """Allow explicit task adapters only for cataloged multi-task checkpoints."""

    if model_id not in multi_adapter_model_ids:
        return False
    if preset_adapter is None or str(preset_adapter) != effective_adapter:
        return False
    expected_kinds = EXPECTED_PROVIDER_KINDS.get(effective_adapter)
    if expected_kinds and provider_kind not in expected_kinds:
        return False
    expected_loss_types = EXPECTED_LOSS_TYPES.get(effective_adapter)
    return not (
        expected_loss_types
        and loss_type is not None
        and str(loss_type) not in expected_loss_types
    )


def _check_entry(
    entry: Mapping[str, Any],
    support_rows: Mapping[str, Mapping[str, Any]],
    multi_adapter_model_ids: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    lane_id = entry.get("lane_id")
    scope = f"evidence_catalog:{lane_id if isinstance(lane_id, str) else '<unknown>'}"
    model = entry.get("model")
    preset_spec = entry.get("preset")
    inputs = entry.get("inputs")
    if not isinstance(model, Mapping):
        return [Finding("error", scope, "model must be an object")]
    if not isinstance(preset_spec, Mapping):
        return [Finding("error", scope, "preset must be an object")]
    if not isinstance(inputs, Mapping):
        return [Finding("error", scope, "inputs must be an object")]

    model_id = model.get("id")
    adapter = model.get("adapter")
    preset_relpath = preset_spec.get("path")
    if not isinstance(model_id, str) or not model_id:
        return [Finding("error", scope, "model.id is required")]
    if adapter not in ALLOWED_ADAPTERS:
        return [Finding("error", scope, f"unknown adapter {adapter!r}")]
    if not isinstance(preset_relpath, str) or not preset_relpath:
        return [Finding("error", scope, "preset.path is required")]

    preset = REPO_ROOT / preset_relpath
    if not preset.is_file():
        return [Finding("error", scope, f"missing preset {preset_relpath}")]
    expected_digest = preset_spec.get("sha256")
    observed_digest = "sha256:" + hashlib.sha256(preset.read_bytes()).hexdigest()
    if expected_digest != observed_digest:
        findings.append(Finding("error", scope, "preset digest does not match bytes"))

    data = _load_yaml(preset)
    preset_adapter = _nested(data, "model", "adapter")
    effective_adapter = str(adapter)
    provider_kind = _provider_kind(_nested(data, "dataset", "provider"))
    loss_type = _nested(data, "eval", "loss", "type")
    expected_auto = resolve_auto_adapter(model_id)
    explicit_task_adapter = _allows_explicit_task_adapter(
        model_id=model_id,
        multi_adapter_model_ids=multi_adapter_model_ids,
        effective_adapter=effective_adapter,
        preset_adapter=preset_adapter,
        provider_kind=provider_kind,
        loss_type=loss_type,
    )
    if expected_auto != effective_adapter and not explicit_task_adapter:
        findings.append(
            Finding(
                "error",
                scope,
                f"adapter:auto resolves {model_id!r} to {expected_auto!r}, "
                f"but the catalog uses {effective_adapter!r}",
            )
        )

    if preset_adapter is not None and str(preset_adapter) != effective_adapter:
        findings.append(
            Finding(
                "error",
                scope,
                f"preset adapter {preset_adapter!r} does not match catalog adapter "
                f"{effective_adapter!r}",
            )
        )

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
    source = inputs.get("source")
    source_provider = source.get("provider") if isinstance(source, Mapping) else None
    if source_provider != provider_kind:
        findings.append(
            Finding(
                "error",
                scope,
                "catalog input provider disagrees with the preset provider",
            )
        )

    input_kind = inputs.get("kind")
    if effective_adapter == "hf_multimodal" and input_kind != "vision_text":
        findings.append(
            Finding("error", scope, "multimodal entries require vision_text inputs")
        )
    if effective_adapter != "hf_multimodal" and input_kind == "vision_text":
        findings.append(
            Finding("error", scope, "vision_text inputs require hf_multimodal")
        )

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

    support = support_rows.get(str(lane_id))
    if support is None:
        findings.append(Finding("error", scope, "missing support-matrix lane"))
    else:
        if support.get("adapter") != effective_adapter:
            findings.append(
                Finding("error", scope, "support-matrix adapter disagrees with catalog")
            )
        representatives = support.get("representative_models")
        if (
            isinstance(representatives, list)
            and representatives
            and model_id not in representatives
        ):
            findings.append(
                Finding(
                    "error",
                    scope,
                    "catalog model is not a support-matrix representative",
                )
            )
    return findings


def audit() -> list[Finding]:
    findings: list[Finding] = []
    evidence_catalog = _load_json(EVIDENCE_CATALOG_PATH)
    family_catalog = _load_json(MODEL_FAMILY_CATALOG_PATH)
    support_matrix = _load_json(SUPPORT_MATRIX_PATH)
    entries = _catalog_entries(evidence_catalog)
    support_rows = _support_matrix_rows(support_matrix)

    if evidence_catalog.get("format_version") != CATALOG_FORMAT:
        findings.append(
            Finding("error", "evidence_catalog", "unexpected format_version")
        )
    if evidence_catalog.get("entry_count") != len(entries):
        findings.append(
            Finding("error", "evidence_catalog", "entry_count does not match entries")
        )

    adapters_by_model_id: dict[str, set[str]] = {}
    catalog_model_ids: set[str] = set()
    catalog_lane_ids: set[str] = set()
    for entry in entries:
        model = entry.get("model")
        lane_id = entry.get("lane_id")
        if isinstance(lane_id, str):
            if lane_id in catalog_lane_ids:
                findings.append(
                    Finding("error", f"evidence_catalog:{lane_id}", "duplicate lane")
                )
            catalog_lane_ids.add(lane_id)
        if not isinstance(model, Mapping):
            continue
        model_id = model.get("id")
        adapter = model.get("adapter")
        if isinstance(model_id, str) and isinstance(adapter, str):
            catalog_model_ids.add(model_id)
            adapters_by_model_id.setdefault(model_id, set()).add(adapter)

    multi_adapter_model_ids = {
        model_id
        for model_id, adapters in adapters_by_model_id.items()
        if len(adapters) > 1
    }
    for entry in entries:
        findings.extend(_check_entry(entry, support_rows, multi_adapter_model_ids))

    expected_lane_ids = set(support_rows)
    if catalog_lane_ids != expected_lane_ids:
        findings.append(
            Finding(
                "error",
                "evidence_catalog",
                "lane IDs must exactly match the public support matrix: "
                f"missing={sorted(expected_lane_ids - catalog_lane_ids)!r} "
                f"extra={sorted(catalog_lane_ids - expected_lane_ids)!r}",
            )
        )

    routed_model_ids = ModelFamilyRouteIndex.from_contracts(
        catalog=family_catalog,
        support_matrix=support_matrix,
    ).routed_model_ids()
    missing_catalog_routes = sorted(
        _model_ids_from_catalog(family_catalog) - catalog_model_ids - routed_model_ids
    )
    for model_id in missing_catalog_routes:
        findings.append(
            Finding(
                "error",
                "model_family_catalog",
                f"representative model {model_id!r} lacks a catalog entry or route",
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
