"""Shipped deterministic text scorers using the existing extension contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from invarlock.core.scorer_extension import (
    ScorerExtensionDescriptor,
    ScorerExtensionResult,
    ScorerReplayRequest,
    build_scorer_result,
    scorer_configuration_schema_sha256,
)
from invarlock.pipeline.metrics import KINDS, SCORER_VERSION, score

BUILTIN_SCORER_IDS = tuple(
    f"invarlock.{kind}" for kind in KINDS if kind != "exact_match"
)


class BuiltinScorer:
    """First-party pure text implementation; no discovery or third-party code."""

    abi_version = "1"

    def __init__(self, kind: str):
        if f"invarlock.{kind}" not in BUILTIN_SCORER_IDS:
            raise ValueError(f"unknown shipped scorer {kind!r}")
        self.kind = kind

    def configuration_schema(self) -> Mapping[str, object]:
        properties: dict[str, Any] = {}
        required = []
        if self.kind in ("normalized_match", "token_f1"):
            properties = {
                "casefold": {"type": "boolean"},
                "unicode_version": {
                    "type": "string",
                    "pattern": r"^[0-9]+\.[0-9]+\.[0-9]+$",
                    "maxLength": 32,
                },
            }
            required = ["unicode_version"]
        elif self.kind == "numeric_tolerance":
            properties = {
                key: {"type": "number", "minimum": 0}
                for key in ("absolute", "relative")
            }
        elif self.kind == "json_fields":
            required = ["fields"]
            properties = {
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^/", "maxLength": 4096},
                    "minItems": 1,
                    "maxItems": 100,
                    "uniqueItems": True,
                }
            }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

    def descriptor(self) -> ScorerExtensionDescriptor:
        return ScorerExtensionDescriptor(
            scorer_id=f"invarlock.{self.kind}",
            scorer_version=SCORER_VERSION,
            supported_tasks=("text_causal",),
            supported_input_kinds=("text",),
            supported_output_kinds=("text",),
            required_facts=("expected_output", "output_text", "output_sha256"),
            configuration_schema_sha256=scorer_configuration_schema_sha256(
                self.configuration_schema()
            ),
        )

    def replay(self, request: ScorerReplayRequest) -> ScorerExtensionResult:
        # Binding freezes JSON arrays as tuples; convert them to ordinary JSON
        # before invoking the same scorer used by pipeline and qualification.
        configuration = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in request.binding.configuration.items()
        }
        return build_scorer_result(
            request,
            [
                score(
                    self.kind,
                    r.facts["expected_output"],
                    r.facts["output_text"],
                    configuration,
                )
                for r in request.records
            ],
        )
