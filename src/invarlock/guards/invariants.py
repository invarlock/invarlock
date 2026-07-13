"""
InvarLock Guards - Invariants
=========================

Invariant checking for model edits to ensure structural integrity.
"""

import hashlib
from typing import Any

import torch
import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard
from invarlock.core.types import GuardDiagnostic, GuardOutcome, GuardValidationResult
from invarlock.guards import invariant_checks as _invariant_checks
from invarlock.guards import invariants_standard as _invariants_standard
from invarlock.guards.invariant_embeddings import (
    embedding_vocab_size_matches,
)

INVARLOCK_CORE_ABI = CORE_ABI

check_adapter_aware_invariants = _invariants_standard.check_adapter_aware_invariants
_check_standard_invariants = _invariants_standard.check_standard_invariants
_detect_adapter_type = _invariants_standard.detect_adapter_type

_INVARIANT_CAPTURE_ERRORS = (
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)

_TEXT_DECODER_WRAPPERS = ("language_model",)


def _first_decoder_layer(decoder: Any, *, container: str = "layers") -> Any | None:
    """Return the first layer only for supported decoder layer containers."""
    layers = getattr(decoder, container, None)
    if not isinstance(layers, (list, tuple, nn.ModuleList)) or not layers:
        return None
    return layers[0]


def _has_decoder_rotary_embedding(
    decoder: Any, *, layer_container: str = "layers"
) -> bool:
    """Detect RoPE only on a structurally valid decoder backbone."""
    first_layer = _first_decoder_layer(decoder, container=layer_container)
    if first_layer is None:
        return False
    if getattr(decoder, "rotary_emb", None) is not None:
        return True
    self_attn = getattr(first_layer, "self_attn", None)
    return getattr(self_attn, "rotary_emb", None) is not None


def _has_rotary_embedding(model: Any) -> bool:
    """Inspect explicit text-decoder paths without recursively matching names."""
    decoder = getattr(model, "model", None)
    if decoder is not None and _has_decoder_rotary_embedding(decoder):
        return True
    if decoder is not None and any(
        _has_decoder_rotary_embedding(getattr(decoder, wrapper, None))
        for wrapper in _TEXT_DECODER_WRAPPERS
    ):
        return True
    # Falcon exposes the decoder as ``transformer.h`` and its shared rotary
    # embedding on ``transformer``. Keep this explicit so similarly named
    # vision or auxiliary modules cannot satisfy the language-model invariant.
    transformer = getattr(model, "transformer", None)
    return _has_decoder_rotary_embedding(transformer, layer_container="h")


class InvariantsGuard(Guard):
    """
    Guard for checking model invariants and structural integrity.
    """

    name = "invariants"

    def __init__(self, strict_mode: bool = False, on_fail: str = "monitor"):
        """
        Initialize invariants guard.

        Args:
            strict_mode: Whether to use strict validation
            on_fail: Decision to take on failure ("monitor", "rollback", "block")
        """
        self.strict_mode = strict_mode
        self.on_fail = on_fail
        self.prepared = False
        self.baseline_checks: dict[str, Any] = {}
        self.last_current_checks: dict[str, Any] = {}
        self.profile_checks: tuple[str, ...] = ()
        self._adapter_name = ""
        self._assurance_mode = "off"
        self._non_finite_evidence_gaps: list[dict[str, str]] = []

    def set_run_context(self, report: Any) -> None:
        """Apply assurance requirements before guard preparation.

        Strict assurance cannot inherit the guard's compatibility defaults of
        monitoring structural failures.  The run contract therefore upgrades
        invariant checks to fail closed even when a policy omits these knobs.
        """
        context = getattr(report, "context", {}) or {}
        assurance = context.get("assurance") if isinstance(context, dict) else None
        mode = assurance.get("mode") if isinstance(assurance, dict) else None
        self._assurance_mode = str(mode or "off").strip().lower()
        if self._assurance_mode == "strict":
            self.strict_mode = True
            self.on_fail = "block"

    def prepare(
        self, model: Any, adapter: Any, calib: Any, policy: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Prepare invariants guard by capturing baseline state.

        Args:
            model: Model to prepare for
            adapter: ModelAdapter instance
            calib: Calibration data (unused)
            policy: Policy configuration

        Returns:
            Preparation results
        """
        self.prepared = True
        self._adapter_name = (
            str(
                getattr(adapter, "name", adapter if isinstance(adapter, str) else "")
                or ""
            )
            .strip()
            .lower()
        )

        if isinstance(policy, dict):
            if "strict_mode" in policy:
                self.strict_mode = bool(policy["strict_mode"])
            configured_action = policy.get("on_fail")
            if configured_action in {"monitor", "rollback", "block"}:
                self.on_fail = str(configured_action)
        if self._assurance_mode == "strict":
            self.strict_mode = True
            self.on_fail = "block"

        profile_checks = (
            policy.get("profile_checks") if isinstance(policy, dict) else None
        )
        if isinstance(profile_checks, list | tuple | set):
            self.profile_checks = tuple(str(check) for check in profile_checks)
        else:
            self.profile_checks = ()

        # Capture baseline invariants
        self.baseline_checks = self._capture_invariants(model, adapter)

        return {
            "ready": True,
            "baseline_checks": len(self.baseline_checks),
            "strict_mode": self.strict_mode,
        }

    def before_edit(self, model: Any) -> None:
        """Execute before edit (no action needed for invariants)."""
        pass

    def after_edit(self, model: Any) -> None:
        """Execute after edit (no action needed for invariants)."""
        pass

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        """
        Validate model invariants (Guard ABC interface).

        Args:
            model: Model to validate
            adapter: ModelAdapter instance
            context: Validation context

        Returns:
            Typed validation result
        """
        if not self.prepared:
            # Auto-prepare if not already done
            self.prepare(model, adapter, None, {})

        outcome = self.finalize(model)
        violations = tuple(dict(item) for item in (outcome.violations or []))
        diagnostics = tuple(
            GuardDiagnostic(
                kind=str(violation.get("type", "invariant_violation")),
                severity=str(violation.get("severity", "warning")),
                message=str(violation.get("message", "")),
                details={
                    str(key): value
                    for key, value in violation.items()
                    if key not in {"type", "severity", "message"}
                },
            )
            for violation in violations
        )
        return GuardValidationResult(
            passed=bool(outcome.passed),
            decision=str(outcome.decision),
            metrics=dict(outcome.metrics or {}),
            diagnostics=diagnostics,
            policy={
                "strict_mode": bool(self.strict_mode),
                "on_fail": str(self.on_fail),
            },
            details={
                "baseline_checks": dict(self.baseline_checks),
                "current_checks": dict(self.last_current_checks),
            },
            violations=violations,
        )

    def finalize(self, model: Any) -> GuardOutcome:
        """
        Finalize invariants guard by checking for violations.

        Args:
            model: Model to validate

        Returns:
            GuardOutcome with validation results
        """
        if not self.prepared:
            return GuardOutcome(
                name=self.name,
                passed=False,
                decision="block",
                violations=[{"type": "not_prepared", "message": "Guard not prepared"}],
                metrics={},
            )

        # Check current invariants
        current_checks = self._capture_invariants(model, self._adapter_name)
        self.last_current_checks = current_checks
        violations: list[dict[str, Any]] = []
        tokenizer_mismatches: list[dict[str, Any]] = []
        evidence_gaps: list[dict[str, str]] = []

        # Non-finite detection
        non_finite_locations = self._detect_non_finite(model)
        evidence_gaps.extend(self._non_finite_evidence_gaps)
        if non_finite_locations:
            violations.append(
                {
                    "type": "non_finite_tensor",
                    "locations": non_finite_locations,
                    "message": "Non-finite parameter or buffer values detected",
                }
            )

        # LayerNorm coverage check
        baseline_layer_norms = set(self.baseline_checks.get("layer_norm_paths", ()))
        current_layer_norms = set(current_checks.get("layer_norm_paths", ()))
        missing_layer_norms = sorted(baseline_layer_norms - current_layer_norms)
        if missing_layer_norms:
            violations.append(
                {
                    "type": "layer_norm_missing",
                    "missing": missing_layer_norms,
                    "message": "Expected LayerNorm modules are missing after edit",
                }
            )

        # Tokenizer / vocab alignment
        baseline_vocab_sizes = self.baseline_checks.get("embedding_vocab_sizes")
        current_vocab_sizes = current_checks.get("embedding_vocab_sizes")
        if isinstance(baseline_vocab_sizes, dict):
            for module_name, baseline_size in baseline_vocab_sizes.items():
                size_matches, current_size = embedding_vocab_size_matches(
                    baseline_vocab_sizes,
                    current_vocab_sizes,
                    str(module_name),
                    baseline_size,
                )
                if not size_matches:
                    mismatch = {
                        "module": module_name,
                        "baseline": int(baseline_size),
                        "current": current_size,
                    }
                    tokenizer_mismatches.append(mismatch)
                    violations.append(
                        {
                            "type": "tokenizer_mismatch",
                            "message": "Embedding vocabulary size changed",
                            **mismatch,
                        }
                    )

        # Compare remaining invariants with baseline
        handled_keys = {
            "evidence_gaps",
            "layer_norm_paths",
            "embedding_vocab_sizes",
            "config_vocab_size",
        }

        for phase, checks in (
            ("baseline", self.baseline_checks),
            ("current", current_checks),
        ):
            raw_gaps = checks.get("evidence_gaps")
            if not isinstance(raw_gaps, tuple | list):
                continue
            for gap in raw_gaps:
                if not isinstance(gap, dict):
                    continue
                check_name = str(gap.get("check", "unknown"))
                reason = str(gap.get("reason", "unknown"))
                evidence_gaps.append(
                    {
                        "phase": phase,
                        "check": check_name,
                        "reason": reason,
                    }
                )

        if evidence_gaps:
            violations.append(
                {
                    "type": "evidence_gap",
                    "message": "Invariant evidence capture failed",
                    "gaps": evidence_gaps,
                }
            )

        for check_name, baseline_value in self.baseline_checks.items():
            if check_name in handled_keys:
                continue

            current_value = current_checks.get(check_name)

            if check_name.startswith("profile::") and current_value is not True:
                violations.append(
                    {
                        "type": "profile_invariant_failed",
                        "check": check_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "message": f"Required model profile invariant {check_name} is not satisfied",
                    }
                )
                continue

            if current_value != baseline_value:
                violations.append(
                    {
                        "type": "invariant_violation",
                        "check": check_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "message": f"Invariant {check_name} changed from {baseline_value} to {current_value}",
                    }
                )

        # Classify violations by severity
        fatal_violation_types = {"non_finite_tensor", "tokenizer_mismatch"}
        if self.strict_mode:
            fatal_violation_types.update(
                {
                    "layer_norm_missing",
                    "invariant_violation",
                    "profile_invariant_failed",
                    "evidence_gap",
                }
            )

        fatal_violations: list[dict[str, Any]] = []
        warning_violations: list[dict[str, Any]] = []

        for violation in violations:
            violation_type = violation.get("type")
            severity = "fatal" if violation_type in fatal_violation_types else "warning"
            annotated = violation.copy()
            annotated.setdefault("severity", severity)
            if severity == "fatal":
                fatal_violations.append(annotated)
            else:
                warning_violations.append(annotated)

        annotated_violations = fatal_violations + warning_violations

        # Determine if passed based on fatal violations and configured action
        fatal_count = len(fatal_violations)
        warning_count = len(warning_violations)

        if fatal_count:
            passed = False
            if self.on_fail in {"block", "rollback"}:
                decision = self.on_fail
            else:
                decision = "block"
        elif warning_count:
            if self.on_fail in {"block", "rollback"}:
                passed = False
                decision = self.on_fail
            else:
                passed = True
                decision = "monitor"
        else:
            passed = True
            decision = "allow"

        metrics: dict[str, Any] = {
            "checks_performed": len(self.baseline_checks),
            "violations_found": len(annotated_violations),
            "fatal_violations": fatal_count,
            "warning_violations": warning_count,
            "decision": decision,
        }
        if non_finite_locations:
            metrics["non_finite_found"] = len(non_finite_locations)
        if missing_layer_norms:
            metrics["layer_norm_missing"] = missing_layer_norms
        if tokenizer_mismatches:
            metrics["tokenizer_mismatches"] = tokenizer_mismatches
        if evidence_gaps:
            metrics["evidence_gaps"] = len(evidence_gaps)

        return GuardOutcome(
            name=self.name,
            passed=passed,
            decision=decision,
            violations=annotated_violations,
            metrics=metrics,
        )

    def _capture_invariants(self, model: Any, adapter: Any | None) -> dict[str, Any]:
        """
        Capture model invariants for comparison.

        Args:
            model: Model to analyze
            adapter: ModelAdapter (optional)

        Returns:
            Dictionary of invariant checks
        """
        checks: dict[str, Any] = {}
        evidence_gaps: list[dict[str, str]] = []

        # Check parameter count
        try:
            param_count = sum(p.numel() for p in model.parameters())
            checks["parameter_count"] = param_count
        except _INVARIANT_CAPTURE_ERRORS as exc:
            checks["parameter_count"] = None
            evidence_gaps.append(
                {
                    "check": "parameter_count",
                    "reason": type(exc).__name__,
                }
            )

        layer_norm_paths: list[str] = []
        embedding_vocab_sizes: dict[str, int] = {}
        structure_items: list[str] = []
        module_type_paths: dict[str, str] = {}
        linear_dimensions: dict[str, list[int]] = {}
        parameter_shapes: dict[str, list[int]] = {}
        try:
            for name, parameter in model.named_parameters():
                parameter_shapes[str(name)] = [int(part) for part in parameter.shape]
        except _INVARIANT_CAPTURE_ERRORS as exc:
            parameter_shapes = {}
            evidence_gaps.append(
                {
                    "check": "parameter_shapes",
                    "reason": type(exc).__name__,
                }
            )
        try:
            for name, module in model.named_modules():
                module_type = type(module)
                module_fqcn = f"{module_type.__module__}.{module_type.__qualname__}"
                structure_items.append(f"{name}:{module_type.__name__}")
                module_type_paths[str(name)] = module_fqcn
                in_features = getattr(module, "in_features", None)
                out_features = getattr(module, "out_features", None)
                if (
                    isinstance(in_features, int)
                    and not isinstance(in_features, bool)
                    and isinstance(out_features, int)
                    and not isinstance(out_features, bool)
                    and in_features > 0
                    and out_features > 0
                ):
                    linear_dimensions[str(name)] = [in_features, out_features]
                if isinstance(module, nn.LayerNorm):
                    layer_norm_paths.append(name)
                if isinstance(module, nn.Embedding):
                    try:
                        embedding_vocab_sizes[name] = int(module.num_embeddings)
                    except _INVARIANT_CAPTURE_ERRORS:
                        weight = getattr(module, "weight", None)
                        if getattr(weight, "shape", None):
                            embedding_vocab_sizes[name] = int(weight.shape[0])
        except _INVARIANT_CAPTURE_ERRORS as exc:
            layer_norm_paths = []
            embedding_vocab_sizes = {}
            structure_items = []
            evidence_gaps.append(
                {
                    "check": "module_structure",
                    "reason": type(exc).__name__,
                }
            )
        checks["layer_norm_paths"] = tuple(layer_norm_paths)
        if module_type_paths:
            checks["module_type_paths"] = dict(sorted(module_type_paths.items()))
        if linear_dimensions:
            checks["linear_dimensions"] = dict(sorted(linear_dimensions.items()))
        if parameter_shapes:
            checks["parameter_shapes"] = dict(sorted(parameter_shapes.items()))
        if embedding_vocab_sizes:
            checks["embedding_vocab_sizes"] = embedding_vocab_sizes

        config = getattr(model, "config", None)
        config_vocab = getattr(config, "vocab_size", None)
        try:
            if config_vocab is not None:
                checks["config_vocab_size"] = int(config_vocab)
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "check": "config_vocab_size",
                    "reason": type(exc).__name__,
                }
            )

        # Check weight tying (for language models)
        weight_tying_flags: dict[str, bool] = {}

        def _is_tied(left: Any, right: Any) -> bool:
            try:
                return left.data_ptr() == right.data_ptr()
            except _INVARIANT_CAPTURE_ERRORS:
                # guard-fallback-ok: inaccessible tensor pointers are treated as not tied.
                return False

        # GPT-2 style (transformer.wte <-> lm_head)
        try:
            transformer = getattr(model, "transformer", None)
            lm_head = getattr(model, "lm_head", None)
            embed_weight = getattr(getattr(transformer, "wte", None), "weight", None)
            head_weight = getattr(lm_head, "weight", None)
            if embed_weight is not None and head_weight is not None:
                weight_tying_flags["gpt2"] = _is_tied(embed_weight, head_weight)
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "check": "weight_tying_gpt2",
                    "reason": type(exc).__name__,
                }
            )

        # BERT style (bert.embeddings.word_embeddings <-> cls.predictions.decoder)
        try:
            bert = getattr(model, "bert", None)
            embeddings = getattr(bert, "embeddings", None)
            word_embeddings = getattr(embeddings, "word_embeddings", None)
            decoder = getattr(
                getattr(getattr(model, "cls", None), "predictions", None),
                "decoder",
                None,
            )
            embed_weight = getattr(word_embeddings, "weight", None)
            decoder_weight = getattr(decoder, "weight", None)
            if embed_weight is not None and decoder_weight is not None:
                weight_tying_flags["bert"] = _is_tied(embed_weight, decoder_weight)
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "check": "weight_tying_bert",
                    "reason": type(exc).__name__,
                }
            )

        # Decoder embed_tokens style (model.embed_tokens <-> lm_head)
        try:
            decoder_model = getattr(model, "model", None)
            embed_tokens = getattr(decoder_model, "embed_tokens", None)
            embed_weight = getattr(embed_tokens, "weight", None)
            head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
            if embed_weight is not None and head_weight is not None:
                weight_tying_flags["embed_tokens"] = _is_tied(embed_weight, head_weight)
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "check": "weight_tying_embed_tokens",
                    "reason": type(exc).__name__,
                }
            )

        # Wrapped decoder style
        # (model.model.language_model.embed_tokens <-> top-level lm_head). This is
        # the explicit text-decoder path used by unified multimodal models;
        # do not scan vision towers or arbitrary embedding-like modules.
        try:
            decoder_model = getattr(model, "model", None)
            language_model = getattr(decoder_model, "language_model", None)
            if language_model is not None:
                embed_tokens = getattr(language_model, "embed_tokens", None)
                embed_weight = getattr(embed_tokens, "weight", None)
                head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
                if embed_weight is None:
                    evidence_gaps.append(
                        {
                            "check": "weight_tying_language_model_embed_tokens",
                            "reason": "embedding_weight_missing",
                        }
                    )
                elif head_weight is None:
                    evidence_gaps.append(
                        {
                            "check": "weight_tying_language_model_embed_tokens",
                            "reason": "lm_head_weight_missing",
                        }
                    )
                else:
                    weight_tying_flags["language_model_embed_tokens"] = _is_tied(
                        embed_weight, head_weight
                    )
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "check": "weight_tying_language_model_embed_tokens",
                    "reason": type(exc).__name__,
                }
            )

        if weight_tying_flags:
            checks["weight_tying"] = all(weight_tying_flags.values())
            checks["weight_tying_arches"] = weight_tying_flags
        else:
            checks["weight_tying"] = None

        # Check model structure hash (basic)
        try:
            canonical = "\n".join(sorted(structure_items))
            checks["structure_hash"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:16]
        except _INVARIANT_CAPTURE_ERRORS as exc:
            checks["structure_hash"] = None
            evidence_gaps.append(
                {
                    "check": "structure_hash",
                    "reason": type(exc).__name__,
                }
            )

        adapter_name = (
            str(
                getattr(adapter, "name", adapter if isinstance(adapter, str) else "")
                or ""
            )
            .strip()
            .lower()
        )
        if adapter_name == "hf_bnb":
            try:
                from invarlock.core.backend_inventory import (  # noqa: PLC0415
                    _quantized_module_inventory,
                )

                observation = _quantized_module_inventory(
                    model,
                    adapter=adapter_name,
                )
                checks["quantized_runtime_observation"] = {
                    "schema": "invarlock/quantized-structure-observation-v1",
                    "adapter": adapter_name,
                    "count": int(observation.get("count", 0) or 0),
                    "types": list(observation.get("types", [])),
                    "kinds": list(observation.get("kinds", [])),
                    "modules": dict(observation.get("modules", {})),
                }
            except _INVARIANT_CAPTURE_ERRORS as exc:
                evidence_gaps.append(
                    {
                        "check": "quantized_runtime_observation",
                        "reason": type(exc).__name__,
                    }
                )

        # Profile-specific invariants
        if getattr(self, "profile_checks", None):
            for name in self.profile_checks:
                checks[f"profile::{name}"] = self._evaluate_profile_check(model, name)

        if evidence_gaps:
            checks["evidence_gaps"] = tuple(
                {"check": gap["check"], "reason": gap["reason"]}
                for gap in evidence_gaps
            )

        return checks

    def _detect_non_finite(self, model: Any) -> list[str]:
        """Detect parameters or buffers containing non-finite values."""
        locations: list[str] = []
        evidence_gaps: list[dict[str, str]] = []
        try:
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param).all():
                        locations.append(f"parameter::{name}")
                except _INVARIANT_CAPTURE_ERRORS as exc:
                    evidence_gaps.append(
                        {
                            "phase": "current",
                            "check": f"parameter_finiteness::{name}",
                            "reason": type(exc).__name__,
                        }
                    )
                    continue
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "phase": "current",
                    "check": "parameter_finiteness_iteration",
                    "reason": type(exc).__name__,
                }
            )
        try:
            for name, buffer in model.named_buffers():
                try:
                    if not torch.isfinite(buffer).all():
                        locations.append(f"buffer::{name}")
                except _INVARIANT_CAPTURE_ERRORS as exc:
                    evidence_gaps.append(
                        {
                            "phase": "current",
                            "check": f"buffer_finiteness::{name}",
                            "reason": type(exc).__name__,
                        }
                    )
                    continue
        except _INVARIANT_CAPTURE_ERRORS as exc:
            evidence_gaps.append(
                {
                    "phase": "current",
                    "check": "buffer_finiteness_iteration",
                    "reason": type(exc).__name__,
                }
            )
        self._non_finite_evidence_gaps = evidence_gaps
        return locations

    def _evaluate_profile_check(self, model: Any, name: str) -> bool:
        name = str(name).lower()

        if name == "mlm_mask_alignment":
            config = getattr(model, "config", None)
            model_type = getattr(config, "model_type", "") if config else ""
            has_cls_decoder = bool(
                getattr(
                    getattr(getattr(model, "cls", None), "predictions", None),
                    "decoder",
                    None,
                )
            )
            return "bert" in model_type or has_cls_decoder

        if name in {"rope_rotary_embedding", "rotary_embedding"}:
            return _has_rotary_embedding(model)

        if name in {"causal_masking", "causal"}:
            config = getattr(model, "config", None)
            if config and getattr(config, "is_decoder", False):
                return True
            model_type = getattr(config, "model_type", "") if config else ""
            return any(
                keyword in model_type
                for keyword in ("gpt", "mistral", "mixtral", "qwen", "opt", "phi")
            )

        return False


def check_all_invariants(model: Any, threshold: float = 1e-6) -> GuardOutcome:
    """Check basic whole-model structural and numerical invariants."""
    return _invariant_checks.check_all_invariants(model, threshold)


def assert_invariants(model: Any, threshold: float = 1e-6) -> None:
    """Raise when a whole-model invariant check fails."""
    result = check_all_invariants(model, threshold)
    if not result.passed:
        violation_messages = [
            violation.get("message", str(violation))
            for violation in result.violations or []
        ]
        raise AssertionError(
            f"Model invariants violated: {'; '.join(violation_messages)}"
        )


__all__ = [
    "InvariantsGuard",
    "check_adapter_aware_invariants",
    "check_all_invariants",
    "assert_invariants",
]
