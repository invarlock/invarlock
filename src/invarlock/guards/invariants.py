"""
InvarLock Guards - Invariants
=========================

Invariant checking for model edits to ensure structural integrity.
"""

from typing import Any

import torch
import torch.nn as nn

from invarlock.core.api import Guard
from invarlock.core.types import GuardOutcome


class InvariantsGuard(Guard):
    """
    Guard for checking model invariants and structural integrity.
    """

    name = "invariants"

    def __init__(self, strict_mode: bool = False, on_fail: str = "warn"):
        """
        Initialize invariants guard.

        Args:
            strict_mode: Whether to use strict validation
            on_fail: Action to take on failure ("warn", "rollback", "abort")
        """
        self.strict_mode = strict_mode
        self.on_fail = on_fail
        self.prepared = False
        self.baseline_checks: dict[str, Any] = {}
        self.profile_checks: tuple[str, ...] = ()

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
    ) -> dict[str, Any]:
        """
        Validate model invariants (Guard ABC interface).

        Args:
            model: Model to validate
            adapter: ModelAdapter instance
            context: Validation context

        Returns:
            Dictionary with validation results
        """
        if not self.prepared:
            # Auto-prepare if not already done
            self.prepare(model, adapter, None, {})

        outcome = self.finalize(model)

        return {
            "passed": outcome.passed,
            "action": outcome.action,
            "violations": outcome.violations,
            "metrics": outcome.metrics,
        }

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
                action="warn",
                violations=[{"type": "not_prepared", "message": "Guard not prepared"}],
                metrics={},
            )

        # Check current invariants
        current_checks = self._capture_invariants(model, None)
        violations: list[dict[str, Any]] = []
        tokenizer_mismatches: list[dict[str, Any]] = []

        # Non-finite detection
        non_finite_locations = self._detect_non_finite(model)
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
                current_size = None
                if isinstance(current_vocab_sizes, dict):
                    current_size = current_vocab_sizes.get(module_name)
                if current_size is None or int(current_size) != int(baseline_size):
                    mismatch = {
                        "module": module_name,
                        "baseline": int(baseline_size),
                        "current": None if current_size is None else int(current_size),
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
            "layer_norm_paths",
            "embedding_vocab_sizes",
            "config_vocab_size",
        }

        for check_name, baseline_value in self.baseline_checks.items():
            if check_name in handled_keys:
                continue

            current_value = current_checks.get(check_name)

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
            fatal_violation_types.update({"layer_norm_missing", "invariant_violation"})

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
            if self.on_fail in {"abort", "rollback"}:
                action = self.on_fail
            else:
                action = "abort"
        elif warning_count:
            if self.on_fail in {"abort", "rollback"}:
                passed = False
                action = self.on_fail
            else:
                passed = True
                action = "warn"
        else:
            passed = True
            action = "none"

        metrics: dict[str, Any] = {
            "checks_performed": len(self.baseline_checks),
            "violations_found": len(annotated_violations),
            "fatal_violations": fatal_count,
            "warning_violations": warning_count,
        }
        if non_finite_locations:
            metrics["non_finite_found"] = len(non_finite_locations)
        if missing_layer_norms:
            metrics["layer_norm_missing"] = missing_layer_norms
        if tokenizer_mismatches:
            metrics["tokenizer_mismatches"] = tokenizer_mismatches

        return GuardOutcome(
            name=self.name,
            passed=passed,
            action=action,
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
        checks = {}

        # Check parameter count
        try:
            param_count = sum(p.numel() for p in model.parameters())
            checks["parameter_count"] = param_count
        except Exception:
            checks["parameter_count"] = -1

        # Record LayerNorm module paths for later comparison
        layer_norm_paths: list[str] = []
        try:
            for name, module in model.named_modules():
                if isinstance(module, nn.LayerNorm):
                    layer_norm_paths.append(name)
        except Exception:
            layer_norm_paths = []
        checks["layer_norm_paths"] = tuple(layer_norm_paths)

        # Capture embedding vocab sizes (num_embeddings) for tokenizer alignment
        embedding_vocab_sizes: dict[str, int] = {}
        try:
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding):
                    try:
                        embedding_vocab_sizes[name] = int(module.num_embeddings)
                    except Exception:
                        weight = getattr(module, "weight", None)
                        if getattr(weight, "shape", None):
                            embedding_vocab_sizes[name] = int(weight.shape[0])
        except Exception:
            embedding_vocab_sizes = {}
        if embedding_vocab_sizes:
            checks["embedding_vocab_sizes"] = embedding_vocab_sizes

        config = getattr(model, "config", None)
        config_vocab = getattr(config, "vocab_size", None)
        try:
            if config_vocab is not None:
                checks["config_vocab_size"] = int(config_vocab)
        except Exception:
            pass

        # Check weight tying (for language models)
        weight_tying_flags: dict[str, bool] = {}

        def _is_tied(left: Any, right: Any) -> bool:
            if left is None or right is None:
                return False
            try:
                return left.data_ptr() == right.data_ptr()
            except Exception:
                return False

        # GPT-2 style (transformer.wte <-> lm_head)
        try:
            transformer = getattr(model, "transformer", None)
            lm_head = getattr(model, "lm_head", None)
            embed_weight = getattr(getattr(transformer, "wte", None), "weight", None)
            head_weight = getattr(lm_head, "weight", None)
            if embed_weight is not None and head_weight is not None:
                weight_tying_flags["gpt2"] = _is_tied(embed_weight, head_weight)
        except Exception:
            pass

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
        except Exception:
            pass

        # LLaMA style (model.embed_tokens <-> lm_head)
        try:
            llama_model = getattr(model, "model", None)
            embed_tokens = getattr(llama_model, "embed_tokens", None)
            embed_weight = getattr(embed_tokens, "weight", None)
            llama_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
            if embed_weight is not None and llama_head_weight is not None:
                weight_tying_flags["llama"] = _is_tied(embed_weight, llama_head_weight)
        except Exception:
            pass

        if weight_tying_flags:
            checks["weight_tying"] = all(weight_tying_flags.values())
            checks["weight_tying_arches"] = weight_tying_flags
        else:
            checks["weight_tying"] = None

        # Check model structure hash (basic)
        try:
            structure_items = []
            for name, module in model.named_modules():
                structure_items.append(f"{name}:{type(module).__name__}")
            structure_hash = hash(tuple(structure_items))
            checks["structure_hash"] = structure_hash
        except Exception:
            checks["structure_hash"] = 0

        # Profile-specific invariants
        if getattr(self, "profile_checks", None):
            for name in self.profile_checks:
                checks[f"profile::{name}"] = self._evaluate_profile_check(model, name)

        return checks

    def _detect_non_finite(self, model: Any) -> list[str]:
        """Detect parameters or buffers containing non-finite values."""
        locations: list[str] = []
        try:
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param).all():
                        locations.append(f"parameter::{name}")
                except Exception:
                    continue
            for name, buffer in model.named_buffers():
                try:
                    if not torch.isfinite(buffer).all():
                        locations.append(f"buffer::{name}")
                except Exception:
                    continue
        except Exception:
            return locations
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
            # Detect rotary embeddings used by LLaMA-style models
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                first_layer = model.model.layers[0] if model.model.layers else None
            else:
                first_layer = None
            rotary = None
            if first_layer is not None:
                rotary = getattr(
                    getattr(first_layer, "self_attn", None), "rotary_emb", None
                )
            return rotary is not None

        if name in {"causal_masking", "causal"}:
            config = getattr(model, "config", None)
            if config and getattr(config, "is_decoder", False):
                return True
            model_type = getattr(config, "model_type", "") if config else ""
            return any(
                keyword in model_type
                for keyword in ("gpt", "llama", "mistral", "opt", "phi")
            )

        return True


def check_adapter_aware_invariants(
    model: Any, verbose: bool = False
) -> tuple[bool, dict[str, Any]]:
    """
    Check model invariants with adapter awareness.

    Args:
        model: Model to check
        verbose: Whether to print detailed information

    Returns:
        (all_passed, results) tuple
    """
    results: dict[str, Any] = {"adapter_type": "none", "checks": {}, "violations": []}
    all_passed = True
    # Standard model checks only
    standard_checks: dict[str, dict[str, Any]] = _check_standard_invariants(model)
    results["checks"].update(standard_checks)
    for check_name, check_result in standard_checks.items():
        if not check_result.get("passed", True):
            all_passed = False
            results["violations"].append(
                {
                    "type": "standard_violation",
                    "check": check_name,
                    "message": check_result.get("message", "Check failed"),
                }
            )
    return all_passed, results


def _detect_adapter_type(model: Any) -> str:
    """Detect adapter type (disabled). Always returns 'none'."""
    return "none"


def _check_standard_invariants(model: Any) -> dict[str, dict[str, Any]]:
    """Check standard model invariants."""
    checks: dict[str, dict[str, Any]] = {}

    # Check parameter count is reasonable
    try:
        param_count = sum(p.numel() for p in model.parameters())
        checks["parameter_count"] = {
            "passed": param_count > 0,
            "count": param_count,
            "message": f"Parameter count: {param_count}",
        }
    except Exception as e:
        checks["parameter_count"] = {
            "passed": False,
            "message": f"Could not count parameters: {e}",
        }

    # Check for NaN parameters
    try:
        has_nan = False
        for param in model.parameters():
            if hasattr(param, "isnan") and param.isnan().any():
                has_nan = True
                break

        checks["no_nan_parameters"] = {
            "passed": not has_nan,
            "message": "NaN parameters detected" if has_nan else "No NaN parameters",
        }
    except Exception as e:
        checks["no_nan_parameters"] = {
            "passed": False,
            "message": f"Could not check for NaN: {e}",
        }

    return checks


def check_all_invariants(model: Any, threshold: float = 1e-6) -> GuardOutcome:
    """
    Check all basic model invariants.

    Args:
        model: PyTorch model to check
        threshold: Numerical threshold for invariant checks

    Returns:
        GuardOutcome: Result of invariant checking
    """
    violations = []

    # Basic model structure checks
    if not hasattr(model, "named_parameters"):
        violations.append(
            {
                "type": "structure_violation",
                "message": "Model missing named_parameters method",
            }
        )
        return GuardOutcome(
            name="check_all_invariants",
            passed=False,
            action="reject",
            violations=violations,
            metrics={},
        )

    # Check for NaN/Inf in parameters
    for name, param in model.named_parameters():
        if hasattr(param.data, "isnan") and param.data.isnan().any():
            violations.append(
                {
                    "type": "nan_violation",
                    "parameter": name,
                    "message": f"NaN detected in parameter {name}",
                }
            )
        if hasattr(param.data, "isinf") and param.data.isinf().any():
            violations.append(
                {
                    "type": "inf_violation",
                    "parameter": name,
                    "message": f"Inf detected in parameter {name}",
                }
            )

    # Check parameter ranges are reasonable
    for name, param in model.named_parameters():
        if hasattr(param.data, "abs") and hasattr(param.data, "max"):
            max_val = param.data.abs().max()
            if hasattr(max_val, "item"):
                max_val = max_val.item()

            if max_val > 1000:
                violations.append(
                    {
                        "type": "range_violation",
                        "parameter": name,
                        "max_value": max_val,
                        "message": f"Parameter {name} has unusually large values (max: {max_val})",
                    }
                )
            if max_val < threshold:
                violations.append(
                    {
                        "type": "range_violation",
                        "parameter": name,
                        "max_value": max_val,
                        "message": f"Parameter {name} has unusually small values (max: {max_val})",
                    }
                )

    passed = len(violations) == 0
    action = "continue" if passed else "reject"

    return GuardOutcome(
        name="check_all_invariants",
        passed=passed,
        action=action,
        violations=violations,
        metrics={
            "parameters_checked": sum(1 for _ in model.named_parameters()),
            "violations_found": len(violations),
        },
    )


def assert_invariants(model: Any, threshold: float = 1e-6) -> None:
    """
    Assert that all model invariants hold, raising exception if not.

    Args:
        model: PyTorch model to check
        threshold: Numerical threshold for invariant checks

    Raises:
        AssertionError: If any invariants are violated
    """
    result = check_all_invariants(model, threshold)
    if not result.passed:
        violation_messages = [v.get("message", str(v)) for v in result.violations or []]
        raise AssertionError(
            f"Model invariants violated: {'; '.join(violation_messages)}"
        )


__all__ = [
    "InvariantsGuard",
    "check_adapter_aware_invariants",
    "check_all_invariants",
    "assert_invariants",
]
