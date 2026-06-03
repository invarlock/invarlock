from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from invarlock.core.exceptions import GuardError, ValidationError


@dataclass
class RMTPolicy:
    """RMT guard policy configuration."""

    q: float | Literal["auto"] = "auto"
    deadband: float = 0.10
    margin: float = 1.5
    correct: bool = True


class RMTPolicyDict(TypedDict, total=False):
    """TypedDict version of the RMT guard policy."""

    q: float | Literal["auto"]
    deadband: float
    margin: float
    correct: bool
    epsilon_default: float
    epsilon_by_family: dict[str, float]
    activation_required: bool
    estimator: dict[str, Any]
    activation: dict[str, Any]


__all__ = [
    "RMTPolicy",
    "RMTPolicyDict",
    "apply_rmt_policy_overrides",
    "build_rmt_guard_policy",
    "compute_epsilon_violations",
    "create_custom_rmt_policy",
    "get_rmt_policy",
]


def build_rmt_guard_policy(guard: Any) -> RMTPolicyDict:
    return RMTPolicyDict(
        q=guard.q,
        deadband=guard.deadband,
        margin=guard.margin,
        correct=guard.correct,
        epsilon_default=float(guard.epsilon_default),
        epsilon_by_family=dict(guard.epsilon_by_family),
    )


def apply_rmt_policy_overrides(guard: Any, policy: dict[str, Any] | None) -> None:
    if not isinstance(policy, dict) or not policy:
        return

    if "epsilon" in policy:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={
                "param": "epsilon",
                "hint": "Use rmt.epsilon_default and rmt.epsilon_by_family instead.",
            },
        )

    if "q" in policy:
        q_val = policy.get("q")
        if q_val == "auto":
            guard.q = "auto"
        else:
            try:
                guard.q = float(q_val)
            except (TypeError, ValueError):
                guard.q = "auto"

    if "deadband" in policy:
        try:
            guard.deadband = float(policy.get("deadband", guard.deadband))
        except (TypeError, ValueError):
            pass

    if "margin" in policy:
        try:
            guard.margin = float(policy.get("margin", guard.margin))
        except (TypeError, ValueError):
            pass

    if "correct" in policy:
        guard.correct = bool(policy.get("correct"))

    if "epsilon_by_family" in policy:
        guard._set_epsilon_by_family(policy["epsilon_by_family"])
    if "epsilon_default" in policy:
        guard._set_epsilon_default(policy["epsilon_default"])

    for family_key in ("attn", "ffn", "embed", "other"):
        guard.epsilon_by_family.setdefault(family_key, guard.epsilon_default)

    if "activation_required" in policy:
        guard._require_activation = bool(policy.get("activation_required"))

    estimator_policy = policy.get("estimator")
    if isinstance(estimator_policy, dict):
        try:
            iters = int(estimator_policy.get("iters", 3) or 3)
        except (TypeError, ValueError):
            iters = 3
        if iters < 1:
            iters = 1
        init = str(estimator_policy.get("init", "ones") or "ones").strip().lower()
        if init not in {"ones", "e0"}:
            init = "ones"
        guard.estimator = {"type": "power_iter", "iters": iters, "init": init}

    activation_policy = policy.get("activation")
    if isinstance(activation_policy, dict):
        sampling = activation_policy.get("sampling")
        if isinstance(sampling, dict):
            windows = sampling.get("windows")
            if isinstance(windows, dict):
                cfg = dict(guard.activation_sampling.get("windows") or {})
                if windows.get("count") is not None:
                    try:
                        cfg["count"] = int(windows.get("count") or 0)
                    except (TypeError, ValueError):
                        pass
                if windows.get("indices_policy") is not None:
                    cfg["indices_policy"] = str(
                        windows.get("indices_policy") or cfg.get("indices_policy")
                    )
                guard.activation_sampling["windows"] = cfg


def compute_epsilon_violations(guard: Any) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    families = set(guard.edge_risk_by_family) | set(guard.baseline_edge_risk_by_family)
    for family in families:
        base = float(guard.baseline_edge_risk_by_family.get(family, 0.0) or 0.0)
        cur = float(guard.edge_risk_by_family.get(family, 0.0) or 0.0)
        if base <= 0.0:
            continue
        epsilon_val = float(guard.epsilon_by_family.get(family, guard.epsilon_default))
        allowed = (1.0 + epsilon_val) * base
        if cur > allowed:
            delta = (cur / base) - 1.0
            violations.append(
                {
                    "family": family,
                    "edge_base": base,
                    "edge_cur": cur,
                    "delta": float(delta),
                    "allowed": allowed,
                    "epsilon": epsilon_val,
                }
            )
    return violations


def get_rmt_policy(name: str = "balanced") -> RMTPolicyDict:
    """Return the in-module fallback policy set.

    The calibrated source of truth used by the runtime is
    `invarlock.guards.policies.get_rmt_policy(..., use_yaml=True)`,
    which overlays values from `runtime/tiers.yaml`. These hardcoded values are
    kept only as a defensive fallback for direct imports of this module.
    """

    fallback_policies = {
        "conservative": RMTPolicyDict(
            q="auto",
            deadband=0.05,
            margin=1.3,
            correct=True,
            epsilon_default=0.06,
            epsilon_by_family={"ffn": 0.06, "attn": 0.05, "embed": 0.07, "other": 0.07},
        ),
        "balanced": RMTPolicyDict(
            q="auto",
            deadband=0.10,
            margin=1.5,
            correct=True,
            epsilon_default=0.10,
            epsilon_by_family={"ffn": 0.10, "attn": 0.08, "embed": 0.12, "other": 0.12},
        ),
        "aggressive": RMTPolicyDict(
            q="auto",
            deadband=0.15,
            margin=1.8,
            correct=True,
            epsilon_default=0.15,
            epsilon_by_family={"ffn": 0.15, "attn": 0.15, "embed": 0.15, "other": 0.15},
        ),
    }

    if name not in fallback_policies:
        available = list(fallback_policies.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return fallback_policies[name]


def create_custom_rmt_policy(
    q: float | Literal["auto"] = "auto",
    deadband: float = 0.10,
    margin: float = 1.5,
    correct: bool = True,
    *,
    epsilon_default: float = 0.1,
    epsilon_by_family: dict[str, float] | None = None,
) -> RMTPolicyDict:
    if isinstance(q, float) and not 0.1 <= q <= 10.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "q", "value": q},
        )

    if not 0.0 <= deadband <= 0.5:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if not margin >= 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "margin", "value": margin},
        )

    try:
        eps_default_val = float(epsilon_default)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "epsilon_default", "value": epsilon_default},
        ) from exc
    if not (math.isfinite(eps_default_val) and eps_default_val >= 0.0):
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "epsilon_default", "value": epsilon_default},
        )

    eps_by_family: dict[str, float] = {}
    if epsilon_by_family is not None:
        if not isinstance(epsilon_by_family, dict):
            raise ValidationError(
                code="E501",
                message="POLICY-PARAM-INVALID",
                details={"param": "epsilon_by_family", "value": epsilon_by_family},
            )
        for family, value in epsilon_by_family.items():
            try:
                eps_val = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "epsilon_by_family",
                        "family": str(family),
                        "value": value,
                    },
                ) from exc
            if not (math.isfinite(eps_val) and eps_val >= 0.0):
                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "epsilon_by_family",
                        "family": str(family),
                        "value": value,
                    },
                )
            eps_by_family[str(family)] = eps_val

    return RMTPolicyDict(
        q=q,
        deadband=deadband,
        margin=margin,
        correct=correct,
        epsilon_default=eps_default_val,
        epsilon_by_family=eps_by_family,
    )
