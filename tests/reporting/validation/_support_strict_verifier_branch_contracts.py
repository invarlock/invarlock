from __future__ import annotations

from invarlock.reporting import verify_baseline as baseline_mod
from invarlock.reporting import verify_bootstrap as bootstrap_mod
from invarlock.reporting import verify_strict_accuracy as accuracy_mod


class _ExplodingFloatInt(int):
    def __float__(self) -> float:
        raise OverflowError("numeric conversion refused")


class _ExplodingIntString(str):
    def __int__(self) -> int:
        raise OverflowError("integer conversion refused")


def _baseline_errors(report: dict, baseline: dict | None) -> list[str]:
    errors: list[str] = []
    baseline_mod.append_strict_baseline_contract_errors(
        errors,
        report=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    return errors


def _accuracy_errors(payload: dict, *, strict: bool = True) -> tuple[bool, list[str]]:
    errors: list[str] = []
    usable = accuracy_mod._append_accuracy_recompute_errors(
        errors,
        cert_obj=payload,
        pm=payload["primary_metric"],
        tol=1e-9,
        require_strict=strict,
    )
    return usable, errors


def _bootstrap_errors(report: dict, baseline: dict | None) -> list[str]:
    errors: list[str] = []
    bootstrap_mod.append_strict_ppl_bootstrap_replay_errors(
        errors,
        report=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    return errors
