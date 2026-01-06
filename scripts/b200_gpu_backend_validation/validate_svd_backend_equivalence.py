#!/usr/bin/env python3
"""
validate_svd_backend_equivalence.py

Re-validation harness for CPU vs CUDA backend changes in SVD-based measurements.

This script is intended for GPU boxes (e.g., 8× B200) and produces a JSON report
you can archive alongside a proposed implementation change.

It measures:
  1) Equivalence: CPU vs CUDA numeric outputs on the same inputs.
  2) Determinism: repeat the same computation per backend and record drift.
  3) Calibration impact estimate: implied epsilon deltas when outlier counts differ.

It does not change InvarLock behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _torch_import() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"ERROR: failed to import torch: {exc}") from exc


def _transformers_import() -> Any:
    try:
        import transformers  # type: ignore  # noqa: F401

        return transformers
    except Exception as exc:
        raise SystemExit(
            "ERROR: failed to import transformers. "
            "Install it (or run inside the InvarLock env) for --model tests.\n"
            f"Details: {exc}"
        ) from exc


def _invarlock_rmt_import() -> tuple[Any, Any]:
    try:
        from invarlock.guards.rmt import RMTGuard, mp_bulk_edge  # type: ignore

        return RMTGuard, mp_bulk_edge
    except Exception as exc:
        raise SystemExit(
            "ERROR: failed to import invarlock.guards.rmt. "
            "Run this script from an environment where InvarLock is importable.\n"
            f"Details: {exc}"
        ) from exc


@dataclass(frozen=True)
class NumericDiff:
    abs_max: float
    rel_max: float


@dataclass(frozen=True)
class DeterminismStats:
    repeats: int
    outlier_count_min: int
    outlier_count_max: int
    sigma_max_min: float
    sigma_max_max: float
    max_ratio_min: float
    max_ratio_max: float


@dataclass(frozen=True)
class RMTOutlierResult:
    ok: bool
    outlier_count: int
    sigma_max: float
    max_ratio: float
    mp_edge: float
    threshold: float
    device: str
    seconds: float
    error: str | None = None


@dataclass(frozen=True)
class RMTBackendComparison:
    cpu: RMTOutlierResult
    cuda: RMTOutlierResult
    outlier_count_equal: bool
    sigma_max_diff: NumericDiff | None
    max_ratio_diff: NumericDiff | None
    required_eps: float | None


@dataclass(frozen=True)
class SpectralSigmaResult:
    ok: bool
    sigma_max: float
    device: str
    seconds: float
    error: str | None = None


@dataclass(frozen=True)
class SpectralBackendComparison:
    cpu: SpectralSigmaResult
    cuda: SpectralSigmaResult
    sigma_max_diff: NumericDiff | None


def _set_strictish_determinism(torch: Any) -> dict[str, Any]:
    """Best-effort determinism configuration; record what we managed to set."""
    applied: dict[str, Any] = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        applied["torch.backends.cuda.matmul.allow_tf32"] = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = False
        applied["torch.backends.cudnn.benchmark"] = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        applied["torch.backends.cudnn.deterministic"] = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
        applied["torch.set_float32_matmul_precision"] = "highest"
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        applied["torch.use_deterministic_algorithms"] = "true_warn_only"
    except Exception:
        try:
            torch.use_deterministic_algorithms(True)
            applied["torch.use_deterministic_algorithms"] = "true"
        except Exception:
            pass
    return applied


def _numeric_diff(a: float, b: float) -> NumericDiff:
    abs_max = float(abs(a - b))
    denom = max(abs(a), abs(b), 1e-12)
    rel_max = float(abs_max / denom)
    return NumericDiff(abs_max=abs_max, rel_max=rel_max)


def _maybe_sync(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _rmt_outliers_for_activation(
    torch: Any,
    mp_bulk_edge: Any,
    activations: Any,
    *,
    margin: float,
    deadband: float,
) -> RMTOutlierResult:
    """Compute activation outliers using the same steps as the strict RMT pipeline."""
    if isinstance(activations, tuple | list):
        activations = activations[0] if activations else None
    if not isinstance(activations, torch.Tensor):
        return RMTOutlierResult(
            ok=False,
            outlier_count=0,
            sigma_max=0.0,
            max_ratio=0.0,
            mp_edge=0.0,
            threshold=0.0,
            device="unknown",
            seconds=0.0,
            error="activations is not a torch.Tensor",
        )

    device = str(activations.device)
    t0 = time.perf_counter()
    try:
        if activations.dim() < 2:
            return RMTOutlierResult(
                ok=True,
                outlier_count=0,
                sigma_max=0.0,
                max_ratio=0.0,
                mp_edge=0.0,
                threshold=0.0,
                device=device,
                seconds=time.perf_counter() - t0,
            )
        mat = activations
        if mat.dim() > 2:
            mat = mat.reshape(-1, mat.shape[-1])
        if mat.numel() == 0:
            return RMTOutlierResult(
                ok=True,
                outlier_count=0,
                sigma_max=0.0,
                max_ratio=0.0,
                mp_edge=0.0,
                threshold=0.0,
                device=device,
                seconds=time.perf_counter() - t0,
            )

        mat = mat.detach().float()
        if not torch.isfinite(mat).all():
            return RMTOutlierResult(
                ok=True,
                outlier_count=0,
                sigma_max=0.0,
                max_ratio=0.0,
                mp_edge=0.0,
                threshold=0.0,
                device=device,
                seconds=time.perf_counter() - t0,
            )

        mat = mat - mat.mean()
        std = float(mat.std().item())
        if not math.isfinite(std) or std <= 0.0:
            return RMTOutlierResult(
                ok=True,
                outlier_count=0,
                sigma_max=0.0,
                max_ratio=0.0,
                mp_edge=0.0,
                threshold=0.0,
                device=device,
                seconds=time.perf_counter() - t0,
            )
        mat = mat / std

        m, n = mat.shape
        mp_edge_val = float(mp_bulk_edge(int(m), int(n), whitened=False))
        threshold = mp_edge_val * (1.0 + float(deadband)) * float(margin)

        _maybe_sync(torch, device)
        s_vals = torch.linalg.svdvals(mat)
        _maybe_sync(torch, device)

        if s_vals.numel() == 0:
            return RMTOutlierResult(
                ok=True,
                outlier_count=0,
                sigma_max=0.0,
                max_ratio=0.0,
                mp_edge=mp_edge_val,
                threshold=threshold,
                device=device,
                seconds=time.perf_counter() - t0,
            )

        sigma_max = float(s_vals.max().item())
        max_ratio = float(sigma_max / max(mp_edge_val, 1e-12))
        outlier_count = int((s_vals > threshold).sum().item())

        return RMTOutlierResult(
            ok=True,
            outlier_count=outlier_count,
            sigma_max=sigma_max,
            max_ratio=max_ratio,
            mp_edge=mp_edge_val,
            threshold=threshold,
            device=device,
            seconds=time.perf_counter() - t0,
        )
    except Exception as exc:
        return RMTOutlierResult(
            ok=False,
            outlier_count=0,
            sigma_max=0.0,
            max_ratio=0.0,
            mp_edge=0.0,
            threshold=0.0,
            device=device,
            seconds=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _required_epsilon(*, bare: int, guarded: int) -> float | None:
    if bare < 0 or guarded < 0:
        return None
    if bare == 0:
        return 0.0 if guarded == 0 else None
    return max(0.0, (guarded / bare) - 1.0)


def _compare_rmt_outliers(
    torch: Any,
    mp_bulk_edge: Any,
    activations: Any,
    *,
    margin: float,
    deadband: float,
) -> RMTBackendComparison:
    cuda_res = _rmt_outliers_for_activation(
        torch, mp_bulk_edge, activations, margin=margin, deadband=deadband
    )
    cpu_acts = None
    try:
        if isinstance(activations, torch.Tensor):
            cpu_acts = activations.detach().to("cpu")
    except Exception:
        cpu_acts = None
    cpu_res = _rmt_outliers_for_activation(
        torch, mp_bulk_edge, cpu_acts, margin=margin, deadband=deadband
    )

    sigma_diff = (
        _numeric_diff(cpu_res.sigma_max, cuda_res.sigma_max)
        if (cpu_res.ok and cuda_res.ok)
        else None
    )
    ratio_diff = (
        _numeric_diff(cpu_res.max_ratio, cuda_res.max_ratio)
        if (cpu_res.ok and cuda_res.ok)
        else None
    )
    required_eps = (
        _required_epsilon(bare=cpu_res.outlier_count, guarded=cuda_res.outlier_count)
        if (cpu_res.ok and cuda_res.ok)
        else None
    )

    return RMTBackendComparison(
        cpu=cpu_res,
        cuda=cuda_res,
        outlier_count_equal=bool(cpu_res.outlier_count == cuda_res.outlier_count),
        sigma_max_diff=sigma_diff,
        max_ratio_diff=ratio_diff,
        required_eps=required_eps,
    )


def _spectral_sigma_max(torch: Any, weight: Any) -> SpectralSigmaResult:
    if not isinstance(weight, torch.Tensor):
        return SpectralSigmaResult(
            ok=False,
            sigma_max=0.0,
            device="unknown",
            seconds=0.0,
            error="weight is not a torch.Tensor",
        )
    device = str(weight.device)
    t0 = time.perf_counter()
    try:
        W = weight.detach().float()
        if W.numel() == 0 or W.ndim != 2:
            return SpectralSigmaResult(
                ok=True,
                sigma_max=0.0,
                device=device,
                seconds=time.perf_counter() - t0,
            )
        _maybe_sync(torch, device)
        s_vals = torch.linalg.svdvals(W)
        _maybe_sync(torch, device)
        sigma_max = float(s_vals.max().item()) if s_vals.numel() else 0.0
        return SpectralSigmaResult(
            ok=True,
            sigma_max=sigma_max,
            device=device,
            seconds=time.perf_counter() - t0,
        )
    except Exception as exc:
        return SpectralSigmaResult(
            ok=False,
            sigma_max=0.0,
            device=device,
            seconds=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _compare_spectral_sigma(torch: Any, weight: Any) -> SpectralBackendComparison:
    cuda_weight = weight
    cpu_weight = None
    try:
        if isinstance(weight, torch.Tensor):
            cpu_weight = weight.detach().to("cpu")
    except Exception:
        cpu_weight = None
    cpu_res = _spectral_sigma_max(torch, cpu_weight)
    cuda_res = _spectral_sigma_max(torch, cuda_weight)
    sigma_diff = (
        _numeric_diff(cpu_res.sigma_max, cuda_res.sigma_max)
        if (cpu_res.ok and cuda_res.ok)
        else None
    )
    return SpectralBackendComparison(
        cpu=cpu_res, cuda=cuda_res, sigma_max_diff=sigma_diff
    )


def _determinism_stats_for_rmt(
    torch: Any,
    mp_bulk_edge: Any,
    activations: Any,
    *,
    margin: float,
    deadband: float,
    repeats: int,
) -> dict[str, DeterminismStats]:
    if repeats < 2:
        repeats = 2
    stats: dict[str, DeterminismStats] = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if not isinstance(activations, torch.Tensor):
            continue
        mat = activations.detach()
        mat = mat.to(device)

        counts: list[int] = []
        sigmas: list[float] = []
        ratios: list[float] = []
        for _ in range(repeats):
            res = _rmt_outliers_for_activation(
                torch, mp_bulk_edge, mat, margin=margin, deadband=deadband
            )
            counts.append(int(res.outlier_count))
            sigmas.append(float(res.sigma_max))
            ratios.append(float(res.max_ratio))
        stats[device] = DeterminismStats(
            repeats=repeats,
            outlier_count_min=min(counts),
            outlier_count_max=max(counts),
            sigma_max_min=min(sigmas),
            sigma_max_max=max(sigmas),
            max_ratio_min=min(ratios),
            max_ratio_max=max(ratios),
        )
    return stats


def _select_modules(
    torch: Any, model: Any, *, name_regex: str, max_modules: int
) -> list[tuple[str, Any]]:
    pat = re.compile(name_regex, flags=re.IGNORECASE)
    selected: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not pat.search(name):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.ndim != 2:
            continue
        selected.append((name, module))
        if len(selected) >= max_modules:
            break
    return selected


def _tokenize_prompts(
    tokenizer: Any, prompts: list[str], *, seq_len: int
) -> dict[str, Any]:
    if not prompts:
        prompts = ["Hello world"]
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(seq_len),
    )


def _resolve_model_input_device(torch: Any, model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_model_tests(
    *,
    model_id: str,
    max_modules: int,
    module_regex: str,
    prompts: list[str],
    seq_len: int,
    margin: float,
    deadband: float,
    repeats: int,
    spectral: bool,
) -> dict[str, Any]:
    torch = _torch_import()
    _transformers_import()
    RMTGuard, mp_bulk_edge = _invarlock_rmt_import()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    result: dict[str, Any] = {
        "model_id": model_id,
        "module_regex": module_regex,
        "max_modules": max_modules,
        "seq_len": seq_len,
        "prompts": prompts,
        "rmt": {"margin": margin, "deadband": deadband},
        "modules": [],
        "rmt_comparisons": [],
        "spectral_comparisons": [],
        "families": {},
    }

    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    result["load_seconds"] = time.perf_counter() - t_load0

    selected = _select_modules(
        torch, model, name_regex=module_regex, max_modules=max_modules
    )
    result["modules"] = [name for name, _ in selected]
    if not selected:
        result["error"] = "no modules selected"
        return result

    if spectral:
        spec_records: list[dict[str, Any]] = []
        for module_name, module in selected:
            comp = _compare_spectral_sigma(torch, getattr(module, "weight", None))
            rec = asdict(comp)
            rec["module"] = module_name
            spec_records.append(rec)
        result["spectral_comparisons"] = spec_records

    device_for_inputs = _resolve_model_input_device(torch, model)
    batch = _tokenize_prompts(tok, prompts, seq_len=seq_len)
    try:
        batch = {k: v.to(device_for_inputs) for k, v in batch.items()}
    except Exception:
        batch = {
            k: v.to("cuda" if torch.cuda.is_available() else "cpu")
            for k, v in batch.items()
        }

    comparisons: list[dict[str, Any]] = []
    handles = []

    def make_hook(module_name: str):
        def hook(_module, _inp, out):
            comp = _compare_rmt_outliers(
                torch, mp_bulk_edge, out, margin=margin, deadband=deadband
            )
            rec = asdict(comp)
            rec["module"] = module_name
            comparisons.append(rec)

        return hook

    for module_name, module in selected:
        handles.append(module.register_forward_hook(make_hook(module_name)))

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**batch)
    result["forward_seconds"] = time.perf_counter() - t0

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    result["rmt_comparisons"] = comparisons

    families: dict[str, dict[str, Any]] = {}
    for rec in comparisons:
        module_name = str(rec.get("module", ""))
        family = RMTGuard._classify_family(module_name)
        fam_entry = families.setdefault(
            family,
            {
                "cpu_outliers": 0,
                "cuda_outliers": 0,
                "required_eps": None,
                "modules": [],
            },
        )
        cpu_count = int((rec.get("cpu") or {}).get("outlier_count") or 0)
        cuda_count = int((rec.get("cuda") or {}).get("outlier_count") or 0)
        fam_entry["cpu_outliers"] += cpu_count
        fam_entry["cuda_outliers"] += cuda_count
        fam_entry["modules"].append(module_name)

    for _fam, entry in families.items():
        entry["required_eps"] = _required_epsilon(
            bare=int(entry["cpu_outliers"]), guarded=int(entry["cuda_outliers"])
        )
    result["families"] = families

    # Optional determinism check: run the strict pipeline on a fixed synthetic matrix.
    # This avoids confounding model-level nondeterminism with backend SVD determinism.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    mat_cpu = torch.randn(
        (1024, 2048), generator=gen, dtype=torch.float32, device="cpu"
    )
    result["determinism"] = {
        k: asdict(v)
        for k, v in _determinism_stats_for_rmt(
            torch,
            mp_bulk_edge,
            mat_cpu,
            margin=margin,
            deadband=deadband,
            repeats=repeats,
        ).items()
    }

    return result


def _run_synthetic_tests(
    *,
    shapes: list[tuple[int, int]],
    margin: float,
    deadband: float,
    repeats: int,
) -> dict[str, Any]:
    torch = _torch_import()
    _RMTGuard, mp_bulk_edge = _invarlock_rmt_import()

    out: dict[str, Any] = {
        "shapes": [list(s) for s in shapes],
        "rmt": {"margin": margin, "deadband": deadband, "repeats": repeats},
        "cases": [],
    }

    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)

    for m, n in shapes:
        mat_cpu = torch.randn((m, n), generator=gen, dtype=torch.float32, device="cpu")
        case: dict[str, Any] = {"shape": [m, n]}
        if torch.cuda.is_available():
            mat_cuda = mat_cpu.to("cuda")
            comp = _compare_rmt_outliers(
                torch, mp_bulk_edge, mat_cuda, margin=margin, deadband=deadband
            )
            case["comparison"] = asdict(comp)
            det = _determinism_stats_for_rmt(
                torch,
                mp_bulk_edge,
                mat_cpu,
                margin=margin,
                deadband=deadband,
                repeats=repeats,
            )
            case["determinism"] = {k: asdict(v) for k, v in det.items()}
        out["cases"].append(case)

    return out


def _summarize_rmt(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    count_mismatch = 0
    sigma_rel_max = 0.0
    ratio_rel_max = 0.0
    req_eps_max = 0.0
    req_eps_missing = 0

    for rec in records:
        total += 1
        if not rec.get("outlier_count_equal", False):
            count_mismatch += 1
        sd = rec.get("sigma_max_diff") or {}
        rd = rec.get("max_ratio_diff") or {}
        try:
            sigma_rel_max = max(sigma_rel_max, float(sd.get("rel_max", 0.0)))
        except Exception:
            pass
        try:
            ratio_rel_max = max(ratio_rel_max, float(rd.get("rel_max", 0.0)))
        except Exception:
            pass
        eps = rec.get("required_eps")
        if eps is None:
            req_eps_missing += 1
        else:
            req_eps_max = max(req_eps_max, float(eps))

    mismatch_rate = (count_mismatch / total) if total else 0.0
    return {
        "total": total,
        "outlier_count_mismatch": count_mismatch,
        "outlier_count_mismatch_rate": mismatch_rate,
        "sigma_rel_max": sigma_rel_max,
        "max_ratio_rel_max": ratio_rel_max,
        "required_eps_max": req_eps_max,
        "required_eps_missing": req_eps_missing,
    }


def _summarize_spectral(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    sigma_rel_max = 0.0
    errors = 0
    for rec in records:
        total += 1
        cpu_ok = bool((rec.get("cpu") or {}).get("ok", False))
        cuda_ok = bool((rec.get("cuda") or {}).get("ok", False))
        if not (cpu_ok and cuda_ok):
            errors += 1
        sd = rec.get("sigma_max_diff") or {}
        try:
            sigma_rel_max = max(sigma_rel_max, float(sd.get("rel_max", 0.0)))
        except Exception:
            pass
    return {"total": total, "errors": errors, "sigma_rel_max": sigma_rel_max}


def _collect_system_info(torch: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "env": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_VISIBLE_DEVICES",
            )
            if os.environ.get(k) is not None
        },
        "torch": {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        },
    }
    if torch.cuda.is_available():
        devices = []
        try:
            for i in range(int(torch.cuda.device_count())):
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                    }
                )
        except Exception:
            pass
        info["torch"]["cuda_devices"] = devices
    return info


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_svd_backend_equivalence.py",
        description="Validate CPU vs CUDA equivalence/determinism for SVD-based strict measurements.",
    )
    parser.add_argument("--out", type=str, default="svd_backend_validation.json")
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Enable best-effort deterministic flags (may reduce throughput).",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--synthetic-shapes",
        type=str,
        default="1024x4096,1024x8192,2048x2048",
        help="Comma-separated list like 1024x8192,2048x2048",
    )
    parser.add_argument("--rmt-margin", type=float, default=1.5)
    parser.add_argument("--rmt-deadband", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--module-regex",
        type=str,
        default=r"(?:\\.attn\\.|attention|\\.mlp\\.|ffn|\\.c_fc|\\.c_proj)",
    )
    parser.add_argument("--max-modules", type=int, default=4)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Also compare CPU vs CUDA sigma_max on selected module weight matrices.",
    )
    parser.add_argument(
        "--fail-rmt-mismatch-rate",
        type=float,
        default=None,
        help="Exit non-zero if RMT outlier-count mismatch rate exceeds this value.",
    )
    parser.add_argument(
        "--fail-rmt-sigma-rel-max",
        type=float,
        default=None,
        help="Exit non-zero if max rel error on RMT sigma_max exceeds this value.",
    )
    parser.add_argument(
        "--fail-rmt-required-eps-max",
        type=float,
        default=None,
        help="Exit non-zero if required_eps_max exceeds this value (ignores None).",
    )
    parser.add_argument(
        "--fail-spectral-sigma-rel-max",
        type=float,
        default=None,
        help="Exit non-zero if max rel error on spectral sigma_max exceeds this value.",
    )

    args = parser.parse_args(argv)

    # Best-effort: set CUBLAS_WORKSPACE_CONFIG before torch CUDA init for determinism.
    determinism_env_applied: dict[str, Any] = {}
    if args.strict_determinism and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        determinism_env_applied["env.CUBLAS_WORKSPACE_CONFIG"] = os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ]

    torch = _torch_import()
    determinism_applied = dict(determinism_env_applied)
    if args.strict_determinism:
        determinism_applied.update(_set_strictish_determinism(torch))

    report: dict[str, Any] = {
        "started_at": _now_iso(),
        "system": _collect_system_info(torch),
        "determinism": determinism_applied,
        "results": {},
    }

    prompts = args.prompt or [
        "The quick brown fox jumps over the lazy dog.",
        "Explain why the sky appears blue during the day.",
    ]

    if args.synthetic:
        shapes: list[tuple[int, int]] = []
        for entry in str(args.synthetic_shapes).split(","):
            entry = entry.strip()
            if not entry:
                continue
            m_str, x, n_str = entry.partition("x")
            if not x:
                raise SystemExit(f"Invalid shape: {entry} (expected like 1024x4096)")
            shapes.append((int(m_str), int(n_str)))
        report["results"]["synthetic"] = _run_synthetic_tests(
            shapes=shapes,
            margin=float(args.rmt_margin),
            deadband=float(args.rmt_deadband),
            repeats=int(args.repeats),
        )

    if args.model:
        model_results: list[dict[str, Any]] = []
        for model_id in args.model:
            model_results.append(
                _run_model_tests(
                    model_id=model_id,
                    max_modules=int(args.max_modules),
                    module_regex=str(args.module_regex),
                    prompts=prompts,
                    seq_len=int(args.seq_len),
                    margin=float(args.rmt_margin),
                    deadband=float(args.rmt_deadband),
                    repeats=int(args.repeats),
                    spectral=bool(args.spectral),
                )
            )
        report["results"]["models"] = model_results

    # Summaries
    rmt_records: list[dict[str, Any]] = []
    syn = (report.get("results") or {}).get("synthetic") or {}
    for case in syn.get("cases", []) if isinstance(syn, dict) else []:
        comp = case.get("comparison")
        if isinstance(comp, dict):
            rmt_records.append(comp)
    models = (report.get("results") or {}).get("models") or []
    spectral_records: list[dict[str, Any]] = []
    if isinstance(models, list):
        for mrec in models:
            for comp in (
                mrec.get("rmt_comparisons", []) if isinstance(mrec, dict) else []
            ):
                if isinstance(comp, dict):
                    rmt_records.append(comp)
            for comp in (
                mrec.get("spectral_comparisons", []) if isinstance(mrec, dict) else []
            ):
                if isinstance(comp, dict):
                    spectral_records.append(comp)

    summary = {
        "rmt": _summarize_rmt(rmt_records),
        "spectral": _summarize_spectral(spectral_records) if spectral_records else None,
    }
    report["summary"] = summary
    report["finished_at"] = _now_iso()

    # Fail criteria
    failures: list[str] = []
    rmt_sum = summary.get("rmt") or {}
    if args.fail_rmt_mismatch_rate is not None:
        if float(rmt_sum.get("outlier_count_mismatch_rate", 0.0)) > float(
            args.fail_rmt_mismatch_rate
        ):
            failures.append("rmt.mismatch_rate")
    if args.fail_rmt_sigma_rel_max is not None:
        if float(rmt_sum.get("sigma_rel_max", 0.0)) > float(
            args.fail_rmt_sigma_rel_max
        ):
            failures.append("rmt.sigma_rel_max")
    if args.fail_rmt_required_eps_max is not None:
        if float(rmt_sum.get("required_eps_max", 0.0)) > float(
            args.fail_rmt_required_eps_max
        ):
            failures.append("rmt.required_eps_max")
    if (
        args.fail_spectral_sigma_rel_max is not None
        and summary.get("spectral") is not None
    ):
        spec_sum = summary.get("spectral") or {}
        if float(spec_sum.get("sigma_rel_max", 0.0)) > float(
            args.fail_spectral_sigma_rel_max
        ):
            failures.append("spectral.sigma_rel_max")
    report["failures"] = failures

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote: {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
