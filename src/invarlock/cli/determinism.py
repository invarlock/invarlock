"""Determinism presets for CI/release runs.

Centralizes:
- Seeds (python/numpy/torch)
- Thread caps (OMP/MKL/etc + torch threads)
- TF32 policy
- torch deterministic algorithms
- A structured "determinism level" for certificate provenance
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from invarlock.model_utils import set_seed

try:  # optional torch
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_profile(profile: str | None) -> str:
    try:
        return (profile or "").strip().lower()
    except Exception:
        return ""


def _coerce_device(device: str | None) -> str:
    try:
        return (device or "").strip().lower()
    except Exception:
        return "cpu"


def apply_determinism_preset(
    *,
    profile: str | None,
    device: str | None,
    seed: int,
    threads: int = 1,
) -> dict[str, Any]:
    """Apply a determinism preset and return a provenance payload."""

    prof = _coerce_profile(profile)
    dev = _coerce_device(device)
    threads_i = max(1, _coerce_int(threads, 1))

    requested = "off"
    if prof in {"ci", "release"}:
        requested = "strict"

    env_set: dict[str, Any] = {}
    torch_flags: dict[str, Any] = {}
    notes: list[str] = []

    # Thread caps (best-effort): make CPU determinism explicit and reduce drift.
    if requested == "strict":
        for var in _THREAD_ENV_VARS:
            os.environ[var] = str(threads_i)
            env_set[var] = os.environ.get(var)

    # CUDA determinism: cuBLAS workspace config.
    if requested == "strict" and dev.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        env_set["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    # Seed all RNGs (python/numpy/torch) using the existing helper for parity.
    set_seed(int(seed))

    # Derive a stable seed bundle for provenance.
    seed_bundle = {
        "python": int(seed),
        "numpy": int(seed),
        "torch": None,
    }
    try:
        numpy_seed = int(np.random.get_state()[1][0])
        seed_bundle["numpy"] = int(numpy_seed)
    except Exception:
        pass
    if torch is not None:
        try:
            seed_bundle["torch"] = int(torch.initial_seed())
        except Exception:
            seed_bundle["torch"] = int(seed)

    # Torch-specific controls.
    level = "off" if requested == "off" else "strict"
    if requested == "strict":
        if torch is None:
            level = "tolerance"
            notes.append("torch_unavailable")
        else:
            # Thread caps.
            try:
                if hasattr(torch, "set_num_threads"):
                    torch.set_num_threads(threads_i)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(threads_i)
                torch_flags["torch_threads"] = threads_i
            except Exception:
                level = "tolerance"
                notes.append("torch_thread_caps_failed")

            # Disable TF32 for determinism.
            try:
                matmul = getattr(
                    getattr(torch.backends, "cuda", object()), "matmul", None
                )
                if matmul is not None and hasattr(matmul, "allow_tf32"):
                    matmul.allow_tf32 = False
                cudnn_mod = getattr(torch.backends, "cudnn", None)
                if cudnn_mod is not None and hasattr(cudnn_mod, "allow_tf32"):
                    cudnn_mod.allow_tf32 = False
            except Exception:
                level = "tolerance"
                notes.append("tf32_policy_failed")

            # Deterministic algorithms.
            try:
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                # Downgrade to tolerance-based determinism rather than crashing.
                level = "tolerance"
                notes.append("deterministic_algorithms_unavailable")
                try:
                    if hasattr(torch, "use_deterministic_algorithms"):
                        torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass

            # cuDNN knobs.
            try:
                cudnn_mod = getattr(torch.backends, "cudnn", None)
                if cudnn_mod is not None:
                    cudnn_mod.benchmark = False
                    if hasattr(cudnn_mod, "deterministic"):
                        cudnn_mod.deterministic = True
            except Exception:
                level = "tolerance"
                notes.append("cudnn_determinism_failed")

            # Snapshot applied flags for provenance.
            try:
                det_enabled = getattr(
                    torch, "are_deterministic_algorithms_enabled", None
                )
                if callable(det_enabled):
                    torch_flags["deterministic_algorithms"] = bool(det_enabled())
            except Exception:
                pass
            try:
                cudnn_mod = getattr(torch.backends, "cudnn", None)
                if cudnn_mod is not None:
                    torch_flags["cudnn_deterministic"] = bool(
                        getattr(cudnn_mod, "deterministic", False)
                    )
                    torch_flags["cudnn_benchmark"] = bool(
                        getattr(cudnn_mod, "benchmark", False)
                    )
                    if hasattr(cudnn_mod, "allow_tf32"):
                        torch_flags["cudnn_allow_tf32"] = bool(
                            getattr(cudnn_mod, "allow_tf32", False)
                        )
            except Exception:
                pass
            try:
                matmul = getattr(
                    getattr(torch.backends, "cuda", object()), "matmul", None
                )
                if matmul is not None and hasattr(matmul, "allow_tf32"):
                    torch_flags["cuda_matmul_allow_tf32"] = bool(matmul.allow_tf32)
            except Exception:
                pass

    # Normalized level is always one of these.
    if level not in {"off", "strict", "tolerance"}:
        level = "tolerance" if requested == "strict" else "off"

    # Extra breadcrumb: random module state is not easily serializable; include a coarse marker.
    try:
        torch_flags["python_random"] = isinstance(random.random(), float)
    except Exception:
        pass

    payload = {
        "requested": requested,
        "level": level,
        "profile": prof or None,
        "device": dev,
        "threads": threads_i if requested == "strict" else None,
        "seed": int(seed),
        "seeds": seed_bundle,
        "env": env_set,
        "torch": torch_flags,
        "notes": notes,
    }

    # Remove empty sections for stable artifacts.
    if not payload["env"]:
        payload.pop("env", None)
    if not payload["torch"]:
        payload.pop("torch", None)
    if not payload["notes"]:
        payload.pop("notes", None)
    if payload.get("threads") is None:
        payload.pop("threads", None)
    if payload.get("profile") is None:
        payload.pop("profile", None)

    return payload


__all__ = ["apply_determinism_preset"]
