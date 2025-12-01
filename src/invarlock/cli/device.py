"""Minimal device helpers for the CLI."""

from __future__ import annotations

from typing import Any


def resolve_device(requested: str | None) -> str:
    req = (requested or "auto").lower()
    if req != "auto":
        if not is_device_available(req):
            raise RuntimeError(f"Device '{req}' is not available")
        return req
    # Prefer CUDA → MPS → CPU
    if is_device_available("cuda"):
        # Resolve to first CUDA device explicitly
        return "cuda:0"
    if is_device_available("mps"):
        return "mps"
    return "cpu"


def is_device_available(device: str) -> bool:
    d = (device or "cpu").lower()
    # Normalize CUDA variants like 'cuda:0' → 'cuda'
    if d.startswith("cuda"):
        d = "cuda"
    if d == "cpu":
        return True
    try:
        import torch  # noqa: F401

        if d == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            return True
        if (
            d == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):  # type: ignore[attr-defined]
            return True
    except Exception:
        return False
    return False


def validate_device_for_config(
    device: str, config_requirements: dict[str, Any] | None = None
) -> tuple[bool, str]:
    # Simple validation stub; extend with model/profile specific checks as needed
    valid = {"cpu", "cuda", "cuda:0", "mps"}
    if device not in valid:
        return False, f"Unsupported device '{device}'"
    if config_requirements and config_requirements.get("required_device"):
        req = str(config_requirements.get("required_device")).lower()
        if device != req:
            return (
                False,
                f"Configuration requires device '{req}' but '{device}' was selected",
            )
    return True, ""


def get_device_info() -> dict[str, dict]:
    """Return a structured snapshot of device availability.

    Keys: 'cpu', 'cuda', 'mps', and 'auto_selected'.
    """
    info: dict[str, dict] = {
        "cpu": {"available": True, "info": "Available"},
        "cuda": {"available": False, "info": "Not available"},
        "mps": {"available": False, "info": "Not available"},
    }
    auto = resolve_device("auto")
    try:
        import torch  # noqa: F401

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            info["mps"]["available"] = True
            info["mps"]["info"] = "Available"
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            props = torch.cuda.get_device_properties(0)
            name = getattr(props, "name", "CUDA")
            mem = getattr(props, "total_memory", 0)
            info["cuda"]["available"] = True
            info["cuda"]["info"] = "Available"
            info["cuda"]["device_count"] = torch.cuda.device_count()
            info["cuda"]["device_name"] = name
            info["cuda"]["memory_total"] = f"{mem / 1e9:.1f} GB"
    except Exception:
        pass
    info["auto_selected"] = auto
    return info
