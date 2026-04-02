from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "activation_edge_risk",
    "activation_svd_outliers",
    "batch_token_weight",
    "collect_calibration_batches",
    "compute_activation_edge_risk",
    "compute_activation_outliers",
    "get_activation_modules",
    "prepare_activation_inputs",
]


def collect_calibration_batches(
    calib: Any,
    max_windows: int,
    *,
    activation_sampling: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Collect a deterministic slice of calibration batches."""
    if calib is None or max_windows <= 0:
        return []

    source = getattr(calib, "dataloader", None) or calib
    try:
        if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
            source_items: Any = source
            n = int(len(source_items))
            if n <= 0:
                return []
            count = min(int(max_windows), n)
            policy = (
                (activation_sampling or {}).get("windows", {})
                if isinstance(activation_sampling, Mapping)
                else {}
            )
            indices_policy = "evenly_spaced"
            if isinstance(policy, Mapping):
                indices_policy = str(
                    policy.get("indices_policy", indices_policy) or indices_policy
                )
            policy_name = indices_policy.strip().lower()
            if policy_name == "last":
                idxs = list(range(max(0, n - count), n))
            elif policy_name == "evenly_spaced":
                if count <= 1:
                    idxs = [0]
                else:
                    idxs = [
                        int(round(i * (n - 1) / float(count - 1))) for i in range(count)
                    ]
            else:
                idxs = list(range(count))
            batches: list[Any] = []
            for idx in idxs:
                try:
                    batches.append(source_items[idx])
                except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
                    continue
            return batches
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass

    try:
        iterator = iter(source)
    except TypeError:
        return []
    return list(itertools.islice(iterator, max_windows))


def prepare_activation_inputs(
    batch: Any, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Normalize batch inputs to tensors on the target device."""
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids", batch.get("inputs"))
        attention_mask = batch.get("attention_mask")
    elif isinstance(batch, tuple | list) and batch:
        input_ids = batch[0]
        attention_mask = batch[1] if len(batch) > 1 else None
    else:
        input_ids = batch
        attention_mask = None

    if input_ids is None:
        return None, None

    try:
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        try:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.clone()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None, None

    if attention_mask is not None:
        try:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            try:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.as_tensor(attention_mask)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                attention_mask = attention_mask.clone()
            except (AttributeError, TypeError, ValueError, RuntimeError):
                attention_mask = None

    return input_ids, attention_mask


def batch_token_weight(
    input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
) -> int:
    """Compute token-weight for a batch (used for activation outlier weighting)."""
    weight = 0
    if isinstance(attention_mask, torch.Tensor):
        try:
            weight = int(attention_mask.sum().item())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            weight = 0
    if weight <= 0 and isinstance(input_ids, torch.Tensor):
        try:
            weight = int(input_ids.numel())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            weight = 0
    return max(weight, 1)


def get_activation_modules(
    model: nn.Module, *, allowed_suffixes: Sequence[str]
) -> list[tuple[str, nn.Module]]:
    """Return modules to analyze for activation-based RMT."""
    modules: list[tuple[str, nn.Module]] = []
    try:
        from transformers.pytorch_utils import Conv1D

        module_types: tuple[type[nn.Linear], type[nn.Conv1d], type[Conv1D]]
        module_types = (nn.Linear, nn.Conv1d, Conv1D)
    except ImportError:
        module_types = (nn.Linear, nn.Conv1d)

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            modules.append((name, module))
            continue
        if isinstance(module, nn.LayerNorm):
            modules.append((name, module))
            continue
        if isinstance(module, module_types) and hasattr(module, "weight"):
            name_lower = name.lower()
            if any(name.endswith(suffix) for suffix in allowed_suffixes) or any(
                tok in name_lower
                for tok in (
                    "attn",
                    "attention",
                    "mlp",
                    "ffn",
                    "router",
                    "expert",
                    "moe",
                    "gate",
                    "gating",
                    "switch",
                )
            ):
                modules.append((name, module))

    modules.sort(key=lambda t: t[0])
    return modules


def activation_edge_risk(
    activations: Any,
    *,
    estimator: Mapping[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    """Compute activation edge-risk score r = σ̂max(A') / σ_MP(m,n)."""
    if isinstance(activations, tuple | list):
        activations = activations[0] if activations else None
    if not isinstance(activations, torch.Tensor):
        return None
    if activations.dim() < 2:
        return None
    if activations.dim() > 2:
        activations = activations.reshape(-1, activations.shape[-1])
    if activations.numel() == 0:
        return None

    mat = activations.detach()
    if mat.shape[0] <= 0 or mat.shape[1] <= 0:
        return None
    if not torch.isfinite(mat).all():
        return None

    eps = 1e-12
    try:
        mu = mat.mean(dtype=torch.float32)
        norm = torch.linalg.vector_norm(mat.reshape(-1), ord=2, dtype=torch.float32)
        mean_sq = (norm * norm) / float(mat.numel())
        var = mean_sq - (mu * mu)
        std = torch.sqrt(var.clamp_min(eps))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not torch.isfinite(mu) or not torch.isfinite(std):
        return None
    std_val = float(std.item())
    if not math.isfinite(std_val) or std_val <= 0.0:
        return None

    try:
        from . import rmt_math

        mp_edge_val = rmt_math.mp_bulk_edge(
            int(mat.shape[0]), int(mat.shape[1]), whitened=False
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not (math.isfinite(mp_edge_val) and mp_edge_val > 0.0):
        return None

    try:
        iters = int((estimator or {}).get("iters", 3) or 3)
    except (AttributeError, TypeError, ValueError, OverflowError):
        iters = 3
    if iters < 1:
        iters = 1
    init = str((estimator or {}).get("init", "ones") or "ones").strip().lower()
    if init not in {"ones", "e0"}:
        init = "ones"

    device = mat.device
    dtype = mat.dtype

    with torch.inference_mode():
        if init == "ones":
            v = torch.ones((mat.shape[1],), device=device, dtype=dtype)
        else:
            v = torch.zeros((mat.shape[1],), device=device, dtype=dtype)
            v[0] = 1
        v = v / torch.linalg.vector_norm(v.float()).clamp_min(eps).to(dtype)

        mu_d = mu.to(dtype)
        inv_std_d = (1.0 / std.clamp_min(eps)).to(dtype)
        ones_n = torch.ones((mat.shape[1],), device=device, dtype=dtype)

        sigma = 0.0
        for _ in range(iters):
            v_sum = torch.sum(v.float())
            u = mat @ v
            u = (u - (mu_d * v_sum.to(dtype))) * inv_std_d
            u_norm = torch.linalg.vector_norm(u.float()).clamp_min(eps)
            sigma_val = float(u_norm.item())
            if not math.isfinite(sigma_val):
                return None
            u = u / u_norm.to(dtype)

            u_sum = torch.sum(u.float())
            v = mat.T @ u
            v = (v - (mu_d * u_sum.to(dtype) * ones_n)) * inv_std_d
            v_norm = torch.linalg.vector_norm(v.float()).clamp_min(eps)
            v = v / v_norm.to(dtype)
            sigma = sigma_val

    risk = float(sigma) / max(float(mp_edge_val), eps)
    return float(risk), float(sigma), float(mp_edge_val)


def activation_svd_outliers(
    activations: Any, *, margin: float, deadband: float
) -> tuple[int, float, float]:
    """Count activation singular values beyond the MP edge."""
    if isinstance(activations, tuple | list):
        activations = activations[0] if activations else None
    if not isinstance(activations, torch.Tensor):
        return 0, 0.0, 0.0

    if activations.dim() < 2:
        return 0, 0.0, 0.0

    if activations.dim() > 2:
        activations = activations.reshape(-1, activations.shape[-1])

    if activations.numel() == 0:
        return 0, 0.0, 0.0

    try:
        mat = activations.detach().float().cpu()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0, 0.0, 0.0

    if not torch.isfinite(mat).all():
        return 0, 0.0, 0.0

    mat = mat - mat.mean()
    std = float(mat.std().item())
    if not math.isfinite(std) or std <= 0.0:
        return 0, 0.0, 0.0

    mat = mat / std
    m, n = mat.shape
    from . import rmt_math

    mp_edge_val = rmt_math.mp_bulk_edge(m, n, whitened=False)
    threshold = mp_edge_val * (1.0 + deadband) * margin

    try:
        s_vals = torch.linalg.svdvals(mat)
    except (RuntimeError, torch.linalg.LinAlgError):
        return 0, 0.0, 0.0

    if s_vals.numel() == 0:
        return 0, 0.0, 0.0

    sigma_max = float(s_vals.max().item())
    max_ratio = sigma_max / max(mp_edge_val, 1e-12)
    outlier_count = int((s_vals > threshold).sum().item())
    return outlier_count, float(max_ratio), sigma_max


def compute_activation_edge_risk(
    model: nn.Module,
    batches: list[Any],
    *,
    allowed_suffixes: Sequence[str],
    activation_sampling: Mapping[str, Any] | None,
    estimator: Mapping[str, Any] | None,
    deadband: float,
    margin: float,
    classify_family_fn: Callable[[str], str],
) -> dict[str, Any] | None:
    """Compute token-weighted activation edge-risk scores per module/family."""
    if not batches:
        return None

    modules = get_activation_modules(model, allowed_suffixes=allowed_suffixes)
    if not modules:
        return None

    acc: dict[str, dict[str, float]] = {}
    for name, _module in modules:
        acc[name] = {"weighted_sum": 0.0, "weight": 0.0, "max_risk": 0.0}

    batch_weight_holder = {"weight": 1}
    handles: list[Any] = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            out = activation_edge_risk(output, estimator=estimator)
            if out is None:
                return
            risk, _sigma, _edge = out
            try:
                weight = int(batch_weight_holder.get("weight", 1) or 1)
            except (TypeError, ValueError, RuntimeError):
                weight = 1
            row = acc.get(name)
            if row is None:
                return
            row["weighted_sum"] = float(row.get("weighted_sum", 0.0)) + float(
                risk
            ) * float(weight)
            row["weight"] = float(row.get("weight", 0.0)) + float(weight)
            row["max_risk"] = max(float(row.get("max_risk", 0.0)), float(risk))

        return _hook

    for name, module in modules:
        try:
            handles.append(module.register_forward_hook(_make_hook(name)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue

    batches_used = 0
    token_weight_total = 0

    model_was_training = model.training
    model.eval()
    try:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return None
        with torch.inference_mode():
            for batch in batches:
                inputs, attention_mask = prepare_activation_inputs(batch, device)
                if inputs is None:
                    continue
                batch_weight = batch_token_weight(inputs, attention_mask)
                batch_weight_holder["weight"] = batch_weight
                try:
                    if attention_mask is not None:
                        model(inputs, attention_mask=attention_mask)
                    else:
                        model(inputs)
                except TypeError:
                    try:
                        model(inputs)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        continue
                except (AttributeError, RuntimeError, ValueError):
                    continue
                batches_used += 1
                token_weight_total += batch_weight
    finally:
        for handle in handles:
            try:
                handle.remove()
            except (AttributeError, RuntimeError):
                pass
        if model_was_training:
            model.train()

    if batches_used <= 0:
        return None

    edge_risk_by_module: dict[str, float] = {}
    for name, row in acc.items():
        w = float(row.get("weight", 0.0) or 0.0)
        if w <= 0.0:
            continue
        edge_risk_by_module[name] = float(row.get("weighted_sum", 0.0) or 0.0) / w

    if not edge_risk_by_module:
        return None

    edge_risk_by_family: dict[str, float] = {}
    for name, risk in edge_risk_by_module.items():
        family = classify_family_fn(name)
        edge_risk_by_family[family] = max(
            float(edge_risk_by_family.get(family, 0.0)), float(risk)
        )

    for family_key in ("attn", "ffn", "embed", "other"):
        edge_risk_by_family.setdefault(family_key, 0.0)

    _ = activation_sampling
    _ = deadband
    _ = margin
    return {
        "analysis_source": "activations_edge_risk",
        "edge_risk_by_module": edge_risk_by_module,
        "edge_risk_by_family": edge_risk_by_family,
        "token_weight_total": int(token_weight_total),
        "batches_used": int(batches_used),
    }


def compute_activation_outliers(
    guard: Any, model: nn.Module, batches: list[Any]
) -> dict[str, Any] | None:
    """Compute activation-based RMT outlier counts using a guard facade."""
    if not batches:
        return None

    modules = guard._get_activation_modules(model)
    if not modules:
        return None

    per_layer_map: dict[str, dict[str, Any]] = {}
    batch_weight_holder = {"weight": 1}
    for idx, (module_name, _module) in enumerate(modules):
        per_layer_map[module_name] = {
            "layer": idx,
            "module_name": module_name,
            "sigma_max": 0.0,
            "worst_ratio": 0.0,
            "outlier_count": 0,
            "has_outlier": False,
        }

    handles: list[Any] = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            try:
                outliers, max_ratio, sigma_max = guard._activation_svd_outliers(
                    output, margin=guard.margin, deadband=guard.deadband
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return
            stats = per_layer_map.get(name)
            if stats is None:
                return
            weight = int(batch_weight_holder.get("weight", 1) or 1)
            if outliers > 0:
                increment = int(outliers) * weight
                stats["outlier_count"] = int(stats.get("outlier_count", 0)) + increment
                stats["has_outlier"] = True
            stats["worst_ratio"] = max(
                float(stats.get("worst_ratio", 0.0)), float(max_ratio)
            )
            stats["sigma_max"] = max(
                float(stats.get("sigma_max", 0.0)), float(sigma_max)
            )

        return _hook

    for name, module in modules:
        try:
            handles.append(module.register_forward_hook(_make_hook(name)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue

    batches_used = 0
    token_weight_total = 0

    model_was_training = model.training
    model.eval()
    try:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return None
        with torch.inference_mode():
            for batch in batches:
                inputs, attention_mask = guard._prepare_activation_inputs(batch, device)
                if inputs is None:
                    continue
                batch_weight = guard._batch_token_weight(inputs, attention_mask)
                batch_weight_holder["weight"] = batch_weight
                try:
                    if attention_mask is not None:
                        model(inputs, attention_mask=attention_mask)
                    else:
                        model(inputs)
                    batches_used += 1
                    token_weight_total += batch_weight
                except TypeError:
                    try:
                        model(inputs)
                        batches_used += 1
                        token_weight_total += batch_weight
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        continue
                except (AttributeError, RuntimeError, ValueError):
                    continue
    finally:
        for handle in handles:
            try:
                handle.remove()
            except (AttributeError, RuntimeError):
                pass
        if model_was_training:
            model.train()

    if batches_used == 0:
        return None

    per_layer = [per_layer_map[name] for name, _module in modules]
    flagged_layers = [info["layer"] for info in per_layer if info.get("has_outlier")]
    outlier_total = sum(int(info.get("outlier_count", 0) or 0) for info in per_layer)
    max_ratio = max(
        (float(info.get("worst_ratio", 0.0)) for info in per_layer), default=0.0
    )

    return {
        "has_outliers": bool(flagged_layers),
        "n_layers_flagged": len(flagged_layers),
        "outlier_count": outlier_total,
        "max_ratio": max_ratio,
        "threshold": (1.0 + guard.deadband) * guard.margin,
        "per_layer": per_layer,
        "flagged_layers": flagged_layers,
        "analysis_source": "activations",
        "token_weight_total": int(token_weight_total),
        "token_weighted": True,
    }
