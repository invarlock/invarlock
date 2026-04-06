"""Runtime config ownership for InvarLock.

Provides the lightweight, dict-backed config model plus runtime config loading
and profile application used by shells and services.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources as _ires
from pathlib import Path
from typing import Any

import yaml

from .config_dependencies import load_raw_config_payload as _load_raw_config_payload


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalize_section_value(value: Any) -> Any:
    if is_dataclass(value):
        return _section_dataclass_payload(value)
    if isinstance(value, dict):
        normalized_dict: dict[str, Any] = {}
        for key, item in value.items():
            normalized_item = _normalize_section_value(item)
            if normalized_item is None:
                continue
            normalized_dict[key] = normalized_item
        return normalized_dict
    if isinstance(value, list):
        return [
            normalized_item
            for item in value
            if (normalized_item := _normalize_section_value(item)) is not None
        ]
    if isinstance(value, tuple):
        return tuple(
            normalized_item
            for item in value
            if (normalized_item := _normalize_section_value(item)) is not None
        )
    return copy.deepcopy(value)


def _section_dataclass_payload(instance: Any) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance).__name__}")
    payload: dict[str, Any] = {}
    for section_field in fields(instance):
        if not section_field.init or section_field.name == "_extra":
            continue
        value = getattr(instance, section_field.name)
        if value is None:
            continue
        payload[section_field.name] = _normalize_section_value(value)
    extra = getattr(instance, "_extra", None)
    if isinstance(extra, dict):
        payload.update(_normalize_section_value(extra))
    return payload


def _split_known_fields(
    cls: type[Any], value: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    known_names = {f.name for f in fields(cls) if f.init and f.name != "_extra"}
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, item in value.items():
        if key in known_names:
            known[key] = item
        else:
            extra[key] = item
    return known, extra


class SectionMixin:
    def __getitem__(self, key: str) -> Any:
        payload = _section_dataclass_payload(self)
        if key in payload and hasattr(self, key):
            return getattr(self, key)
        if key in payload:
            return payload[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
            return
        extra = getattr(self, "_extra", None)
        if isinstance(extra, dict):
            extra[key] = value
            return
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in _section_dataclass_payload(self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):  # pragma: no cover - debug/test helper
        return _section_dataclass_payload(self).items()


@dataclass
class ModelConfig(SectionMixin):
    id: str | None = None
    adapter: str | None = None
    device: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditConfig(SectionMixin):
    name: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig(SectionMixin):
    dir: Path | str = Path(".")
    model_dir: str | Path | None = None
    model_subdir: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)


@dataclass
class GuardsConfig(SectionMixin):
    order: list[str] = field(default_factory=list)
    variance: VarianceGuardConfig | dict[str, Any] | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalLossConfig(SectionMixin):
    type: str | None = None
    mask_prob: float | None = None
    seed: int | None = None
    random_token_prob: float | None = None
    original_token_prob: float | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig(SectionMixin):
    metric: dict[str, Any] | None = None
    bootstrap: EvalBootstrapConfig | dict[str, Any] = field(default_factory=dict)
    loss: EvalLossConfig | dict[str, Any] | None = None
    spike_threshold: float = 2.0
    max_pm_ratio: float = 1.5
    capacity_fast: bool = False
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bootstrap:
            self.bootstrap = EvalBootstrapConfig()
        elif isinstance(self.bootstrap, dict):
            known, extra = _split_known_fields(EvalBootstrapConfig, self.bootstrap)
            self.bootstrap = EvalBootstrapConfig(_extra=extra, **known)
        if self.loss is not None and isinstance(self.loss, dict):
            known, extra = _split_known_fields(EvalLossConfig, self.loss)
            self.loss = EvalLossConfig(_extra=extra, **known)


def _normalize_top_level_section(name: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    if name == "model":
        known, extra = _split_known_fields(ModelConfig, value)
        return ModelConfig(_extra=extra, **known)
    if name == "edit":
        known, extra = _split_known_fields(EditConfig, value)
        plan = known.get("plan")
        if not isinstance(plan, dict):
            known["plan"] = {}
        return EditConfig(_extra=extra, **known)
    if name == "dataset":
        known, extra = _split_known_fields(DatasetConfig, value)
        cfg = DatasetConfig(**known)
        if extra:
            cfg._extra = extra
        return cfg
    if name == "output":
        known, extra = _split_known_fields(OutputConfig, value)
        return OutputConfig(_extra=extra, **known)
    if name == "auto":
        known, extra = _split_known_fields(AutoConfig, value)
        return AutoConfig(_extra=extra, **known)
    if name == "guards":
        variance = value.get("variance")
        if isinstance(variance, dict):
            known, extra = _split_known_fields(VarianceGuardConfig, variance)
            variance = VarianceGuardConfig(_extra=extra, **known)
        guard_value = {k: v for k, v in value.items() if k != "variance"}
        known, extra = _split_known_fields(GuardsConfig, guard_value)
        return GuardsConfig(variance=variance, _extra=extra, **known)
    if name == "eval":
        known, extra = _split_known_fields(EvalConfig, value)
        return EvalConfig(_extra=extra, **known)
    return value


class InvarLockConfig(MutableMapping[str, Any]):
    """Explicit mutable mapping for runtime configuration.

    Stores runtime configuration as a plain nested mapping. Nested sections are
    accessed deliberately via mapping operations or `section()` /
    `require_section()`.
    """

    data: dict[str, Any]

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = copy.deepcopy(dict(data))
        for key, value in list(self.data.items()):
            self.data[key] = _normalize_top_level_section(key, value)

    @classmethod
    def from_sections(cls, **sections: Any) -> InvarLockConfig:
        return cls(sections)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = _normalize_top_level_section(key, value)

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def model_dump(self) -> dict[str, Any]:
        dumped: dict[str, Any] = {}
        for key, value in self.data.items():
            if is_dataclass(value):
                dumped[key] = _section_dataclass_payload(value)
            else:
                dumped[key] = copy.deepcopy(value)
        return dumped

    def section(self, name: str) -> dict[str, Any] | None:
        value = self.data.get(name)
        if value is None:
            return None
        if is_dataclass(value):
            return _section_dataclass_payload(value)
        if isinstance(value, dict):
            return copy.deepcopy(value)
        raise TypeError(
            f"Config section '{name}' must be a mapping, got {type(value).__name__}."
        )

    def require_section(self, name: str) -> dict[str, Any]:
        value = self.section(name)
        if value is None:
            raise KeyError(f"Config section '{name}' is required.")
        return value

    @property
    def model(self) -> ModelConfig:
        value = self.data.get("model")
        if not isinstance(value, ModelConfig):
            raise KeyError("Config section 'model' is required.")
        return value

    @property
    def edit(self) -> EditConfig:
        value = self.data.get("edit")
        if not isinstance(value, EditConfig):
            raise KeyError("Config section 'edit' is required.")
        return value

    @property
    def dataset(self) -> DatasetConfig:
        value = self.data.get("dataset")
        if not isinstance(value, DatasetConfig):
            raise KeyError("Config section 'dataset' is required.")
        return value

    @property
    def output(self) -> OutputConfig:
        value = self.data.get("output")
        if not isinstance(value, OutputConfig):
            raise KeyError("Config section 'output' is required.")
        return value

    @property
    def auto(self) -> AutoConfig:
        value = self.data.get("auto")
        if not isinstance(value, AutoConfig):
            raise KeyError("Config section 'auto' is required.")
        return value

    @property
    def guards(self) -> GuardsConfig:
        value = self.data.get("guards")
        if not isinstance(value, GuardsConfig):
            raise KeyError("Config section 'guards' is required.")
        return value

    @property
    def eval(self) -> EvalConfig:
        value = self.data.get("eval")
        if not isinstance(value, EvalConfig):
            raise KeyError("Config section 'eval' is required.")
        return value

    @property
    def context(self) -> dict[str, Any]:
        value = self.data.get("context")
        if value is None:
            raise KeyError("Config section 'context' is required.")
        if not isinstance(value, dict):
            raise TypeError("Config section 'context' must be a mapping.")
        return value


@dataclass
class DatasetConfig(SectionMixin):
    id: str | None = None
    seq_len: int = 512
    stride: int = 512
    provider: str | dict[str, Any] | None = None
    split: str = "validation"
    preview_n: int | None = None
    final_n: int | None = None
    seed: int | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stride > self.seq_len:
            raise ValueError("stride must be <= seq_len")


@dataclass
class EvalBootstrapConfig:
    replicates: int = 1000
    alpha: float = 0.05
    ci_band: float = 0.10
    method: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replicates <= 0:
            raise ValueError("replicates must be > 0")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0,1)")


@dataclass
class SpectralGuardConfig(SectionMixin):
    sigma_quantile: float | None = None
    family_caps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # normalize family_caps: scalar → {"kappa": value}
        caps = {}
        for k, v in (self.family_caps or {}).items():
            if isinstance(v, dict):
                caps[k] = {"kappa": float(v.get("kappa", 0.0))}
            else:
                caps[k] = {"kappa": float(v)}
        self.family_caps = caps


@dataclass
class RMTGuardConfig(SectionMixin):
    epsilon: dict[str, float] | float | None = None


@dataclass
class VarianceGuardConfig(SectionMixin):
    clamp: list[float] | None = None
    mode: str | None = None
    deadband: float | None = None
    min_gain: float | None = None
    min_rel_gain: float | None = None
    min_abs_adjust: float | None = None
    max_scale_step: float | None = None
    min_effect_lognll: float | None = None
    predictive_one_sided: bool | None = None
    topk_backstop: int | None = None
    max_adjusted_modules: int | None = None
    predictive_gate: bool | None = None
    target_modules: list[str] | None = None
    scope: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    max_calib: int | None = None
    absolute_floor_ppl: float | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clamp is not None:
            if not (isinstance(self.clamp, list) and len(self.clamp) == 2):
                raise ValueError("clamp must be [low, high]")
            low, high = float(self.clamp[0]), float(self.clamp[1])
            if low >= high:
                raise ValueError("clamp lower bound must be < upper bound")
        if self.absolute_floor_ppl is None:
            # Provide conservative default when not specified
            self.absolute_floor_ppl = 0.05


@dataclass
class AutoConfig:
    enabled: bool = True
    tier: str = "balanced"
    probes: int = 0
    target_pm_ratio: float = 1.0
    _extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0 <= int(self.probes) <= 10):
            raise ValueError("probes must be between 0 and 10")
        if float(self.target_pm_ratio) < 1.0:
            raise ValueError("target_pm_ratio must be >= 1.0")


def load_config(path: str | Path) -> InvarLockConfig:
    raw = _load_raw_config_payload(path)
    defaults = raw.pop("defaults", None)
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping when present")
    if isinstance(defaults, dict):
        raw = _deep_merge(defaults, raw)

    edit_block = raw.get("edit")
    if isinstance(edit_block, dict) and "parameters" in edit_block:
        raise ValueError("edit.parameters is not supported; use edit.plan.")
    if isinstance(edit_block, dict) and "kind" in edit_block:
        raise ValueError(
            "edit.kind is not supported; use edit.name with a canonical edit plugin name."
        )

    # "assurance" (strict/fast) was removed in the GPU/MPS-first measurement-contract
    # world. Fail closed so outdated configs are updated explicitly.
    if raw.get("assurance") is not None:
        raise ValueError(
            "assurance.* is not supported; configure measurement contracts under guards.* "
            "(e.g., guards.spectral.estimator, guards.rmt.activation.sampling)."
        )

    # Per-guard strict/fast mode overrides were also removed. Fail closed to avoid
    # silently accepting configs that no longer apply.
    guards_block = raw.get("guards")
    if isinstance(guards_block, dict):
        for guard_name in ("spectral", "rmt"):
            node = guards_block.get(guard_name)
            if isinstance(node, dict) and "mode" in node:
                raise ValueError(
                    f"guards.{guard_name}.mode is not supported; remove it and configure "
                    "measurement-contract knobs under guard policy fields instead."
                )

    # Coerce known guard configs for friendlier attribute access
    guards = raw.get("guards")
    if isinstance(guards, dict):
        var = guards.get("variance")
        if isinstance(var, dict):
            # Pick only recognized keys
            vkw = {
                k: var.get(k)
                for k in [
                    "clamp",
                    "mode",
                    "deadband",
                    "min_gain",
                    "min_rel_gain",
                    "min_abs_adjust",
                    "max_scale_step",
                    "min_effect_lognll",
                    "predictive_one_sided",
                    "topk_backstop",
                    "max_adjusted_modules",
                    "predictive_gate",
                    "target_modules",
                    "scope",
                    "calibration",
                    "absolute_floor_ppl",
                ]
            }
            if vkw.get("mode") is None:
                vkw["mode"] = "ci"
            guards["variance"] = VarianceGuardConfig(
                **{k: v for k, v in vkw.items() if v is not None}
            )
    return InvarLockConfig(raw)


def _load_runtime_yaml(*rel_parts: str) -> dict[str, Any] | None:
    """Load YAML from the runtime config locations.

    Search order:
      1) $INVARLOCK_CONFIG_ROOT/runtime/...
      2) invarlock._data.runtime package resources
    Returns mapping or None if not found.
    """
    # 1) Environment override
    root = os.getenv("INVARLOCK_CONFIG_ROOT")
    if root:
        p = Path(root) / "runtime"
        for part in rel_parts:
            p = p / part
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                if not isinstance(data, dict):
                    raise ValueError("Runtime YAML must be a mapping")
                return data

    # 2) Package data
    try:
        base = _ires.files("invarlock._data.runtime")
        res = base
        for part in rel_parts:
            res = res.joinpath(part)
        # Traversable API: try reading if file-like
        try:
            is_file = getattr(res, "is_file", None)
            read_text = getattr(res, "read_text", None)
            if callable(is_file) and is_file() and callable(read_text):
                text = read_text(encoding="utf-8")
                data = yaml.safe_load(text) or {}
                if not isinstance(data, dict):
                    raise ValueError("Runtime YAML must be a mapping")
                return data
        except FileNotFoundError:
            pass
    except Exception:
        # Importlib resources may not be available in certain environments
        pass
    return None


def load_tiers() -> dict[str, Any]:
    """Load tier policies from runtime locations."""
    data = _load_runtime_yaml("tiers.yaml")
    if data is not None:
        return data
    raise FileNotFoundError(
        "tiers.yaml not found in package runtime (and no INVARLOCK_CONFIG_ROOT override)"
    )


def apply_profile(cfg: InvarLockConfig, profile: str) -> InvarLockConfig:
    # First, try packaged/runtime profiles
    overrides: dict[str, Any] | None = _load_runtime_yaml("profiles", f"{profile}.yaml")

    if overrides is None:
        raise ValueError(f"Unknown profile: {profile}")
    base_cfg = cfg.model_dump()
    merged = _deep_merge(base_cfg, overrides)

    # Runtime profiles provide defaults, but model/preset-specific primary-metric
    # policy must remain authoritative when explicitly configured.
    base_primary_metric = (
        base_cfg.get("primary_metric")
        if isinstance(base_cfg.get("primary_metric"), dict)
        else {}
    )
    merged_primary_metric = (
        merged.get("primary_metric")
        if isinstance(merged.get("primary_metric"), dict)
        else {}
    )
    if base_primary_metric and merged_primary_metric is not None:
        for key, value in base_primary_metric.items():
            merged_primary_metric[key] = copy.deepcopy(value)
        merged["primary_metric"] = merged_primary_metric

    return InvarLockConfig(merged)
