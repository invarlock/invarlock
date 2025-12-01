"""Minimal CLI config implementation for invarlock.cli.

Provides a lightweight, dict-backed configuration object plus helpers used by
the CLI commands (load_config, apply_profile, apply_edit_override, resolve_edit_kind).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from importlib import resources as _ires
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class _Obj:
    def __init__(self, data: Any):
        self._data = data

    def __getattr__(self, item):
        # Only return values for existing keys; otherwise raise AttributeError
        # so hasattr/getattr(..., default) behave correctly.
        if item in self._data:
            v = self._data[item]
            if isinstance(v, dict):
                return _Obj(v)
            return v
        raise AttributeError(item)

    def __getitem__(self, key):  # enable dict-like access in tests
        if isinstance(self._data, dict):
            return self._data[key]
        raise TypeError("Object is not subscriptable")

    # Provide dict-like helpers where tests use mapping semantics
    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._data, dict):
            return self._data.get(key, default)
        return default

    def items(self):  # pragma: no cover - convenience for debug/tests
        if isinstance(self._data, dict):
            return self._data.items()
        return []


@dataclass
class InvarLockConfig:
    """Lightweight, dict-backed config with ergonomic attribute access.

    Accepts either a single `data` mapping or keyword sections like `model=`,
    `edit=`, `dataset=`, etc., and stores them internally as a dict.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __init__(self, data: dict[str, Any] | None = None, **sections: Any) -> None:
        if data is not None and sections:
            merged = _deep_merge(data, sections)
            self.data = merged
        elif data is not None:
            self.data = copy.deepcopy(data)
        else:
            self.data = copy.deepcopy(sections)

        # Basic validation hooks for well-known edits (none required here)

    def model_dump(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def __getattr__(self, item):
        if item in self.data:
            v = self.data[item]
            if isinstance(v, dict):
                return _Obj(v)
            return v
        raise AttributeError(item)


# Typed sub-configs used by tests (minimal validation only)
@dataclass
class OutputConfig:
    dir: Path | str

    def __post_init__(self) -> None:
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)


@dataclass
class DatasetConfig:
    seq_len: int = 512
    stride: int = 512
    provider: str | None = None
    split: str = "validation"
    preview_n: int | None = None
    final_n: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.stride > self.seq_len:
            raise ValueError("stride must be <= seq_len")


@dataclass
class EvalBootstrapConfig:
    replicates: int = 1000
    alpha: float = 0.05
    ci_band: float = 0.10

    def __post_init__(self) -> None:
        if self.replicates <= 0:
            raise ValueError("replicates must be > 0")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0,1)")


@dataclass
class SpectralGuardConfig:
    sigma_quantile: float | None = None
    contraction: float | None = None
    family_caps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # contraction is an alias for sigma_quantile
        if self.contraction is not None and self.sigma_quantile is None:
            self.sigma_quantile = float(self.contraction)
            self.contraction = None
        # normalize family_caps: scalar → {"kappa": value}
        caps = {}
        for k, v in (self.family_caps or {}).items():
            if isinstance(v, dict):
                caps[k] = {"kappa": float(v.get("kappa", 0.0))}
            else:
                caps[k] = {"kappa": float(v)}
        self.family_caps = caps


@dataclass
class RMTGuardConfig:
    epsilon: dict[str, float] | float | None = None


@dataclass
class VarianceGuardConfig:
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
    absolute_floor_ppl: float | None = None

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
class EditConfig:
    name: str
    plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoConfig:
    probes: int = 0
    target_pm_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= int(self.probes) <= 10):
            raise ValueError("probes must be between 0 and 10")
        if float(self.target_pm_ratio) < 1.0:
            raise ValueError("target_pm_ratio must be >= 1.0")


def _create_loader(base_dir: Path):
    class Loader(yaml.SafeLoader):
        pass

    Loader._base_dir = Path(base_dir)

    def _construct_include(loader: yaml.SafeLoader, node: yaml.Node):
        rel = loader.construct_scalar(node)
        path = (loader._base_dir / rel).resolve()
        with path.open(encoding="utf-8") as fh:
            inc_loader = _create_loader(path.parent)
            return yaml.load(fh, Loader=inc_loader)

    Loader.add_constructor("!include", _construct_include)
    return Loader


def load_config(path: str | Path) -> InvarLockConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    loader = _create_loader(p.parent)
    with p.open(encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=loader)
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")
    defaults = raw.pop("defaults", None)
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping when present")
    if isinstance(defaults, dict):
        raw = _deep_merge(defaults, raw)
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
            if getattr(res, "is_file", None) and res.is_file():  # type: ignore[attr-defined]
                text = res.read_text(encoding="utf-8")  # type: ignore[assignment]
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
        # Provide sensible CI defaults when 'ci' profile file is absent
        if profile.lower() == "ci":
            try:
                prev = int(os.getenv("INVARLOCK_CI_PREVIEW", "200"))
            except Exception:
                prev = 200
            try:
                fin = int(os.getenv("INVARLOCK_CI_FINAL", "200"))
            except Exception:
                fin = 200
            overrides = {
                "dataset": {"preview_n": prev, "final_n": fin},
                "eval": {"bootstrap": {"replicates": 1200, "alpha": 0.05}},
            }
        else:
            raise ValueError(f"Unknown profile: {profile}")
    return InvarLockConfig(_deep_merge(cfg.model_dump(), overrides))


def resolve_edit_kind(kind: str) -> str:
    kind = kind.lower().strip()
    # Aliases for common edit types
    mapping = {
        "prune": "quant_rtn",
        "quant": "quant_rtn",
        "mixed": "orchestrator",
    }
    # Direct mapping for aliased kinds
    if kind in mapping:
        return mapping[kind]
    # Check if the kind is a registered edit name (e.g., "noop", "quant_rtn")
    try:
        from invarlock.edits.registry import get_registry

        registry = get_registry()
        if registry.get_plugin(kind) is not None:
            return kind
    except ImportError:
        pass
    # Also allow well-known edit names directly
    known_edits = {"quant_rtn", "noop"}
    if kind in known_edits:
        return kind
    raise ValueError(f"Unknown edit kind: {kind}")


def apply_edit_override(cfg: InvarLockConfig, kind: str) -> InvarLockConfig:
    cfgd = cfg.model_dump()
    resolved = resolve_edit_kind(kind)
    edit_section = cfgd.setdefault("edit", {})
    edit_section["name"] = resolved
    edit_section["kind"] = kind
    return InvarLockConfig(cfgd)


# Backward-compat helper name expected by tests
def _deep_merge_dicts(a: dict, b: dict) -> dict:  # pragma: no cover - trivial alias
    return _deep_merge(a, b)


def create_example_config() -> InvarLockConfig:  # pragma: no cover - test helper
    return InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_gpt2", "device": "auto"},
        edit={"name": "quant_rtn", "plan": {}},
        dataset={"provider": "wikitext2", "seq_len": 512, "stride": 512},
        output={"dir": "runs"},
    )
