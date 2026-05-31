from __future__ import annotations

import importlib

import pytest

from invarlock.cli import run_config as run_serial_mod
from invarlock.core.config_loader import load_config


def test_run_config_guard_overrides_prune_none(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
guards:
  variance:
    calibration:
      windows: 6
      seed: 42
      min_coverage: 4
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    variance_cfg = cfg.guards.variance

    as_dict = run_serial_mod._to_serialisable_dict(variance_cfg)
    assert as_dict["mode"] == "ci"
    assert as_dict["calibration"]["windows"] == 6

    # These fields exist on the dataclass with default None values, but they must not
    # be passed through as explicit policy overrides (they would clobber tier defaults).
    for key in (
        "clamp",
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
    ):
        assert key not in as_dict


def test_prune_none_values_drops_list_and_tuple_entries():
    payload = {"a": None, "b": [1, None, {"c": None, "d": 2}], "e": (None, 3)}
    assert run_serial_mod._prune_none_values(payload) == {
        "b": [1, {"d": 2}],
        "e": (3,),
    }


def test_to_serialisable_dict_falls_back_when_dict_method_raises():
    class ExplodingDict:
        def __init__(self):
            self._data = {"x": 1}

        def dict(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_serial_mod._to_serialisable_dict(ExplodingDict())


def test_to_serialisable_dict_exercises_non_model_dump_fallback_sources():
    class ModelDumpDict:
        def model_dump(self):
            return {"answer": 42}

    assert run_serial_mod._to_serialisable_dict(ModelDumpDict()) == {"answer": 42}

    class ModelDumpNonDict:
        def __init__(self):
            self.answer = 42

        def model_dump(self):
            return ["not", "a", "dict"]

    assert run_serial_mod._to_serialisable_dict(ModelDumpNonDict()) == {"answer": 42}

    class DictReturnsMapping:
        def dict(self):
            return {"from_dict": True}

    assert run_serial_mod._to_serialisable_dict(DictReturnsMapping()) == {
        "from_dict": True
    }

    class DictReturnsNonMapping:
        def __init__(self):
            self.value = "from-vars"

        def dict(self):
            return ["not", "a", "dict"]

    assert run_serial_mod._to_serialisable_dict(DictReturnsNonMapping()) == {
        "value": "from-vars"
    }

    class DictRaisesTypeError:
        def __init__(self):
            self._data = {"fallback": True}

        def dict(self):
            raise TypeError("bad dict")

    assert run_serial_mod._to_serialisable_dict(DictRaisesTypeError()) == {
        "fallback": True
    }

    class DataObject:
        def __init__(self):
            self._data = {"x": 1}

    assert run_serial_mod._to_serialisable_dict(DataObject()) == {"x": 1}

    class DataRaisesAttributeError:
        def __init__(self):
            self.value = "kept"

        def __getattribute__(self, name):
            if name == "_data":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    assert run_serial_mod._to_serialisable_dict(DataRaisesAttributeError()) == {
        "value": "kept"
    }

    class VarsDataFallback:
        def __init__(self):
            self._data = {"from_vars": True}

        def __getattribute__(self, name):
            if name == "_data":
                return "not-a-dict"
            return object.__getattribute__(self, name)

    assert run_serial_mod._to_serialisable_dict(VarsDataFallback()) == {
        "from_vars": True
    }


def test_to_serialisable_dict_uses_vars_when_data_getattr_raises():
    class Weird:
        def __init__(self):
            object.__setattr__(self, "_data", {"y": 2})

        def __getattribute__(self, name):
            if name == "_data":
                raise RuntimeError("boom")
            return object.__getattribute__(self, name)

    with pytest.raises(RuntimeError, match="boom"):
        run_serial_mod._to_serialisable_dict(Weird())


def test_to_serialisable_dict_returns_empty_dict_when_vars_fails():
    class NoVars:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

    assert run_serial_mod._to_serialisable_dict(NoVars()) == {}


def test_to_serialisable_dict_returns_empty_when_vars_payload_is_not_mapping():
    class NonMappingVars:
        @property
        def __dict__(self):  # noqa: D401
            return []

    assert run_serial_mod._to_serialisable_dict(NonMappingVars()) == {}


def test_prepare_config_for_run_returns_loaded_config_when_auto_adapter_unavailable(
    monkeypatch,
) -> None:
    cfg = {"loaded": True}
    real_import_module = importlib.import_module

    def _missing_adapter(name: str):
        if name == "invarlock.adapters.auto":
            raise ImportError(name)
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", _missing_adapter)

    assert (
        run_serial_mod.prepare_config_for_run(
            config_path="config.yaml",
            profile=None,
            edit=None,
            tier=None,
            probes=None,
            load_config_fn=lambda _path: cfg,
        )
        is cfg
    )
