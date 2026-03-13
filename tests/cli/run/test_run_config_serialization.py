from __future__ import annotations

from invarlock.cli.commands import run
from invarlock.cli.config import load_config


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

    as_dict = run._to_serialisable_dict(variance_cfg)
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
    assert run._prune_none_values(payload) == {"b": [1, {"d": 2}], "e": (3,)}


def test_to_serialisable_dict_falls_back_when_dict_method_raises():
    class ExplodingDict:
        def __init__(self):
            self._data = {"x": 1}

        def dict(self):
            raise RuntimeError("boom")

    assert run._to_serialisable_dict(ExplodingDict()) == {"x": 1}


def test_to_serialisable_dict_uses_vars_when_data_getattr_raises():
    class Weird:
        def __init__(self):
            object.__setattr__(self, "_data", {"y": 2})

        def __getattribute__(self, name):
            if name == "_data":
                raise RuntimeError("boom")
            return object.__getattribute__(self, name)

    assert run._to_serialisable_dict(Weird()) == {"y": 2}


def test_to_serialisable_dict_returns_empty_dict_when_vars_fails():
    class NoVars:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

    assert run._to_serialisable_dict(NoVars()) == {}
