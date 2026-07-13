from __future__ import annotations

from invarlock.core.runtime_observation import observe_model_runtime


def test_runtime_observation_records_stable_named_module_and_weight_paths() -> None:
    class Weight:
        pass

    class Module:
        weight = Weight()

    module = Module()

    class Model:
        def named_modules(self):
            return [("encoder.block.0", module)]

    observed, observations = observe_model_runtime(Model())
    assert observed is True
    assert [(item.kind, item.path) for item in observations] == [
        ("module", "encoder.block.0"),
        ("direct_weight", "encoder.block.0.weight"),
    ]


def test_runtime_observation_fails_closed_for_non_module_model() -> None:
    observed, observations = observe_model_runtime(object())
    assert observed is False
    assert observations == ()
