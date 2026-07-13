from __future__ import annotations

import builtins
import math
from types import SimpleNamespace

import click
import pytest

from invarlock.cli import run_config as run_config_mod
from invarlock.cli import run_pairing as pairing_mod
from invarlock.cli import run_pairing as run_pairing_mod
from invarlock.cli import run_runtime_exec as run_runtime_exec_mod
from invarlock.cli.run_runtime_snapshot import SnapshotRestoreFailed
from invarlock.core.exceptions import InvarlockError


def test_hash_sequences_falls_back_when_len_unavailable():
    # Inner generator sequences do not support len(), exercising the fallback path.
    seqs = ((i for i in [1, 2, 3]), (i for i in [4, 5]))
    assert pairing_mod._hash_sequences(seqs) == "e08215eb1a73f6d493dfb9f17c0de613"


def test_tensor_or_list_to_ints_reraises_click_exit(monkeypatch):
    class ExplodingIter:
        def __iter__(self):
            raise click.exceptions.Exit(2)

    fake_tensor = SimpleNamespace(tolist=lambda: ExplodingIter())
    monkeypatch.setattr(pairing_mod, "torch", SimpleNamespace(), raising=False)
    with pytest.raises(click.exceptions.Exit):
        pairing_mod._tensor_or_list_to_ints(fake_tensor)


def test_tensor_or_list_to_ints_returns_empty_for_nonconvertible_tensor_like(
    monkeypatch,
) -> None:
    class BadRaw:
        def __iter__(self):
            raise TypeError("not iterable")

    fake_tensor = SimpleNamespace(tolist=lambda: BadRaw())
    monkeypatch.setattr(pairing_mod, "torch", SimpleNamespace(), raising=False)

    assert pairing_mod._tensor_or_list_to_ints(fake_tensor) == []


def test_get_torch_returns_none_when_torch_import_fails(monkeypatch) -> None:
    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "torch":
            raise ImportError("torch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setattr(pairing_mod, "torch", pairing_mod._IMPORT_UNSET)

    assert pairing_mod._get_torch() is None


def test_tensor_or_list_to_ints_returns_empty_for_non_iterable_without_torch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pairing_mod, "torch", None, raising=False)

    assert pairing_mod._tensor_or_list_to_ints(object()) == []


def test_resolve_provider_and_split_provider_and_split_access_errors():
    class BadDataset:
        @property
        def provider(self):
            raise RuntimeError("boom")

        @property
        def split(self):
            raise RuntimeError("boom")

    def _get_provider(name, **kwargs):  # noqa: ARG001
        class Provider:
            def available_splits(self):
                return ["train", "validation"]

        return Provider()

    cfg = SimpleNamespace(dataset=BadDataset())
    with pytest.raises(RuntimeError, match="boom"):
        run_config_mod.resolve_provider_and_split(
            cfg,
            model_profile=SimpleNamespace(default_provider="synthetic"),
            get_provider_fn=_get_provider,
            provider_kwargs=None,
            resolved_device="cpu",
        )


def test_extract_model_load_kwargs_rejects_removed_dtype_aliases_and_preserves_custom_strings():
    class _Cfg:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "fp16",
                }
            }

    with pytest.raises(InvarlockError) as excinfo:
        run_config_mod.extract_model_load_kwargs(
            _Cfg(), invarlock_error_cls=InvarlockError
        )

    assert excinfo.value.code == "E007"
    assert excinfo.value.details == {
        "removed_values": ["model.dtype=fp16"],
        "replacement": "model.dtype=float16",
    }

    class _Cfg2:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "custom_dtype",
                }
            }

    assert run_config_mod.extract_model_load_kwargs(
        _Cfg2(),
        invarlock_error_cls=InvarlockError,
    ) == {"dtype": "custom_dtype"}


def test_run_bare_control_skip_model_load_requires_live_model(monkeypatch):
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.set_seed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(execute=lambda **kwargs: None),
    )

    with pytest.raises(
        SnapshotRestoreFailed,
        match="bare control without a live model instance",
    ):
        run_runtime_exec_mod.run_bare_control(
            adapter=SimpleNamespace(),
            edit_op=None,
            cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
            model=None,
            run_config=SimpleNamespace(event_path=None, context={}),
            calibration_data=[],
            auto_config=None,
            edit_config={},
            preview_count=1,
            final_count=1,
            seed_bundle={"python": 1},
            resolved_device="cpu",
            restore_fn=None,
            resolved_loss_type="causal",
            skip_model_load=True,
        )


@pytest.mark.parametrize("metric_kind", ["ppl_causal", "accuracy"])
def test_run_bare_control_retains_real_metric_specific_evidence(
    monkeypatch, metric_kind: str
) -> None:
    if metric_kind == "accuracy":
        metrics = {
            "primary_metric": {"kind": "accuracy", "preview": 0.8, "final": 0.8},
            "classification": {"final": {"correct_total": 8, "total": 10}},
        }
        windows = {"final": {"example_ids": list(range(10))}}
        loss_type = "classification"
    else:
        metrics = {
            "primary_metric": {"kind": "ppl_causal", "preview": 2.0, "final": 2.0},
            "logloss_final": math.log(2.0),
            "final_total_tokens": 2,
        }
        windows = {
            "final": {
                "window_ids": [1, 2],
                "logloss": [math.log(2.0), math.log(2.0)],
                "token_counts": [1, 1],
            }
        }
        loss_type = "causal"
    report = SimpleNamespace(
        metrics=metrics,
        evaluation_windows=windows,
        status="success",
    )
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.set_seed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(execute=lambda **kwargs: report),
    )
    monkeypatch.setattr(
        run_runtime_exec_mod, "_capture_backend_inventory", lambda **_: None
    )
    monkeypatch.setattr(run_runtime_exec_mod, "release_process_memory", lambda: None)

    payload = run_runtime_exec_mod.run_bare_control(
        adapter=SimpleNamespace(),
        edit_op=None,
        cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
        model=object(),
        run_config=SimpleNamespace(event_path=None, context={}),
        calibration_data=[],
        auto_config=None,
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 1},
        resolved_device="cpu",
        restore_fn=lambda: None,
        resolved_loss_type=loss_type,
        profile_normalized="ci",
    )

    assert payload is not None
    assert payload["bare_report"]["primary_metric"]["kind"] == metric_kind
    assert payload["bare_report"]["status"] == "success"
    assert payload["bare_facts"]["example_ids_digest"]


def test_run_bare_control_rejects_non_success_status_before_report_assembly(
    monkeypatch,
) -> None:
    report = SimpleNamespace(status="failed")
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.set_seed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(execute=lambda **kwargs: report),
    )
    monkeypatch.setattr(
        run_runtime_exec_mod, "_capture_backend_inventory", lambda **_: None
    )
    monkeypatch.setattr(run_runtime_exec_mod, "release_process_memory", lambda: None)

    with pytest.raises(InvarlockError) as exc_info:
        run_runtime_exec_mod.run_bare_control(
            adapter=SimpleNamespace(),
            edit_op=None,
            cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
            model=object(),
            run_config=SimpleNamespace(event_path=None, context={}),
            calibration_data=[],
            auto_config=None,
            edit_config={},
            preview_count=1,
            final_count=1,
            seed_bundle={"python": 1},
            resolved_device="cpu",
            restore_fn=lambda: None,
            resolved_loss_type="causal",
            profile_normalized="ci",
        )

    assert exc_info.value.code == "E009"
    assert "GUARD-METRIC-BARE-CONTROL-FAILED" in exc_info.value.message


def test_run_bare_control_surfaces_the_underlying_runner_error(monkeypatch) -> None:
    report = SimpleNamespace(
        status="failed", error="vision_text batch is missing image_path"
    )
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.set_seed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(execute=lambda **kwargs: report),
    )
    monkeypatch.setattr(
        run_runtime_exec_mod, "_capture_backend_inventory", lambda **_: None
    )
    monkeypatch.setattr(run_runtime_exec_mod, "release_process_memory", lambda: None)

    with pytest.raises(InvarlockError, match="missing image_path"):
        run_runtime_exec_mod.run_bare_control(
            adapter=SimpleNamespace(),
            edit_op=None,
            cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
            model=object(),
            run_config=SimpleNamespace(event_path=None, context={}),
            calibration_data=[],
            auto_config=None,
            edit_config={},
            preview_count=1,
            final_count=1,
            seed_bundle={"python": 1},
            resolved_device="cpu",
            restore_fn=lambda: None,
            resolved_loss_type="causal",
            profile_normalized="ci",
        )


def test_execute_guarded_run_skip_model_load_requires_live_model():
    with pytest.raises(
        SnapshotRestoreFailed,
        match="guarded execution without a live model instance",
    ):
        run_runtime_exec_mod.execute_guarded_run(
            runner=SimpleNamespace(execute=lambda **kwargs: None),
            adapter=SimpleNamespace(),
            model=None,
            cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
            edit_op=None,
            run_config=SimpleNamespace(event_path=None, context={}),
            guards=[],
            calibration_data=[],
            auto_config=None,
            edit_config={},
            preview_count=1,
            final_count=1,
            restore_fn=None,
            resolved_device="cpu",
            skip_model_load=True,
        )


def test_validate_baseline_schedule_reraises_unexpected_runtime_errors():
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=1, final_n=1))
    schedule = {
        "preview": {
            "window_ids": [0],
            "input_ids": [[1, 2]],
            "attention_masks": [[1, 1]],
        },
        "final": {
            "window_ids": [1],
            "input_ids": [[3, 4]],
            "attention_masks": [[1, 1]],
        },
    }

    def _broken_tensor_or_list_to_ints(_value):  # noqa: ANN001
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_pairing_mod.validate_and_harvest_baseline_schedule(
            cfg,
            schedule,
            baseline_report_data={"data": {}},
            tokenizer_hash="hash",
            resolved_loss_type="causal",
            tensor_or_list_to_ints_fn=_broken_tensor_or_list_to_ints,
        )


def test_validate_baseline_schedule_typed_failures_raise_invarlock_error():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )
    )
    schedule = {
        "preview": {"window_ids": [0], "input_ids": [[1, 2]]},
        "final": {"window_ids": [1], "input_ids": [[3, 4]]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 4,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    with pytest.raises(InvarlockError, match="PAIRING-EVIDENCE-MISSING"):
        run_pairing_mod.validate_and_harvest_baseline_schedule(
            cfg,
            schedule,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            typed_failures=True,
        )
