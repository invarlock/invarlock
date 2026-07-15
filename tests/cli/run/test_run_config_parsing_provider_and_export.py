from __future__ import annotations

import json
from collections import UserDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer
from rich.console import Console

from invarlock.cli.commands.run import run_command
from invarlock.eval.data import EvaluationWindow
from tests.cli.run._support_run_config_parsing import (
    ConfigParsingCfg as _Cfg,
)
from tests.cli.run._support_run_config_parsing import (
    config_parsing_core_report as _core_report,
)
from tests.cli.run._support_run_config_parsing import (
    config_parsing_detect_profile as _detect_profile,
)
from tests.cli.run._support_run_config_parsing import (
    config_parsing_tokenizer as _tok,
)


class _DictNoItems(dict):
    def __getattribute__(self, name: str):  # noqa: ANN001
        if name == "items":
            raise AttributeError(name)
        return super().__getattribute__(name)


class _TruthyEmptyDict(dict):
    def __bool__(self) -> bool:
        return True


def _pm_stub(*_a, **_k):
    return {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 1.0,
    }


def _provider_windows(
    preview_n: int, final_n: int
) -> tuple[EvaluationWindow, EvaluationWindow]:
    prev = EvaluationWindow(
        input_ids=[
            [1 + 4 * i, 2 + 4 * i, 3 + 4 * i, 4 + 4 * i] for i in range(preview_n)
        ],
        attention_masks=[[1, 1, 1, 1] for _ in range(preview_n)],
        indices=list(range(preview_n)),
    )
    fin = EvaluationWindow(
        input_ids=[
            [101 + 4 * i, 102 + 4 * i, 103 + 4 * i, 104 + 4 * i] for i in range(final_n)
        ],
        attention_masks=[[1, 1, 1, 1] for _ in range(final_n)],
        indices=[1000 + i for i in range(final_n)],
    )
    return prev, fin


def _run_with_common_patches(
    *, cfg: _Cfg, exec_stub, post_stub, extra_patches=(), run_kwargs=None
):
    run_kwargs = run_kwargs or {}

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    patches = [
        patch("invarlock.cli.run_config.prepare_config_for_run", lambda **k: cfg),
        patch("invarlock.cli.run_runtime_exec.detect_model_profile", _detect_profile),
        patch(
            "invarlock.cli.run_runtime_exec.resolve_tokenizer", lambda *_a, **_k: _tok()
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch("invarlock.core.registry.get_registry", lambda: Registry()),
        patch(
            "invarlock.core.orchestration.execute._should_measure_metric_impact_impl",
            lambda *_a: (False, False, None),
        ),
        patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
        patch("invarlock.cli.run_execution.postprocess_and_summarize", post_stub),
        patch(
            "invarlock.cli.run_pairing.resolve_metric_and_provider",
            lambda *_a, **_k: ("ppl_causal", None, {}),
        ),
        patch(
            "invarlock.eval.primary_metric.compute_primary_metric_from_report", _pm_stub
        ),
    ]
    patches.extend(extra_patches)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(cfg.output.dir),
            until_pass=False,
            **run_kwargs,
        )


def test_run_command_provider_dict_unwraps_nested_kwargs(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def resolver(*_a, **kwargs):  # noqa: ANN001
        captured["provider_kwargs"] = kwargs.get("provider_kwargs")
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider={
            "kind": "hf",
            "dataset_name": "wikitext",
            "cache_dir": "",
            "max_samples": None,
        },
    )

    _run_with_common_patches(
        cfg=cfg,
        exec_stub=exec_stub,
        post_stub=post_stub,
        extra_patches=(
            patch("invarlock.cli.run_config.resolve_provider_and_split", resolver),
        ),
    )

    provider_kwargs = captured["provider_kwargs"]
    assert isinstance(provider_kwargs, dict)
    assert provider_kwargs.get("dataset_name") == "wikitext"
    assert "cache_dir" not in provider_kwargs
    assert "max_samples" not in provider_kwargs


def test_run_command_provider_mapping_like_unwraps_data_and_items_fallback(
    tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []

    def resolver(*_a, **kwargs):  # noqa: ANN001
        captured.append(dict(kwargs.get("provider_kwargs") or {}))
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    class ProviderObj:
        def __init__(
            self, data: dict[str, object], *, break_data: bool = False
        ) -> None:
            self._data = None if break_data else data
            self._items = data

        def get(self, key: str, default=None):  # noqa: ANN001
            return self._items.get(key, default)

        def items(self):
            return self._items.items()

        def __bool__(self) -> bool:
            return True

    cfg_data = {"kind": "hf", "dataset_name": "wikitext", "cache_dir": ""}
    cfg = _Cfg(outdir=tmp_path / "runs", dataset_provider=ProviderObj(cfg_data))
    cfg2 = _Cfg(
        outdir=tmp_path / "runs2",
        dataset_provider=ProviderObj(cfg_data, break_data=True),
    )

    extra = (patch("invarlock.cli.run_config.resolve_provider_and_split", resolver),)
    _run_with_common_patches(
        cfg=cfg, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )
    _run_with_common_patches(
        cfg=cfg2, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )

    assert captured and captured[0].get("dataset_name") == "wikitext"
    assert captured[1].get("dataset_name") == "wikitext"


def test_run_command_extracts_edit_config_from_plan(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def resolver(*_a, **_k):
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        ec = kwargs.get("edit_config") or {}
        captured.append(dict(ec))
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    # plan unwrap: non-dict mapping with .items and no _data
    plan = UserDict({"alpha": 1, "beta": 2})
    cfg_plan = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        edit_plan=plan,
    )

    extra = (
        patch("invarlock.cli.run_config.resolve_provider_and_split", resolver),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: SimpleNamespace(
                windows=lambda **_kw: _provider_windows(1, 1)
            ),
        ),
    )
    _run_with_common_patches(
        cfg=cfg_plan, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )
    assert captured[0] == {"alpha": 1, "beta": 2}


def test_run_command_rejects_legacy_edit_parameters(tmp_path: Path) -> None:
    cfg = _Cfg(outdir=tmp_path / "runs", dataset_provider="synthetic", edit_plan={})
    cfg.edit.parameters = _DictNoItems({"delta": 4})

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    extra = (
        patch(
            "invarlock.cli.run_config.resolve_provider_and_split",
            lambda *_a, **_k: (
                SimpleNamespace(windows=lambda **_kw: _provider_windows(1, 1)),
                "validation",
                False,
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: SimpleNamespace(
                windows=lambda **_kw: _provider_windows(1, 1)
            ),
        ),
    )
    with pytest.raises(typer.Exit) as excinfo:
        _run_with_common_patches(
            cfg=cfg, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
        )
    assert excinfo.value.exit_code == 2


def test_run_command_rejects_legacy_edit_kind(tmp_path: Path) -> None:
    cfg = _Cfg(outdir=tmp_path / "runs", dataset_provider="synthetic", edit_plan={})
    cfg.edit.kind = "quant"

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    extra = (
        patch(
            "invarlock.cli.run_config.resolve_provider_and_split",
            lambda *_a, **_k: (
                SimpleNamespace(windows=lambda **_kw: _provider_windows(1, 1)),
                "validation",
                False,
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: SimpleNamespace(
                windows=lambda **_kw: _provider_windows(1, 1)
            ),
        ),
    )
    with pytest.raises(typer.Exit) as excinfo:
        _run_with_common_patches(
            cfg=cfg, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
        )
    assert excinfo.value.exit_code == 2


def test_run_command_writes_telemetry_report(tmp_path: Path) -> None:
    def resolver(*_a, **_k):
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        report = _core_report(evaluation_windows={})
        report.metrics["timings"] = {"guards": 0.1}
        report.metrics["memory_mb_peak"] = 12.0
        return report, kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        edit_plan={},
    )

    extra = (
        patch("invarlock.cli.run_config.resolve_provider_and_split", resolver),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: SimpleNamespace(
                windows=lambda **_kw: _provider_windows(1, 1)
            ),
        ),
    )
    _run_with_common_patches(
        cfg=cfg,
        exec_stub=exec_stub,
        post_stub=post_stub,
        extra_patches=extra,
        run_kwargs={"telemetry": True, "timing": True},
    )

    telemetry_files = list((tmp_path / "runs").rglob("telemetry.json"))
    assert telemetry_files, "telemetry.json was not written"
    payload = json.loads(telemetry_files[0].read_text(encoding="utf-8"))
    assert payload["timings"]["guards"] == 0.1


def test_run_command_baseline_token_counts_provider_parity_export_and_classification(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "provenance": {
                    "provider_digest": {"dataset": "synthetic", "split": "validation"}
                },
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0, 1],
                        "input_ids": [[1, 2], [3, 4]],
                        "attention_masks": [[1, 1], [1, 1]],
                        "example_correct": [True, True],
                        "logloss": [1.0, 1.0],
                        "token_counts": [2, 2],
                    },
                    "final": {
                        "window_ids": [2, 3],
                        "input_ids": [[5, 6], [7, 8]],
                        "attention_masks": [[1, 1], [1, 1]],
                        "example_correct": [True, True],
                        "logloss": [1.0, 1.0],
                        "token_counts": [2, 2],
                    },
                },
            }
        )
    )

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

        def snapshot(self, model):  # noqa: ANN001
            return b"x"

        def restore(self, model, blob):  # noqa: ANN001,ARG002
            return None

        def save_pretrained(self, model, export_dir: Path):  # noqa: ANN001
            captured["save_pretrained_called"] = True
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "config.json").write_text("{}", encoding="utf-8")
            return True

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def exec_stub(**kwargs):  # noqa: ANN001
        captured["run_config"] = kwargs.get("run_config")
        captured["model_in_exec"] = kwargs.get("model")
        return _core_report(
            evaluation_windows={
                "preview": {
                    "input_ids": [[1, 2], [3, 4]],
                    "example_correct": [True, True],
                },
                "final": {
                    "input_ids": [[5, 6], [7, 8]],
                    "example_correct": [True, True],
                },
            }
        ), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        captured["report"] = kwargs.get("report")
        return {"json": str(tmp_path / "report.json")}

    def provider_digest(_report):  # noqa: ANN001
        return {"dataset": "synthetic", "split": "validation"}

    def enforce_parity(digest, base_digest, profile=None):  # noqa: ANN001,ARG001
        captured["base_digest"] = base_digest

    monkeypatch.setenv("INVARLOCK_EXPORT_MODEL", "1")
    monkeypatch.delenv("INVARLOCK_EXPORT_DIR", raising=False)
    monkeypatch.setenv("DEBUG_METRIC_DIFFS", "1")

    rec_console = Console(record=True)
    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        loss_type="classification",
        output={"model_dir": "exported_model"},
    )

    with ExitStack() as stack:
        for p in (
            patch("invarlock.cli.run_execution.console", rec_console),
            patch("invarlock.cli.run_config.prepare_config_for_run", lambda **k: cfg),
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile", _detect_profile
            ),
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda *_a, **_k: _tok(),
            ),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            ),
            patch(
                "invarlock.core.orchestration.execute._should_measure_metric_impact_impl",
                lambda *_a: (False, False, None),
            ),
            patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
            patch("invarlock.cli.run_execution.postprocess_and_summarize", post_stub),
            patch(
                "invarlock.cli.run_pairing.resolve_metric_and_provider",
                lambda *_a, **_k: ("ppl_causal", None, {}),
            ),
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                _pm_stub,
            ),
            patch("invarlock.core.registry.get_registry", lambda: Registry()),
            patch("invarlock.cli.run_pairing.compute_provider_digest", provider_digest),
            patch(
                "invarlock.core.run_policy.enforce_provider_parity",
                enforce_parity,
            ),
            patch(
                "invarlock.reporting.run_report_metrics_contract.format_debug_metric_diffs",
                lambda *_a, **_k: "diffs",
            ),
        ):
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    rc = captured["run_config"]
    assert rc is not None
    ctx = rc.context
    assert isinstance(ctx.get("baseline_eval_windows"), dict)
    assert ctx["baseline_eval_windows"]["final"]["token_counts"] == [2, 2]

    report = captured["report"]
    assert isinstance(report, dict)
    assert captured.get("base_digest") == {
        "dataset": "synthetic",
        "split": "validation",
    }
    assert captured.get("save_pretrained_called") is True
    assert report.get("artifacts", {}).get("checkpoint_path")
    assert (
        report.get("metrics", {}).get("classification", {}).get("counts_source")
        == "measured"
    )
    assert "DEBUG_METRIC_DIFFS" in rec_console.export_text()
