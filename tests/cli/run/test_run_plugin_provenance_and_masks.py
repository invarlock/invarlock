from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path, preview=4, final=4) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: structured
  plan: {{}}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview}
  final_n: {final}

guards:
  order: []

eval:
  spike_threshold: 2.0
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _common_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device=None: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: SimpleNamespace(name=name),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=set(),
                cert_lints=[],
                family="gpt",
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=1000),
                "tokhash123",
            ),
        ),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def _provider_simple():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]),
            SimpleNamespace(input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]),
        )
    )


@pytest.mark.parametrize("profile", ["ci", "release"])
def test_parity_mismatch_evalwindow_exit(tmp_path: Path, profile: str):
    from invarlock.eval.data import EvaluationWindow

    cfg = _cfg(tmp_path, 2, 2)

    class Provider:
        def windows(self, **kwargs):
            prev = EvaluationWindow(
                input_ids=[[1, 2]], attention_masks=[[1, 1]], indices=[0]
            )
            fin = EvaluationWindow(
                input_ids=[[3, 4], [5, 6]],
                attention_masks=[[1, 1], [1, 1]],
                indices=[0, 1],
            )
            return prev, fin

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile=profile,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_edit_override_invalid_raises(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.config.resolve_edit_kind", lambda name: name)
        )
        stack.enter_context(
            patch(
                "invarlock.cli.config.apply_edit_override",
                side_effect=ValueError("bad edit"),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={},
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                edit="quant",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_unknown_guards_skipped_and_known_kept(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    class Reg:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device=None: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            if name == "foo":
                raise KeyError("not found")
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    def runner_exec(**kwargs):
        captured["guards"] = kwargs.get("guards")
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(order=["foo", "bar"])
            self.dataset = SimpleNamespace(
                provider="synthetic",
                id="synthetic",
                split="validation",
                seq_len=8,
                stride=4,
                preview_n=1,
                final_n=1,
                seed=42,
            )
            self.eval = SimpleNamespace(
                spike_threshold=2.0, loss=SimpleNamespace(type="auto")
            )
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: Reg())
        )
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    assert [g.name for g in captured["guards"]] == ["bar"]


def test_dataset_meta_stratification_scorer_profile(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class Provider:
        stratification_stats = {"bins": [1, 2]}
        scorer_profile = {"score": "ok"}

        def windows(self, **kwargs):
            return (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )

    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context=getattr(k.get("config"), "context", {}),
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    d = captured["r"]["data"]
    assert d.get("stratification") == {"bins": [1, 2]}
    assert d.get("scorer_profile") == {"score": "ok"}


def test_metrics_window_plan_stats_map(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    ctx_wp = {
        "requested_preview": 1,
        "requested_final": 1,
        "actual_preview": 1,
        "actual_final": 1,
        "coverage_ok": True,
    }

    def runner_exec(**kwargs):
        ctx = {"dataset_meta": {}, "window_plan": ctx_wp}
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context=ctx,
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    stats = captured["r"]["metrics"].get("stats", {})
    assert (
        stats.get("requested_preview") == 1
        and stats.get("actual_final") == 1
        and stats.get("coverage_ok") is True
    )


def test_metrics_window_plan_capacity_map(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    ctx_wp = {
        "requested_preview": 1,
        "requested_final": 1,
        "actual_preview": 1,
        "actual_final": 1,
        "coverage_ok": True,
        "capacity": {"available_unique": 123},
    }

    def runner_exec(**kwargs):
        ctx = {"dataset_meta": {}, "window_plan": ctx_wp}
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context=ctx,
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    metrics = captured["r"].get("metrics", {})
    assert (
        isinstance(metrics.get("window_capacity"), dict)
        and metrics["window_capacity"].get("available_unique") == 123
    )


def test_persist_ref_masks_artifact(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    def runner_exec(**kwargs):
        edit = {
            "artifacts": {
                "mask_payload": {
                    "meta": {"note": "x"},
                    "keep_indices": [1, 2, 3],
                }
            }
        }
        return SimpleNamespace(
            edit=edit,
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    path = Path(captured["r"]["artifacts"]["masks_path"])
    assert path.exists() and path.name == "masks.json"


def test_overhead_bare_warning_present(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            if kwargs.get("guards") == []:
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                    status="error",
                )
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 2.0, "ppl_ratio": 2.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Validator passes but we expect bare warning to be captured
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
        ):
            stack.enter_context(
                patch(
                    target,
                    lambda *a, **k: SimpleNamespace(
                        passed=True,
                        messages=[],
                        warnings=[],
                        errors=[],
                        checks={},
                        metrics={"overhead_ratio": 0.0, "overhead_percent": 0.0},
                    ),
                )
            )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    gh = captured["r"].get("guard_overhead", {})
    assert (
        isinstance(gh, dict)
        and gh.get("evaluated") is True
        and gh.get("passed") is True
    )


def test_dedupe_min_floor_error_path(tmp_path: Path):
    # Provide many exact duplicates to trigger reduction below floor and exit
    cfg = _cfg(tmp_path, 8, 8)

    class Provider:
        def windows(self, **kwargs):
            n_prev = int(kwargs.get("preview_n"))
            n_fin = int(kwargs.get("final_n"))
            seq = [1, 1, 1, 1]
            prev_ids = [seq for _ in range(n_prev)]
            fin_ids = [seq for _ in range(n_fin)]
            return (
                SimpleNamespace(input_ids=prev_ids, attention_masks=[[1] * 4] * n_prev),
                SimpleNamespace(input_ids=fin_ids, attention_masks=[[1] * 4] * n_fin),
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={},
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_plugin_provenance_counts_present(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    class Reg:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device=None: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    def runner_exec(**kwargs):
        captured["plugins"] = kwargs.get("config").context.get("plugins", {})
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(order=["a", "b"])
            self.dataset = SimpleNamespace(
                provider="synthetic",
                id="synthetic",
                split="validation",
                seq_len=8,
                stride=4,
                preview_n=1,
                final_n=1,
                seed=42,
            )
            self.eval = SimpleNamespace(
                spike_threshold=2.0, loss=SimpleNamespace(type="auto")
            )
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: Reg())
        )
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    plugins = captured["plugins"]
    assert isinstance(plugins.get("guards"), list) and len(plugins.get("guards")) == 2


def test_baseline_attention_masks_inferred_and_labels_sanitized(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 0, 2]],
                        "labels": [[1]],
                    },
                    "final": {"window_ids": [1], "input_ids": [[3, 4, 5]]},
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Runner must provide evaluation_windows to satisfy release+baseline, but we will run in default profile
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_baseline_labels_longer_than_input_trimmed(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    # Labels longer than input_ids should be trimmed safely
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2]],
                        "labels": [[1, 2, 3, 4, 5]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[3, 4]],
                        "labels": [[6, 7, 8]],
                    },
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        # No exception implies trimming path executed safely
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_provider_attention_mask_tolist_tuple_path(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class AM:
        def __init__(self, n):
            self.n = n

        def tolist(self):
            return tuple([1] * self.n)

    class Provider:
        def windows(self, **kwargs):
            return (
                SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[AM(3)]),
                SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[AM(3)]),
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


@pytest.mark.parametrize(
    "key",
    [
        "window_pairing_reason",
        "window_pairing_preview",
        "window_pairing_final",
        "paired_windows",
        "paired_delta_summary",
    ],
)
def test_metrics_optional_pairing_fields_passthrough(tmp_path: Path, key: str):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    runner_metrics = {
        "ppl_preview": 1.0,
        "ppl_final": 1.0,
        "ppl_ratio": 1.0,
        key: {"ok": True} if key.endswith("summary") else 1,
    }
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics=runner_metrics,
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    assert key in captured["r"].get("metrics", {})


@pytest.mark.parametrize(
    "opt_key",
    [
        "algorithm_version",
        "implementation",
        "scope",
        "ranking",
        "grouping",
        "budgets",
        "seed",
        "mask_digest",
    ],
)
def test_edit_optional_fields_transfer(tmp_path: Path, opt_key: str):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    edit_payload = {opt_key: "X"}
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit=edit_payload,
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    assert opt_key in captured["r"]["edit"]


@pytest.mark.parametrize("indices_type", ["list", "tuple", "generator"])
def test_provider_indices_various_types(tmp_path: Path, indices_type: str):
    cfg = _cfg(tmp_path, 1, 1)

    def _indices():
        return (i for i in [0])

    class Provider:
        def windows(self, **kwargs):
            if indices_type == "list":
                idx = [0]
            elif indices_type == "tuple":
                idx = (0,)
            else:
                idx = _indices()
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=idx
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=idx
            )
            return prev, fin

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
