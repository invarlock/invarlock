import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.cli.commands.run import run_command

_SNS = SimpleNamespace


def _basic_yaml(tmp_path: Path, extra: str = "") -> Path:
    cfg = tmp_path / "config.yaml"
    base = """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
"""
    cfg.write_text(base + ("\n" + extra if extra else ""))
    return cfg


def _provider_min():
    class P:
        def windows(self, **kwargs):
            prev = SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]])
            fin = SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])
            return prev, fin

        def estimate_capacity(self, **kwargs):
            return {
                "available_unique": 100,
                "available_nonoverlap": 100,
                "total_tokens": 10000,
                "dedupe_rate": 0.05,
                "candidate_unique": 80,
                "candidate_limit": 160,
            }

    return P()


def _common_min(monkeypatch, tmp_path):
    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
                snapshot_chunked=lambda _m=None: str(tmp_path / "snapdir"),
                restore_chunked=lambda _m, _d=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={
                "plan_digest": "abcd",
                "deltas": {
                    "params_changed": 0,
                    "heads_pruned": 0,
                    "neurons_pruned": 0,
                    "layers_modified": 0,
                },
            },
            metrics={
                "ppl_preview": 10.0,
                "ppl_final": 10.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
                "loss_type": "ce",
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda model_id=None, adapter=None: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    # default tokenizer to a simple object with eos/pad
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )

    # guard overhead validator OK by default
    class _OverheadOK:
        def __init__(self):
            self.passed = True
            self.messages = []
            self.warnings = []
            self.errors = []
            self.checks = {}
            self.metrics = {"overhead_ratio": 1.0, "overhead_percent": 0.0}

    monkeypatch.setattr(
        "invarlock.cli.commands.run.validate_guard_overhead",
        lambda *args, **kwargs: _OverheadOK(),
    )


def test_tokenizer_digest_with_get_vocab_str_and_nonstr(monkeypatch, tmp_path):
    _common_min(monkeypatch, tmp_path)

    class Tok:
        def __init__(self):
            self.eos_token = "</s>"
            self.pad_token = "</s>"
            self.vocab_size = 10

        def get_vocab(self):
            # includes a non-string key to exercise else branch
            return {"a": 1, 2: 3}

    # ensure tokenizer_hash is None so _tokenizer_digest is used
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (Tok(), None),
    )

    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


# ---- Merged from test_run_branch_gap_fill_more.py (renamed helpers) ----
def _cfg_gfm(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "config.yaml"
    base = """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 2
  final_n: 2

guards:
  order: []
  invariants:
    profile_checks: some_existing

eval:
  loss:
    type: auto

output:
  dir: runs
"""
    p.write_text(base + ("\n" + extra if extra else ""))
    return p


def _provider_min_gfm():
    class P:
        def windows(self, **kwargs):
            prev = _SNS(input_ids=[[1, 2]], attention_masks=[[1, 1]])
            fin = _SNS(input_ids=[[3, 4]], attention_masks=[[1, 1]])
            return prev, fin

    return P()


def _base_patches_gfm(monkeypatch):
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min_gfm()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *_: (
            _SNS(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )

    class _OverheadOK:
        def __init__(self):
            self.passed = True
            self.metrics = {"overhead_ratio": 1.0, "overhead_percent": 0.0}
            self.messages = []
            self.warnings = []
            self.errors = []
            self.checks = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.run.validate_guard_overhead",
        lambda *a, **k: _OverheadOK(),
    )


def _std_core_report_gfm(ctx=None):
    if ctx is None:
        ctx = {"dataset_meta": {}}
    return _SNS(
        edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
        metrics={
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
            "loss_type": "ce",
        },
        guards={},
        context=ctx,
        evaluation_windows={},
        status="success",
    )


def test_gfm_invariants_profile_checks_existing_string_and_model_invariants_merge(
    monkeypatch, tmp_path
):
    _base_patches_gfm(monkeypatch)

    class Registry:
        def get_adapter(self, name):
            return _SNS(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return _SNS(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: Registry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: _SNS(execute=lambda **k: _std_core_report_gfm()),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: _SNS(
            default_loss="ce",
            invariants=["dim_check"],
            cert_lints=[],
            module_selectors={},
            family="gpt2",
        ),
    )
    cfg = _cfg_gfm(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


# ---- Selected snapshot mode edge scenarios (from edges) ----


def test_snapshot_mode_bytes(monkeypatch, tmp_path):
    cfg = _cfg_gfm(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return _SNS(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
                snapshot_chunked=lambda _m=None: str(tmp_path / "snapdir"),
                restore_chunked=lambda _m, _d=None: None,
            )

        def get_edit(self, name):
            return _SNS(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: _SNS(execute=lambda **k: _std_core_report_gfm()),
    )
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "bytes")
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_mode_chunked(monkeypatch, tmp_path):
    cfg = _cfg_gfm(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return _SNS(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
                snapshot_chunked=lambda _m=None: str(tmp_path / "snapdir"),
                restore_chunked=lambda _m, _d=None: None,
            )

        def get_edit(self, name):
            return _SNS(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: _SNS(execute=lambda **k: _std_core_report_gfm()),
    )
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_mode_reload_fallback(monkeypatch, tmp_path):
    cfg = _cfg_gfm(tmp_path)

    class MinimalRegistry:
        def get_adapter(self, name):
            return _SNS(name=name, load_model=lambda *args, **kwargs: object())

        def get_edit(self, name):
            return _SNS(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    monkeypatch.setattr(
        "invarlock.core.registry.get_registry", lambda: MinimalRegistry()
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: _SNS(execute=lambda **k: _std_core_report_gfm()),
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_mode_auto_prefers_bytes(monkeypatch, tmp_path):
    cfg = _cfg_gfm(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return _SNS(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
                snapshot_chunked=lambda _m=None: str(tmp_path / "snapdir"),
                restore_chunked=lambda _m, _d=None: None,
            )

        def get_edit(self, name):
            return _SNS(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: _SNS(execute=lambda **k: _std_core_report_gfm()),
    )
    monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)

    class VM:
        available = 2 * 1024 * 1024 * 1024  # 2GB

    monkeypatch.setattr("invarlock.cli.commands.run.psutil.virtual_memory", lambda: VM)

    class DU:
        free = 50 * 1024 * 1024 * 1024  # 50GB

    monkeypatch.setattr("invarlock.cli.commands.run.shutil.disk_usage", lambda path: DU)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_tokenizer_digest_no_get_vocab_vocab_list(monkeypatch, tmp_path):
    _common_min(monkeypatch, tmp_path)

    class Tok:
        def __init__(self):
            self.eos_token = "</s>"
            self.pad_token = "</s>"
            self.vocab_size = 10
            # list without items() attribute
            self.vocab = [("a", 1), (3, 5)]

    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (Tok(), None),
    )

    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_tokenizer_digest_get_vocab_raises(monkeypatch, tmp_path):
    _common_min(monkeypatch, tmp_path)

    class Tok:
        def __init__(self):
            self.eos_token = "</s>"
            self.pad_token = "</s>"
            self.vocab_size = 10

        def get_vocab(self):  # noqa: D401
            raise RuntimeError("nope")

    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (Tok(), None),
    )

    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_baseline_sanitize_attention_masks_and_labels(monkeypatch, tmp_path):
    # Missing attention_masks should be rebuilt; labels should be padded/trimmed
    _common_min(monkeypatch, tmp_path)
    baseline = tmp_path / "baseline.json"
    schedule = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "input_ids": [[1, 2, 3]],
                # no attention_masks provided
                "labels": [[-100, 1]],  # shorter than input_ids
            },
            "final": {
                "window_ids": [2],
                "input_ids": [[4, 5]],
                # labels longer than input_ids
                "labels": [[-100, -100, -100, -100]],
            },
        }
    }
    baseline.write_text(json.dumps(schedule))
    cfg = _basic_yaml(tmp_path)
    run_command(
        config=str(cfg),
        device="cpu",
        out=str(tmp_path / "runs"),
        profile=None,
        baseline=str(baseline),
    )


def test_baseline_sanitize_non_list_input_ids_ignored(monkeypatch, tmp_path):
    # If input_ids is not a list, _sanitize should return None and fall back
    _common_min(monkeypatch, tmp_path)
    baseline = tmp_path / "baseline.json"
    schedule = {
        "evaluation_windows": {
            "preview": {"window_ids": [1], "input_ids": 42},  # not a list
            "final": {
                "window_ids": [2],
                "input_ids": [[7, 8]],
                "attention_masks": [[1, 1]],
            },
        }
    }
    baseline.write_text(json.dumps(schedule))
    cfg = _basic_yaml(tmp_path)
    run_command(
        config=str(cfg),
        device="cpu",
        out=str(tmp_path / "runs"),
        profile=None,
        baseline=str(baseline),
    )


def test_snapshot_cfg_bytes_fallback_to_chunked(monkeypatch, tmp_path):
    class Registry:
        def get_adapter(self, name):
            # Only chunked supported
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot_chunked=lambda _m=None: str(tmp_path / "snap"),
                restore_chunked=lambda *_: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, name, t):
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={},
            metrics={
                "ppl_preview": 1.0,
                "ppl_final": 1.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: Registry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tok",
        ),
    )

    cfg = _basic_yaml(
        tmp_path,
        extra="""
context:
  snapshot:
    mode: bytes
        """,
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_cfg_chunked_fallback_to_bytes(monkeypatch, tmp_path):
    class Registry:
        def get_adapter(self, name):
            # Only bytes supported
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda *_: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, name, t):
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={},
            metrics={
                "ppl_preview": 1.0,
                "ppl_final": 1.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: Registry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tok",
        ),
    )

    cfg = _basic_yaml(
        tmp_path,
        extra="""
context:
  snapshot:
    mode: chunked
        """,
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_env_bytes_fallback_to_chunked(monkeypatch, tmp_path):
    class Registry:
        def get_adapter(self, name):
            # Only chunked supported
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot_chunked=lambda _m=None: str(tmp_path / "snap"),
                restore_chunked=lambda *_: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, name, t):
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: Registry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tok",
        ),
    )
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "bytes")

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={},
            metrics={
                "ppl_preview": 1.0,
                "ppl_final": 1.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_snapshot_env_chunked_fallback_to_bytes(monkeypatch, tmp_path):
    class Registry:
        def get_adapter(self, name):
            # Only bytes supported
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda *_: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, name, t):
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: Registry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tok",
        ),
    )
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={},
            metrics={
                "ppl_preview": 1.0,
                "ppl_final": 1.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)


def test_stratification_count_mismatch_final_only(monkeypatch, tmp_path):
    # cause final_n mismatch by forcing dataset_meta to override report['data'] values
    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError

        def get_plugin_metadata(self, name, t):
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _exec(**kwargs):
        # override final_n only
        ctx = {"dataset_meta": {"final_n": 999}}
        return SimpleNamespace(
            edit={},
            metrics={
                "ppl_preview": 1.0,
                "ppl_final": 1.0,
                "ppl_ratio": 1.0,
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
            },
            guards={},
            context=ctx,
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider", lambda *a, **k: _provider_min()
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tok",
        ),
    )
    cfg = _basic_yaml(tmp_path)
    import click

    with pytest.raises(click.exceptions.Exit):
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None
        )


def test_snapshot_auto_both_memory_disk_queries_fail(monkeypatch, tmp_path):
    # Ensure auto path runs with exceptions in RAM and disk usage helpers
    _common_min(monkeypatch, tmp_path)

    # Remove explicit mode and env
    monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)

    # Cause psutil.virtual_memory and shutil.disk_usage to raise
    def _raise_vm():
        raise RuntimeError("vm fail")

    def _raise_du(_):
        raise RuntimeError("du fail")

    monkeypatch.setattr("invarlock.cli.commands.run.psutil.virtual_memory", _raise_vm)
    monkeypatch.setattr("invarlock.cli.commands.run.shutil.disk_usage", _raise_du)

    cfg = _basic_yaml(tmp_path)
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)
