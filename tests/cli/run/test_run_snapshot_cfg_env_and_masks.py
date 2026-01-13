from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command


def _base_cfg(tmp_path: Path, preview=1, final=1) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
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
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
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
    )


def _provider():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def test_snapshot_auto_ram_fraction_env(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace as NS

    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return [
                        ("p", NS(element_size=lambda: 1, nelement=lambda: 900_000_000))
                    ]

                def named_buffers(self):
                    return []

            return M()

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    def vm():
        return SimpleNamespace(available=200 * 1024 * 1024)  # 200MB

    def du(path):
        return SimpleNamespace(total=0, used=0, free=4 * 1024 * 1024 * 1024)  # 4GB

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION", "0.2")
        stack.enter_context(
            patch("invarlock.cli.commands.run.psutil.virtual_memory", vm)
        )
        stack.enter_context(patch("invarlock.cli.commands.run.shutil.disk_usage", du))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: adapter,
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert adapter.rest_chunked >= 1


def test_snapshot_cfg_threshold_and_tempdir(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return [
                        (
                            "p",
                            SimpleNamespace(
                                element_size=lambda: 1, nelement=lambda: 50_000_000
                            ),
                        )
                    ]  # 50MB

                def named_buffers(self):
                    return []

            return M()

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    def load_cfg(p):
        class Cfg:
            def __init__(self):
                self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
                self.edit = SimpleNamespace(name="structured", plan={})
                self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
                self.guards = SimpleNamespace(order=[])
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
                self.context = {
                    "snapshot": {
                        "threshold_mb": 10.0,
                        "disk_free_margin_ratio": 1.1,
                        "temp_dir": str(tmp_path),
                    }
                }

            def model_dump(self):
                return {}

        return Cfg()

    def vm():
        return SimpleNamespace(available=0)

    def du(path):
        return SimpleNamespace(total=0, used=0, free=100 * 1024 * 1024)  # 100MB

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.cli.config.load_config", load_cfg))
        stack.enter_context(
            patch("invarlock.cli.commands.run.psutil.virtual_memory", vm)
        )
        stack.enter_context(patch("invarlock.cli.commands.run.shutil.disk_usage", du))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: adapter,
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert adapter.rest_chunked >= 1


def test_snapshot_bytes_supported_but_ram_low_prefers_chunked(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    # Large est_mb ~ 900MB
                    return [
                        (
                            "p",
                            SimpleNamespace(
                                element_size=lambda: 1, nelement=lambda: 900_000_000
                            ),
                        )
                    ]

                def named_buffers(self):
                    return []

            return M()

        def snapshot(self, model):
            return b"blob"

        def restore(self, model, blob):
            pass

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    def vm():
        # Low available RAM (100MB) → est_mb >= max(64, avail*0.8) triggers chunked in bytes-supported branch
        return SimpleNamespace(available=100 * 1024 * 1024)

    def du(path):
        # Plenty of disk space
        return SimpleNamespace(total=0, used=0, free=2 * 1024 * 1024 * 1024)

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: adapter,
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.cli.commands.run.psutil.virtual_memory", vm)
        )
        stack.enter_context(patch("invarlock.cli.commands.run.shutil.disk_usage", du))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert adapter.rest_chunked >= 1
