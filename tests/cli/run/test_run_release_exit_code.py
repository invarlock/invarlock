import textwrap
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as cli


def _cfg(tmp_path: Path) -> str:
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        textwrap.dedent(
            """
            model:
              adapter: hf_gpt2
              id: gpt2
              device: auto
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

            eval:
              metric: { kind: ppl_causal }
              loss: { type: auto }

            output:
              dir: runs_cfg
            """
        )
    )
    return str(yml)


def _stub_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    # Device OK
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )

    # Registry and runner
    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())

    # Provider
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            ),
            estimate_capacity=lambda **kw: {
                "available_unique": 100,
                "available_nonoverlap": 100,
                "total_tokens": 1000,
                "dedupe_rate": 0.0,
            },
        ),
    )

    # Model profile + tokenizer
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: (_ for _ in ()).throw(
            __import__(
                "invarlock.cli.commands.run", fromlist=["InvarlockError"]
            ).InvarlockError("E003", "MASK-PARITY-MISMATCH", {})
        ),
        raising=True,
    )


def test_release_profile_hard_abort_exit_3(tmp_path: Path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "release"])
    assert r.exit_code == 3
    assert "[INVARLOCK:E003]" in r.stdout


def test_out_flag_precedence_over_config(tmp_path: Path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    out_dir = tmp_path / "flag_out"

    # Ensure we don't trigger the parity error path in this case
    def _profile():
        tok = SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10)
        return SimpleNamespace(
            default_loss="causal",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
            make_tokenizer=lambda: (tok, "tokhash123"),
        )

    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile", lambda *a, **k: _profile()
    )
    r = CliRunner().invoke(
        cli, ["run", "-c", cfg, "--profile", "dev", "--out", str(out_dir)]
    )
    s = r.stdout
    # Output dir precedence: anywhere in output paths the base dir should be the flag value
    assert "flag_out" in s
    assert "runs_cfg" not in s

    # (cleanup-on-early-error is left for a follow-up PR that moves cleanup print to a finally block)
