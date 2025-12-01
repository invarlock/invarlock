from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as cli


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "\n".join(json.dumps({"text": t}) for t in texts) + "\n", encoding="utf-8"
    )


def _cfg(tmp_path: Path, data_file: Path) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_gpt2
  id: gpt2
  device: auto
edit:
  name: noop
  plan: {{}}

dataset:
  provider:
    kind: local_jsonl
  file: {data_file.as_posix()}
  split: validation
  seq_len: 16
  stride: 8
  preview_n: 2
  final_n: 2

guards:
  order: []

eval:
  metric: {{ kind: ppl_causal }}
  loss: {{ type: auto }}

output:
  dir: runs
""",
        encoding="utf-8",
    )
    return str(p)


def _stub_runtime(monkeypatch, tmp_path: Path):
    # Device
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )

    # Minimal registry that avoids loading real models/edits/guards
    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())

    # Core runner that returns empty guard results and injects minimal context
    def _exec(**kwargs):
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0, "layers_modified": 0}},
            metrics={"window_overlap_fraction": 0.0, "window_match_fraction": 1.0},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )

    # Model profile and tokenizer stub
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
            default_provider="synthetic",
        ),
    )

    # Tokenizer with encode method
    def _enc(text: str, truncation=True, max_length=16):
        # Derive simple token ids from characters to avoid duplicates
        ids = [((ord(c) % 13) + 1) for c in text][:max_length]
        # pad if needed
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        return ids

    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(encode=_enc, pad_token_id=0, vocab_size=256),
            "tokhash123",
        ),
    )


def test_certify_byod_local_quick(tmp_path: Path, monkeypatch) -> None:
    # Prepare a small JSONL with a few text samples
    data_file = tmp_path / "byod.jsonl"
    _write_jsonl(
        data_file,
        ["hello world", "bring your own data", "invarlock local jsonl", "test sample"],
    )

    # Prepare config and stub env
    cfg = _cfg(tmp_path, data_file)
    _stub_runtime(monkeypatch, tmp_path)

    # Produce baseline and subject runs
    r1 = CliRunner().invoke(
        cli, ["run", "-c", cfg, "--profile", "dev", "--out", str(tmp_path / "run_base")]
    )
    assert r1.exit_code == 0, r1.stdout
    r2 = CliRunner().invoke(
        cli, ["run", "-c", cfg, "--profile", "dev", "--out", str(tmp_path / "run_subj")]
    )
    assert r2.exit_code == 0, r2.stdout

    # Generate certificate from runs
    # Each run creates a timestamped subdirectory; pick the only subdir
    def _pick_run_dir(root: Path) -> Path:
        subs = [p for p in root.iterdir() if p.is_dir()]
        assert subs, f"no run subdir under {root}"
        # choose the most recently modified
        subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return subs[0]

    rep_base = _pick_run_dir(tmp_path / "run_base")
    rep_subj = _pick_run_dir(tmp_path / "run_subj")
    rcert = CliRunner().invoke(
        cli,
        [
            "report",
            "--run",
            str(rep_subj),
            "--format",
            "cert",
            "--baseline",
            str(rep_base),
            "--output",
            str(tmp_path / "cert"),
        ],
    )
    assert rcert.exit_code == 0, rcert.stdout

    # Assert artifacts exist and provider digest recorded
    cert_dir = tmp_path / "cert"
    assert (cert_dir / "evaluation.cert.json").exists()
    assert (cert_dir / "evaluation_certificate.md").exists()
    assert (cert_dir / "manifest.json").exists()
    cert = json.loads((cert_dir / "evaluation.cert.json").read_text("utf-8"))
    prov = cert.get("provenance") or {}
    assert isinstance(prov.get("provider_digest"), dict) and prov["provider_digest"]
