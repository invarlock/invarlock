from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as public_cli
from tests.cli.run._internal_cli import internal_run_app


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "\n".join(json.dumps({"text": t}) for t in texts) + "\n", encoding="utf-8"
    )


def _cfg(tmp_path: Path, data_file: Path) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_causal
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
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                describe=lambda _model: {"n_layer": 1, "total_params": 1024},
            )

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
            # Provide finite paired losses so PM computation and baseline normalization
            # remain valid under fail-closed baseline checks.
            evaluation_windows={
                "preview": {
                    "window_ids": [0, 1],
                    "logloss": [2.20, 2.10],
                    "token_counts": [16, 16],
                },
                "final": {
                    "window_ids": [2, 3],
                    "logloss": [2.05, 2.15],
                    "token_counts": [16, 16],
                },
            },
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )

    # Model profile and tokenizer stub
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
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
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(encode=_enc, pad_token_id=0, vocab_size=256),
            "tokhash123",
        ),
    )


def test_evaluate_byod_local_quick(tmp_path: Path, monkeypatch) -> None:
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
        internal_run_app,
        ["run", "-c", cfg, "--profile", "dev", "--out", str(tmp_path / "run_base")],
    )
    assert r1.exit_code == 0, r1.stdout
    r2 = CliRunner().invoke(
        internal_run_app,
        ["run", "-c", cfg, "--profile", "dev", "--out", str(tmp_path / "run_subj")],
    )
    assert r2.exit_code == 0, r2.stdout

    # Generate evaluation report from runs
    # Each run creates a timestamped subdirectory; pick the only subdir
    def _pick_run_dir(root: Path) -> Path:
        subs = [p for p in root.iterdir() if p.is_dir()]
        assert subs, f"no run subdir under {root}"
        # choose the most recently modified
        subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return subs[0]

    rep_base = _pick_run_dir(tmp_path / "run_base")
    rep_subj = _pick_run_dir(tmp_path / "run_subj")
    from tests.cli._support_runtime_policy import bind_runtime_policy

    for run_dir in (rep_base, rep_subj):
        run_report_path = run_dir / "report.json"
        run_report = json.loads(run_report_path.read_text("utf-8"))
        bind_runtime_policy(run_report, profile="dev")
        run_report_path.write_text(json.dumps(run_report), encoding="utf-8")
    rcert = CliRunner().invoke(
        public_cli,
        [
            "report",
            "generate",
            "--run",
            str(rep_subj),
            "--format",
            "report",
            "--baseline-run-report",
            str(rep_base),
            "--output",
            str(tmp_path / "reports"),
        ],
    )
    assert rcert.exit_code == 0, rcert.stdout

    # Assert artifacts exist and provider digest recorded
    report_dir = tmp_path / "reports"
    assert (report_dir / "evaluation.report.json").exists()
    assert (report_dir / "evaluation_report.md").exists()
    assert (report_dir / "manifest.json").exists()
    report = json.loads((report_dir / "evaluation.report.json").read_text("utf-8"))
    evaluation_windows = report.get("evaluation_windows")
    assert isinstance(evaluation_windows, dict)
    final_windows = evaluation_windows.get("final")
    assert isinstance(final_windows, dict)
    assert final_windows.get("window_ids") == [2, 3]
    assert final_windows.get("logloss") == [2.05, 2.15]
    assert final_windows.get("token_counts") == [16, 16]
    prov = report.get("provenance") or {}
    assert isinstance(prov.get("provider_digest"), dict) and prov["provider_digest"]

    verify_result = CliRunner().invoke(
        public_cli,
        [
            "verify",
            "--profile",
            "ci",
            "--json",
            str(report_dir / "evaluation.report.json"),
        ],
    )
    verify_payload = json.loads(verify_result.stdout)
    assert verify_result.exit_code == 1
    assert verify_payload["results"][0]["recompute"] == {
        "family": "ppl",
        "performed": True,
        "ok": True,
        "reason": None,
    }
