from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from invarlock.cli.commands.evaluate import evaluate_command
from invarlock.core.assurance_contract import CANONICAL_GUARD_CHAIN
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports


def _stub_run(out_dir: Path) -> Path:
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_causal", "device": "cpu"},
        "edit": {"name": "noop"},
        "metrics": {"primary_metric": {"preview": 1.0, "final": 1.0}},
        "data": {"preview_n": 1, "final_n": 1},
    }
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def _strict_stub_report(*, model_id: str, edit_name: str, context: dict) -> dict:
    windows = {
        "preview": {
            "window_ids": [1, 2],
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "logloss": [0.6931471805599453, 0.6931471805599453],
            "token_counts": [10, 10],
        },
        "final": {
            "window_ids": [3, 4],
            "input_ids": [[7, 8, 9], [10, 11, 12]],
            "logloss": [0.6931471805599453, 0.6931471805599453],
            "token_counts": [10, 10],
        },
    }
    invariant_metrics = {
        "checks_performed": 1,
        "violations_found": 0,
        "fatal_violations": 0,
        "warning_violations": 0,
    }
    measurement_contract = {"kind": "activation_edge_risk", "version": "test-v1"}
    return {
        "meta": {
            "model_id": model_id,
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 7, "numpy": 7, "torch": 7},
            "auto": {"tier": "balanced"},
            "tokenizer_hash": "strict-tokenizer",
        },
        "context": dict(context),
        "edit": {"name": edit_name, "deltas": {"params_changed": 0}},
        "data": {
            "dataset": "strict-local-jsonl",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
            "tokenizer_hash": "strict-tokenizer",
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 2.0,
                "final": 2.0,
                "ratio_vs_baseline": 1.0,
                "ci": [0.0, 0.0],
                "display_ci": [1.0, 1.0],
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": windows,
        "provenance": {"provider_digest": {"ids_sha256": "strict-provider-ids"}},
        "artifacts": {},
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "decision": "allow",
                "metrics": invariant_metrics,
            },
            {
                "name": "spectral",
                "passed": True,
                "decision": "allow",
                "metrics": {
                    "stable": True,
                    "caps_applied": 0,
                    "modules_checked": 1,
                    "families": {"linear": {"violations": 0}},
                    "measurement_contract": {"kind": "spectral", "version": "test-v1"},
                },
            },
            {
                "name": "rmt",
                "passed": True,
                "decision": "allow",
                "metrics": {
                    "stable": True,
                    "edge_risk_by_family_base": {"linear": 1.0},
                    "edge_risk_by_family": {"linear": 1.0},
                    "measurement_contract": measurement_contract,
                },
            },
            {
                "name": "variance",
                "passed": True,
                "decision": "allow",
                "metrics": {"ve_enabled": False, "gain": 0.0},
            },
            {
                "name": "invariants",
                "passed": True,
                "decision": "allow",
                "metrics": invariant_metrics,
            },
        ],
    }


def _write_strict_stub_run(out_dir: Path, config_path: str, *, call_index: int) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    context = cfg.get("context") if isinstance(cfg, dict) else {}
    if not isinstance(context, dict):
        context = {}
    ts_dir = out_dir / f"20250101_00000{call_index}"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = _strict_stub_report(
        model_id=str(cfg.get("model", {}).get("id", "stub"))
        if isinstance(cfg, dict)
        else "stub",
        edit_name=str(cfg.get("edit", {}).get("name", "noop"))
        if isinstance(cfg, dict)
        else "noop",
        context=context,
    )
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_evaluate_command_smoke_for_bridge(monkeypatch, tmp_path) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    calls = {"runs": 0, "reports": 0}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"] += 1
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        calls["reports"] += 1

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert calls["runs"] == 2
    assert calls["reports"] == 1


def test_evaluate_command_strict_path_generates_verifiable_pending_report(
    monkeypatch,
    tmp_path,
) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.evaluate_output as evaluate_output_mod
    from invarlock.cli.commands import evaluate as eval_mod

    calls = {"runs": 0}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"] += 1
        return str(
            _write_strict_stub_run(
                Path(kwargs["out"]),
                str(kwargs["config"]),
                call_index=calls["runs"],
            )
        )

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(eval_mod, "maybe_delegate_model_command", lambda: None)
    monkeypatch.setattr(
        evaluate_output_mod,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:test",
    )
    monkeypatch.setattr(
        evaluate_output_mod,
        "resolve_runtime_image_digest",
        lambda: "sha256:" + "1" * 64,
    )

    report_dir = tmp_path / "reports"
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        tier="balanced",
        out=str(tmp_path / "runs"),
        report_out=str(report_dir),
        timing=False,
        progress=False,
    )

    report_path = report_dir / "evaluation.report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert calls["runs"] == 2
    assert payload["assurance"]["mode"] == "strict"
    assert payload["assurance"]["verdict"] == "pending_verifier"
    assert payload["assurance"]["report_local_verdict"] == "pass"
    assert payload["assurance"]["verified_assurance_verdict"] == "pending"
    assert payload["assurance"]["guard_chain_observed"] == list(CANONICAL_GUARD_CHAIN)
    assert payload["assurance"]["runtime_provenance_verification_status"] == "pending"
    assert payload["assurance"]["runtime_provenance_declared"] == "container"
    assert payload["report_build"] == {
        "synthesized_fields": [],
        "repaired_fields": [],
        "fallback_fields": [],
    }

    result = run_verify_reports(
        [report_path],
        profile="ci",
        assurance_mode="strict",
    )

    assert result.outcome == VerifyOutcome.OK


def test_evaluate_command_reuses_baseline_report_for_bridge(monkeypatch, tmp_path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    baseline_dir = tmp_path / "baseline_report_dir"
    baseline_dir.mkdir()
    baseline_report = baseline_dir / "report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {
                    "model_id": str(src),
                    "adapter": "hf_causal",
                    "device": "cpu",
                },
                "context": {
                    "profile": "ci",
                    "auto": {"tier": "balanced"},
                    "assurance": {"mode": "off"},
                },
                "edit": {"name": "noop"},
                "data": {
                    "provider": "wikitext2",
                    "split": "validation",
                    "seq_len": 512,
                    "stride": 512,
                    "preview_n": 64,
                    "final_n": 64,
                    "seed": 43,
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [1], "input_ids": [[1, 2]]},
                    "final": {"window_ids": [2], "input_ids": [[3, 4]]},
                },
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                        "ratio_vs_baseline": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calls = {"runs": [], "reports": []}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"].append(kwargs)
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**kwargs):  # noqa: ANN001
        calls["reports"].append(kwargs)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_report=str(baseline_report),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert len(calls["runs"]) == 1
    assert Path(calls["runs"][0]["baseline"]).resolve() == baseline_report.resolve()
    assert calls["reports"] and calls["reports"][0]["baseline"] == str(
        baseline_report.resolve()
    )


def test_evaluate_command_local_mode_prefers_local_files_only(monkeypatch, tmp_path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    captured_runs: list[dict[str, object]] = []

    def fake_run(**kwargs):  # noqa: ANN001
        captured_runs.append(kwargs)
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        execution_mode="host",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert len(captured_runs) == 2
    for call in captured_runs:
        assert call["prefer_local_files_only"] is True


def test_evaluate_command_passes_host_mode_execution_mode_to_runs(
    monkeypatch, tmp_path
) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    captured_runs: list[dict[str, object]] = []

    def fake_run(**kwargs):  # noqa: ANN001
        captured_runs.append(kwargs)
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as eval_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(eval_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="dev",
        execution_mode="host",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert len(captured_runs) == 2
    for call in captured_runs:
        assert call["allow_unverified_provenance"] is True


def test_evaluate_command_resets_runtime_security_on_success(
    monkeypatch, tmp_path
) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    import invarlock.cli.commands.evaluate as eval_mod
    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.security_helpers as security_helpers

    state = {"active": False, "configured": None, "enter": 0, "exit": 0}

    @contextmanager
    def fake_configure(**kwargs):
        state["configured"] = kwargs
        state["active"] = True
        state["enter"] += 1
        try:
            yield
        finally:
            state["active"] = False
            state["exit"] += 1

    def fake_run(**kwargs):
        out = Path(kwargs["out"])
        return str(_stub_run(out))

    monkeypatch.setattr(security_helpers, "configure_runtime_security", fake_configure)
    monkeypatch.setattr(eval_mod, "maybe_delegate_model_command", lambda: None)
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(
        eval_mod, "generate_reports", lambda **_kwargs: None, raising=False
    )

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        allow_network=True,
        assurance="off",
    )

    assert state["configured"]["allow_network"] is True
    assert state["active"] is False
    assert state["enter"] == 1
    assert state["exit"] == 1


def test_evaluate_command_resets_runtime_security_on_raise(
    monkeypatch, tmp_path
) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    import invarlock.cli.commands.evaluate as eval_mod
    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.security_helpers as security_helpers

    state = {"active": False, "configured": None, "enter": 0, "exit": 0}

    @contextmanager
    def fake_configure(**kwargs):
        state["configured"] = kwargs
        state["active"] = True
        state["enter"] += 1
        try:
            yield
        finally:
            state["active"] = False
            state["exit"] += 1

    def failing_run(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(security_helpers, "configure_runtime_security", fake_configure)
    monkeypatch.setattr(eval_mod, "maybe_delegate_model_command", lambda: None)
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)
    monkeypatch.setattr(
        eval_mod, "generate_reports", lambda **_kwargs: None, raising=False
    )

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="auto",
            subject_adapter="auto",
            profile="ci",
            out=str(tmp_path / "runs"),
            report_out=str(tmp_path / "reports"),
            timing=False,
            progress=False,
            allow_network=True,
            assurance="off",
        )

    assert state["configured"]["allow_network"] is True
    assert state["active"] is False
    assert state["enter"] == 1
    assert state["exit"] == 1


def test_evaluate_command_passes_concrete_run_defaults(monkeypatch, tmp_path) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    captured_runs: list[dict[str, object]] = []

    def fake_run(**kwargs):  # noqa: ANN001
        captured_runs.append(kwargs)
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert len(captured_runs) == 2
    for call in captured_runs:
        assert call["until_pass"] is False
        assert call["max_attempts"] == 1
        assert call["timeout"] is None
        assert call["allow_network"] is False
        assert call["allow_host_execution"] is False
        assert call["allow_third_party_plugins"] is False
        assert call["allow_remote_code"] is False
