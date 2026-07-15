from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from invarlock.cli.commands.evaluate import evaluate_command
from invarlock.core.assurance_contract import CANONICAL_GUARD_CHAIN
from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports
from tests.cli._support_effective_config import preserve_effective_config
from tests.cli.run._support_run_evaluate_command_bridge import _stub_run
from tests.cli.verify._support_runtime_provenance import (
    _write_matching_strict_policy_pack,
    bind_runtime_policy_receipt,
)
from tests.core._support_assurance_contract import (
    _plugin_metadata,
    bind_noop_variance_evidence,
    bind_raw_guard_evidence,
)


def _strict_stub_report(*, model_id: str, edit_name: str, context: dict) -> dict:
    assert edit_name == "noop"
    window_count = 180
    windows = {
        "preview": {
            "window_ids": list(range(window_count)),
            "input_ids": [
                [index, index + 1, index + 2] for index in range(window_count)
            ],
            "logloss": [0.6931471805599453] * window_count,
            "token_counts": [10] * window_count,
        },
        "final": {
            "window_ids": list(range(window_count, window_count * 2)),
            "input_ids": [
                [index, index + 1, index + 2]
                for index in range(window_count, window_count * 2)
            ],
            "logloss": [0.6931471805599453] * window_count,
            "token_counts": [10] * window_count,
        },
    }
    invariant_metrics = {
        "checks_performed": 1,
        "violations_found": 0,
        "fatal_violations": 0,
        "warning_violations": 0,
    }
    measurement_contract = {"kind": "activation_edge_risk", "version": "test-v1"}
    strict_context = dict(context)
    strict_context["guard_chain_observed"] = list(CANONICAL_GUARD_CHAIN)
    report = {
        "meta": {
            "model_id": model_id,
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 7, "numpy": 7, "torch": 7},
            "auto": {"tier": "balanced"},
            "tokenizer_hash": "strict-tokenizer",
            "plugins": {
                "adapter": _plugin_metadata("adapters", "hf_causal"),
                "edit": _plugin_metadata("edits", "noop"),
                "guards": [
                    _plugin_metadata("guards", name) for name in CANONICAL_GUARD_CHAIN
                ],
            },
        },
        "context": strict_context,
        "edit": {"name": edit_name, "deltas": {"params_changed": 0}},
        "data": {
            "dataset": "strict-local-jsonl",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": window_count,
            "final_n": window_count,
            "dataset_hash": "strict-dataset",
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
            "bootstrap": {
                "enabled": True,
                "replicates": 1200,
                "alpha": 0.05,
                "method": "bca_paired_delta_log",
                "seed": 7,
                "preview_final_delta_basis": "independent_disjoint_slices",
                "preview_final_delta_method": ("independent_percentile_delta_log"),
                "preview_final_delta_seed": 104,
                "coverage": {
                    "tier": "balanced",
                    "preview": {
                        "used": window_count,
                        "required": window_count,
                        "ok": True,
                    },
                    "final": {
                        "used": window_count,
                        "required": window_count,
                        "ok": True,
                    },
                    "replicates": {
                        "used": 1200,
                        "required": 1200,
                        "ok": True,
                    },
                },
            },
            "preview_final_slice_delta_summary": {
                "mean": 0.0,
                "ci": [0.0, 0.0],
                "basis": "independent_disjoint_slices",
                "paired": False,
                "ci_method": "independent_percentile_delta_log",
                "ci_reason": None,
                "preview_windows": window_count,
                "final_windows": window_count,
                "degenerate": True,
                "degenerate_reason": "constant_bootstrap_distribution",
            },
        },
        "evaluation_windows": windows,
        "provenance": {
            "provider_digest": {
                "ids_sha256": "strict-provider-ids",
                "tokenizer_sha256": "strict-tokenizer",
            }
        },
        "artifacts": {},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "guards": [
            {
                "name": "invariants",
                "stage": "pre",
                "supported": True,
                "passed": True,
                "decision": "allow",
                "violations": [],
                "metrics": invariant_metrics,
            },
            {
                "name": "spectral",
                "supported": True,
                "passed": True,
                "decision": "allow",
                "violations": [],
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
                "supported": True,
                "passed": True,
                "decision": "allow",
                "violations": [],
                "metrics": {
                    "stable": True,
                    "edge_risk_by_family_base": dict.fromkeys(
                        ("attn", "embed", "ffn", "other"), 1.0
                    ),
                    "edge_risk_by_family": dict.fromkeys(
                        ("attn", "embed", "ffn", "other"), 1.0
                    ),
                    "epsilon_by_family": dict.fromkeys(
                        ("attn", "embed", "ffn", "other"), 0.01
                    ),
                    "measurement_contract": measurement_contract,
                },
            },
            {
                "name": "variance",
                "supported": True,
                "passed": True,
                "decision": "allow",
                "violations": [],
                "metrics": {
                    "ve_enabled": False,
                    "monitor_only": False,
                    "gain": 0.0,
                    "predictive_gate": {
                        "evaluated": True,
                        "passed": True,
                        "reason": "no_adjustment_required",
                    },
                    "calibration": {
                        "status": "no_scaling_required",
                        "coverage": 8,
                        "min_coverage": 6,
                    },
                },
            },
            {
                "name": "invariants",
                "stage": "post",
                "supported": True,
                "passed": True,
                "decision": "allow",
                "violations": [],
                "metrics": invariant_metrics,
            },
        ],
        "guard_metric_impact": {
            "evaluated": True,
            "metric_kind": "ppl_causal",
            "bare_value": 2.0,
            "guarded_value": 2.0,
            "degradation_limit": 0.01,
            "passed": True,
        },
    }
    report["variance"] = {
        "enabled": False,
        "monitor_only": False,
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "no_adjustment_required",
        },
        "calibration": {
            "status": "no_scaling_required",
            "coverage": 8,
            "min_coverage": 6,
        },
    }
    bound = bind_noop_variance_evidence(bind_raw_guard_evidence(report))
    resolved_policy = resolve_tier_policies("balanced", profile="ci")
    for guard in bound["guards"]:
        guard_name = guard.get("name")
        guard_policy = guard.get("policy")
        resolved_guard_policy = resolved_policy.get(guard_name)
        if not isinstance(guard_policy, dict) or not isinstance(
            resolved_guard_policy, dict
        ):
            continue
        if guard_name == "spectral":
            guard_policy["family_caps"] = copy.deepcopy(
                resolved_guard_policy["family_caps"]
            )
        elif guard_name == "rmt":
            guard_policy["epsilon_by_family"] = copy.deepcopy(
                resolved_guard_policy["epsilon_by_family"]
            )
    bound["resolved_policy"] = resolved_policy
    return bind_runtime_policy_receipt(bound)


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
    report["meta"]["run_id"] = f"strict-stub-run-{call_index}"
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, dict):
        for field in ("model_identity",):
            if field in model_cfg:
                report["meta"][field] = model_cfg[field]
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
        return str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs))

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


def test_evaluate_command_strict_path_rejects_incomplete_stub_evidence(
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

    calls: dict[str, object] = {"runs": 0, "run_reports": []}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"] = int(calls["runs"]) + 1
        preserve_effective_config(kwargs)
        report = _write_strict_stub_run(
            Path(kwargs["out"]),
            str(kwargs["config"]),
            call_index=int(calls["runs"]),
        )
        run_reports = calls["run_reports"]
        assert isinstance(run_reports, list)
        run_reports.append(report)
        return str(report)

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
    assert payload["assurance"]["verdict"] == "fail"
    assert payload["assurance"]["report_local_verdict"] == "fail"
    assert payload["assurance"]["blocking_reasons"]
    assert payload["assurance"]["guard_chain_observed"] == list(CANONICAL_GUARD_CHAIN)
    assert payload["assurance"]["runtime_provenance_verification_status"] == "pending"
    assert payload["assurance"]["runtime_provenance_declared"] == "container"
    assert payload["report_build"] == {
        "synthesized_fields": [],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    policy_path = _write_matching_strict_policy_pack(report_path, payload)

    result = run_verify_reports(
        [report_path],
        baseline=calls["run_reports"][0],
        policy_pack=policy_path,
        profile="ci",
        assurance_mode="strict",
        expected_runtime_image_digest="sha256:" + "1" * 64,
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL


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
    baseline_digest = checkpoint_tree_sha256(src)
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {
                    "model_id": str(src),
                    "adapter": "hf_causal",
                    "device": "cpu",
                    "model_identity": {
                        "kind": "local_checkpoint_tree",
                        "sha256": baseline_digest,
                    },
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
                    "preview_n": 240,
                    "final_n": 240,
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
        return str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs))

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
        return str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs))

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
        return str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs))

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
        return str(_stub_run(out, run_kwargs=kwargs))

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
