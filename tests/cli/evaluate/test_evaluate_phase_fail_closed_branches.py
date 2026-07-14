from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
import typer

from invarlock.cli import evaluate_phases


def test_provided_baseline_preserves_typer_exit_from_contract_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        evaluate_phases,
        "load_validated_baseline_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(typer.Exit(7)),
    )
    request = evaluate_phases.BaselineEvaluationRequest(
        baseline_report=str(tmp_path / "baseline.report.json"),
        profile_name="dev",
        tier_name="balanced",
        adapter="hf_causal",
        out=str(tmp_path / "runs"),
        device=None,
        allow_network=False,
        allow_host_execution=False,
        allow_third_party_plugins=False,
        allow_remote_code=False,
        allow_unverified_provenance=False,
        prefer_local_files_only=True,
        no_color=True,
        baseline_cfg={
            "model": {"id": "model"},
            "assurance": {"mode": "strict"},
            "dataset": {},
        },
        baseline_label="baseline",
        tmp_dir=tmp_path,
    )
    runtime = evaluate_phases.EvaluatePhaseRuntime(
        console=None,
        output_style=None,
        timings={},
        verbosity=1,
        progress=False,
        info_fn=lambda *_args, **_kwargs: None,
        debug_fn=lambda *_args, **_kwargs: None,
        phase_fn=lambda *_args, **_kwargs: None,
        fail_fn=lambda *_args, **_kwargs: pytest.fail(
            "the original CLI exit must not be translated"
        ),
        suppress_child_output_fn=lambda *_args, **_kwargs: nullcontext(None),
        load_yaml_fn=lambda *_args, **_kwargs: {},
        dump_yaml_fn=lambda *_args, **_kwargs: None,
        run_command_fn=lambda *_args, **_kwargs: None,
        json_load_fn=lambda *_args, **_kwargs: {},
    )

    with pytest.raises(typer.Exit) as exc_info:
        evaluate_phases.run_baseline_evaluation_phase(request, runtime)

    assert exc_info.value.exit_code == 7
