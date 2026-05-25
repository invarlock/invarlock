from __future__ import annotations

from pathlib import Path

import invarlock.cli.commands.run as run_mod
import invarlock.cli.internal_config_run as internal_config_run
import invarlock.cli.runtime_launch_plan as runtime_launch_plan
from invarlock.cli.config_execution import ConfigExecutionRequest
from invarlock.runtime_security import ContainerLaunchPlan


def _execution_field_values(request: ConfigExecutionRequest) -> dict[str, object]:
    policy_fields = set(ConfigExecutionRequest.POLICY_FIELDS)
    return {
        name: getattr(request, name)
        for name in ConfigExecutionRequest.field_names()
        if name not in policy_fields
    }


def test_run_command_request_delegated_argv_internal_runner_round_trip(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _run_request(request: ConfigExecutionRequest, **kwargs: object) -> Path:
        captured["request"] = request
        captured.update(kwargs)
        return Path("reports/roundtrip.report.json")

    monkeypatch.setattr(run_mod, "run_request", _run_request, raising=True)

    out = run_mod.run_command(
        config="configs/demo.yaml",
        device="cpu",
        profile="ci",
        out="runs/demo",
        edit="quant_rtn",
        edit_label="nightly",
        tier="balanced",
        metric_kind="ppl_causal",
        probes=4,
        until_pass=True,
        max_attempts=5,
        timeout=120,
        baseline="baseline.json",
        no_cleanup=True,
        style="audit",
        progress=True,
        timing=True,
        telemetry=True,
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unverified_provenance=True,
        prefer_local_files_only=True,
        no_color=True,
    )

    assert out == Path("reports/roundtrip.report.json")
    request = captured["request"]
    assert isinstance(request, ConfigExecutionRequest)
    assert captured["command_name"] == "run"
    assert request.runtime_policy_kwargs() == {
        "allow_network": True,
        "allow_host_execution": True,
        "allow_third_party_plugins": True,
        "allow_remote_code": True,
        "allow_unverified_provenance": True,
    }

    def _normalize(argv: list[str], *, cwd: Path) -> ContainerLaunchPlan:
        return ContainerLaunchPlan(
            argv=tuple(argv),
            argv_mounts=(),
            needs_cwd_host_mirror=False,
            gpu_passthrough=False,
        )

    monkeypatch.setattr(
        runtime_launch_plan,
        "normalize_delegated_argv",
        _normalize,
        raising=True,
    )

    plan = runtime_launch_plan.build_request_container_launch_plan("run", request)
    assert "--allow-network" not in plan.argv
    assert "--allow-host-execution" not in plan.argv
    assert "--allow-third-party-plugins" not in plan.argv
    assert "--allow-remote-code" not in plan.argv
    assert "--allow-unverified-provenance" not in plan.argv

    parser = internal_config_run._build_parser()
    parsed = parser.parse_args(list(plan.argv))
    parsed_request = ConfigExecutionRequest.from_argparse(parsed)

    assert _execution_field_values(parsed_request) == _execution_field_values(request)
    assert parsed.invoked_command == "run"

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        internal_config_run,
        "run_request",
        lambda request, **kwargs: seen.update({"request": request, **kwargs}),
        raising=True,
    )

    assert internal_config_run.main(list(plan.argv)) == 0
    assert seen == {
        "request": parsed_request,
        "command_name": "run",
        "delegate": False,
    }


def test_config_execution_request_rejects_unknown_kwargs() -> None:
    try:
        ConfigExecutionRequest.from_kwargs(config="configs/demo.yaml", unknown=True)
    except TypeError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown request fields should fail closed")
