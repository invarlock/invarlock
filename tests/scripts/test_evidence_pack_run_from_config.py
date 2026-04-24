from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root / "scripts" / "evidence_packs" / "python" / "run_from_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evidence_pack_run_from_config",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_pack_run_from_config"] = module
    spec.loader.exec_module(module)
    return module


def test_delegate_if_needed_builds_container_launch_plan(monkeypatch) -> None:
    module = _load_script_module()
    seen: dict[str, object] = {}

    monkeypatch.setattr(module, "apply_runtime_allowances", lambda **kwargs: None)
    monkeypatch.setattr(module, "running_inside_container", lambda: False)
    monkeypatch.setattr(module, "host_execution_allowed", lambda: False)
    monkeypatch.setattr(
        module,
        "build_current_process_container_launch_plan",
        lambda argv: seen.setdefault("plan", tuple(argv)) or tuple(argv),
    )

    def _delegate(script_path, plan):
        seen["script_path"] = script_path
        seen["delegated_plan"] = plan
        return 7

    monkeypatch.setattr(module, "delegate_python_script_to_container", _delegate)

    args = module._parse_args(["--config", "demo.yaml"])
    result = module._delegate_if_needed(
        args,
        ["--config", "demo.yaml", "--device", "cuda"],
    )

    assert result == 7
    assert seen["script_path"] == Path(module.__file__)
    assert seen["plan"] == ("--config", "demo.yaml", "--device", "cuda")
    assert seen["delegated_plan"] == ("--config", "demo.yaml", "--device", "cuda")


def test_delegate_if_needed_honors_remote_code_env(monkeypatch) -> None:
    module = _load_script_module()
    seen: dict[str, object] = {}

    def _apply_runtime_allowances(**kwargs):
        seen["policy"] = kwargs.get("policy")
        return None

    monkeypatch.setattr(module, "apply_runtime_allowances", _apply_runtime_allowances)
    monkeypatch.setattr(module, "running_inside_container", lambda: False)
    monkeypatch.setattr(module, "host_execution_allowed", lambda: False)
    monkeypatch.setattr(
        module,
        "build_current_process_container_launch_plan",
        lambda argv: tuple(argv),
    )
    monkeypatch.setattr(
        module,
        "delegate_python_script_to_container",
        lambda script_path, plan: 0,
    )
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")

    args = module._parse_args(["--config", "demo.yaml"])
    result = module._delegate_if_needed(args, ["--config", "demo.yaml"])

    assert result == 0
    policy = seen["policy"]
    assert policy is not None
    assert policy.allow_remote_code is True


def test_delegate_if_needed_defaults_device_auto_for_delegation(monkeypatch) -> None:
    module = _load_script_module()
    seen: dict[str, object] = {}

    monkeypatch.setattr(module, "apply_runtime_allowances", lambda **kwargs: None)
    monkeypatch.setattr(module, "running_inside_container", lambda: False)
    monkeypatch.setattr(module, "host_execution_allowed", lambda: False)
    monkeypatch.setattr(
        module,
        "build_current_process_container_launch_plan",
        lambda argv: seen.setdefault("plan", tuple(argv)) or tuple(argv),
    )
    monkeypatch.setattr(
        module,
        "delegate_python_script_to_container",
        lambda script_path, plan: 0,
    )

    args = module._parse_args(["--config", "demo.yaml"])
    result = module._delegate_if_needed(args, ["--config", "demo.yaml"])

    assert result == 0
    assert seen["plan"] == ("--config", "demo.yaml", "--device", "auto")
