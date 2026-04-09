from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root / "scripts" / "proof_packs" / "python" / "run_from_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "proof_pack_run_from_config",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["proof_pack_run_from_config"] = module
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
