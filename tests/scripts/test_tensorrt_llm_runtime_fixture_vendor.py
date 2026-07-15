from __future__ import annotations

from pathlib import Path

from tests.scripts._tensorrt_llm_fixture_support import fixture


def test_docker_prefix_initializes_vendor_runtime_without_human_banner(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    command = fixture._docker_prefix(
        engine="docker",
        selector="device=1",
        worker=worker,
        image="candidate",
    )
    assert command[:6] == ["docker", "run", "--rm", "--gpus", "device=1", "--network"]
    assert "none" in command
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in command
    assert "--privileged" not in command
    assert command[command.index("--entrypoint") + 1] == "/bin/bash"
    assert all(value in command for value in fixture._boundary.VENDOR_CACHE_ENV_ARGS)
    assert command[-5:] == [
        "candidate",
        "-c",
        'exec "$@"',
        "--",
        "/opt/invarlock/bin/vendor-python",
    ]
