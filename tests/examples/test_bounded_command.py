from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from examples.integrations import bounded_command
from examples.integrations.bounded_command import run_bounded_command


def test_bounded_command_captures_output_and_environment(tmp_path: Path) -> None:
    completed = run_bounded_command(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['INVARLOCK_TEST_VALUE']); "
            "print('diagnostic', file=__import__('sys').stderr)",
        ],
        cwd=tmp_path,
        environment={"INVARLOCK_TEST_VALUE": "bounded"},
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == "bounded\n"
    assert completed.stderr == "diagnostic\n"


def test_bounded_command_streams_uncaptured_output_and_preserves_check_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    streamed = run_bounded_command(
        [
            sys.executable,
            "-c",
            "print('stdout'); print('stderr', file=__import__('sys').stderr)",
        ],
    )
    assert streamed.stdout is None
    assert streamed.stderr is None
    captured = capsys.readouterr()
    assert "stdout" in captured.out
    assert "stderr" in captured.err

    run_bounded_command([sys.executable, "-c", "print('only stdout')"])
    assert "only stdout" in capsys.readouterr().out
    run_bounded_command(
        [sys.executable, "-c", "print('only stderr', file=__import__('sys').stderr)"]
    )
    assert "only stderr" in capsys.readouterr().err

    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_bounded_command(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)",
            ],
            check=True,
            capture_output=True,
        )
    assert failure.value.returncode == 7
    assert failure.value.output == "out\n"
    assert failure.value.stderr == "err\n"


def test_bounded_command_rejects_unbounded_or_unstartable_commands(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_bounded_command([sys.executable, "-c", "pass"], timeout_seconds=0)
    with pytest.raises(ValueError, match="output limits"):
        run_bounded_command([sys.executable, "-c", "pass"], stdout_limit=0)
    with pytest.raises(RuntimeError, match="could not start"):
        run_bounded_command([str(tmp_path / "missing-command")])


def test_bounded_command_enforces_output_and_time_limits() -> None:
    with pytest.raises(RuntimeError, match="stdout limit exceeded"):
        run_bounded_command(
            [sys.executable, "-c", "print('x' * 4096)"],
            stdout_limit=32,
        )
    with pytest.raises(RuntimeError, match="timed out"):
        run_bounded_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=1,
        )


def test_bounded_command_supports_file_stdin_and_bounded_stdout_transfer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    destination = tmp_path / "output.txt"
    source.write_text("from file\n", encoding="utf-8")

    completed = run_bounded_command(
        [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"],
        stdin_path=source,
        stdout_path=destination,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert destination.read_text(encoding="utf-8") == "from file\n"

    failed_destination = tmp_path / "failed-output.txt"
    with pytest.raises(RuntimeError, match="stdout limit exceeded"):
        run_bounded_command(
            [sys.executable, "-c", "print('x' * 4096)"],
            stdout_path=failed_destination,
            stdout_limit=32,
        )
    assert not failed_destination.exists()


def test_bounded_command_termination_handles_exited_and_hung_processes() -> None:
    class ExitedProcess:
        def poll(self) -> int:
            return 0

    bounded_command._terminate(ExitedProcess())  # type: ignore[arg-type]  # noqa: SLF001

    class HungProcess:
        def __init__(self) -> None:
            self.killed = False
            self.terminated = False
            self.wait_count = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: int) -> int:
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired("command", timeout)
            return 0

    process = HungProcess()
    bounded_command._terminate(process)  # type: ignore[arg-type]  # noqa: SLF001
    assert process.terminated and process.killed and process.wait_count == 2
