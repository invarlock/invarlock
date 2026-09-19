from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from scripts import runtime_qualification

ROOT = Path(__file__).resolve().parents[2]


def test_qualification_accepts_only_the_core_candidate_distribution() -> None:
    assert runtime_qualification._CANDIDATE_DISTRIBUTION_SOURCES == {
        "invarlock": ("src/invarlock", "invarlock")
    }


def test_qualification_source_inventory_is_core_only() -> None:
    assert runtime_qualification._is_execution_source("src/invarlock/cli/app.py")
    assert runtime_qualification._is_execution_source(
        "scripts/runtime_qualification.py"
    )
    assert not runtime_qualification._is_execution_source("legacy/runtime/provider.py")
    assert not runtime_qualification._is_execution_source("README.md")


def test_candidate_probe_requires_the_core_distribution() -> None:
    assert 'expected.get("invarlock")' in runtime_qualification._CANDIDATE_PROBE
    assert "candidate distribution discovery does not match manifest" in (
        runtime_qualification._CANDIDATE_PROBE
    )


def test_provider_runtime_images_bind_source_identity_and_read_only_execution() -> None:
    for filename in (
        "Dockerfile.gguf",
        "Dockerfile.hf-vision-text",
        "Dockerfile.tensorrt-llm",
    ):
        dockerfile = (ROOT / "runtime" / filename).read_text(encoding="utf-8")
        assert (
            'org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"'
            in dockerfile
        )
        assert (
            'dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"'
            in dockerfile
        )
        assert "--require-hashes" in dockerfile
        assert "--no-deps" in dockerfile


def test_runtime_build_and_qualification_targets_use_authenticated_inputs() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    qualification = (ROOT / "scripts/authenticated_runtime_build.py").read_text(
        encoding="utf-8"
    )
    assert "runtime-image" in makefile
    assert "runtime-qualification-canary" in makefile
    assert "--source-bundle-sha256" in makefile
    assert "source_bundle_sha256" in qualification
    assert "INVARLOCK_ALLOW_NETWORK" in (
        ROOT / "scripts/runtime_qualification.py"
    ).read_text(encoding="utf-8")


def test_tensorrt_image_launcher_runs_with_the_fixed_python_interpreter(
    tmp_path: Path,
) -> None:
    dockerfile = (ROOT / "runtime/Dockerfile.tensorrt-llm").read_text(encoding="utf-8")
    command = shlex.split(
        next(
            line.removesuffix("\\")
            for line in dockerfile.splitlines()
            if " > /opt/invarlock/bin/tensorrt-llm-runner" in line
        )
    )
    offset = command.index("printf")
    assert command[offset + 1] == "%s\\n"
    lines = command[offset + 2 : command.index(">")]
    assert lines[0] == "#!/opt/invarlock/bin/vendor-python"
    launcher = tmp_path / "tensorrt-llm-runner"
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(launcher), "--invarlock-score-v1"],
        input=b"{}",
        capture_output=True,
        timeout=30,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 70
    assert result.stdout == b""
    assert result.stderr == (
        b"TensorRT-LLM runner failed closed: runner request fields are not closed\n"
    )
    for worker_name in ("__worker__", "__mp_main__"):
        imported = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; path,name=sys.argv[1:]; "
                "sys.argv=[path,'--invarlock-score-batch-v1']; "
                "assert callable(runpy.run_path(path,run_name=name)['main'])",
                str(launcher),
                worker_name,
            ],
            input=b"",
            capture_output=True,
            timeout=30,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert imported.returncode == 0
        assert imported.stdout == b""
        assert imported.stderr == b""
