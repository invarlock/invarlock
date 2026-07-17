from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import runtime_qualification

ROOT = Path(__file__).resolve().parents[2]
SHA256 = "sha256:" + "a" * 64


def _qualification_values(tmp_path: Path) -> dict[str, object]:
    return {
        "PYTHON": sys.executable,
        "QUALIFICATION_DRIVER_PYTHON": tmp_path / "qualification driver python",
        "REQUEST": tmp_path / "request with spaces.yaml",
        "SIGNING_KEY": tmp_path / "evidence signer.pem",
        "IMAGE": SHA256,
        "IMAGE_DIGEST": SHA256,
        "EVIDENCE": tmp_path / "qualified evidence",
        "TRUST_PROFILE": tmp_path / "trusted inputs.json",
        "RECEIPT": tmp_path / "verification receipt.json",
        "CANARY_EVIDENCE": tmp_path / "canary evidence",
        "CANARY_RECEIPT": tmp_path / "canary receipt.json",
        "CANARY_TRUST_PROFILE": tmp_path / "canary trusted inputs.json",
        "SUMMARY": tmp_path / "qualification summary.json",
        "SOURCE_COMMIT": "b" * 40,
        "SOURCE_BUNDLE": tmp_path / "source bundle.tar.gz",
        "SOURCE_BUNDLE_SHA256": "sha256:" + "c" * 64,
        "CANDIDATE_WHEEL_MANIFEST": tmp_path / "candidate wheels.json",
        "QUALIFICATION_DEVICE": "cuda:0",
        "QUALIFICATION_CPUS": "8",
        "QUALIFICATION_MEMORY_MIB": "8192",
        "QUALIFICATION_USER": "65532:65532",
    }


def _dry_run(target: str, tmp_path: Path) -> str:
    values = _qualification_values(tmp_path)
    command = ["make", "-n", target]
    command.extend(f"{key}={value}" for key, value in values.items())
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_core_readiness_uses_the_repository_qualification_driver(
    tmp_path: Path,
) -> None:
    output = _dry_run("runtime-qualification-readiness", tmp_path)

    assert "scripts/runtime_qualification.py readiness" in output
    assert (
        f'"{tmp_path / "qualification driver python"}" -I -S '
        "scripts/runtime_qualification.py readiness"
    ) in output
    assert f'--python "{sys.executable}"' in output
    assert '--runtime-device "cuda:0"' in output
    assert '--runtime-cpus "8"' in output
    assert '--runtime-memory-mib "8192"' in output
    assert '--runtime-user "65532:65532"' in output
    assert f'--canary-evidence "{tmp_path / "canary evidence"}"' in output
    assert f'--canary-receipt "{tmp_path / "canary receipt.json"}"' in output
    assert "qualification_precheck.py" not in output
    assert "-m invarlock evaluate" not in output


def test_core_evidence_target_delegates_the_complete_transaction(
    tmp_path: Path,
) -> None:
    output = _dry_run("runtime-qualification-evidence", tmp_path)

    assert "scripts/runtime_qualification.py run" in output
    assert f'--summary "{tmp_path / "qualification summary.json"}"' in output
    assert f'--request "{tmp_path / "request with spaces.yaml"}"' in output
    assert '--source-commit "' + "b" * 40 + '"' in output
    assert f'--source-bundle "{tmp_path / "source bundle.tar.gz"}"' in output
    assert '--source-bundle-sha256 "sha256:' + "c" * 64 + '"' in output
    assert (
        f'--candidate-wheel-manifest "{tmp_path / "candidate wheels.json"}"' in output
    )
    assert (
        f'--canary-trust-profile "{tmp_path / "canary trusted inputs.json"}"' in output
    )
    assert "qualification_precheck.py" not in output
    assert "-m invarlock verify" not in output


def test_core_canary_target_bootstraps_without_a_prior_canary(
    tmp_path: Path,
) -> None:
    output = _dry_run("runtime-qualification-canary", tmp_path)

    assert "scripts/runtime_qualification.py canary" in output
    assert f'--summary "{tmp_path / "qualification summary.json"}"' in output
    assert "--canary-evidence" not in output
    assert "--canary-receipt" not in output


def test_core_operator_journey_preserves_one_explicit_runtime_contract(
    tmp_path: Path,
) -> None:
    stages = {
        "runtime-qualification-canary": "canary",
        "runtime-qualification-readiness": "readiness",
        "runtime-qualification-evidence": "run",
    }

    for target, mode in stages.items():
        output = _dry_run(target, tmp_path)
        assert f"scripts/runtime_qualification.py {mode}" in output
        parser = runtime_qualification._parser()  # noqa: SLF001
        mode_action = next(
            action for action in parser._actions if action.dest == "mode"
        )
        mode_parser = mode_action.choices[mode]
        required_options = {
            option
            for action in mode_parser._actions
            if action.required
            for option in action.option_strings
        }
        for option in required_options:
            assert option in output
        assert '--runtime-device "cuda:0"' in output
        assert '--runtime-cpus "8"' in output
        assert '--runtime-memory-mib "8192"' in output
        assert '--runtime-user "65532:65532"' in output
        if mode == "canary":
            assert "--canary-evidence" not in output
        else:
            assert f'--canary-evidence "{tmp_path / "canary evidence"}"' in output


def test_core_operator_journey_requires_an_explicit_device(tmp_path: Path) -> None:
    values = _qualification_values(tmp_path)
    values.pop("QUALIFICATION_DEVICE")
    command = ["make", "-n", "runtime-qualification-canary"]
    command.extend(f"{key}={value}" for key, value in values.items())

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "QUALIFICATION_DEVICE is required" in completed.stderr
