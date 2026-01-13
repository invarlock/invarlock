import json
import subprocess
import sys
from pathlib import Path


def _write_cov_xml(path: Path, class_specs: list[tuple[str, float, float]]) -> None:
    """Write a minimal coverage.xml with given (filename, branch_rate, line_rate)."""
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<coverage>",
        "  <packages>",
        "    <package name='pkg'>",
        "      <classes>",
    ]
    for filename, br, lr in class_specs:
        lines.append(
            f"        <class name='X' filename='{filename}' branch-rate='{br}' line-rate='{lr}'/>"
        )
    lines += [
        "      </classes>",
        "    </package>",
        "  </packages>",
        "</coverage>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_checker(
    xml_path: Path, json_path: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path("scripts") / "check_coverage_thresholds.py"),
        "--coverage",
        str(xml_path),
        "--json",
        str(json_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_two_tier_policy_enforced(tmp_path: Path) -> None:
    # Create a synthetic report containing a mix of core and non-core files
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("src/invarlock/core/runner.py", 0.89, 0.90),  # core → FAIL (needs 0.90)
            (
                "src/invarlock/cli/commands/run.py",
                0.91,
                0.90,
            ),  # core meets floor → PASS
            (
                "src/invarlock/reporting/certificate.py",
                0.91,
                0.90,
            ),  # override set to 0.90 → PASS
            (
                "src/invarlock/cli/commands/plugins.py",
                0.81,
                0.90,
            ),  # non-core → not enforced (absent from THRESHOLDS)
            (
                "src/invarlock/eval/primary_metric.py",
                0.91,
                0.90,
            ),  # explicit critical file → PASS
            (
                "src/invarlock/eval/metrics.py",
                0.91,
                0.90,
            ),  # explicit critical file → PASS
            ("src/invarlock/guards/spectral.py", 0.89, 0.90),  # core (guards) → FAIL
        ],
    )

    proc = _run_checker(xml, json_out)

    # Expect non-zero due to the two intentional core failures
    assert proc.returncode != 0
    err = proc.stderr
    assert "src/invarlock/core/runner.py" in err
    assert "src/invarlock/guards/spectral.py" in err
    # Sanity: overridden core file shouldn't appear as a failure
    assert "src/invarlock/cli/commands/run.py" not in err


def test_overrides_take_precedence(tmp_path: Path) -> None:
    # Explicit overrides should win over a stricter core-floor flag.
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(xml, [("src/invarlock/reporting/certificate.py", 0.94, 0.90)])
    proc = _run_checker(xml, json_out, extra_args=["--core-floor", "0.95"])

    # Should pass with explicit 90% override applied
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(json_out.read_text())
    assert payload["status"] == "ok"
    files = {f["path"]: f for f in payload["files"]}
    assert (
        abs(files["src/invarlock/reporting/certificate.py"]["threshold"] - 0.90) < 1e-9
    )
