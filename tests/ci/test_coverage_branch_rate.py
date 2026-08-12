from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "checks" / "check_coverage_branch_rate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_coverage_branch_rate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(path: Path, *, covered: int, valid: int) -> None:
    percentage = 100 * covered / valid if valid else 100.0
    path.write_text(
        '<?xml version="1.0" ?>\n'
        '<coverage branches-covered="'
        f"{covered}"
        '" branches-valid="'
        f"{valid}"
        '"><packages><package><classes><class name="module" filename="module.py"><lines>'
        '<line number="1" hits="1" branch="true" condition-coverage="'
        f"{percentage:.2f}% ({covered}/{valid})"
        '"/><line number="2" hits="1"/></lines></class>'
        "</classes></package></packages></coverage>\n",
        encoding="utf-8",
    )


def test_aggregate_branch_rate_passes_at_the_exact_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    _report(first, covered=9, valid=10)
    _report(second, covered=9, valid=10)

    assert module.main([str(first), str(second), "--minimum", "90"]) == 0

    output = capsys.readouterr()
    assert "aggregate branch coverage: 90.00% (18/20)" in output.out
    assert output.err == ""


def test_aggregate_branch_rate_fails_below_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=89, valid=100)

    assert module.main([str(report), "--minimum", "90"]) == 1

    output = capsys.readouterr()
    assert output.out.startswith(f"{report}: 89.00% branch coverage")
    assert "ERROR: aggregate branch coverage: 89.00%" in output.err


@pytest.mark.parametrize(
    "contents",
    [
        "not xml",
        "<coverage />",
        '<coverage branches-covered="x" branches-valid="1"/>',
        '<coverage branches-covered="2" branches-valid="1"/>',
        '<coverage branches-covered="-1" branches-valid="1"/>',
        '<report branches-covered="1" branches-valid="1"/>',
        (
            '<coverage branches-covered="1" branches-valid="1">'
            '<line branch="true" condition-coverage="not coverage"/>'
            "</coverage>"
        ),
    ],
)
def test_invalid_reports_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    report.write_text(contents, encoding="utf-8")

    assert module.main([str(report)]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_zero_branch_report_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=0, valid=0)

    assert module.main([str(report), "--minimum", "100"]) == 2
    assert "invalid branch counts" in capsys.readouterr().err


def test_internal_branch_totals_must_match_root_totals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=9, valid=10)
    contents = report.read_text(encoding="utf-8").replace(
        'condition-coverage="90.00% (9/10)"',
        'condition-coverage="80% (8/10)"',
    )
    report.write_text(contents, encoding="utf-8")

    assert module.main([str(report)]) == 2
    assert "branch totals do not match report contents" in capsys.readouterr().err


def test_class_branch_rate_fails_even_when_aggregate_rate_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<coverage branches-covered="18" branches-valid="20">'
        "<packages><package><classes>"
        '<class name="weak" filename="weak.py"><lines>'
        '<line number="1" hits="1" branch="true" '
        'condition-coverage="80% (8/10)"/>'
        "</lines></class>"
        '<class name="strong" filename="strong.py"><lines>'
        '<line number="1" hits="1" branch="true" '
        'condition-coverage="100% (10/10)"/>'
        "</lines></class>"
        "</classes></package></packages></coverage>",
        encoding="utf-8",
    )

    assert module.main([str(report), "--minimum", "90"]) == 1
    output = capsys.readouterr()
    assert "weak.py: 80.00% branch coverage (8/10)" in output.err
    assert "coverage.xml: 90.00% branch coverage (18/20)" in output.out


def test_report_scoped_class_exemption_preserves_aggregate_enforcement(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<coverage branches-covered="18" branches-valid="20">'
        "<packages><package><classes>"
        '<class name="weak" filename="weak.py"><lines>'
        '<line number="1" hits="1" branch="true" '
        'condition-coverage="80% (8/10)"/>'
        "</lines></class>"
        '<class name="strong" filename="strong.py"><lines>'
        '<line number="1" hits="1" branch="true" '
        'condition-coverage="100% (10/10)"/>'
        "</lines></class>"
        "</classes></package></packages></coverage>",
        encoding="utf-8",
    )
    manifest = tmp_path / "exemptions.txt"
    source_root = tmp_path / "examples"
    source_root.mkdir()
    source_root.joinpath("unreported.py").write_text("pass\n", encoding="utf-8")
    manifest.write_text(
        "examples/weak.py\nexamples/unreported.py\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert (
        module.main(
            [
                str(report),
                "--minimum",
                "90",
                "--class-exemptions",
                str(report),
                "examples",
                str(manifest),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "aggregate branch coverage: 90.00% (18/20)" in output.out
    assert output.err == ""


def test_class_exemption_does_not_remove_counts_from_aggregate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=8, valid=10)
    manifest = tmp_path / "exemptions.txt"
    manifest.write_text("examples/module.py\n", encoding="utf-8")

    assert (
        module.main(
            [
                str(report),
                "--minimum",
                "90",
                "--class-exemptions",
                str(report),
                "examples",
                str(manifest),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "module.py: 80.00%" not in output.err
    assert "ERROR: aggregate branch coverage: 80.00%" in output.err


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("examples/missing.py\n", "does not match a class"),
        ("outside/module.py\n", "outside source root"),
        (
            "examples/module.py\nexamples/module.py\n",
            "duplicate class exemption",
        ),
        ("../module.py\n", "invalid class exemption"),
    ],
)
def test_invalid_class_exemption_manifests_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
    message: str,
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=9, valid=10)
    manifest = tmp_path / "exemptions.txt"
    manifest.write_text(contents, encoding="utf-8")

    assert (
        module.main(
            [
                str(report),
                "--class-exemptions",
                str(report),
                "examples",
                str(manifest),
            ]
        )
        == 2
    )
    assert message in capsys.readouterr().err


def test_class_exemptions_must_bind_once_to_an_input_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    other = tmp_path / "other.xml"
    _report(report, covered=9, valid=10)
    manifest = tmp_path / "exemptions.txt"
    manifest.write_text("examples/module.py\n", encoding="utf-8")

    assert (
        module.main(
            [
                str(report),
                "--class-exemptions",
                str(other),
                "examples",
                str(manifest),
            ]
        )
        == 2
    )
    assert "not an input" in capsys.readouterr().err

    binding = [str(report), "examples", str(manifest)]
    assert (
        module.main(
            [
                str(report),
                "--class-exemptions",
                *binding,
                "--class-exemptions",
                *binding,
            ]
        )
        == 2
    )
    assert "duplicate class-exemption binding" in capsys.readouterr().err


def test_duplicate_resolved_report_paths_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    _report(report, covered=9, valid=10)

    assert module.main([str(report), str(report.resolve())]) == 2
    assert "duplicate coverage report" in capsys.readouterr().err


def test_duplicate_class_paths_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<coverage branches-covered="18" branches-valid="20">'
        "<packages><package><classes>"
        '<class name="first" filename="same.py"><lines>'
        '<line number="1" hits="1" branch="true" '
        'condition-coverage="90% (9/10)"/>'
        "</lines></class>"
        '<class name="second" filename="same.py"><lines>'
        '<line number="2" hits="1" branch="true" '
        'condition-coverage="90% (9/10)"/>'
        "</lines></class>"
        "</classes></package></packages></coverage>",
        encoding="utf-8",
    )

    assert module.main([str(report)]) == 2
    assert "duplicate coverage class" in capsys.readouterr().err


@pytest.mark.parametrize("minimum", ["-0.01", "100.01"])
def test_invalid_threshold_is_rejected(minimum: str) -> None:
    module = _load_module()
    with pytest.raises(SystemExit, match="2"):
        module.main(["coverage.xml", "--minimum", minimum])
