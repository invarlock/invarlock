from __future__ import annotations

import importlib.util
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


def _write_line_only_cov_xml(path: Path, filename: str, line_rate: float) -> None:
    path.write_text(
        "\n".join(
            [
                "<?xml version='1.0' encoding='UTF-8'?>",
                "<coverage>",
                "  <packages>",
                "    <package name='pkg'>",
                "      <classes>",
                (
                    "        <class name='X' "
                    f"filename='{filename}' line-rate='{line_rate}'/>"
                ),
                "      </classes>",
                "    </package>",
                "  </packages>",
                "</coverage>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_checker(
    xml_path: Path,
    json_path: Path,
    extra_args: list[str] | None = None,
    *,
    allow_missing_threshold_files: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path("scripts") / "coverage" / "check_coverage_thresholds.py"),
        "--coverage",
        str(xml_path),
        "--json",
        str(json_path),
    ]
    if allow_missing_threshold_files:
        cmd.append("--allow-missing-threshold-files")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
