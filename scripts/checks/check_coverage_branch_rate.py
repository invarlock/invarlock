from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import NamedTuple

_CONDITION_COVERAGE_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)?%\s*\((\d+)\s*/\s*(\d+)\)\s*$"
)


class CoverageCounts(NamedTuple):
    covered: int
    valid: int

    @property
    def rate(self) -> float:
        return 100.0 * self.covered / self.valid


class ClassCoverage(NamedTuple):
    name: str
    counts: CoverageCounts


class CoverageReport(NamedTuple):
    counts: CoverageCounts
    classes: tuple[ClassCoverage, ...]
    class_names: frozenset[str]


def _internal_branch_counts(root: ET.Element, path: Path) -> CoverageCounts:
    covered = 0
    valid = 0
    for line in root.findall(".//line"):
        if line.attrib.get("branch") != "true":
            continue
        condition_coverage = line.attrib.get("condition-coverage", "")
        match = _CONDITION_COVERAGE_PATTERN.fullmatch(condition_coverage)
        if match is None:
            raise ValueError(
                f"invalid condition coverage in {path}: {condition_coverage!r}"
            )
        line_covered, line_valid = (int(value) for value in match.groups())
        if line_valid <= 0 or line_covered < 0 or line_covered > line_valid:
            raise ValueError(
                f"invalid condition counts in {path}: "
                f"covered={line_covered}, valid={line_valid}"
            )
        covered += line_covered
        valid += line_valid
    return CoverageCounts(covered=covered, valid=valid)


def _branch_counts(path: Path) -> CoverageReport:
    try:
        root = ET.parse(path).getroot()
        if root.tag != "coverage":
            raise ValueError(f"unexpected XML root {root.tag!r}")
        covered = int(root.attrib["branches-covered"])
        valid = int(root.attrib["branches-valid"])
    except (OSError, ET.ParseError, KeyError, ValueError) as exc:
        raise ValueError(f"cannot read branch coverage from {path}: {exc}") from exc
    if covered < 0 or valid <= 0 or covered > valid:
        raise ValueError(
            f"invalid branch counts in {path}: covered={covered}, valid={valid}"
        )
    counts = CoverageCounts(covered=covered, valid=valid)
    internal_counts = _internal_branch_counts(root, path)
    if internal_counts != counts:
        raise ValueError(
            f"branch totals do not match report contents in {path}: "
            f"root={counts.covered}/{counts.valid}, "
            f"lines={internal_counts.covered}/{internal_counts.valid}"
        )
    classes: list[ClassCoverage] = []
    class_names: set[str] = set()
    for index, class_element in enumerate(root.findall(".//class"), start=1):
        class_name = (
            class_element.attrib.get("filename")
            or class_element.attrib.get("name")
            or f"class #{index}"
        )
        if class_name in class_names:
            raise ValueError(f"duplicate coverage class in {path}: {class_name}")
        class_names.add(class_name)
        class_counts = _internal_branch_counts(class_element, path)
        if class_counts.valid == 0:
            continue
        classes.append(ClassCoverage(name=class_name, counts=class_counts))
    class_totals = CoverageCounts(
        covered=sum(class_coverage.counts.covered for class_coverage in classes),
        valid=sum(class_coverage.counts.valid for class_coverage in classes),
    )
    if class_totals != counts:
        raise ValueError(
            f"branch totals do not match class contents in {path}: "
            f"root={counts.covered}/{counts.valid}, "
            f"classes={class_totals.covered}/{class_totals.valid}"
        )
    return CoverageReport(
        counts=counts,
        classes=tuple(classes),
        class_names=frozenset(class_names),
    )


def _reject_duplicate_reports(paths: list[Path]) -> None:
    seen: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(
                f"duplicate coverage report {path}: resolves to the same path as "
                f"{seen[resolved]}"
            )
        seen[resolved] = path


def _normalized_repo_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return path


def _load_exemption_manifest(
    manifest: Path,
    *,
    source_root: PurePosixPath,
    report_path: Path,
    report: CoverageReport,
) -> frozenset[str]:
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"cannot read class exemptions from {manifest}: {exc}"
        ) from exc

    entries: dict[PurePosixPath, int] = {}
    class_names: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        entry = _normalized_repo_path(
            value,
            label=f"class exemption in {manifest}:{line_number}",
        )
        if entry in entries:
            raise ValueError(
                f"duplicate class exemption in {manifest}:{line_number}: {entry}; "
                f"first declared on line {entries[entry]}"
            )
        entries[entry] = line_number
        try:
            class_name = entry.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"class exemption in {manifest}:{line_number} is outside source root "
                f"{source_root}: {entry}"
            ) from exc
        source_path = Path(entry)
        if class_name == "." or (
            class_name not in report.class_names
            and (source_path.is_symlink() or not source_path.is_file())
        ):
            raise ValueError(
                f"class exemption in {manifest}:{line_number} does not match a class "
                f"in {report_path} or an existing regular source file: {entry}"
            )
        if class_name in report.class_names:
            class_names.add(class_name)
    return frozenset(class_names)


def _class_exemptions_by_report(
    bindings: list[list[str]],
    reports: list[tuple[Path, CoverageReport]],
) -> dict[Path, frozenset[str]]:
    reports_by_path = {path.resolve(): (path, report) for path, report in reports}
    exemptions: dict[Path, frozenset[str]] = {}
    for report_value, source_root_value, manifest_value in bindings:
        bound_path = Path(report_value).resolve()
        if bound_path not in reports_by_path:
            raise ValueError(
                f"class exemptions reference a report that is not an input: "
                f"{report_value}"
            )
        if bound_path in exemptions:
            raise ValueError(f"duplicate class-exemption binding for {report_value}")
        source_root = _normalized_repo_path(
            source_root_value,
            label="class-exemption source root",
        )
        report_path, report = reports_by_path[bound_path]
        exemptions[bound_path] = _load_exemption_manifest(
            Path(manifest_value),
            source_root=source_root,
            report_path=report_path,
            report=report,
        )
    return exemptions


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce aggregate and per-module branch rates in coverage.py XML reports."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--minimum", type=float, default=90.0)
    parser.add_argument(
        "--class-exemptions",
        action="append",
        default=[],
        nargs=3,
        metavar=("REPORT", "SOURCE_ROOT", "MANIFEST"),
        help=(
            "exclude exact entries in MANIFEST from line-coverage enforcement for "
            "REPORT after removing SOURCE_ROOT; use only where an alternate "
            "execution gate is documented"
        ),
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.minimum <= 100.0:
        parser.error("--minimum must be between 0 and 100")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _reject_duplicate_reports(args.reports)
        reports = [(path, _branch_counts(path)) for path in args.reports]
        exemptions_by_report = _class_exemptions_by_report(
            args.class_exemptions,
            reports,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    class_failures: list[str] = []
    for path, report in reports:
        counts = report.counts
        class_exemptions = exemptions_by_report.get(path.resolve(), frozenset())
        print(
            f"{path}: {counts.rate:.2f}% branch coverage "
            f"({counts.covered}/{counts.valid})"
        )
        for class_coverage in report.classes:
            class_counts = class_coverage.counts
            if (
                class_coverage.name not in class_exemptions
                and class_counts.rate < args.minimum
            ):
                class_failures.append(
                    f"{path}: {class_coverage.name}: {class_counts.rate:.2f}% "
                    f"branch coverage ({class_counts.covered}/{class_counts.valid}); "
                    f"required: {args.minimum:.2f}%"
                )

    enforced_counts: list[CoverageCounts] = []
    for path, report in reports:
        class_exemptions = exemptions_by_report.get(path.resolve(), frozenset())
        exempt_counts = CoverageCounts(
            covered=sum(
                class_coverage.counts.covered
                for class_coverage in report.classes
                if class_coverage.name in class_exemptions
            ),
            valid=sum(
                class_coverage.counts.valid
                for class_coverage in report.classes
                if class_coverage.name in class_exemptions
            ),
        )
        enforced_counts.append(
            CoverageCounts(
                covered=report.counts.covered - exempt_counts.covered,
                valid=report.counts.valid - exempt_counts.valid,
            )
        )
    total = CoverageCounts(
        covered=sum(counts.covered for counts in enforced_counts),
        valid=sum(counts.valid for counts in enforced_counts),
    )
    if total.valid <= 0:
        print("ERROR: no enforceable branch coverage remains", file=sys.stderr)
        return 2
    summary = (
        f"aggregate branch coverage: {total.rate:.2f}% "
        f"({total.covered}/{total.valid}); required: {args.minimum:.2f}%"
    )
    for failure in class_failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    if total.rate < args.minimum:
        print(f"ERROR: {summary}", file=sys.stderr)
        return 1
    if class_failures:
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
