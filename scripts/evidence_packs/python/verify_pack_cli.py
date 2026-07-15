"""Command-line parser for structured evidence-pack verification checks."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


def parse_args(
    argv: list[str] | None, *, handlers: dict[str, Callable[[argparse.Namespace], int]]
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured JSON/path checks for evidence-pack shell entrypoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_field = subparsers.add_parser("manifest-field")
    manifest_field.add_argument("manifest", type=Path)
    manifest_field.add_argument("field")
    manifest_field.set_defaults(func=handlers["manifest-field"])

    path_within = subparsers.add_parser("path-within")
    path_within.add_argument("root", type=Path)
    path_within.add_argument("candidate", type=Path)
    path_within.set_defaults(func=handlers["path-within"])

    scenario_strictness = subparsers.add_parser("scenario-strictness")
    scenario_strictness.add_argument("scenarios", type=Path)
    scenario_strictness.add_argument("scenario_id")
    scenario_strictness.set_defaults(func=handlers["scenario-strictness"])

    report_scenario_id = subparsers.add_parser("report-scenario-id")
    report_scenario_id.add_argument("pack_dir", type=Path)
    report_scenario_id.add_argument("report", type=Path)
    report_scenario_id.set_defaults(func=handlers["report-scenario-id"])

    report_failure = subparsers.add_parser("report-expects-verify-failure")
    report_failure.add_argument("pack_dir", type=Path)
    report_failure.add_argument("report", type=Path)
    report_failure.set_defaults(func=handlers["report-expects-verify-failure"])

    verify_reports = subparsers.add_parser("verify-reports")
    verify_reports.add_argument("pack_dir", type=Path)
    verify_reports.add_argument("--json-out", type=Path)
    verify_reports.add_argument("--profile", default="dev")
    verify_reports.add_argument(
        "--report-assurance", default="report", choices=("report", "strict", "off")
    )
    verify_reports.add_argument("--expected-runtime-image-digest")
    verify_reports.add_argument("--policy-pack", type=Path)
    verify_reports.add_argument("--require-clean", action="store_true")
    verify_reports.add_argument("--write-sidecars", action="store_true")
    verify_reports.add_argument("--summary-out", type=Path)
    verify_reports.add_argument("--staged-baselines", action="store_true")
    verify_reports.set_defaults(func=handlers["verify-reports"])

    policy_materials = subparsers.add_parser("policy-materials")
    policy_materials.add_argument("pack_dir", type=Path)
    policy_materials.add_argument(
        "--report-assurance", default="report", choices=("report", "strict", "off")
    )
    policy_materials.add_argument("--policy-pack", type=Path)
    policy_materials.set_defaults(func=handlers["policy-materials"])

    json_object = subparsers.add_parser("json-object")
    json_object.add_argument("path", type=Path)
    json_object.add_argument("--label", default="metadata file")
    json_object.set_defaults(func=handlers["json-object"])

    scenarios_manifest = subparsers.add_parser("scenarios-manifest")
    scenarios_manifest.add_argument("path", type=Path)
    scenarios_manifest.set_defaults(func=handlers["scenarios-manifest"])

    extra_files = subparsers.add_parser("extra-files")
    extra_files.add_argument("pack_dir", type=Path)
    extra_files.add_argument("--strict", action="store_true")
    extra_files.set_defaults(func=handlers["extra-files"])

    validate_manifest = subparsers.add_parser("validate-manifest")
    validate_manifest.add_argument("manifest", type=Path)
    validate_manifest.set_defaults(func=handlers["validate-manifest"])

    provenance = subparsers.add_parser("manifest-provenance")
    provenance.add_argument("pack_dir", type=Path)
    provenance.set_defaults(func=handlers["manifest-provenance"])

    verdict_binding = subparsers.add_parser("final-verdict-binding")
    verdict_binding.add_argument("pack_dir", type=Path)
    verdict_binding.add_argument("--require-binding", action="store_true")
    verdict_binding.set_defaults(func=handlers["final-verdict-binding"])

    baseline_materials = subparsers.add_parser("baseline-materials")
    baseline_materials.add_argument("pack_dir", type=Path)
    baseline_materials.add_argument(
        "--report-assurance", default="report", choices=("report", "strict", "off")
    )
    baseline_materials.set_defaults(func=handlers["baseline-materials"])

    signature = subparsers.add_parser(
        "signature", help="Verify a package-native evidence-pack signature bundle."
    )
    signature.add_argument("pack_dir", type=Path)
    signature.add_argument(
        "--strict",
        action="store_true",
        help="Fail closed when manifest.signature.json is missing.",
    )
    signature.add_argument(
        "--expected-fingerprint",
        help="Require the signer to match this sha256:... key fingerprint.",
    )
    signature.set_defaults(func=handlers["signature"])
    return parser.parse_args(argv)
