#!/usr/bin/env python3
"""Audit public evidence classification and verifier metadata."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from invarlock.reporting.report_schema import validate_report  # noqa: E402
from scripts.checks.public_evidence_checks.artifacts import (  # noqa: E402
    _check_guard_value_demo,
    _check_published_basis_multimodal_quality,
    _check_signed_pack,
    _is_direct_published_basis_artifact,
    _require_path,
)
from scripts.checks.public_evidence_checks.common import (  # noqa: E402
    META_FILENAME,
    PUBLIC_EVIDENCE_ROOT,
    SCHEMA,
    _artifact_dirs,
    _check_duplicate_root_evaluation_reports,
    _check_public_evidence_privacy,
    _load_json,
    _relative,
)
from scripts.checks.public_evidence_checks.index import (  # noqa: E402
    _check_packaged_public_evidence_index,
)
from scripts.checks.public_evidence_checks.negative_fixtures import (  # noqa: E402
    check_current_negative_fixture_index,
)
from scripts.checks.public_evidence_checks.summaries import (  # noqa: E402
    _check_attention_backend_compatibility,
    _check_runtime_backend_compatibility,
)

EVIDENCE_CLASS_REGISTRY: dict[str, dict[str, str | None]] = {
    "contract_fixture": {"kind": "fixture", "specialized_checker": None},
    "strict_pass_fixture": {"kind": "fixture", "specialized_checker": None},
    "historical_archived_fixture": {
        "kind": "historical",
        "specialized_checker": None,
    },
    "historical_archived_run": {
        "kind": "historical",
        "specialized_checker": None,
    },
    "caught_regression_fixture": {"kind": "fixture", "specialized_checker": None},
    "policy_failure_fixture": {"kind": "fixture", "specialized_checker": None},
    "byoe_subject_fixture": {"kind": "fixture", "specialized_checker": None},
    "real_model_run": {"kind": "real", "specialized_checker": None},
    "real_guard_value_demo": {
        "kind": "real",
        "specialized_checker": "guard_value_demo",
    },
    "signed_real_model_pack": {"kind": "real", "specialized_checker": None},
    "runtime_backend_compatibility": {
        "kind": "summary",
        "specialized_checker": "runtime_backend_compatibility",
    },
    "attention_backend_compatibility": {
        "kind": "summary",
        "specialized_checker": "attention_backend_compatibility",
    },
}

EvidenceChecker = Callable[[list[str], Path, dict[str, Any]], None]

SPECIALIZED_EVIDENCE_CHECKERS: dict[str, EvidenceChecker] = {
    "guard_value_demo": _check_guard_value_demo,
    "runtime_backend_compatibility": _check_runtime_backend_compatibility,
    "attention_backend_compatibility": _check_attention_backend_compatibility,
}


def check_public_evidence(
    root: Path = PUBLIC_EVIDENCE_ROOT,
    *,
    fetch_external_assets: bool = False,
    require_current_negative_evidence: bool = False,
) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    if not (root / "README.md").is_file():
        errors.append(f"{_relative(root)}: README.md is required")
    if not root.is_dir():
        return [f"public evidence root not found: {root}"]
    _check_public_evidence_privacy(errors, root)
    _check_duplicate_root_evaluation_reports(errors, root)
    current_negative_evidence_valid = check_current_negative_fixture_index(errors, root)
    if require_current_negative_evidence and not current_negative_evidence_valid:
        errors.append(
            f"{_relative(root)}: release closure requires a validated "
            "current negative-evidence index"
        )
    if root == PUBLIC_EVIDENCE_ROOT.resolve():
        _check_packaged_public_evidence_index(
            errors,
            root,
            fetch_external_assets=fetch_external_assets,
        )

    for artifact_dir in sorted(_artifact_dirs(root)):
        meta_path = artifact_dir / META_FILENAME
        if not meta_path.is_file():
            errors.append(f"{_relative(artifact_dir)}: missing {META_FILENAME}")
            continue

        metadata, error = _load_json(meta_path)
        if error:
            errors.append(error)
            continue
        assert metadata is not None

        if metadata.get("schema") != SCHEMA:
            errors.append(f"{_relative(meta_path)}: schema must be {SCHEMA}")

        evidence_class = metadata.get("evidence_class")
        if not isinstance(evidence_class, str):
            errors.append(f"{_relative(meta_path)}: invalid evidence_class")
            continue
        class_spec = EVIDENCE_CLASS_REGISTRY.get(evidence_class)
        if class_spec is None:
            errors.append(f"{_relative(meta_path)}: invalid evidence_class")
            continue

        summary = str(metadata.get("summary") or "").lower()
        class_kind = class_spec["kind"]
        specialized_checker = class_spec["specialized_checker"]

        if class_kind == "fixture" and "fixture" not in summary:
            errors.append(f"{_relative(meta_path)}: fixture evidence must say fixture")
        if class_kind == "historical":
            allowed_historical_kinds = (
                {
                    "caught_regression_fixture",
                    "policy_failure_fixture",
                    "byoe_subject_fixture",
                }
                if evidence_class == "historical_archived_fixture"
                else {"real_model_run"}
            )
            if metadata.get("historical_evidence_kind") not in allowed_historical_kinds:
                errors.append(
                    f"{_relative(meta_path)}: historical evidence must retain "
                    "its prior evidence classification"
                )
            historical_status = metadata.get("current_verifier_status")
            if (
                not isinstance(historical_status, str)
                or not historical_status.startswith("historical artifact;")
                or "current verifier rejects" not in historical_status
            ):
                errors.append(
                    f"{_relative(meta_path)}: historical evidence must state its "
                    "non-current verifier status"
                )
            if metadata.get("verifier_command_expectation") != (
                "expected_nonzero_current_contract_rejection"
            ):
                errors.append(
                    f"{_relative(meta_path)}: historical evidence must expect "
                    "current-contract rejection"
                )

        artifact_paths = metadata.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            errors.append(f"{_relative(meta_path)}: artifact_paths must be an object")
            continue

        report_path: Path | None = None
        if "evaluation_report" in artifact_paths:
            report_path = _require_path(
                errors, artifact_dir, artifact_paths, "evaluation_report"
            )
            _require_path(errors, artifact_dir, artifact_paths, "runtime_manifest")
        elif (artifact_dir / "evaluation.report.json").is_file():
            report_path = _require_path(
                errors, artifact_dir, artifact_paths, "evaluation_report"
            )
            _require_path(errors, artifact_dir, artifact_paths, "runtime_manifest")
        if report_path is not None and _is_direct_published_basis_artifact(
            artifact_dir, root
        ):
            _check_published_basis_multimodal_quality(errors, artifact_dir, report_path)
        if (
            report_path is not None
            and root == PUBLIC_EVIDENCE_ROOT.resolve()
            and class_kind != "historical"
        ):
            report_payload, report_error = _load_json(report_path)
            if report_error:
                errors.append(report_error)
            elif report_payload is None or not validate_report(report_payload):
                errors.append(
                    f"{_relative(report_path)}: evaluation report is not valid under the current schema"
                )

        if class_kind == "real":
            _require_path(errors, artifact_dir, artifact_paths, "run_command")
            if "invarlock evaluate" not in str(metadata.get("generated_by") or ""):
                errors.append(
                    f"{_relative(meta_path)}: real runs must record invarlock evaluate"
                )
            if "fixture" in summary:
                errors.append(
                    f"{_relative(meta_path)}: real-run summary must not say fixture"
                )

        if "evidence_pack" in artifact_paths:
            _check_signed_pack(errors, artifact_dir, metadata, artifact_paths)

        if specialized_checker is not None:
            checker = SPECIALIZED_EVIDENCE_CHECKERS[specialized_checker]
            checker(errors, artifact_dir, artifact_paths)

        commands = metadata.get("verifier_commands")
        if not isinstance(commands, list) or not commands:
            errors.append(
                f"{_relative(meta_path)}: verifier_commands must be a non-empty list"
            )

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PUBLIC_EVIDENCE_ROOT,
        help="Public evidence root to audit.",
    )
    parser.add_argument(
        "--fetch-external-assets",
        action="store_true",
        help="Download external public-evidence assets and verify size/SHA256.",
    )
    parser.add_argument(
        "--require-current-negative-evidence",
        action="store_true",
        help=(
            "Fail closed unless a typed current negative-evidence index is present "
            "and every named failure replays under strict release verification."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = check_public_evidence(
        args.root,
        fetch_external_assets=args.fetch_external_assets,
        require_current_negative_evidence=args.require_current_negative_evidence,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Public evidence audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
