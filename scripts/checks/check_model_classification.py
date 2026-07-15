#!/usr/bin/env python3
"""Audit model lifecycle classification across public static contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_CATALOG_PATH = REPO_ROOT / "contracts" / "evidence_catalog_v1.json"
MODEL_CLASSIFICATION_PATH = REPO_ROOT / "contracts" / "model_classification.json"
MODEL_FAMILY_CATALOG_PATH = REPO_ROOT / "contracts" / "model_family_catalog.json"
SUPPORT_MATRIX_PATH = REPO_ROOT / "contracts" / "support_matrix.json"

CLASSIFICATION_FORMAT = "model-classification-v1"
AUDIT_FORMAT = "invarlock/model-classification-audit-v1"
CLASSIFICATIONS = {
    "published",
    "backlog",
    "blocked",
    "smoke_canary",
    "catalog_only",
    "usage_only",
    "out_of_scope",
}
ELIGIBILITY = {"eligible", "blocked", "not_applicable"}
SUPPORT_MATRIX_CLASSIFICATIONS = {"published", "backlog", "blocked"}
EVIDENCE_CATALOG_FORMAT = "invarlock/evidence-catalog-v1"
ENTRY_FIELDS = {
    "id",
    "classification",
    "eligibility",
    "support_matrix_lane_id",
    "catalog_family_ids",
    "candidate_id",
    "blockers",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    scope: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "scope": self.scope, "message": self.message}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _entry_id(entry: Mapping[str, Any]) -> str:
    for key in ("id", "support_matrix_lane_id", "candidate_id"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return "<unknown>"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("entries")
    return (
        [entry for entry in entries if isinstance(entry, dict)]
        if isinstance(entries, list)
        else []
    )


def _index_one(
    entries: Sequence[Mapping[str, Any]], key: str
) -> tuple[dict[str, Mapping[str, Any]], list[Finding]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    findings: list[Finding] = []
    for entry in entries:
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in indexed:
            findings.append(
                Finding(
                    "error",
                    f"model_classification:{key}",
                    f"{value!r} is declared by multiple entries",
                )
            )
        indexed[value] = entry
    return indexed, findings


def _index_many(
    entries: Sequence[Mapping[str, Any]], key: str
) -> tuple[dict[str, Mapping[str, Any]], list[Finding]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    findings: list[Finding] = []
    for entry in entries:
        for value in _string_list(entry.get(key)):
            if value in indexed:
                findings.append(
                    Finding(
                        "error",
                        f"model_classification:{key}",
                        f"{value!r} is declared by multiple entries",
                    )
                )
            indexed[value] = entry
    return indexed, findings


def _catalog_families(
    catalog: Mapping[str, Any],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    indexed: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for section in (
        "declared_support",
        "implemented_coverage",
        "usage_only",
        "recommended_additions",
    ):
        families = catalog.get(section)
        if not isinstance(families, list):
            continue
        for family in families:
            if not isinstance(family, Mapping):
                continue
            family_id = family.get("family_id")
            if isinstance(family_id, str) and family_id:
                indexed[family_id] = (section, family)
    return indexed


def _published_basis_candidates(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    section = catalog.get("published_basis_candidates_text_le_14b")
    if not isinstance(section, Mapping):
        return []
    candidates = section.get("candidates")
    return (
        [item for item in candidates if isinstance(item, Mapping)]
        if isinstance(candidates, list)
        else []
    )


def _support_rows(support_matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    lanes = support_matrix.get("lanes")
    if not isinstance(lanes, list):
        return {}
    return {
        lane["lane_id"]: lane
        for lane in lanes
        if isinstance(lane, Mapping)
        and isinstance(lane.get("lane_id"), str)
        and lane["lane_id"]
    }


def _collect_named_model_sources(
    *,
    catalog: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    support_matrix: Mapping[str, Any],
) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = defaultdict(set)

    for lane_id, row in _support_rows(support_matrix).items():
        for model_id in _string_list(row.get("representative_models")):
            sources[model_id].add(f"support_matrix:{lane_id}")

    for section, family in _catalog_families(catalog).values():
        family_id = family.get("family_id", "<unknown>")
        for model_id in _string_list(family.get("representative_models")):
            sources[model_id].add(f"model_family_catalog:{section}:{family_id}")

    for candidate in _published_basis_candidates(catalog):
        model_id = candidate.get("representative_model")
        candidate_id = candidate.get("candidate_id", "<unknown>")
        if isinstance(model_id, str) and model_id:
            sources[model_id].add(f"published_basis_candidate:{candidate_id}")

    entries = evidence_catalog.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            lane_id = entry.get("lane_id", "<unknown>")
            model = entry.get("model")
            if isinstance(model, Mapping):
                model_id = model.get("id")
                if isinstance(model_id, str) and model_id:
                    sources[model_id].add(f"evidence_catalog:{lane_id}")

    return sources


def _check_manifest_shape(
    payload: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> list[Finding]:
    findings: list[Finding] = []
    if payload.get("format_version") != CLASSIFICATION_FORMAT:
        findings.append(
            Finding(
                "error",
                "model_classification",
                f"format_version must be {CLASSIFICATION_FORMAT!r}",
            )
        )
    classification_values = _string_list(payload.get("classification_values"))
    if set(classification_values) != CLASSIFICATIONS or len(
        classification_values
    ) != len(CLASSIFICATIONS):
        findings.append(
            Finding(
                "error",
                "model_classification.classification_values",
                "classification_values must match the checker classification enum",
            )
        )
    eligibility_values = _string_list(payload.get("eligibility_values"))
    if set(eligibility_values) != ELIGIBILITY or len(eligibility_values) != len(
        ELIGIBILITY
    ):
        findings.append(
            Finding(
                "error",
                "model_classification.eligibility_values",
                "eligibility_values must match the checker eligibility enum",
            )
        )
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        findings.append(
            Finding("error", "model_classification.policy", "missing policy object")
        )
    else:
        allowed = _string_list(policy.get("allowed_named_checkpoint_license_ids"))
        if allowed != ["apache-2.0", "mit"]:
            findings.append(
                Finding(
                    "error",
                    "model_classification.policy",
                    "strict named-checkpoint policy must list apache-2.0 then mit",
                )
            )

    for entry in entries:
        scope = f"model_classification:{_entry_id(entry)}"
        unexpected = sorted(set(entry) - ENTRY_FIELDS)
        if unexpected:
            findings.append(
                Finding(
                    "error",
                    scope,
                    "entry contains fields outside the public classification "
                    f"contract: {', '.join(unexpected)}",
                )
            )
        classification = entry.get("classification")
        eligibility = entry.get("eligibility")
        if classification not in CLASSIFICATIONS:
            findings.append(
                Finding("error", scope, f"unknown classification {classification!r}")
            )
        if eligibility not in ELIGIBILITY:
            findings.append(
                Finding("error", scope, f"unknown eligibility {eligibility!r}")
            )
        if classification == "published" and eligibility != "eligible":
            findings.append(
                Finding("error", scope, "published entries must be eligible")
            )
        if classification == "blocked" and not _string_list(entry.get("blockers")):
            findings.append(
                Finding("error", scope, "blocked entries must list blockers")
            )

    blocked = payload.get("blocked_named_checkpoints")
    if not isinstance(blocked, list):
        findings.append(
            Finding(
                "error",
                "model_classification.blocked_named_checkpoints",
                "blocked_named_checkpoints must be a list",
            )
        )
    else:
        for index, item in enumerate(blocked):
            scope = f"model_classification.blocked_named_checkpoints[{index}]"
            if not isinstance(item, Mapping):
                findings.append(Finding("error", scope, "entry must be an object"))
                continue
            for key in ("model_id", "license_id", "reason"):
                value = item.get(key)
                if not isinstance(value, str) or not value:
                    findings.append(Finding("error", scope, f"missing {key}"))
    return findings


def _check_support_matrix(
    support_matrix: Mapping[str, Any],
    by_support_lane: Mapping[str, Mapping[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    for lane_id, row in _support_rows(support_matrix).items():
        entry = by_support_lane.get(lane_id)
        scope = f"support_matrix:{lane_id}"
        if entry is None:
            findings.append(
                Finding("error", scope, "missing model_classification entry")
            )
            continue
        classification = entry.get("classification")
        if classification not in SUPPORT_MATRIX_CLASSIFICATIONS:
            findings.append(
                Finding(
                    "error",
                    scope,
                    f"support-matrix lane cannot be classified as {classification!r}",
                )
            )
            continue

        support_tier = row.get("support_tier")
        docs_label = row.get("docs_label")
        if classification == "published":
            if support_tier != "published_basis":
                findings.append(
                    Finding(
                        "error",
                        scope,
                        "published classification requires support_tier='published_basis'",
                    )
                )
            if docs_label != "Yes":
                findings.append(
                    Finding(
                        "error",
                        scope,
                        "published classification requires docs_label='Yes'",
                    )
                )
        else:
            if support_tier == "published_basis":
                findings.append(
                    Finding(
                        "error",
                        scope,
                        f"{classification} classification cannot use published_basis",
                    )
                )
            if docs_label != "No":
                findings.append(
                    Finding(
                        "error",
                        scope,
                        f"{classification} classification requires docs_label='No'",
                    )
                )
    return findings


def _check_catalog(
    catalog: Mapping[str, Any],
    by_catalog_family: Mapping[str, Mapping[str, Any]],
    by_candidate: Mapping[str, Mapping[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    catalog_families = _catalog_families(catalog)

    for family_id, (section, family) in catalog_families.items():
        state = family.get("state")
        entry = by_catalog_family.get(family_id)
        if section == "declared_support" or state == "published_basis":
            if entry is None:
                findings.append(
                    Finding(
                        "error",
                        f"model_family_catalog:{family_id}",
                        "published/declared family missing model_classification entry",
                    )
                )
                continue
        if entry is None:
            continue
        classification = entry.get("classification")
        if classification == "published" and state != "published_basis":
            findings.append(
                Finding(
                    "error",
                    f"model_family_catalog:{family_id}",
                    "published classification requires catalog state published_basis",
                )
            )
        if classification == "blocked" and state == "published_basis":
            findings.append(
                Finding(
                    "error",
                    f"model_family_catalog:{family_id}",
                    "blocked classification cannot use catalog state published_basis",
                )
            )

    for candidate in _published_basis_candidates(catalog):
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        decision = candidate.get("decision")
        if decision not in {"blocked_missing_artifacts", "explicitly_out_of_scope"}:
            continue
        entry = by_candidate.get(candidate_id)
        scope = f"published_basis_candidate:{candidate_id}"
        if entry is None:
            findings.append(
                Finding("error", scope, "missing model_classification entry")
            )
            continue
        expected = (
            "blocked" if decision == "blocked_missing_artifacts" else "out_of_scope"
        )
        if entry.get("classification") != expected:
            findings.append(
                Finding(
                    "error",
                    scope,
                    f"candidate decision {decision!r} requires classification {expected!r}",
                )
            )
    return findings


def _check_evidence_catalog(
    evidence_catalog: Mapping[str, Any],
    support_matrix: Mapping[str, Any],
    by_support_lane: Mapping[str, Mapping[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    if evidence_catalog.get("format_version") != EVIDENCE_CATALOG_FORMAT:
        findings.append(
            Finding(
                "error",
                "evidence_catalog",
                f"format_version must be {EVIDENCE_CATALOG_FORMAT!r}",
            )
        )
    raw_entries = evidence_catalog.get("entries")
    if not isinstance(raw_entries, list):
        return [
            *findings,
            Finding("error", "evidence_catalog", "entries must be a list"),
        ]

    support_rows = _support_rows(support_matrix)
    seen: set[str] = set()
    for index, catalog_entry in enumerate(raw_entries):
        scope = f"evidence_catalog:entries[{index}]"
        if not isinstance(catalog_entry, Mapping):
            findings.append(Finding("error", scope, "entry must be an object"))
            continue
        lane_id = catalog_entry.get("lane_id")
        if not isinstance(lane_id, str) or not lane_id:
            findings.append(Finding("error", scope, "lane_id is required"))
            continue
        scope = f"evidence_catalog:{lane_id}"
        if lane_id in seen:
            findings.append(Finding("error", scope, "lane_id is duplicated"))
            continue
        seen.add(lane_id)
        support = support_rows.get(lane_id)
        classification = by_support_lane.get(lane_id)
        if support is None:
            findings.append(
                Finding("error", scope, "lane is absent from the support matrix")
            )
            continue
        if classification is None:
            findings.append(Finding("error", scope, "lane lacks model classification"))
        model = catalog_entry.get("model")
        if not isinstance(model, Mapping):
            findings.append(Finding("error", scope, "model must be an object"))
            continue
        if model.get("adapter") != support.get("adapter"):
            findings.append(
                Finding(
                    "error",
                    scope,
                    "adapter disagrees with the support matrix",
                )
            )
        representatives = set(_string_list(support.get("representative_models")))
        if representatives and model.get("id") not in representatives:
            findings.append(
                Finding(
                    "error",
                    scope,
                    "model is not a support-matrix representative",
                )
            )

    expected = set(support_rows)
    if seen != expected:
        findings.append(
            Finding(
                "error",
                "evidence_catalog",
                "lane IDs must exactly match the public support matrix: "
                f"missing={sorted(expected - seen)!r} extra={sorted(seen - expected)!r}",
            )
        )
    return findings


def _check_blocked_named_checkpoints(
    classification: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    support_matrix: Mapping[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    sources = _collect_named_model_sources(
        catalog=catalog,
        evidence_catalog=evidence_catalog,
        support_matrix=support_matrix,
    )
    blocked = classification.get("blocked_named_checkpoints")
    if not isinstance(blocked, list):
        return findings
    for item in blocked:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if model_id in sources:
            findings.append(
                Finding(
                    "error",
                    f"blocked_named_checkpoint:{model_id}",
                    "blocked checkpoint appears in structured metadata: "
                    + ", ".join(sorted(sources[model_id])),
                )
            )
    return findings


def audit() -> list[Finding]:
    classification = _load_json(MODEL_CLASSIFICATION_PATH)
    support_matrix = _load_json(SUPPORT_MATRIX_PATH)
    catalog = _load_json(MODEL_FAMILY_CATALOG_PATH)
    evidence_catalog = _load_json(EVIDENCE_CATALOG_PATH)
    entries = _entries(classification)

    findings: list[Finding] = []
    findings.extend(_check_manifest_shape(classification, entries))

    by_support_lane, support_index_findings = _index_one(
        entries, "support_matrix_lane_id"
    )
    by_candidate, candidate_index_findings = _index_one(entries, "candidate_id")
    by_catalog_family, catalog_index_findings = _index_many(
        entries, "catalog_family_ids"
    )
    findings.extend(support_index_findings)
    findings.extend(candidate_index_findings)
    findings.extend(catalog_index_findings)

    findings.extend(_check_support_matrix(support_matrix, by_support_lane))
    findings.extend(_check_catalog(catalog, by_catalog_family, by_candidate))
    findings.extend(
        _check_evidence_catalog(
            evidence_catalog,
            support_matrix,
            by_support_lane,
        )
    )
    findings.extend(
        _check_blocked_named_checkpoints(
            classification,
            catalog=catalog,
            evidence_catalog=evidence_catalog,
            support_matrix=support_matrix,
        )
    )

    return sorted(findings, key=lambda item: (item.severity, item.scope, item.message))


def _print_text(findings: Sequence[Finding]) -> None:
    if not findings:
        print("Model classification audit OK.")
        return
    print("Model classification audit failures:", file=sys.stderr)
    for finding in findings:
        print(
            f"  {finding.severity.upper()} {finding.scope}: {finding.message}",
            file=sys.stderr,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON audit output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    findings = audit()
    payload = {
        "schema": AUDIT_FORMAT,
        "ok": not findings,
        "finding_count": len(findings),
        "findings": [finding.as_dict() for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(findings)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
