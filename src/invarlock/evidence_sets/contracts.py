"""Closed composition inputs; submitted indexes carry no recipient authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from invarlock.captured_contracts import atomic_write, read_file, sha
from invarlock.core.evaluation_request import _reference_parts
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.public_contracts import (
    load_evidence_set_recipient_policy_schema,
    load_evidence_set_schema,
)

INDEX_FILE = "evidence-set.json"
CONTROL_LIMIT = 256 * 1024
SCOPE = "same-answer-component-conjunction-v1"
STATISTICAL_SCOPE = "component-methods-no-joint-confidence"
MEMBERS = ("deterministic", "judge")
STATEMENTS = {"deterministic": "manifest.json", "judge": "envelope.json"}
DETERMINISTIC_KINDS = frozenset(
    {
        "exact_match",
        "normalized_match",
        "numeric_tolerance",
        "json_fields",
        "json_exact",
        "token_f1",
    }
)


class EvidenceSetError(ValueError):
    """A bounded composition is malformed, mismatched, or unsupported."""


def read_object(
    path: Path, *, maximum: int = CONTROL_LIMIT
) -> tuple[dict[str, Any], bytes]:
    raw = read_file(Path(path), maximum)
    value = parse_json_bytes(raw, label=path.name)
    if not isinstance(value, dict):
        raise EvidenceSetError(f"{path.name} must contain a JSON object")
    return value, raw


def validate(value: dict[str, Any], schema: dict[str, Any], *, label: str) -> None:
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error is not None:
        raise EvidenceSetError(f"{label} is invalid: {error.message[:240]}")


def is_evidence_set(path: Path) -> bool:
    candidate = Path(path) / INDEX_FILE
    return candidate.exists() or candidate.is_symlink()


def member_path(root: Path, reference: str) -> Path:
    return (
        Path(root)
        .absolute()
        .joinpath(*_reference_parts(reference, label="member path"))
    )


def require_external(path: Path, root: Path) -> None:
    if Path(path).resolve().is_relative_to(Path(root).resolve()):
        raise EvidenceSetError(
            "recipient inputs and outputs must remain outside the evidence set"
        )


def load_index(root: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = read_object(Path(root) / INDEX_FILE)
    validate(value, load_evidence_set_schema(), label="evidence set index")
    paths = [member_path(root, value["members"][name]["path"]) for name in MEMBERS]
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise EvidenceSetError("evidence set member paths overlap")
    if any(
        (Path(root) / filename).exists() or (Path(root) / filename).is_symlink()
        for filename in STATEMENTS.values()
    ):
        raise EvidenceSetError(
            "evidence set contains conflicting format discriminators"
        )
    return value, raw


def load_policy(path: Path, root: Path) -> tuple[dict[str, Any], bytes]:
    require_external(path, root)
    value, raw = read_object(Path(path))
    validate(
        value,
        load_evidence_set_recipient_policy_schema(),
        label="evidence set recipient policy",
    )
    return value, raw


def check_statements(root: Path, index: dict[str, Any]) -> None:
    for name in MEMBERS:
        member = index["members"][name]
        path = member_path(root, member["path"])
        if is_evidence_set(path):
            raise EvidenceSetError("nested evidence sets are unsupported")
        other = "envelope.json" if name == "deterministic" else "manifest.json"
        if (path / other).exists() or (path / other).is_symlink():
            raise EvidenceSetError("member contains conflicting format discriminators")
        if (
            sha(read_file(path / STATEMENTS[name], CONTROL_LIMIT))
            != member["statement_sha256"]
        ):
            raise EvidenceSetError(f"{name} statement differs from index pin")


def write_evidence_set_index(root: Path, *, deterministic: str, judge: str) -> Path:
    """Index two existing packs without copying credentials or conferring trust.

    Statement format and shared inputs are checked by verification. This helper
    only constructs bounded transport pins; no acceptance claim is produced.
    """
    root = Path(root).absolute()
    value: dict[str, Any] = {
        "format": "invarlock/evidence-set-v1",
        "members": {
            name: {
                "kind": kind,
                "path": reference,
                "statement_sha256": sha(
                    read_file(
                        member_path(root, reference) / STATEMENTS[name], CONTROL_LIMIT
                    )
                ),
            }
            for name, kind, reference in (
                ("deterministic", "captured", deterministic),
                ("judge", "judge", judge),
            )
        },
    }
    validate(value, load_evidence_set_schema(), label="evidence set index")
    paths = [member_path(root, value["members"][name]["path"]) for name in MEMBERS]
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise EvidenceSetError("evidence set member paths overlap")
    for filename in STATEMENTS.values():
        if (root / filename).exists() or (root / filename).is_symlink():
            raise EvidenceSetError(
                "evidence set contains conflicting format discriminators"
            )
    check_statements(root, value)
    destination = root / INDEX_FILE
    atomic_write(destination, canonical_json_bytes(value))
    return destination
