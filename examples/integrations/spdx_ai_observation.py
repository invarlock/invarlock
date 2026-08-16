#!/usr/bin/env python3
"""Build one bounded SPDX 3.0.1 AI observation payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from invarlock.core.runtime_provider import GGUFArtifactIdentity
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
    encode_artifact_identity,
)

SPDX_CONTEXT = "https://spdx.org/rdf/3.0.1/spdx-context.jsonld"
SPDX_SPEC_VERSION = "3.0.1"
SPDX_SPEC_REPOSITORY = "https://github.com/spdx/spdx-spec"
SPDX_SPEC_COMMIT = "61a649da8ca27924ac1ca8d2a061cb228839b24c"
OBSERVATION_PAYLOAD_FORMAT = "invarlock/example-spdx-ai-observation-v1"
NO_ASSERTION_LICENSE = "https://spdx.org/rdf/3.0.1/terms/Licensing/NoAssertion"

_ROOT = Path(__file__).resolve().parent / "spdx-ai-observation"
DEFAULT_SOURCE = _ROOT / "source/model-aibom.spdx3.json"
DEFAULT_ARTIFACT_IDENTITY = _ROOT / "source/model-artifact.identity.json"
DEFAULT_EXPECTED = _ROOT / "observation-payload.json"
_MAX_SOURCE_BYTES = 1024 * 1024
_MAX_GRAPH_ELEMENTS = 64
_SHA256_HEX_LENGTH = 64


class SpdxObservationError(ValueError):
    """Raised when the bounded observation cannot be built safely."""


def _validate_spdx_canonical_value(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or any(not 0x21 <= ord(character) <= 0x7F for character in key)
            ):
                raise SpdxObservationError(
                    f"{label} object names must contain only SPDX canonical ASCII"
                )
            _validate_spdx_canonical_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_spdx_canonical_value(item, label=f"{label}[{index}]")
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    raise SpdxObservationError(
        f"{label} contains a value outside the SPDX canonical JSON subset"
    )


def spdx_canonical_json_bytes(value: object) -> bytes:
    """Encode the SPDX canonical-JSON subset without a final line feed."""

    _validate_spdx_canonical_value(value, label="SPDX value")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpdxObservationError(f"SPDX value is not canonical JSON: {exc}") from exc


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SpdxObservationError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _object_array(value: object, *, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise SpdxObservationError(f"{label} must be a non-empty array")
    if len(value) > _MAX_GRAPH_ELEMENTS:
        raise SpdxObservationError(
            f"{label} exceeds the {_MAX_GRAPH_ELEMENTS}-element example limit"
        )
    return [
        _object(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _string_array(value: object, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise SpdxObservationError(f"{label} must be a non-empty string array")
    return cast(list[str], value)


def _one_by_type(
    graph: Sequence[Mapping[str, object]], type_name: str
) -> Mapping[str, object]:
    matches = [item for item in graph if item.get("type") == type_name]
    if len(matches) != 1:
        raise SpdxObservationError(
            f"SPDX graph must contain exactly one {type_name} object"
        )
    return matches[0]


def _element_id(element: Mapping[str, object], *, label: str) -> str:
    value = element.get("spdxId", element.get("@id"))
    if not isinstance(value, str) or not value:
        raise SpdxObservationError(f"{label} must have a non-empty spdxId or @id")
    return value


def _validate_spdx_subset(document: Mapping[str, object]) -> tuple[str, ...]:
    """Validate only the explicitly documented example subset."""

    if set(document) != {"@context", "@graph"}:
        raise SpdxObservationError(
            "SPDX document must contain only @context and @graph"
        )
    if document.get("@context") != SPDX_CONTEXT:
        raise SpdxObservationError("SPDX document context is not pinned to 3.0.1")
    graph = _object_array(document.get("@graph"), label="SPDX @graph")

    seen_ids: set[str] = set()
    for index, element in enumerate(graph):
        element_id = _element_id(element, label=f"SPDX @graph[{index}]")
        if element_id in seen_ids:
            raise SpdxObservationError(f"SPDX graph has duplicate ID {element_id!r}")
        seen_ids.add(element_id)

    creation = _one_by_type(graph, "CreationInfo")
    creation_id = _element_id(creation, label="SPDX CreationInfo")
    if creation.get("specVersion") != SPDX_SPEC_VERSION:
        raise SpdxObservationError("SPDX CreationInfo specVersion must be 3.0.1")
    _string_array(creation.get("createdBy"), label="SPDX CreationInfo.createdBy")
    if not isinstance(creation.get("created"), str):
        raise SpdxObservationError("SPDX CreationInfo.created must be a timestamp")

    spdx_document = _one_by_type(graph, "SpdxDocument")
    profiles = set(
        _string_array(
            spdx_document.get("profileConformance"),
            label="SPDX SpdxDocument.profileConformance",
        )
    )
    required_profiles = {"ai", "core", "expandedLicensing", "software"}
    if not required_profiles.issubset(profiles):
        raise SpdxObservationError(
            "SPDX SpdxDocument profileConformance is missing the example profiles"
        )

    package = _one_by_type(graph, "ai_AIPackage")
    package_id = _element_id(package, label="SPDX ai_AIPackage")
    required_package_fields = {
        "ai_typeOfModel",
        "creationInfo",
        "name",
        "releaseTime",
        "software_downloadLocation",
        "software_packageVersion",
        "software_primaryPurpose",
        "spdxId",
        "suppliedBy",
        "type",
        "verifiedUsing",
    }
    if not required_package_fields.issubset(package):
        raise SpdxObservationError(
            "SPDX ai_AIPackage is missing required example fields"
        )
    if package.get("creationInfo") != creation_id:
        raise SpdxObservationError("SPDX ai_AIPackage creationInfo binding is invalid")
    if package.get("software_primaryPurpose") != "model":
        raise SpdxObservationError("SPDX ai_AIPackage primary purpose must be model")
    _string_array(package.get("ai_typeOfModel"), label="SPDX ai_typeOfModel")

    supplier_id = package.get("suppliedBy")
    if not isinstance(supplier_id, str):
        raise SpdxObservationError("SPDX ai_AIPackage suppliedBy must be one agent ID")
    suppliers = [
        item
        for item in graph
        if item.get("type") in {"Agent", "Organization", "Person", "SoftwareAgent"}
        and item.get("spdxId") == supplier_id
    ]
    if len(suppliers) != 1:
        raise SpdxObservationError("SPDX suppliedBy agent is missing or ambiguous")

    hashes = _object_array(
        package.get("verifiedUsing"), label="SPDX ai_AIPackage.verifiedUsing"
    )
    if len(hashes) != 1:
        raise SpdxObservationError(
            "SPDX ai_AIPackage must have exactly one example integrity method"
        )
    integrity_hash = hashes[0]
    if set(integrity_hash) != {"algorithm", "hashValue", "type"}:
        raise SpdxObservationError("SPDX example Hash fields are invalid")
    hash_value = integrity_hash.get("hashValue")
    if (
        integrity_hash.get("type") != "Hash"
        or integrity_hash.get("algorithm") != "sha256"
        or not isinstance(hash_value, str)
        or len(hash_value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in hash_value)
    ):
        raise SpdxObservationError("SPDX example Hash must be lowercase SHA-256")

    document_elements = set(
        _string_array(spdx_document.get("element"), label="SPDX SpdxDocument.element")
    )
    document_roots = set(
        _string_array(
            spdx_document.get("rootElement"),
            label="SPDX SpdxDocument.rootElement",
        )
    )
    if package_id not in document_elements or document_roots != {package_id}:
        raise SpdxObservationError("SPDX document does not root the AI package")

    relationships = [item for item in graph if item.get("type") == "Relationship"]
    expected_relationships = {"hasConcludedLicense", "hasDeclaredLicense"}
    actual_relationships: list[str] = []
    for relationship in relationships:
        if relationship.get("from") != package_id:
            continue
        relationship_type = relationship.get("relationshipType")
        if relationship_type not in expected_relationships:
            continue
        if relationship.get("to") != [NO_ASSERTION_LICENSE]:
            raise SpdxObservationError(
                f"SPDX {relationship_type} must point to NoAssertion"
            )
        relationship_id = _element_id(relationship, label="SPDX Relationship")
        if relationship_id not in document_elements:
            raise SpdxObservationError(
                f"SPDX {relationship_type} is not listed in document elements"
            )
        actual_relationships.append(cast(str, relationship_type))
    if sorted(actual_relationships) != sorted(expected_relationships):
        raise SpdxObservationError(
            "SPDX AI package requires one declared and one concluded license relationship"
        )

    return (
        "canonical_json_bytes",
        "pinned_3_0_1_context_and_creation_info",
        "unique_graph_identifiers",
        "single_rooted_ai_package",
        "ai_package_required_example_fields",
        "single_sha256_integrity_method",
        "supplier_reference",
        "declared_and_concluded_license_relationships",
    )


def load_spdx_document(
    source_bytes: bytes,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Parse canonical SPDX source bytes and run the bounded subset checks."""

    try:
        decoded = parse_json_bytes(source_bytes, label="SPDX source document")
    except StrictJsonError as exc:
        raise SpdxObservationError(str(exc)) from exc
    document = _object(decoded, label="SPDX source document")
    if spdx_canonical_json_bytes(document) != source_bytes:
        raise SpdxObservationError(
            "SPDX source must be compact sorted-key JSON with no final line feed"
        )
    return document, _validate_spdx_subset(document)


def _gguf_cross_binding(
    document: Mapping[str, object], artifact_identity_bytes: bytes
) -> dict[str, object]:
    try:
        identity = decode_artifact_identity(artifact_identity_bytes)
    except RuntimeProviderEvidenceError as exc:
        raise SpdxObservationError(str(exc)) from exc
    if not isinstance(identity, GGUFArtifactIdentity):
        raise SpdxObservationError("artifact identity must describe one GGUF artifact")
    if encode_artifact_identity(identity) != artifact_identity_bytes:
        raise SpdxObservationError("artifact identity must use canonical JSON bytes")

    graph = cast(list[dict[str, object]], document["@graph"])
    package = _one_by_type(graph, "ai_AIPackage")
    hashes = cast(list[dict[str, object]], package["verifiedUsing"])
    spdx_hash = cast(str, hashes[0]["hashValue"])
    if spdx_hash != identity.sha256:
        raise SpdxObservationError(
            "SPDX package SHA-256 does not match the InvarLock GGUF identity"
        )
    return {
        "binding_basis": "spdx.ai_AIPackage.verifiedUsing.Hash.sha256",
        "invarlock_artifact_content_digest": f"sha256:{identity.sha256}",
        "invarlock_artifact_identity_digest": sha256_digest(artifact_identity_bytes),
        "invarlock_artifact_name": identity.artifact_name,
        "spdx_package_id": package["spdxId"],
        "status": "matched",
    }


def build_observation_payload(
    source_bytes: bytes, artifact_identity_bytes: bytes
) -> dict[str, object]:
    """Build the example-owned wrapper preserved by EvidenceObservation."""

    document, checks = load_spdx_document(source_bytes)
    return {
        "artifact_cross_binding": _gguf_cross_binding(
            document, artifact_identity_bytes
        ),
        "document": document,
        "document_identity": {
            "byte_length": len(source_bytes),
            "canonicalization": ("utf8-json-sorted-keys-compact-no-final-line-feed"),
            "digest": sha256_digest(source_bytes),
            "media_type": "application/ld+json",
            "serialization": "spdx-3.0.1-json-ld",
        },
        "field_provenance": {
            "artifact_cross_binding": "computed_by_example_mapper",
            "document": "document_author_supplied_exact_bytes",
            "document_identity": "computed_from_exact_source_bytes",
            "profileConformance": (
                "document_author_declared_not_independently_validated"
            ),
        },
        "format": OBSERVATION_PAYLOAD_FORMAT,
        "interpretation_limits": [
            (
                "This payload is authenticated context and has no InvarLock "
                "acceptance authority."
            ),
            (
                "The cross-binding matches the SPDX document author's declared "
                "SHA-256 to the supplied canonical InvarLock GGUF identity; this "
                "example does not acquire or hash model bytes."
            ),
            ("Passing the bounded example checks is not an SPDX conformance claim."),
        ],
        "specification": {
            "context": SPDX_CONTEXT,
            "repository": SPDX_SPEC_REPOSITORY,
            "revision": SPDX_SPEC_COMMIT,
            "version": SPDX_SPEC_VERSION,
        },
        "validation": {
            "example_subset_checks": {
                "checks": list(checks),
                "status": "passed",
            },
            "official_json_schema": {
                "reason": (
                    "official SPDX JSON Schema validation is not run by this example"
                ),
                "status": "not_evaluated",
            },
            "owl_shacl_semantics": {
                "reason": (
                    "SPDX OWL/SHACL semantic validation is not run by this example"
                ),
                "status": "not_evaluated",
            },
            "spdx_profile_conformance": {
                "reason": (
                    "full structural and semantic validation is required before "
                    "claiming profile conformance"
                ),
                "status": "not_evaluated",
            },
        },
    }


def _read(path: Path, *, label: str) -> bytes:
    try:
        return read_regular_file_bytes(path, label=label, max_bytes=_MAX_SOURCE_BYTES)
    except StrictJsonError as exc:
        raise SpdxObservationError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--artifact-identity", type=Path, default=DEFAULT_ARTIFACT_IDENTITY
    )
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the rebuilt payload with the committed observation payload",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_observation_payload(
            _read(args.source, label="SPDX source document"),
            _read(args.artifact_identity, label="GGUF artifact identity"),
        )
        encoded = canonical_json_bytes(payload)
        if args.check:
            expected = _read(args.expected, label="expected observation payload")
            if expected != encoded:
                raise SpdxObservationError(
                    "committed observation payload does not match rebuilt bytes"
                )
            print(
                "SPDX AI observation example: PASS "
                f"({hashlib.sha256(encoded).hexdigest()})"
            )
        else:
            sys.stdout.buffer.write(encoded)
    except (OSError, SpdxObservationError) as exc:
        print(f"SPDX AI observation example: FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
