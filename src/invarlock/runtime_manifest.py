"""Canonical runtime manifest construction and provider-sidecar binding."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invarlock.runtime_security_helpers import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
    _runtime_provenance_image_ref,
    _sha256_bytes,
    _sha256_path,
    current_execution_mode,
    network_allowed,
    remote_code_allowed,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    running_inside_container,
    serialize_canonical_json,
    third_party_plugins_allowed,
)


def _sibling_reference(
    path: str | os.PathLike[str],
    *,
    report_directory: Path,
    label: str,
) -> dict[str, str]:
    candidate = Path(path).resolve()
    if candidate.parent != report_directory:
        raise ValueError(f"{label} must be a sibling of the runtime report")
    return {"filename": candidate.name, "sha256": _sha256_path(candidate)}


def _config_reference(
    *,
    report_directory: Path,
    config_path: str | os.PathLike[str] | None,
    config_payload: Any | None,
) -> dict[str, Any]:
    if config_path is not None:
        candidate = Path(config_path).resolve()
        if candidate.parent != report_directory:
            raise ValueError("config_path must be a sibling of the runtime report")
        return {
            "path": candidate.name,
            "sha256": _sha256_path(candidate),
            "source": "file",
        }
    if config_payload is not None:
        payload = serialize_canonical_json(config_payload).encode("utf-8")
        return {"path": None, "sha256": _sha256_bytes(payload), "source": "inline"}
    return {"path": None, "sha256": None, "source": "missing"}


def _execution_or_current(
    execution: RuntimeManifestExecution | None,
) -> RuntimeManifestExecution:
    return execution or RuntimeManifestExecution(
        execution_mode=current_execution_mode(),
        container_execution=running_inside_container(),
        image_ref=resolve_runtime_image(),
        image_digest=resolve_runtime_image_digest(),
        allow_network=network_allowed(),
        allow_remote_code=remote_code_allowed(),
        allow_third_party_plugins=third_party_plugins_allowed(),
    )


def write_runtime_manifest(
    report_path: str | os.PathLike[str],
    *,
    provider_files: RuntimeProviderManifestFiles,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
    execution: RuntimeManifestExecution | None = None,
    generated_at_utc: str | None = None,
) -> Path:
    """Write the one closed manifest binding runtime and provider evidence."""

    report = Path(report_path).resolve()
    runtime_execution = _execution_or_current(execution)
    if (
        runtime_execution.execution_mode != "container"
        or runtime_execution.container_execution is not True
    ):
        raise ValueError("runtime manifest requires container execution")
    image_digest = runtime_execution.image_digest
    if (
        not isinstance(image_digest, str)
        or not image_digest.startswith("sha256:")
        or len(image_digest) != 71
        or any(character not in "0123456789abcdef" for character in image_digest[7:])
    ):
        raise ValueError("runtime manifest requires a lowercase image digest")

    provider_paths = (
        Path(provider_files.receipt).resolve(),
        Path(provider_files.scoring_observation).resolve(),
        Path(provider_files.artifact_identity).resolve(),
    )
    if len(set(provider_paths)) != len(provider_paths):
        raise ValueError("runtime provider binding files must be distinct")

    manifest_path = report.parent / RUNTIME_MANIFEST_FILENAME
    reserved_paths = {report, manifest_path.resolve()}
    if config_path is not None:
        reserved_paths.add(Path(config_path).resolve())
    if reserved_paths.intersection(provider_paths):
        raise ValueError(
            "runtime provider binding files must be distinct from the report, "
            "manifest, and file config"
        )
    if config_path is not None and Path(config_path).resolve() in {
        report,
        manifest_path.resolve(),
    }:
        raise ValueError("file config must be distinct from the report and manifest")

    if generated_at_utc is None:
        generated_at = datetime.now(UTC).isoformat()
    else:
        try:
            parsed = datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("generated_at_utc must be an ISO 8601 timestamp") from exc
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None:
            raise ValueError("generated_at_utc must include a UTC offset")
        if offset.total_seconds() != 0:
            raise ValueError("generated_at_utc must use UTC")
        generated_at = generated_at_utc

    manifest: dict[str, Any] = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": generated_at,
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": report.name,
            "filename": report.name,
            "sha256": _sha256_path(report),
        },
        "config": _config_reference(
            report_directory=report.parent,
            config_path=config_path,
            config_payload=config_payload,
        ),
        "execution_mode": "container",
        "outer_container": {
            "image_ref": _runtime_provenance_image_ref(
                runtime_execution.image_ref,
                image_digest,
            ),
            "image_digest": image_digest,
            "container_execution": True,
            "allow_network": runtime_execution.allow_network,
            "allow_remote_code": runtime_execution.allow_remote_code,
            "allow_third_party_plugins": runtime_execution.allow_third_party_plugins,
        },
        "runtime_provider": {
            "receipt": _sibling_reference(
                provider_files.receipt,
                report_directory=report.parent,
                label="runtime provider receipt",
            ),
            "scoring_observation": _sibling_reference(
                provider_files.scoring_observation,
                report_directory=report.parent,
                label="runtime scoring observation",
            ),
            "artifact_identity": _sibling_reference(
                provider_files.artifact_identity,
                report_directory=report.parent,
                label="model artifact identity",
            ),
        },
    }
    try:
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except FileExistsError as exc:
        raise ValueError("runtime manifest destination must not already exist") from exc
    except OSError as exc:
        raise ValueError("runtime manifest could not be written safely") from exc
    return manifest_path


__all__ = ["write_runtime_manifest"]
