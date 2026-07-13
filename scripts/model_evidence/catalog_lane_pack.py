"""Assemble one verified catalog lane into a signed evidence pack."""

from __future__ import annotations

import base64
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_catalog import entry_digest, load_evidence_catalog
from invarlock.evidence_pack_integrity import (
    EVIDENCE_PACK_SIGNATURE_FORMAT,
    public_key_fingerprint,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)

_COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")
_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")


class CatalogLaneError(RuntimeError):
    """Raised when one catalog lane cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class CatalogLaneArtifacts:
    catalog: Path
    lane_id: str
    evaluation_report: Path
    runtime_manifest: Path
    baseline_report: Path
    policy_pack: Path
    resolved_inputs: Path
    resolved_config: Path
    preset: Path
    evaluation_input_binding: Path
    verification_receipt: Path
    source_commit: str
    source_bundle_sha256: str
    input_materialization: Path | None = None
    expected_runtime_image_digest: str | None = None
    network_mode: str = "offline"


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        _raw, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise CatalogLaneError(str(exc)) from exc
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", errors="strict") as handle:
            handle.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        raise CatalogLaneError(f"could not write {path.name}: {exc}") from exc


def _copy_regular(source: Path, destination: Path, *, label: str) -> None:
    try:
        payload = read_regular_file_bytes(source, label=label)
    except StrictJsonError as exc:
        raise CatalogLaneError(str(exc)) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except OSError as exc:
        raise CatalogLaneError(f"could not stage {label}: {exc}") from exc


def _digest(path: Path) -> str:
    try:
        return sha256_prefixed(read_regular_file_bytes(path, label=path.name))
    except StrictJsonError as exc:
        raise CatalogLaneError(str(exc)) from exc


def _verification_passed(path: Path) -> None:
    receipt = _read_object(path, label="strict report verification receipt")
    summary = receipt.get("summary")
    results = receipt.get("results")
    if (
        receipt.get("format_version") != "verify-v1"
        or not isinstance(summary, Mapping)
        or summary.get("ok") is not True
        or not isinstance(results, list)
        or not results
        or any(
            not isinstance(item, Mapping) or item.get("ok") is not True
            for item in results
        )
    ):
        raise CatalogLaneError("strict report verification did not pass")


def _report_run_id(report: Mapping[str, object]) -> str | None:
    meta = report.get("meta")
    run_id = meta.get("run_id") if isinstance(meta, Mapping) else None
    return run_id.strip() if isinstance(run_id, str) and run_id.strip() else None


def _load_signing_key(path: Path) -> ed25519.Ed25519PrivateKey:
    try:
        key_bytes = read_regular_file_bytes(path, label="evidence-pack signing key")
        key = serialization.load_pem_private_key(key_bytes, password=None)
    except (StrictJsonError, TypeError, ValueError) as exc:
        raise CatalogLaneError(
            f"could not load evidence-pack signing key: {exc}"
        ) from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise CatalogLaneError("evidence-pack signing key must be Ed25519")
    return key


def _write_signature(
    manifest_path: Path,
    *,
    private_key: ed25519.Ed25519PrivateKey,
) -> str:
    public_key = private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    manifest_bytes = manifest_path.read_bytes()
    signature = private_key.sign(manifest_bytes)
    bundle = {
        "format": EVIDENCE_PACK_SIGNATURE_FORMAT,
        "algorithm": "ed25519",
        "signing_key_fingerprint": fingerprint,
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    _write_json(manifest_path.with_name("manifest.signature.json"), bundle)
    return fingerprint


def _stage_materials(
    pack_dir: Path,
    artifacts: CatalogLaneArtifacts,
) -> dict[str, str]:
    sources = {
        "reports/report-001/evaluation.report.json": artifacts.evaluation_report,
        "reports/report-001/runtime.manifest.json": artifacts.runtime_manifest,
        "baselines/baseline-001/evaluation.report.json": artifacts.baseline_report,
        "policy/policy-pack.json": artifacts.policy_pack,
        "metadata/catalog.json": artifacts.catalog,
        "metadata/resolved-inputs.json": artifacts.resolved_inputs,
        "metadata/runtime-config.yaml": artifacts.resolved_config,
        "metadata/preset.yaml": artifacts.preset,
        "metadata/evaluation-input-binding.json": artifacts.evaluation_input_binding,
    }
    if artifacts.input_materialization is not None:
        sources["metadata/input-materialization.json"] = artifacts.input_materialization
    for relative, source in sources.items():
        _copy_regular(source, pack_dir / relative, label=relative)

    source_payload = {
        "format_version": "invarlock/source-provenance-v1",
        "commit": artifacts.source_commit,
        "source_bundle_sha256": artifacts.source_bundle_sha256,
        "dirty": False,
    }
    _write_json(pack_dir / "metadata/source_repo.json", source_payload)
    return {relative: _digest(pack_dir / relative) for relative in sources}


def _write_verdict(pack_dir: Path) -> None:
    report_path = pack_dir / "reports/report-001/evaluation.report.json"
    report = _read_object(report_path, label="evaluation report")
    verdict: dict[str, object] = {
        "verdict": "PASS",
        "report_path": "reports/report-001/evaluation.report.json",
        "report_sha256": _digest(report_path).removeprefix("sha256:"),
    }
    run_id = _report_run_id(report)
    if run_id is not None:
        verdict["run_id"] = run_id
    _write_json(pack_dir / "results/final_verdict.json", verdict)


def _write_scenario_contract(pack_dir: Path) -> None:
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "schema": "evidence_pack_scenarios_v1",
            "schema_version": 1,
            "scenarios": [
                {
                    "id": "report-001",
                    "strictness": "must_pass",
                    "intent": "submitted_report",
                    "primary_guard": "primary_metric",
                    "artifact_class": "evidence_only_pack",
                    "generation": {"kind": "evidence_only"},
                }
            ],
        },
    )


def _write_checksums(pack_dir: Path) -> None:
    relative_paths = sorted(
        path.relative_to(pack_dir).as_posix()
        for path in pack_dir.rglob("*")
        if path.is_file()
    )
    lines = [
        f"{_digest(pack_dir / relative).removeprefix('sha256:')}  {relative}"
        for relative in relative_paths
    ]
    try:
        (pack_dir / "checksums.sha256").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", errors="strict"
        )
    except OSError as exc:
        raise CatalogLaneError(f"could not write checksums: {exc}") from exc


def _manifest_materials(pack_dir: Path, *, vision: bool) -> list[dict[str, str]]:
    named_paths = [
        ("catalog", "metadata/catalog.json"),
        ("resolved-inputs", "metadata/resolved-inputs.json"),
        ("runtime-config", "metadata/runtime-config.yaml"),
        ("preset", "metadata/preset.yaml"),
        ("evaluation-input-binding", "metadata/evaluation-input-binding.json"),
        ("scenarios", "metadata/scenarios.json"),
    ]
    if vision:
        named_paths.append(
            ("input-materialization", "metadata/input-materialization.json")
        )
    return [
        {"name": name, "path": relative, "digest": _digest(pack_dir / relative)}
        for name, relative in named_paths
    ]


def _write_manifest(
    pack_dir: Path,
    *,
    artifacts: CatalogLaneArtifacts,
    signing_fingerprint: str,
) -> None:
    catalog = load_evidence_catalog(pack_dir / "metadata/catalog.json")
    entry = catalog.entries.get(artifacts.lane_id)
    if entry is None:
        raise CatalogLaneError("catalog lane is not present")
    inputs = entry.get("inputs")
    vision = isinstance(inputs, Mapping) and inputs.get("kind") == "vision_text"
    policy = _read_object(pack_dir / "policy/policy-pack.json", label="policy pack")
    policy_digest = policy.get("policy_digest")
    if not isinstance(policy_digest, str) or not policy_digest:
        raise CatalogLaneError("policy pack has no policy_digest")
    checksums = pack_dir / "checksums.sha256"
    manifest = {
        "format": "evidence-pack-v1",
        "evidence_level": "high",
        "network_mode": artifacts.network_mode,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _digest(checksums).removeprefix("sha256:"),
        "signing_key_fingerprint": signing_fingerprint,
        "subject": {
            "name": "final_verdict",
            "path": "results/final_verdict.json",
            "digest": _digest(pack_dir / "results/final_verdict.json"),
        },
        "invocation": {
            "config_source": {
                "path": "metadata/source_repo.json",
                "digest": _digest(pack_dir / "metadata/source_repo.json"),
            }
        },
        "materials": _manifest_materials(pack_dir, vision=vision),
        "verification": {
            "clean_reports": 1,
            "error_injection_reports": 0,
            "failed_reports": 0,
            "report_assurance": "strict",
            "subject_mode": "catalog_bound_noop",
        },
        "verification_baselines": [
            {
                "name": "baseline-001",
                "path": "baselines/baseline-001/evaluation.report.json",
                "digest": _digest(
                    pack_dir / "baselines/baseline-001/evaluation.report.json"
                ),
                "report_paths": ["reports/report-001/evaluation.report.json"],
            }
        ],
        "verification_policy_pack": {
            "path": "policy/policy-pack.json",
            "digest": _digest(pack_dir / "policy/policy-pack.json"),
            "policy_digest": policy_digest,
        },
        "catalog": {
            "path": "metadata/catalog.json",
            "digest": catalog.digest,
            "entry_id": artifacts.lane_id,
            "entry_digest": entry_digest(entry),
        },
    }
    _write_json(pack_dir / "manifest.json", manifest)


def _validate_artifact_contract(artifacts: CatalogLaneArtifacts) -> None:
    if _COMMIT_RE.fullmatch(artifacts.source_commit) is None:
        raise CatalogLaneError(
            "source commit must be 40 lowercase hexadecimal characters"
        )
    if _DIGEST_RE.fullmatch(artifacts.source_bundle_sha256) is None:
        raise CatalogLaneError("source bundle digest must be sha256:<64 lowercase hex>")
    if artifacts.network_mode not in {"offline", "online"}:
        raise CatalogLaneError("network mode must be offline or online")
    _verification_passed(artifacts.verification_receipt)
    catalog = load_evidence_catalog(artifacts.catalog)
    entry = catalog.entries.get(artifacts.lane_id)
    if entry is None:
        raise CatalogLaneError("catalog lane is not present")
    inputs = entry.get("inputs")
    vision = isinstance(inputs, Mapping) and inputs.get("kind") == "vision_text"
    if vision != (artifacts.input_materialization is not None):
        raise CatalogLaneError(
            "vision lanes require exactly one input materialization artifact"
        )
    runtime = _read_object(artifacts.runtime_manifest, label="runtime manifest")
    runtime_section = runtime.get("runtime")
    observed_image = (
        runtime_section.get("image_digest")
        if isinstance(runtime_section, Mapping)
        else None
    )
    expected_image = artifacts.expected_runtime_image_digest
    if expected_image is not None and observed_image != expected_image:
        raise CatalogLaneError(
            "runtime manifest image digest does not match expected input"
        )


def assemble_signed_catalog_pack(
    artifacts: CatalogLaneArtifacts,
    out_dir: Path,
    *,
    signing_key: Path,
) -> tuple[Path, str]:
    """Create one signed pack without publishing it or mutating the evidence index."""

    _validate_artifact_contract(artifacts)
    if out_dir.exists() or out_dir.is_symlink():
        raise CatalogLaneError(f"output already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    pack_dir = temporary_root / "pack"
    pack_dir.mkdir()
    try:
        _stage_materials(pack_dir, artifacts)
        _write_verdict(pack_dir)
        _write_scenario_contract(pack_dir)
        _write_checksums(pack_dir)
        private_key = _load_signing_key(signing_key)
        fingerprint = public_key_fingerprint(private_key.public_key())
        _write_manifest(
            pack_dir,
            artifacts=artifacts,
            signing_fingerprint=fingerprint,
        )
        observed_fingerprint = _write_signature(
            pack_dir / "manifest.json", private_key=private_key
        )
        if observed_fingerprint != fingerprint:
            raise CatalogLaneError("signing fingerprint changed during pack assembly")
        pack_dir.replace(out_dir)
        return out_dir, fingerprint
    except BaseException:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = [
    "CatalogLaneArtifacts",
    "CatalogLaneError",
    "assemble_signed_catalog_pack",
]
