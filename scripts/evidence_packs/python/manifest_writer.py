from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from source_repo_metadata import SourceRepoMetadataError, build_source_repo_payload

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017
_JSON_READ_ERRORS = (OSError, TypeError, ValueError)
_VERSION_READ_ERRORS = (AttributeError, ImportError, ModuleNotFoundError)
_CHECKSUM_ERRORS = (OSError, ValueError)
_COERCE_ERRORS = (TypeError, ValueError, OverflowError)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except _JSON_READ_ERRORS:
        return None


def _evidence_pack_counts_from_verification(
    verification: dict[str, Any] | None,
) -> tuple[int | None, int | None]:
    if not isinstance(verification, dict):
        return None, None
    clean_reports = verification.get("clean_reports")
    failed_reports = verification.get("failed_reports")
    return (
        int(clean_reports) if isinstance(clean_reports, int) else None,
        int(failed_reports) if isinstance(failed_reports, int) else None,
    )


def _derive_evidence_level(
    *,
    subject_present: bool,
    clean_reports: int | None,
    failed_reports: int | None,
    has_source_repo_ref: bool,
    has_environment_ref: bool,
) -> str:
    if (
        subject_present
        and isinstance(clean_reports, int)
        and clean_reports > 0
        and failed_reports == 0
        and has_source_repo_ref
        and has_environment_ref
    ):
        return "high"
    if subject_present and isinstance(clean_reports, int) and clean_reports > 0:
        return "medium"
    return "low"


def _collect_model_revisions(pack_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    revisions_path = pack_dir / "state" / "model_revisions.json"
    if not revisions_path.is_file():
        revisions_path = pack_dir / "metadata" / "model_revisions.json"
    if not revisions_path.is_file():
        return [], []

    data = _load_json(revisions_path)
    if not isinstance(data, dict):
        return [], []

    model_list_raw = data.get("model_list") or []
    model_list = [str(item) for item in model_list_raw if isinstance(item, str)]

    models: list[dict[str, str]] = []
    models_obj = data.get("models") or {}
    if isinstance(models_obj, dict):
        for model_id, info in models_obj.items():
            if not isinstance(model_id, str):
                continue
            if not isinstance(info, dict):
                info = {}
            revision = info.get("revision") or ""
            models.append({"model_id": model_id, "revision": str(revision)})

    return model_list, sorted(models, key=lambda item: item.get("model_id", ""))


def _collect_artifacts(pack_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for path in pack_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(pack_dir)
        if rel.name in {
            "manifest.json",
            "manifest.signature.json",
            "checksums.sha256",
        }:
            continue
        artifacts.append(str(rel))
    return sorted(artifacts)


def _sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_prefixed(path: Path) -> str:
    return f"sha256:{_sha256_hex(path)}"


def _maybe_get_invarlock_version() -> str:
    try:
        invarlock = importlib.import_module("invarlock")
        version = getattr(invarlock, "__version__", "")
        return str(version) if isinstance(version, str) else ""
    except _VERSION_READ_ERRORS:
        return ""


def _model_licenses_for(model_ids: set[str]) -> dict[str, str]:
    known_licenses = {
        "mistralai/Mistral-7B-v0.1": "Apache-2.0",
    }
    return {mid: lic for mid, lic in known_licenses.items() if mid in model_ids}


def _first_file(pack_dir: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        path = pack_dir / candidate
        if path.is_file():
            return path
    return None


def _load_metadata_object(pack_dir: Path, rel_path: str) -> dict[str, Any]:
    payload = _load_json(pack_dir / rel_path)
    return payload if isinstance(payload, dict) else {}


def _load_required_metadata_object(
    pack_dir: Path, rel_path: str
) -> dict[str, Any] | None:
    path = pack_dir / rel_path
    if not path.is_file():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{rel_path} must contain a JSON object")
    return payload


def _file_reference(
    pack_dir: Path, rel_path: str, *, name: str | None = None
) -> dict[str, Any] | None:
    path = pack_dir / rel_path
    if not path.is_file():
        return None
    payload: dict[str, Any] = {
        "path": rel_path,
        "digest": _sha256_prefixed(path),
    }
    if name:
        payload["name"] = name
    return payload


def _subject(pack_dir: Path) -> dict[str, Any] | None:
    path = _first_file(
        pack_dir,
        "results/verdicts/final_verdict.json",
        "results/final_verdict.json",
    )
    if path is None:
        return None
    return {
        "name": "final_verdict",
        "path": str(path.relative_to(pack_dir)),
        "digest": _sha256_prefixed(path),
    }


def _config_source(pack_dir: Path) -> dict[str, Any]:
    payload = _load_required_metadata_object(pack_dir, "metadata/source_repo.json")
    reference = _file_reference(pack_dir, "metadata/source_repo.json")
    if payload is None:
        payload = build_source_repo_payload()
    uri = payload.get("uri")
    commit = payload.get("commit")
    branch = payload.get("branch")
    describe = payload.get("describe")
    dirty = payload.get("dirty")
    if not isinstance(uri, str) or not uri.strip():
        raise RuntimeError("source repo provenance must include a non-empty uri")
    if not isinstance(commit, str) or not commit.strip():
        raise RuntimeError("source repo provenance must include a non-empty commit")
    if not isinstance(branch, str) or not branch.strip():
        raise RuntimeError("source repo provenance must include a non-empty branch")
    if not isinstance(describe, str) or not describe.strip():
        raise RuntimeError("source repo provenance must include a non-empty describe")
    if not isinstance(dirty, bool):
        raise RuntimeError("source repo provenance must include a boolean dirty flag")
    result: dict[str, Any] = {
        "uri": uri,
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }
    if reference is not None:
        result.update(reference)
    return result


def _dataset_provider_parameters() -> dict[str, Any]:
    kind = str(os.environ.get("INVARLOCK_DATASET", "")).strip() or "wikitext2"
    payload: dict[str, Any] = {"kind": kind}
    if kind == "hf_text":
        dataset_name = (
            os.environ.get("INVARLOCK_HF_DATASET_NAME")
            or os.environ.get("INVARLOCK_HF_DATASET")
            or "allenai/c4"
        )
        if dataset_name == "c4":
            dataset_name = "allenai/c4"
        payload["dataset_name"] = dataset_name
        config_name = (
            os.environ.get("INVARLOCK_HF_CONFIG_NAME")
            or os.environ.get("INVARLOCK_HF_DATASET_CONFIG_NAME")
            or ""
        )
        if config_name:
            payload["config_name"] = config_name
    elif kind == "local_jsonl":
        for key in (
            "INVARLOCK_LOCAL_JSONL_FILE",
            "INVARLOCK_LOCAL_JSONL_PATH",
            "INVARLOCK_LOCAL_JSONL_DATA_FILES",
        ):
            value = os.environ.get(key, "").strip()
            if value:
                payload[key.removeprefix("INVARLOCK_LOCAL_JSONL_").lower()] = value
    return payload


def _environment(pack_dir: Path) -> dict[str, Any] | None:
    payload = _load_metadata_object(pack_dir, "metadata/environment.json")
    reference = _file_reference(pack_dir, "metadata/environment.json")
    if not payload and reference is None:
        return None
    result: dict[str, Any] = {}
    if reference is not None:
        result.update(reference)
    for field in (
        "recorded_at",
        "platform",
        "python_version",
        "gpu_name",
        "gpu_count",
        "gpu_memory_gb",
        "fp8_native_support",
    ):
        if field in payload:
            result[field] = payload[field]
    return result


def _materials(pack_dir: Path) -> list[dict[str, Any]]:
    materials: list[dict[str, Any]] = []
    for rel_path, name in (
        ("metadata/model_revisions.json", "model_revisions"),
        ("metadata/scenarios.json", "scenarios"),
        ("metadata/tuned_edit_params.json", "tuned_edit_params"),
        ("results/analysis/edit_artifact_summary.json", "edit_artifact_summary"),
    ):
        reference = _file_reference(pack_dir, rel_path, name=name)
        if reference is not None:
            materials.append(reference)
    return materials


def _edit_artifact_summary(pack_dir: Path) -> dict[str, Any]:
    payload = _load_json(
        pack_dir / "results" / "analysis" / "edit_artifact_summary.json"
    )
    return payload if isinstance(payload, dict) else {}


def _scenario_ids() -> list[str]:
    raw = str(os.environ.get("PACK_SCENARIO_IDS", ""))
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_manifest(
    *,
    pack_dir: Path,
    run_dir: Path,
    suite: str,
    net: str,
    determinism: str,
    repeats: int,
) -> None:
    model_list, models = _collect_model_revisions(pack_dir)

    determinism_repeats = None
    det_path = pack_dir / "results" / "determinism_repeats.json"
    if det_path.is_file():
        determinism_repeats = _load_json(det_path)

    verification_summary = None
    verification_path = pack_dir / "results" / "verification_summary.json"
    if verification_path.is_file():
        verification_summary = _load_json(verification_path)

    artifacts = _collect_artifacts(pack_dir)

    checksums_digest = ""
    checksums_path = pack_dir / "checksums.sha256"
    if checksums_path.is_file():
        try:
            checksums_digest = _sha256_hex(checksums_path)
        except _CHECKSUM_ERRORS:
            checksums_digest = ""

    used_models: set[str] = set(model_list)
    for item in models:
        model_id = item.get("model_id")
        if model_id:
            used_models.add(model_id)
    model_licenses = _model_licenses_for(used_models)

    payload: dict[str, Any] = {
        "format": "evidence-pack-v1",
        "generated_at": dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite": suite,
        "network_mode": "online"
        if str(net) in {"1", "true", "yes", "on"}
        else "offline",
        "determinism": determinism,
        "repeats": repeats,
        "determinism_repeats": determinism_repeats,
        "run_dir": str(run_dir),
        "invarlock_version": _maybe_get_invarlock_version(),
        "model_list": model_list,
        "models": models,
        "artifacts": artifacts,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": checksums_digest,
        "builder": {
            "id": "invarlock/evidence-pack@v1",
            "name": "InvarLock Evidence Pack Runner",
        },
        "invocation": {
            "config_source": _config_source(pack_dir),
            "parameters": {
                "suite": suite,
                "network_mode": "online"
                if str(net) in {"1", "true", "yes", "on"}
                else "offline",
                "determinism": determinism,
                "repeats": repeats,
                "scenario_ids": _scenario_ids(),
                "dataset_provider": _dataset_provider_parameters(),
            },
        },
        "materials": _materials(pack_dir),
    }

    edit_summary = _edit_artifact_summary(pack_dir)
    if edit_summary:
        lanes = edit_summary.get("evidence_lanes")
        if isinstance(lanes, dict):
            payload["evidence_lanes"] = lanes
        deployable = edit_summary.get("deployable_subjects")
        if isinstance(deployable, dict):
            payload["deployable_subjects"] = deployable
        counts = edit_summary.get("counts")
        if isinstance(counts, dict):
            payload["artifact_class_counts"] = counts

    if payload["builder"]["id"] and payload["builder"]["name"]:
        version = payload.get("invarlock_version") or ""
        if isinstance(version, str) and version:
            payload["builder"]["version"] = version

    subject = _subject(pack_dir)
    if subject is not None:
        payload["subject"] = subject

    environment = _environment(pack_dir)
    if environment is not None:
        payload["environment"] = environment

    clean_reports, failed_reports = _evidence_pack_counts_from_verification(
        verification_summary if isinstance(verification_summary, dict) else None
    )
    payload["evidence_level"] = _derive_evidence_level(
        subject_present=subject is not None,
        clean_reports=clean_reports,
        failed_reports=failed_reports,
        has_source_repo_ref=bool(
            isinstance(payload.get("invocation"), dict)
            and isinstance(payload["invocation"].get("config_source"), dict)
            and payload["invocation"]["config_source"].get("path")
            and payload["invocation"]["config_source"].get("digest")
        ),
        has_environment_ref=bool(
            isinstance(environment, dict)
            and environment.get("path")
            and environment.get("digest")
        ),
    )

    if model_licenses:
        payload["model_licenses"] = model_licenses
    if isinstance(verification_summary, dict) and verification_summary:
        payload["verification"] = verification_summary

    out_path = pack_dir / "manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an evidence-pack manifest.json")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--determinism", required=True)
    parser.add_argument("--repeats", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repeats = int(args.repeats)
    except _COERCE_ERRORS:
        repeats = 0
    try:
        write_manifest(
            pack_dir=Path(args.pack_dir),
            run_dir=Path(args.run_dir),
            suite=str(args.suite),
            net=str(args.net),
            determinism=str(args.determinism),
            repeats=repeats,
        )
    except (RuntimeError, SourceRepoMetadataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
