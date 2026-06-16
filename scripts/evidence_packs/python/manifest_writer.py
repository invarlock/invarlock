from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    from scripts.evidence_packs.python import manifest_writer_provenance as _provenance
except ImportError:  # pragma: no cover - direct script execution path
    import manifest_writer_provenance as _provenance

try:
    from scripts.evidence_workflows.workflow_state import (
        collect_artifact_paths,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution path
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2]
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from evidence_workflows.workflow_state import (
        collect_artifact_paths,
        sha256_file,
        write_json,
    )

SourceRepoMetadataError = _provenance.SourceRepoMetadataError
_ensure_repo_src_path = _provenance._ensure_repo_src_path
_git_text = _provenance._git_text
_load_run_state_environment = _provenance._load_run_state_environment
_maybe_number = _provenance._maybe_number
_repo_root = _provenance._repo_root
_snapshot_marker_payload = _provenance._snapshot_marker_payload
_truthy = _provenance._truthy
_utc_now = _provenance._utc_now
build_environment_payload = _provenance.build_environment_payload
build_source_repo_payload = _provenance.build_source_repo_payload
write_environment_metadata = _provenance.write_environment_metadata
write_source_repo_metadata = _provenance.write_source_repo_metadata

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
    return collect_artifact_paths(
        pack_dir,
        patterns=["**/*"],
        exclude_names={
            "manifest.json",
            "manifest.signature.json",
            "checksums.sha256",
        },
    )


def _sha256_hex(path: Path) -> str:
    return sha256_file(path)


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
        "generated_at": _utc_now(),
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
    write_json(out_path, payload)


def _render_pack_readme(
    *,
    evidence_level: str,
    clean_reports: int | None,
    error_reports: int | None,
    failed_reports: int | None,
    policy_profile: str | None,
) -> str:
    lines = [
        "# InvarLock Evidence Pack",
        "",
        "This evidence pack bundles reports, summary reports, and metadata for offline",
        "verification. No model weights are included.",
        "",
        f"Evidence level: {evidence_level}",
        (
            "Review summary: "
            f"clean_reports={clean_reports if clean_reports is not None else 'unknown'}, "
            f"error_injection_reports={error_reports if error_reports is not None else 'unknown'}, "
            f"failed_reports={failed_reports if failed_reports is not None else 'unknown'}, "
            f"profile={policy_profile or 'unknown'}."
        ),
        "",
        "Why it might be wrong:",
    ]
    if failed_reports not in (None, 0):
        lines.append(
            "- Unexpected report verification failures were recorded; inspect results/verification_summary.json before trusting final conclusions."
        )
    else:
        lines.append(
            "- Nested report verification succeeded for the bundled clean reports, but reviewers should still inspect the underlying evaluation.report.json files."
        )
    lines.extend(
        [
            "- Error-injection reports are expected-failure evidence and should not be interpreted as clean PASS runs.",
            "- Current validation edit artifacts are checkpoint-shaped subjects, not optimized deployment backends; inspect results/analysis/edit_artifact_summary.json and report-local edit_metadata.json sidecars.",
            "- By default this is evidence-grade packaging. For strong distributable evidence, require a signed manifest, strict verification, and a PASS final verdict.",
            "",
            "## Verify",
            "",
            "1) Verify the manifest signature (if present):",
            "   invarlock advanced evidence-pack verify <pack-dir> --strict --report-assurance strict",
            "",
            "2) Verify file checksums:",
            "   sha256sum -c checksums.sha256",
            "   # macOS: shasum -a 256 -c checksums.sha256",
            "",
            "3) Verify report integrity:",
            "   invarlock verify --json reports/**/evaluation.report.json",
            "",
            "Or use:",
            "  invarlock advanced evidence-pack verify <pack-dir>",
            "  invarlock advanced evidence-pack verify <pack-dir> --strict --report-assurance strict",
            "Repo workflow alternative:",
            "  scripts/evidence_packs/verify_pack.sh --pack <pack-dir> --strict --report-assurance strict",
        ]
    )
    return "\n".join(lines) + "\n"


def _verification_int(verification: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(verification, dict):
        return None
    value = verification.get(key)
    return int(value) if isinstance(value, int) else None


def write_pack_readme(pack_dir: Path) -> None:
    verification = _load_json(pack_dir / "results" / "verification_summary.json")
    verification_obj = verification if isinstance(verification, dict) else None
    clean_reports = _verification_int(verification_obj, "clean_reports")
    error_reports = _verification_int(verification_obj, "error_injection_reports")
    failed_reports = _verification_int(verification_obj, "failed_reports")
    policy_profile = (
        str(verification_obj.get("policy_profile"))
        if isinstance(verification_obj, dict)
        and isinstance(verification_obj.get("policy_profile"), str)
        else None
    )
    evidence_level = _derive_evidence_level(
        subject_present=(pack_dir / "results" / "final_verdict.json").is_file(),
        clean_reports=clean_reports,
        failed_reports=failed_reports,
        has_source_repo_ref=(pack_dir / "metadata" / "source_repo.json").is_file(),
        has_environment_ref=(pack_dir / "metadata" / "environment.json").is_file(),
    )
    (pack_dir / "README.md").write_text(
        _render_pack_readme(
            evidence_level=evidence_level,
            clean_reports=clean_reports,
            error_reports=error_reports,
            failed_reports=failed_reports,
            policy_profile=policy_profile,
        ),
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an evidence-pack manifest.json")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--determinism", required=True)
    parser.add_argument("--repeats", default="0")
    return parser.parse_args(argv)


def _parse_source_repo_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write evidence-pack source repository metadata."
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _parse_environment_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write evidence-pack environment metadata."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir")
    return parser.parse_args(argv)


def _parse_readme_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an evidence-pack README.md.")
    parser.add_argument("pack_dir")
    return parser.parse_args(argv)


def _parse_sign_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign an evidence-pack manifest with a package-native Ed25519 key."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.json.")
    parser.add_argument(
        "--signing-key",
        help="Optional Ed25519 private key PEM. When omitted, an ephemeral key is generated.",
    )
    parser.add_argument(
        "--signature-out",
        help="Optional output path for manifest.signature.json.",
    )
    parser.add_argument(
        "--generate-ephemeral",
        action="store_true",
        help="Generate an ephemeral Ed25519 key when --signing-key is omitted.",
    )
    return parser.parse_args(argv)


def _load_manifest_for_signing(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must decode to a JSON object")
    return payload


def _sign_with_key(
    manifest_path: Path,
    *,
    signing_key_path: Path,
    signature_path: Path | None,
) -> str:
    _ensure_repo_src_path()
    from invarlock.evidence_pack_integrity import (
        load_private_signing_key,
        public_key_fingerprint,
        sign_manifest,
    )

    fingerprint = public_key_fingerprint(
        load_private_signing_key(signing_key_path).public_key()
    )
    payload = _load_manifest_for_signing(manifest_path)
    payload["signing_key_fingerprint"] = fingerprint
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sign_manifest(
        manifest_path,
        signing_key_path=signing_key_path,
        signature_path=signature_path,
    )
    return fingerprint


def sign_manifest_command(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    signature_path = Path(args.signature_out) if args.signature_out else None

    if args.signing_key:
        fingerprint = _sign_with_key(
            manifest_path,
            signing_key_path=Path(args.signing_key),
            signature_path=signature_path,
        )
        print(fingerprint)
        return 0

    if not args.generate_ephemeral:
        print(
            "either --signing-key or --generate-ephemeral is required",
            file=sys.stderr,
        )
        return 2

    _ensure_repo_src_path()
    from invarlock.evidence_pack_integrity import generate_signing_keypair

    with TemporaryDirectory(prefix="invarlock-evidence-pack-signing-") as tmp_dir:
        private_key_path = Path(tmp_dir) / "ephemeral-signing-key.pem"
        public_key_path = Path(tmp_dir) / "ephemeral-signing-key.pub.pem"
        generate_signing_keypair(
            private_key_path,
            public_key_path=public_key_path,
        )
        fingerprint = _sign_with_key(
            manifest_path,
            signing_key_path=private_key_path,
            signature_path=signature_path,
        )
    print(fingerprint)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "source-repo":
        args = _parse_source_repo_args(argv[1:])
        try:
            write_source_repo_metadata(Path(args.out))
        except SourceRepoMetadataError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0
    if argv and argv[0] == "environment":
        args = _parse_environment_args(argv[1:])
        run_dir = Path(args.run_dir) if args.run_dir else None
        write_environment_metadata(out_path=Path(args.out), run_dir=run_dir)
        return 0
    if argv and argv[0] == "readme":
        args = _parse_readme_args(argv[1:])
        write_pack_readme(Path(args.pack_dir))
        return 0
    if argv and argv[0] == "sign":
        args = _parse_sign_args(argv[1:])
        return sign_manifest_command(args)

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
