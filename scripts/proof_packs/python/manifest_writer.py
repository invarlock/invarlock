from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


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
        import invarlock  # type: ignore[import-not-found]

        version = getattr(invarlock, "__version__", "")
        return str(version) if isinstance(version, str) else ""
    except Exception:
        return ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_text(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


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
    payload = _load_metadata_object(pack_dir, "metadata/source_repo.json")
    reference = _file_reference(pack_dir, "metadata/source_repo.json")
    result: dict[str, Any] = {
        "uri": payload.get("uri") or "",
        "commit": payload.get("commit") or _git_text("rev-parse", "HEAD"),
        "branch": payload.get("branch")
        or _git_text("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": payload.get("describe")
        or _git_text("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(payload.get("dirty"))
        if payload
        else bool(_git_text("status", "--porcelain")),
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
    ):
        reference = _file_reference(pack_dir, rel_path, name=name)
        if reference is not None:
            materials.append(reference)
    return materials


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
        except Exception:
            checksums_digest = ""

    used_models: set[str] = set(model_list)
    for item in models:
        model_id = item.get("model_id")
        if model_id:
            used_models.add(model_id)
    model_licenses = _model_licenses_for(used_models)

    payload: dict[str, Any] = {
        "format": "proof-pack-v1",
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
            "id": "invarlock/proof-pack@v1",
            "name": "InvarLock Proof Pack Runner",
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

    if model_licenses:
        payload["model_licenses"] = model_licenses
    if isinstance(verification_summary, dict) and verification_summary:
        payload["verification"] = verification_summary

    out_path = pack_dir / "manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a proof-pack manifest.json")
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
    except Exception:
        repeats = 0
    write_manifest(
        pack_dir=Path(args.pack_dir),
        run_dir=Path(args.run_dir),
        suite=str(args.suite),
        net=str(args.net),
        determinism=str(args.determinism),
        repeats=repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
