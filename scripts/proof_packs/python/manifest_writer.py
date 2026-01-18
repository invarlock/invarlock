from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        if rel.name in {"manifest.json", "manifest.json.asc", "checksums.sha256"}:
            continue
        artifacts.append(str(rel))
    return sorted(artifacts)


def _maybe_get_invarlock_version() -> str:
    try:
        import invarlock  # type: ignore[import-not-found]

        version = getattr(invarlock, "__version__", "")
        return str(version) if isinstance(version, str) else ""
    except Exception:
        return ""


def _model_licenses_for(model_ids: set[str]) -> dict[str, str]:
    known_licenses = {
        "mistralai/Mistral-7B-v0.1": "Apache-2.0",
    }
    return {mid: lic for mid, lic in known_licenses.items() if mid in model_ids}


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

    used_models: set[str] = set(model_list)
    for item in models:
        model_id = item.get("model_id")
        if model_id:
            used_models.add(model_id)
    model_licenses = _model_licenses_for(used_models)

    payload: dict[str, Any] = {
        "format": "proof-pack-v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    }

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
