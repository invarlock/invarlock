from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOW_HOST_EXECUTION_ENV = "INVARLOCK_ALLOW_HOST_EXECUTION"
ALLOW_NETWORK_ENV = "INVARLOCK_ALLOW_NETWORK"
ALLOW_REMOTE_CODE_ENV = "INVARLOCK_ALLOW_REMOTE_CODE"
ALLOW_THIRD_PARTY_PLUGINS_ENV = "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"
ALLOW_UNATTESTED_ARTIFACTS_ENV = "INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS"
CONTAINER_EXECUTION_ENV = "INVARLOCK_CONTAINER_EXECUTION"
RUNTIME_IMAGE_ENV = "INVARLOCK_RUNTIME_IMAGE"
RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_RUNTIME_IMAGE_DIGEST"
RUNTIME_MANIFEST_FILENAME = "runtime.manifest.json"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_VERIFIER_BINARY_ENV = "INVARLOCK_RUNTIME_VERIFIER"
RUNTIME_VERIFIER_BINARY_DEFAULT = "invarlock-runtime-verify"
RUNTIME_VERIFIER_CONTRACT_VERSION = "runtime-manifest-v1"
RUNTIME_IMAGE_LOCAL_DEFAULT = "invarlock-runtime:local"
RUNTIME_IMAGE_DEFAULT = "ghcr.io/invarlock/invarlock-runtime:latest"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

__all__ = [
    "ALLOW_HOST_EXECUTION_ENV",
    "ALLOW_NETWORK_ENV",
    "ALLOW_REMOTE_CODE_ENV",
    "ALLOW_THIRD_PARTY_PLUGINS_ENV",
    "ALLOW_UNATTESTED_ARTIFACTS_ENV",
    "CONTAINER_EXECUTION_ENV",
    "RUNTIME_IMAGE_ENV",
    "RUNTIME_IMAGE_DIGEST_ENV",
    "RUNTIME_MANIFEST_FILENAME",
    "RUNTIME_MANIFEST_VERSION",
    "RUNTIME_VERIFIER_BINARY_ENV",
    "RUNTIME_VERIFIER_BINARY_DEFAULT",
    "RUNTIME_VERIFIER_CONTRACT_VERSION",
    "apply_runtime_allowances",
    "build_container_command",
    "container_image_available_locally",
    "current_execution_mode",
    "delegate_current_process_to_container",
    "network_allowed",
    "host_execution_allowed",
    "load_runtime_manifest",
    "remote_code_allowed",
    "resolve_container_engine",
    "resolve_runtime_image",
    "resolve_runtime_image_digest",
    "runtime_verifier_binary",
    "running_inside_container",
    "serialize_canonical_json",
    "third_party_plugins_allowed",
    "unattested_artifacts_allowed",
    "write_runtime_manifest",
]


def _coerce_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _set_env_flag(name: str, enabled: bool | None) -> None:
    if enabled is True:
        os.environ[name] = "1"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value.__class__.__name__ == "OptionInfo":
        return _json_safe(getattr(value, "default", None))
    return str(value)


def serialize_canonical_json(payload: Any) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def network_allowed() -> bool:
    return _coerce_bool(os.environ.get(ALLOW_NETWORK_ENV)) is True


def host_execution_allowed() -> bool:
    return _coerce_bool(os.environ.get(ALLOW_HOST_EXECUTION_ENV)) is True


def remote_code_allowed() -> bool:
    return _coerce_bool(os.environ.get(ALLOW_REMOTE_CODE_ENV)) is True


def unattested_artifacts_allowed() -> bool:
    return _coerce_bool(os.environ.get(ALLOW_UNATTESTED_ARTIFACTS_ENV)) is True


def third_party_plugins_allowed() -> bool:
    explicit = _coerce_bool(os.environ.get(ALLOW_THIRD_PARTY_PLUGINS_ENV))
    return explicit is True


def running_inside_container() -> bool:
    return _coerce_bool(os.environ.get(CONTAINER_EXECUTION_ENV)) is True


def current_execution_mode() -> str:
    return "container" if running_inside_container() else "host-bypass"


def resolve_runtime_image() -> str:
    image = os.environ.get(RUNTIME_IMAGE_ENV, "").strip()
    if image:
        return image
    engine = resolve_container_engine()
    if engine is not None and container_image_available_locally(
        RUNTIME_IMAGE_LOCAL_DEFAULT, engine=engine
    ):
        return RUNTIME_IMAGE_LOCAL_DEFAULT
    return RUNTIME_IMAGE_DEFAULT


def resolve_runtime_image_digest() -> str | None:
    explicit = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "").strip()
    if explicit:
        return explicit
    image = resolve_runtime_image()
    if "@sha256:" in image:
        return image.split("@", 1)[1]
    engine = resolve_container_engine()
    if engine is None:
        return None
    _, digest = _inspect_container_image(engine, image)
    return digest


def _inspect_container_image(engine: str, image: str) -> tuple[bool, str | None]:
    completed = subprocess.run(
        [
            engine,
            "image",
            "inspect",
            image,
            "--format",
            "{{json .RepoDigests}}\n{{.Id}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False, None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    repo_digests: list[str] = []
    if lines:
        try:
            payload = json.loads(lines[0])
        except Exception:
            payload = None
        if isinstance(payload, list):
            repo_digests = [str(item) for item in payload if isinstance(item, str)]
    for digest_ref in repo_digests:
        if "@sha256:" in digest_ref:
            return True, digest_ref.split("@", 1)[1]
    if len(lines) >= 2 and lines[1].startswith("sha256:"):
        return True, lines[1]
    return True, None


def resolve_container_engine() -> str | None:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    return None


def container_image_available_locally(
    image: str | None = None, *, engine: str | None = None
) -> bool:
    resolved_engine = engine or resolve_container_engine()
    if resolved_engine is None:
        return False
    resolved_image = image or resolve_runtime_image()
    exists, _ = _inspect_container_image(resolved_engine, resolved_image)
    return exists


def runtime_verifier_binary() -> str:
    binary = os.environ.get(RUNTIME_VERIFIER_BINARY_ENV, "").strip()
    if binary:
        return binary
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "target" / "debug" / RUNTIME_VERIFIER_BINARY_DEFAULT,
        repo_root / "target" / "release" / RUNTIME_VERIFIER_BINARY_DEFAULT,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    script_dirs = [Path(sys.executable).parent]
    argv0 = Path(sys.argv[0]).parent if sys.argv else None
    if argv0 is not None:
        script_dirs.append(argv0)
    for script_dir in script_dirs:
        for candidate in (
            script_dir / RUNTIME_VERIFIER_BINARY_DEFAULT,
            script_dir / f"{RUNTIME_VERIFIER_BINARY_DEFAULT}.exe",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return RUNTIME_VERIFIER_BINARY_DEFAULT


def apply_runtime_allowances(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unattested_artifacts: bool = False,
) -> None:
    _set_env_flag(ALLOW_NETWORK_ENV, allow_network)
    _set_env_flag(ALLOW_HOST_EXECUTION_ENV, allow_host_execution)
    _set_env_flag(ALLOW_THIRD_PARTY_PLUGINS_ENV, allow_third_party_plugins)
    _set_env_flag(ALLOW_REMOTE_CODE_ENV, allow_remote_code)
    _set_env_flag(ALLOW_UNATTESTED_ARTIFACTS_ENV, allow_unattested_artifacts)
    if allow_network:
        try:
            from invarlock.security import enforce_network_policy

            enforce_network_policy(True)
        except Exception:
            pass


def build_container_command(argv: list[str] | None = None) -> list[str]:
    engine = resolve_container_engine()
    if engine is None:
        raise RuntimeError(
            "Host execution is disabled by default and no container engine "
            "(docker/podman) is available. Set "
            f"{ALLOW_HOST_EXECUTION_ENV}=1 or install docker/podman."
        )

    cwd = Path.cwd().resolve()
    image = resolve_runtime_image()
    digest = resolve_runtime_image_digest() or ""
    if not network_allowed() and not container_image_available_locally(
        image, engine=engine
    ):
        raise RuntimeError(
            "Host execution is disabled by default and runtime image "
            f"{image!r} is not available locally. Build it with `make runtime-image` "
            f"or set {ALLOW_NETWORK_ENV}=1 to allow pulling the image."
        )
    if argv is None:
        argv = list(sys.argv[1:])
    env_pairs = {
        ALLOW_NETWORK_ENV: "1" if network_allowed() else "0",
        ALLOW_REMOTE_CODE_ENV: "1" if remote_code_allowed() else "0",
        ALLOW_THIRD_PARTY_PLUGINS_ENV: "1" if third_party_plugins_allowed() else "0",
        ALLOW_UNATTESTED_ARTIFACTS_ENV: (
            "1" if unattested_artifacts_allowed() else "0"
        ),
        CONTAINER_EXECUTION_ENV: "1",
        RUNTIME_IMAGE_DIGEST_ENV: digest,
        "PYTHONPATH": "/workspace/src",
    }
    command = [engine, "run", "--rm"]
    if not network_allowed():
        command.extend(["--network", "none"])
    command.extend(["-v", f"{cwd}:/workspace", "-w", "/workspace"])
    for key, value in env_pairs.items():
        command.extend(["-e", f"{key}={value}"])
    # The runtime image already sets `python -m invarlock` as its entrypoint.
    command.extend([image, *argv])
    return command


def delegate_current_process_to_container(argv: list[str] | None = None) -> int:
    command = build_container_command(argv=argv)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _config_digest(
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
) -> tuple[str | None, str]:
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            return _sha256_path(path), "file"
    if config_payload is not None:
        payload = serialize_canonical_json(config_payload).encode("utf-8")
        return _sha256_bytes(payload), "inline"
    return None, "missing"


def write_runtime_manifest(
    report_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    report = Path(report_path).resolve()
    digest, digest_source = _config_digest(
        config_path=config_path, config_payload=config_payload
    )
    manifest: dict[str, Any] = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report),
            "filename": report.name,
            "sha256": _sha256_path(report),
        },
        "config": {
            "path": str(Path(config_path).resolve())
            if config_path is not None
            else None,
            "sha256": digest,
            "source": digest_source,
        },
        "execution_mode": current_execution_mode(),
        "runtime": {
            "image_ref": resolve_runtime_image(),
            "image_digest": resolve_runtime_image_digest(),
            "container_execution": running_inside_container(),
            "allow_network": network_allowed(),
            "allow_remote_code": remote_code_allowed(),
            "allow_third_party_plugins": third_party_plugins_allowed(),
        },
    }
    if isinstance(extra, dict) and extra:
        manifest["context"] = _json_safe(extra)
    manifest_path = report.parent / RUNTIME_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_runtime_manifest(
    report_path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any] | None]:
    report = Path(report_path)
    manifest_path = report.parent / RUNTIME_MANIFEST_FILENAME
    if not manifest_path.exists():
        return manifest_path, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return manifest_path, None
    return manifest_path, payload if isinstance(payload, dict) else None
