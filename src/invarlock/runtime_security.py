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

from invarlock.core.config_dependencies import inspect_config_dependencies

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


def _host_nvidia_visible() -> bool:
    if Path("/dev/nvidiactl").exists():
        return True
    return shutil.which("nvidia-smi") is not None


def _command_tokens(argv: list[str]) -> list[str]:
    return [token for token in argv if not token.startswith("-")]


def _requested_device(argv: list[str]) -> str | None:
    if "--device" in argv:
        idx = argv.index("--device")
        if idx + 1 < len(argv):
            return str(argv[idx + 1]).strip().lower()
        return None

    command_tokens = _command_tokens(argv)
    if not command_tokens:
        return None
    if command_tokens[0] in {"evaluate", "run", "calibrate"}:
        return "auto"
    return None


_CONFIG_SCAN_ARG_FLAGS = {"--config", "-c", "--preset", "--edit-config"}
_CONFIG_PATH_ARG_FLAGS = _CONFIG_SCAN_ARG_FLAGS | {"--baseline-report"}
_OUTPUT_PATH_ARG_FLAGS = {"--out", "--report-out"}
_PATH_ARG_FLAGS = _OUTPUT_PATH_ARG_FLAGS | _CONFIG_PATH_ARG_FLAGS
_LOCAL_MODEL_ARG_FLAGS = {"--baseline", "--subject", "--source", "--edited"}
_PATH_ENV_VARS = {
    "INVARLOCK_CONFIG_ROOT",
    "INVARLOCK_EVALUATE_TMP_DIR",
    "INVARLOCK_EXPORT_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "TMPDIR",
    "TMP",
}
_BEHAVIOR_ENV_VARS = {
    "INVARLOCK_STORE_EVAL_WINDOWS",
    "INVARLOCK_SNAPSHOT_MODE",
    "INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION",
    "INVARLOCK_SNAPSHOT_THRESHOLD_MB",
    "INVARLOCK_SKIP_OVERHEAD_CHECK",
    "INVARLOCK_WINDOW_OVERLAP_FRACTION",
    "INVARLOCK_DETERMINISM",
    "PACK_DETERMINISM",
    "INVARLOCK_DETERMINISM_WARN_ONLY",
    "INVARLOCK_OMP_THREADS",
    "INVARLOCK_DEDUP_TEXTS",
    "INVARLOCK_CAPACITY_FAST",
    "INVARLOCK_TINY_RELAX",
    "TOKENIZERS_PARALLELISM",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE",
}


def _iter_path_args(argv: list[str], *, flags: set[str] | None = None) -> list[Path]:
    active_flags = flags or _PATH_ARG_FLAGS
    paths: list[Path] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in active_flags and idx + 1 < len(argv):
            paths.append(Path(argv[idx + 1]).expanduser())
            idx += 2
            continue
        for flag in active_flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                paths.append(Path(token[len(prefix) :]).expanduser())
                break
        idx += 1
    return paths


def _minimize_mounts(mounts: list[Path] | set[Path]) -> list[Path]:
    ordered = sorted(set(mounts), key=lambda item: (len(item.parts), str(item)))
    minimized: list[Path] = []
    for mount in ordered:
        if any(
            existing == mount or existing in mount.parents for existing in minimized
        ):
            continue
        minimized.append(mount)
    return minimized


def _iter_flag_occurrences(
    argv: list[str], *, flags: set[str]
) -> list[tuple[int, str, str, int | None]]:
    occurrences: list[tuple[int, str, str, int | None]] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in flags and idx + 1 < len(argv):
            occurrences.append((idx, token, argv[idx + 1], idx + 1))
            idx += 2
            continue
        matched = False
        for flag in flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                occurrences.append((idx, flag, token[len(prefix) :], None))
                matched = True
                break
        idx += 1 if not matched else 1
    return occurrences


def _replace_flag_value(
    argv: list[str],
    *,
    token_index: int,
    flag: str,
    value_index: int | None,
    new_value: str,
) -> None:
    if value_index is None:
        argv[token_index] = f"{flag}={new_value}"
    else:
        argv[value_index] = new_value


def _absolute_host_path(path: str | Path, *, cwd: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.abspath(str(candidate)))
    return Path(os.path.abspath(str(cwd / candidate)))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _workspace_path(path: Path, *, cwd: Path) -> str:
    return str(Path("/workspace") / path.relative_to(cwd))


def _mount_root_for_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.exists() and expanded.is_dir() else expanded.parent


def _mount_root_for_resolved_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    return resolved if resolved.exists() and resolved.is_dir() else resolved.parent


def _mount_is_already_covered(mount: Path, *, cwd: Path) -> bool:
    return mount == cwd or cwd in mount.parents


def _iter_external_symlink_target_mounts(
    path: Path, *, cwd: Path, recursive: bool = True
) -> list[Path]:
    expanded = path.expanduser()
    root_mount = _mount_root_for_path(expanded)
    mounts: set[Path] = set()

    def _record_symlink_target(link_path: Path) -> None:
        target_mount = _mount_root_for_resolved_path(link_path)
        if target_mount == root_mount or root_mount in target_mount.parents:
            return
        if _mount_is_already_covered(target_mount, cwd=cwd):
            return
        mounts.add(target_mount)

    if expanded.is_symlink():
        _record_symlink_target(expanded)

    if not recursive:
        return sorted(mounts, key=lambda item: (len(item.parts), str(item)))

    walk_root = expanded.resolve(strict=False) if expanded.is_symlink() else expanded
    if not walk_root.exists() or not walk_root.is_dir():
        return sorted(mounts, key=lambda item: (len(item.parts), str(item)))

    for current, dirnames, filenames in os.walk(walk_root, followlinks=False):
        current_path = Path(current)
        for name in (*dirnames, *filenames):
            entry = current_path / name
            if entry.is_symlink():
                _record_symlink_target(entry)

    return sorted(mounts, key=lambda item: (len(item.parts), str(item)))


def _iter_absolute_pythonpath_entries() -> list[Path]:
    raw_value = os.environ.get("PYTHONPATH", "")
    if not raw_value:
        return []
    entries: list[Path] = []
    seen: set[Path] = set()
    for raw_entry in raw_value.split(os.pathsep):
        text = raw_entry.strip()
        if not text:
            continue
        entry = Path(text).expanduser()
        if not entry.is_absolute():
            continue
        resolved = entry.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(resolved)
    return entries


def _container_pythonpath_entries(*, cwd: Path) -> tuple[list[str], list[Path]]:
    host_entries = _iter_absolute_pythonpath_entries()
    if not host_entries:
        return ["/workspace/src"], []

    container_entries: list[str] = []
    mounts: set[Path] = set()
    for entry in host_entries:
        if entry == cwd or cwd in entry.parents:
            rel = entry.relative_to(cwd)
            container_entries.append(str(Path("/workspace") / rel))
            continue
        container_entries.append(str(entry))
        mount = _mount_root_for_path(entry)
        if not _mount_is_already_covered(mount, cwd=cwd):
            mounts.add(mount)
        mounts.update(_iter_external_symlink_target_mounts(entry, cwd=cwd))
    return container_entries, _minimize_mounts(mounts)


def _record_path_dependencies(
    path: Path,
    mounts: set[Path],
    *,
    cwd: Path,
    recursive_symlink_scan: bool = True,
) -> bool:
    mounts.update(
        _iter_external_symlink_target_mounts(
            path,
            cwd=cwd,
            recursive=recursive_symlink_scan,
        )
    )
    if _path_is_within(path, cwd):
        return True
    mount = _mount_root_for_path(path)
    if not _mount_is_already_covered(mount, cwd=cwd):
        mounts.add(mount)
    return False


def _normalize_output_path_for_container(
    raw_value: str, *, cwd: Path
) -> tuple[str, set[Path]]:
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    mounts: set[Path] = set()
    inside_cwd = _record_path_dependencies(host_path, mounts, cwd=cwd)
    if inside_cwd:
        return raw_value, mounts
    return str(host_path), mounts


def _normalize_local_model_path_for_container(
    raw_value: str, *, cwd: Path
) -> tuple[str, set[Path], bool]:
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    if not host_path.exists():
        return raw_value, set(), False
    mounts: set[Path] = set()
    inside_cwd = _record_path_dependencies(host_path, mounts, cwd=cwd)
    if inside_cwd:
        return _workspace_path(host_path, cwd=cwd), mounts, True
    return str(host_path), mounts, True


def _normalize_config_path_for_container(
    raw_value: str,
    *,
    cwd: Path,
    scan_dependencies: bool,
) -> tuple[str, set[Path], bool]:
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    mounts: set[Path] = set()
    needs_cwd_host_mirror = _record_path_dependencies(host_path, mounts, cwd=cwd)
    if scan_dependencies:
        try:
            scan = inspect_config_dependencies(host_path)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Delegated runtime config {host_path} is not mountable: {exc}"
            ) from exc
        for config_path in scan.config_paths:
            if _record_path_dependencies(config_path, mounts, cwd=cwd):
                needs_cwd_host_mirror = True
        for referenced_path in scan.referenced_paths:
            if _record_path_dependencies(referenced_path, mounts, cwd=cwd):
                needs_cwd_host_mirror = True
    return str(host_path), mounts, needs_cwd_host_mirror


def _normalize_delegated_argv(
    argv: list[str], *, cwd: Path
) -> tuple[list[str], list[Path], bool]:
    rewritten = list(argv)
    mounts: set[Path] = set()
    needs_cwd_host_mirror = False

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_CONFIG_PATH_ARG_FLAGS
    ):
        scan_dependencies = flag in _CONFIG_SCAN_ARG_FLAGS
        new_value, extra_mounts, needs_mirror = _normalize_config_path_for_container(
            value,
            cwd=cwd,
            scan_dependencies=scan_dependencies,
        )
        mounts.update(extra_mounts)
        needs_cwd_host_mirror = needs_cwd_host_mirror or needs_mirror
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_OUTPUT_PATH_ARG_FLAGS
    ):
        new_value, extra_mounts = _normalize_output_path_for_container(
            value,
            cwd=cwd,
        )
        mounts.update(extra_mounts)
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    for token_index, flag, value, value_index in _iter_flag_occurrences(
        rewritten, flags=_LOCAL_MODEL_ARG_FLAGS
    ):
        new_value, extra_mounts, treated_as_path = (
            _normalize_local_model_path_for_container(
                value,
                cwd=cwd,
            )
        )
        if not treated_as_path:
            continue
        mounts.update(extra_mounts)
        _replace_flag_value(
            rewritten,
            token_index=token_index,
            flag=flag,
            value_index=value_index,
            new_value=new_value,
        )

    return rewritten, _minimize_mounts(mounts), needs_cwd_host_mirror


def _path_env_value_for_container(
    raw_value: str,
    *,
    cwd: Path,
) -> tuple[str, list[Path]]:
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    mounts: set[Path] = set()
    # Path-based env vars like TMPDIR and cache roots can point at very large
    # trees. Mount the declared root and any direct symlink target, but avoid
    # recursively scanning the full directory contents while building commands.
    inside_cwd = _record_path_dependencies(
        host_path,
        mounts,
        cwd=cwd,
        recursive_symlink_scan=False,
    )
    if inside_cwd:
        return _workspace_path(host_path, cwd=cwd), _minimize_mounts(mounts)
    return str(host_path), _minimize_mounts(mounts)


def _delegated_env_pairs(*, cwd: Path) -> tuple[dict[str, str], list[Path]]:
    env_pairs: dict[str, str] = {
        ALLOW_NETWORK_ENV: "1" if network_allowed() else "0",
        ALLOW_REMOTE_CODE_ENV: "1" if remote_code_allowed() else "0",
        ALLOW_THIRD_PARTY_PLUGINS_ENV: "1" if third_party_plugins_allowed() else "0",
        ALLOW_UNATTESTED_ARTIFACTS_ENV: (
            "1" if unattested_artifacts_allowed() else "0"
        ),
        CONTAINER_EXECUTION_ENV: "1",
    }
    mounts: list[Path] = []
    for name in sorted(_BEHAVIOR_ENV_VARS):
        value = os.environ.get(name)
        if value is not None:
            env_pairs[name] = value
    for name in sorted(_PATH_ENV_VARS):
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        container_value, extra_mounts = _path_env_value_for_container(
            value,
            cwd=cwd,
        )
        env_pairs[name] = container_value
        mounts.extend(extra_mounts)
    return env_pairs, _minimize_mounts(mounts)


def _needs_gpu_passthrough(argv: list[str]) -> bool:
    requested = _requested_device(argv)
    if requested not in {"auto", "cuda"}:
        return False
    return _host_nvidia_visible()


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
    argv, argv_mounts, needs_cwd_host_mirror = _normalize_delegated_argv(argv, cwd=cwd)
    pythonpath_entries, pythonpath_mounts = _container_pythonpath_entries(cwd=cwd)
    env_pairs, env_mounts = _delegated_env_pairs(cwd=cwd)
    env_pairs[RUNTIME_IMAGE_DIGEST_ENV] = digest
    env_pairs["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    command = [engine, "run", "--rm"]
    if _needs_gpu_passthrough(argv):
        command.extend(["--gpus", "all"])
    if not network_allowed():
        command.extend(["--network", "none"])
    command.extend(["-v", f"{cwd}:/workspace", "-w", "/workspace"])
    if needs_cwd_host_mirror:
        command.extend(["-v", f"{cwd}:{cwd}"])
    extra_mounts = list(argv_mounts)
    extra_mounts.extend(env_mounts)
    for mount in pythonpath_mounts:
        if any(existing == mount for existing in extra_mounts):
            continue
        extra_mounts.append(mount)
    for mount in _minimize_mounts(extra_mounts):
        command.extend(["-v", f"{mount}:{mount}"])
    for key, value in env_pairs.items():
        command.extend(["-e", f"{key}={value}"])
    # The runtime image already sets `python -m invarlock` as its entrypoint.
    command.extend([image, *argv])
    return command


def delegate_current_process_to_container(argv: list[str] | None = None) -> int:
    command = build_container_command(argv=argv)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def build_container_python_command(
    script_path: str | os.PathLike[str],
    argv: list[str] | None = None,
) -> list[str]:
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
    argv, argv_mounts, needs_cwd_host_mirror = _normalize_delegated_argv(argv, cwd=cwd)
    pythonpath_entries, pythonpath_mounts = _container_pythonpath_entries(cwd=cwd)
    env_pairs, env_mounts = _delegated_env_pairs(cwd=cwd)
    env_pairs[RUNTIME_IMAGE_DIGEST_ENV] = digest
    env_pairs["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    script_host_path = _absolute_host_path(script_path, cwd=cwd)
    script_mounts: set[Path] = set()
    if _record_path_dependencies(script_host_path, script_mounts, cwd=cwd):
        container_script = _workspace_path(script_host_path, cwd=cwd)
    else:
        container_script = str(script_host_path)

    command = [engine, "run", "--rm", "--entrypoint", "python"]
    if _needs_gpu_passthrough(["run", *argv]):
        command.extend(["--gpus", "all"])
    if not network_allowed():
        command.extend(["--network", "none"])
    command.extend(["-v", f"{cwd}:/workspace", "-w", "/workspace"])
    if needs_cwd_host_mirror:
        command.extend(["-v", f"{cwd}:{cwd}"])
    extra_mounts = list(argv_mounts)
    extra_mounts.extend(env_mounts)
    extra_mounts.extend(pythonpath_mounts)
    extra_mounts.extend(script_mounts)
    for mount in _minimize_mounts(extra_mounts):
        command.extend(["-v", f"{mount}:{mount}"])
    for key, value in env_pairs.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend([image, container_script, *argv])
    return command


def delegate_python_script_to_container(
    script_path: str | os.PathLike[str],
    argv: list[str] | None = None,
) -> int:
    command = build_container_python_command(script_path, argv=argv)
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
