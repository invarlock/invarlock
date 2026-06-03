from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def _helpers():
    from invarlock import runtime_security_helpers as helpers

    return helpers


_CONFIG_SCAN_ARG_FLAGS = {"--config", "-c", "--preset", "--edit-config"}
_CONFIG_PATH_ARG_FLAGS = _CONFIG_SCAN_ARG_FLAGS | {"--baseline-report"}
_OUTPUT_PATH_ARG_FLAGS = {"--out", "--report-out"}
_LOCAL_MODEL_ARG_FLAGS = {"--baseline", "--subject"}
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
_FORWARDED_ENV_VARS = {
    "CUDA_VISIBLE_DEVICES",
    "HF_DATASETS_OFFLINE",
    "INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE",
    "INVARLOCK_DETERMINISM",
    "INVARLOCK_DETERMINISM_WARN_ONLY",
    "INVARLOCK_RUNTIME_IMAGE",
    "INVARLOCK_TINY_RELAX",
    "NVIDIA_VISIBLE_DEVICES",
    "INVARLOCK_SKIP_OVERHEAD_CHECK",
    "INVARLOCK_SNAPSHOT_MODE",
    "INVARLOCK_STORE_EVAL_WINDOWS",
    "PACK_DETERMINISM",
}


def _host_nvidia_visible() -> bool:
    if Path("/dev/nvidiactl").exists():
        return True
    return shutil.which("nvidia-smi") is not None


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
    helpers = _helpers()
    expanded = path.expanduser()
    root_mount = _mount_root_for_path(expanded)
    mounts: set[Path] = set()

    def _record_symlink_target(link_path: Path) -> None:
        target_mount = _mount_root_for_resolved_path(link_path)
        if target_mount == root_mount or root_mount in target_mount.parents:
            return
        if helpers._mount_is_already_covered(target_mount, cwd=cwd):
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
    helpers = _helpers()
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
        if not helpers._mount_is_already_covered(mount, cwd=cwd):
            mounts.add(mount)
        mounts.update(helpers._iter_external_symlink_target_mounts(entry, cwd=cwd))
    return container_entries, helpers._minimize_mounts(mounts)


def _record_path_dependencies(
    path: Path,
    mounts: set[Path],
    *,
    cwd: Path,
    recursive_symlink_scan: bool = True,
) -> bool:
    helpers = _helpers()
    mounts.update(
        helpers._iter_external_symlink_target_mounts(
            path,
            cwd=cwd,
            recursive=recursive_symlink_scan,
        )
    )
    if _path_is_within(path, cwd):
        return True
    mount = _mount_root_for_path(path)
    if not helpers._mount_is_already_covered(mount, cwd=cwd):
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
    helpers = _helpers()
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    mounts: set[Path] = set()
    needs_cwd_host_mirror = _record_path_dependencies(host_path, mounts, cwd=cwd)
    if scan_dependencies:
        try:
            scan = helpers.inspect_config_dependencies(host_path)
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


def _path_env_value_for_container(
    raw_value: str,
    *,
    cwd: Path,
) -> tuple[str, list[Path]]:
    host_path = _absolute_host_path(raw_value, cwd=cwd)
    mounts: set[Path] = set()
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
    helpers = _helpers()
    env_pairs: dict[str, str] = {
        helpers.ALLOW_NETWORK_ENV: "1" if helpers.network_allowed() else "0",
        helpers.ALLOW_REMOTE_CODE_ENV: "1" if helpers.remote_code_allowed() else "0",
        helpers.ALLOW_THIRD_PARTY_PLUGINS_ENV: (
            "1" if helpers.third_party_plugins_allowed() else "0"
        ),
        helpers.ALLOW_UNVERIFIED_PROVENANCE_ENV: (
            "1" if helpers.unverified_provenance_allowed() else "0"
        ),
        helpers.CONTAINER_EXECUTION_ENV: "1",
    }
    mounts: list[Path] = []
    for name in sorted(_PATH_ENV_VARS):
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        container_value, extra_mounts = _path_env_value_for_container(value, cwd=cwd)
        env_pairs[name] = container_value
        mounts.extend(extra_mounts)
    for name in sorted(_FORWARDED_ENV_VARS):
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        env_pairs[name] = value
    return env_pairs, _minimize_mounts(mounts)


def _merge_container_mounts(*groups: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    merged: list[Path] = []
    for group in groups:
        for mount in group:
            if any(existing == mount for existing in merged):
                continue
            merged.append(mount)
    return tuple(_minimize_mounts(merged))


def _resolve_container_launch_context(plan: Any) -> Any:
    helpers = _helpers()
    engine = helpers.resolve_container_engine()
    if engine is None:
        raise RuntimeError(
            "Host execution is disabled by default and no container engine "
            "such as Podman or Docker is available. Set "
            f"{helpers.CONTAINER_ENGINE_ENV}=podman|docker, "
            f"{helpers.ALLOW_HOST_EXECUTION_ENV}=1 or install Podman or Docker."
        )

    cwd = Path.cwd().resolve()
    image = helpers.resolve_runtime_image()
    digest = helpers.resolve_runtime_image_digest() or ""
    if not helpers.network_allowed() and not helpers.container_image_available_locally(
        image, engine=engine
    ):
        raise RuntimeError(
            "Host execution is disabled by default and runtime image "
            f"{image!r} is not available locally. Build it with "
            f"`{helpers._runtime_image_build_command(image)}` or set "
            f"{helpers.ALLOW_NETWORK_ENV}=1 to allow pulling the image."
        )

    pythonpath_entries, pythonpath_mounts = helpers._container_pythonpath_entries(
        cwd=cwd
    )
    env_pairs, env_mounts = helpers._delegated_env_pairs(cwd=cwd)
    env_pairs[helpers.RUNTIME_IMAGE_DIGEST_ENV] = digest
    env_pairs["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    base_mounts = _merge_container_mounts(
        plan.argv_mounts,
        tuple(env_mounts),
        tuple(pythonpath_mounts),
    )
    return helpers._ContainerLaunchContext(
        engine=engine,
        cwd=cwd,
        image=image,
        env_pairs=env_pairs,
        base_mounts=base_mounts,
    )


def _compose_container_run_args(
    context: Any,
    plan: Any,
    *,
    entrypoint: tuple[str, ...] = (),
    extra_mounts: tuple[Path, ...] = (),
    argv: tuple[str, ...],
) -> list[str]:
    helpers = _helpers()
    command = [context.engine, "run", "--rm", *entrypoint]
    if plan.gpu_passthrough:
        command.extend(["--gpus", "all"])
    if not helpers.network_allowed():
        command.extend(["--network", "none"])
    command.extend(["-v", f"{context.cwd}:/workspace", "-w", "/workspace"])
    if plan.needs_cwd_host_mirror:
        command.extend(["-v", f"{context.cwd}:{context.cwd}"])
    for mount in _merge_container_mounts(context.base_mounts, extra_mounts):
        command.extend(["-v", f"{mount}:{mount}"])
    for key, value in context.env_pairs.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend([context.image, *argv])
    return command


__all__ = [
    "_CONFIG_SCAN_ARG_FLAGS",
    "_CONFIG_PATH_ARG_FLAGS",
    "_OUTPUT_PATH_ARG_FLAGS",
    "_LOCAL_MODEL_ARG_FLAGS",
    "_PATH_ENV_VARS",
    "_FORWARDED_ENV_VARS",
    "_host_nvidia_visible",
    "_minimize_mounts",
    "_absolute_host_path",
    "_path_is_within",
    "_workspace_path",
    "_mount_root_for_path",
    "_mount_root_for_resolved_path",
    "_mount_is_already_covered",
    "_iter_external_symlink_target_mounts",
    "_iter_absolute_pythonpath_entries",
    "_container_pythonpath_entries",
    "_record_path_dependencies",
    "_normalize_output_path_for_container",
    "_normalize_local_model_path_for_container",
    "_normalize_config_path_for_container",
    "_path_env_value_for_container",
    "_delegated_env_pairs",
    "_merge_container_mounts",
    "_resolve_container_launch_context",
    "_compose_container_run_args",
]
