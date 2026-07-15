"""Named runtime boundary for the optional GPTQModel backend.

GPTQModel 7.x can import symbols that moved out of newer Transformers hub
namespaces.  The compatibility bridge belongs here, immediately before the
optional backend import, rather than being an invisible global side effect of
an adapter.  A successful import is deliberately only an import proof: model
loading, CUDA kernel compilation, and inference must still be exercised by a
caller that needs those guarantees.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import sysconfig
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import ModuleType

_RUNTIME_IMPORT_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_VERSION_ERRORS = (PackageNotFoundError, OSError, RuntimeError, TypeError, ValueError)
_BRIDGED_HUB_SYMBOLS = ("create_repo", "list_repo_tree")
_JIT_MISSING_REQUIREMENT_MESSAGES = {
    "python_headers": "Python.h missing",
}


@dataclass(frozen=True)
class GPTQModelJITToolchainStatus:
    """Non-sensitive availability facts for a CUDA JIT preflight.

    This status does not expose executable or host paths.  It is requested
    explicitly because an ordinary Python import must remain possible on a
    CPU-only host or before a CUDA lane is selected.
    """

    required: bool
    interpreter_bin_added_to_path: bool
    ninja_available: bool
    python_headers_available: bool
    nvcc_available: bool
    cxx_available: bool
    ready: bool
    missing_requirements: tuple[str, ...]


@dataclass(frozen=True)
class GPTQModelRuntimeStatus:
    """Observed state of the named GPTQModel runtime preparation boundary."""

    importable: bool
    gptqmodel_version: str | None
    import_error_type: str | None
    compatibility_bridge_required: bool
    compatibility_bridge_applied: bool
    compatibility_bridge_missing_symbols: tuple[str, ...]
    compatibility_bridge_error_type: str | None
    jit_toolchain: GPTQModelJITToolchainStatus | None

    @property
    def ready(self) -> bool:
        """Whether this request's import and optional JIT requirements hold."""

        return (
            self.importable
            and self.compatibility_bridge_error_type is None
            and (
                not self.compatibility_bridge_required
                or self.compatibility_bridge_applied
            )
            and (self.jit_toolchain is None or self.jit_toolchain.ready)
        )


@dataclass(frozen=True)
class _CompatibilityBridgeStatus:
    required: bool
    applied: bool
    missing_symbols: tuple[str, ...]
    error_type: str | None


@dataclass(frozen=True)
class _BridgeObservation:
    """Private process-bound record of a successful named bridge."""

    hub: object
    pid: int
    status: _CompatibilityBridgeStatus


_BRIDGE_OBSERVATION: _BridgeObservation | None = None


def _observed_bridge_status(
    transformers_hub: object,
    *,
    pid: int,
) -> _CompatibilityBridgeStatus | None:
    """Return an exact in-process observation, rejecting inherited state.

    The bridge adds missing attributes to the live Transformers hub module.
    A later proof collection therefore sees the attributes and would otherwise
    incorrectly report that no bridge was required.  Record only a successful
    named bridge in module-private state tied to one PID.  A fork inherits the
    patched module but not an in-process bridge event; it must fail closed
    rather than claim that the child applied the bridge itself.
    """

    observed = _BRIDGE_OBSERVATION
    if observed is None or observed.hub is not transformers_hub:
        return None
    if observed.pid == pid:
        return observed.status
    return _CompatibilityBridgeStatus(
        required=True,
        applied=False,
        missing_symbols=(),
        error_type="InheritedProcessState",
    )


def _record_successful_bridge_observation(
    transformers_hub: object,
    status: _CompatibilityBridgeStatus,
    *,
    pid: int,
) -> _CompatibilityBridgeStatus:
    """Bind a successful bridge event to this process only."""

    global _BRIDGE_OBSERVATION
    if status.required and status.applied:
        _BRIDGE_OBSERVATION = _BridgeObservation(
            hub=transformers_hub,
            pid=pid,
            status=status,
        )
    return status


def _module_version(module: ModuleType) -> str | None:
    try:
        installed_version = package_version("gptqmodel")
    except _VERSION_ERRORS:
        installed_version = None
    if installed_version:
        return installed_version
    raw_module_version = getattr(module, "__version__", None)
    return raw_module_version if isinstance(raw_module_version, str) else None


def _transformers_hub_compatibility_bridge() -> _CompatibilityBridgeStatus:
    """Install only the GPTQModel hub symbols absent from Transformers.

    The bridge intentionally has no effect when the selected Transformers
    release still exports the two names.  It is narrow enough to keep the
    optional GPTQModel integration visible and independently testable.
    """

    try:
        transformers = importlib.import_module("transformers")
    except _RUNTIME_IMPORT_ERRORS as exc:
        return _CompatibilityBridgeStatus(
            required=False,
            applied=False,
            missing_symbols=(),
            error_type=type(exc).__name__,
        )

    transformers_utils = getattr(transformers, "utils", None)
    transformers_hub = getattr(transformers_utils, "hub", None)
    if transformers_hub is None:
        return _CompatibilityBridgeStatus(
            required=True,
            applied=False,
            missing_symbols=("transformers.utils.hub",),
            error_type="AttributeError",
        )

    process_id = os.getpid()
    observed = _observed_bridge_status(transformers_hub, pid=process_id)
    if observed is not None:
        return observed

    missing_symbols = tuple(
        symbol
        for symbol in _BRIDGED_HUB_SYMBOLS
        if not hasattr(transformers_hub, symbol)
    )
    if not missing_symbols:
        return _CompatibilityBridgeStatus(
            required=False,
            applied=False,
            missing_symbols=(),
            error_type=None,
        )

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        bridge_values: dict[str, object] = {}
        if "create_repo" in missing_symbols:
            # Hugging Face Hub exposes some public names lazily through
            # module ``__getattr__``; direct module dictionaries miss them.
            bridge_values["create_repo"] = getattr(  # noqa: B009
                huggingface_hub,
                "create_repo",
            )
        if "list_repo_tree" in missing_symbols:
            hf_api_type = getattr(huggingface_hub, "HfApi")  # noqa: B009
            bridge_values["list_repo_tree"] = hf_api_type().list_repo_tree
    except _RUNTIME_IMPORT_ERRORS as exc:
        return _CompatibilityBridgeStatus(
            required=True,
            applied=False,
            missing_symbols=missing_symbols,
            error_type=type(exc).__name__,
        )

    assigned_symbols: list[str] = []
    try:
        for symbol, value in bridge_values.items():
            setattr(transformers_hub, symbol, value)
            assigned_symbols.append(symbol)
    except _RUNTIME_IMPORT_ERRORS as exc:
        # Never leave a half-installed compatibility bridge.  The symbols were
        # absent before this attempt, so a best-effort deletion restores the
        # observed upstream namespace before reporting the failure.
        for symbol in reversed(assigned_symbols):
            try:
                delattr(transformers_hub, symbol)
            except (AttributeError, TypeError):
                pass
        return _CompatibilityBridgeStatus(
            required=True,
            applied=False,
            missing_symbols=missing_symbols,
            error_type=type(exc).__name__,
        )

    still_missing = tuple(
        symbol for symbol in missing_symbols if not hasattr(transformers_hub, symbol)
    )
    return _record_successful_bridge_observation(
        transformers_hub,
        _CompatibilityBridgeStatus(
            required=True,
            applied=not still_missing,
            missing_symbols=still_missing,
            error_type=None if not still_missing else "AttributeError",
        ),
        pid=process_id,
    )


def _active_interpreter_bin() -> Path:
    executable = Path(sys.executable)
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    # Do not resolve this path: virtualenv Python executables are commonly
    # symlinks to a system interpreter, while the sibling ``ninja`` lives in
    # the virtualenv's own bin directory.
    return executable.parent


def _ensure_active_interpreter_bin_on_path() -> bool:
    """Expose a venv-installed ``ninja`` to a process launched by absolute Python.

    A shell can invoke ``/venv/bin/python`` while retaining a PATH that omits
    ``/venv/bin``.  GPTQModel's first-use JIT invokes ``ninja`` by name, so in
    that narrow case we add only the active interpreter's bin directory and
    only when it actually contains the needed executable.
    """

    if shutil.which("ninja") is not None:
        return False
    interpreter_bin = _active_interpreter_bin()
    candidate = interpreter_bin / "ninja"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return False

    current_path = os.environ.get("PATH", "")
    entries = tuple(entry for entry in current_path.split(os.pathsep) if entry)
    try:
        already_present = any(
            Path(entry).resolve() == interpreter_bin for entry in entries
        )
    except OSError:
        already_present = str(interpreter_bin) in entries
    if already_present:
        return False
    os.environ["PATH"] = (
        f"{interpreter_bin}{os.pathsep}{current_path}"
        if current_path
        else str(interpreter_bin)
    )
    return True


def _python_headers_available() -> bool:
    try:
        candidate_dirs = {
            sysconfig.get_paths().get("include"),
            sysconfig.get_paths().get("platinclude"),
            sysconfig.get_config_var("INCLUDEPY"),
            sysconfig.get_config_var("CONFINCLUDEPY"),
        }
    except (AttributeError, OSError, TypeError):
        return False
    return any(
        include_dir and (Path(str(include_dir)) / "Python.h").is_file()
        for include_dir in candidate_dirs
    )


def _nvcc_available() -> bool:
    if shutil.which("nvcc") is not None:
        return True
    cuda_homes = [os.environ.get("CUDA_HOME")]
    try:
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
        cuda_homes.append(getattr(cpp_extension, "CUDA_HOME", None))  # noqa: B009
    except _RUNTIME_IMPORT_ERRORS:
        pass
    # This is the conventional toolkit location when neither PATH nor the
    # environment points at it.  It is still accepted only if ``nvcc`` exists
    # and is executable, not because the directory name is present.
    cuda_homes.append("/usr/local/cuda")
    for cuda_home in cuda_homes:
        if not cuda_home:
            continue
        candidate = Path(str(cuda_home)) / "bin" / "nvcc"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def _cxx_available() -> bool:
    return any(
        shutil.which(candidate) is not None for candidate in ("c++", "g++", "gcc")
    )


def inspect_gptqmodel_jit_toolchain() -> GPTQModelJITToolchainStatus:
    """Inspect the prerequisites required by GPTQModel CUDA first-use JIT."""

    interpreter_bin_added_to_path = _ensure_active_interpreter_bin_on_path()
    ninja_available = shutil.which("ninja") is not None
    python_headers_available = _python_headers_available()
    nvcc_available = _nvcc_available()
    cxx_available = _cxx_available()
    requirements = {
        "ninja": ninja_available,
        "python_headers": python_headers_available,
        "nvcc": nvcc_available,
        "cxx": cxx_available,
    }
    missing_requirements = tuple(
        name for name, available in requirements.items() if not available
    )
    return GPTQModelJITToolchainStatus(
        required=True,
        interpreter_bin_added_to_path=interpreter_bin_added_to_path,
        ninja_available=ninja_available,
        python_headers_available=python_headers_available,
        nvcc_available=nvcc_available,
        cxx_available=cxx_available,
        ready=not missing_requirements,
        missing_requirements=missing_requirements,
    )


def prepare_gptqmodel_runtime(
    *,
    require_jit_toolchain: bool = False,
) -> GPTQModelRuntimeStatus:
    """Prepare and observe GPTQModel without pretending an import proves a load.

    Set ``require_jit_toolchain`` only for a CUDA lane that may trigger
    GPTQModel's first-use compilation.  Normal adapter discovery and CPU
    imports intentionally leave that preflight disabled.
    """

    bridge_status = _transformers_hub_compatibility_bridge()
    jit_toolchain = inspect_gptqmodel_jit_toolchain() if require_jit_toolchain else None
    try:
        gptqmodel = importlib.import_module("gptqmodel")
    except _RUNTIME_IMPORT_ERRORS as exc:
        return GPTQModelRuntimeStatus(
            importable=False,
            gptqmodel_version=None,
            import_error_type=type(exc).__name__,
            compatibility_bridge_required=bridge_status.required,
            compatibility_bridge_applied=bridge_status.applied,
            compatibility_bridge_missing_symbols=bridge_status.missing_symbols,
            compatibility_bridge_error_type=bridge_status.error_type,
            jit_toolchain=jit_toolchain,
        )
    return GPTQModelRuntimeStatus(
        importable=True,
        gptqmodel_version=_module_version(gptqmodel),
        import_error_type=None,
        compatibility_bridge_required=bridge_status.required,
        compatibility_bridge_applied=bridge_status.applied,
        compatibility_bridge_missing_symbols=bridge_status.missing_symbols,
        compatibility_bridge_error_type=bridge_status.error_type,
        jit_toolchain=jit_toolchain,
    )


def require_gptqmodel_runtime(
    *,
    require_jit_toolchain: bool = False,
) -> GPTQModelRuntimeStatus:
    """Return prepared runtime status or raise a clear optional-runtime error."""

    status = prepare_gptqmodel_runtime(
        require_jit_toolchain=require_jit_toolchain,
    )
    if not status.importable:
        error_type = status.import_error_type or "UnknownImportError"
        raise ImportError(f"GPTQModel runtime import failed: {error_type}")
    if status.compatibility_bridge_error_type is not None or (
        status.compatibility_bridge_required and not status.compatibility_bridge_applied
    ):
        error_type = status.compatibility_bridge_error_type or "UnknownBridgeError"
        raise ImportError(f"GPTQModel compatibility bridge unavailable: {error_type}")
    if require_jit_toolchain and (
        status.jit_toolchain is None or not status.jit_toolchain.ready
    ):
        missing = (
            status.jit_toolchain.missing_requirements
            if status.jit_toolchain is not None
            else ("jit_toolchain",)
        )
        missing_descriptions = tuple(
            _JIT_MISSING_REQUIREMENT_MESSAGES.get(requirement, requirement)
            for requirement in missing
        )
        raise RuntimeError(
            "GPTQModel CUDA JIT toolchain unavailable: "
            + ", ".join(missing_descriptions)
        )
    return status


def import_gptqmodel(
    *,
    require_jit_toolchain: bool = False,
) -> ModuleType:
    """Import GPTQModel through the named compatibility/runtime boundary."""

    require_gptqmodel_runtime(require_jit_toolchain=require_jit_toolchain)
    return importlib.import_module("gptqmodel")


__all__ = [
    "GPTQModelJITToolchainStatus",
    "GPTQModelRuntimeStatus",
    "import_gptqmodel",
    "inspect_gptqmodel_jit_toolchain",
    "prepare_gptqmodel_runtime",
    "require_gptqmodel_runtime",
]
