"""Strict contract loader for one baseline-versus-subject evaluation."""

from __future__ import annotations

import errno
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, TypeAliasType, cast

import jsonschema
import yaml

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    ModelRuntimeSpec,
    RuntimeProvider,
    RuntimeTask,
    require_runtime_task,
)
from invarlock.core.scorer_extension import (
    ScorerExtensionBinding,
    ScorerExtensionError,
    decode_scorer_binding,
)
from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes
from invarlock.public_contracts import (
    EVALUATION_REQUEST_FORMAT_VERSION as EVALUATION_REQUEST_FORMAT,
)
from invarlock.public_contracts import load_evaluation_request_schema

from .schedule_preparation import LocalDatasetRequest

MAX_EVALUATION_REQUEST_BYTES = 1024 * 1024
MAX_EVALUATION_REQUEST_NODES = 10_000
MAX_EVALUATION_REQUEST_DEPTH = 64

type JSONScalar = str | int | float | bool | None
ProviderResolver = TypeAliasType(  # noqa: UP040
    "ProviderResolver", Callable[[str], RuntimeProvider]
)

_JSON_INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_JSON_FLOAT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_FORBIDDEN_INCLUDE_KEYS = frozenset(
    {"include", "includes", "include_file", "include_files", "extends", "ref"}
)
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


class EvaluationRequestError(ValueError):
    """Raised when an evaluation request is ambiguous, unsafe, or invalid."""


class _StrictRequestYamlLoader(yaml.SafeLoader):
    """Safe YAML loader with unambiguous JSON-compatible scalar semantics."""


def _construct_mapping(
    loader: _StrictRequestYamlLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge" or key_node.value == "<<":
            raise EvaluationRequestError("request YAML merge keys are not allowed")
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or not key:
            raise EvaluationRequestError(
                "request YAML object keys must be non-empty strings"
            )
        if key in result:
            raise EvaluationRequestError(f"request YAML has duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


def _construct_bool(_loader: _StrictRequestYamlLoader, node: yaml.ScalarNode) -> bool:
    if node.value not in {"true", "false"}:
        raise EvaluationRequestError(
            "request YAML booleans must use lowercase true or false"
        )
    return bool(node.value == "true")


def _construct_int(_loader: _StrictRequestYamlLoader, node: yaml.ScalarNode) -> int:
    if _JSON_INTEGER_RE.fullmatch(node.value) is None:
        raise EvaluationRequestError(
            "request YAML integers must use canonical JSON syntax"
        )
    return int(node.value)


def _construct_float(_loader: _StrictRequestYamlLoader, node: yaml.ScalarNode) -> float:
    if _JSON_FLOAT_RE.fullmatch(node.value) is None:
        raise EvaluationRequestError("request YAML numbers must use finite JSON syntax")
    value = float(node.value)
    if not math.isfinite(value):
        raise EvaluationRequestError("request YAML numbers must be finite")
    return value


def _construct_null(_loader: _StrictRequestYamlLoader, node: yaml.ScalarNode) -> None:
    if node.value != "null":
        raise EvaluationRequestError("request YAML nulls must use lowercase null")
    return None


_StrictRequestYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)
for _tag, _constructor in (
    ("tag:yaml.org,2002:bool", _construct_bool),
    ("tag:yaml.org,2002:int", _construct_int),
    ("tag:yaml.org,2002:float", _construct_float),
    ("tag:yaml.org,2002:null", _construct_null),
):
    _StrictRequestYamlLoader.add_constructor(_tag, _constructor)


@dataclass(frozen=True)
class RuntimeRequest:
    provider: str
    settings: Mapping[str, JSONScalar]


@dataclass(frozen=True)
class ArtifactRequest:
    path: Path | None
    model_id: str
    locator: str | None


@dataclass(frozen=True)
class ComparisonSideRequest:
    artifact: ArtifactRequest
    runtime: RuntimeRequest


@dataclass(frozen=True)
class ComparisonRequest:
    baseline: ComparisonSideRequest
    subject: ComparisonSideRequest
    dataset: Path | LocalDatasetRequest
    policy: Path
    task: RuntimeTask
    metric: Literal["exact_match", "normalized_nll_per_utf8_byte"] | None
    scorer_extension: ScorerExtensionBinding | None = None

    @property
    def collection_metric(
        self,
    ) -> Literal["exact_match", "normalized_nll_per_utf8_byte"]:
        """Provider-owned facts to collect before verifier replay."""

        if self.scorer_extension is not None:
            return "exact_match"
        if self.metric is None:  # pragma: no cover - loader enforces exclusivity
            raise EvaluationRequestError("comparison metric selection is invalid")
        return self.metric


@dataclass(frozen=True)
class ImportSideRequest:
    identity: Path
    receipt: Path
    observation: Path
    run_report: Path
    runtime_manifest: Path
    runtime_config: Path


@dataclass(frozen=True)
class ExecutionRequest:
    mode: Literal["run", "import"]
    records: Path | None
    schedule: Path | None
    baseline: ImportSideRequest | None
    subject: ImportSideRequest | None


@dataclass(frozen=True)
class OutputRequest:
    evidence: Path


@dataclass(frozen=True)
class ObservationRequest:
    """One root-confined observation payload attached by the host transaction."""

    observation_id: str
    kind: str
    scope: Literal["baseline", "subject", "comparison"]
    path: Path


@dataclass(frozen=True)
class EvaluationRequest:
    format_version: str
    root: Path
    comparison: ComparisonRequest
    execution: ExecutionRequest
    output: OutputRequest
    observations: tuple[ObservationRequest, ...] = ()


def _scan_yaml_limits_and_features(text: str) -> None:
    collection_starts = (
        yaml.tokens.BlockMappingStartToken,
        yaml.tokens.BlockSequenceStartToken,
        yaml.tokens.FlowMappingStartToken,
        yaml.tokens.FlowSequenceStartToken,
    )
    collection_ends = (
        yaml.tokens.BlockEndToken,
        yaml.tokens.FlowMappingEndToken,
        yaml.tokens.FlowSequenceEndToken,
    )
    node_count = 0
    depth = 0
    try:
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)):
                raise EvaluationRequestError(
                    "request YAML aliases and anchors are not allowed"
                )
            if isinstance(token, (yaml.tokens.TagToken, yaml.tokens.DirectiveToken)):
                raise EvaluationRequestError(
                    "request YAML explicit tags and directives are not allowed"
                )
            if isinstance(token, collection_starts):
                node_count += 1
                depth += 1
                if depth > MAX_EVALUATION_REQUEST_DEPTH:
                    raise EvaluationRequestError(
                        "request exceeds the 64-level nesting limit"
                    )
            elif isinstance(token, collection_ends):
                depth = max(0, depth - 1)
            elif isinstance(token, yaml.tokens.ScalarToken):
                node_count += 1
            if node_count > MAX_EVALUATION_REQUEST_NODES:
                raise EvaluationRequestError("request exceeds the 10,000-node limit")
    except EvaluationRequestError:
        raise
    except (RecursionError, yaml.YAMLError) as exc:
        raise EvaluationRequestError(
            f"request YAML could not be scanned: {exc}"
        ) from exc


def _load_yaml(payload: bytes) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationRequestError("request must be UTF-8 encoded") from exc
    _scan_yaml_limits_and_features(text)
    try:
        return yaml.load(text, Loader=_StrictRequestYamlLoader)
    except EvaluationRequestError:
        raise
    except (RecursionError, yaml.YAMLError) as exc:
        raise EvaluationRequestError(
            f"request YAML could not be decoded: {exc}"
        ) from exc


def _reject_include_directives(value: Any) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if isinstance(key, str) and key.lower() in _FORBIDDEN_INCLUDE_KEYS:
                    raise EvaluationRequestError(
                        "request YAML include directives are not allowed"
                    )
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)


def _schema_error_message(error: jsonschema.ValidationError) -> str:
    pending = [error]
    details: list[str] = []
    while pending:
        candidate = pending.pop(0)
        path = ".".join(str(component) for component in candidate.absolute_path)
        prefix = f"{path}: " if path else ""
        details.append(prefix + candidate.message)
        pending.extend(candidate.context)
    return "; ".join(dict.fromkeys(details))


def _validate_schema(value: Any) -> dict[str, Any]:
    schema = load_evaluation_request_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: tuple(str(component) for component in error.absolute_path),
    )
    if errors:
        raise EvaluationRequestError(
            "request does not match evaluation_request.schema.json: "
            + _schema_error_message(errors[0])
        )
    return cast(dict[str, Any], value)


def _reference_parts(reference: str, *, label: str) -> tuple[str, ...]:
    if (
        not reference
        or reference.startswith(("/", "~", "\\"))
        or _WINDOWS_DRIVE_RE.match(reference) is not None
        or "://" in reference
        or "\\" in reference
    ):
        raise EvaluationRequestError(f"{label} must be a safe relative reference")
    parts = PurePosixPath(reference).parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "//" in reference
        or reference.endswith("/")
    ):
        raise EvaluationRequestError(f"{label} must be a safe relative reference")
    return tuple(parts)


def _is_symlink_at(parent_fd: int, name: str) -> bool:
    try:
        return stat.S_ISLNK(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode
        )
    except OSError:
        return False


def _unsafe_component_error(
    *, label: str, parent_fd: int, component: str, exc: OSError
) -> EvaluationRequestError:
    if exc.errno == errno.ELOOP or _is_symlink_at(parent_fd, component):
        return EvaluationRequestError(f"{label} traverses a symlink")
    return EvaluationRequestError(f"{label} is not a root-confined regular path: {exc}")


def _resolve_existing_reference(
    root: Path,
    reference: str,
    *,
    label: str,
    expected: Literal["file", "artifact"],
) -> Path:
    parts = _reference_parts(reference, label=label)
    root_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    current_fd = root_fd
    try:
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            flags = _FILE_OPEN_FLAGS if final else _DIRECTORY_OPEN_FLAGS
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise _unsafe_component_error(
                    label=label,
                    parent_fd=current_fd,
                    component=component,
                    exc=exc,
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        mode = os.fstat(current_fd).st_mode
        if expected == "file" and not stat.S_ISREG(mode):
            raise EvaluationRequestError(f"{label} must reference a regular file")
        if expected == "artifact" and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise EvaluationRequestError(
                f"{label} must reference a regular file or directory"
            )
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    return root.joinpath(*parts)


def _resolve_output_reference(root: Path, reference: str, *, label: str) -> Path:
    parts = _reference_parts(reference, label=label)
    root_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    current_fd = root_fd
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                return root.joinpath(*parts)
            except OSError as exc:
                raise _unsafe_component_error(
                    label=label,
                    parent_fd=current_fd,
                    component=component,
                    exc=exc,
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        destination = parts[-1]
        try:
            os.stat(destination, dir_fd=current_fd, follow_symlinks=False)
        except FileNotFoundError:
            return root.joinpath(*parts)
        except OSError as exc:
            raise EvaluationRequestError(f"{label} cannot be inspected: {exc}") from exc
        raise EvaluationRequestError(f"{label} already exists")
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _default_provider_resolver(provider_name: str) -> RuntimeProvider:
    if provider_name != "hf_transformers":
        raise EvaluationRequestError(
            f"runtime provider {provider_name!r} is not installed or authorized"
        )
    from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

    return HFTransformersProvider()


def _resolve_provider(
    provider_name: str,
    *,
    resolver: ProviderResolver,
) -> RuntimeProvider:
    try:
        provider = resolver(provider_name)
    except EvaluationRequestError:
        raise
    except (KeyError, LookupError, ModuleNotFoundError) as exc:
        raise EvaluationRequestError(
            f"runtime provider {provider_name!r} is not installed or authorized"
        ) from exc
    except Exception as exc:
        raise EvaluationRequestError(
            f"runtime provider {provider_name!r} could not be resolved: {exc}"
        ) from exc
    if provider.name != provider_name:
        raise EvaluationRequestError(
            f"runtime provider identity mismatch: requested {provider_name!r}, "
            f"resolved {provider.name!r}"
        )
    if provider.abi_version != INVARLOCK_RUNTIME_PROVIDER_ABI:
        raise EvaluationRequestError(
            f"runtime provider {provider_name!r} uses unsupported ABI "
            f"{provider.abi_version!r}"
        )
    return provider


def _build_runtime(
    runtime: dict[str, Any],
    artifact: dict[str, Any],
    *,
    side_name: str,
    provider_cache: dict[str, RuntimeProvider],
    provider_resolver: ProviderResolver,
) -> RuntimeRequest:
    provider_name = cast(str, runtime["provider"])
    settings = MappingProxyType(
        cast(dict[str, JSONScalar], dict(cast(dict[str, Any], runtime["settings"])))
    )
    provider = provider_cache.get(provider_name)
    if provider is None:
        provider = _resolve_provider(provider_name, resolver=provider_resolver)
        provider_cache[provider_name] = provider
    try:
        spec = ModelRuntimeSpec(
            provider_name=provider_name,
            model_id=cast(str, artifact["model_id"]),
            settings=settings,
        )
        provider.validate_config(spec)
    except (TypeError, ValueError) as exc:
        raise EvaluationRequestError(
            f"comparison.{side_name}.runtime is invalid: {exc}"
        ) from exc
    return RuntimeRequest(provider=provider_name, settings=spec.settings)


def _build_side(
    value: dict[str, Any],
    *,
    side_name: str,
    execution_mode: Literal["run", "import"],
    root: Path,
    provider_cache: dict[str, RuntimeProvider],
    provider_resolver: ProviderResolver,
) -> ComparisonSideRequest:
    artifact = cast(dict[str, Any], value["artifact"])
    runtime = cast(dict[str, Any], value["runtime"])
    artifact_path = cast(str | None, artifact.get("path"))
    locator = cast(str | None, artifact.get("locator"))
    if execution_mode == "run":
        if artifact_path is None:
            raise EvaluationRequestError(
                f"comparison.{side_name}.artifact.path is required in run mode"
            )
        resolved_artifact_path = _resolve_existing_reference(
            root,
            artifact_path,
            label=f"comparison.{side_name}.artifact.path",
            expected="artifact",
        )
    else:
        if locator is None:
            raise EvaluationRequestError(
                f"comparison.{side_name}.artifact.locator is required in import mode"
            )
        resolved_artifact_path = None
    return ComparisonSideRequest(
        artifact=ArtifactRequest(
            path=resolved_artifact_path,
            model_id=cast(str, artifact["model_id"]),
            locator=locator,
        ),
        runtime=_build_runtime(
            runtime,
            artifact,
            side_name=side_name,
            provider_cache=provider_cache,
            provider_resolver=provider_resolver,
        ),
    )


def _build_import_side(
    value: dict[str, Any], *, side_name: str, root: Path
) -> ImportSideRequest:
    prefix = f"execution.{side_name}"
    return ImportSideRequest(
        identity=_resolve_existing_reference(
            root,
            cast(str, value["identity"]),
            label=f"{prefix}.identity",
            expected="file",
        ),
        receipt=_resolve_existing_reference(
            root,
            cast(str, value["receipt"]),
            label=f"{prefix}.receipt",
            expected="file",
        ),
        observation=_resolve_existing_reference(
            root,
            cast(str, value["observation"]),
            label=f"{prefix}.observation",
            expected="file",
        ),
        run_report=_resolve_existing_reference(
            root,
            cast(str, value["run_report"]),
            label=f"{prefix}.run_report",
            expected="file",
        ),
        runtime_manifest=_resolve_existing_reference(
            root,
            cast(str, value["runtime_manifest"]),
            label=f"{prefix}.runtime_manifest",
            expected="file",
        ),
        runtime_config=_resolve_existing_reference(
            root,
            cast(str, value["runtime_config"]),
            label=f"{prefix}.runtime_config",
            expected="file",
        ),
    )


def _build_dataset(
    value: object,
    *,
    execution_mode: Literal["run", "import"],
    root: Path,
) -> Path | LocalDatasetRequest:
    if execution_mode == "import":
        if not isinstance(value, str):
            raise EvaluationRequestError(
                "comparison.dataset must reference the canonical schedule in import mode"
            )
        return _resolve_existing_reference(
            root,
            value,
            label="comparison.dataset",
            expected="file",
        )
    if not isinstance(value, dict):
        raise EvaluationRequestError(
            "comparison.dataset must be a pinned local dataset object in run mode"
        )
    try:
        return LocalDatasetRequest(
            path=_resolve_existing_reference(
                root,
                cast(str, value["path"]),
                label="comparison.dataset.path",
                expected="file",
            ),
            sha256=cast(str, value["sha256"]),
            format=cast(str, value["format"]),
            name=cast(str, value["name"]),
            split=cast(str, value["split"]),
            input_field=cast(str, value["input_field"]),
            expected_output_field=cast(str, value["expected_output_field"]),
            id_field=cast(str | None, value.get("id_field")),
            content_role=cast(str | None, value.get("content_role")),
            content_id_field=cast(str | None, value.get("content_id_field")),
            content_sha256_field=cast(str | None, value.get("content_sha256_field")),
            content_byte_length_field=cast(
                str | None, value.get("content_byte_length_field")
            ),
            content_media_type_field=cast(
                str | None, value.get("content_media_type_field")
            ),
            limit=cast(int | None, value.get("limit")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationRequestError(f"comparison.dataset is invalid: {exc}") from exc


def _build_request(
    value: dict[str, Any],
    *,
    root: Path,
    provider_resolver: ProviderResolver,
) -> EvaluationRequest:
    comparison = cast(dict[str, Any], value["comparison"])
    execution = cast(dict[str, Any], value["execution"])
    output = cast(dict[str, Any], value["output"])
    execution_mode = cast(Literal["run", "import"], execution["mode"])

    observation_requests: list[ObservationRequest] = []
    observation_ids: set[str] = set()
    for index, item in enumerate(value.get("observations", [])):
        if not isinstance(item, dict) or "path" not in item:
            raise EvaluationRequestError(
                f"observations[{index}] must be an authored observation with path"
            )
        observation_id = cast(str, item["id"])
        if observation_id in observation_ids:
            raise EvaluationRequestError(
                f"duplicate observation id: {observation_id!r}"
            )
        observation_ids.add(observation_id)
        observation_requests.append(
            ObservationRequest(
                observation_id=observation_id,
                kind=cast(str, item["kind"]),
                scope=cast(Literal["baseline", "subject", "comparison"], item["scope"]),
                path=_resolve_existing_reference(
                    root,
                    cast(str, item["path"]),
                    label=f"observations[{index}].path",
                    expected="file",
                ),
            )
        )
    provider_cache: dict[str, RuntimeProvider] = {}
    baseline = _build_side(
        cast(dict[str, Any], comparison["baseline"]),
        side_name="baseline",
        execution_mode=execution_mode,
        root=root,
        provider_cache=provider_cache,
        provider_resolver=provider_resolver,
    )
    subject = _build_side(
        cast(dict[str, Any], comparison["subject"]),
        side_name="subject",
        execution_mode=execution_mode,
        root=root,
        provider_cache=provider_cache,
        provider_resolver=provider_resolver,
    )
    metric = cast(
        Literal["exact_match", "normalized_nll_per_utf8_byte"] | None,
        comparison.get("metric"),
    )
    try:
        scorer_extension = (
            decode_scorer_binding(comparison["scorer_extension"])
            if "scorer_extension" in comparison
            else None
        )
    except ScorerExtensionError as exc:
        raise EvaluationRequestError(str(exc)) from exc
    if (metric is None) == (scorer_extension is None):
        raise EvaluationRequestError(
            "comparison must select exactly one built-in metric or scorer_extension"
        )
    collection_metric = "exact_match" if scorer_extension is not None else metric
    assert collection_metric is not None
    try:
        task = require_runtime_task(comparison["task"], field_name="comparison.task")
    except ValueError as exc:
        raise EvaluationRequestError(str(exc)) from exc
    for side_name, side in (("baseline", baseline), ("subject", subject)):
        provider = provider_cache[side.runtime.provider]
        if task not in provider.capabilities().tasks:
            raise EvaluationRequestError(
                f"comparison.{side_name}.runtime provider {provider.name!r} "
                f"does not support task {task!r}"
            )
        if collection_metric not in provider.capabilities().metrics:
            raise EvaluationRequestError(
                f"comparison.{side_name}.runtime provider {provider.name!r} "
                f"does not support metric {collection_metric!r}"
            )

    if execution_mode == "import":
        execution_request = ExecutionRequest(
            mode="import",
            records=_resolve_existing_reference(
                root,
                cast(str, execution["records"]),
                label="execution.records",
                expected="file",
            ),
            schedule=_resolve_existing_reference(
                root,
                cast(str, execution["schedule"]),
                label="execution.schedule",
                expected="file",
            ),
            baseline=_build_import_side(
                cast(dict[str, Any], execution["baseline"]),
                side_name="baseline",
                root=root,
            ),
            subject=_build_import_side(
                cast(dict[str, Any], execution["subject"]),
                side_name="subject",
                root=root,
            ),
        )
    else:
        execution_request = ExecutionRequest(
            mode="run",
            records=None,
            schedule=None,
            baseline=None,
            subject=None,
        )
    return EvaluationRequest(
        format_version=cast(str, value["format_version"]),
        root=root,
        comparison=ComparisonRequest(
            baseline=baseline,
            subject=subject,
            dataset=_build_dataset(
                comparison["dataset"],
                execution_mode=execution_mode,
                root=root,
            ),
            policy=_resolve_existing_reference(
                root,
                cast(str, comparison["policy"]),
                label="comparison.policy",
                expected="file",
            ),
            task=task,
            metric=metric,
            scorer_extension=scorer_extension,
        ),
        execution=execution_request,
        observations=tuple(observation_requests),
        output=OutputRequest(
            evidence=_resolve_output_reference(
                root,
                cast(str, output["evidence"]),
                label="output.evidence",
            )
        ),
    )


def load_evaluation_request(
    path: str | Path,
    *,
    provider_resolver: ProviderResolver | None = None,
    request_root: Path | None = None,
) -> EvaluationRequest:
    """Load one strict request anchored to its file's real parent directory.

    The built-in resolver intentionally exposes only the canonical Hugging Face
    provider. Optional provider add-ins must supply an explicit, authorized
    resolver. Callers must repeat no-follow resolution when opening returned
    paths so a later filesystem mutation cannot cross the request root.
    """

    request_path = Path(path)
    try:
        payload = read_regular_file_bytes(
            request_path,
            label="evaluation request",
            max_bytes=MAX_EVALUATION_REQUEST_BYTES,
        )
        root = (request_root or request_path.parent).resolve(strict=True)
    except (OSError, StrictJsonError) as exc:
        raise EvaluationRequestError(str(exc)) from exc
    value = _load_yaml(payload)
    _reject_include_directives(value)
    validated = _validate_schema(value)
    return _build_request(
        validated,
        root=root,
        provider_resolver=provider_resolver or _default_provider_resolver,
    )


__all__ = [
    "EVALUATION_REQUEST_FORMAT",
    "MAX_EVALUATION_REQUEST_BYTES",
    "MAX_EVALUATION_REQUEST_DEPTH",
    "MAX_EVALUATION_REQUEST_NODES",
    "ArtifactRequest",
    "ComparisonRequest",
    "ComparisonSideRequest",
    "EvaluationRequest",
    "EvaluationRequestError",
    "ExecutionRequest",
    "ImportSideRequest",
    "LocalDatasetRequest",
    "ObservationRequest",
    "OutputRequest",
    "ProviderResolver",
    "RuntimeRequest",
    "load_evaluation_request",
]
