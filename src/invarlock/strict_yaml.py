"""Unambiguous YAML parsing over immutable regular-file snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import yaml

from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes


class StrictYamlError(ValueError):
    """Raised when YAML is unsafe, ambiguous, or not a stable file snapshot."""


_CORE_TAGS = frozenset(
    {
        "tag:yaml.org,2002:map",
        "tag:yaml.org,2002:seq",
        "tag:yaml.org,2002:str",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:null",
    }
)
_MERGE_TAG = "tag:yaml.org,2002:merge"


class StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader restricted to deterministic config-oriented YAML values."""

    allowed_custom_tags: frozenset[str] = frozenset()

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        if node.tag not in _CORE_TAGS | self.allowed_custom_tags:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"unsupported or ambiguous YAML tag {node.tag!r}",
                node.start_mark,
            )
        return super().construct_object(node, deep=deep)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG or key_node.value == "<<":
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "YAML merge keys are not supported",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "YAML mapping keys must be scalar and hashable",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate YAML key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return cast(dict[Any, Any], super().construct_mapping(node, deep=deep))


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictYamlError(f"{label} contains a non-finite YAML number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(key, label=label)
            _reject_nonfinite(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item, label=label)


def parse_yaml_bytes(
    payload: bytes,
    *,
    label: str,
    loader_cls: type[yaml.SafeLoader] = StrictSafeLoader,
) -> Any:
    """Parse exactly one UTF-8 YAML document with deterministic semantics."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictYamlError(f"{label} is not UTF-8 YAML") from exc
    try:
        value = yaml.load(text, Loader=loader_cls)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise StrictYamlError(f"{label} is not strict YAML: {exc}") from exc
    _reject_nonfinite(value, label=label)
    return value


def parse_yaml_documents_bytes(
    payload: bytes,
    *,
    label: str,
    loader_cls: type[yaml.SafeLoader] = StrictSafeLoader,
) -> list[Any]:
    """Parse every document in a UTF-8 YAML stream under strict semantics."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictYamlError(f"{label} is not UTF-8 YAML") from exc
    try:
        values = list(yaml.load_all(text, Loader=loader_cls))
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise StrictYamlError(f"{label} is not strict YAML: {exc}") from exc
    for value in values:
        _reject_nonfinite(value, label=label)
    return values


def read_yaml_snapshot(
    path: Path,
    *,
    label: str,
    loader_cls: type[yaml.SafeLoader] = StrictSafeLoader,
) -> tuple[bytes, Any]:
    """Read and parse one YAML file from the same regular-file byte snapshot."""

    try:
        payload = read_regular_file_bytes(path, label=label)
    except StrictJsonError as exc:
        raise StrictYamlError(str(exc)) from exc
    return payload, parse_yaml_bytes(payload, label=label, loader_cls=loader_cls)


def load_yaml_object(
    path: Path,
    *,
    label: str,
    loader_cls: type[yaml.SafeLoader] = StrictSafeLoader,
) -> dict[str, Any]:
    """Load a strict YAML mapping from one immutable file snapshot."""

    _, value = read_yaml_snapshot(path, label=label, loader_cls=loader_cls)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StrictYamlError(f"{label} must decode to a mapping")
    return value


__all__ = [
    "StrictSafeLoader",
    "StrictYamlError",
    "load_yaml_object",
    "parse_yaml_bytes",
    "parse_yaml_documents_bytes",
    "read_yaml_snapshot",
]
