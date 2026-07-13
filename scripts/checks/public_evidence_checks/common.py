"""Shared filesystem and privacy checks for public evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_EVIDENCE_ROOT = REPO_ROOT / "public_evidence"
PACKAGED_PUBLIC_EVIDENCE_INDEX = (
    REPO_ROOT
    / "src"
    / "invarlock"
    / "_data"
    / "public_evidence"
    / "published_basis_index.json"
)
META_FILENAME = "evidence.meta.json"
SCHEMA = "invarlock.public_evidence.meta.v1"
PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION = "public-evidence-index-v1"
CANONICAL_PACK_REPORT = (
    Path("evidence_pack") / "reports" / "report-001" / "evaluation.report.json"
)
MAX_DUPLICATE_ROOT_REPORT_BYTES = 0
PUBLIC_TEXT_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_EXECUTION_PATTERNS = (
    (
        "root_ssh_target",
        re.compile(r"\broot@[A-Za-z0-9._-]+\b"),
        "replace root SSH targets with a generic CUDA validation host label",
    ),
    (
        "private_ip_address",
        re.compile(
            r"(?<![A-Za-z0-9])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![A-Za-z0-9])"
        ),
        "replace private host IP addresses with a generic host label",
    ),
    (
        "absolute_root_path",
        re.compile(r"(?<![A-Za-z0-9._-])/root(?:/[^\s\"'`,)}\]]*)?"),
        "replace absolute root paths with generic validation-root placeholders",
    ),
    (
        "private_tmp_path",
        re.compile(r"(?<![A-Za-z0-9._-])/private/tmp(?:/[^\s\"'`,)}\]]*)?"),
        "replace private temporary paths with generic local-run placeholders",
    ),
    (
        "macos_user_home_path",
        re.compile(r"(?<![A-Za-z0-9._-])/Users/[A-Za-z0-9._-]+(?:/[^\s\"'`,)}\]]*)?"),
        "replace macOS user-home paths with generic validation-root placeholders",
    ),
    (
        "private_macos_var_folder_path",
        re.compile(r"(?<![A-Za-z0-9._-])/private/var/folders(?:/[^\s\"'`,)}\]]*)?"),
        "replace private macOS temporary paths with generic local-temp placeholders",
    ),
    (
        "macos_var_folder_path",
        re.compile(r"(?<![A-Za-z0-9._-])/var/folders(?:/[^\s\"'`,)}\]]*)?"),
        "replace macOS temporary paths with generic local-temp placeholders",
    ),
    (
        "home_directory_path",
        re.compile(r"(?<![A-Za-z0-9._-])/home/[A-Za-z0-9._-]+(?:/[^\s\"'`,)}\]]*)?"),
        "replace home-directory paths with generic validation-root placeholders",
    ),
    (
        "absolute_host_path",
        re.compile(r"(?:^|(?<=[\s\"'=:(\[{,]))/(?!/)[^\s\"'`,)}\]]+"),
        "remove host-local absolute paths from public evidence",
    ),
    (
        "windows_host_path",
        re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'`,)}\]]+"),
        "remove host-local Windows paths from public evidence",
    ),
    (
        "home_expansion_path",
        re.compile(r"(?<![A-Za-z0-9._/-])~[\\/][^\s\"'`,)}\]]+"),
        "remove home-directory expansions from public evidence",
    ),
    (
        "file_uri_path",
        re.compile(r"\bfile:(?://)?/[^\s\"'`,)}\]]*", re.IGNORECASE),
        "remove file URI paths from public evidence",
    ),
)

_CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_ENDPOINT_FIELD_NAMES = frozenset(
    {
        "api_endpoint",
        "api_url",
        "endpoint",
        "host",
        "provider_endpoint",
        "provider_url",
        "url",
    }
)
_PRIVATE_ENDPOINT_HOST = re.compile(
    r"(?:^|[.-])(?:internal|intranet|private|corp|local|lan)(?:$|[.-])",
    re.IGNORECASE,
)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"{path}: unable to read JSON: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path}: invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: expected JSON object"
    return payload, None


def _is_inside_special_dir(path: Path, root: Path) -> bool:
    parts = set(path.relative_to(root).parts)
    return bool(parts & {"artifact_package", "evidence_pack"})


def _artifact_dirs(root: Path) -> set[Path]:
    dirs: set[Path] = set()
    for metadata in root.rglob(META_FILENAME):
        if metadata.is_file() and not _is_inside_special_dir(metadata, root):
            dirs.add(metadata.parent)
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if _is_inside_special_dir(path, root):
            continue
        if path.name in {
            "evaluation.report.json",
            "runtime.manifest.json",
            "checkpoint_refs.json",
            "evidence_pack_recipe.json",
        }:
            dirs.add(path.parent)
    for manifest in root.rglob("evidence_pack/manifest.json"):
        dirs.add(manifest.parent.parent)
    return dirs


def _relative(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _directory_counts(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _resolve_public_evidence_path(public_evidence_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.parts and path.parts[0] == "public_evidence":
        return public_evidence_root.joinpath(*path.parts[1:])
    return REPO_ROOT / path


def _structured_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key, item in value.items():
            values.extend(_structured_string_values(key))
            values.extend(_structured_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_structured_string_values(item))
        return values
    return []


def _decoded_structured_strings(path: Path, text: str) -> list[str]:
    try:
        if path.suffix == ".json":
            documents = [json.loads(text)]
        elif path.suffix == ".jsonl":
            documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif path.suffix in {".yaml", ".yml"}:
            documents = list(yaml.safe_load_all(text))
        else:
            return []
    except (json.JSONDecodeError, yaml.YAMLError):
        return []
    values: list[str] = []
    for document in documents:
        values.extend(_structured_string_values(document))
    return values


def _structured_key_values(value: Any) -> list[tuple[str, Any]]:
    """Return JSON/YAML object field names and scalar values for privacy policy."""

    if isinstance(value, dict):
        pairs: list[tuple[str, Any]] = []
        for key, child in value.items():
            if isinstance(key, str):
                pairs.append((key, child))
            pairs.extend(_structured_key_values(child))
        return pairs
    if isinstance(value, (list, tuple, set)):
        return [pair for child in value for pair in _structured_key_values(child)]
    return []


def _is_private_endpoint(value: str) -> bool:
    parsed = urlparse(value)
    hostname = parsed.hostname
    if hostname is None:
        hostname = value.split("/", 1)[0].split(":", 1)[0]
    normalized = hostname.strip().strip(".").lower()
    return bool(
        normalized
        and (
            normalized == "localhost"
            or normalized.endswith((".internal", ".intranet", ".local", ".lan"))
            or _PRIVATE_ENDPOINT_HOST.search(normalized)
        )
    )


def _structured_privacy_errors(path: Path, text: str) -> list[str]:
    try:
        if path.suffix == ".json":
            documents = [json.loads(text)]
        elif path.suffix == ".jsonl":
            documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif path.suffix in {".yaml", ".yml"}:
            documents = list(yaml.safe_load_all(text))
        else:
            return []
    except (json.JSONDecodeError, yaml.YAMLError):
        return []
    errors: list[str] = []
    for key, value in (
        pair for document in documents for pair in _structured_key_values(document)
    ):
        normalized = key.strip().lower().replace("-", "_")
        if normalized in _CREDENTIAL_FIELD_NAMES:
            errors.append("credential_field")
        if (
            normalized in _ENDPOINT_FIELD_NAMES
            and isinstance(value, str)
            and _is_private_endpoint(value)
        ):
            errors.append("private_endpoint")
    return sorted(set(errors))


def _check_public_evidence_privacy(errors: list[str], root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in PUBLIC_TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{_relative(path)}: unable to scan public text: {exc}")
            continue
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            for name, pattern, message in PRIVATE_EXECUTION_PATTERNS:
                if pattern.search(line):
                    errors.append(f"{_relative(path)}:{line_number}: {name}: {message}")
        for value in _decoded_structured_strings(path, text):
            for name, pattern, message in PRIVATE_EXECUTION_PATTERNS:
                if not pattern.search(value) or value in text:
                    continue
                errors.append(f"{_relative(path)}: decoded value: {name}: {message}")
        for name in _structured_privacy_errors(path, text):
            if name == "credential_field":
                errors.append(
                    f"{_relative(path)}: credential_field: remove credential-bearing fields from public evidence"
                )
            else:
                errors.append(
                    f"{_relative(path)}: private_endpoint: replace private provider endpoints with public-safe identifiers"
                )


def _check_duplicate_root_evaluation_reports(errors: list[str], root: Path) -> None:
    duplicate_pairs: list[tuple[Path, Path, int]] = []
    for root_report in sorted(root.glob("*/*/evaluation.report.json")):
        pack_report = root_report.parent / CANONICAL_PACK_REPORT
        if not pack_report.is_file():
            continue
        try:
            root_bytes = root_report.read_bytes()
            pack_bytes = pack_report.read_bytes()
        except OSError as exc:
            errors.append(f"{_relative(root_report)}: unable to compare reports: {exc}")
            continue
        if root_bytes == pack_bytes:
            duplicate_pairs.append((root_report, pack_report, len(root_bytes)))

    duplicate_bytes = sum(size for _, _, size in duplicate_pairs)
    if duplicate_bytes <= MAX_DUPLICATE_ROOT_REPORT_BYTES:
        return
    errors.append(
        f"{_relative(root)}: duplicate root evaluation reports waste "
        f"{duplicate_bytes} bytes across {len(duplicate_pairs)} file(s); "
        f"budget is {MAX_DUPLICATE_ROOT_REPORT_BYTES} bytes"
    )
    for root_report, pack_report, _ in duplicate_pairs:
        errors.append(
            f"{_relative(root_report)}: duplicate of canonical pack report "
            f"{_relative(pack_report)}"
        )
