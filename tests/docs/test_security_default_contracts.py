from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_BOUNDARY_PATTERNS = (
    re.compile(r"\baiware\b", re.IGNORECASE),
    re.compile(r"\bplatform repo\b", re.IGNORECASE),
    re.compile(r"\bprivate repo\b", re.IGNORECASE),
    re.compile(r"\binternal repo\b", re.IGNORECASE),
    re.compile(r"\bcompanion repo\b", re.IGNORECASE),
    re.compile(r"\bsibling repo\b", re.IGNORECASE),
    re.compile(r"\bclosed-source\b", re.IGNORECASE),
    re.compile(r"\bcommercial platform\b", re.IGNORECASE),
    re.compile(r"\bproprietary platform\b", re.IGNORECASE),
)


def _iter_docs() -> list[Path]:
    return sorted((REPO_ROOT / "docs").rglob("*.md"))


def _iter_markdown_surfaces() -> list[Path]:
    surfaces = {
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
    }
    surfaces.update((REPO_ROOT / "docs").rglob("*.md"))
    surfaces.update((REPO_ROOT / "configs").rglob("*.md"))
    return sorted(path for path in surfaces if path.is_file())


def _iter_public_boundary_surfaces() -> list[Path]:
    surfaces = set(_iter_markdown_surfaces())
    for suffix in ("*.yaml", "*.yml"):
        surfaces.update((REPO_ROOT / "configs").rglob(suffix))
    surfaces.update((REPO_ROOT / "src" / "invarlock" / "cli").rglob("*.py"))
    surfaces.add(REPO_ROOT / "src" / "invarlock" / "core" / "report_inputs.py")
    return sorted(path for path in surfaces if path.is_file())


def _iter_notebooks() -> list[Path]:
    return sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))


def _iter_fenced_code_blocks(text: str) -> Iterable[str]:
    in_block = False
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if in_block:
                yield "\n".join(current)
                current = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            current.append(line)


def _has_model_loading_example(blocks: Iterable[str]) -> bool:
    return any(
        "invarlock run" in block
        or "invarlock evaluate" in block
        or "invarlock calibrate" in block
        for block in blocks
    )


def _has_verify_example(blocks: Iterable[str]) -> bool:
    return any("invarlock verify" in block for block in blocks)


def test_markdown_surfaces_with_verify_examples_explain_attestation() -> None:
    missing: list[str] = []
    for path in _iter_markdown_surfaces():
        text = path.read_text(encoding="utf-8")
        if not _has_verify_example(_iter_fenced_code_blocks(text)):
            continue
        if "runtime.manifest.json" in text:
            continue
        missing.append(str(path.relative_to(REPO_ROOT)))

    assert not missing, (
        "Markdown surfaces with verify examples must mention runtime.manifest.json: "
        + ", ".join(missing)
    )


def test_markdown_surfaces_with_model_loading_examples_explain_execution_context() -> (
    None
):
    missing: list[str] = []
    for path in _iter_markdown_surfaces():
        text = path.read_text(encoding="utf-8")
        if not _has_model_loading_example(_iter_fenced_code_blocks(text)):
            continue
        if any(
            marker in text
            for marker in (
                "--assurance trusted-local",
                "--assurance attested",
                "--allow-host-execution",
                "INVARLOCK_ALLOW_HOST_EXECUTION=1",
                "runtime container",
                "runtime-container",
            )
        ):
            continue
        missing.append(str(path.relative_to(REPO_ROOT)))

    assert not missing, (
        "Markdown surfaces with model-loading examples must explain secure-default container execution or explicit host bypass: "
        + ", ".join(missing)
    )


def test_deprecated_plugin_disable_env_is_absent_from_repo_surfaces() -> None:
    hits: list[str] = []
    scan_roots = [
        REPO_ROOT / ".github",
        REPO_ROOT / "docs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "src" / "invarlock",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "Makefile",
    ]
    for root in scan_roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "INVARLOCK_DISABLE_PLUGIN_DISCOVERY" not in text:
                continue
            hits.append(str(path.relative_to(REPO_ROOT)))

    assert not hits, (
        "Deprecated plugin disable env should not appear in repo surfaces: "
        + ", ".join(hits)
    )


def test_public_docs_use_logical_runtime_tiers_path() -> None:
    hits: list[str] = []
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        if (
            "invarlock._data.runtime/tiers.yaml" in text
            or "src/invarlock/_data/runtime/tiers.yaml" in text
        ):
            hits.append(str(path.relative_to(REPO_ROOT)))

    assert not hits, (
        "Public docs must refer to runtime tiers via logical runtime/tiers.yaml path: "
        + ", ".join(hits)
    )


def test_public_docs_and_workflows_do_not_teach_unattested_verify_bypass() -> None:
    hits: list[str] = []
    for root in (REPO_ROOT / ".github", REPO_ROOT / "docs"):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if (
                "--allow-unattested-artifacts" in text
                or "INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS" in text
            ):
                hits.append(str(path.relative_to(REPO_ROOT)))

    assert not hits, (
        "Public docs and workflows must not teach unattested verify bypasses: "
        + ", ".join(hits)
    )


def test_public_docs_do_not_teach_cert_terminology_for_proof_packs() -> None:
    hits: list[str] = []
    option_pattern = re.compile(r"(?<![A-Za-z0-9_-])--cert(?![A-Za-z0-9_-])")
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        if "proof-pack" not in text:
            continue
        if option_pattern.search(text) or any(
            marker in text
            for marker in (
                " bundled cert",
                " clean cert",
                " error-injection cert",
                " certs/",
            )
        ):
            hits.append(str(path.relative_to(REPO_ROOT)))

    assert not hits, (
        "Public proof-pack docs must use report terminology instead of cert terminology: "
        + ", ".join(hits)
    )


def test_public_surfaces_do_not_use_repo_boundary_language() -> None:
    hits: list[str] = []
    for path in _iter_public_boundary_surfaces():
        text = path.read_text(encoding="utf-8")
        for pattern in PUBLIC_BOUNDARY_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path.relative_to(REPO_ROOT)} -> {pattern.pattern}")

    assert not hits, (
        "Public surfaces must stay standalone and repo-agnostic: "
        + ", ".join(sorted(hits))
    )


def test_proof_pack_remote_code_requests_are_explicit() -> None:
    bad: list[str] = []
    for path in sorted((REPO_ROOT / "scripts" / "proof_packs").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        if "/tests/" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        if "trust_remote_code" not in text:
            continue
        if any(
            marker in text
            for marker in (
                "INVARLOCK_ALLOW_REMOTE_CODE",
                "require_remote_code_opt_in",
                "pack_remote_code_allowed",
                "pack_model_trust_remote_code_yaml",
            )
        ):
            continue
        bad.append(str(path.relative_to(REPO_ROOT)))

    assert not bad, (
        "Proof-pack helpers must wire remote code through explicit opt-in controls: "
        + ", ".join(bad)
    )


def test_notebooks_with_model_loading_examples_explain_execution_context() -> None:
    missing: list[str] = []
    for path in _iter_notebooks():
        text = path.read_text(encoding="utf-8")
        if not any(
            token in text
            for token in (
                "invarlock run",
                "invarlock evaluate",
                "invarlock calibrate",
            )
        ):
            continue
        if any(
            marker in text
            for marker in (
                "--assurance trusted-local",
                "--assurance attested",
                "--allow-host-execution",
                "INVARLOCK_ALLOW_HOST_EXECUTION=1",
                "runtime container",
                "runtime-container",
            )
        ):
            continue
        missing.append(str(path.relative_to(REPO_ROOT)))

    assert not missing, (
        "Notebooks with model-loading examples must explain secure-default container execution or explicit host bypass: "
        + ", ".join(missing)
    )


def test_notebooks_with_verify_examples_explain_attestation_or_bypass() -> None:
    missing_manifest_context: list[str] = []
    missing_bypass_context: list[str] = []
    for path in _iter_notebooks():
        text = path.read_text(encoding="utf-8")
        if "invarlock verify" not in text:
            continue
        if "runtime.manifest.json" not in text:
            missing_manifest_context.append(str(path.relative_to(REPO_ROOT)))
        if "invarlock verify --assurance trusted-local" in text:
            continue
        if "runtime container" in text or "runtime-container" in text:
            continue
        missing_bypass_context.append(str(path.relative_to(REPO_ROOT)))

    assert not missing_manifest_context, (
        "Notebooks with verify examples must mention runtime.manifest.json or its absence: "
        + ", ".join(missing_manifest_context)
    )
    assert not missing_bypass_context, (
        "Notebook verify examples must either use the explicit trusted-local assurance mode or explain the runtime-container path: "
        + ", ".join(missing_bypass_context)
    )
