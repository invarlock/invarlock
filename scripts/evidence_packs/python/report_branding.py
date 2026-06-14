from __future__ import annotations

import re
from pathlib import Path

BRAND_NAME = "InvarLock"
BRAND_TAGLINE = "Auditable verification for edited model checkpoints."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _version_from_source() -> str | None:
    init_path = _repo_root() / "src" / "invarlock" / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.MULTILINE)
    return match.group(1) if match else None


def version_label() -> str:
    try:
        from invarlock import __version__
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        __version__ = _version_from_source()
    return __version__ if isinstance(__version__, str) and __version__ else "unknown"


def evidence_pack_text_header(title: str) -> list[str]:
    return [
        f"{BRAND_NAME.upper()} EVIDENCE PACK - {title.upper()}",
        f"{BRAND_NAME} {version_label()}",
        BRAND_TAGLINE,
    ]


def evidence_pack_markdown_header(title: str) -> list[str]:
    return [
        f"# {BRAND_NAME} {title}",
        "",
        f"*{BRAND_NAME} {version_label()} · {BRAND_TAGLINE}*",
        "",
    ]


__all__ = [
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "evidence_pack_markdown_header",
    "evidence_pack_text_header",
    "version_label",
]
