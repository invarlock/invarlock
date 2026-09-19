"""Generate release-tagged package descriptions from branch-relative READMEs."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (".",)
LINK = re.compile(
    r'(?P<html>\b(?:href|src|srcset)=")(?P<url>[^"\n]+)(?P<end>")|(?P<md>!?\[[^\]\n]*\]\()(?P<target>[^)\s]+)(?P<close>\))'
)


def render(root: Path, project: Path) -> str:
    """Resolve local resources against a package's versioned repository tree."""
    version = tomllib.loads((project / "pyproject.toml").read_text())["project"][
        "version"
    ]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)", version):
        raise ValueError("invalid package version")
    source = (project / "README.md").read_text()

    def replace(match: re.Match[str]) -> str:
        url = match.group("url") or match.group("target")
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            if parsed.scheme not in {"https", "mailto"}:
                raise ValueError(f"unsupported README URL: {url}")
            repository_prefixes = {
                "github.com": (
                    "/invarlock/invarlock/blob/",
                    "/invarlock/invarlock/tree/",
                ),
                "raw.githubusercontent.com": ("/invarlock/invarlock/",),
            }
            if any(
                parsed.path.startswith(prefix + branch + "/")
                for prefix in repository_prefixes.get(parsed.hostname, ())
                for branch in ("main", "staging/next")
            ):
                raise ValueError(f"unversioned repository resource: {url}")
            return match.group()
        if not parsed.path:
            return match.group()
        target = (project / parsed.path).resolve()
        relative = target.relative_to(root.resolve()).as_posix()
        if not target.exists():
            raise ValueError(f"missing README resource: {url}")
        image = (match.group("html") or "").startswith(("src=", "srcset=")) or (
            match.group("md") or ""
        ).startswith("!")
        prefix = (
            "https://raw.githubusercontent.com/invarlock/invarlock"
            if image
            else "https://github.com/invarlock/invarlock/"
            + ("tree" if target.is_dir() else "blob")
        )
        resolved = f"{prefix}/v{version}/{relative}"
        if parsed.query:
            resolved += "?" + parsed.query
        if parsed.fragment:
            resolved += "#" + parsed.fragment
        return (
            (match.group("html") or match.group("md"))
            + resolved
            + (match.group("end") or match.group("close"))
        )

    def markup(part: str) -> str:
        # Keep the fallback in the surrounding HTML block. Removing only the
        # wrapper leaves blank lines and an indented Markdown code block.
        part = re.sub(
            r"<picture>(.*?)</picture>",
            lambda match: re.sub(r"<source\b[^>]*>", "", match[1]).strip(),
            part,
            flags=re.S,
        )
        part = re.sub(r"<source\b[^>]*>", "", part)
        return re.sub(r"(?m)^[ \t]+$", "", LINK.sub(replace, part))

    # Examples are literal shell/Python input, not rendered resource references.
    parts = re.split(r"(^```[^\n]*\n.*?^```[^\n]*(?:\n|$))", source, flags=re.M | re.S)
    return "".join(
        part if index % 2 else markup(part) for index, part in enumerate(parts)
    )


def synchronize(root: Path, *, write: bool) -> None:
    for relative in PROJECTS:
        project = root / relative
        expected = render(root, project)
        destination = project / "packaging/README.md"
        if write:
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(expected)
        elif not destination.is_file() or destination.read_text() != expected:
            raise ValueError(
                f"stale package README: {relative}; run scripts/release/package_readme.py --write"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    synchronize(ROOT, write=args.write)


if __name__ == "__main__":
    main()
