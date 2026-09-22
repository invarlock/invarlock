"""Render built package descriptions and check their public image elements."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import stat
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

try:
    from scripts.release.first_party_distribution_validation import _artifact_pair
    from scripts.release.package_readme import PROJECTS
    from scripts.release.release_distribution_validation import (
        MAX_ARCHIVE_MEMBERS,
        DistributionValidationSpec,
        ReleasePreflightError,
        _normalize_description,
        _read_tar_metadata,
        _read_zip_metadata,
        _require_regular_file,
        read_distribution_project,
    )
except ImportError:  # pragma: no cover - direct script execution
    from first_party_distribution_validation import _artifact_pair
    from package_readme import PROJECTS
    from release_distribution_validation import (
        MAX_ARCHIVE_MEMBERS,
        DistributionValidationSpec,
        ReleasePreflightError,
        _normalize_description,
        _read_tar_metadata,
        _read_zip_metadata,
        _require_regular_file,
        read_distribution_project,
    )


def _description(raw: bytes, spec: DistributionValidationSpec) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if message.defects:
        raise ReleasePreflightError("malformed package description metadata")
    for field, expected in (
        ("Name", spec.distribution_name),
        ("Version", spec.version),
        ("Description-Content-Type", "text/markdown"),
    ):
        if message.get_all(field) != [expected]:
            raise ReleasePreflightError(f"package description {field} mismatch")
    payload = message.get_payload(decode=True)
    if not isinstance(payload, bytes):
        raise ReleasePreflightError("package description is not text")
    try:
        description = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleasePreflightError("package description is not UTF-8") from exc
    if not description.strip():
        raise ReleasePreflightError("package description is empty")
    return description


class _Images(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.sources.append(dict(attrs).get("src"))


def _render(description: str, spec: DistributionValidationSpec) -> str:
    try:
        if importlib.metadata.version("readme-renderer") != "45.0":
            raise ReleasePreflightError("requires readme-renderer[md]==45.0")
        markdown = importlib.import_module("readme_renderer.markdown")
    except ImportError as exc:
        raise ReleasePreflightError("requires readme-renderer[md]==45.0") from exc
    rendered = markdown.render(description)
    if not rendered or not rendered.strip():
        raise ReleasePreflightError(
            "Markdown rendering failed; install readme-renderer[md]==45.0"
        )
    if spec.distribution_name == "invarlock":
        images = _Images()
        images.feed(rendered)
        for filename in ("invarlock-logo.svg", "evaluation-verification-flow.svg"):
            expected = (
                "https://raw.githubusercontent.com/invarlock/invarlock/"
                f"v{spec.version}/docs/assets/{filename}"
            )
            if images.sources.count(expected) != 1:
                raise ReleasePreflightError(
                    f"rendered core description needs one image: {expected}"
                )
    return rendered


def render_pair(spec: DistributionValidationSpec, wheel: Path, sdist: Path) -> str:
    """Read only the exact metadata members; never extract archive paths."""
    for path in (wheel, sdist):
        _require_regular_file(path, "description archive")
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        matching = [
            member
            for member in members
            if member.filename == f"{spec.dist_info_root}/METADATA"
        ]
        if len(members) > MAX_ARCHIVE_MEMBERS or len(matching) != 1:
            raise ReleasePreflightError("wheel description metadata is ambiguous")
        member = matching[0]
        if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
            raise ReleasePreflightError("wheel description metadata is not regular")
        wheel_description = _description(
            _read_zip_metadata(archive, member, label="wheel description"), spec
        )
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        matching = [
            member for member in members if member.name == f"{spec.sdist_root}/PKG-INFO"
        ]
        if len(members) > MAX_ARCHIVE_MEMBERS or len(matching) != 1:
            raise ReleasePreflightError("sdist description metadata is ambiguous")
        member = matching[0]
        if not member.isfile():
            raise ReleasePreflightError("sdist description metadata is not regular")
        sdist_description = _description(
            _read_tar_metadata(archive, member, label="sdist description"), spec
        )
    if _normalize_description(wheel_description) != _normalize_description(
        sdist_description
    ):
        raise ReleasePreflightError("wheel and sdist descriptions differ")
    rendered = _render(wheel_description, spec)
    _render(sdist_description, spec)
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--core-dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        for relative in PROJECTS:
            project = root / relative
            name, version = read_distribution_project(project)
            spec = DistributionValidationSpec(project, name, version, "")
            wheel, sdist = _artifact_pair(
                dist_dir=root / args.core_dist_dir, distribution_name=name
            )
            rendered = render_pair(spec, wheel, sdist)
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / f"{name}.html").write_text(
                    rendered, encoding="utf-8"
                )
            print(f"{name} {version}: wheel/sdist description rendering passed")
    except (
        ReleasePreflightError,
        OSError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
