"""Exercise the release renderer against actual archive metadata and Markdown."""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile

import pytest

from scripts.release import package_rendering as rendering


def _core_description(version="1.2.3"):
    base = (
        f"https://raw.githubusercontent.com/invarlock/invarlock/v{version}/docs/assets"
    )
    return (
        '<p align="center">\n'
        f'  <img src="{base}/invarlock-logo.svg" alt="InvarLock">\n'
        "</p>\n\n# Release\n\n"
        f"![Workflow]({base}/evaluation-verification-flow.svg)\n"
    )


def _pair(
    tmp_path,
    description=None,
    *,
    name="invarlock",
    sdist_description=None,
    wheel_path=None,
    sdist_link=False,
    metadata_overrides=None,
):
    description = _core_description() if description is None else description
    spec = rendering.DistributionValidationSpec(tmp_path, name, "1.2.3", "")
    fields = {
        "Metadata-Version": "2.4",
        "Name": name,
        "Version": spec.version,
        "Description-Content-Type": "text/markdown",
    }
    fields.update(metadata_overrides or {})

    def metadata(body):
        return (
            "".join(f"{key}: {value}\n" for key, value in fields.items()) + "\n" + body
        ).encode("utf-8")

    tmp_path.mkdir(parents=True, exist_ok=True)
    wheel = tmp_path / f"{spec.normalized_name}-1.2.3-py3-none-any.whl"
    sdist = tmp_path / f"{spec.sdist_root}.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            wheel_path or f"{spec.dist_info_root}/METADATA", metadata(description)
        )
    with tarfile.open(sdist, "w:gz") as archive:
        raw = metadata(description if sdist_description is None else sdist_description)
        member = tarfile.TarInfo(f"{spec.sdist_root}/PKG-INFO")
        if sdist_link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        else:
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return spec, wheel, sdist


def test_actual_archive_descriptions_render_images_and_unicode(tmp_path):
    result = rendering.render_pair(*_pair(tmp_path, _core_description() + "\nCafé.\n"))
    assert "Café" in result
    images = rendering._Images()
    images.feed(result)
    assert len(images.sources) == 2
    assert all("/v1.2.3/" in source for source in images.sources)


def test_original_indented_logo_regression_fails(tmp_path):
    # Removing the picture wrapper left a blank line and four-space indentation.
    broken = _core_description().replace("\n  <img", "\n\n    <img")
    with pytest.raises(rendering.ReleasePreflightError, match="invarlock-logo.svg"):
        rendering.render_pair(*_pair(tmp_path, broken))


@pytest.mark.parametrize(
    "change", ["missing_logo", "missing_workflow", "wrong_version", "wrong_host"]
)
def test_missing_or_incorrect_image_sources_fail(tmp_path, change):
    description = _core_description()
    if change == "missing_logo":
        description = "\n".join(
            line for line in description.splitlines() if "<img" not in line
        )
    elif change == "missing_workflow":
        description = "\n".join(
            line for line in description.splitlines() if "![Workflow]" not in line
        )
    elif change == "wrong_version":
        description = description.replace("/v1.2.3/", "/v1.2.2/")
    else:
        description = description.replace("raw.githubusercontent.com", "example.com")
    with pytest.raises(rendering.ReleasePreflightError, match="needs one image"):
        rendering.render_pair(*_pair(tmp_path, description))


def test_non_core_description_does_not_require_core_images(tmp_path):
    rendered = rendering.render_pair(
        *_pair(tmp_path, "# Diagnostics\n", name="invarlock-diagnostics")
    )
    assert "<h1" in rendered


def test_different_archive_descriptions_fail(tmp_path):
    with pytest.raises(rendering.ReleasePreflightError, match="descriptions differ"):
        rendering.render_pair(
            *_pair(tmp_path, sdist_description=_core_description() + "Changed\n")
        )


def test_archive_newline_conventions_are_allowed(tmp_path):
    description = _core_description().rstrip("\n")
    assert rendering.render_pair(
        *_pair(
            tmp_path,
            description,
            sdist_description=(description + "\n").replace("\n", "\r\n"),
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("Name", "other"),
        ("Version", "0.0.1"),
        ("Description-Content-Type", "text/plain"),
    ],
)
def test_wrong_description_metadata_fails(tmp_path, field, value):
    with pytest.raises(rendering.ReleasePreflightError, match=f"{field} mismatch"):
        rendering.render_pair(*_pair(tmp_path, metadata_overrides={field: value}))


def test_only_expected_wheel_metadata_path_is_read(tmp_path):
    with pytest.raises(rendering.ReleasePreflightError, match="metadata is ambiguous"):
        rendering.render_pair(*_pair(tmp_path, wheel_path="../../other/METADATA"))


def test_sdist_metadata_symlink_is_rejected(tmp_path):
    with pytest.raises(rendering.ReleasePreflightError, match="not regular"):
        rendering.render_pair(*_pair(tmp_path, sdist_link=True))


def test_wheel_metadata_symlink_is_rejected(tmp_path):
    spec, wheel, sdist = _pair(tmp_path)
    with zipfile.ZipFile(wheel, "w") as archive:
        member = zipfile.ZipInfo(f"{spec.dist_info_root}/METADATA")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "../../outside")
    with pytest.raises(rendering.ReleasePreflightError, match="not regular"):
        rendering.render_pair(spec, wheel, sdist)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_duplicate_metadata_members_are_rejected(tmp_path, kind):
    spec, wheel, sdist = _pair(tmp_path)
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "a") as archive:
            name = f"{spec.dist_info_root}/METADATA"
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(name, archive.read(name))
    else:
        name = f"{spec.sdist_root}/PKG-INFO"
        with tarfile.open(sdist) as archive:
            raw = archive.extractfile(name).read()
        with tarfile.open(sdist, "w:gz") as archive:
            for _ in range(2):
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
    with pytest.raises(rendering.ReleasePreflightError, match="metadata is ambiguous"):
        rendering.render_pair(spec, wheel, sdist)


@pytest.mark.parametrize(
    "raw,error",
    [
        (b"not a metadata header\n\nREADME", "malformed"),
        (
            b"Name: invarlock\nVersion: 1.2.3\nDescription-Content-Type: text/markdown\n\n\xff",
            "not UTF-8",
        ),
        (
            b"Name: invarlock\nVersion: 1.2.3\nDescription-Content-Type: text/markdown\n"
            b'Content-Type: multipart/mixed; boundary="part"\n\n'
            b"--part\nContent-Type: text/plain\n\nREADME\n--part--\n",
            "not text",
        ),
    ],
)
def test_malformed_description_payload_fails(tmp_path, raw, error):
    spec, wheel, sdist = _pair(tmp_path)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{spec.dist_info_root}/METADATA", raw)
    with pytest.raises(rendering.ReleasePreflightError, match=error):
        rendering.render_pair(spec, wheel, sdist)


def test_empty_description_fails(tmp_path):
    with pytest.raises(rendering.ReleasePreflightError, match="description is empty"):
        rendering.render_pair(*_pair(tmp_path, "\n"))


def test_missing_renderer_fails(tmp_path, monkeypatch):
    def missing(name):
        raise rendering.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(rendering.importlib.metadata, "version", missing)
    with pytest.raises(
        rendering.ReleasePreflightError, match="requires readme-renderer"
    ):
        rendering.render_pair(*_pair(tmp_path))


def test_unpinned_renderer_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(rendering.importlib.metadata, "version", lambda name: "44.0")
    with pytest.raises(
        rendering.ReleasePreflightError, match="requires readme-renderer"
    ):
        rendering.render_pair(*_pair(tmp_path))


def test_missing_markdown_extra_fails(tmp_path, monkeypatch):
    from readme_renderer import markdown

    monkeypatch.setattr(markdown, "variants", {})
    with pytest.warns(UserWarning, match="Markdown renderers are not available"):
        with pytest.raises(
            rendering.ReleasePreflightError, match="Markdown rendering failed"
        ):
            rendering.render_pair(*_pair(tmp_path))


@pytest.mark.parametrize("result", [None, "", " \n"])
def test_empty_rendering_fails(tmp_path, monkeypatch, result):
    from readme_renderer import markdown

    monkeypatch.setattr(markdown, "render", lambda description: result)
    with pytest.raises(
        rendering.ReleasePreflightError, match="Markdown rendering failed"
    ):
        rendering.render_pair(*_pair(tmp_path))


def test_cli_renders_the_core_distribution_pair(tmp_path, capsys):
    for relative in rendering.PROJECTS:
        assert relative == "."
        name = "invarlock"
        project = tmp_path / relative
        project.mkdir(parents=True, exist_ok=True)
        (project / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1.2.3"\n'
        )
        _pair(
            tmp_path / "dist",
            None,
            name=name,
        )
    output = tmp_path / "previews"
    assert (
        rendering.main(["--repo-root", str(tmp_path), "--output-dir", str(output)]) == 0
    )
    assert len(list(output.glob("*.html"))) == 1
    assert capsys.readouterr().out.count("rendering passed") == 1
    next((tmp_path / "dist").glob("*.whl")).unlink()
    with pytest.raises(SystemExit) as exc:
        rendering.main(["--repo-root", str(tmp_path)])
    assert exc.value.code == 2
