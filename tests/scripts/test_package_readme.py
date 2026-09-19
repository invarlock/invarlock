"""Package descriptions keep repository context without mutable release links."""

from html.parser import HTMLParser
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from scripts.release import package_readme as readme


def project(tmp_path, text, version="1.2.3"):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion="{version}"\n')
    (tmp_path / "README.md").write_text(text)
    return tmp_path


def test_resources_and_literal_examples(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.md").write_text("a")
    text = '[Doc](docs/a.md?plain=1#part) [Dir](docs) ![Image](docs/a.md)\n<img src="docs/a.md"> [Anchor](#part) [Web](https://example.com)\n```sh\n[Literal](missing)\n```\n<picture><source srcset="docs/a.md"><img src="docs/a.md"></picture>'
    result = readme.render(tmp_path, project(tmp_path, text))
    assert "/blob/v1.2.3/docs/a.md?plain=1#part" in result
    assert "/tree/v1.2.3/docs" in result
    assert (
        "https://raw.githubusercontent.com/invarlock/invarlock/v1.2.3/docs/a.md"
        in result
    )
    assert "[Literal](missing)" in result
    assert "[Anchor](#part)" in result
    assert "<source" not in result and "<picture" not in result


@pytest.mark.parametrize(
    "url",
    [
        "missing",
        "../outside",
        "javascript:alert",
        "//example.com/file",
        "https://github.com/invarlock/invarlock/blob/main/README.md",
        "https://raw.githubusercontent.com/invarlock/invarlock/staging/next/a",
    ],
)
def test_invalid_resource(tmp_path, url):
    with pytest.raises(ValueError):
        readme.render(tmp_path, project(tmp_path, f"[Bad]({url})"))


def test_repository_context(tmp_path):
    (tmp_path / "LICENSE").write_text("license")
    result = readme.render(tmp_path, project(tmp_path, "[License](LICENSE)"))
    assert "/blob/v1.2.3/LICENSE" in result


def test_invalid_version(tmp_path):
    with pytest.raises(ValueError, match="version"):
        readme.render(tmp_path, project(tmp_path, "", "../main"))


def test_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(readme, "PROJECTS", (".",))
    project(tmp_path, "description")
    with pytest.raises(ValueError, match="stale"):
        readme.synchronize(tmp_path, write=False)
    readme.synchronize(tmp_path, write=True)
    readme.synchronize(tmp_path, write=False)
    (tmp_path / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        readme.synchronize(tmp_path, write=False)


def test_repository_descriptions_current():
    readme.synchronize(Path(__file__).resolve().parents[2], write=False)


def test_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(readme, "ROOT", project(tmp_path, "description"))
    monkeypatch.setattr(readme, "PROJECTS", (".",))
    monkeypatch.setattr("sys.argv", ["package_readme", "--write"])
    readme.main()
    monkeypatch.setattr("sys.argv", ["package_readme", "--check"])
    readme.main()


def test_picture_cleanup_preserves_literal_examples(tmp_path):
    source = '<p>\n  <picture>\n    <source srcset="x">\n  </picture>\n</p>\n```html\n<picture><source srcset="x"></picture>\n  \n```\n'
    result = readme.render(tmp_path, project(tmp_path, source))
    assert result.startswith("<p>\n\n</p>")
    assert '```html\n<picture><source srcset="x"></picture>\n  \n```' in result


def test_package_logo_renders_as_image():
    root = Path(__file__).resolve().parents[2]
    rendered = MarkdownIt("commonmark").render(readme.render(root, root))

    class Images(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = []

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.images.append(dict(attrs))

    images = Images()
    images.feed(rendered)
    logos = [item for item in images.images if item.get("alt") == "InvarLock"]
    assert len(logos) == 1
    assert logos[0]["src"].endswith("/docs/assets/invarlock-logo.svg")
    assert "&lt;img" not in rendered


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/invarlock/invarlock/tree/staging/next/docs",
        "https://GITHUB.COM/invarlock/invarlock/blob/main/README.md",
        "https://raw.githubusercontent.com/invarlock/invarlock/main/README.md",
    ],
)
def test_mutable_repository_urls_are_rejected(tmp_path, url):
    with pytest.raises(ValueError, match="unversioned repository resource"):
        readme.render(tmp_path, project(tmp_path, f"[Link]({url})"))


@pytest.mark.parametrize(
    "url",
    [
        "https://githubXcom/invarlock/invarlock/blob/main/README.md",
        "https://rawXgithubusercontent.com/invarlock/invarlock/main/README.md",
        "https://example.com/github.com/invarlock/invarlock/blob/main/README.md",
        "https://github.com/invarlock/invarlock/blob/v1.2.3/README.md",
        "https://raw.githubusercontent.com/invarlock/invarlock/v1.2.3/README.md",
        "https://github.com/another/repo/blob/main/README.md",
    ],
)
def test_other_hosts_paths_and_versioned_resources_are_preserved(tmp_path, url):
    source = f"[Link]({url})"
    assert readme.render(tmp_path, project(tmp_path, source)) == source
