from __future__ import annotations

import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_documentation_url_matches_published_latest_directory() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    assert config["site_url"] == "https://invarlock.github.io/invarlock/latest/"


@pytest.mark.parametrize("status", [200, 404])
def test_rendered_link_check_follows_nested_pages_and_rejects_broken_links(
    tmp_path: Path, status: int
) -> None:
    requests: list[str] = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self.send_response(status if self.path == "/destination" else 404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Destination</body></html>")

        def log_message(self, *_args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Destination) as destination:
        thread = Thread(target=destination.serve_forever, daemon=True)
        thread.start()
        try:
            site = tmp_path / "site"
            (site / "nested").mkdir(parents=True)
            (site / "index.html").write_text('<a href="nested/">Nested page</a>')
            (site / "nested/index.html").write_text(
                f'<a href="http://localhost:{destination.server_port}/destination">'
                "External destination</a>"
            )
            # These generated files are not part of the navigable page graph.
            (site / "404.html").write_text('<a href="/missing">Not found</a>')
            (site / "sitemap.xml").write_text(
                "<urlset><url><loc>https://unpublished.invalid/</loc></url></urlset>"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/checks/check_docs_links.py"),
                    "--site-dir",
                    str(site),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        finally:
            destination.shutdown()
            thread.join(timeout=5)

    assert "/destination" in requests, result.stdout + result.stderr
    assert (result.returncode == 0) == (status == 200), result.stdout + result.stderr
