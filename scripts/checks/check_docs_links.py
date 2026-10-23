"""Check rendered documentation through HTTP, including external destinations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    site = args.site_dir.resolve()
    if not (site / "index.html").is_file():
        parser.error(f"Missing rendered entry page: {site / 'index.html'}")

    handler = partial(_Handler, directory=str(site))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "linkcheck",
                    "--check-extern",
                    "--no-status",
                    f"http://127.0.0.1:{server.server_port}/",
                ],
                check=False,
            )
        finally:
            server.shutdown()
            thread.join()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
