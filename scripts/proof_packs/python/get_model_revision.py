from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: get_model_revision.py <revisions_json> <model_id>", file=sys.stderr
        )
        return 2

    path = Path(argv[0])
    model_id = argv[1]

    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0

    if not isinstance(data, dict):
        return 0

    revision = ""
    models = data.get("models")
    if isinstance(models, dict):
        entry = models.get(model_id)
        if isinstance(entry, dict):
            revision = str(entry.get("revision") or "")

    if revision:
        print(revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
