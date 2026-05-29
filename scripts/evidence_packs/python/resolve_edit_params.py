from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .editing.specs import resolve_edit_spec
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.specs import resolve_edit_spec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: resolve_edit_params.py <model_output_dir> <edit_spec> [version_hint]",
            file=sys.stderr,
        )
        return 2

    resolved = resolve_edit_spec(
        model_output_dir=Path(argv[0]),
        edit_spec=argv[1],
        version_hint=argv[2] if len(argv) > 2 else "",
    )
    print(json.dumps(resolved.to_shell_payload()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
