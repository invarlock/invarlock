from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 5:
        print(
            "Usage: write_verification_summary.py <out_path> <count_clean> <count_error> <count_failed> <profile>",
            file=sys.stderr,
        )
        return 2

    out_path = Path(argv[0])
    count_clean = int(argv[1])
    count_error = int(argv[2])
    count_failed = int(argv[3])
    profile = argv[4]

    payload = {
        "clean_certs": count_clean,
        "error_injection_certs": count_error,
        "failed_certs": count_failed,
        "policy_profile": profile,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
