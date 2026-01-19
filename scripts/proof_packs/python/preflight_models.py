from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: preflight_models.py <out_file> <model_id> [model_id...]",
            file=sys.stderr,
        )
        return 2

    out_file = Path(argv[0])
    model_ids = argv[1:]

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(
            "ERROR: huggingface_hub is required for preflight; install it before running with --net 1.",
            file=sys.stderr,
        )
        print(f"       Details: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, object] = {
        "generated_at": _utc_now(),
        "suite": str(os.environ.get("PACK_SUITE", "")),
        "model_list": list(model_ids),
        "models": {},
    }

    api = HfApi(token=False)

    errors: list[str] = []
    models_out: dict[str, dict[str, object]] = {}
    for model_id in model_ids:
        try:
            info = api.model_info(model_id, token=False)
        except Exception as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status in (401, 403):
                msg = "requires authentication (gated/private)"
            else:
                msg = str(err)
            print(
                f"ERROR: {model_id} is not publicly accessible ({msg})",
                file=sys.stderr,
            )
            errors.append(model_id)
            continue

        gated = bool(getattr(info, "gated", False))
        private = bool(getattr(info, "private", False))
        if gated or private:
            print(
                f"ERROR: {model_id} is gated/private; proof packs require ungated models.",
                file=sys.stderr,
            )
            errors.append(model_id)
            continue

        models_out[model_id] = {
            "revision": str(getattr(info, "sha", "") or ""),
            "resolved_at": _utc_now(),
            "gated": gated,
            "private": private,
        }

    payload["models"] = models_out

    if errors:
        return 2

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote model revisions to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
