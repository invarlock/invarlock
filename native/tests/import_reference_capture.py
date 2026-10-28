"""Retain source adapter inputs/outcomes while exercising its existing tests.

Use with pytest -p native.tests.import_reference_capture and an explicit
INVARLOCK_IMPORT_CORPUS destination. This does not alter the source outcomes.
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path


def pytest_configure(config):
    from invarlock.evaluation_records import adapters

    destination = os.environ.get("INVARLOCK_IMPORT_CORPUS")
    if not destination:
        return
    path = Path(destination)
    if path.exists():
        raise RuntimeError("Refusing to overwrite retained import corpus")
    original = adapters._parse_run_bytes
    seen = set()

    def capture(raw, **options):
        outcome = None
        try:
            value = original(raw, **options)
            outcome = {"ok": value}
            return value
        except Exception as exc:
            outcome = {"error": str(exc), "type": type(exc).__name__}
            raise
        finally:
            # Tests that replace production limits test that altered configuration.
            # They are not reference observations for the shipping native limits.
            if (
                isinstance(raw, bytes)
                and outcome is not None
                and adapters.MAX_RECORDS == 50_000
                and adapters.MAX_INPUT_BYTES == 128 * 1024 * 1024
            ):
                request = {
                    "raw_base64": base64.b64encode(raw).decode(),
                    "options": options,
                }
                try:
                    encoded = json.dumps(request, sort_keys=True, allow_nan=False)
                    limits = {
                        name: {
                            key: getattr(module, key)
                            for key in ("MAX_RECORDS", "MAX_INPUT_BYTES")
                            if hasattr(module, key)
                        }
                        for name, module in list(sys.modules.items())
                        if name.startswith("invarlock.evaluation_records.")
                    }
                    item = json.dumps(
                        {
                            "request": request,
                            "expected": outcome,
                            "reference_limits": limits,
                        },
                        allow_nan=False,
                    )
                except (TypeError, ValueError):
                    pass
                else:
                    key = hashlib.sha256(encoded.encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        with path.open("a") as stream:
                            stream.write(item + "\n")

    adapters._parse_run_bytes = capture
