"""Execute a current scalar profile with complete profile and native source binding."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from maintained.batch_semantics import validate_cases
from maintained.profile_binding import require_current_profile
from maintained.scalar_native import build_scorer
from maintained.scalar_semantics import CONFIGURATIONS, validate_pair, validate_result
from runner_support import (
    arguments,
    finish_deterministic,
    load_inputs,
    require_profile_package,
)


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parent
    definitions = json.loads((root / "scalar-profiles.json").read_bytes())["profiles"]
    with tempfile.TemporaryDirectory(prefix="invarlock-scalar-inputs-") as temporary:
        frozen = argparse.Namespace(**vars(args))
        for name in ("cases", "profile", "schedule", "dependency_lock"):
            path = Path(temporary) / name
            path.write_bytes(getattr(args, name).read_bytes())
            setattr(frozen, name, path)
        profile, _, cases = load_inputs(frozen)
        definition = next(
            (
                item
                for item in definitions
                if item["profile_id"] == profile["profile_id"]
            ),
            None,
        )
        if definition is None:
            raise ValueError("scalar execution requires a separate current profile")
        provider = definition["historical_profile"]
        if definition["scorer_configuration"] != CONFIGURATIONS[provider]:
            raise ValueError("current scalar scorer configuration changed")
        require_current_profile(profile, definition, root.parent)
        require_profile_package(profile)
        validate_cases(cases)
        for case in cases:
            validate_pair(provider, case)
        score, sources = build_scorer(provider)
        if sources != definition["source_bindings"]:
            raise ValueError("native source bindings do not match the current profile")
        scores, details = [], []
        for case in cases:
            native = score(case)
            value = validate_result(provider, case, native)
            scores.append(value)
            details.append({"native": native, "source_bindings": sources})
        finish_deterministic(
            args=frozen,
            entrypoint=f"current literal {provider} scalar",
            scores=scores,
            details=details,
        )


if __name__ == "__main__":
    main()
