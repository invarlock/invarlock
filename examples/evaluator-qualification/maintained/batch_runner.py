"""Run a strict batch profile against a frozen copy of its declared inputs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from maintained.batch_native import execute
from maintained.batch_semantics import PROVIDERS, project, validate_domain
from maintained.profile_binding import require_current_profile
from runner_support import (
    arguments,
    finish_deterministic,
    load_inputs,
    require_profile_package,
)


def main() -> None:
    args = arguments()
    with tempfile.TemporaryDirectory(prefix="invarlock-batch-inputs-") as temporary:
        frozen = argparse.Namespace(**vars(args))
        for name in ("cases", "profile", "schedule", "dependency_lock"):
            path = Path(temporary) / name
            path.write_bytes(getattr(args, name).read_bytes())
            setattr(frozen, name, path)
        profile, _, cases = load_inputs(frozen)
        profiles = {f"{provider}-strict-batch-v1": provider for provider in PROVIDERS}
        provider = profiles.get(profile["profile_id"])
        if provider is None:
            raise ValueError("batch runner requires its separate versioned profile")
        root = Path(__file__).resolve().parent
        definition = next(
            item
            for item in json.loads((root / "batch-profiles.json").read_bytes())[
                "profiles"
            ]
            if item["profile_id"] == profile["profile_id"]
        )
        require_current_profile(profile, definition, root.parent)
        require_profile_package(profile)
        validate_domain(provider, cases)
        native, environment = execute(
            provider,
            cases,
            version=profile["upstream"]["package"]["version"],
            dependency_lock=frozen.dependency_lock,
        )
        scores, details = project(provider, cases, native)
        finish_deterministic(
            args=frozen,
            entrypoint=f"maintained strict {provider} batch",
            scores=scores,
            details=details,
            environment=environment,
        )


if __name__ == "__main__":
    main()
