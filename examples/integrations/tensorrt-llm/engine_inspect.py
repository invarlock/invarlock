#!/usr/bin/env python3
"""Inspect two TensorRT-LLM engines inside their authenticated runtime image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from invarlock_addins.tensorrt_llm.execution import (
    official_tensorrt_llm_runner_path,
)
from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import TensorRTLLMRuntimeBindings

from invarlock.core.runtime_provider import artifact_identity_sha256
from invarlock.runtime_security_helpers import (
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    strict_container_boundary_present,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    arguments = parser.parse_args()
    if not strict_container_boundary_present():
        raise SystemExit("inspection requires the authenticated container boundary")
    image = os.environ.get(RUNTIME_IMAGE_ENV, "")
    digest = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    if image != digest and not image.endswith("@" + digest):
        raise SystemExit("runtime image does not embed its authenticated digest")
    root = arguments.resource_root.resolve(strict=True)
    provider = TensorRTLLMProvider()
    result = {}
    for role in ("baseline", "subject"):
        spec = provider.inspect_runtime_spec(
            TensorRTLLMRuntimeBindings(
                engine_bundle_path=root / f"{role}-engine",
                tokenizer_contract_path=root / "tokenizer-contract.json",
                runner_executable_path=official_tensorrt_llm_runner_path(),
            ),
            seed=0,
            context_length=arguments.context_length,
            batch_size=1,
            max_output_tokens=arguments.max_output_tokens,
            timeout_seconds=arguments.timeout_seconds,
        )
        result[role] = {
            "artifact_identity_sha256": "sha256:"
            + artifact_identity_sha256(provider.identify_artifact(spec)),
            "model_id": spec.model_id,
            "settings": dict(spec.settings),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
