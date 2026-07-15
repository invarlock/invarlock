from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from invarlock.core.runtime_provider import (
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)


def load_script(name: str) -> ModuleType:
    path = Path.cwd() / "scripts" / "release" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_script("tensorrt_llm_runtime_fixture")


def identity(*, tree: str = "1" * 64) -> TensorRTLLMArtifactIdentity:
    return TensorRTLLMArtifactIdentity(
        bundle_name=f"tensorrt-llm-sha256-{tree}",
        engine_bundle_tree_sha256=tree,
        file_inventory_sha256="2" * 64,
        builder_config_sha256="3" * 64,
        tokenizer_metadata_sha256="4" * 64,
        engine_metadata_sha256="5" * 64,
        target_compute_capability="9.0",
    )


def canary_payload(
    *,
    artifact: str | None = None,
    engine_tree: str = "1" * 64,
    tokenizer: str = "4" * 64,
    output: str = "7" * 64,
) -> dict[str, object]:
    return {
        "artifact_identity_sha256": artifact or artifact_identity_sha256(identity()),
        "engine_bundle_tree_sha256": engine_tree,
        "format_version": fixture.CANARY_FORMAT,
        "ok": True,
        "output_sha256": output,
        "scoring_observation_sha256": "9" * 64,
        "tokenizer_metadata_sha256": tokenizer,
    }


def valid_manifest(identity_value: TensorRTLLMArtifactIdentity) -> dict[str, object]:
    identity_payload = fixture.asdict(identity_value)
    return {
        "backend_version": fixture.BACKEND_VERSION,
        "build_recipe": dict(fixture.BUILD_RECIPE),
        "candidate_image_digest": "sha256:" + "a" * 64,
        "engine_builds": {
            "primary": identity_payload,
            "secondary": identity_payload,
        },
        "engine_byte_reproduction": "matched",
        "expected_output_sha256": "7" * 64,
        "format_version": fixture.MANIFEST_FORMAT,
        "model": {
            "inventory_sha256": "6" * 64,
            "repository": fixture.MODEL_REPOSITORY,
            "revision": fixture.MODEL_REVISION,
        },
        "selected_engine_identity": identity_payload,
        "tokenizer_sha256": identity_value.tokenizer_metadata_sha256,
        "worker": {"sha256": "8" * 64},
    }


def qualification_summary() -> dict[str, object]:
    return {
        "candidate_image_digest": "sha256:" + "a" * 64,
        "engine_bundle_tree_sha256": "1" * 64,
        "format_version": fixture.QUALIFICATION_FORMAT,
        "gpu_count": 2,
        "ok": True,
        "output_sha256": "7" * 64,
        "tokenizer_sha256": "4" * 64,
    }


def inspection(*, digest: str = "sha256:" + "a" * 64) -> bytes:
    return json.dumps(
        [
            {
                "Config": {
                    "Labels": {
                        "dev.invarlock.runtime-provider": "tensorrt_llm",
                        "dev.invarlock.tensorrt-llm.base-digest": fixture.BASE_DIGEST,
                        "dev.invarlock.tensorrt-llm.version": fixture.BACKEND_VERSION,
                    }
                },
                "Id": digest,
            }
        ]
    ).encode()
