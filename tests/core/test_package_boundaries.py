"""Package-boundary regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from invarlock.core.builtin_plugin_catalog import builtin_plugin_specs

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
OBSERVABILITY_ROOT = REPO_ROOT / "src/invarlock/observability"
GGUF_ADDIN = REPO_ROOT / "addins/gguf"
MULTIMODAL_ADDIN = REPO_ROOT / "addins/multimodal"
TENSORRT_LLM_ADDIN = REPO_ROOT / "addins/tensorrt_llm"

NATIVE_EXECUTION_MODULES = (
    "invarlock.runtime_providers.llama_cpp",
    "invarlock.runtime_providers.llama_cpp_session",
    "invarlock.runtime_providers.tensorrt_llm",
    "invarlock.runtime_providers.tensorrt_llm_session",
    "invarlock.runtime_providers._tensorrt_llm_execution",
    "invarlock.runtime_providers._tensorrt_llm_inspection",
    "invarlock.runtime_providers.tensorrt_llm_runner",
)
GUARD_EXECUTION_MODULES = (
    "invarlock.guards.invariants",
    "invarlock.guards.rmt",
    "invarlock.guards.spectral",
    "invarlock.guards.variance",
)

_ENUMERATION_PROBE = r"""
import json
import sys

from invarlock.core.registry import CoreRegistry
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)

registry = CoreRegistry()
providers = registry.list_runtime_providers()
blocked = tuple(json.loads(sys.argv[1]))
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)
)
print(
    json.dumps(
        {
            "providers": providers,
            "loaded": loaded,
            "identity_readers": [
                read_gguf_artifact_identity.__name__,
                read_tensorrt_llm_artifact_identity.__name__,
            ],
        }
    )
)
"""


def _run_enumeration_probe() -> dict[str, object]:
    env = os.environ.copy()
    env["INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _ENUMERATION_PROBE,
            json.dumps(NATIVE_EXECUTION_MODULES + GUARD_EXECUTION_MODULES),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_core_distribution_registers_only_canonical_hf_provider() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    providers = metadata["project"]["entry-points"]["invarlock.runtime_providers"]
    scripts = metadata["project"]["scripts"]

    assert providers == {
        "hf_transformers": (
            "invarlock.runtime_providers.hf_transformers:HFTransformersProvider"
        )
    }
    assert "invarlock-tensorrt-llm-runner" not in scripts


def test_builtin_provider_catalog_rejects_unknown_catalog_types() -> None:
    with pytest.raises(ValueError, match="Unknown plugin catalog type"):
        builtin_plugin_specs("scorers")


def test_core_distribution_declares_its_supported_posix_platforms() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    classifiers = set(metadata["project"]["classifiers"])

    assert "Operating System :: OS Independent" not in classifiers
    assert "Operating System :: MacOS :: MacOS X" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers


def test_native_provider_implementations_live_in_optional_addin_distributions() -> None:
    core_provider_root = REPO_ROOT / "src/invarlock/runtime_providers"
    forbidden_core_files = (
        "llama_cpp.py",
        "llama_cpp_session.py",
        "tensorrt_llm.py",
        "tensorrt_llm_session.py",
        "_tensorrt_llm_execution.py",
        "_tensorrt_llm_inspection.py",
        "tensorrt_llm_canary.py",
        "tensorrt_llm_runner.py",
    )
    assert all(
        not core_provider_root.joinpath(name).exists() for name in forbidden_core_files
    )

    gguf = tomllib.loads((GGUF_ADDIN / "pyproject.toml").read_text(encoding="utf-8"))
    multimodal = tomllib.loads(
        (MULTIMODAL_ADDIN / "pyproject.toml").read_text(encoding="utf-8")
    )
    tensorrt = tomllib.loads(
        (TENSORRT_LLM_ADDIN / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert gguf["project"]["entry-points"]["invarlock.runtime_providers"] == {
        "llama_cpp": "invarlock_addins.gguf.provider:LlamaCppProvider"
    }
    assert multimodal["project"]["entry-points"]["invarlock.runtime_providers"] == {
        "hf_vision_text": ("invarlock_addins.multimodal.provider:HFVisionTextProvider")
    }
    assert tensorrt["project"]["entry-points"]["invarlock.runtime_providers"] == {
        "tensorrt_llm": ("invarlock_addins.tensorrt_llm.provider:TensorRTLLMProvider")
    }
    assert (GGUF_ADDIN / "runtime/Dockerfile").is_file()
    assert (TENSORRT_LLM_ADDIN / "runtime/Dockerfile").is_file()


def test_core_enumeration_does_not_import_addin_or_guard_execution_modules() -> None:
    payload = _run_enumeration_probe()

    assert payload["providers"] == ["hf_transformers"]
    assert payload["loaded"] == []
    assert payload["identity_readers"] == [
        "read_gguf_artifact_identity",
        "read_tensorrt_llm_artifact_identity",
    ]


def test_custom_observability_package_is_not_part_of_core() -> None:
    assert not list(OBSERVABILITY_ROOT.glob("*.py"))
    assert not (OBSERVABILITY_ROOT / "py.typed").exists()
