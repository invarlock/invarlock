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
DIAGNOSTICS_ROOT = REPO_ROOT / "src/invarlock/diagnostics"

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


def test_core_distribution_registers_all_first_party_providers() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    providers = metadata["project"]["entry-points"]["invarlock.runtime_providers"]
    scripts = metadata["project"]["scripts"]

    assert providers == {
        "hf_transformers": "invarlock.runtime_providers.hf_transformers:HFTransformersProvider",
        "llama_cpp": "invarlock.runtime_providers.llama_cpp:LlamaCppProvider",
        "tensorrt_llm": "invarlock.runtime_providers.tensorrt_llm:TensorRTLLMProvider",
        "hf_vision_text": "invarlock.runtime_providers.hf_vision_text:HFVisionTextProvider",
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


def test_native_provider_implementations_live_in_the_core_distribution() -> None:
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
        core_provider_root.joinpath(name).is_file() for name in forbidden_core_files
    )
    assert not (REPO_ROOT / "addins").exists()


def test_core_enumeration_does_not_import_execution_backends() -> None:
    payload = _run_enumeration_probe()

    assert payload["providers"] == [
        "hf_transformers",
        "llama_cpp",
        "tensorrt_llm",
        "hf_vision_text",
    ]
    assert payload["loaded"] == []
    assert payload["identity_readers"] == [
        "read_gguf_artifact_identity",
        "read_tensorrt_llm_artifact_identity",
    ]


def test_custom_observability_package_is_not_part_of_core() -> None:
    observability_root = REPO_ROOT / "src/invarlock/observability"
    assert not list(observability_root.glob("*.py"))
    assert not (observability_root / "py.typed").exists()


def test_public_extras_do_not_resolve_an_unqualified_model_runtime() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]
    assert set(extras) == {"diagnostics", "vision-text", "judge"}
    assert not any(
        "+invarlock." in item
        for requirements in extras.values()
        for item in requirements
    )
    assert not any(
        item.startswith("invarlock-")
        for requirements in extras.values()
        for item in requirements
    )

    groups = metadata["dependency-groups"]
    assert "accelerate==1.14.0+invarlock.1" in groups["hf"]
    assert "pillow>=11.3,<13" in groups["runtime-test"]
    for name in ("runtime-test", "example-peft", "example-torchao"):
        assert {"include-group": "hf"} in groups[name]
    assert metadata["tool"]["uv"]["find-links"] == ["runtime/wheels"]
    assert (REPO_ROOT / "runtime/wheels/README.md").is_file()


def test_core_public_surface_and_provider_capabilities_block_optional_imports() -> None:
    probe = """
import importlib.abc
import sys
blocked = {
    "torch", "transformers", "accelerate", "numpy", "PIL", "inspect_ai",
    "openai", "anthropic", "google", "httpx", "tensorrt", "tensorrt_llm",
}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise RuntimeError("Optional dependency imported: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from invarlock import engine
from invarlock.cli.app import app
from invarlock.core.registry import CoreRegistry
import invarlock.diagnostics
import invarlock.judge_measurements
registry = CoreRegistry()
for name in registry.list_runtime_providers():
    provider = registry.get_runtime_provider(name)
    assert provider.capabilities().provider_name == name
assert not blocked.intersection(sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
