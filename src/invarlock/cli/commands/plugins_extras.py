"""
Optional-extra checks for the plugins CLI command.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import io
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import NotRequired, TypedDict

from invarlock.core.plugins_inventory import bitsandbytes_runtime_available

_EXTRA_CHECK_ERRORS = (AttributeError, ImportError, OSError, RuntimeError, ValueError)


class _PluginExtraInfo(TypedDict):
    packages: list[str]
    extra: str
    minimum_versions: NotRequired[dict[str, str]]


def check_plugin_extras(plugin_name: str, plugin_type: str) -> str:
    """Check if plugin requires missing optional extras."""
    extras_map: dict[str, _PluginExtraInfo] = {
        "quant_rtn": {"packages": [], "extra": ""},
        "invariants": {"packages": [], "extra": ""},
        "spectral": {"packages": [], "extra": ""},
        "variance": {"packages": [], "extra": ""},
        "rmt": {"packages": [], "extra": ""},
        "demo_hello_guard": {"packages": [], "extra": ""},
        "hf_causal": {
            "packages": ["transformers"],
            "extra": "invarlock[adapters]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_mlm": {
            "packages": ["transformers"],
            "extra": "invarlock[adapters]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_seq2seq": {
            "packages": ["transformers"],
            "extra": "invarlock[adapters]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_multimodal": {
            "packages": ["transformers", "torchvision", "PIL"],
            "extra": "invarlock[multimodal]",
            "minimum_versions": {
                "transformers": "5.12.0",
                "torchvision": "0.26.0",
            },
        },
        "hf_auto": {
            "packages": ["transformers"],
            "extra": "invarlock[adapters]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_gptq": {
            "packages": ["gptqmodel", "transformers"],
            "extra": "invarlock[gptq]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_awq": {
            "packages": ["gptqmodel", "transformers"],
            "extra": "invarlock[awq]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_bnb": {
            "packages": ["bitsandbytes", "transformers"],
            "extra": "invarlock[gpu]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_torchao": {
            "packages": ["torchao", "transformers"],
            "extra": "invarlock[torchao]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_hqq": {
            "packages": ["hqq", "transformers"],
            "extra": "invarlock[hqq]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_quanto": {
            "packages": ["optimum.quanto", "transformers"],
            "extra": "invarlock[quanto]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
        "hf_ct": {
            "packages": ["compressed_tensors", "transformers"],
            "extra": "invarlock[compressed-tensors]",
            "minimum_versions": {"transformers": "5.12.0"},
        },
    }

    plugin_info = extras_map.get(plugin_name)
    if not plugin_info or not plugin_info["packages"]:
        return ""

    missing_packages: list[str] = []
    minimum_versions = plugin_info.get("minimum_versions", {})
    for pkg in plugin_info["packages"]:
        try:
            _plugin_package_importable(str(pkg))
            minimum_version = minimum_versions.get(pkg)
            if minimum_version and not _package_version_at_least(
                pkg, str(minimum_version)
            ):
                raise ImportError(f"{pkg}>={minimum_version} not available")
        except _EXTRA_CHECK_ERRORS:
            missing_packages.append(pkg)

    if not missing_packages:
        if plugin_info["extra"]:
            return f"✓ {plugin_info['extra']}"
        return "✓ Available"
    if plugin_info["extra"]:
        return f"⚠️ missing {plugin_info['extra']}"
    return f"⚠️ missing {', '.join(missing_packages)}"


def _plugin_package_importable(package_name: str) -> bool:
    if package_name == "bitsandbytes":
        if not bitsandbytes_runtime_available():
            raise ImportError("bitsandbytes not importable")
        return True
    if package_name == "gptqmodel":
        from invarlock.plugins import _patch_gptqmodel_transformers_hub_compat

        _patch_gptqmodel_transformers_hub_compat()
    with (
        warnings.catch_warnings(),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        __import__(package_name)
    return True


def _package_version_at_least(package_name: str, minimum: str) -> bool:
    try:
        installed = importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return _version_key(installed) >= _version_key(minimum)


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(match.group(0)) for match in re.finditer(r"\d+", value)]
    padded = (parts + [0, 0, 0])[:3]
    return (padded[0], padded[1], padded[2])
