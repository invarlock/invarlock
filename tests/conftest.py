from __future__ import annotations

import os
import sys
from importlib import import_module

import pytest

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)
_PATCH_TARGET_MODULES = (
    "invarlock.cli.bench",
    "invarlock.core.auto_tuning",
    "invarlock.core.bootstrap",
    "invarlock.core.config_loader",
    "invarlock.core.determinism_policy",
    "invarlock.core.metric_provider_resolution",
    "invarlock.core.registry",
    "invarlock.core.orchestration.execute",
    "invarlock.core.runner",
    "invarlock.runtime_verify",
    "invarlock.eval.bench_runner",
    "invarlock.eval.data",
    "invarlock.eval.metrics_activation",
    "invarlock.eval.metrics_support",
    "invarlock.eval.primary_metric",
    "invarlock.model_profile",
    "invarlock.observability.core",
    "invarlock.observability.health",
    "invarlock.plugins.bitsandbytes",
    "invarlock.evidence_pack",
    "invarlock.reporting.report_summary",
    "invarlock.reporting.report_make",
    "invarlock.reporting.report_builder_support",
)


def _reattach_parent_package_attrs(module_name: str) -> None:
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        child_name = parts[idx]
        child_module_name = ".".join(parts[: idx + 1])
        parent_module = sys.modules.get(parent_name)
        child_module = sys.modules.get(child_module_name)
        if parent_module is None or child_module is None:
            continue
        if not hasattr(parent_module, child_name):
            setattr(parent_module, child_name, child_module)


@pytest.fixture(autouse=True)
def _restore_invarlock_env():
    # Snapshot environment variables that some tests may mutate without cleanup
    keys = [
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_RUNTIME_IMAGE",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _materialize_patch_targets(request: pytest.FixtureRequest):
    nodeid = request.node.nodeid
    if "import_safety" in nodeid:
        yield
        return

    for module_name in _PATCH_TARGET_MODULES:
        try:
            import_module(module_name)
            _reattach_parent_package_attrs(module_name)
        except Exception:
            continue
    yield


@pytest.fixture
def allow_host_execution_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _VALID_TEST_IMAGE_DIGEST)
    yield
