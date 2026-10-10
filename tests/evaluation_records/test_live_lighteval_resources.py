"""Authenticating opaque setup bytes does not deserialize benchmark objects."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location("live_resource_test", HERE / "harness.py")
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)


@pytest.fixture
def resource(tmp_path, monkeypatch):
    package = tmp_path / "lighteval"
    (package / "tasks/tasks").mkdir(parents=True)
    target = package / "tasks/tasks/tinyBenchmarks.pkl"
    archive = tmp_path / "asset.pkl"
    raw = b"opaque setup fixture; deliberately not a pickle"
    archive.write_bytes(raw)
    monkeypatch.setattr(
        LIVE.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    monkeypatch.setattr(
        LIVE,
        "LIGHTEVAL_REGISTRY_RESOURCE",
        {
            "url": "https://example.invalid/pinned",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        },
    )
    return archive, target, raw


def test_stage_then_verify_exact_opaque_bytes(resource):
    archive, target, raw = resource
    result = LIVE.lighteval_resource(archive)
    assert target.read_bytes() == raw
    assert LIVE.lighteval_resource() == result
    assert LIVE.lighteval_resource(archive) == result
    assert "does not deserialize" in result["scope"]


@pytest.mark.parametrize("mutation", ["hash", "size"])
def test_resource_refuses_changed_or_oversize_bytes(resource, mutation):
    archive, target, raw = resource
    archive.write_bytes(raw + b"x" if mutation == "size" else b"x" * len(raw))
    with pytest.raises(ValueError, match="independent pin"):
        LIVE.lighteval_resource(archive)
    assert not target.exists()


def test_resource_preserves_unexpected_existing_file(resource):
    archive, target, _ = resource
    target.write_bytes(b"original unrelated bytes")
    with pytest.raises(ValueError, match="refuse to replace"):
        LIVE.lighteval_resource(archive)
    assert target.read_bytes() == b"original unrelated bytes"


def test_missing_resource_or_sdk_has_actionable_error(resource, monkeypatch):
    with pytest.raises(ValueError, match="before capture"):
        LIVE.lighteval_resource()
    monkeypatch.setattr(LIVE.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="install the pinned"):
        LIVE.lighteval_resource()


@pytest.mark.parametrize("fail", [False, True])
def test_registry_bridge_forwards_real_results_and_restores_api(monkeypatch, fail):
    calls = []
    result = iter([object()])

    class Api:
        def list_models(self, *, search=None, **kwargs):
            calls.append((search, kwargs))
            return result

    metrics = ModuleType("lighteval.metrics")
    metrics.metrics_sample = SimpleNamespace(HfApi=Api)
    monkeypatch.setitem(sys.modules, "lighteval.metrics", metrics)
    try:
        with LIVE._lighteval_registry_api():
            client = metrics.metrics_sample.HfApi()
            assert client.list_models(model_name="example/model", limit=2) is result
            assert client.list_models(search="modern-query", limit=1) is result
            with pytest.raises(ValueError, match="cannot combine"):
                client.list_models(model_name="legacy", search="modern")
            if fail:
                raise RuntimeError("registry import failed")
    except RuntimeError as exc:
        assert fail and str(exc) == "registry import failed"
    assert metrics.metrics_sample.HfApi is Api
    assert calls == [("example/model", {"limit": 2}), ("modern-query", {"limit": 1})]
    assert len(list(result)) == 1  # The bridge neither consumes nor replaces results.
