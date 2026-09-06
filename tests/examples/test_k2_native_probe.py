"""Exercise native-probe rejection boundaries; actual imports run in the image."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples.qualification import k2_native_probe as probe


def _environment(tmp_path, monkeypatch, *, mode="good"):
    root = tmp_path / "facts"
    root.mkdir()
    package = tmp_path / "sglang"
    package.mkdir()
    source = package / "model.py"
    source.write_bytes(b"reviewed native source fixture")
    inputs = {
        "derived_distribution_version": "fixture",
        "reviewed_source_files": {
            "python/sglang/model.py": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python/pyproject.toml": "0" * 64,
        },
    }
    (root / "build-inputs.json").write_text(json.dumps(inputs))
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(probe.platform, "platform", lambda: "Linux fixture")
    distributions = [
        SimpleNamespace(
            metadata={"Name": "sglang"},
            version="wrong" if mode == "version" else "fixture",
        )
    ]
    if mode == "package":
        distributions.append(
            SimpleNamespace(metadata={"Name": "diskcache"}, version="fixture")
        )
    monkeypatch.setattr(
        probe.importlib.metadata, "distributions", lambda: distributions
    )
    calls = []

    def import_module(name):
        calls.append(name)
        if "outlines" in name:
            if mode == "optional_pass":
                return SimpleNamespace()
            raise RuntimeError(
                "different import failure"
                if mode == "optional_error"
                else "unavailable in the restricted K2 runtime"
            )
        return SimpleNamespace(__file__=str(package / "__init__.py"))

    monkeypatch.setattr(probe.importlib, "import_module", import_module)
    restored = []
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_context = lambda: SimpleNamespace(
        override_server_args=lambda **kw: SimpleNamespace(
            install=lambda: None, restore=lambda: restored.append(True)
        )
    )
    grammar = ModuleType("sglang.srt.constrained.base_grammar_backend")

    def dispatch(*args):
        if mode == "dispatch_pass":
            return None
        raise RuntimeError(
            "different dispatch failure"
            if mode == "dispatch_error"
            else "unavailable in the restricted K2 runtime"
        )

    grammar.create_grammar_backend = dispatch
    monkeypatch.setitem(sys.modules, runtime_context.__name__, runtime_context)
    monkeypatch.setitem(sys.modules, grammar.__name__, grammar)

    def run(args, **kwargs):
        if mode == "help_timeout":
            raise subprocess.TimeoutExpired(args, 180)
        return SimpleNamespace(
            stdout="missing"
            if mode == "help_missing"
            else "--model-path --grammar-backend"
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    if mode == "source":
        source.write_bytes(b"substituted native source")
    return root, calls, restored


def test_cpu_probe_records_its_limited_scope(tmp_path, monkeypatch):
    root, calls, restored = _environment(tmp_path, monkeypatch)
    report = probe.inspect_native(root)
    assert report["status"] == "cpu_imports_passed_not_gpu_qualified"
    assert report["gpu_execution"] is False
    assert "sglang.srt.models.xllm" in calls
    assert report["excluded_dispatch_rejected"]
    assert restored == [True]


@pytest.mark.parametrize(
    "mode,exception",
    [
        ("version", ValueError),
        ("package", ValueError),
        ("source", ValueError),
        ("optional_pass", ValueError),
        ("optional_error", RuntimeError),
        ("dispatch_pass", ValueError),
        ("dispatch_error", RuntimeError),
        ("help_timeout", subprocess.TimeoutExpired),
        ("help_missing", ValueError),
    ],
)
def test_probe_does_not_convert_bad_boundaries_to_success(
    tmp_path, monkeypatch, mode, exception
):
    root, _, _ = _environment(tmp_path, monkeypatch, mode=mode)
    with pytest.raises(exception):
        probe.inspect_native(root)


def test_probe_requires_the_declared_platform(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="platform|Linux"):
        probe.inspect_native()


@pytest.mark.parametrize(
    "names",
    [
        ("Example-Package", "example_package"),
        ("example.package", "example--package"),
        ("packaging", "packaging"),
    ],
)
def test_duplicate_real_installed_metadata_cannot_hide_a_version(tmp_path, names):
    from importlib.metadata import distributions

    roots = []
    for index, (name, version) in enumerate(zip(names, ("0.0.0", "26.0"), strict=True)):
        root = tmp_path / str(index)
        metadata = root / f"example_{index}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        roots.append(str(root))
    observed = list(distributions(path=roots))
    assert len(observed) == 2
    with pytest.raises(ValueError, match="duplicate installed distribution"):
        probe.package_inventory(observed)


def test_package_inventory_normalizes_unique_distribution_names():
    assert probe.package_inventory(
        [
            SimpleNamespace(metadata={"Name": "Example._Package"}, version="1.0"),
            SimpleNamespace(metadata={"Name": "Other"}, version="2.0"),
        ]
    ) == {"example-package": "1.0", "other": "2.0"}


def test_native_probe_rejects_duplicates_before_importing(tmp_path, monkeypatch):
    root, calls, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "sglang"}, version="wrong"),
            SimpleNamespace(metadata={"Name": "SGLang"}, version="fixture"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate installed distribution"):
        probe.inspect_native(root)
    assert calls == []
