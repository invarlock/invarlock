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
    monkeypatch.setattr(probe, "host_compiler_probe", lambda: {"status": "fixture"})
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


def host_report():
    return {
        "status": "fixed_cpu_host_compile_and_call_passed",
        "source_sha256": hashlib.sha256(probe.HOST_SOURCE).hexdigest(),
        "compiled_library_sha256": "a" * 64,
        "triton_version": "fixture",
        "result": 42,
        "gpu_execution": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"result": 0},
        {"gpu_execution": True},
        {"source_sha256": "0" * 64},
        {"compiled_library_sha256": ""},
        {"triton_version": "other"},
        {"status": "skipped"},
    ],
)
def test_host_compiler_report_must_bind_fixture_and_runtime(change):
    value = host_report()
    value.update(change)
    with pytest.raises(ValueError):
        probe.validate_host_compiler(value, "fixture")


def test_host_compiler_missing_report_rejected():
    with pytest.raises(ValueError):
        probe.validate_host_compiler(None, "fixture")


@pytest.mark.parametrize("mode", ["good", "failure", "large", "timeout"])
def test_host_compiler_child_result_and_timeout(tmp_path, monkeypatch, mode):
    child = tmp_path / "child.py"
    if mode == "good":
        code = "print(" + repr(json.dumps(host_report())) + ")"
    elif mode == "failure":
        code = "raise SystemExit(3)"
    elif mode == "large":
        code = 'print("x" * 70000)'
    else:
        code = "import time; time.sleep(20)"
    child.write_text(code)
    monkeypatch.setattr(probe, "__file__", str(child))
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(probe, "HOST_COMPILER_TIMEOUT", 0.1 if mode == "timeout" else 5)
    if mode == "good":
        assert probe.host_compiler_probe() == host_report()
    else:
        with pytest.raises(
            subprocess.TimeoutExpired if mode == "timeout" else ValueError
        ):
            probe.host_compiler_probe()


def test_host_compiler_failed_child_retains_bounded_diagnostic(tmp_path, monkeypatch):
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\n"
        "sys.stderr.buffer.write(b'failed to map segment: \\xff' + b'x' * 5000)\n"
        "raise SystemExit(3)\n"
    )
    monkeypatch.setattr(probe, "__file__", str(child))
    with pytest.raises(ValueError) as caught:
        probe.host_compiler_probe()
    message = str(caught.value)
    assert "(exit 3): failed to map segment: \ufffd" in message
    assert len(message.split("(exit 3): ", 1)[1]) == 4096


@pytest.mark.parametrize("mode", ["good", "wrong_result", "noexec", "escape"])
def test_fixed_host_compile_uses_real_interface_and_checks_load(
    tmp_path, monkeypatch, mode
):
    import ctypes
    import resource

    limits = []
    calls = []
    monkeypatch.setattr(
        resource, "setrlimit", lambda kind, value: limits.append((kind, value))
    )
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "fixture")
    module = ModuleType("triton.runtime.build")

    def build(name, src, directory, library_dirs, include_dirs, libraries, ccflags):
        from pathlib import Path

        assert Path(src).read_bytes() == probe.HOST_SOURCE
        calls.append((name, library_dirs, include_dirs, libraries, ccflags))
        path = (
            tmp_path / "escaped.so"
            if mode == "escape"
            else Path(directory) / "fixed.so"
        )
        path.write_bytes(b"compiled fixture")
        return str(path)

    module._build = build
    monkeypatch.setitem(sys.modules, "triton.runtime.build", module)

    def load(path):
        if mode == "noexec":
            raise OSError("failed to map segment")
        return SimpleNamespace(
            fixed_host_probe=lambda: 0 if mode == "wrong_result" else 42
        )

    monkeypatch.setattr(ctypes, "CDLL", load)
    if mode == "good":
        assert probe.fixed_host_compile()["result"] == 42
    else:
        with pytest.raises(OSError if mode == "noexec" else ValueError):
            probe.fixed_host_compile()
    assert calls == [("fixed_host_probe", [], [], [], [])]
    assert (resource.RLIMIT_CPU, (30, 30)) in limits
    assert (resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024)) in limits
