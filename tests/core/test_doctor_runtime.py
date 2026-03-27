from __future__ import annotations

from types import SimpleNamespace

from invarlock.core import doctor_runtime as mod


def test_find_spec_safe_swallows_broken_hook() -> None:
    def _boom(_name: str) -> object | None:
        raise RuntimeError("broken hook")

    assert mod.find_spec_safe("torch", find_spec_fn=_boom) is None


def test_collect_torch_runtime_facts_collects_cuda_and_memory() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.0.0",
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _index: SimpleNamespace(total_memory=8e9),
        ),
    )

    facts = mod.collect_torch_runtime_facts(
        import_torch_fn=lambda: fake_torch,
        get_device_info_fn=lambda: {"cpu": {"available": True, "info": "ok"}},
        which_fn=lambda name: "/usr/bin/" + name if name == "nvcc" else None,
    )

    assert facts.version == "2.0.0"
    assert facts.device_info["cpu"]["available"] is True
    assert facts.cuda_toolkit_found is True
    assert facts.torch_cuda_build is True
    assert facts.cuda_available is True
    assert facts.gpu_memory_gb == 8.0
    assert facts.gpu_memory_low is False


def test_collect_torch_runtime_facts_tolerates_cuda_probe_errors() -> None:
    fake_torch = SimpleNamespace(
        __version__=None,
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError())
        ),
    )

    facts = mod.collect_torch_runtime_facts(
        import_torch_fn=lambda: fake_torch,
        get_device_info_fn=lambda: {},
        which_fn=lambda _name: None,
    )

    assert facts.version is None
    assert facts.cuda_toolkit_found is None
    assert facts.torch_cuda_build is None
    assert facts.cuda_available is None
    assert facts.gpu_memory_gb is None
    assert facts.gpu_memory_low is False


def test_collect_optional_dependency_facts_marks_bitsandbytes_runtime() -> None:
    specs = {
        "datasets": object(),
        "transformers": object(),
        "bitsandbytes": object(),
    }

    facts = mod.collect_optional_dependency_facts(
        has_cuda=False,
        bitsandbytes_runtime_available_fn=lambda: False,
        find_spec_fn=lambda name: specs.get(name),
    )

    by_name = {fact.name: fact for fact in facts}
    assert by_name["datasets"].present is True
    assert by_name["transformers"].present is True
    assert by_name["auto_gptq"].present is False
    assert by_name["bitsandbytes"].present is True
    assert by_name["bitsandbytes"].runtime_available is False
    assert by_name["bitsandbytes"].extra_hint == "gpu"
