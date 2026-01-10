# ruff: noqa: I001,E402,F811
from __future__ import annotations

from types import SimpleNamespace

from invarlock.cli.determinism import apply_determinism_preset


def test_determinism_preset_coercion_failure_paths(monkeypatch) -> None:
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(
        profile=object(),  # type: ignore[arg-type]
        device=object(),  # type: ignore[arg-type]
        seed=1,
        threads=object(),  # type: ignore[arg-type]
    )
    assert payload["requested"] == "off"
    assert payload["level"] == "off"
    assert payload["device"] == "cpu"


def test_determinism_preset_off_in_dev_profile(monkeypatch) -> None:
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)
    payload = apply_determinism_preset(profile="dev", device="cpu", seed=123, threads=2)
    assert payload["requested"] == "off"
    assert payload["level"] == "off"
    assert "env" not in payload


def test_determinism_preset_downgrades_when_torch_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("invarlock.cli.determinism.torch", None)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(
        profile="release", device="cuda:0", seed=7, threads=2
    )
    assert payload["requested"] == "strict"
    assert payload["level"] == "tolerance"
    assert payload["device"] == "cuda:0"
    assert "env" in payload
    assert payload["env"].get("CUBLAS_WORKSPACE_CONFIG") in {":16:8", ":4096:8", None}
    assert payload["env"].get("OMP_NUM_THREADS") == "2"


def test_determinism_preset_marks_tolerance_when_deterministic_algos_fail(
    monkeypatch,
) -> None:
    # Build a tiny torch stub that fails hard-determinism but supports warn_only fallback.
    class _Matmul:
        allow_tf32 = True

    class _Cuda:
        matmul = _Matmul()

    class _Cudnn:
        benchmark = True
        deterministic = False
        allow_tf32 = True

    class _Backends:
        cuda = _Cuda()
        cudnn = _Cudnn()

    calls: list[tuple[bool, bool]] = []

    def _use_deterministic_algorithms(enabled: bool, warn_only: bool = False) -> None:
        calls.append((bool(enabled), bool(warn_only)))
        if enabled and not warn_only:
            raise RuntimeError("not supported")

    fake_torch = SimpleNamespace(
        backends=_Backends(),
        set_num_threads=lambda *_a, **_k: None,
        set_num_interop_threads=lambda *_a, **_k: None,
        use_deterministic_algorithms=_use_deterministic_algorithms,
        initial_seed=lambda: 7,
        are_deterministic_algorithms_enabled=lambda: False,
    )

    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="ci", device="cuda:0", seed=7, threads=1)
    assert payload["requested"] == "strict"
    assert payload["level"] == "tolerance"
    assert calls and calls[0] == (True, False)


def test_determinism_preset_seed_bundle_fallbacks(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("boom")

    fake_torch = SimpleNamespace(initial_seed=_boom)
    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)
    monkeypatch.setattr("invarlock.cli.determinism.np.random.get_state", _boom)

    payload = apply_determinism_preset(profile="dev", device="cpu", seed=3, threads=1)
    assert payload["seeds"]["numpy"] == 3
    assert payload["seeds"]["torch"] == 3


def test_determinism_preset_marks_tolerance_when_backend_access_raises(
    monkeypatch,
) -> None:
    class _BackendsRaises:
        def __getattr__(self, _name: str):
            raise RuntimeError("nope")

    fake_torch = SimpleNamespace(
        backends=_BackendsRaises(),
        set_num_threads=lambda *_a, **_k: None,
        set_num_interop_threads=lambda *_a, **_k: None,
        use_deterministic_algorithms=lambda *_a, **_k: None,
        initial_seed=lambda: 7,
        are_deterministic_algorithms_enabled=lambda: True,
    )
    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="ci", device="cuda:0", seed=7, threads=1)
    assert payload["level"] == "tolerance"
    assert payload.get("notes") is not None


def test_determinism_preset_strict_with_minimal_torch_stub_hits_false_branches(
    monkeypatch,
) -> None:
    class _Backends:
        cuda = SimpleNamespace()
        cudnn = None

    fake_torch = SimpleNamespace(backends=_Backends(), initial_seed=lambda: 7)
    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="ci", device="cuda:0", seed=7, threads=1)
    assert payload["level"] == "strict"


def test_determinism_preset_warn_only_fallback_failure(monkeypatch) -> None:
    class _Matmul:
        allow_tf32 = True

    class _Cuda:
        matmul = _Matmul()

    class _Cudnn:
        benchmark = True
        deterministic = False
        allow_tf32 = True

    class _Backends:
        cuda = _Cuda()
        cudnn = _Cudnn()

    def _always_fail(*_a, **_k) -> None:
        raise RuntimeError("fail")

    fake_torch = SimpleNamespace(
        backends=_Backends(),
        set_num_threads=lambda *_a, **_k: None,
        set_num_interop_threads=lambda *_a, **_k: None,
        use_deterministic_algorithms=_always_fail,
        initial_seed=lambda: 7,
    )

    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="ci", device="cuda:0", seed=7, threads=1)
    assert payload["level"] == "tolerance"


def test_determinism_preset_cuda_low_memory_selects_cublas_fallback(monkeypatch) -> None:
    class _CudaProps:
        total_memory = 1 * 1024**3

    class _Cuda:
        def get_device_properties(self, _idx: int):  # type: ignore[no-untyped-def]
            return _CudaProps()

    class _CudnnNoDeterministic:
        benchmark = True

    class _Matmul:
        allow_tf32 = True

    class _Backends:
        cuda = SimpleNamespace(matmul=_Matmul())
        cudnn = _CudnnNoDeterministic()

    fake_torch = SimpleNamespace(
        cuda=_Cuda(),
        backends=_Backends(),
        set_num_threads=lambda *_a, **_k: None,
        set_num_interop_threads=lambda *_a, **_k: None,
        use_deterministic_algorithms=lambda *_a, **_k: None,
        initial_seed=lambda: 7,
        are_deterministic_algorithms_enabled=lambda: True,
    )

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr("invarlock.cli.determinism.torch", fake_torch)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="ci", device="cuda:0", seed=7, threads=2)
    assert payload["requested"] == "strict"
    assert payload["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_determinism_preset_prunes_empty_torch_payload_when_random_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr("invarlock.cli.determinism.random.random", lambda: 1 / 0)
    monkeypatch.setattr("invarlock.cli.determinism.set_seed", lambda *_a, **_k: None)

    payload = apply_determinism_preset(profile="dev", device="cpu", seed=1, threads=1)
    assert "torch" not in payload
