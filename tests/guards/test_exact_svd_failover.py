from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from invarlock.guards import exact_svd as exact_svd_module


@dataclass(frozen=True)
class _Device:
    type: str


class _FakeMatrix:
    """Minimal tensor-like object that makes accelerator failover observable."""

    def __init__(self, device_type: str, calls: list[str]) -> None:
        self.device = _Device(device_type)
        self._calls = calls

    def float(self) -> _FakeMatrix:
        self._calls.append(f"float:{self.device.type}")
        return self

    def cpu(self) -> _FakeMatrix:
        self._calls.append(f"cpu:{self.device.type}")
        return _FakeMatrix("cpu", self._calls)


def test_accelerator_runtime_failure_retries_once_with_cpu_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise failover without making this contract conditional on local CUDA."""

    calls: list[str] = []
    accelerator_value = _FakeMatrix("cuda", calls)
    original = _FakeMatrix("cuda", calls)
    expected = torch.tensor([3.0, 1.0])

    monkeypatch.setattr(
        exact_svd_module,
        "_exact_svd_input",
        lambda _matrix: accelerator_value,
    )

    def svdvals(value: _FakeMatrix) -> torch.Tensor:
        calls.append(f"svd:{value.device.type}")
        if value.device.type != "cpu":
            raise torch.linalg.LinAlgError("accelerator solver failed")
        return expected

    monkeypatch.setattr(torch.linalg, "svdvals", svdvals)

    actual = exact_svd_module.exact_svdvals(original)  # type: ignore[arg-type]

    assert actual is expected
    assert calls == ["svd:cuda", "float:cuda", "cpu:cuda", "svd:cpu"]


def test_cpu_linalg_failure_is_not_hidden_by_a_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fail(_matrix: torch.Tensor) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        raise torch.linalg.LinAlgError("invalid singular-value input")

    monkeypatch.setattr(torch.linalg, "svdvals", fail)

    with pytest.raises(torch.linalg.LinAlgError, match="invalid singular-value input"):
        exact_svd_module.exact_svdvals(torch.eye(2))

    assert attempts == 1
