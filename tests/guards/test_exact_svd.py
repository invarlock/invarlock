from __future__ import annotations

import pytest
import torch

from invarlock.guards.exact_svd import _exact_svd_input, exact_svdvals


def test_exact_svdvals_matches_torch_cpu_values_and_order() -> None:
    generator = torch.Generator().manual_seed(1701)
    matrix = torch.randn((31, 13), generator=generator, dtype=torch.float64)

    actual = exact_svdvals(matrix)
    expected = torch.linalg.svdvals(matrix.float())

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.all(actual[:-1] >= actual[1:])


def test_exact_svd_input_preserves_cpu_layout() -> None:
    matrix = torch.randn((4096, 8), dtype=torch.float32)

    prepared = _exact_svd_input(matrix)

    assert prepared is matrix
    assert prepared.shape == (4096, 8)


def test_exact_svdvals_preserves_cpu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_matrix: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("svd failed")

    monkeypatch.setattr(torch.linalg, "svdvals", fail)

    with pytest.raises(RuntimeError, match="svd failed"):
        exact_svdvals(torch.eye(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_exact_svdvals_preserves_cuda_execution_and_values() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1701)
    matrix = torch.randn((4096, 1024), generator=generator, device="cuda")

    prepared = _exact_svd_input(matrix)
    actual = exact_svdvals(matrix)
    expected = torch.linalg.svdvals(matrix)

    assert prepared.device.type == "cuda"
    assert prepared.shape == (4096, 1024)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_exact_svdvals_cuda_failure_retries_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.linalg.svdvals
    devices: list[str] = []

    def fail_cuda(value: torch.Tensor) -> torch.Tensor:
        devices.append(value.device.type)
        if value.device.type == "cuda":
            raise RuntimeError("cuda svd failed")
        return original(value)

    monkeypatch.setattr(torch.linalg, "svdvals", fail_cuda)

    actual = exact_svdvals(torch.eye(4, device="cuda"))

    assert devices == ["cuda", "cpu"]
    torch.testing.assert_close(actual, torch.ones(4), rtol=0.0, atol=0.0)
