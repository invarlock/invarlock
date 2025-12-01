from types import SimpleNamespace

import torch

from invarlock.core.api import RunConfig
from invarlock.core.runner import CoreRunner


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ensure parameters() is non-empty to provide a device
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        # produce a tiny scalar loss; avoid NaNs/zeros
        val = torch.tensor(0.01, device=input_ids.device)
        return SimpleNamespace(loss=val)


class IndexableCalib:
    """Indexable provider that raises on single-index access for 0,
    but supports slicing (to exercise debug snapshot branches)."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __len__(self):  # length hint path
        return len(self._batches)

    def __getitem__(self, idx):  # indexable path
        if isinstance(idx, int) and idx == 0:
            # Force the debug snapshot try/except branch
            raise RuntimeError("first item access not allowed")
        return self._batches[idx]


def test_eval_phase_debug_snapshot_with_indexable(monkeypatch):
    runner = CoreRunner()
    model = DummyModel()
    adapter = object()  # adapter is not used by _eval_phase/_compute_real_metrics

    # Minimal calibration dataset; supports slicing and batching
    calib = IndexableCalib(
        [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6]},
        ]
    )

    # Enable debug snapshot path
    monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")

    # Provide a basic config; use small window counts
    cfg = RunConfig(context={})

    metrics = runner._eval_phase(
        model=model,
        adapter=adapter,
        calibration_data=calib,
        report=SimpleNamespace(metrics={}, meta={}, evaluation_windows={}),
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    # Sanity check that evaluation produced expected keys
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")
    assert "latency_ms_per_tok" in metrics
