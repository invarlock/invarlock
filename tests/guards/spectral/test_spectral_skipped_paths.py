import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_before_after_edit_skipped_when_not_prepared():
    model = nn.Linear(4, 4)
    guard = SpectralGuard()
    # Not prepared → before_edit and after_edit should be skipped without error
    guard.before_edit(model)
    guard.after_edit(model)
    # No violations/events required, but code path executed without raising
    assert isinstance(guard.events, list)
