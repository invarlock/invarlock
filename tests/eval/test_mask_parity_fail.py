from __future__ import annotations

import pytest

from invarlock.cli.commands.run import _enforce_provider_parity
from invarlock.cli.errors import InvarlockError


def test_mask_parity_mismatch_aborts_in_ci():
    subj = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-A"}
    base = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-B"}
    with pytest.raises(InvarlockError) as ei:
        _enforce_provider_parity(subj, base, profile="ci")
    assert str(ei.value).startswith("[INVARLOCK:E003]")


def test_mask_parity_dev_profile_no_abort():
    subj = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-A"}
    base = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-B"}
    # Should not raise outside CI/Release profiles
    _enforce_provider_parity(subj, base, profile="dev")
