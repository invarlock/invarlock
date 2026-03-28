from __future__ import annotations

import pytest

from invarlock.cli.run_pairing import enforce_provider_parity
from invarlock.core.exceptions import InvarlockError


def test_enforce_provider_parity_missing_tokenizer_digest_in_ci_raises():
    with pytest.raises(InvarlockError) as ei:
        enforce_provider_parity(
            {"ids_sha256": "abc"}, {"ids_sha256": "def"}, profile="ci"
        )
    assert ei.value.code == "E004"
