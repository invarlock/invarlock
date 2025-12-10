"""
Placeholder for golden-run offline regression coverage.

The public repo does not bundle the offline golden artifacts used in
internal CI, so this test is skipped in OSS. Keeping the file ensures
workflow test lists remain stable.
"""

import pytest


@pytest.mark.skip(reason="Offline golden runs are not shipped in the OSS repo.")
def test_offline_golden_runs_placeholder() -> None:
    pass
