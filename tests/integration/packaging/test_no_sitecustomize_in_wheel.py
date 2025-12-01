import os
import subprocess
import sys
import zipfile

import pytest


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_sitecustomize_not_in_wheel(tmp_path):
    # Build wheel locally
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        check=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert "sitecustomize.py" not in names


pytestmark = pytest.mark.integration
