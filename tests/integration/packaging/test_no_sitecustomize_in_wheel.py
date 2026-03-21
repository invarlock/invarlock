import os
import subprocess
import sys
import zipfile

import pytest


def _build_wheel(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        check=True,
    )
    return next(tmp_path.glob("*.whl"))


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_sitecustomize_not_in_wheel(tmp_path):
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert "sitecustomize.py" not in names


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_proof_pack_repo_assets_not_in_wheel(tmp_path):
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert "invarlock/public_contracts.py" in names
        assert "invarlock/_data/contracts/proof_pack_manifest.schema.json" in names
        assert "invarlock/_data/contracts/policy_pack.schema.json" in names
        assert "invarlock/_data/contracts/runtime_manifest.schema.json" in names
        assert "invarlock/_data/contracts/support_matrix.json" in names
        assert "invarlock/_data/contracts/model_family_catalog.json" in names
        assert not any(name.startswith("contracts/") for name in names)
        assert not any(name.startswith("scripts/proof_packs/") for name in names)


pytestmark = pytest.mark.integration
