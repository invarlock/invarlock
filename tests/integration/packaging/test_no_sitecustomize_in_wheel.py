import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _build_wheel(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    shutil.rmtree(repo_root / "build", ignore_errors=True)
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
        assert (
            "invarlock/_data/public_evidence/published_basis/gpt2/evaluation.report.json"
            in names
        )
        assert (
            "invarlock/_data/public_evidence/published_basis/bert/proof_pack_recipe.json"
            in names
        )
        assert not any(name.startswith("contracts/") for name in names)
        assert not any(name.startswith("scripts/proof_packs/") for name in names)
        assert "invarlock/core/config_dependencies.py" not in names
        assert "invarlock/core/run_orchestrator_execute_prepare.py" not in names


pytestmark = pytest.mark.integration
