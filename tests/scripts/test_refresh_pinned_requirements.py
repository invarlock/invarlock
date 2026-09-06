from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFRESH_SCRIPT = ROOT / "scripts/security/refresh_pinned_requirements.sh"


@pytest.mark.parametrize("missing_dependency", (False, True))
def test_refresh_filters_evaluator_locks_and_fails_closed(
    tmp_path: Path, missing_dependency: bool
) -> None:
    scripts = tmp_path / "scripts/security"
    scripts.mkdir(parents=True)
    shutil.copyfile(REFRESH_SCRIPT, scripts / REFRESH_SCRIPT.name)
    for name in (
        "build_cache_free_lm_eval_wheel.py",
        "build_restricted_openai_evals_wheel.py",
    ):
        # Subprocess coverage must retain the helpers' canonical source identity.
        (scripts / name).symlink_to(ROOT / "scripts/security" / name)

    # The compiler is an external dependency. Exercise the actual refresh and
    # filter commands against a deterministic compiled closure without network.
    compiler = tmp_path / "uv"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('--output-file') + 1])\n"
        "packages = 'retained==1.0\\n'\n"
        "if 'lm-evaluation-harness' in sys.argv[3]:\n"
        "    packages += 'lm-eval==0.4.12\\nsqlitedict==2.1.0\\n'\n"
        "    packages += 'rouge-score==0.1.2\\nnltk==3.10.3\\n'\n"
        "elif 'openai-evals-runtime' in sys.argv[3]:\n"
        "    packages += 'evals==3.0.1.post1\\n'\n"
        "    if os.environ['MISSING_EVALUATOR_DEPENDENCY'] != '1':\n"
        "        packages += 'nltk==3.10.3\\n'\n"
        "output.write_text(packages)\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / REFRESH_SCRIPT.name), "--write"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "MISSING_EVALUATOR_DEPENDENCY": "1" if missing_dependency else "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    workflow = tmp_path / "requirements/workflows"
    if missing_dependency:
        assert result.returncode != 0
        assert "nltk" in result.stderr
        assert not (workflow / "openai-evals-runtime-py312.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        for name in (
            "lm-evaluation-harness-py312.txt",
            "lm-evaluation-harness-py312-cu129.txt",
            "openai-evals-runtime-py312.txt",
            "openai-evals-runtime-py312-cu129.txt",
        ):
            assert (workflow / name).read_text() == "retained==1.0\n"
    assert not list(workflow.glob(".*full*.txt"))
