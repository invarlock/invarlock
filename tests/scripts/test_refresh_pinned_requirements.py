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
@pytest.mark.parametrize("retained_lock", (False, True))
def test_refresh_filters_evaluator_locks_and_fails_closed(
    tmp_path: Path, missing_dependency: bool, retained_lock: bool
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

    # Wheel derivation has its own authenticated-artifact tests. This fixture
    # verifies that refresh calls bootstrap before asking the compiler to resolve.
    (scripts / "build_hardened_accelerate_wheel.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "assert sys.argv[1:] == ['bootstrap']\n"
        "Path('bootstrap-verified').touch()\n",
        encoding="utf-8",
    )

    workflow = tmp_path / "requirements/workflows"
    if retained_lock:
        workflow.mkdir(parents=True)
        for name in (
            "lm-evaluation-harness-py312.txt",
            "lm-evaluation-harness-py312-cu129.txt",
            "openai-evals-runtime-py312.txt",
            "openai-evals-runtime-py312-cu129.txt",
        ):
            (workflow / name).write_text("retained==1.0\n")

    # The compiler is an external dependency. Exercise the actual refresh and
    # filter commands against a deterministic compiled closure without network.
    compiler = tmp_path / "uv"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert Path('bootstrap-verified').is_file()\n"
        "assert sys.argv[sys.argv.index('--find-links') + 1] == 'runtime/wheels'\n"
        "output = Path(sys.argv[sys.argv.index('--output-file') + 1])\n"
        "if '-full-' in output.name and os.environ['RETAINED_LOCK'] == '1':\n"
        "    assert output.read_text() == 'retained==1.0\\n'\n"
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
            "RETAINED_LOCK": "1" if retained_lock else "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    workflow = tmp_path / "requirements/workflows"
    if missing_dependency:
        assert result.returncode != 0
        assert "nltk" in result.stderr
        output = workflow / "openai-evals-runtime-py312.txt"
        if retained_lock:
            assert output.read_text() == "retained==1.0\n"
        else:
            assert not output.exists()
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
