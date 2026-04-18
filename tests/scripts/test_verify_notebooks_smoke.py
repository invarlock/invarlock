from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "verify_notebooks_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "tests_verify_notebooks_smoke", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_script_rewrites_notebook_shell_commands(tmp_path: Path) -> None:
    module = _load_script_module()
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["!invarlock doctor --json || true\n"],
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "%%bash\n",
                            "invarlock evaluate --baseline gpt2 --subject gpt2\n",
                            "make runtime-image-podman\n",
                            "test -f reports/eval/runtime.manifest.json\n",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_py = tmp_path / "demo.py"

    module.write_script(nb_path=nb_path, out_py=out_py, skip_pip=True)

    rendered = out_py.read_text(encoding="utf-8")
    assert "sys.executable" in rendered
    assert 'replacement = f"{indent}{env_prefix}{py} -m invarlock"' in rendered
    assert "return f\"echo '[skip-host] {stripped}'\"" in rendered
    assert "_run_bash('invarlock doctor --json || true')" in rendered
    assert (
        "_run_bash('invarlock evaluate --baseline sshleifer/tiny-gpt2 --subject sshleifer/tiny-gpt2\\n"
        "make runtime-image-podman\\n"
        "test -f reports/eval/runtime.manifest.json\\n')"
    ) in rendered
    assert 'stripped.startswith("make runtime-image")' in rendered
    assert 'stripped.startswith("make runtime-smoke")' in rendered


def test_write_script_can_skip_curated_model_loading_cells(tmp_path: Path) -> None:
    module = _load_script_module()
    nb_path = tmp_path / "invarlock_policy_tiers.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "%%bash\n",
                            "invarlock evaluate --baseline gpt2 --subject gpt2\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "source": ["!invarlock doctor --json || true\n"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_py = tmp_path / "demo.py"

    module.write_script(
        nb_path=nb_path,
        out_py=out_py,
        skip_pip=True,
        skip_model_loading=True,
    )

    rendered = out_py.read_text(encoding="utf-8")
    assert "(skip-model-loading)" in rendered
    assert (
        "_run_bash('invarlock evaluate --baseline gpt2 --subject gpt2\\n')"
        not in rendered
    )
    assert "_run_bash('invarlock doctor --json || true')" in rendered


def test_rewrite_live_smoke_shell_script_uses_smoke_assets_for_heavy_examples() -> None:
    module = _load_script_module()

    rendered = module._rewrite_live_smoke_shell_script(
        "invarlock evaluate \\\n"
        "  --baseline gpt2 \\\n"
        "  --subject distilgpt2 \\\n"
        "  --preset configs/presets/causal_lm/wikitext2_512.yaml\n"
    )

    assert "--baseline sshleifer/tiny-gpt2" in rendered
    assert "--subject sshleifer/tiny-gpt2" in rendered
    assert "configs/presets/causal_lm/gpt2_smoke_128.yaml" in rendered
    assert "--baseline gpt2" not in rendered
    assert "--subject distilgpt2" not in rendered


def test_rewrite_live_smoke_shell_script_normalizes_profiles_and_seed_counts() -> None:
    module = _load_script_module()

    rendered = module._rewrite_live_smoke_shell_script(
        "invarlock advanced calibrate null-sweep \\\n"
        "  --config configs/calibration/null_sweep_ci.yaml \\\n"
        "  --profile release \\\n"
        "  --n-seeds 10\n"
    )

    assert "configs/calibration/null_sweep_smoke.yaml" in rendered
    assert "--profile dev" in rendered
    assert "--n-seeds 1" in rendered
    assert "--profile release" not in rendered
    assert "--n-seeds 10" not in rendered


def test_seed_curated_demo_outputs_writes_expected_reports(tmp_path: Path) -> None:
    module = _load_script_module()
    policy_run_dir = tmp_path / "policy"
    module._seed_curated_demo_outputs(
        nb_path=tmp_path / "invarlock_policy_tiers.ipynb",
        run_dir=policy_run_dir,
    )
    python_api_run_dir = tmp_path / "python_api"
    module._seed_curated_demo_outputs(
        nb_path=tmp_path / "invarlock_python_api.ipynb",
        run_dir=python_api_run_dir,
    )
    compare_run_dir = tmp_path / "compare"
    module._seed_curated_demo_outputs(
        nb_path=tmp_path / "invarlock_compare_evaluate.ipynb",
        run_dir=compare_run_dir,
    )
    quickstart_run_dir = tmp_path / "quickstart"
    module._seed_curated_demo_outputs(
        nb_path=tmp_path / "invarlock_quickstart_cpu.ipynb",
        run_dir=quickstart_run_dir,
    )

    assert (
        policy_run_dir / "reports" / "tier_conservative" / "evaluation.report.json"
    ).is_file()
    assert (
        policy_run_dir / "reports" / "tier_balanced" / "evaluation.report.json"
    ).is_file()
    assert (
        policy_run_dir / "reports" / "tier_aggressive" / "evaluation.report.json"
    ).is_file()
    assert (
        python_api_run_dir / "reports" / "python_api" / "evaluation.report.json"
    ).is_file()
    assert (
        compare_run_dir / "reports" / "compare_evaluate" / "evaluation.report.json"
    ).is_file()
    assert (
        quickstart_run_dir / "reports" / "eval" / "evaluation.report.json"
    ).is_file()
    conservative = json.loads(
        (
            policy_run_dir / "reports" / "tier_conservative" / "evaluation.report.json"
        ).read_text(encoding="utf-8")
    )
    assert conservative["primary_metric"]["ratio_vs_baseline"] == 2.0
    assert (
        conservative["resolved_policy"]["metrics"]["pm_ratio"]["ratio_limit_base"]
        == 1.05
    )
    python_api_report = json.loads(
        (
            python_api_run_dir / "reports" / "python_api" / "evaluation.report.json"
        ).read_text(encoding="utf-8")
    )
    final_logs = python_api_report["evaluation_windows"]["final"]["logloss"]
    final_counts = python_api_report["evaluation_windows"]["final"]["token_counts"]
    weighted_final = sum(
        float(logloss) * int(count)
        for logloss, count in zip(final_logs, final_counts, strict=False)
    ) / sum(int(count) for count in final_counts)
    assert python_api_report["primary_metric"]["final"] == pytest.approx(
        math.exp(weighted_final)
    )
    assert python_api_report["primary_metric"]["ratio_vs_baseline"] == pytest.approx(
        1.0
    )
