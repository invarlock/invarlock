#!/usr/bin/env python3
"""
Smoke-run Jupyter notebooks by extracting code cells into temp .py scripts.

This is a lightweight verifier intended for CI/dev sanity checks when full
notebook execution (via Jupyter kernels) isn't available.

What it does:
  - Reads `notebooks/*.ipynb`
  - Writes a runnable `.py` per notebook into a temp run directory
  - Converts Jupyter shell escapes (`!cmd`) into `bash -c ...` subprocess calls
  - Converts `%%bash` cells into `bash -c ...` subprocess calls
  - Runs each generated script in an isolated temp working directory

Defaults:
  - Skips `pip install ...` lines inside notebooks (assumes deps already present)
  - Sets a cautious env for HF/transformers + InvarLock demos
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST_EXECUTION_ENV = "INVARLOCK_ALLOW_HOST_EXECUTION"
SMOKE_MODEL_SCRIPT_REWRITES = (
    (re.compile(r"(?m)(--baseline\s+)distilgpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--baseline\s+)gpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--subject\s+)distilgpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--subject\s+)gpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--profile\s+)(?:ci|release)\b"), r"\1dev"),
    (re.compile(r"(?m)(--n-seeds\s+)\d+\b"), r"\g<1>1"),
    (
        re.compile(r"configs/presets/causal_lm/wikitext2_512\.yaml"),
        "configs/presets/causal_lm/gpt2_smoke_128.yaml",
    ),
    (
        re.compile(r"configs/calibration/null_sweep_ci\.yaml"),
        "configs/calibration/null_sweep_smoke.yaml",
    ),
    (
        re.compile(r"configs/calibration/rmt_ve_sweep_ci\.yaml"),
        "configs/calibration/rmt_ve_sweep_smoke.yaml",
    ),
)
_ENV_PREFIX_PATTERN = r"(?P<env>(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*)"
_INVARLOCK_PREFIX = re.compile(rf"^(?P<indent>\s*){_ENV_PREFIX_PATTERN}invarlock(?=\s)")
_PY_INVARLOCK_PREFIX = re.compile(
    rf"^(?P<indent>\s*){_ENV_PREFIX_PATTERN}(?:python|python3)\s+-m\s+invarlock(?=\s)"
)
_CURATED_NOTEBOOK_SKIP_MARKERS = {
    "invarlock_compare_evaluate.ipynb": ("invarlock evaluate",),
    "invarlock_custom_datasets.ipynb": ("invarlock evaluate",),
    "invarlock_evaluation_report_deep_dive.ipynb": ("invarlock evaluate",),
    "invarlock_policy_tiers.ipynb": ("invarlock evaluate",),
    "invarlock_python_api.ipynb": (
        "from transformers import AutoTokenizer",
        "from invarlock.adapters.auto import HF_Auto_Adapter",
    ),
    "invarlock_quickstart_cpu.ipynb": ("invarlock evaluate",),
}
_POLICY_TIER_RATIO_LIMITS = {
    "conservative": 1.05,
    "balanced": 1.10,
    "aggressive": 1.20,
}


def _preferred_invarlock_python() -> str:
    workspace_python = ROOT / ".venv" / "bin" / "python"
    if workspace_python.is_file():
        return str(workspace_python)
    return sys.executable


def _demo_window_summary(section: dict[str, object]) -> tuple[float, float, int] | None:
    loglosses = section.get("logloss")
    token_counts = section.get("token_counts")
    if not isinstance(loglosses, list) or not loglosses:
        return None
    if not isinstance(token_counts, list) or len(token_counts) != len(loglosses):
        token_counts = [1] * len(loglosses)

    weighted_total = 0.0
    total_tokens = 0
    for logloss, token_count in zip(loglosses, token_counts, strict=False):
        try:
            logloss_f = float(logloss)
            token_count_i = int(token_count)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(logloss_f) or token_count_i <= 0:
            return None
        weighted_total += logloss_f * token_count_i
        total_tokens += token_count_i
    if total_tokens <= 0:
        return None
    mean_logloss = weighted_total / total_tokens
    return mean_logloss, math.exp(mean_logloss), total_tokens


def _rewrite_live_smoke_shell_script(script: str) -> str:
    rewritten = script
    for pattern, replacement in SMOKE_MODEL_SCRIPT_REWRITES:
        rewritten = pattern.sub(replacement, rewritten)
    return rewritten


def _iter_code_cells(nb: dict) -> list[tuple[int, str]]:
    cells = nb.get("cells", [])
    out: list[tuple[int, str]] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        out.append((idx, src))
    return out


def _is_pip_install(cmd: str) -> bool:
    tokens = cmd.strip().split()
    if not tokens:
        return False
    if tokens[0] == "pip":
        return "install" in tokens[1:]
    if (
        len(tokens) >= 3
        and tokens[0] in {"python", "python3"}
        and tokens[1] == "-m"
        and tokens[2] == "pip"
    ):
        return "install" in tokens[3:]
    return False


def _convert_cell(
    *,
    cell_index: int,
    cell_source: str,
    notebook_name: str,
    skip_pip: bool,
    skip_model_loading: bool,
) -> list[str]:
    lines = cell_source.splitlines()
    if not lines:
        return []
    if _should_skip_cell(
        notebook_name=notebook_name,
        cell_source=cell_source,
        skip_model_loading=skip_model_loading,
    ):
        return [
            f"print({(f'[{notebook_name}] cell {cell_index} (skip-model-loading)').__repr__()})\n",
            "\n",
        ]

    first = lines[0].lstrip()
    if first.startswith("%%bash"):
        script = "\n".join(lines[1:]).rstrip() + "\n"
        if not skip_model_loading:
            script = _rewrite_live_smoke_shell_script(script)
        return [
            f"print({(f'[{notebook_name}] cell {cell_index} (%%bash)').__repr__()})\n",
            f"_run_bash({script!r})\n",
            "\n",
        ]

    out: list[str] = [f"print({(f'[{notebook_name}] cell {cell_index}').__repr__()})\n"]
    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("!"):
            cmd = stripped[1:].strip()
            if skip_pip and _is_pip_install(cmd):
                out.append(f"print({(f'  (skip) {cmd}').__repr__()})\n")
                continue
            if not skip_model_loading:
                cmd = _rewrite_live_smoke_shell_script(cmd)
            out.append(f"_run_bash({cmd!r})\n")
            continue
        out.append(raw + "\n")
    out.append("\n")
    return out


def _should_skip_shell_line(stripped: str) -> bool:
    if (
        stripped.startswith("make runtime-image")
        or stripped.startswith("make runtime-smoke")
        or stripped.startswith("make container-default-smoke")
        or stripped.startswith("make container-front-door-smoke")
    ):
        return True
    if stripped.startswith(("docker ", "podman ")):
        return True
    return "runtime.manifest.json" in stripped and (
        stripped.startswith("test -f ")
        or stripped.startswith("[ -f ")
        or stripped.startswith("stat ")
    )


def _should_skip_cell(
    *,
    notebook_name: str,
    cell_source: str,
    skip_model_loading: bool,
) -> bool:
    if not skip_model_loading:
        return False
    markers = _CURATED_NOTEBOOK_SKIP_MARKERS.get(notebook_name, ())
    return any(marker in cell_source for marker in markers)


def _demo_verify_pass_report() -> dict:
    src_root = ROOT / "src"
    if (src_root / "invarlock").is_dir():
        src_path = str(src_root)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        try:
            from invarlock.core.auto_tuning import resolve_tier_policies
            from invarlock.reporting.report_make import make_report
            from invarlock.reporting.runtime_policy_receipt import (
                build_runtime_policy_receipt,
            )

            run_report = {
                "meta": {
                    "model_id": "docs-demo-model",
                    "adapter": "hf_causal",
                    "commit": "docs-demo",
                    "seed": 42,
                    "device": "cpu",
                    "ts": "2026-04-03T00:00:00+00:00",
                    "auto": {
                        "enabled": False,
                        "tier": "balanced",
                        "probes_used": 0,
                        "target_pm_ratio": None,
                    },
                },
                "data": {
                    "dataset": "unit",
                    "split": "validation",
                    "seq_len": 8,
                    "stride": 8,
                    "preview_n": 2,
                    "final_n": 2,
                },
                "edit": {
                    "name": "noop",
                    "plan_digest": "docs-demo",
                    "deltas": {
                        "params_changed": 0,
                        "sparsity": None,
                        "bitwidth_map": None,
                        "layers_modified": 0,
                    },
                },
                "guards": [],
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 10.0,
                        "final": 10.0,
                    },
                    "bootstrap": {
                        "method": "percentile",
                        "replicates": 50,
                        "alpha": 0.05,
                        "seed": 0,
                        "coverage": {
                            "preview": {"used": 2},
                            "final": {"used": 2},
                        },
                    },
                    "preview_final_slice_delta_summary": {
                        "mean": 0.0,
                        "ci": [-0.01, 0.01],
                        "basis": "independent_disjoint_slices",
                        "paired": False,
                        "ci_method": "independent_percentile_delta_log",
                        "ci_reason": None,
                        "preview_windows": 2,
                        "final_windows": 2,
                        "degenerate": False,
                        "degenerate_reason": None,
                    },
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "preview_total_tokens": 50000,
                    "final_total_tokens": 50000,
                    "logloss_delta": 0.0,
                    "logloss_delta_ci": [-0.01, 0.01],
                },
                "artifacts": {
                    "events_path": "",
                    "logs_path": "",
                    "checkpoint_path": None,
                },
                "flags": {
                    "guard_recovered": False,
                    "rollback_reason": None,
                },
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [3, 4],
                        "logloss": [2.30, 2.30],
                        "token_counts": [100, 100],
                    },
                    "final": {
                        "window_ids": [1, 2],
                        "logloss": [2.30, 2.30],
                        "token_counts": [100, 100],
                    },
                },
            }
            baseline_report = {
                "run_id": "docs-demo-base",
                "model_id": "docs-demo-model",
                "meta": {
                    "seed": 0,
                    "model_id": "docs-demo-model",
                    "adapter": "hf_causal",
                    "auto": {"tier": "balanced"},
                },
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [3, 4],
                        "logloss": [2.30, 2.30],
                        "token_counts": [100, 100],
                    },
                    "final": {
                        "window_ids": [1, 2],
                        "logloss": [2.30, 2.30],
                        "token_counts": [100, 100],
                    },
                },
                "data": {
                    "seq_len": 8,
                    "preview_n": 2,
                    "final_n": 2,
                    "dataset": "unit",
                    "split": "validation",
                    "stride": 8,
                },
                "edit": {
                    "name": "none",
                    "plan_digest": "0",
                    "deltas": {
                        "params_changed": 0,
                        "layers_modified": 0,
                        "sparsity": None,
                        "bitwidth_map": None,
                    },
                },
                "guards": [],
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 10.0,
                        "final": 10.0,
                    },
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
                "artifacts": {
                    "events_path": "",
                    "logs_path": "",
                    "checkpoint_path": None,
                },
                "flags": {
                    "guard_recovered": False,
                    "rollback_reason": None,
                },
            }
            baseline_final_summary = _demo_window_summary(
                baseline_report["evaluation_windows"]["final"]
            )
            subject_final_summary = _demo_window_summary(
                run_report["evaluation_windows"]["final"]
            )
            if baseline_final_summary is not None:
                baseline_mean_logloss, baseline_ppl, baseline_tokens = (
                    baseline_final_summary
                )
                run_report["metrics"]["primary_metric"]["preview"] = baseline_ppl
                run_report["metrics"]["preview_total_tokens"] = baseline_tokens
                baseline_report["metrics"]["primary_metric"]["preview"] = baseline_ppl
                baseline_report["metrics"]["primary_metric"]["final"] = baseline_ppl
            else:
                baseline_mean_logloss = None
            if subject_final_summary is not None:
                subject_mean_logloss, subject_ppl, subject_tokens = (
                    subject_final_summary
                )
                run_report["metrics"]["primary_metric"]["final"] = subject_ppl
                run_report["metrics"]["final_total_tokens"] = subject_tokens
            else:
                subject_mean_logloss = None
            if baseline_mean_logloss is not None and subject_mean_logloss is not None:
                delta_mean_logloss = subject_mean_logloss - baseline_mean_logloss
                run_report["metrics"]["preview_final_slice_delta_summary"]["mean"] = (
                    delta_mean_logloss
                )
                run_report["metrics"]["logloss_delta"] = delta_mean_logloss
            for raw_report in (run_report, baseline_report):
                meta = raw_report["meta"]
                auto = meta.get("auto") if isinstance(meta, dict) else None
                tier = str(auto.get("tier") if isinstance(auto, dict) else "balanced")
                edit_name = str(raw_report["edit"]["name"])
                policies = resolve_tier_policies(tier, edit_name, profile="dev")
                resolved, receipt = build_runtime_policy_receipt(
                    policies,
                    raw_report["guards"],
                    tier=tier,
                    profile="dev",
                    edit_name=edit_name,
                )
                raw_report["resolved_policy"] = resolved
                raw_report["policy_resolution"] = receipt
            report = make_report(run_report, baseline_report)
            report.setdefault("meta", {})["demo_input_mode"] = (
                "canonical_generated_fixture"
            )
            report.setdefault("provenance", {}).setdefault(
                "provider_digest", {"ids_sha256": "docs-demo-provider"}
            )
            return report
        except Exception as exc:
            raise RuntimeError(
                "Cannot build canonical notebook demo evidence."
            ) from exc


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seed_curated_demo_outputs(*, nb_path: Path, run_dir: Path) -> None:
    if nb_path.name == "invarlock_policy_tiers.ipynb":
        base_report = _demo_verify_pass_report()
        for tier, ratio_limit_base in _POLICY_TIER_RATIO_LIMITS.items():
            report = json.loads(json.dumps(base_report))
            report.setdefault("primary_metric", {})["ratio_vs_baseline"] = 2.0
            resolved_policy = report.setdefault("resolved_policy", {})
            metrics = resolved_policy.setdefault("metrics", {})
            pm_ratio = metrics.setdefault("pm_ratio", {})
            pm_ratio["ratio_limit_base"] = ratio_limit_base
            _write_json(
                run_dir / "reports" / f"tier_{tier}" / "evaluation.report.json", report
            )
        return

    if nb_path.name == "invarlock_python_api.ipynb":
        _write_json(
            run_dir / "reports" / "python_api" / "evaluation.report.json",
            _demo_verify_pass_report(),
        )
        return

    report_targets = {
        "invarlock_compare_evaluate.ipynb": (
            run_dir / "reports" / "compare_evaluate" / "evaluation.report.json"
        ),
        "invarlock_custom_datasets.ipynb": (
            run_dir / "reports" / "byod" / "evaluation.report.json"
        ),
        "invarlock_evaluation_report_deep_dive.ipynb": (
            run_dir / "reports" / "eval_deep_dive" / "evaluation.report.json"
        ),
        "invarlock_quickstart_cpu.ipynb": (
            run_dir / "reports" / "eval" / "evaluation.report.json"
        ),
    }
    target = report_targets.get(nb_path.name)
    if target is not None:
        report = _demo_verify_pass_report()
        if nb_path.name == "invarlock_custom_datasets.ipynb":
            report.setdefault("dataset", {})["provider"] = "local_jsonl"
            report.setdefault("provenance", {})["provider_digest"] = {
                "ids_sha256": "docs-demo-provider",
                "provider": "local_jsonl",
            }
        _write_json(target, report)


def write_script(
    *,
    nb_path: Path,
    out_py: Path,
    skip_pip: bool,
    skip_model_loading: bool = False,
) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = _iter_code_cells(nb)

    header = "\n".join(
        [
            "#!/usr/bin/env python3",
            f"# Generated from: {nb_path}",
            f"# Generated at: {datetime.now(tz=UTC).isoformat()}",
            "",
            "from __future__ import annotations",
            "",
            "import os",
            "import re",
            "import shlex",
            "import subprocess",
            "import sys",
            "",
            "",
            f"REPO_INVARLOCK_PYTHON = {_preferred_invarlock_python()!r}",
            f"HOST_EXECUTION_ENV = {HOST_EXECUTION_ENV!r}",
            f"_INVARLOCK_PREFIX = re.compile({_INVARLOCK_PREFIX.pattern!r})",
            f"_PY_INVARLOCK_PREFIX = re.compile({_PY_INVARLOCK_PREFIX.pattern!r})",
            "",
            "",
            "def _resolve_invarlock_python() -> str:",
            "    return REPO_INVARLOCK_PYTHON if os.path.isfile(REPO_INVARLOCK_PYTHON) else sys.executable",
            "",
            "",
            "def _normalize_shell_line(line: str) -> str:",
            "    stripped = line.strip()",
            '    if not stripped or stripped.startswith("#"):',
            "        return line",
            "    if (",
            '        stripped.startswith("make runtime-image")',
            '        or stripped.startswith("make runtime-smoke")',
            '        or stripped.startswith(("docker ", "podman "))',
            "        or (",
            '            "runtime.manifest.json" in stripped',
            "            and (",
            '                stripped.startswith("test -f ")',
            '                or stripped.startswith("[ -f ")',
            '                or stripped.startswith("stat ")',
            "            )",
            "        )",
            "    ):",
            "        return f\"echo '[skip-host] {stripped}'\"",
            "",
            "    py = shlex.quote(_resolve_invarlock_python())",
            "",
            "    def _replace(match: re.Match[str]) -> str:",
            '        indent = match.group("indent")',
            '        env_prefix = match.group("env") or ""',
            '        replacement = f"{indent}{env_prefix}{py} -m invarlock"',
            "        if (",
            "            HOST_EXECUTION_ENV not in env_prefix",
            '            and " advanced calibrate" in line',
            "        ):",
            '            replacement = f"{indent}{env_prefix}{HOST_EXECUTION_ENV}=1 {py} -m invarlock"',
            "        return replacement",
            "",
            "    normalized = _PY_INVARLOCK_PREFIX.sub(_replace, line, count=1)",
            "    normalized = _INVARLOCK_PREFIX.sub(_replace, normalized, count=1)",
            "    return normalized",
            "",
            "",
            "def _run_bash(cmd: str) -> None:",
            "    # Use bash so notebook-style commands (pipes, heredocs, exports) behave.",
            "    # Avoid `bash -l` (login shell), which can reset PATH and break venv/conda",
            "    # command resolution for `invarlock`, `python`, etc.",
            '    normalized = "\\n".join(_normalize_shell_line(line) for line in cmd.splitlines())',
            '    if cmd.endswith("\\n"):',
            '        normalized += "\\n"',
            '    subprocess.run(["bash", "-c", normalized], check=True, env=os.environ.copy())',
            "",
            "",
            "def main() -> None:",
            "",
        ]
    )

    body: list[str] = []
    notebook_name = nb_path.name
    for cell_index, cell_source in code_cells:
        body.extend(
            _convert_cell(
                cell_index=cell_index,
                cell_source=cell_source,
                notebook_name=notebook_name,
                skip_pip=skip_pip,
                skip_model_loading=skip_model_loading,
            )
        )

    footer = """\


if __name__ == "__main__":
    main()
"""

    # Indent body under main().
    indented_body = []
    for line in body:
        if line.strip():
            indented_body.append("    " + line)
        else:
            indented_body.append(line)

    out_py.write_text(header + "".join(indented_body) + footer, encoding="utf-8")


def _env_for_run() -> dict[str, str]:
    env = os.environ.copy()
    # Prefer local repo code when invoked from source checkout.
    env["PYTHONPATH"] = str(ROOT / "src") + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_script(*, script_path: Path, cwd: Path, timeout_s: int) -> None:
    env = _env_for_run()
    stdout_path = cwd / "stdout.txt"
    stderr_path = cwd / "stderr.txt"
    with (
        stdout_path.open("w", encoding="utf-8") as out,
        stderr_path.open("w", encoding="utf-8") as err,
    ):
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=err,
            timeout=timeout_s,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Notebook smoke failed: {script_path.name} (exit={proc.returncode})\n"
            f"  cwd: {cwd}\n"
            f"  stdout: {stdout_path}\n"
            f"  stderr: {stderr_path}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook paths (default: notebooks/*.ipynb)",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root (default: /tmp/invarlock_notebook_smoke_<ts>)",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help="Per-notebook timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--run-pip",
        action="store_true",
        help="Do not skip `pip install ...` lines from notebooks.",
    )
    parser.add_argument(
        "--skip-model-loading",
        action="store_true",
        help=(
            "Skip curated heavyweight model-loading cells and reuse seeded demo "
            "reports for later verification steps."
        ),
    )
    args = parser.parse_args(argv)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else Path("/tmp") / f"invarlock_notebook_smoke_{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    nb_paths = [Path(p) for p in args.notebooks] if args.notebooks else []
    if not nb_paths:
        nb_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    nb_paths = [p for p in nb_paths if p.exists()]
    if not nb_paths:
        raise SystemExit("No notebooks found.")

    print(f"Output root: {out_root}")
    print(f"Notebooks: {len(nb_paths)}")

    skip_pip = not bool(args.run_pip)
    for nb in nb_paths:
        run_dir = out_root / f"{nb.stem}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_model_loading:
            _seed_curated_demo_outputs(nb_path=nb, run_dir=run_dir)
        script_path = run_dir / f"{nb.stem}.py"
        write_script(
            nb_path=nb,
            out_py=script_path,
            skip_pip=skip_pip,
            skip_model_loading=args.skip_model_loading,
        )
        print(f"Running: {nb.name}")
        run_script(script_path=script_path, cwd=run_dir, timeout_s=int(args.timeout_s))
        print(f"OK: {nb.name}")

    print("All notebook smoke runs passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
