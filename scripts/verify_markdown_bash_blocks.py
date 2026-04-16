#!/usr/bin/env python3
"""Live-run concrete bash code blocks from Markdown docs.

Blocks are executed in file-scoped temporary workspaces copied from the current
checkout so workflows like:

- build image
- evaluate
- verify
- render report

can run in order without mutating the developer's working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
EXECUTION_MODES = ("container", "host")
HOST_EXECUTION_ENV = "INVARLOCK_ALLOW_HOST_EXECUTION"
MODEL_LOADING_COMMANDS = {"evaluate", "run", "calibrate"}
DEFAULT_EVALUATE_SMOKE_PRESET = "configs/presets/causal_lm/gpt2_smoke_128.yaml"
SMOKE_MODEL_ID_MAP = {
    "distilgpt2": "sshleifer/tiny-gpt2",
    "gpt2": "sshleifer/tiny-gpt2",
}
SMOKE_PATH_MAP = {
    "configs/calibration/null_sweep_ci.yaml": "configs/calibration/null_sweep_smoke.yaml",
    "configs/calibration/rmt_ve_sweep_ci.yaml": "configs/calibration/rmt_ve_sweep_smoke.yaml",
    "configs/presets/causal_lm/wikitext2_512.yaml": DEFAULT_EVALUATE_SMOKE_PRESET,
}
SMOKE_SCRIPT_REWRITES = (
    (re.compile(r"(?m)(--baseline\s+)distilgpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--baseline\s+)gpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--subject\s+)distilgpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--subject\s+)gpt2\b"), r"\1sshleifer/tiny-gpt2"),
    (re.compile(r"(?m)(--profile\s+)(?:ci|release)\b"), r"\1dev"),
    (re.compile(r"(?m)(--n-seeds\s+)\d+\b"), r"\g<1>1"),
    (
        re.compile(r"configs/presets/causal_lm/wikitext2_512\.yaml"),
        DEFAULT_EVALUATE_SMOKE_PRESET,
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
DEMO_EVALUATION_REPORT_FIXTURE = (
    ROOT / "tests" / "artifacts" / "golden_runs" / "gpt2" / "evaluation.report.json"
)
DEMO_RUNTIME_MANIFEST_FIXTURE = (
    ROOT / "tests" / "fixtures" / "runtime_attestation" / "runtime.manifest.json"
)

EXCLUDE_TOP_LEVEL_DIRS = {
    "build",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-release",
    ".evaluate_tmp",
    "node_modules",
    "reports",
    "runs",
    "site",
    "tmp",
    "venv",
}

ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<[^>]+>")
RUN_ID_PLACEHOLDER_PATTERN = re.compile(r"\bruns/\d{8}_\d{6}\b")
INVARLOCK_COMMAND_PATTERN = re.compile(
    r"^(?:[A-Z_][A-Z0-9_]*=[^\s]+\s+)*(?:invarlock\s+|python\s+-m\s+invarlock(?:\.[^\s]+)?\s+).*$"
)

SKIP_TOKENS = (
    "$CONFIG_FILE",
    "...",
    "…",
    "config.yaml",
    "custom_format",
    "make dev-install",
    "my_plugin",
    "my_config.yaml",
    "run_pack.sh",
    "run_suite.sh",
    "runs/latest",
    "/path/to/",
    "/absolute/path/to/",
    "<BASELINE_MODEL>",
    "<SUBJECT_MODEL>",
    "<model_or_id>",
    "<edited_model_or_dir>",
    "<source>",
    "<edited>",
    "<ts>",
    "<hf_dir_or_id>",
    "<report.json>",
    "<out.html>",
    "<edited_report.json>",
    "<baseline_report.json>",
)


@dataclass(frozen=True)
class BashBlock:
    file: str
    line: int
    block_index: int
    text: str


def _strip_prompt(s: str) -> str:
    s = s.lstrip()
    return s[2:] if s.startswith("$ ") else s


def _should_skip_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if ANGLE_PLACEHOLDER_PATTERN.search(stripped):
        return True
    if RUN_ID_PLACEHOLDER_PATTERN.search(stripped):
        return True
    return any(token in stripped for token in SKIP_TOKENS)


def _contains_invarlock_command(text: str) -> bool:
    for raw in text.splitlines():
        line = _strip_prompt(raw.strip())
        if not line or line.startswith("#"):
            continue
        if INVARLOCK_COMMAND_PATTERN.match(line):
            return True
    return False


def extract_bash_blocks(paths: list[Path]) -> list[BashBlock]:
    blocks: list[BashBlock] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        in_fence = False
        block_lines: list[str] = []
        start_line = 0
        language = ""
        block_index = 0
        for line_no, line in enumerate(lines, start=1):
            if line.startswith("```"):
                if in_fence:
                    if language == "bash":
                        block_text = "\n".join(block_lines).strip()
                        if block_text and _contains_invarlock_command(block_text):
                            block_index += 1
                            blocks.append(
                                BashBlock(
                                    file=str(path),
                                    line=start_line,
                                    block_index=block_index,
                                    text=block_text,
                                )
                            )
                    in_fence = False
                    block_lines = []
                    language = ""
                else:
                    in_fence = True
                    start_line = line_no + 1
                    info = line[3:].strip().split()
                    language = info[0] if info else ""
                continue
            if in_fence:
                block_lines.append(line)
    return blocks


def iter_markdown_files(root: Path, *, paths: list[str] | None = None) -> list[Path]:
    if paths:
        candidates: set[Path] = set()
        for item in paths:
            path = (Path(item) if Path(item).is_absolute() else (root / item)).resolve()
            if path.is_dir():
                candidates.update(
                    p.resolve() for p in path.rglob("*.md") if p.is_file()
                )
            elif path.is_file() and path.suffix.lower() == ".md":
                candidates.add(path)
        return sorted(candidates)

    md_files: list[Path] = []
    for path in root.glob("**/*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and (
            rel_parts[0].startswith(".") or rel_parts[0] in EXCLUDE_TOP_LEVEL_DIRS
        ):
            continue
        md_files.append(path)
    return sorted(md_files, key=lambda p: str(p))


def _ignore_copytree(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in EXCLUDE_TOP_LEVEL_DIRS:
            ignored.add(name)
    return ignored


def _prepare_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(ROOT, workspace, ignore=_ignore_copytree)


def _split_env_prefix(tokens: list[str]) -> tuple[list[str], list[str]]:
    env_prefix: list[str] = []
    idx = 0
    for token in tokens:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
            env_prefix.append(token)
            idx += 1
            continue
        break
    return env_prefix, tokens[idx:]


def _command_tokens(argv: list[str]) -> list[str]:
    if argv[:1] == ["invarlock"]:
        return argv[1:]
    if (
        len(argv) >= 3
        and argv[0] in {"python", "python3"}
        and argv[1] == "-m"
        and argv[2].startswith("invarlock")
    ):
        return argv[3:]
    return []


def _is_assurance_command(command_tokens: list[str]) -> bool:
    return command_tokens[:1] in (["evaluate"], ["verify"]) or command_tokens[:2] == [
        "report",
        "verify",
    ]


def _is_model_loading_command(command_tokens: list[str]) -> bool:
    if command_tokens[:1] and command_tokens[0] in MODEL_LOADING_COMMANDS:
        return True
    return command_tokens[:2] == ["advanced", "calibrate"]


def _should_skip_line_for_host_mode(stripped: str) -> bool:
    if stripped.startswith("make runtime-image") or stripped.startswith(
        "make runtime-smoke"
    ):
        return True
    if stripped.startswith(("docker ", "podman ")):
        return True
    return "runtime.manifest.json" in stripped and (
        stripped.startswith("test -f ")
        or stripped.startswith("[ -f ")
        or stripped.startswith("stat ")
    )


def _rewrite_model_loading_tokens_for_live_smoke(argv: list[str]) -> list[str]:
    if any(flag in argv for flag in ("--help", "-h")):
        return argv

    rewritten: list[str] = []
    saw_baseline_report = False
    saw_profile = False
    saw_preset = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--baseline-report":
            saw_baseline_report = True
        if token == "--profile":
            saw_profile = True
        if token == "--preset":
            saw_preset = True
        if token in {"--baseline", "--subject"} and i + 1 < len(argv):
            rewritten.extend([token, SMOKE_MODEL_ID_MAP.get(argv[i + 1], argv[i + 1])])
            i += 2
            continue
        if token == "--profile" and i + 1 < len(argv):
            profile = argv[i + 1]
            rewritten.extend(
                [token, "dev" if profile in {"ci", "release"} else profile]
            )
            i += 2
            continue
        if token == "--n-seeds" and i + 1 < len(argv):
            rewritten.extend([token, "1"])
            i += 2
            continue
        if token in {"--preset", "--config"} and i + 1 < len(argv):
            rewritten.extend([token, SMOKE_PATH_MAP.get(argv[i + 1], argv[i + 1])])
            i += 2
            continue
        rewritten.append(token)
        i += 1

    command_tokens = _command_tokens(rewritten)
    if rewritten[:1] == ["invarlock"]:
        insert_at = 4 if command_tokens[:2] == ["advanced", "calibrate"] else 2
    elif (
        len(rewritten) >= 3
        and rewritten[0] in {"python", "python3"}
        and rewritten[1] == "-m"
        and rewritten[2].startswith("invarlock")
    ):
        insert_at = 6 if command_tokens[:2] == ["advanced", "calibrate"] else 4
    else:
        return rewritten

    inserts: list[str] = []
    if not saw_profile:
        inserts.extend(["--profile", "dev"])
    if (
        command_tokens[:1] == ["evaluate"]
        and not saw_preset
        and not saw_baseline_report
    ):
        inserts.extend(["--preset", DEFAULT_EVALUATE_SMOKE_PRESET])

    if not inserts:
        return rewritten
    return [*rewritten[:insert_at], *inserts, *rewritten[insert_at:]]


def _rewrite_live_smoke_script_text(text: str) -> str:
    rewritten = text
    for pattern, replacement in SMOKE_SCRIPT_REWRITES:
        rewritten = pattern.sub(replacement, rewritten)
    return rewritten


def _insert_option_after_command(argv: list[str], option: str) -> list[str]:
    if argv[:1] == ["invarlock"]:
        insert_at = 2
        if len(argv) >= 3 and argv[1] == "report" and argv[2] == "verify":
            insert_at = 3
        if len(argv) >= 3 and argv[1] == "report" and argv[2] == "html":
            insert_at = 3
        return [*argv[:insert_at], option, *argv[insert_at:]]
    if (
        len(argv) >= 3
        and argv[0] in {"python", "python3"}
        and argv[1] == "-m"
        and argv[2].startswith("invarlock")
    ):
        insert_at = 4
        if len(argv) >= 5 and argv[3] == "report" and argv[4] == "verify":
            insert_at = 5
        if len(argv) >= 5 and argv[3] == "report" and argv[4] == "html":
            insert_at = 5
        return [*argv[:insert_at], option, *argv[insert_at:]]
    return [*argv, option]


def _rewrite_invarlock_tokens(
    *,
    env_prefix: list[str],
    argv: list[str],
    execution_mode: str,
) -> tuple[list[str], list[str]]:
    command_tokens = _command_tokens(argv)
    if not command_tokens:
        return env_prefix, argv

    env_prefix = [
        token for token in env_prefix if not token.startswith(f"{HOST_EXECUTION_ENV}=")
    ]

    if execution_mode == "host" and _is_model_loading_command(command_tokens):
        argv = _rewrite_model_loading_tokens_for_live_smoke(argv)
        command_tokens = _command_tokens(argv)

    def _strip_option_with_value(
        tokens: list[str],
        option: str,
    ) -> list[str]:
        rewritten: list[str] = []
        skip_next = False
        for idx, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if token == option and idx + 1 < len(tokens):
                skip_next = True
                continue
            rewritten.append(token)
        return rewritten

    if execution_mode == "container":
        argv = [token for token in argv if token != "--allow-host-execution"]
        if _is_assurance_command(command_tokens):
            argv = _strip_option_with_value(argv, "--assurance")
        if command_tokens[:2] == ["report", "html"] and "--force" not in argv:
            argv = _insert_option_after_command(argv, "--force")
        return env_prefix, argv

    if _is_assurance_command(command_tokens):
        argv = _strip_option_with_value(argv, "--assurance")
        if "--assurance" not in argv:
            if argv[:1] == ["invarlock"]:
                argv = [*argv[:2], "--assurance", "trusted-local", *argv[2:]]
            elif (
                len(argv) >= 3
                and argv[0] in {"python", "python3"}
                and argv[1] == "-m"
                and argv[2].startswith("invarlock")
            ):
                argv = [*argv[:4], "--assurance", "trusted-local", *argv[4:]]
    elif _is_model_loading_command(command_tokens):
        if "--allow-host-execution" not in argv:
            env_prefix.append(f"{HOST_EXECUTION_ENV}=1")
    if command_tokens[:2] == ["report", "html"] and "--force" not in argv:
        argv = _insert_option_after_command(argv, "--force")
    return env_prefix, argv


def _sanitize_script(
    block: BashBlock,
    *,
    execution_mode: str = "container",
    skip_model_loading: bool = False,
) -> str:
    rendered: list[str] = []
    py = shlex.quote(sys.executable)
    skipping_continuation = False
    for raw in block.text.splitlines():
        stripped = raw.strip()
        if skipping_continuation:
            skipping_continuation = stripped.endswith("\\")
            continue
        if not stripped:
            rendered.append("")
            continue
        if stripped.startswith("#"):
            rendered.append(raw)
            continue
        if execution_mode == "host" and _should_skip_line_for_host_mode(stripped):
            rendered.append(f"echo '[skip-host] {stripped}'")
            continue
        tokens = stripped.split()
        if len(tokens) >= 2 and tokens[0] == "pip" and tokens[1] == "install":
            rendered.append(f"echo '[skip] {stripped}'")
            continue
        if (
            len(tokens) >= 4
            and tokens[0] in {"python", "python3"}
            and tokens[1] == "-m"
            and tokens[2] == "pip"
            and tokens[3] == "install"
        ):
            rendered.append(f"echo '[skip] {stripped}'")
            continue
        line = raw
        lstripped = raw.lstrip()
        indent = raw[: len(raw) - len(lstripped)]
        has_trailing_backslash = lstripped.rstrip().endswith("\\")
        parse_target = lstripped.rstrip()
        if has_trailing_backslash:
            parse_target = parse_target[:-1].rstrip()
        try:
            parsed_tokens = shlex.split(parse_target, posix=True)
        except ValueError:
            parsed_tokens = []
        if parsed_tokens:
            env_prefix, argv = _split_env_prefix(parsed_tokens)
            command_tokens = _command_tokens(argv)
            if skip_model_loading and _is_model_loading_command(command_tokens):
                rendered.append(f"echo '[skip-model-loading] {stripped}'")
                skipping_continuation = has_trailing_backslash
                continue
            env_prefix, argv = _rewrite_invarlock_tokens(
                env_prefix=env_prefix,
                argv=argv,
                execution_mode=execution_mode,
            )
            if argv[:1] == ["invarlock"]:
                rebuilt = env_prefix + [py, "-m", "invarlock", *argv[1:]]
                line = indent + shlex.join(rebuilt)
            elif (
                len(argv) >= 3
                and argv[0] in {"python", "python3"}
                and argv[1] == "-m"
                and argv[2].startswith("invarlock")
            ):
                rebuilt = env_prefix + [py, "-m", argv[2], *argv[3:]]
                line = indent + shlex.join(rebuilt)
        if has_trailing_backslash and line != raw:
            line = line.rstrip() + " \\"
        rendered.append(line)
    sanitized = "\n".join(rendered).strip() + "\n"
    if execution_mode == "host" and not skip_model_loading:
        sanitized = _rewrite_live_smoke_script_text(sanitized)
    return sanitized


def _default_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(workspace / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_demo_evaluation_report(
    run_report: dict[str, object],
    baseline_report: dict[str, object],
) -> dict[str, object] | None:
    src_root = ROOT / "src"
    if not (src_root / "invarlock").is_dir():
        return None
    src_path = str(src_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from invarlock.reporting.report_make import make_report

    evaluation_report = make_report(run_report, baseline_report)
    validation = evaluation_report.get("validation")
    if isinstance(validation, dict):
        validation["primary_metric_acceptable"] = True
    evaluation_report["resolved_policy"] = {
        "metrics": {
            "pm_ratio": {
                "ratio_limit_base": 1.1,
                "min_tokens": 1,
                "min_token_fraction": 0.0,
                "hysteresis_ratio": 0.0,
            }
        }
    }
    return evaluation_report


def _seed_demo_inputs(workspace: Path) -> None:
    evaluation_targets = (
        workspace / "reports" / "eval" / "evaluation.report.json",
        workspace / "report_bundle" / "evaluation.report.json",
        workspace / "reports" / "baseline_calib" / "evaluation.report.json",
        workspace / "reports" / "baseline_cpu" / "evaluation.report.json",
        workspace / "reports" / "baseline_mps" / "evaluation.report.json",
    )
    manifest_targets = (
        workspace / "reports" / "eval" / "runtime.manifest.json",
        workspace / "report_bundle" / "runtime.manifest.json",
    )

    if DEMO_RUNTIME_MANIFEST_FIXTURE.is_file():
        for target in manifest_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEMO_RUNTIME_MANIFEST_FIXTURE, target)

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
            "paired_delta_summary": {"mean": 0.0},
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
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.31],
                "token_counts": [100, 100],
            }
        },
    }
    baseline_report = {
        "run_id": "docs-demo-base",
        "model_id": "docs-demo-model",
        "meta": {"seed": 0, "model_id": "docs-demo-model"},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            }
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
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
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
    for target in (
        workspace / "runs" / "baseline" / "report.json",
        workspace / "runs" / "source" / "report.json",
        workspace / "runs" / "baseline_calib" / "source" / "demo" / "report.json",
    ):
        _write_json(target, baseline_report)
    _write_json(workspace / "runs" / "subject" / "report.json", run_report)

    evaluation_report = _build_demo_evaluation_report(run_report, baseline_report)
    if evaluation_report is not None:
        target_payloads = {
            workspace / "reports" / "baseline_cpu" / "evaluation.report.json": {
                **evaluation_report,
                "meta": {**evaluation_report.get("meta", {}), "device": "cpu"},
            },
            workspace / "reports" / "baseline_mps" / "evaluation.report.json": {
                **evaluation_report,
                "meta": {**evaluation_report.get("meta", {}), "device": "mps"},
            },
        }
        for target in evaluation_targets:
            _write_json(target, target_payloads.get(target, evaluation_report))
    elif DEMO_EVALUATION_REPORT_FIXTURE.is_file():
        for target in evaluation_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEMO_EVALUATION_REPORT_FIXTURE, target)

    _write_json(
        workspace / "resolved_policy.json",
        {"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )
    _write_json(
        workspace / "overrides.json",
        [{"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}],
    )
    _write_json(
        workspace / "compatibility.json",
        {"support_tiers": ["published_basis"]},
    )


def run_blocks(
    blocks: list[BashBlock],
    *,
    output_root: Path,
    execution_mode: str = "container",
    skip_model_loading: bool = False,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.jsonl"
    workspace_root = output_root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    grouped: dict[str, list[BashBlock]] = {}
    for block in blocks:
        grouped.setdefault(block.file, []).append(block)

    with results_path.open("w", encoding="utf-8") as out:
        for file_index, (file_path, file_blocks) in enumerate(
            sorted(grouped.items()), start=1
        ):
            workspace = (
                workspace_root / f"{file_index:03d}_{Path(file_path).stem}_{run_stamp}"
            )
            _prepare_workspace(workspace)
            _seed_demo_inputs(workspace)
            env = _default_env(workspace)
            for block in file_blocks:
                block_id = f"{file_index:03d}-{block.block_index:02d}"
                script_path = workspace / f".docs_live_{block_id}.sh"
                log_dir = output_root / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{block_id}.log"
                if _should_skip_block(block.text):
                    record = {
                        "id": block_id,
                        "file": block.file,
                        "line": block.line,
                        "status": "skipped-template",
                        "log_path": str(log_path),
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    continue

                script_path.write_text(
                    _sanitize_script(
                        block,
                        execution_mode=execution_mode,
                        skip_model_loading=skip_model_loading,
                    ),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    ["bash", "-euo", "pipefail", str(script_path.name)],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log_path.write_text(
                    (completed.stdout or "")
                    + ("\n" if completed.stdout and completed.stderr else "")
                    + (completed.stderr or ""),
                    encoding="utf-8",
                )
                record = {
                    "id": block_id,
                    "file": block.file,
                    "line": block.line,
                    "execution_mode": execution_mode,
                    "status": "ok" if completed.returncode == 0 else "failed",
                    "exit_code": int(completed.returncode),
                    "log_path": str(log_path),
                    "stdout": (completed.stdout or "")[-4000:],
                    "stderr": (completed.stderr or "")[-4000:],
                }
                out.write(json.dumps(record) + "\n")
                out.flush()

    failures = 0
    for raw in results_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("status") == "failed":
            failures += 1
    print(f"Verified {len(blocks)} bash block(s) → {results_path}")
    if failures:
        print(f"Markdown bash block failures: {failures}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Markdown files or directories to scan (default: repo-wide).",
    )
    parser.add_argument(
        "--output-root",
        default=str(TMP / "markdown_live"),
        help="Output directory for logs, workspaces, and result JSONL.",
    )
    parser.add_argument(
        "--execution-mode",
        default="container",
        choices=EXECUTION_MODES,
        help=(
            "Replay markdown commands as secure-default container commands or "
            "as explicit trusted-host commands."
        ),
    )
    parser.add_argument(
        "--skip-model-loading",
        action="store_true",
        help=(
            "Skip model-loading commands (`evaluate`, `run`, `calibrate`) while "
            "still replaying downstream verify/report steps against seeded demo data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    md_files = iter_markdown_files(ROOT, paths=args.paths)
    blocks = extract_bash_blocks(md_files)
    return run_blocks(
        blocks,
        output_root=Path(args.output_root).expanduser().resolve(),
        execution_mode=args.execution_mode,
        skip_model_loading=args.skip_model_loading,
    )


if __name__ == "__main__":
    raise SystemExit(main())
