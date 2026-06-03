"""Command rewriting for markdown bash block replay."""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
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


class BashBlockLike(Protocol):
    text: str


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


def _is_evaluate_command(command_tokens: list[str]) -> bool:
    return command_tokens[:1] == ["evaluate"]


def _is_verify_command(command_tokens: list[str]) -> bool:
    return command_tokens[:1] == ["verify"] or command_tokens[:2] == [
        "report",
        "verify",
    ]


def _is_model_loading_command(command_tokens: list[str]) -> bool:
    if command_tokens[:1] and command_tokens[0] in MODEL_LOADING_COMMANDS:
        return True
    return command_tokens[:2] == ["advanced", "calibrate"]


def _is_optional_environment_command(command_tokens: list[str]) -> bool:
    return command_tokens[:1] == ["doctor"]


def _should_skip_line_for_host_mode(
    stripped: str,
    *,
    host_supports_mps: Callable[[], bool] | None = None,
) -> bool:
    if (
        stripped.startswith("make runtime-image")
        or stripped.startswith("make runtime-smoke")
        or stripped.startswith("make container-default-smoke")
        or stripped.startswith("make container-front-door-smoke")
    ):
        return True
    if stripped.startswith(("docker ", "podman ")):
        return True
    supports_mps = (
        _host_supports_mps if host_supports_mps is None else host_supports_mps
    )
    if "--device mps" in stripped and not supports_mps():
        return True
    return "runtime.manifest.json" in stripped and (
        stripped.startswith("test -f ")
        or stripped.startswith("[ -f ")
        or stripped.startswith("stat ")
    )


def _host_supports_mps() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import torch
    except Exception:
        return False
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


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
    return _insert_tokens_after_command(argv, [option])


def _insert_tokens_after_command(argv: list[str], tokens: list[str]) -> list[str]:
    if argv[:1] == ["invarlock"]:
        insert_at = 2
        if len(argv) >= 3 and argv[1] == "report" and argv[2] == "verify":
            insert_at = 3
        if len(argv) >= 3 and argv[1] == "report" and argv[2] == "html":
            insert_at = 3
        return [*argv[:insert_at], *tokens, *argv[insert_at:]]
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
        return [*argv[:insert_at], *tokens, *argv[insert_at:]]
    return [*argv, *tokens]


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
        if _is_evaluate_command(command_tokens):
            argv = _strip_option_with_value(argv, "--execution-mode")
        if _is_verify_command(command_tokens):
            argv = _strip_option_with_value(argv, "--runtime-provenance")
        if command_tokens[:2] == ["report", "html"] and "--force" not in argv:
            argv = _insert_option_after_command(argv, "--force")
        return env_prefix, argv

    if _is_evaluate_command(command_tokens):
        argv = _strip_option_with_value(argv, "--execution-mode")
        if "--execution-mode" not in argv:
            if argv[:1] == ["invarlock"]:
                argv = [*argv[:2], "--execution-mode", "host", *argv[2:]]
            elif (
                len(argv) >= 3
                and argv[0] in {"python", "python3"}
                and argv[1] == "-m"
                and argv[2].startswith("invarlock")
            ):
                argv = [
                    *argv[:4],
                    "--execution-mode",
                    "host",
                    *argv[4:],
                ]
    elif _is_verify_command(command_tokens):
        argv = _strip_option_with_value(argv, "--runtime-provenance")
        if "--runtime-provenance" not in argv:
            if argv[:1] == ["invarlock"]:
                argv = [
                    *argv[:2],
                    "--runtime-provenance",
                    "host",
                    *argv[2:],
                ]
            elif (
                len(argv) >= 3
                and argv[0] in {"python", "python3"}
                and argv[1] == "-m"
                and argv[2].startswith("invarlock")
            ):
                argv = [
                    *argv[:4],
                    "--runtime-provenance",
                    "host",
                    *argv[4:],
                ]
    elif _is_model_loading_command(command_tokens):
        if "--allow-host-execution" not in argv:
            env_prefix.append(f"{HOST_EXECUTION_ENV}=1")
    if command_tokens[:2] == ["report", "html"] and "--force" not in argv:
        argv = _insert_option_after_command(argv, "--force")
    return env_prefix, argv


def _sanitize_script(
    block: BashBlockLike,
    *,
    execution_mode: str = "container",
    skip_model_loading: bool = False,
    host_supports_mps: Callable[[], bool] | None = None,
    root: Path = ROOT,
) -> str:
    rendered: list[str] = []
    workspace_python = root / ".venv" / "bin" / "python"
    selected_python = (
        str(workspace_python) if workspace_python.is_file() else sys.executable
    )
    py = shlex.quote(selected_python)
    skipping_continuation = False
    block_lines = block.text.splitlines()
    for line_index, raw in enumerate(block_lines):
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
        continuation_parts = [stripped]
        if stripped.endswith("\\"):
            probe_index = line_index + 1
            while probe_index < len(block_lines):
                continuation = block_lines[probe_index].strip()
                continuation_parts.append(continuation)
                if not continuation.endswith("\\"):
                    break
                probe_index += 1
        if execution_mode == "host" and any(
            _should_skip_line_for_host_mode(part, host_supports_mps=host_supports_mps)
            for part in continuation_parts
            if part
        ):
            rendered.append(f"echo '[skip-host] {stripped}'")
            skipping_continuation = stripped.endswith("\\")
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
            if skip_model_loading and (
                _is_model_loading_command(command_tokens)
                or _is_optional_environment_command(command_tokens)
            ):
                rendered.append(f"echo '[skip-model-loading] {stripped}'")
                skipping_continuation = has_trailing_backslash
                continue
            env_prefix, argv = _rewrite_invarlock_tokens(
                env_prefix=env_prefix,
                argv=argv,
                execution_mode=execution_mode,
            )
            if (
                skip_model_loading
                and _is_verify_command(command_tokens)
                and "--assurance" not in argv
            ):
                argv = _insert_tokens_after_command(argv, ["--assurance", "off"])
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
            elif argv[:1] and argv[0] in {"python", "python3"}:
                rebuilt = env_prefix + [py, *argv[1:]]
                line = indent + shlex.join(rebuilt)
        if has_trailing_backslash and line != raw:
            line = line.rstrip() + " \\"
        rendered.append(line)
    sanitized = "\n".join(rendered).strip() + "\n"
    if execution_mode == "host" and not skip_model_loading:
        sanitized = _rewrite_live_smoke_script_text(sanitized)
    return sanitized


__all__ = [
    "DEFAULT_EVALUATE_SMOKE_PRESET",
    "HOST_EXECUTION_ENV",
    "MODEL_LOADING_COMMANDS",
    "SMOKE_MODEL_ID_MAP",
    "SMOKE_PATH_MAP",
    "SMOKE_SCRIPT_REWRITES",
    "_command_tokens",
    "_host_supports_mps",
    "_insert_option_after_command",
    "_insert_tokens_after_command",
    "_is_evaluate_command",
    "_is_model_loading_command",
    "_is_optional_environment_command",
    "_is_verify_command",
    "_rewrite_invarlock_tokens",
    "_rewrite_live_smoke_script_text",
    "_rewrite_model_loading_tokens_for_live_smoke",
    "_sanitize_script",
    "_should_skip_line_for_host_mode",
    "_split_env_prefix",
]
