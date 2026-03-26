"""
InvarLock CLI Evaluate Command
=========================

Hero path: Compare & Evaluate (BYOE). Provide baseline (`--baseline`) and
subject (`--subject`) checkpoints and InvarLock will run paired windows and emit a
evaluation report. Optionally, pass `--edit-config` to run the built‑in quant_rtn demo.

Steps:
  1) Baseline (no-op edit) on baseline model
  2) Subject (no-op or provided edit config) on subject model with --baseline pairing
  3) Emit evaluation report via `invarlock report --format report`
"""

from __future__ import annotations

import builtins as _builtins
import inspect
import io
import json
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from invarlock import __version__ as INVARLOCK_VERSION

from ...core.evaluate_contract import (
    apply_edited_primary_metric_policy,
    load_validated_baseline_report,
    require_run_report_artifact,
)
from ...core.evaluate_plan import (
    build_baseline_run_config,
    build_evaluation_report_kwargs,
    build_subject_edit_run_config,
    build_subject_noop_run_config,
    default_preset_data_for_adapter,
    determine_subject_label,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
)
from ...core.evaluate_plan import (
    normalize_model_id as _normalize_model_id,
)
from ...core.exceptions import ConfigError, ValidationError
from ..adapter_auto import resolve_auto_adapter
from ..security_helpers import (
    configure_runtime_security,
    emit_runtime_manifest,
    maybe_delegate_model_command,
)

# Use the report group's programmatic entry for report generation
from .report import report_command as _report

_LAZY_RUN_IMPORT = True

PHASE_BAR_WIDTH = 67
VERBOSITY_QUIET = 0
VERBOSITY_DEFAULT = 1
VERBOSITY_VERBOSE = 2

console = Console()


def _render_banner_lines(title: str, context: str) -> list[str]:
    return [
        title,
        context,
        "-" * max(len(title), len(context)),
    ]


def _print_header_banner(
    console: Console, *, version: str, profile: str, tier: str, adapter: str
) -> None:
    title = f"INVARLOCK v{version} · Evaluation Pipeline"
    context = f"Profile: {profile} · Tier: {tier} · Adapter: {adapter}"
    for line in _render_banner_lines(title, context):
        console.print(line)


def _phase_title(index: int, total: int, title: str) -> str:
    return f"PHASE {index}/{total} · {title}"


def _print_phase_header(console: Console, title: str) -> None:
    console.print(title)
    console.print("-" * max(PHASE_BAR_WIDTH, len(title)))


def _format_ratio(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    return f"{val:.3f}"


def _resolve_verbosity(quiet: bool, verbose: bool) -> int:
    if quiet and verbose:
        console.print("--quiet and --verbose are mutually exclusive")
        raise typer.Exit(2)
    if quiet:
        return VERBOSITY_QUIET
    if verbose:
        return VERBOSITY_VERBOSE
    return VERBOSITY_DEFAULT


@contextmanager
def _override_console(module: Any, new_console: Console) -> Iterator[None]:
    original_console = getattr(module, "console", None)
    module.console = new_console
    try:
        yield
    finally:
        module.console = original_console


@contextmanager
def _suppress_child_output(enabled: bool) -> Iterator[io.StringIO | None]:
    if not enabled:
        yield None
        return
    from . import report as report_mod
    from . import run as run_mod

    buffer = io.StringIO()
    quiet_console = Console(file=buffer, force_terminal=False, color_system=None)
    with (
        _override_console(run_mod, quiet_console),
        _override_console(report_mod, quiet_console),
    ):
        yield buffer


def _print_quiet_summary(
    *,
    report_out: Path,
    source: str,
    edited: str,
    profile: str,
) -> None:
    report_path = report_out / "evaluation.report.json"
    console.print(f"INVARLOCK v{INVARLOCK_VERSION} · EVALUATE")
    console.print(f"Baseline: {source} -> Subject: {edited} · Profile: {profile}")
    if not report_path.exists():
        console.print(f"Output: {report_out}")
        return
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            evaluation_report = json.load(fh)
    except Exception:
        console.print(f"Output: {report_path}")
        return
    if not isinstance(evaluation_report, dict):
        console.print(f"Output: {report_path}")
        return
    try:
        from invarlock.reporting.render import (
            compute_console_validation_block as _console_block,
        )

        block = _console_block(evaluation_report)
        rows = block.get("rows", [])
        total = len(rows) if isinstance(rows, list) else 0
        passed = (
            sum(1 for row in rows if row.get("ok")) if isinstance(rows, list) else 0
        )
        status = "PASS" if block.get("overall_pass") else "FAIL"
    except Exception:
        total = 0
        passed = 0
        status = "UNKNOWN"
    pm_ratio = _format_ratio(
        (evaluation_report.get("primary_metric") or {}).get("ratio_vs_baseline")
    )
    gate_summary = f"{passed}/{total} passed" if total else "N/A"
    console.print(f"Status: {status} · Gates: {gate_summary}")
    if pm_ratio != "N/A":
        console.print(f"Primary metric ratio: {pm_ratio}")
    console.print(f"Output: {report_path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Preset must be a mapping")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _resolve_evaluate_tmp_dir() -> Path:
    """Return the on-disk scratch directory for `invarlock evaluate`.

    Evaluate generates merged YAML configs for baseline/subject runs so
    downstream `invarlock run` flows remain traceable. We keep these files
    under `./tmp/.evaluate` by default to avoid cluttering the working tree.
    Each invocation gets an isolated subdirectory so concurrent evaluate
    commands cannot overwrite each other's generated YAMLs.
    """

    candidate = os.environ.get("INVARLOCK_EVALUATE_TMP_DIR")
    if candidate:
        tmp_dir = Path(candidate).expanduser()
    else:
        scratch_root = Path("tmp") / ".evaluate"
        scratch_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(scratch_root))).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def evaluate_command(
    # Primary names for programmatic/test compatibility
    source: str = typer.Option(
        ..., "--source", "--baseline", help="Baseline model dir or Hub ID"
    ),
    edited: str = typer.Option(
        ..., "--edited", "--subject", help="Subject model dir or Hub ID"
    ),
    baseline_report: str | None = typer.Option(
        None,
        "--baseline-report",
        help=(
            "Reuse an existing baseline run report.json file (explicit path; skips baseline evaluation). "
            "Must include stored evaluation windows (e.g., set INVARLOCK_STORE_EVAL_WINDOWS=1)."
        ),
    ),
    adapter: str = typer.Option(
        "auto", "--adapter", help="Adapter name or 'auto' to resolve"
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device override for runs (auto|cuda|mps|cpu)",
    ),
    profile: str = typer.Option(
        "ci", "--profile", help="Profile (ci|release|ci_cpu|dev)"
    ),
    tier: str = typer.Option("balanced", "--tier", help="Tier label for context"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=(
            "Universal preset path to use (defaults to causal or masked preset"
            " based on adapter)"
        ),
    ),
    out: str = typer.Option("runs", "--out", help="Base output directory"),
    report_out: str = typer.Option(
        "reports/eval", "--report-out", help="Evaluation report output directory"
    ),
    edit_config: str | None = typer.Option(
        None, "--edit-config", help="Edit preset to apply a demo edit"
    ),
    edit_label: str | None = typer.Option(
        None,
        "--edit-label",
        help=(
            "Edit algorithm label for BYOE models. Use 'noop' for baseline, "
            "'quant_rtn' etc. for built-in edits, 'custom' for pre-edited models."
        ),
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Minimal output (suppress run/report detail)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output (include debug details)"
    ),
    banner: bool = typer.Option(
        True, "--banner/--no-banner", help="Show header banner"
    ),
    style: str = typer.Option("audit", "--style", help="Output style (audit|friendly)"),
    timing: bool = typer.Option(False, "--timing", help="Show timing summary"),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress done messages"
    ),
    mode: str = typer.Option(
        "attested",
        "--mode",
        help="Execution mode for model-loading steps (attested|local).",
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Explicitly allow outbound network access for this command.",
    ),
    allow_host_execution: bool = typer.Option(
        False,
        "--allow-host-execution",
        help="Run on the host instead of auto-delegating to the runtime container.",
    ),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Enable third-party entry-point plugin discovery for this command.",
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="Allow trust_remote_code-style model loading for this command.",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    """Evaluate two checkpoints (baseline vs subject) with pinned windows."""
    # Support programmatic calls and Typer-invoked calls uniformly
    try:
        from typer.models import OptionInfo as _TyperOptionInfo
    except Exception:  # pragma: no cover - typer internals may change
        _TyperOptionInfo = ()  # type: ignore[assignment]

    def _coerce_option(value, fallback=None):
        if isinstance(value, _TyperOptionInfo):
            return getattr(value, "default", fallback)
        return value if value is not None else fallback

    source = _coerce_option(source)
    edited = _coerce_option(edited)
    baseline_report = _coerce_option(baseline_report)
    adapter = _coerce_option(adapter, "auto")
    device = _coerce_option(device)
    profile = _coerce_option(profile, "ci")
    tier = _coerce_option(tier, "balanced")
    preset = _coerce_option(preset)
    out = _coerce_option(out, "runs")
    report_out = _coerce_option(report_out, "reports/eval")
    edit_config = _coerce_option(edit_config)
    edit_label = _coerce_option(edit_label)
    quiet = _coerce_option(quiet, False)
    verbose = _coerce_option(verbose, False)
    banner = _coerce_option(banner, True)
    style = _coerce_option(style, "audit")
    timing = bool(_coerce_option(timing, False))
    progress = bool(_coerce_option(progress, True))
    mode = str(_coerce_option(mode, "attested")).strip().lower()
    allow_network = bool(_coerce_option(allow_network, False))
    allow_host_execution = bool(_coerce_option(allow_host_execution, False))
    allow_third_party_plugins = bool(_coerce_option(allow_third_party_plugins, False))
    allow_remote_code = bool(_coerce_option(allow_remote_code, False))
    no_color = bool(_coerce_option(no_color, False))

    if mode not in {"attested", "local"}:
        raise typer.BadParameter(
            "Execution mode must be one of: attested, local.",
            param_hint="--mode",
        )
    allow_host_execution = allow_host_execution or mode == "local"

    configure_runtime_security(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
    )
    maybe_delegate_model_command()

    verbosity = _resolve_verbosity(bool(quiet), bool(verbose))

    if verbosity == VERBOSITY_QUIET:
        progress = False
        timing = False

    from invarlock.cli.output import (
        make_console,
        perf_counter,
        print_event,
        print_timing_summary,
        resolve_output_style,
        timed_step,
    )

    output_style = resolve_output_style(
        style=str(style),
        profile=str(profile),
        progress=bool(progress),
        timing=bool(timing),
        no_color=bool(no_color),
    )
    console = make_console(no_color=not output_style.color)
    timings: dict[str, float] = {}
    total_start: float | None = perf_counter() if output_style.timing else None

    def _stable_text(value: object, fallback: str = "") -> str:
        if isinstance(value, _builtins.str):
            return value
        try:
            return str(value)
        except Exception:
            return fallback

    def _info(message: str, *, tag: str = "INFO", emoji: str | None = None) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            print_event(console, tag, message, style=output_style, emoji=emoji)

    def _debug(msg: str) -> None:
        if verbosity >= VERBOSITY_VERBOSE:
            console.print(msg, markup=False)

    def _fail(message: str, *, exit_code: int = 2) -> NoReturn:
        print_event(console, "FAIL", message, style=output_style, emoji="❌")
        raise typer.Exit(exit_code)

    def _phase(index: int, total: int, title: str) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            console.print("")
            _print_phase_header(console, _phase_title(index, total, title))

    src_id = str(source)
    edt_id = str(edited)
    profile_name = _stable_text(profile, "dev")
    tier_name = _stable_text(tier, "balanced")

    # Resolve adapter when requested
    eff_adapter = adapter
    adapter_auto = False
    if str(adapter).strip().lower() in {"auto", "auto_hf"}:
        eff_adapter = resolve_auto_adapter(src_id)
        adapter_auto = True

    show_banner = bool(banner) and verbosity >= VERBOSITY_DEFAULT
    if show_banner:
        _print_header_banner(
            console,
            version=INVARLOCK_VERSION,
            profile=profile_name,
            tier=tier_name,
            adapter=str(eff_adapter),
        )
        console.print("")

    if adapter_auto:
        _debug(f"Adapter:auto -> {eff_adapter}")

    # Choose preset. If none provided and repo preset is missing (pip install
    # scenario), fall back to a minimal built-in universal preset so the
    # flag-only quick start works without cloning the repo.
    default_universal = (
        Path("configs/presets/masked_lm/wikitext2_128.yaml")
        if eff_adapter == "hf_mlm"
        else Path("configs/presets/causal_lm/wikitext2_512.yaml")
    )
    preset_path = Path(preset) if preset is not None else default_universal

    preset_data: dict[str, Any]
    if preset is None and not preset_path.exists():
        # Inline minimal preset (wikitext2 universal) for pip installs
        preset_data = default_preset_data_for_adapter(str(eff_adapter))
    else:
        if not preset_path.exists():
            print_event(
                console,
                "FAIL",
                f"Preset not found: {preset_path}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1)
        preset_data = sanitize_preset_data_for_evaluate(_load_yaml(preset_path))

    guards_order = resolve_guards_order(preset_data)

    # Create temp baseline config (no-op edit)
    # Normalize possible "hf:" prefixes for HF adapters
    norm_src_id = _normalize_model_id(src_id, eff_adapter)
    norm_edt_id = _normalize_model_id(edt_id, eff_adapter)

    baseline_cfg = build_baseline_run_config(
        preset_data,
        model_id=norm_src_id,
        adapter_name=str(eff_adapter),
        output_dir=str(Path(out) / "source"),
        profile=profile_name,
        tier=tier_name,
        guards_order=guards_order,
    )

    baseline_label = "noop"
    subject_label = determine_subject_label(
        edit_label=edit_label,
        edit_config=edit_config,
        source_model_id=norm_src_id,
        subject_model_id=norm_edt_id,
    )

    tmp_dir = _resolve_evaluate_tmp_dir()

    baseline_report_path: Path
    if baseline_report:
        _info(
            "Using provided baseline report (skipping baseline evaluation)",
            tag="EXEC",
            emoji="♻️",
        )
        try:
            baseline_report_path, _ = load_validated_baseline_report(
                Path(baseline_report),
                expected_profile=profile_name,
                expected_tier=tier_name,
                expected_adapter=str(eff_adapter),
            )
        except ValidationError as exc:
            _fail(str(getattr(exc, "message", exc)), exit_code=2)
        _debug(f"Baseline report: {baseline_report_path}")
    else:
        baseline_yaml = tmp_dir / "baseline_noop.yaml"
        _dump_yaml(baseline_yaml, baseline_cfg)

        _phase(1, 3, "BASELINE EVALUATION")
        _info("Running baseline (no-op edit)", tag="EXEC", emoji="🏁")
        _debug(f"Baseline config: {baseline_yaml}")
        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="baseline",
                    tag="EXEC",
                    message="Baseline",
                    emoji="🏁",
                ):
                    baseline_run_result = _run(
                        config=str(baseline_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "source"),
                        tier=tier_name,
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=baseline_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

        try:
            baseline_report_path = require_run_report_artifact(
                baseline_run_result,
                stage="Baseline",
            )
        except ConfigError as exc:
            _fail(str(getattr(exc, "message", exc)), exit_code=1)
        _debug(f"Baseline report: {baseline_report_path}")

    # Edited run: either no-op (Compare & Evaluate) or provided edit_config (demo edit)
    _phase(2, 3, "SUBJECT EVALUATION")
    if edit_config:
        edited_yaml = Path(edit_config)
        if not edited_yaml.exists():
            print_event(
                console,
                "FAIL",
                f"Edit config not found: {edited_yaml}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1)
        _info("Running edited (demo edit via --edit-config)", tag="EXEC", emoji="✂️")
        # Overlay subject model id/adapter and output/context onto the provided edit config
        try:
            cfg_loaded: dict[str, Any] = _load_yaml(edited_yaml)
        except Exception as exc:  # noqa: BLE001
            print_event(
                console,
                "FAIL",
                f"Failed to load edit config: {exc}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        merged_edited_cfg = build_subject_edit_run_config(
            preset_data,
            cfg_loaded,
            subject_model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
        )

        # Persist a temporary merged config for traceability
        edited_merged_yaml = tmp_dir / "edited_merged.yaml"
        _dump_yaml(edited_merged_yaml, merged_edited_cfg)
        _debug(f"Edited config (merged): {edited_merged_yaml}")

        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="✂️",
                ):
                    edited_run_result = _run(
                        config=str(edited_merged_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=str(baseline_report_path),
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label if edit_label else None,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
    else:
        edited_cfg = build_subject_noop_run_config(
            preset_data,
            model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
        )
        edited_yaml = tmp_dir / "edited_noop.yaml"
        _dump_yaml(edited_yaml, edited_cfg)
        _info("Running edited (no-op, Compare & Evaluate)", tag="EXEC", emoji="🧪")
        _debug(f"Edited config: {edited_yaml}")
        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="🧪",
                ):
                    edited_run_result = _run(
                        config=str(edited_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=str(baseline_report_path),
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

    try:
        edited_report = require_run_report_artifact(
            edited_run_result,
            stage="Edited",
        )
    except ConfigError as exc:
        _fail(str(getattr(exc, "message", exc)), exit_code=1)
    _debug(f"Edited report: {edited_report}")

    _phase(3, 3, "EVALUATION REPORT GENERATION")

    def _emit_evaluation_report() -> None:
        _info("Emitting evaluation report", tag="EXEC", emoji="📜")
        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="evaluation_report",
                    tag="EXEC",
                    message="Evaluation Report",
                    emoji="📜",
                ):
                    # Use a wall-clock perf counter here (not the output module's
                    # test-patched counter) so timing tests remain deterministic.
                    from time import perf_counter as _wall_perf_counter

                    report_start = _wall_perf_counter()
                    report_kwargs = build_evaluation_report_kwargs(
                        edited_report=str(edited_report),
                        baseline_report=str(baseline_report_path),
                        report_out=str(report_out),
                        style=output_style.name,
                        no_color=bool(no_color),
                        baseline_seconds=float(timings.get("baseline", 0.0)),
                        subject_seconds=float(timings.get("subject", 0.0)),
                        report_start=float(report_start),
                    )
                    try:
                        sig = inspect.signature(_report)
                    except (TypeError, ValueError):
                        _report(**report_kwargs)
                    else:
                        if any(
                            param.kind == inspect.Parameter.VAR_KEYWORD
                            for param in sig.parameters.values()
                        ):
                            _report(**report_kwargs)
                        else:
                            _report(
                                **{
                                    key: value
                                    for key, value in report_kwargs.items()
                                    if key in sig.parameters
                                }
                            )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
        emit_runtime_manifest(
            Path(report_out) / "evaluation.report.json",
            config_payload={
                "command": "evaluate",
                "source": source,
                "edited": edited,
                "adapter": adapter,
                "profile": profile_name,
                "tier": tier_name,
                "preset": preset,
                "out": out,
                "report_out": report_out,
                "edit_config": edit_config,
                "edit_label": edit_label,
                "allow_network": allow_network,
                "allow_remote_code": allow_remote_code,
                "allow_third_party_plugins": allow_third_party_plugins,
            },
            extra={
                "command": "evaluate",
                "profile": profile_name,
                "tier": tier_name,
            },
        )

    # CI/Release hard‑abort: fail fast when primary metric is not computable.
    try:
        prof = str(profile or "").strip().lower()
    except Exception:
        prof = ""
    if prof in {"ci", "ci_cpu", "release"}:
        try:
            with Path(edited_report).open("r", encoding="utf-8") as fh:
                edited_payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print_event(
                console,
                "FAIL",
                f"Failed to read edited report: {exc}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        outcome = apply_edited_primary_metric_policy(
            edited_payload,
            profile=profile,
        )
        if outcome.warning is not None and outcome.error is not None:
            print_event(
                console,
                "WARN",
                outcome.warning,
                style=output_style,
                emoji="⚠️",
            )
            _emit_evaluation_report()
            raise typer.Exit(int(outcome.exit_code or 1))

    _emit_evaluation_report()
    if timing:
        if total_start is not None:
            timings["total"] = max(0.0, float(perf_counter() - total_start))
        else:
            timings["total"] = (
                float(timings.get("baseline", 0.0))
                + float(timings.get("subject", 0.0))
                + float(timings.get("evaluation_report", 0.0))
            )
        print_timing_summary(
            console,
            timings,
            style=output_style,
            order=[
                ("Baseline", "baseline"),
                ("Subject", "subject"),
                ("Evaluation Report", "evaluation_report"),
                ("Total", "total"),
            ],
        )
    if verbosity == VERBOSITY_QUIET:
        _print_quiet_summary(
            report_out=Path(report_out),
            source=src_id,
            edited=edt_id,
            profile=profile_name,
        )
