"""
InvarLock CLI Certify Command
=========================

Hero path: Compare & Certify (BYOE). Provide baseline (`--baseline`) and
subject (`--subject`) checkpoints and InvarLock will run paired windows and emit a
certificate. Optionally, pass `--edit-config` to run the built‑in quant_rtn demo.

Steps:
  1) Baseline (no-op edit) on baseline model
  2) Subject (no-op or provided edit config) on subject model with --baseline pairing
  3) Emit certificate via `invarlock report --format cert`
"""

from __future__ import annotations

import inspect
import io
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from invarlock import __version__ as INVARLOCK_VERSION

from ...core.exceptions import MetricsError
from ..adapter_auto import resolve_auto_adapter
from ..config import _deep_merge as _merge  # reuse helper

# Use the report group's programmatic entry for report generation
from .report import report_command as _report
from .run import _resolve_exit_code as _resolve_exit_code

_LAZY_RUN_IMPORT = True

PHASE_BAR_WIDTH = 67
VERBOSITY_QUIET = 0
VERBOSITY_DEFAULT = 1
VERBOSITY_VERBOSE = 2

console = Console()


def _render_banner_lines(title: str, context: str) -> list[str]:
    width = max(len(title), len(context))
    border = "─" * (width + 2)
    return [
        f"┌{border}┐",
        f"│ {title.ljust(width)} │",
        f"│ {context.ljust(width)} │",
        f"└{border}┘",
    ]


def _print_header_banner(
    console: Console, *, version: str, profile: str, tier: str, adapter: str
) -> None:
    title = f"INVARLOCK v{version} · Certification Pipeline"
    context = f"Profile: {profile} · Tier: {tier} · Adapter: {adapter}"
    for line in _render_banner_lines(title, context):
        console.print(line)


def _phase_title(index: int, total: int, title: str) -> str:
    return f"PHASE {index}/{total} · {title}"


def _print_phase_header(console: Console, title: str) -> None:
    bar_width = max(PHASE_BAR_WIDTH, len(title))
    bar = "═" * bar_width
    console.print(bar)
    console.print(title)
    console.print(bar)


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
    cert_out: Path,
    source: str,
    edited: str,
    profile: str,
) -> None:
    cert_path = cert_out / "evaluation.cert.json"
    console.print(f"INVARLOCK v{INVARLOCK_VERSION} · CERTIFY")
    console.print(f"Baseline: {source} -> Subject: {edited} · Profile: {profile}")
    if not cert_path.exists():
        console.print(f"Output: {cert_out}")
        return
    try:
        with cert_path.open("r", encoding="utf-8") as fh:
            certificate = json.load(fh)
    except Exception:
        console.print(f"Output: {cert_path}")
        return
    if not isinstance(certificate, dict):
        console.print(f"Output: {cert_path}")
        return
    try:
        from invarlock.reporting.render import (
            compute_console_validation_block as _console_block,
        )

        block = _console_block(certificate)
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
        (certificate.get("primary_metric") or {}).get("ratio_vs_baseline")
    )
    gate_summary = f"{passed}/{total} passed" if total else "N/A"
    console.print(f"Status: {status} · Gates: {gate_summary}")
    if pm_ratio != "N/A":
        console.print(f"Primary metric ratio: {pm_ratio}")
    console.print(f"Output: {cert_path}")


def _latest_run_report(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    candidates = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not candidates:
        return None
    latest = candidates[-1]
    for f in [latest / "report.json", latest / f"{latest.name}.json"]:
        if f.exists():
            return f
    # Fallback: first JSON in the directory
    jsons = list(latest.glob("*.json"))
    return jsons[0] if jsons else None


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


def _normalize_model_id(model_id: str, adapter_name: str) -> str:
    """Normalize model identifiers for adapters.

    - Accepts optional "hf:" prefix for Hugging Face repo IDs and strips it
      before passing to transformers APIs.
    """
    mid = str(model_id or "").strip()
    try:
        if str(adapter_name).startswith("hf_") and mid.startswith("hf:"):
            return mid.split(":", 1)[1]
    except Exception:
        pass
    return mid


def certify_command(
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
            "Reuse an existing baseline run report.json (skips baseline evaluation). "
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
    cert_out: str = typer.Option(
        "reports/cert", "--cert-out", help="Certificate output directory"
    ),
    edit_config: str | None = typer.Option(
        None, "--edit-config", help="Edit preset to apply a demo edit (quant_rtn)"
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
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    """Certify two checkpoints (baseline vs subject) with pinned windows."""
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
    cert_out = _coerce_option(cert_out, "reports/cert")
    edit_config = _coerce_option(edit_config)
    edit_label = _coerce_option(edit_label)
    quiet = _coerce_option(quiet, False)
    verbose = _coerce_option(verbose, False)
    banner = _coerce_option(banner, True)
    style = _coerce_option(style, "audit")
    timing = bool(_coerce_option(timing, False))
    progress = bool(_coerce_option(progress, True))
    no_color = bool(_coerce_option(no_color, False))

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
            profile=profile,
            tier=tier,
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
        preset_data = {
            "dataset": {
                "provider": "wikitext2",
                "split": "validation",
                "seq_len": 512,
                "stride": 512,
                "preview_n": 64,
                "final_n": 64,
                "seed": 42,
            }
        }
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
        preset_data = _load_yaml(preset_path)
        # Do not hard-code device from presets in auto-generated certify configs;
        # allow device resolution to pick CUDA/MPS/CPU via 'auto' or CLI overrides.
        model_block = preset_data.get("model")
        if isinstance(model_block, dict) and "device" in model_block:
            model_block = dict(model_block)
            model_block.pop("device", None)
            preset_data["model"] = model_block

    default_guards_order = ["invariants", "spectral", "rmt", "variance", "invariants"]
    guards_order = None
    preset_guards = preset_data.get("guards")
    if isinstance(preset_guards, dict):
        preset_order = preset_guards.get("order")
        if (
            isinstance(preset_order, list)
            and preset_order
            and all(isinstance(item, str) for item in preset_order)
        ):
            guards_order = list(preset_order)
    if guards_order is None:
        guards_order = list(default_guards_order)

    def _load_and_validate_baseline_report(
        report_path: Path,
        *,
        expected_profile: str,
        expected_tier: str,
        expected_adapter: str,
    ) -> Path:
        candidate = Path(report_path).expanduser()
        if not candidate.exists():
            _fail(f"Baseline report not found: {candidate}")
        resolved_report: Path | None = None
        if candidate.is_dir():
            direct = candidate / "report.json"
            if direct.is_file():
                resolved_report = direct
            else:
                resolved_report = _latest_run_report(candidate)
        elif candidate.is_file():
            resolved_report = candidate
        if resolved_report is None or not resolved_report.is_file():
            _fail(f"Baseline report not found: {candidate}")
        resolved_report = resolved_report.resolve()
        try:
            with resolved_report.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            _fail(f"Baseline report is not valid JSON: {resolved_report} ({exc})")
        if not isinstance(payload, dict):
            _fail(f"Baseline report must be a JSON object: {resolved_report}")

        edit_block = payload.get("edit")
        edit_name = edit_block.get("name") if isinstance(edit_block, dict) else None
        if edit_name != "noop":
            _fail(
                "Baseline report must be a no-op run (edit.name == 'noop'). "
                f"Got edit.name={edit_name!r} in {resolved_report}"
            )

        meta = payload.get("meta")
        if isinstance(meta, dict):
            baseline_adapter = meta.get("adapter")
            if (
                isinstance(baseline_adapter, str)
                and baseline_adapter != expected_adapter
            ):
                _fail(
                    "Baseline report adapter mismatch. "
                    f"Expected {expected_adapter!r}, got {baseline_adapter!r} in {resolved_report}"
                )

        context = payload.get("context")
        if isinstance(context, dict):
            baseline_profile = context.get("profile")
            if (
                isinstance(baseline_profile, str)
                and baseline_profile.strip().lower() != expected_profile.strip().lower()
            ):
                _fail(
                    "Baseline report profile mismatch. "
                    f"Expected {expected_profile!r}, got {baseline_profile!r} in {resolved_report}"
                )
            auto_ctx = context.get("auto")
            if isinstance(auto_ctx, dict):
                baseline_tier = auto_ctx.get("tier")
                if isinstance(baseline_tier, str) and baseline_tier != expected_tier:
                    _fail(
                        "Baseline report tier mismatch. "
                        f"Expected {expected_tier!r}, got {baseline_tier!r} in {resolved_report}"
                    )

        eval_windows = payload.get("evaluation_windows")
        if not isinstance(eval_windows, dict):
            _fail(
                "Baseline report missing evaluation window payloads. "
                "Re-run baseline with INVARLOCK_STORE_EVAL_WINDOWS=1."
            )

        for phase_name in ("preview", "final"):
            phase = eval_windows.get(phase_name)
            if not isinstance(phase, dict):
                _fail(
                    f"Baseline report missing evaluation_windows.{phase_name} payloads. "
                    "Re-run baseline with INVARLOCK_STORE_EVAL_WINDOWS=1."
                )
            window_ids = phase.get("window_ids")
            input_ids = phase.get("input_ids")
            if not isinstance(window_ids, list) or not window_ids:
                _fail(
                    f"Baseline report missing evaluation_windows.{phase_name}.window_ids."
                )
            if not isinstance(input_ids, list) or not input_ids:
                _fail(
                    f"Baseline report missing evaluation_windows.{phase_name}.input_ids."
                )
            if len(input_ids) != len(window_ids):
                _fail(
                    "Baseline report has inconsistent evaluation window payloads "
                    f"for {phase_name}: input_ids={len(input_ids)} window_ids={len(window_ids)}."
                )

        return resolved_report

    # Create temp baseline config (no-op edit)
    # Normalize possible "hf:" prefixes for HF adapters
    norm_src_id = _normalize_model_id(src_id, eff_adapter)
    norm_edt_id = _normalize_model_id(edt_id, eff_adapter)

    baseline_cfg = _merge(
        preset_data,
        {
            "model": {
                "id": norm_src_id,
                "adapter": eff_adapter,
            },
            "edit": {"name": "noop", "plan": {}},
            "eval": {},
            "guards": {"order": guards_order},
            "output": {"dir": str(Path(out) / "source")},
            "context": {"profile": profile, "tier": tier},
        },
    )

    baseline_label = "noop"
    subject_label: str | None = None
    if edit_label:
        subject_label = edit_label
    elif not edit_config:
        subject_label = "custom" if norm_src_id != norm_edt_id else "noop"

    tmp_dir = Path(".certify_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    baseline_report_path: Path
    if baseline_report:
        _info(
            "Using provided baseline report (skipping baseline evaluation)",
            tag="EXEC",
            emoji="♻️",
        )
        baseline_report_path = _load_and_validate_baseline_report(
            Path(baseline_report),
            expected_profile=profile,
            expected_tier=tier,
            expected_adapter=str(eff_adapter),
        )
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
                    _run(
                        config=str(baseline_yaml),
                        profile=profile,
                        out=str(Path(out) / "source"),
                        tier=tier,
                        device=device,
                        edit_label=baseline_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

        baseline_report_path_candidate = _latest_run_report(Path(out) / "source")
        if not baseline_report_path_candidate:
            _fail("Could not locate baseline report after run", exit_code=1)
        baseline_report_path = baseline_report_path_candidate
        _debug(f"Baseline report: {baseline_report_path}")

    # Edited run: either no-op (Compare & Certify) or provided edit_config (demo edit)
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

        # Ensure model.id/adapter point to the requested subject
        model_block = dict(cfg_loaded.get("model") or {})
        # Replace placeholder IDs like "<MODEL_ID>" or "<set-your-model-id>"
        if not isinstance(model_block.get("id"), str) or model_block.get(
            "id", ""
        ).startswith("<"):
            model_block["id"] = norm_edt_id
        else:
            # Always normalize when adapter is HF family
            model_block["id"] = _normalize_model_id(str(model_block["id"]), eff_adapter)
        # Respect explicit device from edit config; only set adapter if missing
        if not isinstance(model_block.get("adapter"), str) or not model_block.get(
            "adapter"
        ):
            model_block["adapter"] = eff_adapter
        cfg_loaded["model"] = model_block

        # Apply the same preset to the edited run to avoid duplicating dataset/task
        # settings in edit configs; then overlay the edit, output, and context.
        merged_edited_cfg = _merge(
            _merge(preset_data, cfg_loaded),
            {
                "output": {"dir": str(Path(out) / "edited")},
                "context": {"profile": profile, "tier": tier},
            },
        )
        # Ensure the edited run always has a guard chain. Presets/edit configs
        # often omit it, but `invarlock run` expects guards.order.
        guards_block = merged_edited_cfg.get("guards")
        guards_order_cfg = (
            guards_block.get("order") if isinstance(guards_block, dict) else None
        )
        if not (
            isinstance(guards_order_cfg, list)
            and guards_order_cfg
            and all(isinstance(item, str) for item in guards_order_cfg)
        ):
            merged_edited_cfg = _merge(
                merged_edited_cfg, {"guards": {"order": guards_order}}
            )

        # Persist a temporary merged config for traceability
        tmp_dir = Path(".certify_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
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
                    _run(
                        config=str(edited_merged_yaml),
                        profile=profile,
                        out=str(Path(out) / "edited"),
                        tier=tier,
                        baseline=str(baseline_report_path),
                        device=device,
                        edit_label=subject_label if edit_label else None,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
    else:
        edited_cfg = _merge(
            preset_data,
            {
                "model": {"id": norm_edt_id, "adapter": eff_adapter},
                "edit": {"name": "noop", "plan": {}},
                "eval": {},
                "guards": {"order": guards_order},
                "output": {"dir": str(Path(out) / "edited")},
                "context": {"profile": profile, "tier": tier},
            },
        )
        edited_yaml = tmp_dir / "edited_noop.yaml"
        _dump_yaml(edited_yaml, edited_cfg)
        _info("Running edited (no-op, Compare & Certify)", tag="EXEC", emoji="🧪")
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
                    _run(
                        config=str(edited_yaml),
                        profile=profile,
                        out=str(Path(out) / "edited"),
                        tier=tier,
                        baseline=str(baseline_report_path),
                        device=device,
                        edit_label=subject_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        no_color=no_color,
                    )
            except Exception:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

    edited_report = _latest_run_report(Path(out) / "edited")
    if not edited_report:
        print_event(
            console,
            "FAIL",
            "Could not locate edited report after run",
            style=output_style,
            emoji="❌",
        )
        raise typer.Exit(1)
    _debug(f"Edited report: {edited_report}")

    _phase(3, 3, "CERTIFICATE GENERATION")

    def _emit_certificate() -> None:
        _info("Emitting certificate", tag="EXEC", emoji="📜")
        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="certificate",
                    tag="EXEC",
                    message="Certificate",
                    emoji="📜",
                ):
                    report_kwargs = {
                        "run": str(edited_report),
                        "format": "cert",
                        "baseline": str(baseline_report_path),
                        "output": cert_out,
                        "style": output_style.name,
                        "no_color": no_color,
                    }
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

        def _finite(x: Any) -> bool:
            try:
                return isinstance(x, (int | float)) and math.isfinite(float(x))
            except Exception:
                return False

        meta = (
            edited_payload.get("meta", {}) if isinstance(edited_payload, dict) else {}
        )
        metrics = (
            edited_payload.get("metrics", {})
            if isinstance(edited_payload, dict)
            else {}
        )
        pm = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
        pm_prev = pm.get("preview") if isinstance(pm, dict) else None
        pm_final = pm.get("final") if isinstance(pm, dict) else None
        pm_ratio = pm.get("ratio_vs_baseline")
        device = meta.get("device") or "unknown"
        adapter_name = meta.get("adapter") or "unknown"
        edit_name = (
            (edited_payload.get("edit", {}) or {}).get("name")
            if isinstance(edited_payload, dict)
            else None
        ) or "unknown"

        # Enforce only when a primary_metric block is present; allow degraded-but-flagged metrics to emit certificates, but fail the task.
        has_metric_block = isinstance(pm, dict) and bool(pm)
        if has_metric_block:
            degraded = bool(pm.get("invalid") or pm.get("degraded"))
            if degraded or not _finite(pm_final):
                fallback = pm_prev if _finite(pm_prev) else pm_final
                if not _finite(fallback) or fallback <= 0:
                    fallback = 1.0
                degraded_reason = pm.get("degraded_reason") or (
                    "non_finite_pm"
                    if (not _finite(pm_prev) or not _finite(pm_final))
                    else "primary_metric_degraded"
                )
                print_event(
                    console,
                    "WARN",
                    "Primary metric degraded or non-finite; emitting certificate and marking task degraded. Primary metric computation failed.",
                    style=output_style,
                    emoji="⚠️",
                )
                pm["degraded"] = True
                pm["invalid"] = pm.get("invalid") or True
                pm["preview"] = pm_prev if _finite(pm_prev) else fallback
                pm["final"] = pm_final if _finite(pm_final) else fallback
                pm["ratio_vs_baseline"] = pm_ratio if _finite(pm_ratio) else 1.0
                pm["degraded_reason"] = degraded_reason
                metrics["primary_metric"] = pm
                edited_payload.setdefault("metrics", {}).update(metrics)

                # Emit the certificate for inspection, then exit with a CI-visible error.
                _emit_certificate()
                err = MetricsError(
                    code="E111",
                    message=f"Primary metric degraded or non-finite ({degraded_reason}).",
                    details={
                        "reason": degraded_reason,
                        "adapter": adapter_name,
                        "device": device,
                        "edit": edit_name,
                    },
                )
                raise typer.Exit(_resolve_exit_code(err, profile=profile))

    _emit_certificate()
    if timing:
        if total_start is not None:
            timings["total"] = max(0.0, float(perf_counter() - total_start))
        else:
            timings["total"] = (
                float(timings.get("baseline", 0.0))
                + float(timings.get("subject", 0.0))
                + float(timings.get("certificate", 0.0))
            )
        print_timing_summary(
            console,
            timings,
            style=output_style,
            order=[
                ("Baseline", "baseline"),
                ("Subject", "subject"),
                ("Certificate", "certificate"),
                ("Total", "total"),
            ],
        )
    if verbosity == VERBOSITY_QUIET:
        _print_quiet_summary(
            cert_out=Path(cert_out),
            source=src_id,
            edited=edt_id,
            profile=profile,
        )
