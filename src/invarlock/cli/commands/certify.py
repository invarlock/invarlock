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

import json
import math
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from ..adapter_auto import resolve_auto_adapter
from ..config import _deep_merge as _merge  # reuse helper
from ..errors import InvarlockError

# Use the report group's programmatic entry for report generation
from .report import report_command as _report
from .run import _resolve_exit_code as _resolve_exit_code

_LAZY_RUN_IMPORT = True

console = Console()


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
    adapter: str = typer.Option(
        "auto", "--adapter", help="Adapter name or 'auto' to resolve"
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device override for runs (auto|cuda|mps|cpu)",
    ),
    profile: str = typer.Option("ci", "--profile", help="Profile (ci|release)"),
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
    adapter = _coerce_option(adapter, "auto")
    device = _coerce_option(device)
    profile = _coerce_option(profile, "ci")
    tier = _coerce_option(tier, "balanced")
    preset = _coerce_option(preset)
    out = _coerce_option(out, "runs")
    cert_out = _coerce_option(cert_out, "reports/cert")
    edit_config = _coerce_option(edit_config)

    src_id = str(source)
    edt_id = str(edited)

    # Resolve adapter when requested
    eff_adapter = adapter
    if str(adapter).strip().lower() in {"auto", "hf_auto", "auto_hf"}:
        eff_adapter = resolve_auto_adapter(src_id)
        console.print(f"🔎 Adapter:auto → {eff_adapter}")

    # Choose preset. If none provided and repo preset is missing (pip install
    # scenario), fall back to a minimal built-in universal preset so the
    # flag-only quick start works without cloning the repo.
    default_universal = (
        Path("configs/tasks/masked_lm/ci_cpu.yaml")
        if eff_adapter == "hf_bert"
        else Path("configs/tasks/causal_lm/ci_cpu.yaml")
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
            console.print(f"[red]❌ Preset not found: {preset_path}")
            raise typer.Exit(1)
        preset_data = _load_yaml(preset_path)
        # Do not hard-code device from presets in auto-generated certify configs;
        # allow device resolution to pick CUDA/MPS/CPU via 'auto' or CLI overrides.
        model_block = preset_data.get("model")
        if isinstance(model_block, dict) and "device" in model_block:
            model_block = dict(model_block)
            model_block.pop("device", None)
            preset_data["model"] = model_block

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
            "guards": {
                "order": ["invariants", "spectral", "rmt", "variance", "invariants"]
            },
            "output": {"dir": str(Path(out) / "source")},
            "context": {"profile": profile, "tier": tier},
        },
    )

    tmp_dir = Path(".certify_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    baseline_yaml = tmp_dir / "baseline_noop.yaml"
    _dump_yaml(baseline_yaml, baseline_cfg)

    console.print("🏁 Running baseline (no-op edit)")
    from .run import run_command as _run

    _run(
        config=str(baseline_yaml),
        profile=profile,
        out=str(Path(out) / "source"),
        tier=tier,
        device=device,
    )

    baseline_report = _latest_run_report(Path(out) / "source")
    if not baseline_report:
        console.print("[red]❌ Could not locate baseline report after run")
        raise typer.Exit(1)

    # Edited run: either no-op (Compare & Certify) or provided edit_config (demo edit)
    if edit_config:
        edited_yaml = Path(edit_config)
        if not edited_yaml.exists():
            console.print(f"[red]❌ Edit config not found: {edited_yaml}")
            raise typer.Exit(1)
        console.print("✂️  Running edited (demo edit via --edit-config)")
        # Overlay subject model id/adapter and output/context onto the provided edit config
        try:
            cfg_loaded: dict[str, Any] = _load_yaml(edited_yaml)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]❌ Failed to load edit config: {exc}")
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

        # Persist a temporary merged config for traceability
        tmp_dir = Path(".certify_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        edited_merged_yaml = tmp_dir / "edited_merged.yaml"
        _dump_yaml(edited_merged_yaml, merged_edited_cfg)

        from .run import run_command as _run

        _run(
            config=str(edited_merged_yaml),
            profile=profile,
            out=str(Path(out) / "edited"),
            tier=tier,
            baseline=str(baseline_report),
            device=device,
        )
    else:
        edited_cfg = _merge(
            preset_data,
            {
                "model": {"id": norm_edt_id, "adapter": eff_adapter},
                "edit": {"name": "noop", "plan": {}},
                "eval": {},
                "guards": {
                    "order": [
                        "invariants",
                        "spectral",
                        "rmt",
                        "variance",
                        "invariants",
                    ]
                },
                "output": {"dir": str(Path(out) / "edited")},
                "context": {"profile": profile, "tier": tier},
            },
        )
        edited_yaml = tmp_dir / "edited_noop.yaml"
        _dump_yaml(edited_yaml, edited_cfg)
        console.print("🧪 Running edited (no-op, Compare & Certify)")
        from .run import run_command as _run

        _run(
            config=str(edited_yaml),
            profile=profile,
            out=str(Path(out) / "edited"),
            tier=tier,
            baseline=str(baseline_report),
            device=device,
        )

    edited_report = _latest_run_report(Path(out) / "edited")
    if not edited_report:
        console.print("[red]❌ Could not locate edited report after run")
        raise typer.Exit(1)

    # CI/Release hard‑abort: fail fast when primary metric is not computable.
    # Fall back to legacy ppl_* keys when primary_metric block is absent.
    try:
        prof = str(profile or "").strip().lower()
    except Exception:
        prof = ""
    if prof in {"ci", "release"}:
        try:
            with Path(edited_report).open("r", encoding="utf-8") as fh:
                edited_payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]❌ Failed to read edited report: {exc}")
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

        # Enforce only when a metric block is present; skip for minimal stub reports
        # Enforce only when a primary_metric block is present
        has_metric_block = isinstance(pm, dict) and bool(pm)
        if has_metric_block:
            # Treat non‑finite PM as hard error in CI/Release (after legacy fallback).
            # Require a finite final value; preview is optional for legacy reports.
            if not _finite(pm_final):
                err = InvarlockError(
                    code="E111",
                    message=(
                        "Primary metric computation failed (NaN/inf). "
                        f"Context: device={device}, adapter={adapter_name}, edit={edit_name}. "
                        "Baseline ok; edited failed to compute ppl. "
                        "Try: use an accelerator (mps/cuda), force float32, reduce max_modules, "
                        "or lower batch size (INVARLOCK_SCORES_BATCH_SIZE)."
                    ),
                    details={
                        "device": device,
                        "adapter": adapter_name,
                        "edit": edit_name,
                        "pm_preview": pm_prev,
                        "pm_final": pm_final,
                        "pm_ratio": pm_ratio,
                    },
                )
                code = _resolve_exit_code(err, profile=prof)
                console.print(f"[red]{err}[/red]")
                # Do not emit a certificate
                raise typer.Exit(code)

    console.print("📜 Emitting certificate")
    _report(
        run=str(edited_report),
        format="cert",
        baseline=str(baseline_report),
        output=cert_out,
    )
