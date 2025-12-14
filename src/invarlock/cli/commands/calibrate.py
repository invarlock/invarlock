"""Calibration sweep harnesses (null + VE).

These commands run repeatable sweeps and emit stable artifacts for release notes:
- JSON (machine)
- CSV (spreadsheet)
- Markdown (human)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from invarlock.calibration.spectral_null import summarize_null_sweep_reports
from invarlock.calibration.variance_ve import summarize_ve_sweep_reports
from invarlock.guards.tier_config import get_tier_guard_config

console = Console()

calibrate_app = typer.Typer(
    name="calibrate",
    help="Run calibration sweeps and emit reports (JSON/CSV/Markdown).",
    no_args_is_help=True,
)


@dataclass(frozen=True)
class _SweepSpec:
    tier: str
    seed: int
    windows: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Config must be a mapping: {path}")
    return data


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _dump_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _materialize_sweep_specs(
    *,
    tiers: list[str] | None,
    seeds: list[int] | None,
    n_seeds: int,
    seed_start: int,
    windows: list[int] | None = None,
) -> list[_SweepSpec]:
    tier_list = [t.strip().lower() for t in (tiers or []) if str(t).strip()]
    if not tier_list:
        tier_list = ["balanced", "conservative", "aggressive"]

    seed_list = [int(s) for s in (seeds or [])]
    if not seed_list:
        seed_list = [int(seed_start) + i for i in range(int(n_seeds))]

    out: list[_SweepSpec] = []
    if windows:
        for tier in tier_list:
            for win in windows:
                for seed in seed_list:
                    out.append(_SweepSpec(tier=tier, seed=seed, windows=int(win)))
    else:
        for tier in tier_list:
            for seed in seed_list:
                out.append(_SweepSpec(tier=tier, seed=seed))
    return out


def _write_tiers_recommendation(
    out_path: Path,
    *,
    recommendations: dict[str, dict[str, Any]],
) -> None:
    """Write a tiers.yaml-shaped patch file (only keys we recommend)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(recommendations, sort_keys=False),
        encoding="utf-8",
    )


@calibrate_app.command(
    name="null-sweep",
    help="Run a null (no-op edit) sweep and calibrate spectral κ/alpha empirically.",
)
def null_sweep(
    config: Path = typer.Option(
        Path("configs/calibration/null_sweep_ci.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Base null-sweep YAML (noop edit).",
    ),
    out: Path = typer.Option(
        Path("reports/calibration/null_sweep"),
        "--out",
        help="Output directory for calibration artifacts.",
    ),
    tiers: list[str] = typer.Option(
        None,
        "--tier",
        help="Tier(s) to evaluate (repeatable). Defaults to all tiers.",
    ),
    seed: list[int] = typer.Option(
        None,
        "--seed",
        help="Seed(s) to run (repeatable). Overrides --n-seeds/--seed-start.",
    ),
    n_seeds: int = typer.Option(10, "--n-seeds", min=1, help="Number of seeds to run."),
    seed_start: int = typer.Option(42, "--seed-start", help="Starting seed."),
    profile: str = typer.Option("ci", "--profile", help="Run profile (ci|release)."),
    device: str | None = typer.Option(None, "--device", help="Device override."),
    safety_margin: float = typer.Option(
        0.05, "--safety-margin", help="Safety margin applied to κ recommendations."
    ),
    target_any_warning_rate: float = typer.Option(
        0.01,
        "--target-any-warning-rate",
        help="Target run-level spectral warning rate under the null.",
    ),
) -> None:
    # Keep import light: only pull run machinery when invoked.
    from .run import run_command

    base = _load_yaml(config)
    specs = _materialize_sweep_specs(
        tiers=tiers, seeds=seed, n_seeds=n_seeds, seed_start=seed_start
    )
    specs = sorted(specs, key=lambda s: (s.tier, s.seed))

    run_rows: list[dict[str, Any]] = []
    reports_by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)

    run_root = out / "runs"
    cfg_root = out / "configs"
    cfg_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        cfg = json.loads(json.dumps(base))  # safe deep copy without yaml anchors
        cfg.setdefault("dataset", {})["seed"] = int(spec.seed)
        cfg.setdefault("auto", {})["tier"] = spec.tier

        # Per-run config + output roots to avoid timestamp collisions.
        run_out = run_root / spec.tier / f"seed_{spec.seed}"
        cfg.setdefault("output", {})["dir"] = str(run_out)
        cfg_path = cfg_root / f"null_{spec.tier}_{spec.seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        report_path = run_command(
            config=str(cfg_path),
            device=device,
            profile=profile,
            out=str(run_out),
            tier=spec.tier,
        )
        if not isinstance(report_path, str):
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        reports_by_tier[spec.tier].append(report)

        spectral = None
        for g in report.get("guards", []):
            if isinstance(g, dict) and g.get("name") == "spectral":
                spectral = g
                break
        metrics = spectral.get("metrics", {}) if isinstance(spectral, dict) else {}
        selection = (
            metrics.get("multiple_testing_selection")
            if isinstance(metrics.get("multiple_testing_selection"), dict)
            else {}
        )
        fam_z_summary = (
            metrics.get("family_z_summary")
            if isinstance(metrics.get("family_z_summary"), dict)
            else {}
        )
        candidate_counts = (
            selection.get("family_violation_counts")
            if isinstance(selection.get("family_violation_counts"), dict)
            else {}
        )
        selected_families = selection.get("families_selected")
        selected_set = (
            {str(x) for x in selected_families}
            if isinstance(selected_families, list)
            else set()
        )
        selected_by_family: dict[str, int] = defaultdict(int)
        violations = (
            spectral.get("violations", []) if isinstance(spectral, dict) else []
        )
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict) and v.get("family") is not None:
                    selected_by_family[str(v.get("family"))] += 1
        caps_applied = metrics.get("caps_applied")
        try:
            caps_applied = int(caps_applied) if caps_applied is not None else 0
        except Exception:
            caps_applied = 0
        row: dict[str, Any] = {
            "tier": spec.tier,
            "seed": spec.seed,
            "caps_applied": caps_applied,
            "caps_exceeded": bool(metrics.get("caps_exceeded", False)),
            "selected_families": ",".join(sorted(selected_set)),
        }
        for fam, vals in fam_z_summary.items():
            if not isinstance(vals, dict):
                continue
            max_z = vals.get("max")
            try:
                if max_z is not None and max_z == max_z:
                    row[f"max_z_{fam}"] = float(max_z)
            except Exception:
                continue
        for fam, count in candidate_counts.items():
            try:
                row[f"candidate_{fam}"] = int(count)
            except Exception:
                continue
        for fam, count in selected_by_family.items():
            row[f"selected_{fam}"] = int(count)
        run_rows.append(row)

    summaries: dict[str, Any] = {}
    tiers_patch: dict[str, dict[str, Any]] = {}
    for tier_name, tier_reports in sorted(reports_by_tier.items()):
        summary = summarize_null_sweep_reports(
            tier_reports,
            tier=tier_name,
            safety_margin=safety_margin,
            target_any_warning_rate=target_any_warning_rate,
        )
        summaries[tier_name] = summary
        rec = summary.get("recommendations", {}) if isinstance(summary, dict) else {}
        spectral_patch = {
            "spectral_guard": {
                "family_caps": (rec.get("family_caps") or {}),
                "multiple_testing": (rec.get("multiple_testing") or {}),
            }
        }
        tiers_patch[tier_name] = spectral_patch

    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "kind": "spectral_null_sweep",
        "generated_at": stamp,
        "config": {
            "base_config": str(config),
            "profile": str(profile),
            "tiers": sorted(reports_by_tier.keys()),
            "n_runs": int(sum(len(v) for v in reports_by_tier.values())),
        },
        "summaries": summaries,
    }

    _dump_json(out / "null_sweep_report.json", payload)
    _dump_csv(out / "null_sweep_runs.csv", run_rows)
    _write_tiers_recommendation(
        out / "tiers_patch_spectral_null.yaml", recommendations=tiers_patch
    )

    md_lines = [
        "# Spectral null-sweep calibration",
        "",
        f"- Generated: `{stamp}`",
        f"- Base config: `{config}`",
        "",
        "## Recommendations (tiers.yaml patch)",
        f"- `{out / 'tiers_patch_spectral_null.yaml'}`",
        "",
        "## Summary",
        "",
        "| Tier | Runs | Any-warning rate | α (recommended) |",
        "|---|---:|---:|---:|",
    ]
    for tier_name, summary in sorted(summaries.items()):
        obs = summary.get("observed", {}) if isinstance(summary, dict) else {}
        rec = summary.get("recommendations", {}) if isinstance(summary, dict) else {}
        mt = rec.get("multiple_testing", {}) if isinstance(rec, dict) else {}
        md_lines.append(
            f"| {tier_name} | {summary.get('n_runs', 0)} | {obs.get('any_warning_rate', 0.0):.3f} | {float(mt.get('alpha', 0.0)):.6f} |"
        )
    _dump_markdown(out / "null_sweep_summary.md", "\n".join(md_lines))

    console.print(f"[green]✅ Wrote null sweep artifacts under {out}[/green]")


@calibrate_app.command(
    name="ve-sweep",
    help="Run VE predictive-gate sweeps and recommend min_effect_lognll per tier.",
)
def ve_sweep(
    config: Path = typer.Option(
        Path("configs/calibration/rmt_ve_sweep_ci.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Base VE sweep YAML (quant_rtn edit).",
    ),
    out: Path = typer.Option(
        Path("reports/calibration/ve_sweep"),
        "--out",
        help="Output directory for calibration artifacts.",
    ),
    tiers: list[str] = typer.Option(
        None,
        "--tier",
        help="Tier(s) to evaluate (repeatable). Defaults to all tiers.",
    ),
    seed: list[int] = typer.Option(
        None,
        "--seed",
        help="Seed(s) to run (repeatable). Overrides --n-seeds/--seed-start.",
    ),
    n_seeds: int = typer.Option(10, "--n-seeds", min=1, help="Number of seeds to run."),
    seed_start: int = typer.Option(42, "--seed-start", help="Starting seed."),
    window: list[int] = typer.Option(
        None,
        "--window",
        help="Variance calibration window counts (repeatable). Defaults to 6, 8, 12, 16.",
    ),
    target_enable_rate: float = typer.Option(
        0.05,
        "--target-enable-rate",
        help="Target expected VE enable rate (predictive-gate lower bound).",
    ),
    profile: str = typer.Option("ci", "--profile", help="Run profile (ci|release)."),
    device: str | None = typer.Option(None, "--device", help="Device override."),
    safety_margin: float = typer.Option(
        0.0,
        "--safety-margin",
        help="Safety margin applied to min_effect recommendations.",
    ),
) -> None:
    # Keep import light: only pull run machinery when invoked.
    from .run import run_command

    base = _load_yaml(config)
    windows = [int(w) for w in (window or [])] or [6, 8, 12, 16]
    specs = _materialize_sweep_specs(
        tiers=tiers,
        seeds=seed,
        n_seeds=n_seeds,
        seed_start=seed_start,
        windows=windows,
    )
    specs = sorted(specs, key=lambda s: (s.tier, int(s.windows or 0), s.seed))

    run_rows: list[dict[str, Any]] = []
    reports_by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reports_by_tier_window: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )

    run_root = out / "runs"
    cfg_root = out / "configs"
    cfg_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        win = int(spec.windows or 0)
        cfg = json.loads(json.dumps(base))  # safe deep copy without yaml anchors
        cfg.setdefault("dataset", {})["seed"] = int(spec.seed)
        cfg.setdefault("auto", {})["tier"] = spec.tier
        # Keep edit deterministic when it supports a seed knob.
        plan = cfg.setdefault("edit", {}).setdefault("plan", {})
        if isinstance(plan, dict) and "seed" in plan:
            plan["seed"] = int(spec.seed)

        # Override variance calibration windows and ensure min_coverage is feasible.
        gv = cfg.setdefault("guards", {}).setdefault("variance", {})
        if not isinstance(gv, dict):
            gv = {}
            cfg["guards"]["variance"] = gv
        calib = gv.setdefault("calibration", {})
        if not isinstance(calib, dict):
            calib = {}
            gv["calibration"] = calib
        calib["windows"] = int(win)
        calib["seed"] = int(spec.seed)
        calib["min_coverage"] = int(max(1, min(win, win - 2)))

        run_out = run_root / spec.tier / f"windows_{win}" / f"seed_{spec.seed}"
        cfg.setdefault("output", {})["dir"] = str(run_out)
        cfg_path = cfg_root / f"ve_{spec.tier}_w{win}_{spec.seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        report_path = run_command(
            config=str(cfg_path),
            device=device,
            profile=profile,
            out=str(run_out),
            tier=spec.tier,
        )
        if not isinstance(report_path, str):
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        reports_by_tier[spec.tier].append(report)
        reports_by_tier_window[(spec.tier, win)].append(report)

        variance = None
        for g in report.get("guards", []):
            if isinstance(g, dict) and g.get("name") == "variance":
                variance = g
                break
        metrics = variance.get("metrics", {}) if isinstance(variance, dict) else {}
        pg = (
            metrics.get("predictive_gate", {})
            if isinstance(metrics.get("predictive_gate"), dict)
            else {}
        )
        delta_ci = pg.get("delta_ci")
        try:
            ci_width = (
                float(delta_ci[1]) - float(delta_ci[0])
                if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2
                else None
            )
        except Exception:
            ci_width = None
        run_rows.append(
            {
                "tier": spec.tier,
                "seed": spec.seed,
                "windows": win,
                "predictive_evaluated": bool(pg.get("evaluated", False)),
                "predictive_mean_delta": pg.get("mean_delta"),
                "predictive_delta_ci_lo": (
                    delta_ci[0]
                    if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2
                    else None
                ),
                "predictive_delta_ci_hi": (
                    delta_ci[1]
                    if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2
                    else None
                ),
                "predictive_ci_width": ci_width,
            }
        )

    # Per-tier recommendation using all runs (across window values).
    summaries: dict[str, Any] = {}
    tiers_patch: dict[str, dict[str, Any]] = {}
    for tier_name, tier_reports in sorted(reports_by_tier.items()):
        var_cfg = get_tier_guard_config(tier_name, "variance_guard")
        one_sided = bool(var_cfg.get("predictive_one_sided", True))
        summary = summarize_ve_sweep_reports(
            tier_reports,
            tier=tier_name,
            target_enable_rate=target_enable_rate,
            safety_margin=safety_margin,
            predictive_one_sided=one_sided,
        )
        summaries[tier_name] = summary
        rec = summary.get("recommendations", {}) if isinstance(summary, dict) else {}
        tiers_patch[tier_name] = {
            "variance_guard": {"min_effect_lognll": rec.get("min_effect_lognll")}
        }

    # Power curve: mean CI width per (tier, windows).
    power_curve: list[dict[str, Any]] = []
    for (tier_name, win), items in sorted(reports_by_tier_window.items()):
        widths: list[float] = []
        for rep in items:
            g = None
            for gg in rep.get("guards", []):
                if isinstance(gg, dict) and gg.get("name") == "variance":
                    g = gg
                    break
            metrics = g.get("metrics", {}) if isinstance(g, dict) else {}
            pg = (
                metrics.get("predictive_gate", {})
                if isinstance(metrics.get("predictive_gate"), dict)
                else {}
            )
            delta_ci = pg.get("delta_ci")
            if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
                try:
                    widths.append(float(delta_ci[1]) - float(delta_ci[0]))
                except Exception:
                    continue
        power_curve.append(
            {
                "tier": tier_name,
                "windows": int(win),
                "runs": int(len(items)),
                "mean_ci_width": (sum(widths) / len(widths)) if widths else None,
            }
        )

    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "kind": "variance_ve_sweep",
        "generated_at": stamp,
        "config": {
            "base_config": str(config),
            "profile": str(profile),
            "tiers": sorted(reports_by_tier.keys()),
            "windows": windows,
            "n_runs": int(sum(len(v) for v in reports_by_tier.values())),
        },
        "summaries": summaries,
        "power_curve": power_curve,
    }

    _dump_json(out / "ve_sweep_report.json", payload)
    _dump_csv(out / "ve_sweep_runs.csv", run_rows)
    _dump_csv(out / "ve_power_curve.csv", power_curve)
    _write_tiers_recommendation(
        out / "tiers_patch_variance_ve.yaml", recommendations=tiers_patch
    )

    md_lines = [
        "# Variance (DD-VE) sweep calibration",
        "",
        f"- Generated: `{stamp}`",
        f"- Base config: `{config}`",
        "",
        "## Recommendations (tiers.yaml patch)",
        f"- `{out / 'tiers_patch_variance_ve.yaml'}`",
        "",
        "## Per-tier recommendation",
        "",
        "| Tier | Runs | Recommended min_effect_lognll | Expected enable rate |",
        "|---|---:|---:|---:|",
    ]
    for tier_name, summary in sorted(summaries.items()):
        rec = summary.get("recommendations", {}) if isinstance(summary, dict) else {}
        md_lines.append(
            f"| {tier_name} | {summary.get('n_runs', 0)} | {float(rec.get('min_effect_lognll', 0.0)):.6f} | {float(rec.get('expected_enable_rate', 0.0)):.3f} |"
        )
    _dump_markdown(out / "ve_sweep_summary.md", "\n".join(md_lines))

    console.print(f"[green]✅ Wrote VE sweep artifacts under {out}[/green]")


__all__ = ["calibrate_app"]
