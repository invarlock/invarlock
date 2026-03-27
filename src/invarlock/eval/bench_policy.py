from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from invarlock.reporting.report_types import RunReport


@dataclass
class ScenarioConfig:
    """Configuration for a single benchmark scenario."""

    edit: str
    tier: str
    probes: int
    profile: str = "ci"
    model_id: str = "gpt2"
    adapter: str = "hf_causal"
    device: str = "auto"
    seq_len: int = 512
    stride: int = 128
    preview_n: int | None = None
    final_n: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.profile == "ci":
            if self.preview_n is None:
                self.preview_n = 50
            if self.final_n is None:
                self.final_n = 50
        elif self.profile == "release":
            if self.preview_n is None:
                self.preview_n = 100
            if self.final_n is None:
                self.final_n = 100
        else:
            raise ValueError(f"Unknown profile: {self.profile}. Use 'ci' or 'release'")


@dataclass
class BenchmarkConfig:
    """Global configuration for benchmark execution."""

    edits: list[str]
    tiers: list[str]
    probes: list[int]
    profile: str = "ci"
    dataset: str = "wikitext2"
    model_id: str = "gpt2"
    adapter: str = "hf_causal"
    device: str = "auto"
    seq_len: int = 512
    stride: int = 128
    seed: int = 42
    output_dir: Path = Path("benchmarks")
    epsilon: float | None = None
    ppl_overhead_threshold: float = 0.01
    guard_overhead_time_threshold: float = 0.15
    guard_overhead_mem_threshold: float = 0.10
    catastrophic_spike_threshold: float = 2.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class RunResult:
    """Results from a single run (bare or guarded)."""

    run_type: str
    report: RunReport
    success: bool
    error_message: str | None = None


@dataclass
class ScenarioResult:
    """Results from a single benchmark scenario."""

    config: ScenarioConfig
    bare_result: RunResult | None = None
    guarded_result: RunResult | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None
    probes_used: int = 0
    epsilon_used: float = 0.0


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""

    config: BenchmarkConfig
    scenarios: list[ScenarioResult]
    overall_pass: bool
    timestamp: str
    execution_time_seconds: float
    schema_version: str = "bench-v1"


class ConfigurationManager:
    """Manage configuration generation for bare vs guarded runs."""

    @staticmethod
    def create_base_config(scenario: ScenarioConfig) -> dict[str, Any]:
        return {
            "model": {
                "id": scenario.model_id,
                "adapter": scenario.adapter,
                "device": scenario.device,
                "dtype": "float16",
            },
            "dataset": {
                "provider": "wikitext2",
                "seq_len": scenario.seq_len,
                "stride": scenario.stride,
                "preview_n": scenario.preview_n,
                "final_n": scenario.final_n,
                "seed": scenario.seed,
            },
            "edit": {
                "name": scenario.edit,
                "plan": ConfigurationManager._get_edit_plan(
                    scenario.edit, scenario.profile
                ),
            },
            "eval": {"spike_threshold": 2.0},
            "output": {"dir": "runs"},
        }

    @staticmethod
    def _get_edit_plan(edit_name: str, profile: str) -> dict[str, Any]:
        plans = {
            "quant_rtn": {
                "bitwidth": 8,
                "per_channel": True,
                "group_size": 128,
                "clamp_ratio": 0.0,
                "scope": "ffn",
            },
            "gptq": {"bits": 4, "group_size": 128, "damp_percent": 0.01},
        }
        _ = profile
        return plans.get(edit_name, {})

    @classmethod
    def create_bare_config(cls, scenario: ScenarioConfig) -> dict[str, Any]:
        base_config = cls.create_base_config(scenario)
        base_config["auto"] = {"enabled": False, "tier": "balanced", "probes": 0}
        base_config["guards"] = {
            "order": ["invariants"],
            "invariants": {"mode": "warn"},
        }
        return base_config

    @classmethod
    def create_guarded_config(cls, scenario: ScenarioConfig) -> dict[str, Any]:
        base_config = cls.create_base_config(scenario)
        base_config["auto"] = {
            "enabled": True,
            "tier": scenario.tier,
            "probes": scenario.probes,
            "target_pm_ratio": None,
        }
        base_config["guards"] = {
            "order": ["invariants", "spectral", "rmt", "variance", "invariants_post"],
            "invariants": {"mode": "enforce"},
            "invariants_post": {"mode": "enforce"},
        }
        return base_config


class MetricsAggregator:
    """Aggregate and validate metrics from paired runs."""

    @staticmethod
    def extract_core_metrics(report: RunReport) -> dict[str, float]:
        metrics = report.get("metrics", {}) or {}
        meta = report.get("meta", {}) or {}
        pm = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
        pm_preview = float("nan")
        pm_final = float("nan")
        try:
            if isinstance(pm, dict):
                if isinstance(pm.get("preview"), int | float):
                    pm_preview = float(pm["preview"])  # type: ignore[index]
                if isinstance(pm.get("final"), int | float):
                    pm_final = float(pm["final"])  # type: ignore[index]
        except Exception:
            pm_preview = float("nan")
            pm_final = float("nan")
        duration_s = float("nan")
        try:
            if isinstance(meta, dict):
                dur = meta.get("duration_s", meta.get("duration"))
                if isinstance(dur, int | float):
                    duration_s = float(dur)
        except Exception:
            duration_s = float("nan")
        return {
            "primary_metric_preview": pm_preview,
            "primary_metric_final": pm_final,
            "latency_ms_per_tok": metrics.get("latency_ms_per_tok", float("nan")),
            "memory_mb_peak": metrics.get("memory_mb_peak", float("nan")),
            "duration_s": duration_s,
        }

    @staticmethod
    def extract_guard_metrics(report: RunReport) -> dict[str, Any]:
        guard_metrics: dict[str, Any] = {}
        guards = report.get("guards", [])
        if isinstance(guards, list):
            for guard in guards:
                if not isinstance(guard, dict):
                    continue
                name = str(guard.get("name", "")).lower()
                metrics = (
                    guard.get("metrics", {})
                    if isinstance(guard.get("metrics"), dict)
                    else {}
                )
                violations = guard.get("violations", [])
                if name == "rmt":
                    for key in ("outliers_total", "rmt_outliers", "layers_flagged"):
                        val = metrics.get(key)
                        if isinstance(val, int | float):
                            guard_metrics["rmt_outliers"] = int(val)
                            break
                if name == "invariants":
                    val = metrics.get("violations_found")
                    if isinstance(val, int | float):
                        guard_metrics["tying_violations_post"] = int(val)
                    elif isinstance(violations, list):
                        guard_metrics["tying_violations_post"] = len(violations)

        if "rmt_outliers" not in guard_metrics:
            rmt_metrics = report.get("metrics", {}).get("rmt", {})
            if isinstance(rmt_metrics, dict):
                guard_metrics["rmt_outliers"] = int(rmt_metrics.get("outliers", 0) or 0)
            else:
                guard_metrics["rmt_outliers"] = 0

        if "tying_violations_post" not in guard_metrics:
            invariant_metrics = report.get("metrics", {}).get("invariants", {})
            if isinstance(invariant_metrics, dict):
                guard_metrics["tying_violations_post"] = int(
                    invariant_metrics.get("violations", 0) or 0
                )
            else:
                guard_metrics["tying_violations_post"] = 0

        flags = report.get("flags", {}) or {}
        meta = report.get("meta", {}) or {}
        guard_metrics["catastrophic_spike"] = bool(
            (flags.get("guard_recovered") if isinstance(flags, dict) else False)
            or (meta.get("guard_recovered") if isinstance(meta, dict) else False)
            or (meta.get("rollback_reason") if isinstance(meta, dict) else False)
        )
        return guard_metrics

    @classmethod
    def compute_comparison_metrics(
        cls, bare_result: RunResult, guarded_result: RunResult
    ) -> dict[str, Any]:
        if not (
            bare_result
            and guarded_result
            and bare_result.success
            and guarded_result.success
        ):
            return {}

        bare_metrics = cls.extract_core_metrics(bare_result.report)
        guarded_metrics = cls.extract_core_metrics(guarded_result.report)
        bare_guards = cls.extract_guard_metrics(bare_result.report)
        guarded_guards = cls.extract_guard_metrics(guarded_result.report)

        comparison = {
            "primary_metric_bare": bare_metrics.get("primary_metric_final", float("nan")),
            "primary_metric_guarded": guarded_metrics.get(
                "primary_metric_final", float("nan")
            ),
            "latency_bare": bare_metrics.get("latency_ms_per_tok", float("nan")),
            "latency_guarded": guarded_metrics.get("latency_ms_per_tok", float("nan")),
            "duration_bare_s": bare_metrics.get("duration_s", float("nan")),
            "duration_guarded_s": guarded_metrics.get("duration_s", float("nan")),
            "mem_bare": bare_metrics.get("memory_mb_peak", float("nan")),
            "mem_guarded": guarded_metrics.get("memory_mb_peak", float("nan")),
        }

        pm_bare = comparison["primary_metric_bare"]
        pm_guarded = comparison["primary_metric_guarded"]
        if not (math.isnan(pm_bare) or math.isnan(pm_guarded)) and pm_bare > 0:
            comparison["primary_metric_overhead"] = (pm_guarded - pm_bare) / pm_bare
        else:
            comparison["primary_metric_overhead"] = float("nan")

        duration_bare = comparison.get("duration_bare_s", float("nan"))
        duration_guarded = comparison.get("duration_guarded_s", float("nan"))
        if (
            isinstance(duration_bare, int | float)
            and isinstance(duration_guarded, int | float)
            and not (math.isnan(duration_bare) or math.isnan(duration_guarded))
            and float(duration_bare) > 0
        ):
            comparison["guard_overhead_time"] = (
                float(duration_guarded) - float(duration_bare)
            ) / float(duration_bare)
        else:
            latency_bare = comparison["latency_bare"]
            latency_guarded = comparison["latency_guarded"]
            if (
                not (math.isnan(latency_bare) or math.isnan(latency_guarded))
                and latency_bare > 0
            ):
                comparison["guard_overhead_time"] = (
                    latency_guarded - latency_bare
                ) / latency_bare
            else:
                comparison["guard_overhead_time"] = float("nan")

        mem_bare = comparison["mem_bare"]
        mem_guarded = comparison["mem_guarded"]
        if not (math.isnan(mem_bare) or math.isnan(mem_guarded)) and mem_bare > 0:
            comparison["guard_overhead_mem"] = (mem_guarded - mem_bare) / mem_bare
        else:
            comparison["guard_overhead_mem"] = float("nan")

        comparison.update(
            {
                "rmt_outliers_bare": bare_guards.get("rmt_outliers", 0),
                "rmt_outliers_guarded": guarded_guards.get("rmt_outliers", 0),
                "tying_violations_post": guarded_guards.get("tying_violations_post", 0),
                "catastrophic_spike": bare_guards.get("catastrophic_spike", False)
                or guarded_guards.get("catastrophic_spike", False),
            }
        )
        return comparison


class ValidationGates:
    """Validate metrics against Step 14 gates."""

    @staticmethod
    def validate_catastrophic_spike_rate(comparison: dict[str, Any]) -> bool:
        return not comparison.get("catastrophic_spike", False)

    @staticmethod
    def validate_tying_violations(comparison: dict[str, Any]) -> bool:
        return comparison.get("tying_violations_post", 0) == 0

    @staticmethod
    def validate_rmt_outliers(comparison: dict[str, Any], epsilon: float) -> bool:
        bare_outliers = comparison.get("rmt_outliers_bare", 0)
        guarded_outliers = comparison.get("rmt_outliers_guarded", 0)
        allowed = math.ceil(bare_outliers * (1.0 + epsilon))
        return guarded_outliers <= allowed

    @staticmethod
    def validate_primary_metric_overhead(
        comparison: dict[str, Any], threshold: float = 0.01
    ) -> bool:
        overhead = comparison.get("primary_metric_overhead", float("nan"))
        if math.isnan(overhead):
            return True
        return overhead <= threshold

    @staticmethod
    def validate_time_overhead(
        comparison: dict[str, Any], threshold: float = 0.15
    ) -> bool:
        overhead = comparison.get("guard_overhead_time", float("nan"))
        if math.isnan(overhead):
            return True
        return overhead <= threshold

    @staticmethod
    def validate_memory_overhead(
        comparison: dict[str, Any], threshold: float = 0.10
    ) -> bool:
        overhead = comparison.get("guard_overhead_mem", float("nan"))
        if math.isnan(overhead):
            return True
        return overhead <= threshold

    @classmethod
    def validate_all_gates(
        cls, comparison: dict[str, Any], config: BenchmarkConfig, epsilon: float
    ) -> dict[str, bool]:
        return {
            "spike": cls.validate_catastrophic_spike_rate(comparison),
            "tying": cls.validate_tying_violations(comparison),
            "rmt": cls.validate_rmt_outliers(comparison, epsilon),
            "quality": cls.validate_primary_metric_overhead(
                comparison, config.ppl_overhead_threshold
            ),
            "time": cls.validate_time_overhead(
                comparison, config.guard_overhead_time_threshold
            ),
            "mem": cls.validate_memory_overhead(
                comparison, config.guard_overhead_mem_threshold
            ),
        }


def generate_scenarios(config: BenchmarkConfig) -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            edit=edit,
            tier=tier,
            probes=probes,
            profile=config.profile,
            model_id=config.model_id,
            adapter=config.adapter,
            device=config.device,
            seq_len=config.seq_len,
            stride=config.stride,
            seed=config.seed,
        )
        for edit, tier, probes in itertools.product(
            config.edits, config.tiers, config.probes
        )
    ]


def resolve_epsilon_from_runtime(guarded_report: RunReport) -> float:
    guards = guarded_report.get("guards", [])
    for guard in guards:
        if guard.get("name") == "rmt":
            policy = guard.get("policy", {})
            deadband = policy.get("deadband")
            if deadband is not None:
                return float(deadband)
    return 0.10


def summary_to_step14_json(summary: BenchmarkSummary) -> dict[str, Any]:
    scenarios_data = []
    for result in summary.scenarios:
        scenario_data = {
            "edit": result.config.edit,
            "tier": result.config.tier,
            "probes": result.config.probes,
            "probes_used": result.probes_used,
            "skip": result.skipped,
            "skip_reason": result.skip_reason,
            "artifacts": result.artifacts,
        }
        if not result.skipped and result.metrics:
            scenario_data.update(
                {
                    "primary_metric_bare": result.metrics.get("primary_metric_bare"),
                    "primary_metric_guarded": result.metrics.get(
                        "primary_metric_guarded"
                    ),
                    "primary_metric_overhead": result.metrics.get(
                        "primary_metric_overhead"
                    ),
                    "latency_bare": result.metrics.get("latency_bare"),
                    "latency_guarded": result.metrics.get("latency_guarded"),
                    "guard_overhead_time": result.metrics.get("guard_overhead_time"),
                    "mem_bare": result.metrics.get("mem_bare"),
                    "mem_guarded": result.metrics.get("mem_guarded"),
                    "guard_overhead_mem": result.metrics.get("guard_overhead_mem"),
                    "rmt_outliers_bare": result.metrics.get("rmt_outliers_bare"),
                    "rmt_outliers_guarded": result.metrics.get(
                        "rmt_outliers_guarded"
                    ),
                    "tying_violations_post": result.metrics.get(
                        "tying_violations_post"
                    ),
                    "epsilon": result.epsilon_used,
                    "pass": result.gates,
                }
            )
        else:
            scenario_data.update(
                {
                    "primary_metric_bare": None,
                    "primary_metric_guarded": None,
                    "primary_metric_overhead": None,
                    "latency_bare": None,
                    "latency_guarded": None,
                    "guard_overhead_time": None,
                    "mem_bare": None,
                    "mem_guarded": None,
                    "guard_overhead_mem": None,
                    "rmt_outliers_bare": None,
                    "rmt_outliers_guarded": None,
                    "tying_violations_post": None,
                    "epsilon": None,
                    "pass": {
                        "spike": None,
                        "tying": None,
                        "rmt": None,
                        "quality": None,
                        "time": None,
                        "mem": None,
                    },
                }
            )
        scenarios_data.append(scenario_data)

    return {
        "schema_version": summary.schema_version,
        "profile": summary.config.profile,
        "seed": summary.config.seed,
        "epsilon": summary.config.epsilon,
        "scenarios": scenarios_data,
    }


def generate_step14_markdown(summary: BenchmarkSummary) -> str:
    lines = [
        "# InvarLock Guard Effect Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Profile:** {summary.config.profile}",
        f"**Seed:** {summary.config.seed}",
        f"**Epsilon:** {summary.config.epsilon if summary.config.epsilon is not None else 'auto'}",
        f"**Execution Time:** {summary.execution_time_seconds:.1f}s",
        f"**Overall Result:** {'✅ PASS' if summary.overall_pass else '❌ FAIL'}",
        "",
        "## Scenario Results",
        "",
        "| Edit | Tier | Probes | Status | PM Δ | Time Δ | Mem Δ | RMT | Gates |",
        "|------|------|--------|--------|-------|--------|-------|-----|-------|",
    ]
    for result in summary.scenarios:
        if result.skipped:
            status = "⏸️ SKIP"
            ppl_delta = "-"
            time_delta = "-"
            mem_delta = "-"
            rmt_info = "-"
            gates_info = f"SKIP: {result.skip_reason}"
        else:
            all_pass = all(result.gates.values()) if result.gates else False
            status = "✅ PASS" if all_pass else "❌ FAIL"
            pm_overhead = result.metrics.get("primary_metric_overhead")
            if pm_overhead is not None and not math.isnan(pm_overhead):
                ppl_delta = f"{pm_overhead:+.1%}"
                ppl_delta = f"🔴 {ppl_delta}" if pm_overhead > 0.01 else f"🟢 {ppl_delta}"
            else:
                ppl_delta = "-"
            time_overhead = result.metrics.get("guard_overhead_time")
            if time_overhead is not None and not math.isnan(time_overhead):
                time_delta = f"{time_overhead:+.1%}"
                time_delta = (
                    f"🔴 {time_delta}" if time_overhead > 0.15 else f"🟢 {time_delta}"
                )
            else:
                time_delta = "-"
            mem_overhead = result.metrics.get("guard_overhead_mem")
            if mem_overhead is not None and not math.isnan(mem_overhead):
                mem_delta = f"{mem_overhead:+.1%}"
                mem_delta = (
                    f"🔴 {mem_delta}" if mem_overhead > 0.10 else f"🟢 {mem_delta}"
                )
            else:
                mem_delta = "-"
            bare_outliers = result.metrics.get("rmt_outliers_bare", 0)
            guarded_outliers = result.metrics.get("rmt_outliers_guarded", 0)
            rmt_info = f"{bare_outliers}→{guarded_outliers}"
            gates_status = []
            gates_status.append("📈" if result.gates.get("spike", True) else "❌📈")
            gates_status.append("🔬" if result.gates.get("rmt", True) else "❌🔬")
            gates_status.append("📊" if result.gates.get("quality", True) else "❌📊")
            gates_info = " ".join(gates_status)
        lines.append(
            f"| {result.config.edit} | {result.config.tier} | {result.config.probes} | {status} | {ppl_delta} | {time_delta} | {mem_delta} | {rmt_info} | {gates_info} |"
        )
    lines.extend(
        [
            "",
            "## Legend",
            "",
            "- 🟢 Within threshold",
            "- 🔴 Exceeds threshold",
            "- 📈 Spike gate",
            "- 🔬 RMT gate",
            "- 📊 Quality gate",
            "",
        ]
    )
    return "\n".join(lines)


def scenario_result_to_dict(result: ScenarioResult) -> dict[str, Any]:
    return {
        "edit": result.config.edit,
        "tier": result.config.tier,
        "probes": result.config.probes,
        "probes_used": result.probes_used,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "artifacts": result.artifacts,
        "metrics": result.metrics,
        "gates": result.gates,
        "epsilon_used": result.epsilon_used,
        "bare_success": result.bare_result.success if result.bare_result else False,
        "guarded_success": result.guarded_result.success
        if result.guarded_result
        else False,
    }


def config_to_dict(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "edits": config.edits,
        "tiers": config.tiers,
        "probes": config.probes,
        "profile": config.profile,
        "dataset": config.dataset,
        "model_id": config.model_id,
        "adapter": config.adapter,
        "device": config.device,
        "seq_len": config.seq_len,
        "stride": config.stride,
        "seed": config.seed,
        "epsilon": config.epsilon,
        "ppl_overhead_threshold": config.ppl_overhead_threshold,
        "guard_overhead_time_threshold": config.guard_overhead_time_threshold,
        "guard_overhead_mem_threshold": config.guard_overhead_mem_threshold,
        "catastrophic_spike_threshold": config.catastrophic_spike_threshold,
    }


__all__ = [
    "BenchmarkConfig",
    "BenchmarkSummary",
    "ConfigurationManager",
    "MetricsAggregator",
    "RunResult",
    "ScenarioConfig",
    "ScenarioResult",
    "ValidationGates",
    "config_to_dict",
    "generate_scenarios",
    "generate_step14_markdown",
    "resolve_epsilon_from_runtime",
    "scenario_result_to_dict",
    "summary_to_step14_json",
]
