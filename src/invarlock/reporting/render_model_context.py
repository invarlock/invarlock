from __future__ import annotations

from typing import Any

import yaml

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def append_model_context_sections(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    lines.append("## Model Information")
    lines.append("")
    meta = evaluation_report.get("meta", {}) or {}
    lines.append(f"- **Model ID:** {meta.get('model_id')}")
    lines.append(f"- **Adapter:** {meta.get('adapter')}")
    lines.append(f"- **Device:** {meta.get('device')}")
    lines.append(f"- **Timestamp:** {meta.get('ts')}")
    commit_value = meta.get("commit") or ""
    if commit_value:
        short_sha = str(commit_value)[:12]
        lines.append(f"- **Commit:** {short_sha}")
    else:
        lines.append("- **Commit:** (not set)")
    lines.append(f"- **Seed:** {meta.get('seed')}")
    seeds_map = meta.get("seeds", {})
    if isinstance(seeds_map, dict) and seeds_map:
        lines.append(
            "- **Seeds:** "
            f"python={seeds_map.get('python')}, "
            f"numpy={seeds_map.get('numpy')}, "
            f"torch={seeds_map.get('torch')}"
        )
    invarlock_version = meta.get("invarlock_version")
    if invarlock_version:
        lines.append(f"- **InvarLock Version:** {invarlock_version}")
    env_flags = meta.get("env_flags")
    cuda_flags = meta.get("cuda_flags")

    det_parts: list[str] = []
    for label, keys in (
        ("torch_det", ("torch_deterministic_algorithms", "deterministic_algorithms")),
        ("cudnn_det", ("cudnn_deterministic",)),
        ("cudnn_bench", ("cudnn_benchmark",)),
        ("tf32_matmul", ("cuda_matmul_allow_tf32",)),
        ("tf32_cudnn", ("cudnn_allow_tf32",)),
        ("cublas_ws", ("CUBLAS_WORKSPACE_CONFIG",)),
    ):
        val = None
        for key in keys:
            if isinstance(env_flags, dict) and env_flags.get(key) is not None:
                val = env_flags.get(key)
                break
            if isinstance(cuda_flags, dict) and cuda_flags.get(key) is not None:
                val = cuda_flags.get(key)
                break
        if val is not None:
            det_parts.append(f"{label}={val}")
    if det_parts:
        lines.append(f"- **Determinism:** {', '.join(det_parts)}")

    full_flags: dict[str, Any] = {}
    if isinstance(env_flags, dict) and env_flags:
        full_flags["env_flags"] = env_flags
    if isinstance(cuda_flags, dict) and cuda_flags:
        full_flags["cuda_flags"] = cuda_flags
    if full_flags:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Environment flags (full)</summary>")
        lines.append("")
        lines.append("```yaml")
        flags_yaml = yaml.safe_dump(full_flags, sort_keys=True, width=80).strip()
        for line in flags_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
    lines.append("")

    auto = evaluation_report.get("auto", {}) or {}
    auto_tier = auto.get("tier")
    if auto_tier and auto_tier != "none":
        lines.append("## Auto-Tuning Configuration")
        lines.append("")
        lines.append(f"- **Tier:** {auto_tier}")
        lines.append(f"- **Probes Used:** {auto.get('probes_used', 0)}")
    if auto.get("target_pm_ratio"):
        lines.append(
            f"- **Auto Policy Target Ratio (informational):** {auto['target_pm_ratio']:.3f}"
        )
        try:
            if bool(auto.get("tiny_relax")):
                lines.append("- Tiny relax: enabled (dev-only)")
        except _PARSE_EXCEPTIONS:
            pass
        lines.append("")
