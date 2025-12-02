"""
Model capability probes (lightweight, no heavy imports).

Currently supports a simple MoE probe that scans module names to infer whether
the model likely contains mixture-of-experts structures (router/gating and
expert FFNs).
"""

from __future__ import annotations

from typing import Any


def probe_moe_capabilities(model: Any) -> dict[str, Any]:
    """Detect MoE-related modules in a model by name heuristics.

    Returns:
        {
          'moe': bool,
          'families': set[str],           # e.g., {'router','expert_ffn'}
          'counts': dict[str,int],        # occurrences per family
        }
    """
    families: set[str] = set()
    counts: dict[str, int] = {}
    try:
        for name, _module in model.named_modules():
            lname = str(name).lower()
            # Router/gating indicators
            if any(
                tok in lname
                for tok in ("router", "routing", "gate", "gating", "dispatch", "switch")
            ):
                families.add("router")
                counts["router"] = counts.get("router", 0) + 1
            # Expert FFN indicators
            if any(
                tok in lname
                for tok in ("experts", "expert", "moe", "mixture_of_experts")
            ):
                families.add("expert_ffn")
                counts["expert_ffn"] = counts.get("expert_ffn", 0) + 1
    except Exception:
        pass
    return {"moe": bool(families), "families": families, "counts": counts}
