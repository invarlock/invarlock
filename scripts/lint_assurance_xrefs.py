#!/usr/bin/env python3
"""Assurance cross-reference linter.

Validates that:
- `docs/assurance/*.md` references existing pytest tests via `tests/...::...`
- `docs/assurance/*.md` cites report field paths that exist (against a
  representative report sample).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

TEST_REF_RE = re.compile(
    r"(tests/[A-Za-z0-9_./-]+\.py(?:::[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*)?)"
)
FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
PATHISH_RE = re.compile(r"^[A-Za-z0-9_.{}\[\],*]+$")

FIELD_ROOTS = {
    "artifacts",
    "auto",
    "confidence",
    "dataset",
    "edit",
    "guard_overhead",
    "invariants",
    "meta",
    "policies",
    "policy_digest",
    "policy_provenance",
    "primary_metric",
    "provenance",
    "resolved_policy",
    "rmt",
    "spectral",
    "structure",
    "system_overhead",
    "telemetry",
    "validation",
    "variance",
    # Legacy root (intentionally absent from cert output): lints drift when cited.
    "ppl",
}


@dataclass(frozen=True)
class LintError:
    path: Path
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


def _strip_fenced_code_blocks(text: str) -> str:
    return FENCED_CODE_BLOCK_RE.sub("", text)


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_braces(expr: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", expr)
    if not match:
        return [expr]
    options = _split_top_level_commas(match.group(1))
    expanded: list[str] = []
    for opt in options:
        candidate = expr[: match.start()] + opt.strip() + expr[match.end() :]
        expanded.extend(_expand_braces(candidate))
    return expanded


def _looks_like_field_path(expr: str) -> bool:
    expr = expr.strip()
    if not expr:
        return False
    if re.search(r"\s", expr):
        return False
    if not PATHISH_RE.match(expr):
        return False
    if "[" in expr:
        # Only accept wildcard indexing `[*]` (avoid pseudo-expressions like `[family]`).
        tmp = expr.replace("[*]", "")
        if "[" in tmp or "]" in tmp:
            return False
    root = re.split(r"[.{]", expr, maxsplit=1)[0].strip()
    return root in FIELD_ROOTS


def _extract_field_expressions(text: str) -> set[str]:
    cleaned = _strip_fenced_code_blocks(text)
    expressions: set[str] = set()
    for code_span in INLINE_CODE_RE.findall(cleaned):
        for candidate in _split_top_level_commas(code_span):
            c = candidate.strip().strip(".,;:()")
            if _looks_like_field_path(c):
                expressions.add(c)
    return expressions


def _parse_py(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_pytest_node(tree: ast.AST, node_path: list[str]) -> bool:
    body: list[ast.stmt] = getattr(tree, "body", [])
    if not node_path:
        return False

    def _is_func(node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        )

    if len(node_path) == 1:
        name = node_path[0]
        # Accept top-level function or class method with matching name
        for node in body:
            if _is_func(node, name):
                return True
            if isinstance(node, ast.ClassDef):
                if any(_is_func(n, name) for n in node.body):
                    return True
        return False

    cur_body = body
    for cls_name in node_path[:-1]:
        cls = next(
            (n for n in cur_body if isinstance(n, ast.ClassDef) and n.name == cls_name),
            None,
        )
        if cls is None:
            return False
        cur_body = cls.body
    return any(_is_func(n, node_path[-1]) for n in cur_body)


def _path_exists_in_obj(obj: Any, path: str) -> bool:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return False

    def _step(cur: Any, idx: int) -> bool:
        if idx >= len(parts):
            return True
        part = parts[idx]
        if part == "*":
            if isinstance(cur, dict):
                if not cur:
                    return True
                return any(_step(v, idx + 1) for v in cur.values())
            if isinstance(cur, list):
                if not cur:
                    return True
                return any(_step(v, idx + 1) for v in cur)
            return False
        wildcard = part.endswith("[*]")
        key = part[:-3] if wildcard else part

        if not isinstance(cur, dict) or key not in cur:
            return False
        nxt = cur[key]
        if not wildcard:
            return _step(nxt, idx + 1)

        if isinstance(nxt, dict):
            if not nxt:
                # Container exists but empty; accept the structural claim.
                return True
            return any(_step(v, idx + 1) for v in nxt.values())
        if isinstance(nxt, list):
            if not nxt:
                return True
            return any(_step(v, idx + 1) for v in nxt)
        return False

    return _step(obj, 0)


def _sample_reports() -> list[dict[str, Any]]:
    # Keep this sample self-contained. Docs CI intentionally installs a narrow
    # docs-only dependency surface, so the linter must not rely on report-build
    # modules that import runtime stacks such as numpy/torch.
    ppl_report = {
        "schema_version": "v1",
        "run_id": "sample-ppl",
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "now",
            "tokenizer_hash": "tok",
            "seeds": {"python": 1, "numpy": None, "torch": None},
            "env_flags": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "determinism": {"requested": "strict", "level": "strict"},
        },
        "auto": {"tier": "balanced", "policy_digest": "auto-digest"},
        "dataset": {
            "name": "ds",
            "split": "validation",
            "windows": {
                "stats": {
                    "paired_windows": 2,
                    "requested_preview": 2,
                    "requested_final": 2,
                    "actual_preview": 2,
                    "actual_final": 2,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "bootstrap": {
                        "replicates": 200,
                        "alpha": 0.05,
                        "method": "percentile",
                        "seed": 1,
                    },
                    "paired_delta_summary": {
                        "mean": -0.01,
                        "std": 0.02,
                        "degenerate": False,
                    },
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                }
            },
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "artifacts": {
            "report_path": "reports/eval/evaluation.report.json",
            "events_path": "",
            "logs_path": "",
            "checkpoint_path": None,
        },
        "guard_overhead": {
            "bare_ppl": 10.0,
            "guarded_ppl": 10.1,
            "overhead_ratio": 1.01,
            "overhead_percent": 1.0,
            "overhead_threshold": 0.01,
            "evaluated": True,
            "skipped": False,
            "diagnostics": {"mode": "sample"},
        },
        "invariants": {"stable": True, "status": "pass"},
        "policy_digest": {"thresholds_hash": "thresholds", "changed": False},
        "policy_provenance": {
            "policy_digest": "policy-digest",
            "overrides": [],
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.5,
            "ratio_vs_baseline": 1.101,
            "display_ci": [1.01, 1.09],
            "ci": [1.01, 1.09],
            "reps": 200,
        },
        "provenance": {
            "baseline": {
                "model_id": "baseline",
                "report_path": "reports/baseline/evaluation.report.json",
            },
            "edited": {
                "model_id": "edited",
                "report_path": "reports/edited/evaluation.report.json",
            },
            "env_flags": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "provider_digest": {"ids_sha256": "provider-sha"},
        },
        "resolved_policy": {
            "metrics": {
                "pm_ratio": {
                    "ratio_limit_base": 1.10,
                    "hysteresis_ratio": 0.002,
                },
                "accuracy": {"min_examples_fraction": 0.01},
            },
            "rmt": {
                "margin": 1.5,
                "deadband": 0.10,
                "epsilon_by_family": {"ffn": 0.10},
            },
            "spectral": {"max_caps": 5},
            "variance": {
                "min_effect_lognll": 0.0009,
                "predictive_one_sided": True,
                "max_adjusted_modules": 1,
            },
        },
        "rmt": {
            "edge_risk_by_family": {"ffn": 0.1},
            "edge_risk_by_family_base": {"ffn": 0.05},
            "epsilon_by_family": {"ffn": 0.10},
            "epsilon_default": 0.10,
            "epsilon_violations": [],
            "families": [
                {
                    "edge_base": 0.01,
                    "edge_cur": 0.02,
                    "epsilon": 0.10,
                    "allowed": 0.11,
                    "ratio": 1.05,
                    "delta": 0.01,
                }
            ],
            "measurement_contract_hash": "rmt-hash",
            "stable": True,
            "status": "ok",
            "mode": "warn",
            "max_edge_ratio": 1.05,
            "max_edge_delta": 0.01,
        },
        "spectral": {
            "caps_applied": 0,
            "caps_exceeded": False,
            "families": [{"kappa": 3.0, "violations": []}],
            "family_caps": {"ffn": {"kappa": 3.0}},
            "max_caps": 5,
            "measurement_contract_hash": "spectral-hash",
            "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            "sigma_quantile": 0.99,
            "summary": {
                "deadband": 0.10,
                "sigma_quantile": 0.99,
                "modules_checked": 1,
                "max_caps": 5,
                "caps_exceeded": False,
            },
            "top_z_scores": {"ffn": [{"module": "mlp.c_proj", "z": 2.3}]},
        },
        "telemetry": {"latency_ms_per_tok": 1.0},
        "validation": {
            "guard_overhead_acceptable": True,
            "hysteresis_applied": False,
            "invariants_pass": True,
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "primary_metric_tail_acceptable": True,
            "rmt_stable": True,
            "spectral_stable": True,
        },
        "variance": {
            "enabled": False,
            "gain": 0.0,
            "scope": "ffn",
            "target_modules": 1,
            "proposed_scales": {"layer": 1.0},
            "ab_test": {
                "seed": 1,
                "windows_used": 2,
                "provenance": {"window_ids": [1, 2]},
            },
            "predictive_gate": {
                "delta_ci": [-0.01, 0.01],
                "mean_delta": -0.001,
                "reason": "ci_contains_zero",
                "passed": False,
            },
        },
        "structure": {},
        "system_overhead": {},
        "policies": {},
        "confidence": {},
    }
    acc_report = {
        "schema_version": "v1",
        "run_id": "sample-acc",
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu"},
        "dataset": {"name": "ds"},
        "edit": {"name": "noop"},
        "artifacts": {"report_path": "reports/eval/evaluation.report.json"},
        "primary_metric": {
            "kind": "accuracy",
            "preview": 0.79,
            "final": 0.80,
            "display_ci": [0.78, 0.82],
        },
        "resolved_policy": {"metrics": {"accuracy": {"min_examples_fraction": 0.01}}},
        "validation": {"primary_metric_acceptable": True},
    }

    return [ppl_report, acc_report]


def main() -> None:
    errors: list[LintError] = []
    docs = sorted(Path("docs/assurance").glob("*.md"))
    if not docs:
        print("[lint_assurance_xrefs] No docs found; skipping.")
        raise SystemExit(0)

    # ---- test xrefs ----
    trees: dict[Path, ast.AST] = {}
    for md_path in docs:
        text = md_path.read_text(encoding="utf-8")
        for ref in TEST_REF_RE.findall(_strip_fenced_code_blocks(text)):
            file_part, *node_parts = ref.split("::")
            py_path = Path(file_part)
            if not py_path.exists():
                errors.append(LintError(md_path, f"Missing test file: `{ref}`"))
                continue
            if py_path not in trees:
                try:
                    trees[py_path] = _parse_py(py_path)
                except SyntaxError as e:
                    errors.append(
                        LintError(md_path, f"Cannot parse `{py_path}`: {e.msg}")
                    )
                    continue
            if node_parts and not _has_pytest_node(trees[py_path], node_parts):
                errors.append(LintError(md_path, f"Missing test: `{ref}`"))

    # ---- field-path xrefs ----
    try:
        certs = _sample_reports()
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as e:  # pragma: no cover
        errors.append(
            LintError(Path("docs/assurance"), f"Failed to build sample report: {e}")
        )
        certs = []

    field_exprs: set[str] = set()
    for md_path in docs:
        field_exprs |= _extract_field_expressions(md_path.read_text(encoding="utf-8"))

    for expr in sorted(field_exprs):
        expanded = _expand_braces(expr)
        for path in expanded:
            if not any(_path_exists_in_obj(cert, path) for cert in certs):
                errors.append(
                    LintError(
                        Path("docs/assurance"),
                        f"Missing report field path: `{path}` (from `{expr}`)",
                    )
                )

    if errors:
        print("[lint_assurance_xrefs] FAIL", file=sys.stderr)
        for err in errors:
            print(err.format(), file=sys.stderr)
        raise SystemExit(1)

    print("[lint_assurance_xrefs] OK")


if __name__ == "__main__":
    main()
