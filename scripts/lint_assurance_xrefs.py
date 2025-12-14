#!/usr/bin/env python3
"""Assurance cross-reference linter.

Validates that:
- `docs/assurance/*.md` references existing pytest tests via `tests/...::...`
- `docs/assurance/*.md` cites certificate field paths that exist (against a
  representative certificate sample).
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEST_REF_RE = re.compile(
    r"(tests/[A-Za-z0-9_./-]+\.py::[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*)"
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


def _sample_certificates() -> list[dict[str, Any]]:
    from invarlock.reporting.certificate import make_certificate

    ppl_report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "now",
            "auto": {"enabled": True, "tier": "balanced"},
            "tokenizer_hash": "tok",
            "seeds": {"python": 1, "numpy": None, "torch": None},
        },
        "data": {
            "dataset": "ds",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [
            {
                "name": "spectral",
                "policy": {
                    "deadband": 0.10,
                    "max_caps": 5,
                    "family_caps": {"ffn": {"kappa": 3.0}},
                    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                },
                "metrics": {
                    "caps_applied": 0,
                    "caps_exceeded": False,
                    "modules_checked": 1,
                },
            },
            {
                "name": "rmt",
                "policy": {
                    "deadband": 0.10,
                    "margin": 1.5,
                    "epsilon_default": 0.10,
                    "epsilon_by_family": {"ffn": 0.10},
                },
                "metrics": {"outliers": 0, "stable": True},
            },
            {
                "name": "variance",
                "policy": {
                    "deadband": 0.02,
                    "min_abs_adjust": 0.012,
                    "max_scale_step": 0.03,
                    "min_effect_lognll": 0.0009,
                    "predictive_one_sided": True,
                    "topk_backstop": 1,
                    "max_adjusted_modules": 1,
                },
                "metrics": {
                    "ve_enabled": False,
                    "scope": "ffn",
                    "target_modules": 1,
                    "predictive_gate": {
                        "evaluated": True,
                        "passed": False,
                        "reason": "ci_contains_zero",
                        "delta_ci": (-0.01, 0.01),
                        "mean_delta": -0.001,
                    },
                    "proposed_scales": {"layer": 1.0},
                    "ab_seed_used": 1,
                    "ab_windows_used": 2,
                    "ab_provenance": {"condition_a": {"window_ids": [1, 2]}},
                },
            },
        ],
        "guard_overhead": {
            "bare_final": 10.0,
            "guarded_final": 10.1,
            "overhead_threshold": 0.01,
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.5,
                "ratio_vs_baseline": 1.05,
                "display_ci": [1.01, 1.09],
                "reps": 200,
                "ci_level": 0.95,
            },
            "logloss_delta_ci": (-0.02, 0.01),
            "paired_delta_summary": {"mean": -0.01, "std": 0.02, "degenerate": False},
            "bootstrap": {
                "replicates": 200,
                "alpha": 0.05,
                "method": "percentile",
                "seed": 1,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.1, 0.2],
                "token_counts": [1, 1],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    ppl_baseline = {
        "meta": {"model_id": "m", "adapter": "hf", "seed": 1},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.1, 0.2],
                "token_counts": [1, 1],
            }
        },
    }

    acc_report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "now",
            "auto": {"enabled": True, "tier": "balanced"},
        },
        "data": {
            "dataset": "ds",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.8,
                "display_ci": [0.78, 0.82],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    acc_baseline = {"metrics": {"primary_metric": {"kind": "accuracy", "final": 0.79}}}

    return [
        make_certificate(ppl_report, ppl_baseline),
        make_certificate(acc_report, acc_baseline),
    ]


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
            if not _has_pytest_node(trees[py_path], node_parts):
                errors.append(LintError(md_path, f"Missing test: `{ref}`"))

    # ---- field-path xrefs ----
    try:
        certs = _sample_certificates()
    except Exception as e:  # pragma: no cover
        errors.append(
            LintError(
                Path("docs/assurance"), f"Failed to build sample certificate: {e}"
            )
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
                        f"Missing certificate field path: `{path}` (from `{expr}`)",
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
