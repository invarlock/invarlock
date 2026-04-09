#!/usr/bin/env python3
"""Plan proof-pack CI window schedules using effective post-dedupe token counts."""

from __future__ import annotations

import argparse
import json

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.eval.data import get_provider
from invarlock.eval.window_planning import choose_first_token_sufficient_candidate
from invarlock.model_profile import detect_model_profile, resolve_tokenizer


def _parse_candidate(value: str) -> dict[str, int]:
    parts = [segment.strip() for segment in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must be formatted as seq_len:preview_n:final_n"
        )
    try:
        seq_len, preview_n, final_n = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate values must be integers") from exc
    if seq_len <= 0 or preview_n <= 0 or final_n <= 0:
        raise argparse.ArgumentTypeError("candidate values must be positive")
    return {
        "seq_len": seq_len,
        "stride": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
    }


def _resolve_min_tokens_target(tier: str, profile: str | None) -> int:
    resolved = resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        return int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-provider", default="wikitext2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--profile", default="ci")
    parser.add_argument("--headroom-ratio", type=float, default=1.05)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=[],
        help="Candidate schedule as seq_len:preview_n:final_n",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    profile = detect_model_profile(args.model_path, adapter="hf_auto")
    tokenizer, _ = resolve_tokenizer(profile)
    provider = get_provider(args.dataset_provider, device_hint="cpu")

    result = choose_first_token_sufficient_candidate(
        data_provider=provider,
        tokenizer=tokenizer,
        split=args.split,
        seed=int(args.seed),
        candidates=args.candidate,
        min_tokens_target=_resolve_min_tokens_target(args.tier, args.profile),
        headroom_ratio=float(args.headroom_ratio),
        profile=args.profile,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
