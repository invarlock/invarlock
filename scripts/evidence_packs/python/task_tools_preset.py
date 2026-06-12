from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_yaml_module() -> Any:
    try:
        import yaml
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - evidence-pack preflight enforces PyYAML
        raise SystemExit(
            "PyYAML is required to normalize staged evidence-pack presets"
        ) from exc
    return yaml


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalized_preset_payload(
    payload: Any,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> dict[str, Any]:
    normalized = _mapping(payload)

    dataset = _mapping(normalized.get("dataset"))
    dataset["seq_len"] = seq_len
    dataset["stride"] = stride
    dataset["preview_n"] = preview_n
    dataset["final_n"] = final_n
    normalized["dataset"] = dataset

    if skip_overhead_check:
        context = _mapping(normalized.get("context"))
        run = _mapping(context.get("run"))
        run["skip_overhead_check"] = True
        context["run"] = run
        normalized["context"] = context

    return normalized


def normalize_staged_preset(
    preset_path: Path,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> None:
    yaml = _load_yaml_module()
    raw = preset_path.read_text()
    loaded = yaml.safe_load(raw) if raw.strip() else {}
    normalized = _normalized_preset_payload(
        loaded,
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=skip_overhead_check,
    )
    preset_path.write_text(yaml.safe_dump(normalized, sort_keys=False))


def _coerce_required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(
            f"baseline report is missing numeric data.{key} needed to normalize preset"
        )
    return int(value)


def schedule_from_baseline_report(report_path: Path) -> tuple[int, int, int, int]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load baseline report JSON: {report_path}") from exc

    source: dict[str, Any] = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            source = data
        else:
            dataset = payload.get("dataset")
            if isinstance(dataset, dict):
                source = dataset
    if not source:
        raise SystemExit(
            f"baseline report does not contain data/dataset schedule fields: {report_path}"
        )

    return (
        _coerce_required_int(source, "seq_len"),
        _coerce_required_int(source, "stride"),
        _coerce_required_int(source, "preview_n"),
        _coerce_required_int(source, "final_n"),
    )


def _normalize_staged_preset(args: argparse.Namespace) -> int:
    if args.baseline_report:
        seq_len, stride, preview_n, final_n = schedule_from_baseline_report(
            Path(args.baseline_report)
        )
    else:
        values = (args.seq_len, args.stride, args.preview_n, args.final_n)
        if any(value is None for value in values):
            raise SystemExit(
                "either --baseline-report or all of "
                "--seq-len/--stride/--preview-n/--final-n is required"
            )
        seq_len, stride, preview_n, final_n = (
            int(args.seq_len),
            int(args.stride),
            int(args.preview_n),
            int(args.final_n),
        )
    normalize_staged_preset(
        Path(args.preset),
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=bool(args.skip_overhead_check),
    )
    return 0


def _validate_baseline_report(args: argparse.Namespace) -> int:
    report_path = Path(args.baseline_report)
    expected_adapter = str(args.expected_adapter)
    expected_profile = str(args.expected_profile)
    expected_tier = str(args.expected_tier)
    expected_assurance = str(getattr(args, "expected_assurance", "off"))
    expected_preview_n = getattr(args, "expected_preview_n", None)
    expected_final_n = getattr(args, "expected_final_n", None)

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"baseline_report_invalid_json:{exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print("baseline_report_not_object", file=sys.stderr)
        return 1

    edit = payload.get("edit")
    edit_name = edit.get("name") if isinstance(edit, dict) else None
    if edit_name != "noop":
        print(f"baseline_report_edit_not_noop:{edit_name!r}", file=sys.stderr)
        return 1

    meta = payload.get("meta")
    adapter = meta.get("adapter") if isinstance(meta, dict) else None
    if not isinstance(adapter, str) or not adapter:
        print("baseline_report_missing_adapter", file=sys.stderr)
        return 1
    if adapter != expected_adapter:
        print(
            f"baseline_report_adapter_mismatch:{adapter!r}!={expected_adapter!r}",
            file=sys.stderr,
        )
        return 1

    context = payload.get("context")
    if not isinstance(context, dict):
        print("baseline_report_missing_context", file=sys.stderr)
        return 1
    profile = context.get("profile")
    if not isinstance(profile, str) or not profile:
        print("baseline_report_missing_profile", file=sys.stderr)
        return 1
    if profile.strip().lower() != expected_profile.strip().lower():
        print(
            f"baseline_report_profile_mismatch:{profile!r}!={expected_profile!r}",
            file=sys.stderr,
        )
        return 1
    auto = context.get("auto")
    if not isinstance(auto, dict):
        print("baseline_report_missing_auto", file=sys.stderr)
        return 1
    tier = auto.get("tier")
    if not isinstance(tier, str) or not tier:
        print("baseline_report_missing_tier", file=sys.stderr)
        return 1
    if tier != expected_tier:
        print(
            f"baseline_report_tier_mismatch:{tier!r}!={expected_tier!r}",
            file=sys.stderr,
        )
        return 1
    context_assurance = context.get("assurance")
    assurance = (
        context_assurance.get("mode") if isinstance(context_assurance, dict) else None
    )
    if not isinstance(assurance, str) or not assurance:
        report_assurance = payload.get("assurance")
        assurance = (
            report_assurance.get("mode") if isinstance(report_assurance, dict) else None
        )
    if not isinstance(assurance, str) or not assurance:
        print("baseline_report_missing_assurance", file=sys.stderr)
        return 1
    if assurance.strip().lower() != expected_assurance.strip().lower():
        print(
            f"baseline_report_assurance_mismatch:{assurance!r}!={expected_assurance!r}",
            file=sys.stderr,
        )
        return 1

    windows = payload.get("evaluation_windows")
    if not isinstance(windows, dict):
        print("baseline_report_missing_evaluation_windows", file=sys.stderr)
        return 1

    for phase_name in ("preview", "final"):
        phase = windows.get(phase_name)
        if not isinstance(phase, dict):
            print(f"baseline_report_missing_phase:{phase_name}", file=sys.stderr)
            return 1
        window_ids = phase.get("window_ids")
        input_ids = phase.get("input_ids")
        if not isinstance(window_ids, list) or not window_ids:
            print(f"baseline_report_missing_window_ids:{phase_name}", file=sys.stderr)
            return 1
        if not isinstance(input_ids, list) or not input_ids:
            print(f"baseline_report_missing_input_ids:{phase_name}", file=sys.stderr)
            return 1
        if len(window_ids) != len(input_ids):
            print(f"baseline_report_mismatched_windows:{phase_name}", file=sys.stderr)
            return 1
        expected_count = (
            expected_preview_n if phase_name == "preview" else expected_final_n
        )
        if expected_count is not None and len(window_ids) != expected_count:
            print(
                "baseline_report_window_count_mismatch:"
                f"{phase_name}:{len(window_ids)}!={expected_count}",
                file=sys.stderr,
            )
            return 1

    return 0


def stamp_baseline_report_seed(report_path: Path, *, seed: int) -> None:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load baseline report JSON: {report_path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"baseline report must be a JSON object: {report_path}")
    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}
        payload["data"] = data
    if data.get("seed") is None:
        data["seed"] = int(seed)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _stamp_baseline_report_seed(args: argparse.Namespace) -> int:
    stamp_baseline_report_seed(Path(args.report), seed=int(args.seed))
    return 0


def _parse_window_candidate(value: str) -> dict[str, int]:
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
    from invarlock.core.auto_tuning import resolve_tier_policies

    resolved = resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        return int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _plan_effective_windows(args: argparse.Namespace) -> int:
    from invarlock.eval.data import get_provider
    from invarlock.eval.window_planning import choose_first_token_sufficient_candidate
    from invarlock.model_profile import detect_model_profile, resolve_tokenizer

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
