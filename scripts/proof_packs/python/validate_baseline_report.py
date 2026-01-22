from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4:
        print(
            "Usage: validate_baseline_report.py <baseline_report.json> <expected_adapter> <expected_profile> <expected_tier>",
            file=sys.stderr,
        )
        return 2

    report_path = Path(argv[0])
    expected_adapter = argv[1]
    expected_profile = argv[2]
    expected_tier = argv[3]

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
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
    if isinstance(adapter, str) and adapter != expected_adapter:
        print(
            f"baseline_report_adapter_mismatch:{adapter!r}!={expected_adapter!r}",
            file=sys.stderr,
        )
        return 1

    context = payload.get("context")
    if isinstance(context, dict):
        prof = context.get("profile")
        if (
            isinstance(prof, str)
            and prof.strip().lower() != expected_profile.strip().lower()
        ):
            print(
                f"baseline_report_profile_mismatch:{prof!r}!={expected_profile!r}",
                file=sys.stderr,
            )
            return 1
        auto = context.get("auto")
        if isinstance(auto, dict):
            tier = auto.get("tier")
            if isinstance(tier, str) and tier != expected_tier:
                print(
                    f"baseline_report_tier_mismatch:{tier!r}!={expected_tier!r}",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
