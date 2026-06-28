#!/usr/bin/env python3
"""Run the real training evidence campaign.

This maintainer smoke wraps the existing PEFT LoRA and full fine-tune
integration examples. Generated checkpoints and raw runner outputs remain local
by default; the script writes public summaries and hash inventories that can be
checked before any public evidence is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "invarlock.training_evidence_campaign.summary.v1"
HASH_INVENTORY_SCHEMA = "invarlock.training_evidence_campaign.hash_inventory.v1"
CLAIM_BOUNDARY = "empirical training evidence only; no new assurance claim"
DEFAULT_TARGETS = ("peft_lora", "fine_tune")
PUBLISHABLE_ARTIFACTS = {
    "evaluation_report": "evaluation.report.json",
    "runtime_manifest": "runtime.manifest.json",
    "verify_json": "verify.json",
    "checkpoint_refs": "checkpoint_refs.json",
    "external_edit_summary": "external_edit_summary.json",
    "fixture_summary": "fixture_summary.json",
    "lane_artifact": "lane_artifact.json",
    "run_summary": "run_summary.txt",
}
HTML_ARTIFACT = {"evaluation_html": "evaluation.html"}


@dataclass(frozen=True)
class TargetConfig:
    target: str
    edit_family: str
    display_name: str
    runner: Path
    report_dir: str
    subject_dir: str
    fixture_dir: str
    toolchain: str


TARGETS = {
    "peft_lora": TargetConfig(
        target="peft_lora",
        edit_family="lora_merge",
        display_name="PEFT LoRA train-and-merge subject",
        runner=Path("examples/integrations/peft_lora/run_tiny_peft_lora.sh"),
        report_dir="tiny-peft-lora",
        subject_dir="tiny-gpt2-peft-lora-merged",
        fixture_dir="tiny-peft-lora-fixture",
        toolchain="peft",
    ),
    "fine_tune": TargetConfig(
        target="fine_tune",
        edit_family="fine_tune",
        display_name="Full fine-tune subject",
        runner=Path("examples/integrations/fine_tune/run_tiny_fine_tune.sh"),
        report_dir="tiny-fine-tune",
        subject_dir="tiny-gpt2-fine-tuned",
        fixture_dir="tiny-fine-tune-fixture",
        toolchain="transformers",
    ),
}


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} did not contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _repo_relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _campaign_relative(path: Path, work_root: Path) -> str:
    try:
        return path.resolve().relative_to(work_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _public_artifact_record(
    *,
    path: Path,
    work_root: Path,
    artifact_name: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "artifact": artifact_name,
        "bytes": path.stat().st_size,
        "campaign_relative_path": _campaign_relative(path, work_root),
        "sha256": _sha256(path),
    }
    repo_relative = _repo_relative(path)
    if repo_relative is not None:
        record["repo_relative_path"] = repo_relative
    return record


def _parse_run_summary(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.is_file():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep and key in {
            "status",
            "lane_artifact_label",
            "execution_mode",
            "assurance",
            "runtime_provenance",
            "device",
            "verify_status",
            "verify_reason",
            "verify_runtime_provenance_declared",
            "verify_runtime_provenance_status",
            "verify_runtime_provenance_verified",
        }:
            fields[key] = value.strip()
    return fields


def _artifact_paths(report_dir: Path, *, render_html: bool) -> dict[str, Path]:
    artifact_names = dict(PUBLISHABLE_ARTIFACTS)
    if render_html:
        artifact_names.update(HTML_ARTIFACT)
    return {name: report_dir / filename for name, filename in artifact_names.items()}


def _missing_artifacts(report_dir: Path, *, render_html: bool) -> list[str]:
    return [
        name
        for name, path in _artifact_paths(report_dir, render_html=render_html).items()
        if not path.is_file()
    ]


def _lane_paths(
    *,
    config: TargetConfig,
    work_root: Path,
) -> dict[str, Path]:
    target_root = work_root / config.target
    return {
        "target_root": target_root,
        "subject_dir": target_root / "models" / config.subject_dir,
        "fixture_dir": target_root / "fixtures" / config.fixture_dir,
        "report_dir": target_root / "reports" / config.report_dir,
    }


def _command_for_target(
    *,
    config: TargetConfig,
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(config.runner),
        "--subject-dir",
        str(paths["subject_dir"]),
        "--fixture-dir",
        str(paths["fixture_dir"]),
        "--report-out",
        str(paths["report_dir"]),
        "--profile",
        args.profile,
        "--tier",
        args.tier,
        "--lane",
        args.execution_lane,
    ]
    if args.execution_lane == "host":
        command.extend(["--device", args.device])
    if args.allow_network:
        command.append("--allow-network")
    if args.force:
        command.append("--force")
    if not args.render_html:
        command.append("--no-html")
    return command


def _public_command_shape(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "execution_lane": args.execution_lane,
        "profile": args.profile,
        "tier": args.tier,
        "weights_vendored": False,
    }
    if args.execution_lane == "host":
        payload["device"] = args.device
    if args.execution_lane == "cuda":
        payload["validation_host"] = "CUDA-capable validation host"
    if args.allow_network:
        payload["network"] = "enabled by operator flag"
    else:
        payload["network"] = "disabled by default"
    return payload


def _planned_lane_payload(
    *,
    config: TargetConfig,
    paths: dict[str, Path],
    args: argparse.Namespace,
    work_root: Path,
) -> dict[str, Any]:
    report_repo_relative = _repo_relative(paths["report_dir"])
    payload: dict[str, Any] = {
        "target": config.target,
        "edit_family": config.edit_family,
        "display_name": config.display_name,
        "status": "planned",
        "toolchain": config.toolchain,
        "weights_vendored": False,
        "publishable_artifact_names": sorted(PUBLISHABLE_ARTIFACTS.values()),
        "command_shape": _public_command_shape(args),
        "campaign_relative_report_dir": _campaign_relative(
            paths["report_dir"], work_root
        ),
    }
    if args.render_html:
        payload["publishable_artifact_names"].append("evaluation.html")
        payload["publishable_artifact_names"].sort()
    if report_repo_relative is not None:
        payload["repo_relative_report_dir"] = report_repo_relative
    return payload


def _completed_lane_payload(
    *,
    config: TargetConfig,
    paths: dict[str, Path],
    args: argparse.Namespace,
    work_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_dir = paths["report_dir"]
    missing = _missing_artifacts(report_dir, render_html=args.render_html)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"{config.target} did not produce required artifact(s): {missing_list}"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for artifact_name, path in _artifact_paths(
        report_dir, render_html=args.render_html
    ).items():
        record = _public_artifact_record(
            path=path, work_root=work_root, artifact_name=artifact_name
        )
        artifacts[artifact_name] = record
        inventory.append({"target": config.target, **record})

    run_summary = _parse_run_summary(report_dir / "run_summary.txt")
    lane_artifact = _read_json(report_dir / "lane_artifact.json")
    payload: dict[str, Any] = {
        "target": config.target,
        "edit_family": config.edit_family,
        "display_name": config.display_name,
        "status": "completed",
        "toolchain": config.toolchain,
        "weights_vendored": False,
        "subject_checkpoint_policy": (
            "materialized for validation; do not commit checkpoint weights by default"
        ),
        "command_shape": _public_command_shape(args),
        "lane_artifact_label": lane_artifact.get("lane_artifact_label"),
        "verification": {
            "assurance": run_summary.get("assurance"),
            "profile": args.profile,
            "runtime_provenance": run_summary.get("runtime_provenance"),
            "runtime_provenance_status": run_summary.get(
                "verify_runtime_provenance_status"
            ),
            "runtime_provenance_verified": run_summary.get(
                "verify_runtime_provenance_verified"
            ),
            "verify_status": run_summary.get("verify_status"),
        },
        "artifacts": artifacts,
    }
    return payload, inventory


def _build_summary(
    *,
    campaign_id: str,
    status: str,
    work_root: Path,
    lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": campaign_id,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "weights_vendored": False,
        "public_artifact_policy": (
            "publish public summaries, report manifests, checkpoint references, "
            "evaluation reports, verification JSON, and hash inventories only"
        ),
        "lanes": lanes,
    }
    repo_relative = _repo_relative(work_root)
    if repo_relative is not None:
        payload["repo_relative_work_root"] = repo_relative
    return payload


def _build_inventory(
    *,
    campaign_id: str,
    status: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": HASH_INVENTORY_SCHEMA,
        "campaign_id": campaign_id,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "weights_vendored": False,
        "artifacts": sorted(
            artifacts,
            key=lambda item: (str(item.get("target")), str(item.get("artifact"))),
        ),
    }


def _publish_summaries(
    *,
    summary: dict[str, Any],
    inventory: dict[str, Any],
    publish_summary: Path,
) -> None:
    _write_json(publish_summary / "campaign_summary.json", summary)
    _write_json(publish_summary / "hash_inventory.json", inventory)


def _selected_targets(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_TARGETS)
    targets: list[str] = []
    for value in values:
        for item in value.split(","):
            target = item.strip()
            if not target:
                continue
            if target not in TARGETS:
                raise ValueError(
                    f"unknown target {target!r}; expected one of {sorted(TARGETS)}"
                )
            if target not in targets:
                targets.append(target)
    return targets


def run_campaign(args: argparse.Namespace) -> int:
    campaign_id = args.campaign_id or f"training-evidence-{_timestamp()}"
    work_root = Path(
        args.work_root or Path("reports") / "training-evidence-campaign" / campaign_id
    )
    work_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_targets(args.target)
    lanes: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    status = "planned" if args.dry_run else "completed"

    for target in selected:
        config = TARGETS[target]
        paths = _lane_paths(config=config, work_root=work_root)
        command = _command_for_target(config=config, paths=paths, args=args)
        if args.dry_run:
            lanes.append(
                _planned_lane_payload(
                    config=config, paths=paths, args=args, work_root=work_root
                )
            )
            print("DRY-RUN:", " ".join(command))
            continue

        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            status = "failed"
            lanes.append(
                {
                    **_planned_lane_payload(
                        config=config, paths=paths, args=args, work_root=work_root
                    ),
                    "status": "failed",
                    "exit_code": result.returncode,
                }
            )
            break
        lane_payload, lane_inventory = _completed_lane_payload(
            config=config, paths=paths, args=args, work_root=work_root
        )
        lanes.append(lane_payload)
        inventory_records.extend(lane_inventory)

    summary = _build_summary(
        campaign_id=campaign_id,
        status=status,
        work_root=work_root,
        lanes=lanes,
    )
    inventory = _build_inventory(
        campaign_id=campaign_id,
        status=status,
        artifacts=inventory_records,
    )
    _write_json(work_root / "campaign_summary.json", summary)
    _write_json(work_root / "hash_inventory.json", inventory)

    if args.publish_summary:
        _publish_summaries(
            summary=summary,
            inventory=inventory,
            publish_summary=Path(args.publish_summary),
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status in {"planned", "completed"} else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real PEFT LoRA and full fine-tune training evidence campaign "
            "using tiny public-safe integration lanes."
        )
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Target to run: peft_lora, fine_tune, or a comma-separated list. "
            "Defaults to both targets."
        ),
    )
    parser.add_argument(
        "--work-root",
        help=(
            "Campaign output root. Defaults to reports/training-evidence-campaign/"
            "<campaign-id>."
        ),
    )
    parser.add_argument(
        "--campaign-id",
        help="Stable campaign ID. Defaults to training-evidence-<UTC timestamp>.",
    )
    parser.add_argument(
        "--execution-lane",
        choices=("host", "cuda"),
        default="host",
        help=(
            "Integration lane to run. host defaults to CPU/off assurance for local "
            "smoke; cuda runs the strict CUDA/container lane."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Host-lane device. Ignored for --execution-lane cuda. Default: cpu.",
    )
    parser.add_argument("--profile", default="release")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--render-html",
        action="store_true",
        help="Retain HTML rendering in target runners. Skipped by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a planned campaign summary without launching runners.",
    )
    parser.add_argument(
        "--publish-summary",
        help=(
            "Optional directory for public campaign_summary.json and "
            "hash_inventory.json. Check these files before committing them."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        return run_campaign(args)
    except Exception as exc:
        print(f"[training-evidence-campaign] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
