"""Adapt frozen K2 captures to the resource-aware campaign runner."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from examples.qualification import k2_campaign as campaign
from examples.qualification import k2_producer as capture_worker


def checked_snapshot(plan, role, snapshot):
    campaign.require_ready(plan)
    measured = campaign.measure_snapshot(snapshot, plan["model"][role]["files"])
    if measured != plan["model"][role]["materialized"]:
        raise ValueError("actual snapshot differs from frozen materialization")
    return campaign.digest(measured)


def prepare(plan, role, snapshot, output, phase, preflight=None):
    identity = checked_snapshot(plan, role, snapshot)
    if phase == "decision":
        if preflight is None:
            raise ValueError("decision requires its retained preflight")
        if campaign.validate_capture(plan, preflight, phase="preflight") != role:
            raise ValueError("preflight role differs")
        summary = capture_worker.preflight_summary(preflight)
        if (
            not summary["complete"]
            or any(
                not capture_worker.campaign._answer(row)[0].strip()
                for row in preflight["rows"]
            )
            or summary["estimated_decision_seconds"]
            > plan["budget"]["maximum_wall_seconds"]
        ):
            raise ValueError(
                "preflight is incomplete, empty, or exceeds the frozen budget"
            )
    receipt = {
        "plan_digest": campaign.digest(plan),
        "role": role,
        "materialization_digest": identity,
        "phase": phase,
    }
    campaign.write_json(output / "prepared.json", receipt)
    return receipt


def capture(plan, role, phase, prepared, output):
    campaign.require_ready(plan)
    expected = {
        "plan_digest": campaign.digest(plan),
        "role": role,
        "materialization_digest": campaign.digest(plan["model"][role]["materialized"]),
        "phase": phase,
    }
    if prepared != expected:
        raise ValueError("prepared snapshot receipt differs")
    capture_worker.worker(plan, role, phase, output)


def validate(
    plan, role, snapshot, prepared, captured, output, observed_job_id, workload_key
):
    identity = checked_snapshot(plan, role, snapshot)
    phase = captured["phase"]
    if prepared != {
        "plan_digest": campaign.digest(plan),
        "role": role,
        "materialization_digest": identity,
        "phase": phase,
    }:
        raise ValueError("post-capture snapshot differs from preparation")
    if campaign.validate_capture(plan, captured, phase=phase) != role:
        raise ValueError("capture role differs")
    capture_worker.validate_hardware(
        captured["hardware"], plan["runtime"]["tensor_parallel"]
    )
    answers = [campaign._answer(row) for row in captured["rows"]]
    complete = bool(answers) and all(error is None for _, error in answers)
    nonempty = sum(bool(answer.strip()) for answer, error in answers if error is None)
    report = {
        "format": "invarlock/k2-scheduled-validation-v1",
        "plan_digest": campaign.digest(plan),
        "capture_digest": campaign.digest(captured),
        "role": role,
        "phase": phase,
        "rows": len(answers),
        "complete": complete,
        "nonempty_final_answers": nonempty,
        "semantic_ready": complete and nonempty == len(answers),
        "resources": captured["resources"],
        "limit": "Protocol readiness and materialized identity; no model-quality acceptance.",
    }
    campaign.write_json(output / "verified.json", report)
    campaign.write_json(
        output / "sentinel.json",
        {
            "format": "invarlock/campaign-sentinel-v1",
            "observed_job_id": observed_job_id,
            "workload_key": workload_key,
            "units": len(answers),
            "fixed_seconds": captured["resources"]["startup_seconds"],
            "complete": complete,
            "semantic_ready": report["semantic_ready"],
        },
    )
    return report


def make_manifest(
    plan_path, snapshots, host, result_root, worker_source, *, exclusive=False
):
    """Build a preflight DAG; never change the historical plan or image."""
    from examples.qualification.campaign_scheduling import validate_manifest

    plan = campaign.read_json(plan_path)
    campaign.require_ready(plan)
    worker_hash = "sha256:" + hashlib.sha256(worker_source.read_bytes()).hexdigest()
    plan_hash = "sha256:" + hashlib.sha256(plan_path.read_bytes()).hexdigest()
    model_slug = plan["model"]["id"].replace(".", "-")
    jobs = []
    for role in campaign.ROLES:
        prefix = model_slug + "-" + role
        prepare_id, capture_id, validate_id = (
            prefix + "-" + name for name in ("prepare", "capture", "validate")
        )
        common = [
            {
                "source": str(plan_path.resolve()),
                "target": "/plan.json",
                "sha256": plan_hash,
            },
            {
                "source": str(worker_source.resolve()),
                "target": "/opt/campaign/campaign_adapter.py",
                "sha256": worker_hash,
            },
        ]
        model_mount = {
            "source": str(snapshots[role].resolve()),
            "target": "/models/" + role,
        }
        key = campaign.digest(
            {
                "runtime": plan["runtime"],
                "materialization": plan["model"][role]["materialized"],
                "protocol": campaign.digest(plan["preflight_cases"]),
                "adapter": worker_hash,
                "co_execution": "exclusive" if exclusive else "independent-device-jobs",
            }
        )
        for stage, job_id, deps in (
            ("prepare", prepare_id, []),
            ("capture", capture_id, [prepare_id]),
            ("validate", validate_id, [capture_id]),
        ):
            mounts = [*common, model_mount]
            command = [
                "python",
                "/opt/campaign/campaign_adapter.py",
                stage,
                "--plan",
                "/plan.json",
                "--role",
                role,
                "--phase",
                "preflight",
                "--snapshot",
                "/models/" + role,
                "--output",
                "/output",
            ]
            if stage != "prepare":
                mounts.append(
                    {
                        "source": str(result_root / "jobs" / prepare_id / "output"),
                        "target": "/prepared",
                    }
                )
                command.extend(["--prepared", "/prepared/prepared.json"])
            if stage == "validate":
                mounts.append(
                    {
                        "source": str(result_root / "jobs" / capture_id / "output"),
                        "target": "/captured",
                    }
                )
                command.extend(
                    [
                        "--capture",
                        "/captured/capture.json",
                        "--observed-job-id",
                        capture_id,
                        "--workload-key",
                        key,
                    ]
                )
            gpu = plan["runtime"]["tensor_parallel"] if stage == "capture" else 0
            jobs.append(
                {
                    "id": job_id,
                    "depends_on": deps,
                    "resources": {
                        "gpus": gpu,
                        "cpus": 16,
                        "memory_mib": (196608 if gpu == 2 else 65536) if gpu else 16384,
                        "io_slots": 1,
                        "exclusive": exclusive if gpu else False,
                    },
                    "timeout_seconds": min(plan["budget"]["maximum_wall_seconds"], 900)
                    if gpu
                    else 600,
                    "estimate_seconds": 180 if gpu else 120,
                    "workload": {
                        "key": key if gpu else key + ":" + stage,
                        "units": len(plan["preflight_cases"]) if gpu else 1,
                        "fixed_seconds": 60 if gpu else 0,
                    },
                    "container": {
                        "image": plan["runtime"]["image_digest"],
                        "command": command,
                        "mounts": mounts,
                        "environment": {},
                    },
                    "outputs": {
                        "prepare": ["prepared.json"],
                        "capture": ["capture.json", "server.log"],
                        "validate": ["verified.json", "sentinel.json"],
                    }[stage],
                }
            )
    return validate_manifest(
        {
            "format": "invarlock/campaign-schedule-v1",
            "id": model_slug + "-sentinels",
            "host": host,
            "jobs": jobs,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "capture", "validate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--role", choices=campaign.ROLES, required=True)
    parser.add_argument("--phase", choices=("preflight", "decision"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--observed-job-id")
    parser.add_argument("--workload-key")
    args = parser.parse_args(argv)
    plan = campaign.read_json(args.plan)
    if args.command == "prepare":
        prepare(
            plan,
            args.role,
            args.snapshot,
            args.output,
            args.phase,
            campaign.read_json(args.preflight) if args.preflight else None,
        )
    elif args.command == "capture":
        if args.prepared is None:
            parser.error("capture requires --prepared")
        capture(
            plan, args.role, args.phase, campaign.read_json(args.prepared), args.output
        )
    else:
        if None in (
            args.prepared,
            args.capture,
            args.observed_job_id,
            args.workload_key,
        ):
            parser.error(
                "validate requires prepared receipt, capture and observation attribution"
            )
        validate(
            plan,
            args.role,
            args.snapshot,
            campaign.read_json(args.prepared),
            campaign.read_json(args.capture),
            args.output,
            args.observed_job_id,
            args.workload_key,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
