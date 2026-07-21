#!/usr/bin/env python3
"""Run LM Evaluation Harness and import its per-record output into InvarLock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderPluginIdentity,
    canonical_runtime_behavioral_schedule_json,
    load_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from invarlock.runtime_import_authoring import (
    load_external_scoring_records_jsonl,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

VERSION = "0.4.12+invarlock.nocache.1"
MAX_GENERATION_TOKENS = 1
HARNESS_BATCH_SIZE = 8
HARNESS_SEED = 20_260_716
MINIMUM_SIDE_ACCURACY = 0.20
TASK = "invarlock_exact_match"
DATASET_NAME = "qwen3-0.6b-base-to-post-trained"
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_FIELDS = {
    "format",
    "role",
    "harness_version",
    "task_config",
    "task_config_sha256",
    "execution_config",
    "execution_config_sha256",
    "samples",
    "samples_sha256",
    "model_tree_sha256",
    "dataset_sha256",
    "record_count",
    "stable_id_field",
}
SAMPLE_FIELDS = {
    "doc",
    "target",
    "arguments",
    "filtered_resps",
    "filter",
    "doc_hash",
    "prompt_hash",
    "target_hash",
}


class BridgeError(ValueError):
    """The harness output cannot support verifier replay."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def task_config(dataset: str) -> dict[str, Any]:
    return {
        "task": TASK,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": dataset}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{expected}}",
        "generation_kwargs": {
            "do_sample": False,
            "max_gen_toks": MAX_GENERATION_TOKENS,
            "until": ["\n"],
        },
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
        "metadata": {"version": 1},
    }


def execution_config() -> dict[str, Any]:
    """Return the complete fixed execution profile authenticated by the bridge."""

    return {
        "batch_size": HARNESS_BATCH_SIZE,
        "checkpoint_generation_config": "excluded",
        "device": "cpu",
        "dtype": "float32",
        "harness_backend": "causal",
        "harness_model": "hf",
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "seed": HARNESS_SEED,
        "trust_remote_code": False,
    }


def worker(role: str, model: Path, dataset: Path, output: Path) -> None:
    """Run the real upstream CLI and retain its official samples JSONL."""

    if importlib.metadata.version("lm-eval") != VERSION:
        raise BridgeError(f"the runtime must contain lm-eval {VERSION}")
    if (
        output.exists()
        or output.is_symlink()
        or not model.is_dir()
        or not dataset.is_file()
    ):
        raise BridgeError("worker inputs must exist and output must be new")
    generation_defaults = model / "generation_config.json"
    if generation_defaults.exists() or generation_defaults.is_symlink():
        raise BridgeError(
            "Harness model snapshot must leave generation defaults to the task"
        )
    output.mkdir(parents=True)
    model_tree_sha256 = checkpoint_tree_sha256(model)
    dataset_sha256 = digest(dataset.read_bytes())
    config = task_config(str(dataset))
    config_path = output / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_sha256 = digest(config_path.read_bytes())
    raw = output / "upstream"
    execution = execution_config()
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        str(execution["harness_model"]),
        "--model_args",
        (
            f"pretrained={model},backend={execution['harness_backend']},"
            f"dtype={execution['dtype']},"
            f"trust_remote_code={execution['trust_remote_code']}"
        ),
        "--tasks",
        str(config_path),
        "--device",
        str(execution["device"]),
        "--batch_size",
        str(execution["batch_size"]),
        "--seed",
        str(execution["seed"]),
        "--log_samples",
        "--output_path",
        str(raw),
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode:
        raise BridgeError("LM Evaluation Harness execution failed")
    if (
        checkpoint_tree_sha256(model) != model_tree_sha256
        or digest(dataset.read_bytes()) != dataset_sha256
        or digest(config_path.read_bytes()) != config_sha256
    ):
        raise BridgeError("LM Evaluation Harness inputs changed during execution")
    samples = list(raw.rglob("samples_*.jsonl"))
    if len(samples) != 1:
        raise BridgeError("LM Evaluation Harness did not emit one per-record file")
    destination = output / "samples.jsonl"
    shutil.copy2(samples[0], destination)
    bound = config
    lines = destination.read_bytes().splitlines()
    manifest = {
        "format": "invarlock/lm-evaluation-harness-run-v1",
        "role": role,
        "harness_version": VERSION,
        "task_config": bound,
        "task_config_sha256": digest(canonical_json_bytes(bound)),
        "execution_config": execution,
        "execution_config_sha256": digest(canonical_json_bytes(execution)),
        "samples": destination.name,
        "samples_sha256": digest(destination.read_bytes()),
        "model_tree_sha256": model_tree_sha256,
        "dataset_sha256": dataset_sha256,
        "record_count": len(lines),
        "stable_id_field": "id",
    }
    (output / "run-manifest.json").write_bytes(canonical_json_bytes(manifest))


def load_run(path: Path, role: str) -> tuple[dict[str, Any], Path]:
    try:
        run = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{role} run provenance is missing") from exc
    if not isinstance(run, dict) or set(run) != RUN_FIELDS:
        raise BridgeError(f"{role} run provenance is incomplete")
    if (
        run["format"] != "invarlock/lm-evaluation-harness-run-v1"
        or run["role"] != role
        or run["harness_version"] != VERSION
        or run["stable_id_field"] != "id"
        or IMAGE_ID.fullmatch(run["model_tree_sha256"]) is None
        or IMAGE_ID.fullmatch(f"sha256:{run['dataset_sha256']}") is None
        or run["task_config"] != task_config("/records.jsonl")
        or run["task_config_sha256"] != digest(canonical_json_bytes(run["task_config"]))
        or run["execution_config"] != execution_config()
        or run["execution_config_sha256"]
        != digest(canonical_json_bytes(run["execution_config"]))
    ):
        raise BridgeError(f"{role} run provenance is invalid")
    samples = path.parent / run["samples"]
    if (
        run["samples"] != "samples.jsonl"
        or not samples.is_file()
        or samples.is_symlink()
        or digest(samples.read_bytes()) != run["samples_sha256"]
        or len(samples.read_bytes().splitlines()) != run["record_count"]
    ):
        raise BridgeError(f"{role} per-record output was tampered")
    return cast(dict[str, Any], run), samples


def adapt(samples: Path, schedule: Any, destination: Path) -> None:
    """Map upstream records to the strict ABI; never import aggregate scores."""

    lines = samples.read_bytes().splitlines()
    if len(lines) != len(schedule.records):
        raise BridgeError("one Harness sample is required for every schedule record")
    output: list[dict[str, object]] = []
    for index, (raw, expected) in enumerate(
        zip(lines, schedule.records, strict=True), 1
    ):
        sample = json.loads(raw)
        if not isinstance(sample, dict):
            raise BridgeError(f"sample {index} is not an object")
        if "results" in sample and not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError("aggregate-only Harness results are not accepted")
        if not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError(f"sample {index} lacks per-record facts")
        doc = sample["doc"]
        if not isinstance(doc, dict) or doc.get("id") != expected.record_id:
            raise BridgeError(f"sample {index} lacks a stable, ordered ID")
        arguments = sample["arguments"]
        request = arguments.get("gen_args_0") if isinstance(arguments, dict) else None
        prompt = request.get("arg_0") if isinstance(request, dict) else None
        generation = request.get("arg_1") if isinstance(request, dict) else None
        part = expected.input_parts[0] if len(expected.input_parts) == 1 else None
        if not isinstance(prompt, str) or part is None or prompt != part.text:
            raise BridgeError(f"sample {index} prompt does not match the schedule")
        if generation != task_config("/records.jsonl")["generation_kwargs"]:
            raise BridgeError(
                f"sample {index} generation settings do not match the task"
            )
        target = str(sample["target"])
        doc_bytes = json.dumps(doc, indent=2, default=str, ensure_ascii=False).encode()
        if (
            sample["filter"] != "none"
            or target != expected.expected_output
            or sample["doc_hash"] != digest(doc_bytes)
            or sample["prompt_hash"] != digest(prompt.encode())
            or sample["target_hash"] != digest(target.encode())
        ):
            raise BridgeError(f"sample {index} authenticated inputs were tampered")
        responses = sample["filtered_resps"]
        if (
            not isinstance(responses, list)
            or len(responses) != 1
            or not isinstance(responses[0], str)
        ):
            raise BridgeError(f"sample {index} lacks one model response")
        response = responses[0]
        output.append(
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "status": "ok",
                "output_text": response,
                "output_sha256": digest(response.encode()),
            }
        )
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in output))
    load_external_scoring_records_jsonl(destination, schedule=schedule)


def imported(role: str) -> dict[str, str]:
    root = f"imports/{role}"
    names = {
        "identity": "model-artifact.identity.json",
        "receipt": "runtime-provider.receipt.json",
        "observation": "runtime-scoring.observation.json",
        "run_report": "report.json",
        "runtime_manifest": "runtime.manifest.json",
        "runtime_config": "run.yaml",
    }
    return {key: f"{root}/{name}" for key, name in names.items()}


def validate_completed_outputs(evidence: Path, receipt: Path, report: Path) -> None:
    """Require a passing signed transaction, not merely successful processes."""

    try:
        evaluation_report = json.loads(
            (evidence / "reports/evaluation.report.json").read_bytes()
        )
        verification_receipt = json.loads(receipt.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "the completed transaction is missing verified outputs"
        ) from exc
    if not isinstance(evaluation_report, dict) or not isinstance(
        verification_receipt, dict
    ):
        raise BridgeError("the completed transaction returned invalid outputs")
    statement = verification_receipt.get("statement")
    receipt_verdict = statement.get("verdict") if isinstance(statement, dict) else None
    comparison = evaluation_report.get("comparison")
    baseline = evaluation_report.get("baseline")
    subject = evaluation_report.get("subject")
    if (
        evaluation_report.get("verdict") != "pass"
        or evaluation_report.get("metric") != "exact_match"
        or not isinstance(comparison, dict)
        or isinstance(comparison.get("value"), bool)
        or not isinstance(comparison.get("value"), (int, float))
        or not isinstance(baseline, dict)
        or not isinstance(subject, dict)
        or isinstance(baseline.get("mean_score"), bool)
        or not isinstance(baseline.get("mean_score"), (int, float))
        or isinstance(subject.get("mean_score"), bool)
        or not isinstance(subject.get("mean_score"), (int, float))
        or baseline["mean_score"] < MINIMUM_SIDE_ACCURACY
        or subject["mean_score"] < MINIMUM_SIDE_ACCURACY
        or not isinstance(receipt_verdict, dict)
        or receipt_verdict.get("ok") is not True
        or receipt_verdict.get("integrity_ok") is not True
        or receipt_verdict.get("policy_verdict") != "pass"
        or not report.is_file()
    ):
        raise BridgeError("the completed transaction did not verify a passing result")


def complete(root: Path, prepared: Path, image: str) -> tuple[Path, Path, Path]:
    """Author strict import inputs and execute evaluate, verify, and report."""

    from examples.integrations.run import _write_private_key

    if IMAGE_ID.fullmatch(image) is None:
        raise BridgeError("runtime image must be an immutable local sha256 ID")
    if root.exists() or root.is_symlink():
        raise BridgeError("transaction workspace must be new")
    (root / "inputs").mkdir(parents=True)
    (root / "imports").mkdir()
    (root / "verifier/policy").mkdir(parents=True)
    runs = {
        role: load_run(prepared / f"harness/{role}/run-manifest.json", role)
        for role in ("baseline", "subject")
    }
    if (
        runs["baseline"][0]["task_config_sha256"]
        != runs["subject"][0]["task_config_sha256"]
        or runs["baseline"][0]["execution_config_sha256"]
        != runs["subject"][0]["execution_config_sha256"]
    ):
        raise BridgeError("baseline and subject used different Harness configurations")
    request0 = yaml.safe_load((prepared / "evaluation/request.yaml").read_text())
    comparison0 = request0.get("comparison") if isinstance(request0, dict) else None
    if not isinstance(comparison0, dict) or comparison0.get("metric") != "exact_match":
        raise BridgeError("prepared request is not the fixed exact-match transaction")
    if comparison0.get("policy") != "inputs/acceptance.json":
        raise BridgeError("prepared request has an unexpected policy path")
    dataset0 = prepared / "evaluation/inputs/records.jsonl"
    dataset = comparison0.get("dataset")
    if not isinstance(dataset, dict):
        raise BridgeError("prepared request lacks the authenticated dataset")
    raw_dataset = dataset0.read_bytes()
    expected_dataset = {
        "path": "inputs/records.jsonl",
        "sha256": dataset.get("sha256"),
        "format": "jsonl",
        "name": DATASET_NAME,
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "expected",
        "id_field": "id",
    }
    if dataset != expected_dataset:
        raise BridgeError("prepared request has an unexpected dataset descriptor")
    if dataset["sha256"] != digest(raw_dataset):
        raise BridgeError("prepared dataset does not match the request digest")
    for role in ("baseline", "subject"):
        if runs[role][0]["dataset_sha256"] != dataset["sha256"]:
            raise BridgeError(f"{role} run used a different authenticated dataset")
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset0,
            sha256=digest(raw_dataset),
            format="jsonl",
            name=dataset["name"],
            split=dataset["split"],
            input_field=dataset["input_field"],
            expected_output_field=dataset["expected_output_field"],
            id_field=dataset["id_field"],
        ),
        raw_dataset,
    )
    schedule_path = root / "inputs/schedule.json"
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    schedule = load_runtime_behavioral_schedule(schedule_path)
    if any(not record.expected_output for record in schedule.records):
        raise BridgeError("prepared exact-match targets must be non-empty")
    prepared_policy = prepared / "evaluation/inputs/acceptance.json"
    policy = prepared_policy.read_bytes()
    expected_policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -20.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 102,
                }
            }
        }
    }
    if json.loads(policy) != expected_policy:
        raise BridgeError("prepared exact-match policy is not the fixed example policy")
    (root / "inputs/acceptance.json").write_bytes(policy)
    (root / "verifier/policy/acceptance.json").write_bytes(policy)
    provenance = canonical_json_bytes(
        {
            "format": "invarlock/lm-evaluation-harness-provenance-v1",
            "runtime_image_digest": image,
            "task_config": runs["baseline"][0]["task_config"],
            "task_config_sha256": runs["baseline"][0]["task_config_sha256"],
            "execution_config": runs["baseline"][0]["execution_config"],
            "execution_config_sha256": runs["baseline"][0]["execution_config_sha256"],
            "samples_sha256": {role: runs[role][0]["samples_sha256"] for role in runs},
        }
    )
    (root / "inputs/harness-provenance.json").write_bytes(provenance)
    provider = HFTransformersProvider()
    sides: dict[str, Any] = {}
    anchors: dict[str, str] = {}
    for role in ("baseline", "subject"):
        records_path = root / f"imports/{role}-records.jsonl"
        adapt(runs[role][1], schedule, records_path)
        original = comparison0[role]
        settings = original["runtime"]["settings"]
        spec = ModelRuntimeSpec(
            "hf_transformers", original["artifact"]["model_id"], settings
        )
        checkpoint = prepared / f"evaluation/models/{role}"
        if (checkpoint / "generation_config.json").exists() or (
            checkpoint / "generation_config.json"
        ).is_symlink():
            raise BridgeError(
                f"{role} snapshot does not leave generation defaults to the task"
            )
        identity = provider.authenticate_artifact(spec, checkpoint)
        if runs[role][0]["model_tree_sha256"] != settings.get("checkpoint_tree_sha256"):
            raise BridgeError(f"{role} run used a different authenticated checkpoint")
        execution = runs[role][0]["execution_config"]
        if (
            settings.get("seed") != execution["seed"]
            or settings.get("batch_size") != execution["batch_size"]
            or settings.get("max_output_tokens") != execution["max_generation_tokens"]
        ):
            raise BridgeError(f"{role} runtime settings do not match Harness execution")
        side = write_runtime_import_side(
            root / f"imports/{role}",
            role=cast(Literal["baseline", "subject"], role),
            schedule=schedule,
            policy_digest=sha256_digest(policy),
            artifact_identity=identity,
            records=load_external_scoring_records_jsonl(
                records_path, schedule=schedule
            ),
            plugin=RuntimeProviderPluginIdentity(
                "hf_transformers", "invarlock", INVARLOCK_VERSION
            ),
            backend=RuntimeBackendIdentity(
                "lm-evaluation-harness-hf",
                VERSION,
                digest(provenance),
                None,
                runs[role][0]["task_config_sha256"],
            ),
            capabilities=provider.capabilities(),
            execution_settings=RuntimeExecutionSettings(
                settings["seed"],
                settings["context_length"],
                settings["batch_size"],
                settings["max_output_tokens"],
                settings["timeout_seconds"],
                False,
            ),
            device=RuntimeDeviceFacts(
                str(execution["device"]), f"container-{execution['device']}"
            ),
            runtime_image_ref=image,
            runtime_image_digest=image,
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
        )
        sides[role] = side
        anchors[role] = sha256_digest(side.provider_evidence.artifact_identity_bytes)
    write_runtime_import_paired_records(
        root / "imports/paired-records.json",
        schedule=schedule,
        metric="exact_match",
        baseline=sides["baseline"],
        subject=sides["subject"],
    )

    def side(role: str) -> dict[str, Any]:
        original = comparison0[role]
        return {
            "artifact": {
                key: original["artifact"][key] for key in ("model_id", "locator")
            },
            "runtime": original["runtime"],
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": "inputs/schedule.json",
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {
            "mode": "import",
            "records": "imports/paired-records.json",
            "schedule": "inputs/schedule.json",
            "baseline": imported("baseline"),
            "subject": imported("subject"),
        },
        "observations": [
            {
                "id": "lm-evaluation-harness-provenance",
                "kind": "harness_provenance",
                "scope": "comparison",
                "path": "inputs/harness-provenance.json",
            }
        ],
        "output": {"evidence": "evidence"},
    }
    request_path = root / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False))
    evidence_key = root / "keys/evidence.pem"
    verifier_key = root / "verifier/keys/verifier.pem"
    evidence_key.parent.mkdir()
    verifier_key.parent.mkdir()
    evidence_fingerprint = _write_private_key(evidence_key)
    _write_private_key(verifier_key)
    trust = {
        "format": "invarlock/trust-inputs-v1",
        "policy": {"path": "policy/acceptance.json"},
        "anchors": {
            "baseline_artifact_digest": anchors["baseline"],
            "subject_artifact_digest": anchors["subject"],
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": image,
            "subject_runtime_digest": image,
            "evidence_signer_fingerprint": evidence_fingerprint,
        },
        "verifier": {
            "identity": "invarlock-example/lm-evaluation-harness-verifier",
            "signing_key_path": "keys/verifier.pem",
        },
        "allow_installed_scorers": False,
    }
    trust_path = root / "verifier/trusted-inputs.json"
    trust_path.write_bytes(canonical_json_bytes(trust))
    evidence, receipt, report = (
        root / "evidence",
        root / "verifier/verification.receipt.json",
        root / "comparison-report.html",
    )
    commands = [
        ["evaluate", str(request_path), "--signing-key", str(evidence_key), "--json"],
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(trust_path),
            "--receipt",
            str(receipt),
            "--json",
        ],
        ["report", str(evidence), "--html", str(report)],
    ]
    for arguments in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "invarlock", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise BridgeError(completed.stderr or completed.stdout)
    validate_completed_outputs(evidence, receipt, report)
    return evidence, receipt, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    worker_parser.add_argument("--model", type=Path, required=True)
    worker_parser.add_argument("--dataset", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    bridge_parser = subparsers.add_parser("complete")
    bridge_parser.add_argument("--workspace", type=Path, required=True)
    bridge_parser.add_argument("--prepared", type=Path, required=True)
    bridge_parser.add_argument("--runtime-image", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            worker(args.role, args.model, args.dataset, args.output)
        else:
            evidence, receipt, report = complete(
                args.workspace.resolve(), args.prepared.resolve(), args.runtime_image
            )
            print(f"Evidence: {evidence}\nReceipt: {receipt}\nReport: {report}")
    except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
