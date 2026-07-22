#!/usr/bin/env python3
"""Run a real TensorRT-LLM engine comparison through InvarLock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint

_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_DEVICE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")
_REQUEST = "invarlock-tensorrt-example-request.yaml"
_EVIDENCE = "invarlock-tensorrt-example-evidence"


def _image(image: str) -> tuple[str, str]:
    if _DIGEST.fullmatch(image):
        return image, image
    repository, separator, digest = image.rpartition("@")
    if repository and separator and _DIGEST.fullmatch(digest):
        return image, digest
    raise ValueError("--runtime-image must be an immutable sha256 image reference")


def _root(value: Path) -> Path:
    root = value.expanduser().resolve(strict=True)
    if root != value.expanduser().absolute() or not root.is_dir():
        raise ValueError("resource root must be a non-symlink directory")
    for name in (
        "baseline-engine",
        "subject-engine",
        "tokenizer-contract.json",
        "records.jsonl",
        "policy.json",
    ):
        path = root / name
        expected = path.is_dir() if name.endswith("-engine") else path.is_file()
        if path.is_symlink() or not expected:
            raise ValueError(f"required input is missing or unsafe: {name}")
    if (root / "records.jsonl").stat().st_size > 64 * 1024 * 1024:
        raise ValueError("records.jsonl exceeds 64 MiB")
    if any(character in str(root) for character in (",", "\n", "\r")):
        raise ValueError("resource root cannot be represented in a Docker mount")
    for name in (_REQUEST, _EVIDENCE):
        if os.path.lexists(root / name):
            raise FileExistsError(
                f"generated destination already exists: {root / name}"
            )
    return root


def _key(path: Path) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return public_key_fingerprint(key.public_key())


def _inspect(root: Path, image: str, digest: str, device: str) -> dict[str, Any]:
    helper = Path(__file__).with_name("engine_inspect.py").resolve(strict=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--gpus",
        "device=" + device.split(":", 1)[1],
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--user",
        "65532:65532",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=8g",
        "--env",
        "HOME=/tmp",
        "--env",
        "USER=65532",
        "--env",
        "LOGNAME=65532",
        "--env",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "--env",
        f"INVARLOCK_RUNTIME_IMAGE={image}",
        "--env",
        f"INVARLOCK_RUNTIME_IMAGE_DIGEST={digest}",
        "--mount",
        f"type=bind,src={root},dst=/resources,readonly",
        "--mount",
        f"type=bind,src={helper},dst=/example/engine_inspect.py,readonly",
        "--entrypoint",
        "/opt/invarlock/cli-venv/bin/python",
        image,
        "/example/engine_inspect.py",
        "--resource-root",
        "/resources",
        "--context-length",
        "1024",
        "--max-output-tokens",
        "1",
        "--timeout-seconds",
        "300",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "TensorRT-LLM engine inspection failed"
            + (f": {diagnostic}" if diagnostic else "")
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("engine inspection did not return JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"baseline", "subject"}:
        raise ValueError("engine inspection returned an unexpected schema")
    for role in ("baseline", "subject"):
        side = payload[role]
        if (
            not isinstance(side, dict)
            or set(side) != {"artifact_identity_sha256", "model_id", "settings"}
            or not _DIGEST.fullmatch(str(side["artifact_identity_sha256"]))
            or not isinstance(side["settings"], dict)
        ):
            raise ValueError(f"engine inspection returned an invalid {role} side")
    if (
        payload["baseline"]["artifact_identity_sha256"]
        == payload["subject"]["artifact_identity_sha256"]
    ):
        raise ValueError("baseline and subject engines must be distinct")
    return payload


def _prepare(
    root: Path,
    output: Path,
    inspection: dict[str, Any],
    digest: str,
    locators: tuple[str, str],
) -> dict[str, Path]:
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    verifier = output / "verifier"
    for directory in (output / "keys", verifier / "keys", verifier / "policy"):
        directory.mkdir(parents=True, mode=0o700)
    paths = {
        "request": root / _REQUEST,
        "evidence": root / _EVIDENCE,
        "signer": output / "keys/evidence-signer.pem",
        "trust": verifier / "trusted-inputs.json",
        "verifier": verifier / "keys/verifier.pem",
        "receipt": verifier / "verification.receipt.json",
        "report": output / "comparison-report.html",
    }
    records = (root / "records.jsonl").read_bytes()
    dataset = LocalDatasetRequest(
        path=root / "records.jsonl",
        sha256=hashlib.sha256(records).hexdigest(),
        name="tensorrt-llm-engine-comparison",
        split="validation",
        input_field="prompt",
        expected_output_field="expected",
        id_field="id",
    )
    schedule = prepare_local_evaluation_schedule_bytes(
        dataset, records, task="text_causal"
    )
    policy = (root / "policy.json").read_bytes()
    if not isinstance(json.loads(policy), dict):
        raise ValueError("policy.json must contain a JSON object")
    (verifier / "policy/acceptance.json").write_bytes(policy)

    def side(role: str, locator: str) -> dict[str, object]:
        return {
            "artifact": {
                "path": f"{role}-engine",
                "model_id": inspection[role]["model_id"],
                "locator": locator,
            },
            "runtime": {
                "provider": "tensorrt_llm",
                "settings": inspection[role]["settings"],
            },
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline", locators[0]),
            "subject": side("subject", locators[1]),
            "dataset": {
                "path": "records.jsonl",
                "sha256": dataset.sha256,
                "format": "jsonl",
                "name": dataset.name,
                "split": dataset.split,
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": "policy.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "output": {"evidence": _EVIDENCE},
    }
    paths["request"].write_text(
        yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
    )
    signer = _key(paths["signer"])
    _key(paths["verifier"])
    anchors = {
        f"{role}_artifact_digest": inspection[role]["artifact_identity_sha256"]
        for role in ("baseline", "subject")
    }
    anchors.update(
        {
            "schedule_digest": "sha256:" + schedule.schedule_sha256,
            "baseline_runtime_digest": digest,
            "subject_runtime_digest": digest,
            "evidence_signer_fingerprint": signer,
        }
    )
    paths["trust"].write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy/acceptance.json"},
                "anchors": anchors,
                "verifier": {
                    "identity": "invarlock-example/tensorrt-llm-verifier",
                    "signing_key_path": "keys/verifier.pem",
                },
                "allow_installed_scorers": False,
            }
        )
    )
    return paths


def _execute(
    root: Path,
    paths: dict[str, Path],
    image: str,
    digest: str,
    devices: tuple[str, str],
) -> None:
    environment = dict(os.environ)
    environment["INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT"] = str(root)
    environment["INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT"] = "tokenizer-contract.json"
    base = [sys.executable, "-m", "invarlock"]
    evaluate = [
        *base,
        "evaluate",
        str(paths["request"]),
        "--signing-key",
        str(paths["signer"]),
        "--runtime-image",
        image,
        "--runtime-image-digest",
        digest,
        "--baseline-runtime-device",
        devices[0],
        "--subject-runtime-device",
        devices[1],
        "--json",
    ]
    commands = [
        [*evaluate, "--preflight"],
        evaluate,
        [
            *base,
            "verify",
            str(paths["evidence"]),
            "--trust-profile",
            str(paths["trust"]),
            "--receipt",
            str(paths["receipt"]),
            "--json",
        ],
        [*base, "report", str(paths["evidence"]), "--html", str(paths["report"])],
    ]
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    try:
        report = json.loads(
            (paths["evidence"] / "reports/evaluation.report.json").read_text(
                encoding="utf-8"
            )
        )
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "the completed transaction is missing verified outputs"
        ) from exc
    if not isinstance(report, dict) or not isinstance(receipt, dict):
        raise ValueError("the completed transaction returned invalid outputs")
    statement = receipt.get("statement") if isinstance(receipt, dict) else None
    receipt_verdict = statement.get("verdict") if isinstance(statement, dict) else None
    if report.get("verdict") != "pass":
        raise ValueError("the authenticated policy verdict is not pass")
    if (
        not isinstance(receipt_verdict, dict)
        or receipt_verdict.get("ok") is not True
        or receipt_verdict.get("integrity_ok") is not True
        or receipt_verdict.get("policy_verdict") != "pass"
        or not paths["report"].is_file()
        or paths["report"].stat().st_size == 0
    ):
        raise ValueError("the completed transaction did not verify a passing result")
    for role in ("baseline", "subject"):
        side = report.get(role)
        mean_score = side.get("mean_score") if isinstance(side, dict) else None
        if (
            isinstance(mean_score, bool)
            or not isinstance(mean_score, (int, float))
            or mean_score < 0.4
        ):
            raise ValueError(
                f"the {role} engine solved fewer than 40% of the maintained "
                "causal-cloze records"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--baseline-locator", required=True)
    parser.add_argument("--subject-locator", required=True)
    parser.add_argument("--baseline-device", default="cuda:0")
    parser.add_argument("--subject-device", default="cuda:1")
    arguments = parser.parse_args(argv)
    try:
        devices = (arguments.baseline_device, arguments.subject_device)
        if any(_DEVICE.fullmatch(device) is None for device in devices):
            raise ValueError("devices must use cuda:<nonnegative-index>")
        root = _root(arguments.resource_root)
        image, digest = _image(arguments.runtime_image)
        output = root.parent / f"{root.name}-invarlock-output"
        inspection = _inspect(root, image, digest, devices[0])
        paths = _prepare(
            root,
            output,
            inspection,
            digest,
            (arguments.baseline_locator, arguments.subject_locator),
        )
        _execute(root, paths, image, digest, devices)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"PASS evidence={paths['evidence']} receipt={paths['receipt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
