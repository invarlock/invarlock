"""Pack actual evaluated models and check an independent local recipient."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from examples.integrations.bounded_command import run_bounded_command
from invarlock.core.checkpoint_identity import checkpoint_tree_observation

KIT_COMMIT = "6b8162ae5da4d46f1d2af2beb43e7fb077f052f4"
HELPER = Path(__file__).with_name("modelkit_handoff.py")


def read_json(path: Path):
    return json.loads(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()


def command(
    argv: list[str], log: Path, cwd: Path, *, expected: int | tuple[int, ...] = 0
) -> str:
    started = time.monotonic()
    result = run_bounded_command(
        argv,
        cwd=cwd,
        capture_output=True,
        timeout_seconds=3600,
        environment={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    log.with_suffix(".stdout").write_text(result.stdout)
    log.with_suffix(".stderr").write_text(result.stderr)
    write_json(
        log.with_suffix(".command.json"),
        {
            "argv": argv,
            "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    allowed = (expected,) if isinstance(expected, int) else expected
    if result.returncode not in allowed:
        raise RuntimeError(
            f"{log.name}: expected exit {expected}, got exit {result.returncode}"
        )
    return result.stdout


def check_kit(binary: Path, expected: str, workspace: Path) -> str:
    if digest(binary) != "sha256:" + expected.removeprefix("sha256:"):
        raise ValueError("KitOps executable checksum mismatch")
    version = command([str(binary), "version"], workspace / "kit-version", workspace)
    if "Version: 1.15.0" not in version or KIT_COMMIT not in version:
        raise ValueError("requires KitOps 1.15.0 at the reviewed source revision")
    return version


def transfer(source: Path, destination: Path) -> None:
    """Copy actual bytes, with no publisher links in the recipient store."""
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("transfer source must not contain a symlink")
    shutil.copytree(source, destination)


def _selected_source(root: Path, value: object, *, directory: bool) -> Path:
    """Resolve a caller-selected input without letting relative paths leave its request."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("selected input path must be a nonempty path")
    selected = Path(value)
    if ".." in selected.parts:
        raise ValueError("selected input path must not traverse parent directories")
    source = selected if selected.is_absolute() else root / selected
    if not selected.is_absolute():
        component = root
        for part in selected.parts:
            component /= part
            if component.is_symlink():
                raise ValueError("selected input must not traverse a symlink")
    if not selected.is_absolute() and not source.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("relative input path must stay beside the request")
    if source.is_symlink() or (
        not source.is_dir() if directory else not source.is_file()
    ):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"selected input must be a {kind} without a symlink")
    return source


def _selected_keys(root: Path, value: object) -> dict[str, Path]:
    if not isinstance(value, dict) or not value:
        raise ValueError("trusted public keys must be a nonempty mapping")
    selected = {}
    for fingerprint, path in value.items():
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", fingerprint
        ):
            raise ValueError("trusted public key fingerprint must be a SHA-256 digest")
        selected[fingerprint] = _selected_source(root, path, directory=False)
    return selected


def inference_smoke(
    request: Path,
    decision: dict,
    image: str,
    binary: str,
    output: Path,
    *,
    prompt: str,
    tokens: int,
    allow_policy_rejection: bool = False,
    container_engine: str = "docker",
) -> dict:
    """Run the accepted file inside an immutable, offline CPU runtime image."""
    if not decision.get("accepted") and not (
        allow_policy_rejection
        and decision.get("technical_integrity_ok")
        and decision.get("envelope_authenticated")
        and decision.get("receipt_authenticated")
        and decision.get("envelope_evidence_bound")
        and decision.get("exit_code") == 1
    ):
        raise ValueError(
            "inference requires acceptance or explicit isolated usability permission"
        )
    if container_engine not in ("docker", "podman"):
        raise ValueError("container engine must be docker or podman")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("inference requires an immutable local image ID")
    side = read_json(request)["sides"]["subject"]
    if not side.get("artifact_file", "").endswith(".gguf"):
        raise ValueError("this bounded inference smoke requires a GGUF subject")
    model_dir = (request.parent / side["candidate"]).absolute()
    model = model_dir / side["artifact_file"]
    before = checkpoint_tree_observation(model_dir)
    if digest(model) != side["content_digest"]:
        raise ValueError("accepted model changed before inference")
    details = json.loads(
        command(
            [container_engine, "image", "inspect", image],
            output / "inference-image",
            output,
        )
    )
    if (
        not isinstance(details, list)
        or len(details) != 1
        or not isinstance(details[0], dict)
    ):
        raise ValueError("inference image inspection has an invalid identity")
    inspected_id = details[0].get("Id")
    # Podman may return a bare config digest. Require its complete SHA-256,
    # using the same normalization as the evaluator's OCI image inspection.
    if isinstance(inspected_id, str) and re.fullmatch(r"[0-9a-f]{64}", inspected_id):
        inspected_id = "sha256:" + inspected_id
    if inspected_id != image:
        raise ValueError("inference image does not match its selected identity")
    invocation = [
        container_engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        "--memory",
        "12g",
        "--cpus",
        "4",
        "--pids-limit",
        "128",
        "--mount",
        f"type=bind,source={model_dir},target=/model,readonly",
        "--entrypoint",
        binary,
        image,
    ]
    version = command([*invocation, "--version"], output / "llama-version", output)
    text = command(
        [
            *invocation,
            "--model",
            "/model/" + side["artifact_file"],
            "--prompt",
            prompt,
            "--n-predict",
            str(tokens),
            "--ctx-size",
            "256",
            "--threads",
            "4",
            "--seed",
            "0",
            "--temp",
            "0",
            "--n-gpu-layers",
            "0",
            "--no-display-prompt",
            "--no-warmup",
        ],
        output / "inference",
        output,
    )
    if not text.strip():
        raise RuntimeError("inference returned no generated text")
    if checkpoint_tree_observation(model_dir) != before:
        raise ValueError("accepted model changed during inference")
    return {
        "artifact_digest": side["content_digest"],
        "generated_text": text,
        "prompt": prompt,
        "max_new_tokens": tokens,
        "backend_version": version,
        "image_id": image,
        "container_engine": container_engine,
        "device": "cpu",
        "recipient_accepted": bool(decision.get("accepted")),
        "network": "none",
        "scope": "Loadability and bounded generation, separate from evaluation runtime qualification",
    }


def run(args) -> dict:
    workspace = args.output.absolute()
    workspace.mkdir(parents=True, exist_ok=False)
    workspace.chmod(0o700)
    logs = workspace / "logs"
    logs.mkdir()
    kit = args.kit.resolve()
    original = read_json(args.request)
    source_root = args.request.absolute().parent
    inputs = {
        "evidence": _selected_source(source_root, original["evidence"], directory=True),
        "technical_policy": _selected_source(
            source_root, original["technical_policy"], directory=False
        ),
        "envelope": _selected_source(
            source_root, original["envelope"], directory=False
        ),
        "recipient_policy": _selected_source(
            source_root, original["recipient_policy"], directory=False
        ),
    }
    keys = _selected_keys(source_root, original["trusted_public_keys"])
    candidates = {
        role: _selected_source(
            source_root, original["sides"][role]["candidate"], directory=True
        )
        for role in ("baseline", "subject")
    }
    from examples.integrations.modelkit_handoff import _relative

    selected_files = {}
    for role in ("baseline", "subject"):
        selected = original["sides"][role].get("artifact_file")
        if selected is not None:
            member = _relative(selected)
            if member.suffix != ".gguf":
                raise ValueError("artifact_file must select a GGUF file")
            selected_files[role] = member
    check_kit(kit, args.kit_sha256, logs)
    recipient = workspace / "recipient"
    recipient.mkdir()
    publisher = workspace / "publisher-store"
    transferred = recipient / "kit-store"
    helper = recipient / "modelkit_handoff.py"
    shutil.copy2(HELPER, helper)
    request = copy.deepcopy(original)
    # Technical choices come from the caller's independent input, never the pack.
    for key in ("evidence", "technical_policy", "envelope", "recipient_policy"):
        source = inputs[key]
        target = recipient / key
        if source.is_dir():
            transfer(source, target)
        else:
            shutil.copy2(source, target)
        request[key] = key
    request["trusted_public_keys"] = {}
    (recipient / "trust").mkdir()
    for fingerprint, source in keys.items():
        name = "trust/" + fingerprint.removeprefix("sha256:") + ".pem"
        shutil.copy2(source, recipient / name)
        request["trusted_public_keys"][fingerprint] = name
    packages = {}

    def kit_command(actor: Path, label: str, *values: str) -> str:
        return command(
            [str(kit), "--config", str(actor), "--progress", "none", *values],
            logs / label,
            workspace,
        )

    for role in ("baseline", "subject"):
        side = original["sides"][role]
        source = candidates[role]
        observation = checkpoint_tree_observation(source.absolute())
        member = selected_files.get(role)
        if member is not None:
            actual = digest(source.joinpath(*member.parts))
        else:
            actual = observation.digest
        if actual != side["content_digest"]:
            raise ValueError(
                f"{role} model differs from independently selected content"
            )
        context = workspace / (role + "-context")
        context.mkdir()
        transfer(source, context / "model")
        tag = f"example.invalid/modelkit/{role}:review"
        digests = []
        for label in ("original", "repacked"):
            (context / "Kitfile").write_text(
                "manifestVersion: 1.0.0\npackage:\n  name: " + role + "\n"
                "  description: " + label + " evaluated model\nmodel:\n  path: model\n"
            )
            kit_command(
                publisher,
                f"{role}-{label}-pack",
                "pack",
                str(context),
                "--tag",
                tag,
                "--compression",
                "none",
            )
            inspected = json.loads(
                kit_command(publisher, f"{role}-{label}-inspect", "inspect", tag)
            )
            digests.append(inspected["digest"])
        if digests[0] == digests[1]:
            raise RuntimeError("repack did not change the package identity")
        packages[role] = {"original": digests[0], "repacked": digests[1], "tag": tag}
        # The chosen original digest is frozen before delivery. The tag already
        # points elsewhere, so recipient selection cannot silently follow it.
        request["sides"][role].update(
            {
                "blobs": "kit-store/storage/blobs/sha256",
                "package_digest": digests[0],
                "candidate": role + "/model",
            }
        )
        shutil.rmtree(context)
    write_json(workspace / "selected-packages.json", packages)
    transfer(publisher / "storage", transferred / "storage")
    for role, package in packages.items():
        kit_command(
            transferred,
            role + "-unpack",
            "unpack",
            package["tag"].rsplit(":", 1)[0] + "@" + package["original"],
            "--filter",
            "model",
            "--dir",
            str(recipient / role),
        )
    request_path = recipient / "recipient.json"
    write_json(request_path, request)

    def verify(name: str, value: dict, expected: int | tuple[int, ...]) -> dict:
        path = recipient / (name + ".json")
        write_json(path, value)
        result = json.loads(
            command(
                [
                    str(args.python.absolute()),
                    "-I",
                    str(helper),
                    "--request",
                    str(path),
                ],
                logs / name,
                recipient,
                expected=expected,
            )
        )
        allowed = (expected,) if isinstance(expected, int) else expected
        if result["exit_code"] not in allowed or bool(result["accepted"]) != (
            result["exit_code"] == 0
        ):
            raise RuntimeError("recipient decision contradicts process result")
        return result

    accepted = verify("recipient-decision", request, (0, 1))
    if not accepted.get("technical_integrity_ok") or not accepted.get(
        "envelope_evidence_bound"
    ):
        raise RuntimeError(
            "authentic bound evidence is required even for policy rejection"
        )
    expected_decision = accepted["exit_code"]
    variants = {}
    repacked = copy.deepcopy(request)
    for role in packages:
        repacked["sides"][role]["package_digest"] = packages[role]["repacked"]
    variants["repacked_same_contents"] = verify("repacked", repacked, expected_decision)
    wrong_runtime = copy.deepcopy(request)
    wrong_runtime["technical_anchors"]["runtime_digests"]["subject"] = (
        "sha256:" + "0" * 64
    )
    variants["wrong_runtime"] = verify("wrong-runtime", wrong_runtime, 2)
    revoked = read_json(recipient / "recipient_policy")
    for signer in revoked["trusted_signers"]:
        signer["status"] = "revoked"
    write_json(recipient / "revoked-policy.json", revoked)
    revoked_request = {**request, "recipient_policy": "revoked-policy.json"}
    variants["revoked_signer"] = verify("revoked-signer", revoked_request, 1)
    evidence_copy = recipient / "altered-evidence"
    transfer(recipient / "evidence", evidence_copy)
    manifest = evidence_copy / "manifest.json"
    manifest.chmod(0o600)
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    variants["altered_evidence"] = verify(
        "altered-evidence-request", {**request, "evidence": "altered-evidence"}, 2
    )
    blob = (
        recipient
        / request["sides"]["subject"]["blobs"]
        / packages["subject"]["original"].removeprefix("sha256:")
    )
    hidden = blob.with_suffix(".withheld")
    blob.rename(hidden)
    try:
        variants["missing_package"] = verify("missing-package", request, 2)
    finally:
        hidden.rename(blob)
    model = recipient / request["sides"]["subject"]["candidate"]
    artifact_file = request["sides"]["subject"].get("artifact_file")
    selected = (
        model / artifact_file
        if artifact_file
        else next(path for path in sorted(model.rglob("*")) if path.is_file())
    )
    with selected.open("r+b") as stream:
        original_byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([(original_byte[0] if original_byte else 0) ^ 1]))
    try:
        variants["candidate_changed"] = verify("candidate-changed", request, 2)
    finally:
        with selected.open("r+b") as stream:
            stream.write(original_byte)
            if not original_byte:
                stream.truncate(0)
    # Recheck current acceptance immediately before the actual recipient load.
    final = verify("before-inference", request, expected_decision)
    smoke = None
    if not final["accepted"] and not args.smoke_on_policy_rejection:
        smoke = {"executed": False, "reason": "recipient policy rejected"}
    else:
        smoke = inference_smoke(
            request_path,
            final,
            args.smoke_image,
            args.container_llama,
            logs,
            prompt=args.prompt,
            tokens=args.tokens,
            allow_policy_rejection=args.smoke_on_policy_rejection,
            container_engine=args.container_engine,
        )
    result = {
        "format": "invarlock/example-modelkit-real-journey-v1",
        "accepted": accepted,
        "packages": packages,
        "scenarios": variants,
        "inference": smoke,
        "helper_sha256": digest(helper),
        "kit_binary_sha256": digest(kit),
        "source_request_sha256": digest(args.request),
        "scope": "Actual model package transfer through separate local stores; no registry service claim",
    }
    write_json(workspace / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help="Independent native recipient inputs with source model candidates",
    )
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--kit-sha256", required=True)
    parser.add_argument(
        "--python",
        required=True,
        type=Path,
        help="Separate installed-wheel recipient Python",
    )
    parser.add_argument(
        "--smoke-image",
        required=True,
        help="Immutable local image ID in the selected engine for the offline CPU smoke",
    )
    parser.add_argument(
        "--container-engine",
        choices=("docker", "podman"),
        default="docker",
        help="Engine holding the selected local smoke image (default: docker)",
    )
    parser.add_argument("--container-llama", default="/opt/llama.cpp/llama-completion")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New private directory outside the checkout",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", default=8, type=int, choices=range(1, 33))
    parser.add_argument(
        "--smoke-on-policy-rejection",
        action="store_true",
        help="Allow isolated usability inference after authentic policy rejection; never deployment acceptance",
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "result": str(args.output / "result.json"),
                "completed": True,
                "accepted": result["accepted"]["accepted"],
            }
        )
    )
    return result["accepted"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
