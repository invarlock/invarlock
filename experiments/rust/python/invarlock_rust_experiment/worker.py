"""Fresh isolated worker. The parent retains stdout, stderr and wait4 usage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
from pathlib import Path

from .adapters import candidate_adapter
from .fixtures import make_fixture, write_json


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity(require_installed=True):
    import invarlock

    distribution = importlib.metadata.distribution("invarlock")
    direct = distribution.read_text("direct_url.json")
    if (
        require_installed
        and direct
        and json.loads(direct).get("dir_info", {}).get("editable")
    ):
        raise RuntimeError(
            "complete-path qualification requires a non-editable installed wheel"
        )
    package_path = Path(invarlock.__file__).resolve()
    if require_installed and "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"complete-path import is not installed in site-packages: {package_path}"
        )
    facts = {
        "executable": sys.executable,
        "python": sys.version,
        "package": str(package_path),
        "invarlock_version": distribution.version,
        "dependencies": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
    }
    for module in ("evidence_pack_contract.py", "evaluation_comparison/likelihood.py"):
        facts[module] = sha_file(package_path.parent / module)
    try:
        native = importlib.metadata.distribution("invarlock-native-experiment")
        facts["native_version"] = native.version
        facts["native_files"] = {
            str(file): sha_file(native.locate_file(file))
            for file in native.files or []
            if str(file).endswith((".so", ".dylib", ".pyd"))
        }
    except importlib.metadata.PackageNotFoundError:
        facts["native_version"] = None
    return facts


def check_wheel_binding(metadata_path):
    """Bind retained wheel identities to the actual imported installed bytes."""
    import zipfile

    metadata = json.loads(Path(metadata_path).read_bytes())
    bindings = {}
    for key, distribution_name, prefix in (
        ("root_wheel", "invarlock", "invarlock/"),
        ("native_wheel", "invarlock-native-experiment", "invarlock_native/"),
    ):
        item = metadata[key]
        if sha_file(item["path"]) != item["sha256"].removeprefix("sha256:"):
            raise RuntimeError(f"{key} wheel identity changed")
        distribution = importlib.metadata.distribution(distribution_name)
        checked = 0
        with zipfile.ZipFile(item["path"]) as wheel:
            for member in wheel.infolist():
                if member.filename.startswith(prefix) and not member.is_dir():
                    installed = distribution.locate_file(member.filename)
                    if hashlib.sha256(wheel.read(member)).hexdigest() != sha_file(
                        installed
                    ):
                        raise RuntimeError(
                            f"installed bytes differ from {key}: {member.filename}"
                        )
                    checked += 1
        if checked == 0:
            raise RuntimeError(f"no package files bound for {key}")
        bindings[key] = {"sha256": item["sha256"], "installed_files_checked": checked}
    return bindings


def block_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("network access is forbidden in this offline experiment")


def execute(args):
    identity_start = time.perf_counter()
    identity = runtime_identity(not args.allow_source_smoke)
    if args.metadata:
        identity["wheel_bindings"] = check_wheel_binding(args.metadata)
    identity["check_wall_seconds"] = time.perf_counter() - identity_start
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.stage == "fixture":
        start = time.perf_counter()
        result = make_fixture(Path(args.fixture), args.count, args.scenario)
        return {
            "ok": True,
            "stage": args.stage,
            "identity": identity,
            "operation_wall_seconds": time.perf_counter() - start,
            "result": result,
        }
    fixture = Path(args.fixture)
    with candidate_adapter(args.candidate) as counter:
        # Verification is imported after the explicit substitution of both aliases.
        start = time.perf_counter()
        if args.stage == "kernel":
            from invarlock import evidence_pack_contract as contract

            kernel_arguments = json.loads((fixture / "kernel.json").read_bytes())
            start = time.perf_counter()
            interval = contract._paired_resampling_interval(**kernel_arguments)
            elapsed = time.perf_counter() - start
            result = {"interval_hex": [float(value).hex() for value in interval]}
            write_json(output / "kernel-result.json", result)
        elif args.stage == "zero":
            from invarlock import evidence_pack_contract as contract

            start = time.perf_counter()
            try:
                contract._paired_resampling_interval(
                    metric="normalized_nll_per_utf8_byte",
                    baseline_scores=[0.0] * 7,
                    subject_scores=[1.0] * 7,
                    schedule_sha256="0" * 64,
                )
            except (contract.EvidencePackError, ValueError) as exc:
                if "baseline mean must be greater than zero" not in str(exc):
                    raise
                result = {"rejected": True, "category": "nonpositive_baseline"}
            else:
                raise RuntimeError("zero baseline was not rejected")
            elapsed = time.perf_counter() - start
        else:
            from invarlock.captured_verification import verify_captured_receipt
            from invarlock.engine import EvidenceVerificationError, verify_evidence

            pins = json.loads((fixture / "trust-pins.json").read_bytes())
            anchors = {
                key: value
                for key, value in pins.items()
                if key
                not in {"expected_verifier_identity", "expected_verifier_fingerprint"}
            }
            anchors["policy_path"] = fixture / "policy.json"
            rejected = args.stage == "reject"
            if rejected:
                anchors["expected_signer"] = "sha256:" + "0" * 64
            start = time.perf_counter()
            try:
                verification = verify_evidence(
                    fixture / "pack",
                    **anchors,
                    receipt_path=output / "receipt.json",
                    verifier_signing_key_path=fixture / "verifier.pem",
                    verifier_identity=pins["expected_verifier_identity"],
                )
                verification_payload = verification.payload
            except EvidenceVerificationError as exc:
                expected_fail = (
                    json.loads((fixture / "expected.json").read_bytes())["decision"]
                    != "pass"
                )
                if not (
                    (rejected and exc.exit_code == 6)
                    or (not rejected and expected_fail and exc.exit_code == 7)
                ):
                    raise
                verification_payload = exc.payload
            authenticated = verify_captured_receipt(
                output / "receipt.json",
                fixture / "pack",
                **anchors,
                expected_verifier_identity=pins["expected_verifier_identity"],
                expected_verifier_fingerprint=pins["expected_verifier_fingerprint"],
            )
            if not authenticated.ok:
                raise RuntimeError(
                    f"independent receipt authentication failed: {authenticated.errors}"
                )
            elapsed = time.perf_counter() - start
            expected_decision = json.loads((fixture / "expected.json").read_bytes())[
                "decision"
            ]
            if rejected:
                if (
                    verification_payload["integrity_ok"] is not False
                    or counter.calls != 0
                ):
                    raise RuntimeError(
                        "untrusted signer did not fail before kernel replay"
                    )
            elif (
                verification_payload["decision"] != expected_decision
                or verification_payload["integrity_ok"] is not True
            ):
                raise RuntimeError(
                    f"verification disagreed with expected decision: {verification_payload}"
                )
            report = json.loads(
                (fixture / "pack/reports/evaluation.report.json").read_bytes()
            )
            receipt = json.loads((output / "receipt.json").read_bytes())
            # Entire canonical report and receipt statement; no fields are excluded.
            result = {
                "report": report,
                "receipt_statement": receipt["statement"],
                "receipt_sha256": sha_file(output / "receipt.json"),
            }
            write_json(output / "verification-result.json", verification_payload)
            write_json(output / "canonical-result.json", result)
        if args.stage != "reject" and (
            counter.calls < 1 or (args.candidate == "rust" and counter.native_calls < 1)
        ):
            raise RuntimeError("candidate kernel was not actually invoked")
        write_json(output / "result.json", result)
        return {
            "ok": True,
            "stage": args.stage,
            "candidate": args.candidate,
            "identity": identity,
            "operation_wall_seconds": elapsed,
            "candidate_calls": counter.calls,
            "native_calls": counter.native_calls,
            "result_sha256": sha_file(output / "result.json"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("fixture", "kernel", "verify", "reject", "zero"),
        required=True,
    )
    parser.add_argument(
        "--candidate", choices=("reference", "optimized", "rust"), default="reference"
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--scenario", choices=("pass", "regression"), default="pass")
    parser.add_argument("--allow-source-smoke", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)
    try:
        result = execute(args)
    except BaseException as exc:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": args.stage,
                    "candidate": args.candidate,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        return 1
    write_json(Path(args.output) / "worker-status.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
