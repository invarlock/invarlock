"""Derive a checked Harness serialization binding without repeating model tasks."""

from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bindings  # noqa: E402
import common  # noqa: E402
import recipient  # noqa: E402


def recover(protocol_path, protocol_sha256, source, role, output):
    from invarlock.captured_contracts import read_file
    from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES

    protocol, _ = recipient.read(protocol_path)
    if common.digest(protocol) != protocol_sha256 or role not in recipient.ROLES:
        raise ValueError(
            "recovery differs from its independently supplied protocol or role"
        )
    evaluator = "lm-evaluation-harness"
    native, validated, results = recipient.capture(source, protocol, role, evaluator)
    if not isinstance(native, list) or len(native) != len(protocol["cases"]):
        raise ValueError("Harness export differs from the complete frozen schedule")
    cases = {case["id"]: case for case in protocol["cases"]}
    corrected, seen = deepcopy(native), set()
    for row in corrected:
        ident = row["metadata"]["invarlock_id"]
        if ident not in cases or ident in seen:
            raise ValueError("Harness export has an unknown or repeated case")
        seen.add(ident)
        case, result = cases[ident], results[ident]
        expected = bindings.bind_result(
            result, case, evaluator, protocol["versions"][evaluator]
        )
        if (
            not recipient.same(row["target"], case["expected"])
            or not recipient.same(row["filtered_resps"], [result["output"]])
            or any(
                not recipient.same(
                    row["metadata"].get(name), expected["metadata"].get(name)
                )
                for name in (
                    "invarlock_likelihood",
                    "invarlock_capture_binding",
                    "invarlock_model_execution",
                    "invarlock_task_outcome",
                )
            )
        ):
            raise ValueError(
                "Harness observations differ from their original model ledger"
            )
        row["metadata"] = bindings.rebind_harness_metadata(
            row["metadata"], case, protocol["cases"], row["doc"]
        )
    source = Path(source)
    archive, total = {}, 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("original capture archive cannot contain symlinks")
        if path.is_dir():
            continue
        raw = read_file(path, MAX_INPUT_BYTES)
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("original capture archive exceeds the bounded size")
        archive[path.relative_to(source).as_posix()] = raw
    if any(archive.get(name) != raw for name, raw in validated.items()):
        raise ValueError("original capture changed during recovery validation")
    corrected_raw = common.encoded(corrected)
    declaration = {
        "format": "invarlock/harness-serialization-recovery-v1",
        "status": "derived_capture",
        "protocol_digest": protocol_sha256,
        "role": role,
        "model_calls": 0,
        "operation": "checked nullable metadata expansion binding; original measurements unchanged",
        "original_files": {
            name: "sha256:" + hashlib.sha256(raw).hexdigest()
            for name, raw in archive.items()
        },
        "corrected_native_sha256": "sha256:"
        + hashlib.sha256(corrected_raw).hexdigest(),
        "implementations": {
            name: "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (
                ("recovery", __file__),
                ("bindings", bindings.__file__),
                ("recipient", recipient.__file__),
            )
        },
    }
    manifest = common.decode(validated["capture.json"])
    manifest["native_sha256"] = declaration["corrected_native_sha256"]
    manifest["serialization_recovery"] = common.digest(declaration)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, raw in archive.items():
        path = output / "original" / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    for name, raw in validated.items():
        if name in {"capture.json", "native.json"}:
            continue
        path = output / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    common.write(output / "native.json", corrected)
    common.write(output / "capture.json", manifest)
    common.write(output / "recovery.json", declaration)
    return declaration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=recipient.ROLES, required=True)
    args = parser.parse_args()
    recipient.installed_identity()
    print(
        common.encoded(
            recover(
                args.protocol,
                args.protocol_sha256,
                args.capture,
                args.role,
                args.output,
            )
        ).decode()
    )


if __name__ == "__main__":
    main()
