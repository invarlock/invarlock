"""Project an independently pinned native rehearsal into installed pipeline runs.

Copy this file, the capture, and the predeclared protocol to the recipient. Supply
both expected SHA256 values independently. This checks the publisher's retained
records; it does not prove that an untrusted publisher actually executed a model.
Only the three fixed single-completion profiles from native_rehearsal.py are
supported. No evaluator SDK, model weights, signing key or repository is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path

from invarlock.pipeline import load_run, make_run

PROFILES = {
    "inspect": ("inspect-json", ".json", "match"),
    "lm-eval": ("lm-eval-samples", ".jsonl", "exact_match"),
    "promptfoo": ("promptfoo-jsonl", ".jsonl", "score"),
}
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_BYTES = 64 * 1024 * 1024


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read(path, max_bytes=MAX_FILE_BYTES):
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("this example requires no-follow file opening support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError(f"expected a bounded regular file: {path.name}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError(f"expected a bounded regular file: {path.name}")
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("file grew beyond the input bound")
    return raw


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def checked(value, message):
    if not value:
        raise ValueError(message)


def inputs(capture, protocol_path, expected_protocol, expected_capture):
    protocol_bytes = read(protocol_path)
    checked(digest(protocol_bytes) == expected_protocol, "protocol digest mismatch")
    protocol = decode(protocol_bytes)
    checked(
        protocol.get("format") == "invarlock/example-native-rehearsal-v1",
        "unsupported protocol",
    )
    checked(
        not capture.is_symlink() and capture.is_dir(),
        "capture must be an ordinary directory",
    )
    manifest_bytes = read(capture / "capture.json")
    checked(digest(manifest_bytes) == expected_capture, "capture digest mismatch")
    manifest = decode(manifest_bytes)
    checked(
        manifest.get("format") == "invarlock/example-native-capture-v1",
        "unsupported capture",
    )
    checked(
        manifest["protocol_sha256"] == expected_protocol,
        "capture belongs to another protocol",
    )
    evaluator = manifest["evaluator"]
    checked(evaluator in PROFILES, "unsupported evaluator")
    checked(
        manifest["version"] == protocol["evaluators"][evaluator]["version"],
        "evaluator version drift",
    )
    files = manifest["files"]
    checked(
        isinstance(files, dict) and 0 < len(files) <= 100, "invalid capture inventory"
    )
    actual = {
        p.name for p in capture.iterdir() if not p.is_dir() and p.name != "capture.json"
    }
    checked(actual == set(files), "capture inventory differs from the manifest")
    retained = {}
    remaining = MAX_CAPTURE_BYTES
    for name, expected in files.items():
        checked(
            isinstance(name, str)
            and Path(name).name == name
            and name not in (".", "..", "capture.json"),
            "invalid inventory path",
        )
        raw = read(capture / name, max_bytes=min(MAX_FILE_BYTES, remaining))
        checked(digest(raw) == expected, f"capture file digest mismatch: {name}")
        retained[name] = raw
        remaining -= len(raw)
    return protocol, manifest, retained


def provenance(evaluator):
    return {
        "kind": "measurement",
        "source": f"native-cpu-call-timer-{evaluator}",
        "version": "1.0.0",
        "unit": "milliseconds",
        "rubric_digest": None,
    }


def policy_for(protocol, evaluator, kind):
    settings = protocol["policy"]
    scoring, configuration = {
        "classification": ("exact_match", {}),
        "extraction": ("json_fields", {"fields": ["/status"]}),
        "numeric": ("numeric_tolerance", {"absolute": 0, "relative": 0}),
    }[kind]
    return {
        "format": "invarlock/pipeline-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "quality",
                "kind": scoring,
                "configuration": configuration,
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": settings["minimum_count"],
                "maximum_regression": settings["maximum_regression"],
                "maximum_interval_width": settings["maximum_interval_width"],
                "candidate_minimum": settings["candidate_minimum"],
            },
            {
                "name": "latency",
                "kind": "recorded",
                "configuration": {},
                "score_key": "latency_ms",
                "direction": "lower",
                "unit": "milliseconds",
                "aggregation": "mean",
                "accepted_provenance": provenance(evaluator),
                "minimum_count": settings["minimum_count"],
                "maximum_regression": settings["latency_maximum_regression_ms"],
                "maximum_interval_width": settings["latency_maximum_interval_width_ms"],
                "candidate_maximum": settings["latency_candidate_maximum_ms"],
            },
        ],
    }


def native_configuration(evaluator, native, files, stem, kind, side, protocol):
    template = protocol["prompts"][side]
    cases = protocol["cases"][kind]
    if evaluator == "inspect":
        details = native["eval"]
        checked(details["task"] == f"{kind}_{side}", "Inspect task drift")
        checked(
            details["model"] == "native_cpu_rehearsal/" + protocol["model"]["id"],
            "Inspect model drift",
        )
        checked(
            details["packages"]["inspect_ai"]
            == protocol["evaluators"][evaluator]["version"],
            "Inspect version drift",
        )
        checked(
            details["model_generate_config"]
            == {"max_connections": 1, "max_tokens": 8, "temperature": 0.0},
            "Inspect generation settings drift",
        )
    elif evaluator == "lm-eval":
        checked(decode(files[f"{kind}-dataset.json"]) == cases, "LM dataset drift")
    else:
        config = decode(files[f"{stem}-config.json"])
        checked(
            config["prompts"] == [template.replace("{input}", "{{input}}")],
            "Promptfoo prompt configuration drift",
        )
        expected_tests = [
            {
                "vars": {"input": case["input"]},
                "metadata": {"invarlock_expected": case["expected"]},
                "assert": [{"type": "equals", "value": case["expected"]}],
            }
            for case in cases
        ]
        checked(
            config["tests"] == expected_tests, "Promptfoo reference configuration drift"
        )


def project_records(evaluator, native_rows, imported, calls, cases, template):
    checked(
        len(imported) == len(cases) == len(calls) == len(native_rows),
        "missing or extra native records",
    )
    rows = []
    for case, row, native, call in zip(
        cases, imported, native_rows, calls, strict=True
    ):
        prompt = template.format(input=case["input"])
        identifier = case["id"] + ":0" if evaluator == "promptfoo" else case["id"]
        expected_input = (
            {"input": case["input"]}
            if evaluator == "promptfoo"
            else case
            if evaluator == "lm-eval"
            else case["input"]
        )
        checked(
            row["id"] == identifier
            and row["input"] == expected_input
            and row["expected"] == case["expected"],
            "native identity, input or reference drift",
        )
        checked(
            isinstance(row["output"], str) and row["output"] == call["output"],
            "native output differs from the captured model call",
        )
        checked(call["prompt"] == prompt, "captured call prompt drift")
        checked(
            row["error"] is None and call["error"] is None,
            "native execution did not complete without error",
        )
        latency = call["latency_ms"]
        checked(
            type(latency) in (float, int) and math.isfinite(latency) and latency >= 0,
            "invalid call latency",
        )
        if evaluator == "inspect":
            messages = native["messages"]
            checked(
                len(messages) == 2
                and messages[0]["role"] == "user"
                and messages[0]["content"] == prompt
                and messages[1]["role"] == "assistant"
                and messages[1]["content"] == row["output"],
                "Inspect message drift",
            )
        elif evaluator == "lm-eval":
            checked(
                row["context"]["arguments"]
                == [[prompt, {"do_sample": False, "max_gen_toks": 8, "until": []}]],
                "LM generation request drift",
            )
        else:
            checked(
                row["context"]["prompt"] == prompt, "Promptfoo rendered prompt drift"
            )
            checked(
                native["testCase"]["assert"]
                == [{"type": "equals", "value": case["expected"]}],
                "Promptfoo assertion drift",
            )
        score_key = PROFILES[evaluator][2]
        checked(
            row["scores"].get(score_key) == float(row["output"] == case["expected"]),
            "native score contradicts the retained output and reference",
        )
        rows.append(
            {
                "id": case["id"],
                "input": case["input"],
                "expected": case["expected"],
                "output": row["output"],
                "context": {"prompt": prompt, "native_scores": row["scores"]},
                "scores": {"latency_ms": latency},
                "error": None,
            }
        )
    return rows


def project_capture(
    capture, protocol_path, expected_protocol, expected_capture, output
):
    protocol, manifest, files = inputs(
        capture, protocol_path, expected_protocol, expected_capture
    )
    evaluator = manifest["evaluator"]
    checked(
        set(protocol["cases"]) == {"classification", "extraction", "numeric"}
        and set(protocol["prompts"]) == {"baseline", "candidate"},
        "unsupported scenario selection",
    )
    source = {
        "name": protocol["evaluators"][evaluator]["package"],
        "version": manifest["version"],
    }
    adapter, suffix, _ = PROFILES[evaluator]
    pending = {}
    # Parse immutable copies of the authenticated bytes. The public adapter never
    # rereads a publisher-owned path after its source digest has been checked.
    with tempfile.TemporaryDirectory(prefix="invarlock-native-projection-") as temp:
        for kind, cases in protocol["cases"].items():
            checked(
                [case["id"] for case in cases] == [str(i) for i in range(len(cases))],
                "case IDs must preserve declared order",
            )
            pending[f"{kind}/policy.json"] = policy_for(protocol, evaluator, kind)
            for side, template in protocol["prompts"].items():
                stem = f"{kind}-{side}"
                raw = files[stem + suffix]
                native = (
                    decode(raw)
                    if suffix == ".json"
                    else [decode(line) for line in raw.splitlines() if line.strip()]
                )
                native_configuration(
                    evaluator, native, files, stem, kind, side, protocol
                )
                path = Path(temp) / (stem + suffix)
                path.write_bytes(raw)
                artifact_digest = digest(
                    encoded(
                        {
                            "model": protocol["model"],
                            "runtime": protocol["runtime"],
                            "generation": protocol["generation"],
                            "prompt": template,
                            "evaluator": source,
                        }
                    )
                )
                imported = load_run(
                    path,
                    adapter=adapter,
                    source=source,
                    run_id=stem,
                    artifact_digest=artifact_digest,
                )
                checked(
                    imported["source_digest"] == digest(raw),
                    "public adapter source digest mismatch",
                )
                rows = project_records(
                    evaluator,
                    native["samples"] if evaluator == "inspect" else native,
                    imported["records"],
                    decode(files[stem + "-calls.json"]),
                    cases,
                    template,
                )
                pending[f"{kind}/{side}.json"] = make_run(
                    rows,
                    source=source,
                    run_id=stem,
                    artifact_digest=artifact_digest,
                    source_digest=expected_capture,
                    score_provenance={"latency_ms": provenance(evaluator)},
                )
            pending[f"{kind}/pipeline.json"] = {
                "format": "invarlock/pipeline-project-v1",
                "baseline": {"path": "baseline.json", "adapter": "invarlock"},
                "candidate": {"path": "candidate.json", "adapter": "invarlock"},
                "policy": "policy.json",
            }
    output.mkdir()
    for name, value in pending.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(encoded(value))
    receipt = {
        "format": "invarlock/example-native-projection-v1",
        "expected_protocol": expected_protocol,
        "expected_capture": expected_capture,
        "evaluator": evaluator,
        "files": {name: digest(encoded(value)) for name, value in pending.items()},
    }
    (output / "projection.json").write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol", required=True)
    parser.add_argument("--expected-capture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = project_capture(
            args.capture,
            args.protocol,
            args.expected_protocol,
            args.expected_capture,
            args.output,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Native handoff rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
