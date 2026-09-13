"""Prepare captured NLL evidence from independently pinned real Harness results.

Run with only the installed InvarLock core wheel. This checks retained capture
consistency, not model execution. Independent recipients regenerate the same
runs and request from reviewed raw inputs before verifying submitted evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path

from invarlock.engine import (
    capture_evaluator_run,
    captured_request_digest,
    case_set_digest,
    freeze_case_set,
    normalize_captured_request,
    run_digest,
    write_run,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

MAX_CAPTURE_BYTES = 4 * 1024 * 1024
SOURCE = {"name": "lm-eval", "version": "0.4.12"}


def digest(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pinned_json(path, expected):
    raw = read_regular_file_bytes(
        path, label="Harness capture", max_bytes=MAX_CAPTURE_BYTES
    )
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Capture differs from the independently supplied digest")
    value = parse_json_bytes(raw, label="Harness capture")
    if not isinstance(value, dict):
        raise ValueError("Capture must be a JSON object")
    return value


def _tokens(value, maximum=2048):
    return (
        isinstance(value, list)
        and 0 < len(value) <= maximum
        and all(type(token) is int and token >= 0 for token in value)
    )


def captured_runs(manifest, raw, source_digest):
    """Reject malformed capture structures before constructing any output."""
    try:
        return _captured_runs(manifest, raw, source_digest)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed Harness capture structure or numeric fact") from exc


def _captured_runs(manifest, raw, source_digest):
    """Cross-check raw API returns against the fixed cases and identity inventory."""
    if (
        manifest["format"] != "invarlock/harness-likelihood-manifest-v1"
        or raw["format"] != "invarlock/harness-likelihood-results-v1"
        or manifest["source"] != raw["source"]
        or manifest["source"] != SOURCE
        or set(raw["runs"]) != {"baseline", "subject"}
    ):
        raise ValueError("Unsupported or inconsistent Harness capture profile")
    artifact = manifest["model"]["artifact_digest"]
    tokenizer = manifest["tokenizer"]["tokenizer_digest"]
    configuration = manifest["configuration_digest"]
    if (
        artifact != digest(manifest["model"]["files"])
        or tokenizer != digest(manifest["tokenizer"]["files"])
        or configuration != digest(manifest["configuration"])
    ):
        raise ValueError("Capture identity inventory differs from its bindings")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not 4 <= len(cases) <= 8:
        raise ValueError("This rehearsal requires four to eight original cases")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Case identities must be unique")
    maximum = manifest["configuration"]["max_length"]
    if type(maximum) is not int or not 1 <= maximum <= 2048:
        raise ValueError("Invalid declared context limit")
    if manifest["configuration"].get("backend") != "causal" or any(
        manifest["configuration"].get(name) is not False
        for name in ("truncation", "add_bos_token", "logits_cache", "result_cache")
    ):
        raise ValueError("Unsupported continuation or cache configuration")
    runs = {}
    for side in ("baseline", "subject"):
        records = raw["runs"][side]["records"]
        if not isinstance(records, list) or len(records) != len(cases):
            raise ValueError(
                "Raw result inventory differs from the fixed case schedule"
            )
        for index, (case, row) in enumerate(zip(cases, records, strict=True)):
            if {key: row[key] for key in ("id", "input", "expected")} != case:
                raise ValueError("Original context, reference or case order changed")
            context, reference = row["input"], row["expected"]
            if (
                not isinstance(context, str)
                or not context
                or context != context.rstrip()
                or not isinstance(reference, str)
                or not reference
            ):
                raise ValueError(
                    "Empty inputs or trailing context whitespace are outside this profile"
                )
            retained = row["context"]["harness"]
            left, right = (
                retained["context_token_ids"],
                retained["continuation_token_ids"],
            )
            if (
                not _tokens(left)
                or not _tokens(right)
                or len(right) > maximum
                or len(left) + len(right) > maximum + 1
                or type(retained["request_index"]) is not int
                or retained["request_index"] != index
                or not _tokens(retained["joined_token_ids"], maximum + 1)
                or retained["joined_token_ids"] != left + right
                or retained["decoded_continuation"] != reference
                or type(retained["input_utf8_byte_count"]) is not int
                or retained["input_utf8_byte_count"] != len(context.encode("utf-8"))
            ):
                raise ValueError("Invalid token boundary, truncation or request order")
            result = retained["result"]
            likelihood = row["likelihood"]
            if (
                not isinstance(result, list)
                or len(result) != 2
                or type(result[0]) not in (int, float)
                or not math.isfinite(result[0])
                or result[0] > 0
                or type(result[1]) is not bool
                or type(retained["is_greedy"]) is not bool
                or result[0] != likelihood["logprob_sum"]
                or result[1] != retained["is_greedy"]
                or likelihood["token_count"] != len(right)
                or likelihood["artifact_digest"] != artifact
                or likelihood["tokenizer_digest"] != tokenizer
                or likelihood["configuration_digest"] != configuration
            ):
                raise ValueError(
                    "Typed likelihood facts differ from the retained API result or identities"
                )
        rows = copy.deepcopy(records)
        for row in rows:
            row["context"]["model_id"] = manifest["model_identity"]["id"]
            row["context"]["model_revision"] = manifest["model_identity"]["revision"]
        runs[side] = capture_evaluator_run(
            rows,
            source=raw["source"],
            run_id=side,
            artifact_digest=artifact,
            source_digest=source_digest,
        )
    return runs


def prepare(capture, output, *, manifest_sha256, results_sha256):
    manifest = pinned_json(capture / "manifest.json", manifest_sha256)
    raw = pinned_json(capture / "raw-results.json", results_sha256)
    runs = captured_runs(manifest, raw, results_sha256)
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "expected_case_set_digest": case_set_digest(
            freeze_case_set([{**case, "metadata": {}} for case in manifest["cases"]])
        ),
        "metrics": [
            {
                "name": "reference_nll",
                "kind": "normalized_nll_per_utf8_byte",
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": len(manifest["cases"]),
                # Fixed conformance tolerance; never tuned to a retained result.
                "ratio_max": 1.000001,
                "maximum_interval_width": 0.000001,
                "configuration": {
                    "configuration_digest": manifest["configuration_digest"],
                    "baseline_tokenizer_digest": manifest["tokenizer"][
                        "tokenizer_digest"
                    ],
                    "subject_tokenizer_digest": manifest["tokenizer"][
                        "tokenizer_digest"
                    ],
                },
            }
        ],
        "slices": [],
    }
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "policy.json",
            "metric": "normalized_nll_per_utf8_byte",
        },
        "output": {"evidence": "evidence"},
    }
    normalized = normalize_captured_request(
        request, baseline=runs["baseline"], subject=runs["subject"], policy=policy
    )
    anchors = {
        "baseline_run_digest": run_digest(runs["baseline"]),
        "subject_run_digest": run_digest(runs["subject"]),
        "request_digest": captured_request_digest(normalized),
    }
    output.mkdir(parents=True, exist_ok=False)
    for side, run in runs.items():
        write_run(output / f"{side}.json", run)
    for name, value in (
        ("policy.json", policy),
        ("request.yaml", request),
        ("anchors.json", anchors),
    ):
        (output / name).write_bytes(canonical_json_bytes(value))
    return anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--results-sha256", required=True)
    args = parser.parse_args()
    anchors = prepare(
        args.capture,
        args.output,
        manifest_sha256=args.manifest_sha256,
        results_sha256=args.results_sha256,
    )
    print(canonical_json_bytes(anchors).decode(), end="")


if __name__ == "__main__":
    main()
