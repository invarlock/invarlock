"""Replay a pinned 400-case Harness capture into a core-only recipient project.

The protocol pin is supplied independently; each role pin authenticates that
role's raw-results.json. Tokenizations are cross-checked against those pinned
rows. Manifest runtime versions are compared as source assertions, not treated
as authenticated execution evidence. No model, tokenizer or evaluator is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

from invarlock.captured_contracts import secure_directory
from invarlock.engine import (
    capture_evaluator_run,
    captured_request_digest,
    case_set_digest,
    comparison_policy_digest,
    freeze_case_set,
    normalize_captured_request,
    run_digest,
    write_run,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.staged_directory import staged_directory

MAX_FILE_BYTES = 16 * 1024 * 1024
SOURCE = {"name": "lm-eval", "version": "0.4.12"}
MODELS = {
    "baseline": (
        "mistralai/Mistral-7B-v0.1",
        "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    ),
    "subject": (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
    ),
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
}
TOKEN_FIELDS = {
    "context_token_ids",
    "continuation_token_ids",
    "joined_token_ids",
    "decoded_context",
    "decoded_joined",
    "decoded_continuation",
}
PACKAGES = {
    "lm-eval",
    "transformers",
    "torch",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "accelerate",
    "sentencepiece",
}
IMPLEMENTATIONS = {
    "lm_eval.models.huggingface",
    "lm_eval.api.model",
    "lm_eval.api.metrics",
}
RUNTIME_IMPLEMENTATIONS = {
    "lm_eval.api.instance",
    "transformers.models.mistral.modeling_mistral",
    "transformers.tokenization_utils_base",
}


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, fields, label):
    checked(
        isinstance(value, dict) and set(value) == set(fields), f"{label} fields differ"
    )


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def object_digest(value):
    return digest(canonical_json_bytes(value))


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def check_digest(value):
    checked(
        isinstance(value, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", value),
        "expected a lowercase SHA-256 digest",
    )


def read(path):
    with secure_directory(path.parent):
        raw = read_regular_file_bytes(path, label=path.name, max_bytes=MAX_FILE_BYTES)
    return raw, parse_json_bytes(raw, label=path.name)


def validate_protocol(protocol):
    exact(
        protocol,
        {
            "format",
            "models",
            "cases",
            "configuration",
            "policy",
            "source",
            "source_implementations",
            "dataset",
            "capture_script_sha256",
        },
        "protocol",
    )
    checked(
        protocol["format"] == "invarlock/harness-model-comparison-v1"
        and protocol["source"] == SOURCE,
        "unsupported protocol or source",
    )
    check_digest(protocol["capture_script_sha256"])
    exact(protocol["models"], MODELS, "models")
    for role, model in protocol["models"].items():
        exact(
            model,
            {"id", "revision", "files", "artifact_digest", "tokenizer_digest"},
            "model",
        )
        checked(
            (model["id"], model["revision"]) == MODELS[role],
            "model identity differs from the fixed checkpoints",
        )
        files = model["files"]
        checked(
            isinstance(files, list) and 1 <= len(files) <= 100,
            "invalid model inventory",
        )
        names = []
        for fact in files:
            exact(fact, {"path", "byte_size", "sha256"}, "model file")
            name = fact["path"]
            checked(
                isinstance(name, str)
                and (
                    name
                    in TOKENIZER_FILES
                    | {
                        "config.json",
                        "generation_config.json",
                        "model.safetensors.index.json",
                    }
                    or re.fullmatch(
                        r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", name
                    )
                ),
                "unsupported model inventory path",
            )
            checked(
                type(fact["byte_size"]) is int
                and 0 < fact["byte_size"] <= 20 * 1024**3,
                "invalid model file byte count",
            )
            check_digest(fact["sha256"])
            names.append(name)
        tokenizer = [fact for fact in files if fact["path"] in TOKENIZER_FILES]
        checked(
            names == sorted(set(names))
            and "config.json" in names
            and any(name.endswith(".safetensors") for name in names)
            and tokenizer,
            "incomplete or unordered model inventory",
        )
        checked(
            model["artifact_digest"] == object_digest(files)
            and model["tokenizer_digest"] == object_digest(tokenizer),
            "model inventory digest differs",
        )
    checked(
        protocol["models"]["baseline"]["artifact_digest"]
        != protocol["models"]["subject"]["artifact_digest"],
        "model artifacts must differ",
    )
    exact(
        protocol["source_implementations"], IMPLEMENTATIONS, "upstream implementations"
    )
    for value in protocol["source_implementations"].values():
        check_digest("sha256:" + value if isinstance(value, str) else value)
    checked(
        isinstance(protocol["dataset"], dict) and protocol["dataset"],
        "dataset provenance is required",
    )
    configuration = protocol["configuration"]
    checked(
        isinstance(configuration, dict)
        and type(configuration.get("max_length")) is int
        and configuration["max_length"] == 512
        and configuration.get("truncation") is False
        and configuration.get("result_cache") is False
        and configuration.get("logits_cache") is False
        and configuration.get("chat_template") is False,
        "unsupported context or cache configuration",
    )
    cases = protocol["cases"]
    checked(
        isinstance(cases, list) and len(cases) == 400,
        "the complete 400-case schedule is required",
    )
    for case in cases:
        exact(case, {"id", "input", "expected", "metadata"}, "case")
        checked(
            isinstance(case["input"], str)
            and case["input"]
            and case["input"] == case["input"].rstrip()
            and isinstance(case["expected"], str)
            and case["expected"],
            "invalid original context or reference",
        )
        checked(
            len((case["input"] + case["expected"]).encode("utf-8")) <= 65536,
            "case text exceeds its allowance",
        )
    frozen = case_set_digest(freeze_case_set(cases))
    policy = protocol["policy"]
    comparison_policy_digest(policy)
    checked(
        policy.get("expected_case_set_digest") == frozen
        and len(policy["metrics"]) == 1
        and policy["slices"] == [],
        "policy must bind the full frozen schedule",
    )
    metric = policy["metrics"][0]
    checked(
        metric["kind"] == "normalized_nll_per_utf8_byte"
        and metric["minimum_count"] == 400
        and metric["configuration"]
        == {
            "configuration_digest": object_digest(configuration),
            "baseline_tokenizer_digest": protocol["models"]["baseline"][
                "tokenizer_digest"
            ],
            "subject_tokenizer_digest": protocol["models"]["subject"][
                "tokenizer_digest"
            ],
        },
        "policy likelihood bindings differ from protocol",
    )


def validate_manifest(manifest, protocol, role, expected_protocol):
    exact(
        manifest,
        {
            "format",
            "protocol_sha256",
            "role",
            "source",
            "model",
            "configuration",
            "configuration_digest",
            "source_implementations",
            "packages",
            "capture_script_sha256",
        },
        "manifest",
    )
    checked(
        manifest["format"] == "invarlock/harness-model-comparison-manifest-v1",
        "unsupported manifest",
    )
    expected = {
        "protocol_sha256": expected_protocol,
        "role": role,
        "source": SOURCE,
        "model": protocol["models"][role],
        "configuration": protocol["configuration"],
        "configuration_digest": object_digest(protocol["configuration"]),
        "capture_script_sha256": protocol["capture_script_sha256"],
    }
    checked(
        all(
            canonical_json_bytes(manifest[key]) == canonical_json_bytes(value)
            for key, value in expected.items()
        ),
        "manifest differs from protocol declarations",
    )
    exact(manifest["packages"], PACKAGES, "runtime packages")
    checked(
        all(
            isinstance(v, str) and 0 < len(v) <= 128
            for v in manifest["packages"].values()
        )
        and manifest["packages"]["lm-eval"] == SOURCE["version"],
        "invalid runtime package declarations",
    )
    implementations = manifest["source_implementations"]
    checked(
        isinstance(implementations, list) and len(implementations) == 6,
        "incomplete source inventory",
    )
    observed = {}
    for fact in implementations:
        exact(fact, {"path", "byte_size", "sha256"}, "source file")
        checked(
            type(fact["byte_size"]) is int and 0 < fact["byte_size"] <= MAX_FILE_BYTES,
            "invalid source byte count",
        )
        check_digest(fact["sha256"])
        checked(
            isinstance(fact["path"], str) and fact["path"] not in observed,
            "duplicate or invalid source path",
        )
        observed[fact["path"]] = fact["sha256"]
    paths = {
        name.replace(".", "/") + ".py"
        for name in IMPLEMENTATIONS | RUNTIME_IMPLEMENTATIONS
    }
    checked(set(observed) == paths, "source inventory paths differ")
    checked(
        all(
            observed[name.replace(".", "/") + ".py"] == "sha256:" + sha
            for name, sha in protocol["source_implementations"].items()
        ),
        "upstream implementation differs from protocol",
    )


def validate_row(row, tokens, case, model, configuration_digest, index):
    exact(
        row,
        {"id", "input", "expected", "metadata", "output", "likelihood", "context"},
        "raw record",
    )
    checked(
        all(
            canonical_json_bytes(row[key]) == canonical_json_bytes(case[key])
            for key in case
        ),
        "record order, identity, context, reference or metadata changed",
    )
    checked(
        row["output"] is None,
        "reference likelihood must not substitute a generated answer",
    )
    exact(tokens, TOKEN_FIELDS, "tokenization")
    for field in ("context_token_ids", "continuation_token_ids", "joined_token_ids"):
        value = tokens[field]
        checked(
            isinstance(value, list)
            and 0 < len(value) <= 513
            and all(type(token) is int and 0 <= token < 2**31 for token in value),
            "invalid or truncated token sequence",
        )
    checked(
        tokens["context_token_ids"] + tokens["continuation_token_ids"]
        == tokens["joined_token_ids"],
        "context/continuation token boundary changed",
    )
    checked(
        tokens["decoded_context"] == case["input"]
        and tokens["decoded_joined"] == case["input"] + case["expected"]
        and tokens["decoded_continuation"] == case["expected"],
        "contextual tokenizer round trip differs",
    )
    exact(row["context"], {"model_id", "model_revision", "harness"}, "record context")
    checked(
        (row["context"]["model_id"], row["context"]["model_revision"])
        == (model["id"], model["revision"]),
        "record model context differs",
    )
    harness = row["context"]["harness"]
    exact(
        harness,
        TOKEN_FIELDS
        | {"request_index", "result", "is_greedy", "input_utf8_byte_count"},
        "Harness result",
    )
    checked(
        all(
            canonical_json_bytes(harness[key]) == canonical_json_bytes(tokens[key])
            for key in TOKEN_FIELDS
        ),
        "retained tokenization differs from pinned result",
    )
    checked(
        type(harness["request_index"]) is int
        and harness["request_index"] == index
        and type(harness["input_utf8_byte_count"]) is int
        and harness["input_utf8_byte_count"] == len(case["input"].encode("utf-8")),
        "request index or context byte count differs",
    )
    result = harness["result"]
    checked(
        isinstance(result, list)
        and len(result) == 2
        and finite(result[0])
        and result[0] <= 0
        and type(result[1]) is bool
        and type(harness["is_greedy"]) is bool
        and harness["is_greedy"] == result[1],
        "invalid native Harness likelihood/greedy result",
    )
    likelihood = row["likelihood"]
    checked(
        isinstance(likelihood, dict)
        and likelihood.get("logprob_sum") == result[0]
        and type(likelihood.get("token_count")) is int
        and likelihood["token_count"] == len(tokens["continuation_token_ids"])
        and likelihood.get("configuration_digest") == configuration_digest
        and likelihood.get("tokenizer_digest") == model["tokenizer_digest"],
        "typed likelihood differs from native result or configuration",
    )


def load_role(directory, protocol, expected_protocol, expected_raw, role):
    with secure_directory(directory):
        entries = {path.name for path in directory.iterdir()}
        required = {
            "protocol.json",
            "manifest.json",
            "tokenizations.json",
            "raw-results.json",
        }
        checked(
            entries in (required, required | {"progress"}),
            "role inventory is incomplete or contains unexpected entries",
        )
        protocol_raw, retained_protocol = read(directory / "protocol.json")
        checked(
            digest(protocol_raw) == expected_protocol and retained_protocol == protocol,
            "role protocol differs from independent pin",
        )
        _, manifest = read(directory / "manifest.json")
        validate_manifest(manifest, protocol, role, expected_protocol)
        _, tokenizations = read(directory / "tokenizations.json")
        raw, result = read(directory / "raw-results.json")
        checked(
            digest(raw) == expected_raw,
            f"{role} raw results differ from independent pin",
        )
        exact(
            result,
            {"format", "source", "protocol_sha256", "role", "records", "metadata"},
            "raw results",
        )
        checked(
            result["format"] == "invarlock/harness-model-comparison-results-v1"
            and result["source"] == SOURCE
            and result["protocol_sha256"] == expected_protocol
            and result["role"] == role,
            "raw source or protocol binding differs",
        )
        metadata = result["metadata"]
        exact(
            metadata,
            {"status", "timings", "harness_loglikelihood_call_count", "case_count"},
            "capture metadata",
        )
        checked(
            metadata["status"] == "complete"
            and all(
                type(metadata[key]) is int and metadata[key] == 400
                for key in ("harness_loglikelihood_call_count", "case_count")
            ),
            "capture did not complete every planned call",
        )
        timings = metadata["timings"]
        exact(
            timings,
            {"wall_seconds", "started_unix_ns", "finished_unix_ns"},
            "capture timings",
        )
        checked(
            finite(timings["wall_seconds"])
            and timings["wall_seconds"] > 0
            and type(timings["started_unix_ns"]) is int
            and type(timings["finished_unix_ns"]) is int
            and 0 < timings["started_unix_ns"] <= timings["finished_unix_ns"],
            "invalid capture timing facts",
        )
        records = result["records"]
        checked(
            isinstance(records, list)
            and len(records) == 400
            and isinstance(tokenizations, list)
            and len(tokenizations) == 400,
            "missing or extra likelihood records",
        )
        for index, (row, tokens, case) in enumerate(
            zip(records, tokenizations, protocol["cases"], strict=True)
        ):
            validate_row(
                row,
                tokens,
                case,
                protocol["models"][role],
                object_digest(protocol["configuration"]),
                index,
            )
        if "progress" in entries:
            with secure_directory(directory / "progress"):
                checked(
                    {p.name for p in (directory / "progress").iterdir()}
                    == {
                        f"{index:06}{suffix}.json"
                        for index in range(400)
                        for suffix in ("", ".attempt")
                    },
                    "progress inventory differs from the full schedule",
                )
                for index, row in enumerate(records):
                    expected = {
                        f"{index:06}.attempt.json": canonical_json_bytes(
                            {
                                "request_index": index,
                                "case_id": row["id"],
                                "protocol_sha256": expected_protocol,
                            }
                        ),
                        f"{index:06}.json": canonical_json_bytes(row),
                    }
                    for name, expected_bytes in expected.items():
                        raw_progress = read_regular_file_bytes(
                            directory / "progress" / name,
                            label="progress record",
                            max_bytes=len(expected_bytes),
                        )
                        checked(
                            raw_progress == expected_bytes,
                            "progress admission or result differs",
                        )

    run = capture_evaluator_run(
        records,
        source=SOURCE,
        run_id=f"harness-{role}",
        artifact_digest=protocol["models"][role]["artifact_digest"],
        source_digest=expected_raw,
    )
    return run, manifest


def project_capture(capture, output, protocol_sha256, baseline_sha256, subject_sha256):
    capture, output = Path(capture).absolute(), Path(output).absolute()
    checked(
        not output.is_relative_to(capture) and not capture.is_relative_to(output),
        "capture and output directories must remain separate",
    )
    for pin in (protocol_sha256, baseline_sha256, subject_sha256):
        check_digest(pin)
    with secure_directory(capture):
        checked(
            {p.name for p in capture.iterdir()} == set(MODELS),
            "capture must contain exactly baseline and subject",
        )
        raw_protocol, protocol = read(capture / "baseline" / "protocol.json")
        checked(
            digest(raw_protocol) == protocol_sha256,
            "protocol differs from independent pin",
        )
        validate_protocol(protocol)
        baseline, left = load_role(
            capture / "baseline", protocol, protocol_sha256, baseline_sha256, "baseline"
        )
        subject, right = load_role(
            capture / "subject", protocol, protocol_sha256, subject_sha256, "subject"
        )
        checked(
            left["packages"] == right["packages"]
            and left["source_implementations"] == right["source_implementations"],
            "declared runtime versions or source inventories differ across roles",
        )
    policy = protocol["policy"]
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "path": "baseline.json",
                "adapter": "invarlock",
                "expected_run_digest": run_digest(baseline),
            },
            "subject": {
                "path": "subject.json",
                "adapter": "invarlock",
                "expected_run_digest": run_digest(subject),
            },
            "metric": "normalized_nll_per_utf8_byte",
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    anchors = {
        "format": "invarlock/harness-model-handoff-v1",
        "protocol_sha256": protocol_sha256,
        "raw_results_sha256": {"baseline": baseline_sha256, "subject": subject_sha256},
        "baseline_run_digest": run_digest(baseline),
        "subject_run_digest": run_digest(subject),
        "policy_digest": comparison_policy_digest(policy),
        "request_digest": captured_request_digest(
            normalize_captured_request(
                request, baseline=baseline, subject=subject, policy=policy
            )
        ),
        "source_assurance": "captured_inputs",
    }
    files = {
        "baseline.json": canonical_json_bytes(baseline),
        "subject.json": canonical_json_bytes(subject),
        "policy.json": canonical_json_bytes(policy),
        "request.yaml": canonical_json_bytes(request),
        "protocol.json": raw_protocol,
        "anchors.json": canonical_json_bytes(anchors),
    }
    with staged_directory(output, prefix=".harness-handoff-") as stage:
        write_run(stage.path / "baseline.json", baseline)
        write_run(stage.path / "subject.json", subject)
        for name in ("policy.json", "request.yaml", "protocol.json", "anchors.json"):
            (stage.path / name).write_bytes(files[name])
        stage.require_exact_files(files)
        stage.publish()
    return anchors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("protocol-sha256", "baseline-sha256", "subject-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        result = project_capture(
            args.capture,
            args.output,
            args.protocol_sha256,
            args.baseline_sha256,
            args.subject_sha256,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Harness handoff rejected: {exc}\n")
    print(canonical_json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
