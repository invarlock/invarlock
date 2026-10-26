"""Generate nonconstant synthetic captured NLL through public installed APIs."""

from __future__ import annotations

import json
from pathlib import Path

METRIC = "normalized_nll_per_utf8_byte"


def scores(count):
    baseline = [1.0 + (i % 17) / 19 + (i % 5) / 37 for i in range(count)]
    subject = [value * (0.96 + (i % 7) / 100) for i, value in enumerate(baseline)]
    return baseline, subject


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def make_fixture(root, count, scenario="pass"):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invarlock.engine import (
        captured_request_digest,
        case_set_digest,
        digest,
        evaluate_request_file,
        freeze_case_set,
        make_run,
        normalize_captured_request,
        run_digest,
        write_run,
    )
    from invarlock.evidence_pack_integrity import public_key_fingerprint

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source = {"name": "synthetic-nll-fixture", "version": "1"}
    configuration, tokenizer = (
        digest("offline-fixture-config"),
        digest("offline-tokenizer"),
    )
    left, right = scores(count)
    if scenario == "regression":
        right = [value * 1.25 for value in right]
    elif scenario != "pass":
        raise ValueError(f"unsupported fixture scenario: {scenario}")
    write_json(
        root / "expected.json",
        {"decision": "pass" if scenario == "pass" else "regression"},
    )
    runs = {}
    for side, values in (("baseline", left), ("subject", right)):
        artifact = digest(f"synthetic-{side}")
        records = []
        for i, nll in enumerate(values):
            row = {
                "id": f"case-{i:05d}",
                "input": f"question {i}",
                "expected": "é",
                "output": None,
                "metadata": {"group": "all"},
            }
            row["likelihood"] = {
                "basis": "reference_continuation",
                "logprob_sum": -nll * 2,
                "token_count": 1,
                "utf8_byte_count": 2,
                "input_digest": digest(row["input"]),
                "reference_digest": digest(row["expected"]),
                "artifact_digest": artifact,
                "configuration_digest": configuration,
                "tokenizer_digest": tokenizer,
                "source": source,
            }
            records.append(row)
        runs[side] = make_run(
            records, source=source, run_id=side, artifact_digest=artifact
        )
        write_run(root / f"{side}.json", runs[side])
    cases = freeze_case_set(
        [
            {key: row[key] for key in ("id", "input", "expected", "metadata")}
            for row in runs["baseline"]["records"]
        ]
    )
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "nll",
                "kind": METRIC,
                "configuration": {
                    "configuration_digest": configuration,
                    "baseline_tokenizer_digest": tokenizer,
                    "subject_tokenizer_digest": tokenizer,
                },
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": count,
                "ratio_max": 1.1,
                "maximum_interval_width": 0.25,
            }
        ],
        "slices": [],
        "expected_case_set_digest": case_set_digest(cases),
    }
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "policy.json",
            "metric": METRIC,
        },
        "output": {"evidence": "pack"},
    }
    write_json(root / "policy.json", policy)
    write_json(root / "request.json", request)
    keys = {}
    for role in ("signer", "verifier"):
        key = Ed25519PrivateKey.generate()
        keys[role] = public_key_fingerprint(key.public_key())
        path = root / f"{role}.pem"
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        path.chmod(0o600)
    normalized = normalize_captured_request(
        request, baseline=runs["baseline"], subject=runs["subject"], policy=policy
    )
    # These pins are constructed before publication, outside the evidence directory.
    pins = {
        "expected_baseline_run": run_digest(runs["baseline"]),
        "expected_subject_run": run_digest(runs["subject"]),
        "expected_request_digest": captured_request_digest(normalized),
        "expected_signer": keys["signer"],
        "expected_verifier_fingerprint": keys["verifier"],
        "expected_verifier_identity": "offline-comparison-experiment",
    }
    write_json(root / "trust-pins.json", pins)
    evaluate_request_file(root / "request.json", signing_key_path=root / "signer.pem")
    # Derive the kernel schedule from the same public canonical digest as likelihood replay.
    schedule = digest(
        [
            {k: row[k] for k in ("id", "input", "expected", "metadata")}
            for row in runs["baseline"]["records"]
        ]
    ).removeprefix("sha256:")
    kernel = {
        "metric": METRIC,
        "baseline_scores": left,
        "subject_scores": right,
        "schedule_sha256": schedule,
    }
    write_json(root / "kernel.json", kernel)
    return {
        "count": count,
        "kind": "synthetic_captured_nll",
        "scenario": scenario,
        "pins": pins,
    }
