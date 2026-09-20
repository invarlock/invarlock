#!/usr/bin/env python3
"""Replay native evaluator export contracts through an installed offline recipient.

EM/NLL use original retained 400-case Mistral measurements and policies. Judge
uses an explicitly synthetic complete measurement fixture. No evaluator SDK,
model, judge service, or runtime container is executed by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "parity_native_shapes", HERE / "native_shapes.py"
)
assert SPEC and SPEC.loader
SHAPES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHAPES)
SCORERS = ("exact_match", "normalized_nll", "judge")
INPUT_FORMATS = ("envelope", "native-json")


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def profiles():
    return {
        p["profile_id"]: p["upstream"]["version"]
        for p in read(ROOT / "examples/evaluator-qualification/matrix.json")["profiles"]
    }


def retained(scorer):
    if scorer == "judge":
        rows = [
            {
                "id": f"synthetic-{i}",
                "input": prompt,
                "expected": output,
                "output": output,
                "metadata": {"fixture": "synthetic-no-judge-called"},
            }
            for i, (prompt, output) in enumerate(
                (
                    ("Capital of France?", "Paris"),
                    ("Return the requested label.", "approved"),
                )
            )
        ]
        return (
            [
                {"records": deepcopy(rows), "artifact_digest": "sha256:" + marker * 64}
                for marker in ("a", "b")
            ],
            {},
            {
                "scope": "Synthetic two-case full judge contract; ratings are constructed, not observed."
            },
        )
    reference = read(
        ROOT / "examples/captured-results/references/langfuse/reference.json"
    )
    runs, sources = [], {}
    for side in ("baseline", "subject"):
        origin = reference["sources"][f"{scorer}-{side}.json"]
        raw = (ROOT / origin["path"]).read_bytes()
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == origin["sha256"]
        runs.append(json.loads(raw))
        sources[side] = origin
    origin_root = (ROOT / sources["baseline"]["path"]).parents[2]
    policy = read(origin_root / "evidence/inputs/policy.json")
    expected = read(origin_root / "evidence/reports/evaluation.report.json")
    return (
        runs,
        policy,
        {
            "scope": "Native-shape contract replay of retained Mistral measurements; not fresh SDK or model execution.",
            "sources": sources,
            "expected_metrics": expected["metrics"],
            "expected_decision": expected["decision"],
        },
    )


def _key(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invarlock.evidence_pack_integrity import public_key_fingerprint

    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return public_key_fingerprint(key.public_key())


def export_pair(output, evaluator, scorer, *, input_format="envelope"):
    from invarlock.engine import export_evaluator_result
    from invarlock.evaluation_records.adapters import load_run

    originals, policy, origin = retained(scorer)
    version = profiles()[evaluator]
    sources, runs = [], []
    projection = {
        "lm-evaluation-harness": "/context/arguments/0/0",
        "promptfoo": "/context/prompt",
    }.get(evaluator)
    for side, original in zip(("baseline", "subject"), originals, strict=True):
        path = output / f"{side}.json"
        options = {
            "source_version": version,
            "run_id": f"parity-{scorer}-{side}",
            "artifact_digest": original["artifact_digest"],
        }
        if "service_identity" in original:
            options["service_identity"] = original["service_identity"]
        if projection:
            options["input_projection"] = {
                "kind": "json-pointer",
                "pointer": projection,
            }
        native_payload = SHAPES.payload(
            evaluator, original["records"], version, run_id=options["run_id"]
        )
        adapter = {
            "envelope": "evaluator-json",
            "native-json": "evaluator-native-json",
        }[input_format]
        source = {
            "path": path.name,
            "adapter": adapter,
            "source": {"name": evaluator, "version": version},
            **{key: value for key, value in options.items() if key != "source_version"},
        }
        if input_format == "native-json":
            # The producer only writes its native JSON; core normalization and
            # trust preparation happen in the separate recipient environment.
            if evaluator == "langfuse":
                native_payload = native_payload["result"]
            write(path, native_payload)
            run = load_run(path, **{k: v for k, v in source.items() if k != "path"})
        else:
            run = export_evaluator_result(
                evaluator,
                native_payload,
                path,
                expected_ids=[r["id"] for r in original["records"]],
                **options,
            )
        by_id = {r["id"]: r for r in original["records"]}
        assert set(by_id) == {r["id"] for r in run["records"]}
        for row in run["records"]:
            before = by_id[row["id"]]
            for key in ("input", "expected", "output", "metadata"):
                assert row[key] == before[key], (evaluator, scorer, side, key)
            assert row.get("error") == before.get("error"), (
                evaluator,
                scorer,
                side,
                "error",
            )
            if scorer == "normalized_nll":
                assert {
                    k: v
                    for k, v in row["likelihood"].items()
                    if k not in {"source", "input_digest"}
                } == {
                    k: v
                    for k, v in before["likelihood"].items()
                    if k not in {"source", "input_digest"}
                }
        runs.append(run)
        sources.append(source)
    return sources, runs, policy, origin


def synthetic_measurements(plan, runs):
    from invarlock.judge_measurements.contracts import (
        canonical_payload,
        expected_trial_id,
        measurement_plan_digest,
        render_judge_request,
    )

    fixture = read(HERE / "synthetic-judge.json")
    digest = measurement_plan_digest(plan)
    indexed = {
        side: {r["id"]: r for r in run["records"]}
        for side, run in zip(("baseline", "subject"), runs, strict=True)
    }
    trials = []
    for binding in plan["answer_bindings"]:
        for side in ("baseline", "subject"):
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                row = indexed[side][binding["case_id"]]
                request = render_judge_request(
                    plan, input_text=row["input"], answer_text=row["output"]
                )
                index = len(trials)
                attempt = deepcopy(fixture["attempt"])
                attempt.update(
                    resolved_model=plan["judge"]["approved_resolved_models"][0],
                    request={
                        "text": request.decode(),
                        "sha256": hashlib.sha256(request).hexdigest(),
                        "media_type": "application/json",
                    },
                    request_id=f"synthetic-request-{index}",
                )
                attempt["source"].update(
                    source_id="synthetic",
                    record_index=index,
                    model_event_id=f"synthetic-event-{index}",
                )
                trials.append(
                    {
                        "trial_id": expected_trial_id(
                            digest, binding["case_id"], side, repetition
                        ),
                        "case_id": binding["case_id"],
                        "side": side,
                        "repetition": repetition,
                        "answer_sha256": binding[f"{side}_answer_sha256"],
                        "plan_sha256": digest,
                        "status": "complete",
                        "attempts": [attempt],
                        "selected_attempt": 1,
                        "parse": {"status": "ok", "rating": "correct", "value": "1"},
                    }
                )
    source = canonical_payload(
        {"format": "invarlock/retained-judge-json-v1", "trials": trials}
    )
    return {
        "format": "invarlock/judge-measurements-v1",
        "profile_id": plan["profile_id"],
        "plan_sha256": digest,
        "source_profile": "retained-judge-json-v1",
        "sources": [
            {
                "source_id": "synthetic",
                "profile": "retained-judge-json-v1",
                "encoding": "utf-8",
                "byte_size": len(source),
                "media_type": "application/json",
                "content": source.decode(),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        ],
        "trials": trials,
        "completeness": {
            "status": "complete",
            "expected_trials": len(trials),
            "recorded_trials": len(trials),
            "completed_trials": len(trials),
        },
    }


def prepare(output, evaluator, scorer, *, input_format="envelope"):
    from invarlock.engine import captured_request_digest, normalize_captured_request
    from invarlock.evaluation_records.io import run_digest

    sources, runs, policy, origin = export_pair(
        output, evaluator, scorer, input_format=input_format
    )
    fingerprint = _key(output / "signer.pem")
    _key(output / "verifier.pem")
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": sources[0],
            "subject": sources[1],
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    if scorer == "judge":
        from invarlock.judge_measurements.analysis import (
            analyze_measurements,
            decode_analysis_policy,
        )
        from invarlock.judge_measurements.captured_workflow import (
            prepare_evaluator_judge,
        )
        from invarlock.judge_measurements.evidence import DECISION_SCOPE, object_sha256

        recipe = read(HERE / "synthetic-judge.json")["recipe"]
        recipe["plan"]["sampling"]["case_units"] = [
            {"case_id": row["id"], "unit_id": row["id"]} for row in runs[0]["records"]
        ]
        plan, analysis_policy = prepare_evaluator_judge(recipe, *runs)
        measurements = synthetic_measurements(plan, runs)
        analysis = analyze_measurements(
            plan,
            measurements,
            decode_analysis_policy(analysis_policy, plan=plan),
            baseline_run=runs[0],
            subject_run=runs[1],
        ).to_dict()
        assert analysis["decision"] == "pass"
        write(output / "measurements.json", measurements)
        request["comparison"].update(
            metric="judge",
            judge={
                "workspace": "unused-judge-work",
                "signer_identity": "synthetic-parity-signer",
                "measurements": "measurements.json",
            },
        )
        trust = {
            "format": "invarlock/judge-measurement-recipient-policy-v1",
            "decision_scope": DECISION_SCOPE,
            "intended_subject": runs[1]["artifact_digest"],
            "required_metric_name": analysis_policy["metric_name"],
            "trusted_signer": {
                "identity": "synthetic-parity-signer",
                "public_key_sha256": fingerprint,
            },
            "bindings": {
                "baseline_run_sha256": run_digest(runs[0]),
                "subject_run_sha256": run_digest(runs[1]),
                "case_set_sha256": plan["case_set_sha256"],
                "plan_sha256": object_sha256(plan),
                "measurements_sha256": object_sha256(measurements),
                "analysis_policy_sha256": object_sha256(analysis_policy),
                "analysis_result_sha256": object_sha256(analysis),
            },
            "required_decision": "pass",
        }
        policy = recipe
    else:
        trust = {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "policy.json"},
            "anchors": {
                "baseline_run_digest": run_digest(runs[0]),
                "subject_run_digest": run_digest(runs[1]),
                "request_digest": captured_request_digest(
                    normalize_captured_request(
                        request, baseline=runs[0], subject=runs[1], policy=policy
                    )
                ),
                "evidence_signer_fingerprint": fingerprint,
            },
            "verifier": {
                "identity": "evaluator-parity-recipient",
                "signing_key_path": "verifier.pem",
            },
        }
    write(output / "policy.json", policy)
    write(output / "request.json", request)
    write(output / "trust.json", trust)
    write(output / "origin.json", origin)
    return runs, origin


def installed_identity(python):
    code = "import importlib.util,json,invarlock; from pathlib import Path; from sysconfig import get_path; assert Path(invarlock.__file__).resolve().is_relative_to(Path(get_path('purelib')).resolve()); names=('inspect_ai','lm_eval','langfuse','deepeval','ragas','lighteval','evaluate','pydantic_evals','autoevals','openevals','mlflow','garak','evals','phoenix','opik','azure.ai.evaluation','evidently','trulens'); absent=[]\nfor name in names:\n try: spec=importlib.util.find_spec(name)\n except ModuleNotFoundError: spec=None\n assert spec is None, name\n absent.append(name)\nprint(json.dumps({'invarlock':invarlock.__file__,'sdk_modules_absent':absent}))"
    return json.loads(
        subprocess.run(
            [str(python), "-I", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )


def journey(output, evaluator, scorer, python, *, input_format="envelope"):
    runs, origin = prepare(output, evaluator, scorer, input_format=input_format)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_SIGNING_KEY", None)
    commands = []

    def command(*args, allowed=(0,)):
        result = subprocess.run(
            [str(python), "-I", "-m", "invarlock", *map(str, args)],
            cwd=output,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        commands.append(
            {"command": list(map(str, args)), "exit_code": result.returncode}
        )
        assert result.returncode in allowed, result.stdout + result.stderr
        return json.loads(result.stdout)

    status = 7 if scorer == "normalized_nll" else 0
    decision = "regression" if status else "pass"
    evaluated = command(
        "evaluate",
        "request.json",
        "--signing-key",
        "signer.pem",
        "--fail-on-policy",
        "--json",
        allowed=(status,),
    )
    assert evaluated["authentication"] == "signed" and evaluated["decision"] == decision
    receipt_args = (
        [] if scorer == "judge" else ["--receipt", "verification.receipt.json"]
    )
    verified = command(
        "verify",
        "evidence",
        "--trust-profile",
        "trust.json",
        *receipt_args,
        "--json",
        allowed=(status,),
    )
    write(output / "verification.json", verified)
    if scorer == "judge":
        assert (
            verified["authenticated"] and verified["replayed"] and verified["accepted"]
        )
        assert not (output / "unused-judge-work").exists()
    else:
        assert verified["integrity_ok"] and verified["replay_status"] == "completed"
        assert verified["decision"] == origin["expected_decision"]
        actual = read(output / "evidence/reports/evaluation.report.json")
        # Native input wrappers can change the resampling representation. The
        # independent recipient verifies each interval; original point facts and
        # policy outcomes must remain identical.
        assert [
            {k: v for k, v in metric.items() if k != "interval"}
            for metric in actual["metrics"]
        ] == [
            {k: v for k, v in metric.items() if k != "interval"}
            for metric in origin["expected_metrics"]
        ]
    command("report", "evidence", "--html", "report.html", "--json")
    assert evaluator in (output / "report.html").read_text()
    for side, run in zip(("baseline", "subject"), runs, strict=True):
        path = (
            output
            / "evidence"
            / (f"{side}_run.json" if scorer == "judge" else f"records/{side}.json")
        )
        assert read(path) == run
    # An output-byte change must fail original independent trust, even if parseable.
    path = (
        output
        / "evidence"
        / ("subject_run.json" if scorer == "judge" else "records/subject.json")
    )
    raw = path.read_bytes()
    modified = json.loads(raw)
    modified["records"][0]["output"] = "tampered"
    mode = path.stat().st_mode & 0o777
    path.chmod(mode | 0o200)
    try:
        write(path, modified)
        rejected = command(
            "verify",
            "evidence",
            "--trust-profile",
            "trust.json",
            "--json",
            allowed=(4 if scorer == "judge" else 2,),
        )
    finally:
        path.write_bytes(raw)
        path.chmod(mode)
    assert not rejected.get("accepted", False)
    write(output / "tamper-refusal.json", rejected)
    result = {
        "evaluator": evaluator,
        "scorer": scorer,
        "input_format": input_format,
        "decision": decision,
        "verification_exit_code": status,
        "tamper_rejected": True,
        "record_count": len(runs[0]["records"]),
        "scope": origin["scope"],
        "commands": commands,
    }
    write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", choices=profiles(), required=True)
    parser.add_argument("--scorer", choices=SCORERS, required=True)
    parser.add_argument("--input-format", choices=INPUT_FORMATS, default="envelope")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipient-python", type=Path, required=True)
    args = parser.parse_args()
    identity = installed_identity(args.recipient_python)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = journey(
        output,
        args.evaluator,
        args.scorer,
        args.recipient_python.absolute(),
        input_format=args.input_format,
    )
    print(json.dumps({**result, "recipient": identity}, indent=2))


if __name__ == "__main__":
    main()
