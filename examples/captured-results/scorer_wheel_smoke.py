"""Rehearse three captured scorers with an installed core wheel and no provider SDK.

Run outside the checkout with --fixture pointing to the judge-measurements
example and --cli pointing to the candidate wheel's executable. The retained
one-case judge example remains insufficient evidence; this rehearsal never
collects new judgments or changes its acceptance bounds. The NLL rows contain
authored synthetic likelihood facts for contract testing, not measured model
evidence. No model inference or provider request occurs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from invarlock.engine import (
    capture_evaluator_run,
    captured_request_digest,
    evaluator_input_capabilities,
    import_judge_sources,
    normalize_captured_request,
    prepare_evaluator_judge,
    run_digest,
    write_run,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.judge_measurements.analysis import (
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.contracts import measurement_plan_digest
from invarlock.judge_measurements.evidence import object_sha256


def require_core_only() -> None:
    for name in ("inspect_ai", "openai"):
        try:
            available = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            available = False
        if available or name in sys.modules:
            raise RuntimeError(f"core-only rehearsal found optional module: {name}")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="invarlock")
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    require_core_only()
    executable = shutil.which(args.cli)
    if executable is None:
        raise SystemExit(
            "Install the candidate wheel and supply its invarlock executable"
        )
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "INVARLOCK_SIGNING_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        environment.pop(name, None)
    environment.update(PYTHONSAFEPATH="1", PYTHONNOUSERSITE="1")
    with tempfile.TemporaryDirectory(prefix="invarlock-three-scorer-") as directory:
        root = Path(directory).resolve()

        def run(*arguments: str, expected: int = 0) -> dict:
            completed = subprocess.run(
                [executable, *arguments, "--json"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if completed.returncode != expected:
                raise RuntimeError(
                    f"{arguments[0]} returned {completed.returncode}, expected {expected}: {completed.stdout}{completed.stderr}"
                )
            return json.loads(completed.stdout)

        def save(path: Path, value: object) -> None:
            path.write_bytes(canonical_json_bytes(value))

        def fixture(name: str) -> dict:
            return json.loads((args.fixture / f"{name}.json").read_bytes())

        signer = run("evaluate", "--keygen", "signer")["details"]
        verifier = run("evaluate", "--keygen", "verifier")["details"]
        captured = []
        for side in ("baseline", "subject"):
            original = fixture(side + "_run")
            value = capture_evaluator_run(
                original["records"],
                source=original["source"],
                run_id=original["run_id"],
                artifact_digest=original["artifact_digest"],
                source_digest=original["source_digest"],
                score_provenance=original["score_provenance"],
            )
            assert value == original
            assert evaluator_input_capabilities(value)["judge"]["usable_count"] == 1
            captured.append(value)

        for kind in ("exact_match", "normalized_nll_per_utf8_byte", "judge"):
            project = root / kind
            project.mkdir()
            baseline, subject = copy.deepcopy(captured)
            request = {
                "format_version": "invarlock/evaluation-request-v2",
                "execution": {"mode": "captured"},
                "comparison": {
                    "baseline": {"path": "baseline.json", "adapter": "invarlock"},
                    "subject": {"path": "subject.json", "adapter": "invarlock"},
                    "metric": kind,
                    "policy": "policy.json",
                },
                "output": {"evidence": "evidence"},
            }
            metric = {
                "name": kind,
                "kind": kind,
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 1,
                "maximum_interval_width": 2,
            }
            if kind == "normalized_nll_per_utf8_byte":
                configuration, tokenizer = (
                    digest("reference-likelihood-config"),
                    digest("reference-tokenizer"),
                )
                for value in (baseline, subject):
                    for row in value["records"]:
                        row["likelihood"] = {
                            "basis": "reference_continuation",
                            "logprob_sum": -5,
                            "token_count": 1,
                            "utf8_byte_count": len(row["expected"].encode("utf-8")),
                            "input_digest": digest(row["input"]),
                            "reference_digest": digest(row["expected"]),
                            "artifact_digest": value["artifact_digest"],
                            "configuration_digest": configuration,
                            "tokenizer_digest": tokenizer,
                            "source": value["source"],
                        }
                    assert (
                        evaluator_input_capabilities(value)[kind]["usable_count"] == 1
                    )
                del metric["maximum_regression"]
                metric.update(
                    direction="lower",
                    unit="nats_per_utf8_byte",
                    ratio_max=1.1,
                    configuration={
                        "configuration_digest": configuration,
                        "baseline_tokenizer_digest": tokenizer,
                        "subject_tokenizer_digest": tokenizer,
                    },
                )
            policy = {
                "format": "invarlock/comparison-policy-v1",
                "metrics": [metric],
                "slices": [],
            }
            if kind == "judge":
                original_plan = fixture("plan")
                plan_template = copy.deepcopy(original_plan)
                for name in (
                    "case_set_sha256",
                    "baseline_run_sha256",
                    "subject_run_sha256",
                    "answer_bindings",
                ):
                    del plan_template[name]
                del plan_template["rubric"]["sha256"]
                del plan_template["schedule"]["expected_trials"]
                analysis_template = fixture("analysis_policy")
                del analysis_template["plan_sha256"]
                policy = {
                    "format": "invarlock/native-judge-policy-v1",
                    "plan": plan_template,
                    "analysis": analysis_template,
                    "collection": fixture("collection"),
                    "runner": {
                        "scorer_id": "factual-correctness",
                        "invocation_timeout_seconds": 30,
                    },
                }
                plan, analysis_policy = prepare_evaluator_judge(
                    policy, baseline, subject
                )
                assert plan == original_plan
                retained = fixture("measurements")
                measurements = import_judge_sources(
                    {
                        source["source_id"]: source["content"].encode("utf-8")
                        for source in retained["sources"]
                    },
                    plan=plan,
                    baseline_run=baseline,
                    subject_run=subject,
                )
                assert measurements == retained
                save(project / "measurements.json", measurements)
                request["comparison"]["judge"] = {
                    "workspace": "judge-work",
                    "signer_identity": "example-signer",
                    "measurements": "measurements.json",
                }
                analysis = analyze_measurements(
                    plan,
                    measurements,
                    decode_analysis_policy(analysis_policy, plan=plan),
                    baseline_run=baseline,
                    subject_run=subject,
                ).to_dict()
                trust = {
                    "format": "invarlock/judge-measurement-recipient-policy-v1",
                    "decision_scope": "bounded-judge-fixed-benchmark-v1",
                    "intended_subject": subject["artifact_digest"],
                    "required_metric_name": analysis_policy["metric_name"],
                    "trusted_signer": {
                        "identity": "example-signer",
                        "public_key_sha256": signer["public_key_fingerprint"],
                    },
                    "bindings": {
                        "baseline_run_sha256": run_digest(baseline),
                        "subject_run_sha256": run_digest(subject),
                        "case_set_sha256": plan["case_set_sha256"],
                        "plan_sha256": measurement_plan_digest(plan),
                        "measurements_sha256": object_sha256(measurements),
                        "analysis_policy_sha256": object_sha256(analysis_policy),
                        "analysis_result_sha256": object_sha256(analysis),
                    },
                    "required_decision": "pass",
                }
            else:
                trust = {
                    "format": "invarlock/trust-inputs-v2",
                    "kind": "captured",
                    "policy": {"path": f"{kind}/policy.json"},
                    "anchors": {
                        "baseline_run_digest": run_digest(baseline),
                        "subject_run_digest": run_digest(subject),
                        "request_digest": captured_request_digest(
                            normalize_captured_request(
                                request,
                                baseline=baseline,
                                subject=subject,
                                policy=policy,
                            )
                        ),
                        "evidence_signer_fingerprint": signer["public_key_fingerprint"],
                    },
                    "verifier": {
                        "identity": "recipient",
                        "signing_key_path": "verifier/private.pem",
                    },
                }
            write_run(project / "baseline.json", baseline)
            write_run(project / "subject.json", subject)
            save(project / "policy.json", policy)
            save(project / "request.yaml", request)
            trust_path = root / f"{kind}-trust.json"
            save(trust_path, trust)
            preflight = run(
                "evaluate",
                str(project / "request.yaml"),
                "--signing-key",
                signer["private_key"],
                "--preflight",
            )
            assert preflight["ok"] and not (project / "evidence").exists()
            created = run(
                "evaluate",
                str(project / "request.yaml"),
                "--signing-key",
                signer["private_key"],
            )
            assert created["decision"] == (
                "insufficient_evidence" if kind == "judge" else "pass"
            )
            evidence = project / "evidence"
            original_bytes = {
                p.relative_to(evidence): p.read_bytes()
                for p in evidence.rglob("*")
                if p.is_file()
            }
            verify_options = (
                [
                    "--verifier-signing-key",
                    verifier["private_key"],
                    "--verifier-identity",
                    "recipient",
                ]
                if kind == "judge"
                else []
            )
            receipt_path = root / f"{kind}.receipt.json"
            verified = run(
                "verify",
                str(evidence),
                "--trust-profile",
                str(trust_path),
                "--receipt",
                str(receipt_path),
                *verify_options,
                expected=7 if kind == "judge" else 0,
            )
            assert verified["replayed"] if kind == "judge" else verified["integrity_ok"]
            receipt = json.loads(receipt_path.read_bytes())["statement"]
            if kind == "judge":
                assert (
                    not verified["accepted"]
                    and receipt["result"]["bindings"] == trust["bindings"]
                )
            else:
                assert receipt["verification_scope"] == "captured_comparison"
                assert (
                    receipt["scoring_assurance"][0]["scoring_assurance"] == "recomputed"
                )
            report = run(
                "report",
                str(evidence),
                "--html",
                str(project / "report.html"),
                "--markdown",
                str(project / "report.md"),
            )
            assert report["ok"]
            for extension in ("html", "md"):
                assert (project / f"report.{extension}").read_bytes()
            assert original_bytes == {
                p.relative_to(evidence): p.read_bytes()
                for p in evidence.rglob("*")
                if p.is_file()
            }
            assert not (project / "judge-work").exists()
            print(
                f"{kind}: installed SDK, v2 evaluation, independent receipt and reports pass"
            )
    require_core_only()


if __name__ == "__main__":
    main()
