"""Independently import live capture ledgers and publish offline comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from sysconfig import get_path

# Explicit sibling helpers remain separate from the installed recipient package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bindings  # noqa: E402
import common  # noqa: E402

ROLES = ("baseline", "subject")
SCORERS = ("exact_match", "normalized_nll")
SDK_MODULES = (
    "inspect_ai",
    "lm_eval",
    "langfuse",
    "deepeval",
    "ragas",
    "lighteval",
    "evaluate",
    "pydantic_evals",
    "autoevals",
    "openevals",
    "mlflow",
    "garak",
    "evals",
    "phoenix",
    "opik",
    "azure.ai.evaluation",
    "evidently",
    "trulens",
)


def installed_identity():
    import invarlock

    path = Path(invarlock.__file__).resolve()
    if not sys.flags.isolated or not path.is_relative_to(
        Path(get_path("purelib")).resolve()
    ):
        raise ValueError(
            "recipient requires isolated Python and an independently installed core"
        )
    for name in SDK_MODULES:
        try:
            present = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            present = False
        if present:
            raise ValueError(f"recipient environment contains evaluator SDK: {name}")
    if any(
        value
        and (
            name.endswith(("_API_KEY", "_ACCESS_TOKEN"))
            or name
            in {
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "LANGFUSE_SECRET_KEY",
                "AWS_SECRET_ACCESS_KEY",
            }
        )
        for name, value in os.environ.items()
    ):
        raise ValueError("recipient environment must not contain API credentials")
    return {"invarlock": str(path), "sdk_modules_absent": list(SDK_MODULES)}


def read(path, limit=common.MAX_MESSAGE):
    from invarlock.captured_contracts import read_file

    raw = read_file(Path(path), limit)
    return common.decode(raw), raw


def same(left, right):
    return common.encoded(left) == common.encoded(right)


def contains(value, key, expected):
    if isinstance(value, dict):
        return (key in value and same(value[key], expected)) or any(
            contains(child, key, expected) for child in value.values()
        )
    return isinstance(value, list) and any(
        contains(child, key, expected) for child in value
    )


def retained_tokenization(execution, case, configuration):
    """Check retained token facts; this does not execute or authenticate a tokenizer."""
    tokens = execution.get("tokenization")
    arrays = ("context_token_ids", "continuation_token_ids", "joined_token_ids")
    decoded = {
        "decoded_context": case["input"],
        "decoded_continuation": case["expected"],
        "decoded_joined": case["input"] + case["expected"],
    }
    maximum = configuration.get("max_length")
    generated = configuration.get("max_new_tokens")
    if (
        type(maximum) is not int
        or maximum not in (512, 1024)
        or type(generated) is not int
        or generated != 32
        or not isinstance(tokens, dict)
        or set(tokens) != set(arrays) | set(decoded)
    ):
        raise ValueError("missing or unsupported retained tokenization/configuration")
    for name in arrays:
        values = tokens[name]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= maximum + 1
            or any(type(value) is not int or not 0 <= value < 2**31 for value in values)
        ):
            raise ValueError(
                "retained token IDs must be bounded nonnegative integer sequences"
            )
    if (
        tokens["joined_token_ids"]
        != tokens["context_token_ids"] + tokens["continuation_token_ids"]
        or any(tokens[name] != value for name, value in decoded.items())
        or len(tokens["context_token_ids"]) + generated > maximum
    ):
        raise ValueError(
            "retained tokenization differs from complete original task text or boundary"
        )
    expected = {"until": [], "max_gen_toks": generated, "do_sample": False}
    if not same(execution.get("generation_parameters"), expected):
        raise ValueError(
            "retained generation parameters differ from the admitted configuration"
        )
    return len(tokens["continuation_token_ids"])


def capture(directory, protocol, role, evaluator):
    """Check every admitted request and response before normalizing native data."""
    from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("capture must be a real directory")
    manifest, manifest_raw = read(directory / "capture.json")
    recorded_protocol, protocol_raw = read(directory / "protocol.json")
    expected_protocol = {**protocol, "role": role}
    version = protocol["versions"][evaluator]
    if not same(recorded_protocol, expected_protocol) or any(
        not same(manifest.get(key), value)
        for key, value in {
            "format": "invarlock/live-evaluator-capture-v1",
            "status": "captured",
            "evaluator": evaluator,
            "version": version,
            "role": role,
            "protocol_digest": common.digest(expected_protocol),
            "model": protocol["models"][role],
            "case_count": len(protocol["cases"]),
        }.items()
    ):
        raise ValueError(
            "capture manifest or role protocol differs from the admitted campaign"
        )
    native, raw = read(directory / "native.json", MAX_INPUT_BYTES)
    if manifest.get("native_sha256") != "sha256:" + hashlib.sha256(raw).hexdigest():
        raise ValueError("native capture bytes differ from manifest")
    originals = {
        "capture.json": manifest_raw,
        "protocol.json": protocol_raw,
        "native.json": raw,
    }
    results, expected_files = {}, set()
    ledger_bytes = 0
    taskdir = directory / "tasks"
    if taskdir.is_symlink() or not taskdir.is_dir():
        raise ValueError("task ledger must be a real directory")
    for case in protocol["cases"]:
        request = {
            "evaluator": evaluator,
            "case_id": case["id"],
            "protocol_digest": common.digest(expected_protocol),
        }
        stem = common.digest(request).removeprefix("sha256:")
        entries = {}
        for suffix in ("request", "response"):
            name = f"{stem}.{suffix}.json"
            expected_files.add(name)
            entries[suffix], originals["tasks/" + name] = read(taskdir / name)
            ledger_bytes += len(originals["tasks/" + name])
            if ledger_bytes > MAX_INPUT_BYTES:
                raise ValueError("task ledger exceeds the bounded capture size")
        response = entries["response"]
        if (
            not same(entries["request"], request)
            or not isinstance(response, dict)
            or set(response) != {"request", "result"}
            or not same(response["request"], request)
        ):
            raise ValueError("task ledger differs from its admitted request")
        result = response["result"]
        if (
            not isinstance(result, dict)
            or not {"output", "metadata"} <= result.keys()
            or result.keys() - {"output", "metadata", "error"}
            or not isinstance(result["metadata"], dict)
            or (result["output"] is not None and not isinstance(result["output"], str))
            or (
                "error" in result
                and (not isinstance(result["error"], str) or not result["error"])
            )
        ):
            raise ValueError("invalid task observations")
        execution = result["metadata"].get("invarlock_model_execution", {})
        required = {
            "request": request,
            "protocol_digest": request["protocol_digest"],
            "model": protocol["models"][role],
            "configuration": protocol["configuration"],
            "source": {"name": "lm-eval", "version": "0.4.12"},
        }
        if not isinstance(execution, dict) or any(
            not same(execution.get(key), value) for key, value in required.items()
        ):
            raise ValueError(
                "model execution differs from admitted model, configuration or request"
            )
        token_count = retained_tokenization(execution, case, protocol["configuration"])
        if result["output"] is not None and not same(
            execution.get("generation_result"), [result["output"]]
        ):
            raise ValueError("generation observation differs from retained output")
        facts = result["metadata"].get("invarlock_likelihood")
        if facts is not None:
            observed = execution.get("likelihood_result")
            if (
                not isinstance(facts, dict)
                or type(facts.get("token_count")) is not int
                or facts["token_count"] != token_count
                or not isinstance(observed, list)
                or len(observed) != 2
                or type(observed[1]) is not bool
                or not same(observed[0], facts.get("logprob_sum"))
            ):
                raise ValueError(
                    "likelihood observation differs from retained measurement"
                )
        elif result["output"] is None and not result.get("error"):
            raise ValueError("task has neither an observation nor a failure")
        bindings.bind_result(result, case, evaluator, version)
        results[case["id"]] = result
    if {path.name for path in taskdir.iterdir()} != expected_files:
        raise ValueError("task ledger differs from the complete planned schedule")
    if "transport_recovery" in manifest or any(
        "invarlock_transport_replay" in result["metadata"]
        for result in results.values()
    ):
        common.module("recovery").verify_capture(
            directory, manifest, protocol, role, evaluator, results, originals
        )
    return native, originals, results


def key(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invarlock.evidence_pack_integrity import public_key_fingerprint

    private = Ed25519PrivateKey.generate()
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as stream:
        stream.write(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    path.chmod(0o600)
    return public_key_fingerprint(private.public_key())


def prepare(
    protocol_path,
    protocol_sha256,
    baseline_capture,
    subject_capture,
    evaluator,
    route,
    output,
    judge_recipe=None,
):
    from invarlock.engine import (
        captured_request_digest,
        export_evaluator_result,
        load_run,
        normalize_captured_request,
    )
    from invarlock.evaluation_records.io import run_digest

    protocol, protocol_raw = read(protocol_path)
    if common.digest(protocol) != protocol_sha256:
        raise ValueError(
            "protocol differs from its independently supplied semantic digest"
        )
    common.cases(protocol["cases"])
    if (
        evaluator not in protocol["evaluators"]
        or route not in ("envelope", "native-json")
        or set(protocol["acceptance"]) != set(SCORERS)
    ):
        raise ValueError("recipient selection is outside the admitted campaign")
    captures = [
        capture(path, protocol, role, evaluator)
        for role, path in zip(ROLES, (baseline_capture, subject_capture), strict=True)
    ]
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_raw)
    runs, sources = [], []
    version = protocol["versions"][evaluator]
    for role, (native, originals, results) in zip(ROLES, captures, strict=True):
        retained = output / "captures" / role
        (retained / "tasks").mkdir(parents=True)
        for name, raw in originals.items():
            (retained / name).parent.mkdir(parents=True, exist_ok=True)
            (retained / name).write_bytes(raw)
        run_id = (
            native["run_name"]
            if evaluator == "langfuse"
            else f"live-{evaluator}-{role}"
        )
        source = {
            "path": f"{role}.json",
            "adapter": "evaluator-json"
            if route == "envelope"
            else "evaluator-native-json",
            "source": {"name": evaluator, "version": version},
            "run_id": run_id,
            "artifact_digest": protocol["models"][role]["artifact_digest"],
        }
        if bindings.projection(evaluator):
            source["input_projection"] = bindings.projection(evaluator)
        path = output / f"{role}.json"
        if route == "native-json":
            path.write_bytes(originals["native.json"])
            run = load_run(path, **{k: v for k, v in source.items() if k != "path"})
        else:
            if evaluator == "langfuse":
                native = {
                    "format": "invarlock/langfuse-export-v1",
                    "sdk_version": version,
                    "result": native,
                }
            run = export_evaluator_result(
                evaluator,
                native,
                path,
                expected_ids=[case["id"] for case in protocol["cases"]],
                source_version=version,
                **{
                    k: v
                    for k, v in source.items()
                    if k not in ("path", "adapter", "source")
                },
            )
        indexed = {row["id"]: row for row in run["records"]}
        if len(indexed) != len(run["records"]) or set(indexed) != set(results):
            raise ValueError(
                "normalized capture differs from the complete planned schedule"
            )
        for case in protocol["cases"]:
            row, result = indexed[case["id"]], results[case["id"]]
            bound = bindings.check_record(
                row, result, case, evaluator, version, cases=protocol["cases"]
            )
            if any(
                not contains(row, name, value)
                for name, value in case["metadata"].items()
            ):
                raise ValueError(
                    "native export omitted or changed frozen case metadata"
                )
            retained_metadata = {
                "invarlock_model_execution": result["metadata"][
                    "invarlock_model_execution"
                ],
                "invarlock_task_outcome": {
                    "output": result["output"],
                    "error": result.get("error"),
                },
            }
            if "invarlock_capture_binding" in bound["metadata"]:
                retained_metadata["invarlock_capture_binding"] = bound["metadata"][
                    "invarlock_capture_binding"
                ]
            if "invarlock_transport_replay" in result["metadata"]:
                retained_metadata["invarlock_transport_replay"] = result["metadata"][
                    "invarlock_transport_replay"
                ]
            if "invarlock_serialization_binding" in bound["metadata"]:
                retained_metadata["invarlock_serialization_binding"] = bound[
                    "metadata"
                ]["invarlock_serialization_binding"]
            if any(
                not contains(row, name, value)
                for name, value in retained_metadata.items()
            ):
                raise ValueError(
                    "native export omitted or changed original execution metadata"
                )
        runs.append(run)
        sources.append(source)
    for scorer in SCORERS:
        directory = output / scorer
        directory.mkdir()
        for role in ROLES:
            (directory / f"{role}.json").write_bytes(
                (output / f"{role}.json").read_bytes()
            )
        policy = protocol["acceptance"][scorer]
        fingerprint = key(directory / "signer.pem")
        key(directory / "verifier.pem")
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
                "identity": "live-evaluator-recipient",
                "signing_key_path": "verifier.pem",
            },
        }
        for name, value in (("request", request), ("policy", policy), ("trust", trust)):
            common.write(directory / f"{name}.json", value)
    if judge_recipe:
        from invarlock.judge_measurements.captured_workflow import (
            prepare_evaluator_judge,
        )

        recipe, recipe_raw = read(judge_recipe)
        plan, analysis = prepare_evaluator_judge(recipe, *runs)
        (output / "judge-recipe.json").write_bytes(recipe_raw)
        common.write(output / "judge-plan.json", plan)
        common.write(output / "judge-analysis-policy.json", analysis)
    return protocol, runs


def command(directory, *args, allowed=(0,)):
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "INVARLOCK_SIGNING_KEY"}
    }
    process = subprocess.run(
        [sys.executable, "-I", "-m", "invarlock", *args],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if process.returncode not in allowed:
        raise RuntimeError(
            f"recipient command failed ({process.returncode}): {process.stdout}{process.stderr}"
        )
    return common.decode(process.stdout.encode())


def journey(output):
    results = {}
    for scorer in SCORERS:
        directory = Path(output) / scorer
        preflight = command(
            directory,
            "evaluate",
            "request.json",
            "--preflight",
            "--signing-key",
            "signer.pem",
            "--json",
        )
        if not preflight.get("ok") or (directory / "evidence").exists():
            raise ValueError("offline evaluation preflight did not succeed cleanly")
        common.write(directory / "preflight.json", preflight)
        evaluation = command(
            directory,
            "evaluate",
            "request.json",
            "--signing-key",
            "signer.pem",
            "--fail-on-policy",
            "--json",
            allowed=(0, 7),
        )
        verified = command(
            directory,
            "verify",
            "evidence",
            "--trust-profile",
            "trust.json",
            "--receipt",
            "verification.receipt.json",
            "--json",
            allowed=(0, 7),
        )
        if (
            evaluation.get("authentication") != "signed"
            or not verified.get("integrity_ok")
            or verified.get("replay_status") != "completed"
            or evaluation["decision"] != verified["decision"]
        ):
            raise ValueError(
                "independent verification did not authenticate and replay the evaluation"
            )
        common.write(directory / "verification.json", verified)
        rendering = command(
            directory,
            "report",
            "evidence",
            "--html",
            "report.html",
            "--markdown",
            "report.md",
            "--junit",
            "report.xml",
            "--explain",
            "--json",
        )
        common.write(directory / "report.json", rendering)
        results[scorer] = {
            "decision": evaluation["decision"],
            "verification": verified,
            "reports": ["report.html", "report.md", "report.xml", "report.json"],
        }
    common.write(Path(output) / "result.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "baseline-capture", "subject-capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--route", choices=("envelope", "native-json"), required=True)
    parser.add_argument("--judge-recipe", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    identity = installed_identity()
    prepare(
        args.protocol,
        args.protocol_sha256,
        args.baseline_capture,
        args.subject_capture,
        args.evaluator,
        args.route,
        args.output,
        args.judge_recipe,
    )
    common.write(args.output / "recipient.json", identity)
    if not args.prepare_only:
        print(common.encoded(journey(args.output)).decode())


if __name__ == "__main__":
    main()
