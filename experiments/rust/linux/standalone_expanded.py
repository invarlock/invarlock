"""Expanded fresh-process Python, standalone Rust and Python/Rust CLI comparison.

Five balanced rounds follow one qualification per implementation/workload. All
attempts retained. Any mismatch, timeout or launch failure stops the run; no
retry or exclusion. Package source identity and ZIP container metadata differ
by implementation; every other delivered file and every ZIP member is checked.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("INVARLOCK_COMPARISON_ROOT", "/work/standalone-v4"))
SOURCE = ROOT / "source"
PYTHON = Path("/work/optimization-v5-candidate/venv/bin/python")
PYTHON_FFI = ROOT / "ffi-venv/bin/python"
NATIVE = ROOT / "target/release/invarlock-native-verify"
SUPERVISOR = Path("/work/optimization-v5-candidate/measure_v2")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "experiments/rust/python"))
full_fixtures = importlib.import_module("invarlock_rust_experiment.full_fixtures")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, v):
    path.write_text(json.dumps(v, indent=2, ensure_ascii=False) + "\n")


def normalize(v, output):
    if isinstance(v, dict):
        return {k: normalize(x, output) for k, x in v.items()}
    if isinstance(v, list):
        return [normalize(x, output) for x in v]
    if isinstance(v, str):
        return v.replace(str(output), "$OUTPUT")
    return v


def verify_args(config, family):
    first = config["steps"][0]["config"]
    args = ["verify", first["evidence_path"]]
    if family in ["judge", "evidence_set"]:
        args += ["--trust-profile", first["recipient_policy_path"]]
    else:
        args += ["--policy", first["policy_path"]]
        for key, value in first.items():
            if key.startswith("expected_"):
                args += ["--" + key.replace("_", "-"), value]
    if family == "judge":
        receipt = config["steps"][1]["config"]
        args += [
            "--verifier-signing-key",
            receipt["verifier_signing_key_path"],
            "--verifier-identity",
            receipt["verifier_identity"],
        ]
    elif family != "evidence_set":
        args += [
            "--verifier-signing-key",
            first["verifier_signing_key_path"],
            "--verifier-identity",
            first["verifier_identity"],
        ]
    return args + ["--receipt", "$OUTPUT/receipt.json", "--json"]


def prepare_original():
    fixtures = ROOT / "fixtures"
    fixtures.mkdir()
    workloads = []
    for count in [1000, 5000, 12000, 50000]:
        root = fixtures / f"nll-{count}"
        full_fixtures.captured(root, count, "pass")
        workloads.append(
            {
                "name": f"evaluate-nll-{count}",
                "args": [
                    "evaluate",
                    str(root / "request.json"),
                    "--output",
                    str(root / "OUTPUT_PLACEHOLDER"),
                    "--signing-key",
                    str(root / "signer.pem"),
                    "--json",
                ],
                "evaluation_root": str(root),
            }
        )
    for family in ["captured", "native", "judge", "evidence_set"]:
        root = fixtures / family
        cfg = (
            full_fixtures.captured(root, 12, "pass")
            if family == "captured"
            else getattr(full_fixtures, family)(root)
        )
        evidence = cfg["steps"][0]["config"]["evidence_path"]
        context = root / "context.json"
        write(
            context,
            {
                "format": "invarlock/client-review-context-v1",
                "title": "Retained review",
                "review_date": "2026-10-28",
                "byline": "Reviewer",
                "context": [["Intended use", "Offline parity fixture"]],
                "next_action": "Verify independently",
            },
        )
        workloads += [
            {"name": f"verify-{family}", "args": verify_args(cfg, family)},
            {
                "name": f"report-{family}",
                "args": [
                    "report",
                    evidence,
                    "--html",
                    "$OUTPUT/report.html",
                    "--markdown",
                    "$OUTPUT/report.md",
                    "--junit",
                    "$OUTPUT/report.xml",
                    "--explain",
                    "--json",
                ],
            },
            {
                "name": f"package-{family}",
                "args": [
                    "report",
                    evidence,
                    "--package",
                    "$OUTPUT/package",
                    "--review-context",
                    str(context),
                    "--json",
                ],
            },
        ]
    write(ROOT / "workloads.json", workloads)
    return workloads


def prepare():
    """Reuse the qualified old matrix and add retained imports and tiny inputs."""
    import shutil

    import yaml

    from tests.cli.test_import_journey import _key, _materialize_request

    fixtures = ROOT / "fixtures"
    fixtures.mkdir()
    workloads = json.loads(Path("/work/standalone-v3/workloads.json").read_bytes())
    for workload in workloads:
        workload["candidates"] = (
            ["python", "rust"]
            if workload["name"].startswith("package-")
            else ["python", "rust-ffi", "rust"]
        )

    def add(name, root):
        workloads.append(
            {
                "name": name,
                "args": [
                    "evaluate",
                    str(root / "request.json"),
                    "--output",
                    str(root / "OUTPUT_PLACEHOLDER"),
                    "--signing-key",
                    str(root / "signer.pem"),
                    "--json",
                ],
                "evaluation_root": str(root),
                "candidates": ["python", "rust-ffi", "rust"],
            }
        )

    for count in [1, 4, 12]:
        root = fixtures / f"tiny-{count}"
        full_fixtures.captured(root, count, "pass")
        add(f"evaluate-nll-{count}", root)
    for count in [12, 5000]:
        root = fixtures / f"jsonl-{count}"
        full_fixtures.captured(root, count, "pass")
        request = json.loads((root / "request.json").read_bytes())
        for side in ["baseline", "subject"]:
            run = json.loads((root / f"{side}.json").read_bytes())
            (root / f"{side}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in run["records"])
            )
            request["comparison"][side] = {
                "path": f"{side}.jsonl",
                "adapter": "jsonl",
                **{k: run[k] for k in ["source", "run_id", "artifact_digest"]},
            }
        write(root / "request.json", request)
        add(f"evaluate-jsonl-{count}", root)
    root = fixtures / "deepeval-1000"
    full_fixtures.captured(root, 1000, "pass")
    request = json.loads((root / "request.json").read_bytes())
    for side in ["baseline", "subject"]:
        run = json.loads((root / f"{side}.json").read_bytes())
        source = {"name": "deepeval", "version": "fixture-1"}
        rows = []
        for row in run["records"]:
            row["likelihood"]["source"] = source
            rows.append(
                {
                    "id": row["id"],
                    "test_case": {
                        "input": row["input"],
                        "expected_output": row["expected"],
                        "actual_output": row["output"],
                    },
                    "metric_result": {"score": 1.0, "reason": "fixture"},
                    "metadata": {
                        **row["metadata"],
                        "invarlock_likelihood": row["likelihood"],
                    },
                }
            )
        write(root / f"{side}.export.json", rows)
        request["comparison"][side] = {
            "path": f"{side}.export.json",
            "adapter": "evaluator-native-json",
            "source": source,
            "run_id": side,
            "artifact_digest": run["artifact_digest"],
        }
    write(root / "request.json", request)
    add("evaluate-deepeval-1000", root)
    root = fixtures / "runtime-import"
    root.mkdir()
    material = _materialize_request(root)
    _key(root / "signer.pem")
    write(root / "request.json", yaml.safe_load(Path(material["request"]).read_text()))
    add("evaluate-runtime-import", root)
    root = fixtures / "judge-import"
    root.mkdir()
    for name in [
        "plan",
        "measurements",
        "baseline_run",
        "subject_run",
        "analysis_policy",
    ]:
        shutil.copyfile(
            SOURCE / "tests/fixtures/judge_measurements" / f"{name}.json",
            root / f"{name}.json",
        )
    _key(root / "signer.pem")
    write(
        root / "request.json",
        {
            "format_version": "invarlock/evaluation-request-v3",
            "execution": {"mode": "judge_import", "collection": None},
            "comparison": {
                "plan": "plan.json",
                "measurements": "measurements.json",
                "baseline_run": "baseline_run.json",
                "subject_run": "subject_run.json",
                "policy": "analysis_policy.json",
            },
            "output": {"evidence": "pack", "signer_identity": "comparison-signer"},
        },
    )
    add("evaluate-judge-import", root)
    write(ROOT / "workloads.json", workloads)
    # Identity hashing occurs outside every measured child. Retain immutable input
    # bytes only; output directories from prior runs are not fixture inputs.
    manifests = {}
    for base in [Path("/work/standalone-v3/fixtures"), fixtures]:
        for path in sorted(base.rglob("*")):
            if path.is_file() and not any(
                part.startswith(
                    ("qualification-", "0-", "1-", "2-", "3-", "4-", "expanded-")
                )
                for part in path.relative_to(base).parts
            ):
                manifests[str(path)] = sha(path)
    write(ROOT / "fixture-manifest.json", manifests)
    return workloads


def identity():
    info = json.loads(
        subprocess.check_output(
            [
                str(PYTHON),
                "-c",
                'import invarlock,importlib.metadata,json;print(json.dumps({"version":importlib.metadata.version("invarlock"),"source":invarlock.__file__}))',
            ]
        )
    )
    if info["version"] != "0.17.3":
        raise RuntimeError(info)
    package = Path(info["source"]).parent
    ffi_info = json.loads(
        subprocess.check_output(
            [
                str(PYTHON_FFI),
                "-c",
                'import invarlock,importlib.metadata,json;print(json.dumps({"version":importlib.metadata.version("invarlock"),"source":invarlock.__file__}))',
            ]
        )
    )
    ffi_package = Path(ffi_info["source"]).parent
    return {
        "reference_revision": "1dc9cb644684dc915810da77326f6da8764c58c1",
        "python": info,
        "python_ffi": ffi_info,
        "python_ffi_package_files": {
            str(p.relative_to(ffi_package)): sha(p)
            for p in sorted(ffi_package.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "python_ffi_wheel_sha256": sha(next((ROOT / "python-candidate").glob("*.whl"))),
        "python_package_files": {
            str(p.relative_to(package)): sha(p)
            for p in sorted(package.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "native_sha256": sha(NATIVE),
        "native_extension_files": json.loads(
            subprocess.check_output(
                [
                    str(PYTHON_FFI),
                    "-c",
                    'import invarlock_native,pathlib,hashlib,json;print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path(invarlock_native.__file__).parent.glob("*.so")}))',
                ]
            )
        ),
        "harness_sha256": sha(Path(__file__)),
        "workloads_sha256": sha(ROOT / "workloads.json"),
        "fixture_manifest_sha256": sha(ROOT / "fixture-manifest.json"),
        "source_archive_sha256": sha(ROOT / "input/source.tar"),
        "supervisor_sha256": sha(SUPERVISOR),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text(),
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text(),
        "rounds": 5,
        "timeout_seconds": 600,
        "seed": 4127,
        "normalizations": [
            "Per-attempt output directory paths",
            "package-manifest generator_source_sha256",
            "ZIP compressed bytes and metadata; all uncompressed member bytes checked against delivered files",
        ],
        "identity_checks": "outside all measured workers",
        "retries": 0,
        "exclusions": 0,
    }


def outputs(output):
    values = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(output).as_posix()
        if name == "package/handoff.zip":
            with zipfile.ZipFile(path) as archive:
                expected = {
                    str(p.relative_to(path.parent))
                    for p in path.parent.rglob("*")
                    if p.is_file() and p != path
                }
                if set(archive.namelist()) != expected:
                    raise RuntimeError("archive inventory mismatch")
                for item in archive.namelist():
                    if archive.read(item) != (path.parent / item).read_bytes():
                        raise RuntimeError("archive payload mismatch")
            values[name] = {"validated_members": len(expected)}
        elif name == "package/package-manifest.json":
            manifest = json.loads(path.read_bytes())
            generator = manifest.pop("generator_source_sha256")
            if len(generator) != 64:
                raise RuntimeError("missing generator identity")
            values[name] = manifest
        else:
            values[name] = sha(path)
    return values


def attempt(root, workload, candidate, round_id):
    path = root / f"{round_id}-{workload['name']}-{candidate}"
    path.mkdir()
    output = path / "output"
    output.mkdir()
    args = [s.replace("$OUTPUT", str(output)) for s in workload["args"]]
    evaluation = None
    if "evaluation_root" in workload:
        evaluation = Path(workload["evaluation_root"]) / ("expanded-" + path.name)
        args = [
            str(evaluation) if s.endswith("/OUTPUT_PLACEHOLDER") else s for s in args
        ]
    if candidate == "rust-ffi":
        command = [str(PYTHON_FFI), "-m", "invarlock", *args, "--backend", "rust"]
    elif candidate == "python":
        command = [str(PYTHON), "-m", "invarlock", *args]
    else:
        command = [str(NATIVE), *args]
    write(path / "command.json", command)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if candidate == "rust":
        env["PATH"] = "/nonexistent"
    with (path / "stdout").open("wb") as stdout, (path / "stderr").open("wb") as stderr:
        result = subprocess.run(
            [str(SUPERVISOR), str(path / "metrics.json"), "600", "--", *command],
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
    metrics = json.loads((path / "metrics.json").read_bytes())
    if (
        result.returncode
        or metrics["timed_out"]
        or metrics["exec_errno"]
        or metrics["exit_code"] != 0
    ):
        raise RuntimeError(f"failed attempt {path}")
    payload = json.loads((path / "stdout").read_bytes())
    result_files = outputs(output)
    payload = normalize(payload, output)
    if evaluation:
        payload = normalize(payload, evaluation)
        result_files = outputs(evaluation)
    record = {
        "workload": workload["name"],
        "candidate": candidate,
        "round": round_id,
        "metrics": metrics,
        "payload": payload,
        "files": result_files,
        "path": str(path),
    }
    write(path / "record.json", record)
    return record


def main():
    action = sys.argv[1]
    if action in {"prepare", "freeze"}:
        if action == "prepare":
            prepare()
        else:
            # Reuse the exact retained fixture bytes across architectures.
            json.loads((ROOT / "workloads.json").read_bytes())
        write(ROOT / "identity.json", identity())
        return
    workloads = json.loads((ROOT / "workloads.json").read_bytes())
    run = ROOT / "run"
    run.mkdir()
    # Recheck immutable inputs at execution time, before timing any candidate.
    if identity() != json.loads((ROOT / "identity.json").read_bytes()):
        raise RuntimeError("identity changed")
    for path, expected in json.loads(
        (ROOT / "fixture-manifest.json").read_bytes()
    ).items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"fixture changed before comparison: {path}")
    records = []
    rng = random.Random(4127)
    for round_id in ["qualification", *range(5)]:
        ordered = workloads.copy()
        rng.shuffle(ordered)
        for workload in ordered:
            candidates = workload["candidates"].copy()
            rng.shuffle(candidates)
            pair = []
            for candidate in candidates:
                record = attempt(run, workload, candidate, str(round_id))
                records.append(record)
                pair.append(record)
                print(
                    json.dumps(
                        {
                            "round": round_id,
                            "workload": workload["name"],
                            "candidate": candidate,
                            "seconds": record["metrics"]["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
            if any(
                json.dumps(pair[0]["payload"], sort_keys=True)
                != json.dumps(other["payload"], sort_keys=True)
                or pair[0]["files"] != other["files"]
                for other in pair[1:]
            ):
                write(run / "mismatch.json", pair)
                raise RuntimeError(f"parity mismatch: {workload['name']}")
            write(
                run / "progress.json",
                {
                    "attempts": len(records),
                    "last_workload": workload["name"],
                    "round": round_id,
                },
            )
        if round_id == "qualification":
            write(
                run / "qualification.json", {"outcomes": len(records), "matched": True}
            )
    summary = []
    for workload in workloads:
        row = {"workload": workload["name"]}
        for candidate in workload["candidates"]:
            items = [
                r["metrics"]
                for r in records
                if r["workload"] == workload["name"]
                and r["candidate"] == candidate
                and r["round"] != "qualification"
            ]
            row[candidate] = {
                "elapsed_seconds": statistics.median(x["wall_seconds"] for x in items),
                "cpu_seconds": statistics.median(
                    x["user_cpu_seconds"] + x["system_cpu_seconds"] for x in items
                ),
                "peak_rss_mib": statistics.median(
                    x["max_rss_bytes"] / 1024**2 for x in items
                ),
                "samples": len(items),
            }
        summary.append(row)
    write(
        run / "summary.json",
        {
            "identity": identity(),
            "outcomes": len(records),
            "qualified": sum(len(w["candidates"]) for w in workloads),
            "timed": sum(len(w["candidates"]) for w in workloads) * 5,
            "all_matched": True,
            "rows": summary,
        },
    )


if __name__ == "__main__":
    main()
