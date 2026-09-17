"""Collect disjoint coverage shards and validate their combined inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts/ci/coverage.coveragerc"
SHARDS = ("core", "examples", "support", "addins")
FAST_MARKERS = "not integration and not slow and not manual and not gpu"
ADDIN_TESTS = tuple(
    f"addins/{name}/tests"
    for name in ("diagnostics", "gguf", "multimodal", "tensorrt_llm")
)
SUPPORT_TESTS = (
    "tests/ci/test_coverage_branch_rate.py",
    "tests/ci/test_coverage_shards.py",
    "tests/ci/test_public_evidence_audit.py",
    "tests/ci/test_public_text_check.py",
    "tests/ci/test_qualification_precheck.py",
    "tests/judge_measurements/test_statistics_calibration.py",
    "tests/scripts/test_authenticated_runtime_build.py",
    "tests/scripts/test_accelerate_checkpoint_files.py",
    "tests/scripts/test_build_cache_free_lm_eval_wheel.py",
    "tests/scripts/test_build_hardened_accelerate_wheel.py",
    "tests/scripts/test_hardened_accelerate_audit.py",
    "tests/scripts/test_hardened_accelerate_installed.py",
    "tests/scripts/test_build_restricted_openai_evals_wheel.py",
    "tests/scripts/test_check_repo_cruft.py",
    "tests/scripts/test_core_wheel_consumers.py",
    "tests/scripts/test_cve_audit.py",
    "tests/scripts/test_filter_scorecard_sarif.py",
    "tests/scripts/test_first_party_distribution_validation.py",
    "tests/scripts/test_installed_pip_audit.py",
    "tests/scripts/test_package_readme.py",
    "tests/scripts/test_package_rendering.py",
    "tests/scripts/test_prepare_qualification_suites.py",
    "tests/scripts/test_qualification_candidate_wheels.py",
    "tests/scripts/test_qualification_receipt_check.py",
    "tests/scripts/test_qualification_render_preflight.py",
    "tests/scripts/test_qualification_source.py",
    "tests/scripts/test_refresh_pinned_requirements.py",
    "tests/scripts/test_release_distribution_validation_edges.py",
    "tests/scripts/test_release_preflight.py",
    "tests/scripts/test_release_preflight_adversarial.py",
    "tests/scripts/test_release_preflight_edges.py",
    "tests/scripts/test_release_reference_journey.py",
    "tests/scripts/test_run_pip_audit.py",
    "tests/scripts/test_runtime_qualification.py",
    "tests/scripts/test_runtime_qualification_edges.py",
    "tests/scripts/test_runtime_qualification_security.py",
    "tests/scripts/test_sync_packaged_contracts.py",
    "tests/scripts/test_sync_packaged_public_evidence.py",
    "tests/scripts/test_tagged_release_candidate.py",
    "tests/scripts/test_verify_hosted_distributions.py",
    "tests/scripts/test_verify_hosted_distributions_edges.py",
)


def selection(shard: str) -> list[str]:
    if shard == "core":
        return [
            "tests",
            "-m",
            FAST_MARKERS,
            "--ignore=tests/examples",
            *(f"--ignore={path}" for path in SUPPORT_TESTS),
        ]
    if shard == "examples":
        return ["tests/examples"]
    if shard == "support":
        return list(SUPPORT_TESTS)
    if shard == "addins":
        return list(ADDIN_TESTS)
    raise ValueError(f"unknown coverage shard: {shard}")


def source_identity() -> str:
    """Bind local edits as well as the commit, without storing source contents."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    digest = hashlib.sha256(commit)
    for relative in sorted(set(files.split(b"\0")) - {b""}):
        path = ROOT / os.fsdecode(relative)
        digest.update(relative + b"\0")
        if path.is_symlink():
            content = b"link\0" + os.fsencode(os.readlink(path))
        elif path.is_file():
            content = b"file\0" + path.read_bytes()
        else:
            content = b"deleted\0"
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def pytest_collection_finish(session) -> None:
    """The collector plugin records pytest's actual selected node IDs."""
    destination = os.environ.get("INVARLOCK_COVERAGE_INVENTORY")
    if destination and session.items:
        path = Path(destination)
        temporary = path.with_suffix(f".{os.getpid()}.json")
        temporary.write_text(
            json.dumps(sorted(item.nodeid for item in session.items)) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _inventory(path: Path) -> list[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(items, list)
        or not items
        or any(not isinstance(item, str) or "::" not in item for item in items)
        or len(set(items)) != len(items)
    ):
        raise ValueError(f"invalid or empty test inventory: {path}")
    return items


def _data_digest(path: Path) -> str:
    from coverage import CoverageData

    data = CoverageData(basename=str(path))
    data.read()
    if not data.has_arcs() or not data.measured_files():
        raise ValueError(f"missing branch coverage data: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(relative: str) -> str:
    path = PurePosixPath(relative)
    if not relative or "\\" in relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid relative coverage source: {relative}")
    source = (ROOT / path).resolve()
    if not source.is_relative_to(ROOT.resolve()):
        raise ValueError(f"coverage source escapes the checkout: {relative}")
    return str(source)


def run(shard: str, artifact_dir: Path, workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    arguments = selection(shard)
    destination = artifact_dir.resolve() / shard
    destination.mkdir(parents=True, exist_ok=True)
    # Never let an interrupted retry inherit the previous attempt's success.
    manifest = destination / "manifest.json"
    identity = source_identity()
    _write_json(manifest, {"version": 1, "shard": shard, "status": "running"})
    inventory = destination / "inventory.json"
    inventory.unlink(missing_ok=True)
    data = destination / ".coverage"
    data.unlink(missing_ok=True)
    env = dict(os.environ)
    env["COVERAGE_FILE"] = str(data)
    env["INVARLOCK_COVERAGE_INVENTORY"] = str(inventory)
    addin_sources = ()
    if shard == "addins":
        addin_sources = tuple(str(Path(path).parent / "src") for path in ADDIN_TESTS)
    elif shard == "support":
        addin_sources = ("addins/tensorrt_llm/src",)
    env["PYTHONPATH"] = os.pathsep.join(
        str(ROOT / path) for path in ("scripts/ci", "src", ".", *addin_sources)
    )
    command = [sys.executable, "-m", "pytest", "-p", "coverage_runner"]
    if workers:
        command.extend(["-n", str(workers)])
    command.extend(
        [
            "-q",
            *arguments,
            "--cov",
            f"--cov-config={CONFIG}",
            "--cov-branch",
            "--cov-report=",
            "--durations=20",
            f"--junitxml={destination / 'junit.xml'}",
        ]
    )
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    record = {
        "version": 1,
        "shard": shard,
        "status": "failed",
        "source_identity": identity,
        "exit_code": result.returncode,
        "duration_seconds": time.monotonic() - started,
    }
    _write_json(manifest, record)
    if result.returncode:
        return result.returncode
    if source_identity() != identity:
        raise ValueError("source files changed while coverage was running")
    record.update(
        status="passed",
        inventory=_inventory(inventory),
        data_sha256=_data_digest(data),
    )
    _write_json(manifest, record)
    return 0


def combine(artifact_dir: Path, output: Path) -> None:
    from coverage import CoverageData

    identity = source_identity()
    manifests = sorted(artifact_dir.glob("*/manifest.json"))
    if {path.parent.name for path in manifests} != set(SHARDS):
        raise ValueError("coverage requires exactly core, examples, support and addins")
    seen: set[str] = set()
    data_paths = []
    for manifest in manifests:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        shard = manifest.parent.name
        if (
            not isinstance(record, dict)
            or record.get("version") != 1
            or record.get("shard") != shard
            or record.get("status") != "passed"
            or record.get("exit_code") != 0
            or record.get("source_identity") != identity
        ):
            raise ValueError(f"failed or mismatched coverage shard: {shard}")
        inventory = _inventory(manifest.parent / "inventory.json")
        if record.get("inventory") != inventory or seen.intersection(inventory):
            raise ValueError(f"changed or overlapping coverage inventory: {shard}")
        seen.update(inventory)
        data = manifest.parent / ".coverage"
        if record.get("data_sha256") != _data_digest(data):
            raise ValueError(f"changed coverage data: {shard}")
        data_paths.append(str(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep readers and writers separate, and never reuse a previous combined file.
    with TemporaryDirectory(
        prefix=".coverage-combine-", dir=output.parent
    ) as temporary:
        combined = CoverageData(basename=str(Path(temporary) / ".coverage"))
        for path in data_paths:
            shard_data = CoverageData(basename=path)
            shard_data.read()
            source_paths = {
                name: _source_path(name) for name in shard_data.measured_files()
            }
            combined.update(shard_data, map_path=source_paths.__getitem__)
        combined.write()
        os.replace(combined.data_filename(), output)
    print(f"Combined {len(SHARDS)} coverage shards with {len(seen)} unique tests.")


def main(argv: list[str] | None = None) -> int:
    from coverage.exceptions import CoverageException

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("shard", choices=SHARDS)
    runner.add_argument("--artifact-dir", type=Path, default=Path("artifacts/coverage"))
    runner.add_argument("--workers", type=int, default=2)
    merger = commands.add_parser("combine")
    merger.add_argument("artifact_dir", type=Path)
    merger.add_argument("--output", type=Path, default=Path(".coverage"))
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args.shard, args.artifact_dir, args.workers)
        combine(args.artifact_dir, args.output)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        CoverageException,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
