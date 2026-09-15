"""Authenticate Accelerate remediation while retaining its upstream scan identity."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

try:
    from scripts.security.installed_audit_binding import (
        _read,
        _require,
        _sha,
        parse_locked_pins,
    )
except ImportError:  # pragma: no cover - direct script execution
    from installed_audit_binding import _read, _require, _sha, parse_locked_pins

PACKAGE = "accelerate"
UPSTREAM_VERSION = "1.14.0"
HARDENED_VERSION = "1.14.0+invarlock.1"
WHEEL_NAME = f"accelerate-{HARDENED_VERSION}-py3-none-any.whl"
WHEEL_DIRECTORY = "runtime/wheels"
SOURCE_LOCK = "requirements/workflows/accelerate-upstream-wheel.txt"
REMEDIATED_ADVISORIES = frozenset({"GHSA-4j2p-28q2-5m79", "PYSEC-2026-3804"})


def verify_wheel(path: Path) -> dict:
    try:
        from scripts.security.build_hardened_accelerate_wheel import (
            verify_hardened_wheel,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from build_hardened_accelerate_wheel import verify_hardened_wheel
    data = _read(path)
    verification = verify_hardened_wheel(path)
    _require(_read(path) == data, "hardened wheel changed during verification")
    return {
        "package": PACKAGE,
        "version": HARDENED_VERSION,
        "upstream_version": UPSTREAM_VERSION,
        "wheel_sha256": _sha(data),
        "derivation": verification,
        "remediated_advisories": sorted(REMEDIATED_ADVISORIES),
    }


def lock_remediation(path: Path, repo_root: Path) -> dict | None:
    """Authenticate the runtime artifact or the sole source-only build input."""
    data = _read(path)
    # Parse every line once remediation is requested, so options, duplicate pins,
    # markers, includes and additional artifact hashes cannot change its scope.
    pins = parse_locked_pins(data)
    if PACKAGE not in pins:
        return None
    version, hashes = pins[PACKAGE]
    source = path.relative_to(repo_root).as_posix()
    source_only = source == SOURCE_LOCK
    if version != HARDENED_VERSION and not source_only:
        _require("+" not in version, "unsupported local Accelerate version")
        return None
    proof = verify_wheel(repo_root / WHEEL_DIRECTORY / WHEEL_NAME)
    if source_only:
        try:
            from scripts.security.build_hardened_accelerate_wheel import (
                UPSTREAM_WHEEL_SHA256,
            )
        except ImportError:  # pragma: no cover - direct script execution
            from build_hardened_accelerate_wheel import UPSTREAM_WHEEL_SHA256
        _require(
            pins == {PACKAGE: (UPSTREAM_VERSION, {UPSTREAM_WHEEL_SHA256})},
            "source-only Accelerate lock differs from authenticated upstream input",
        )
    else:
        _require(
            version == HARDENED_VERSION and hashes == {proof["wheel_sha256"]},
            "Accelerate lock does not identify only the verified hardened wheel",
        )
    return {**proof, "source_only": source_only}


def uv_remediation(package: dict, repo_root: Path) -> dict:
    proof = verify_wheel(repo_root / WHEEL_DIRECTORY / WHEEL_NAME)
    _require(
        package.get("version") == HARDENED_VERSION
        and package.get("source") == {"registry": WHEEL_DIRECTORY}
        and package.get("wheels") == [{"path": WHEEL_NAME}]
        and "sdist" not in package,
        "uv Accelerate identity differs from verified local wheel source",
    )
    return {**proof, "source_only": False}


def run_requirement_audit(args) -> int:
    try:
        from scripts.security import installed_audit_binding as binding
    except ImportError:  # pragma: no cover - direct script execution
        import installed_audit_binding as binding
    report: dict = {
        "status": "blocked",
        "remediated_findings": [],
        "blocking_findings": [],
        "scope": "Locked artifact identities, with authenticated Accelerate derivation and raw upstream findings; not an installed-environment audit.",
    }
    try:
        _require(
            not args.path, "remediated requirement scan cannot mix installed paths"
        )
        inventory: dict[str, str] = {}
        anchors: dict[str, bytes] = {}
        proof = None
        for requirement in args.requirement:
            path = Path(requirement).absolute()
            anchors[str(path)] = _read(path)
            pins = parse_locked_pins(anchors[str(path)])
            current = lock_remediation(path, Path.cwd())
            if PACKAGE in pins:
                _require(
                    current is not None, "unpatched Accelerate requirement surface"
                )
                _require(
                    proof is None or proof == current,
                    "mixed Accelerate build and runtime inputs",
                )
                proof = current
            for name, (version, _hashes) in pins.items():
                _require(
                    name not in inventory or inventory[name] == version,
                    "conflicting requirement identities",
                )
                inventory[name] = version
        _require(proof is not None, "missing authenticated Accelerate remediation")
        scan_inventory = {**inventory, PACKAGE: UPSTREAM_VERSION}
        evidence = {
            "package": PACKAGE,
            "version": inventory[PACKAGE],
            "installed_distribution_inventory": inventory,
            "scan_distribution_inventory": scan_inventory,
            "approved_advisories": [],
            "remediation": proof,
        }
        report["binding"] = evidence
        report["lock_sha256"] = {path: _sha(data) for path, data in anchors.items()}
        completed = binding.scan_upstream_inventory(scan_inventory, report)
        report.update(
            {
                "scanner_returncode": completed.returncode,
                "raw_stdout": completed.stdout.decode("utf-8", errors="replace"),
                "raw_stderr": completed.stderr.decode("utf-8", errors="replace"),
                "raw_stdout_base64": base64.b64encode(completed.stdout).decode(),
                "raw_stderr_base64": base64.b64encode(completed.stderr).decode(),
            }
        )
        raw = binding._json(completed.stdout)
        report["raw_findings"] = raw
        remediated, blocking = binding._classify(raw, evidence, completed.returncode)
        report["remediated_findings"], report["blocking_findings"] = (
            remediated,
            blocking,
        )
        for path, data in anchors.items():
            _require(_read(Path(path)) == data, "lock changed during audit")
        current = verify_wheel(Path.cwd() / WHEEL_DIRECTORY / WHEEL_NAME)
        _require(
            current
            == {key: value for key, value in proof.items() if key != "source_only"},
            "wheel changed during audit",
        )
        report["status"] = (
            "blocked" if blocking else "remediated" if remediated else "clean"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        UnicodeError,
        subprocess.SubprocessError,
    ) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    binding.write_report(report, args.report)
    return 0 if report["status"] in {"clean", "remediated"} else 1
