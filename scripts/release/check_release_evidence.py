#!/usr/bin/env python3
"""Validate local release-evidence artifacts before cutting a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_GUARDS = {"spectral", "rmt", "variance"}


def _existing_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(matches)


def _require_file(path: Path, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"{label} missing: {path}")


def _require_any(
    root: Path, patterns: tuple[str, ...], label: str, failures: list[str]
) -> None:
    if not _existing_globs(root, patterns):
        joined = ", ".join(patterns)
        failures.append(f"{label} missing under {root}: {joined}")


def _dist_artifacts(dist_root: Path) -> list[Path]:
    return _existing_globs(dist_root, ("*.whl", "*.tar.gz"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str, failures: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"{label} missing: {path}")
    except json.JSONDecodeError as exc:
        failures.append(f"{label} is not valid JSON: {path}: {exc}")
    return None


def _parse_hash_entries(path: Path, failures: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return entries
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0].lower()):
            failures.append(
                f"wheel/sdist hashes line {line_number} must use sha256sum format."
            )
            continue
        digest = parts[0].lower()
        name = parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        name = name.removeprefix("./")
        entries[name] = digest
        entries[Path(name).name] = digest
    return entries


def _validate_dist_hashes(
    *, dist_root: Path, hash_path: Path, failures: list[str]
) -> None:
    artifacts = _dist_artifacts(dist_root)
    if not artifacts or not hash_path.is_file():
        return
    entries = _parse_hash_entries(hash_path, failures)
    if not entries:
        failures.append(f"wheel/sdist hashes file has no valid entries: {hash_path}")
        return
    for artifact in artifacts:
        candidates = {
            artifact.name,
            artifact.relative_to(dist_root).as_posix(),
            f"{dist_root.name}/{artifact.name}",
        }
        expected = next((entries[name] for name in candidates if name in entries), None)
        if expected is None:
            failures.append(f"wheel/sdist hash missing for artifact: {artifact.name}")
            continue
        actual = _sha256(artifact)
        if actual != expected:
            failures.append(f"wheel/sdist hash mismatch for artifact: {artifact.name}")


def _validate_runtime_digest(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        return
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) != 1 or not _RUNTIME_DIGEST_RE.fullmatch(lines[0]):
        failures.append(
            "runtime image digest must contain exactly one sha256:<64 hex> digest."
        )


def _validate_strict_report(path: Path, failures: list[str]) -> None:
    payload = _load_json(path, "strict example report", failures)
    if not isinstance(payload, dict):
        failures.append("strict example report must be a JSON object.")
        return
    assurance = payload.get("assurance")
    if not isinstance(assurance, dict):
        failures.append("strict example report missing assurance object.")
        return
    if assurance.get("mode") != "strict":
        failures.append("strict example report assurance.mode must be strict.")
    if assurance.get("verdict") not in {"pending_verifier", "pass"}:
        failures.append(
            "strict example report assurance.verdict must be pending_verifier or pass."
        )
    if assurance.get("fallback_fields_used") is not False:
        failures.append(
            "strict example report assurance.fallback_fields_used must be false."
        )
    report_build = payload.get("report_build")
    if not isinstance(report_build, dict):
        failures.append("strict example report missing report_build object.")
    else:
        for field in ("synthesized_fields", "repaired_fields", "fallback_fields"):
            value = report_build.get(field, [])
            if value:
                failures.append(
                    f"strict example report report_build.{field} must be empty."
                )


def _validate_strict_verify(path: Path, report_path: Path, failures: list[str]) -> None:
    payload = _load_json(path, "strict verifier output", failures)
    if not isinstance(payload, dict):
        failures.append("strict verifier output must be a JSON object.")
        return
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("ok") is not True:
        failures.append("strict verifier output summary.ok must be true.")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        failures.append("strict verifier output must include at least one result.")
        return
    result_names = {
        Path(str(result.get("id", ""))).name
        for result in results
        if isinstance(result, dict)
    }
    if report_path.name not in result_names:
        failures.append("strict verifier output does not reference the strict report.")
    provenance_verified = False
    for result in results:
        if not isinstance(result, dict):
            continue
        verification = result.get("verification")
        if not isinstance(verification, dict):
            continue
        provenance = verification.get("runtime_provenance")
        if not isinstance(provenance, dict):
            continue
        provenance_verified = (
            provenance.get("status") == "verified"
            and provenance.get("verified") is True
        )
        if provenance_verified:
            break
    if not provenance_verified:
        failures.append(
            "strict verifier output must contain verified runtime provenance."
        )


def _validate_guard_validation(
    *, json_path: Path, markdown_path: Path, failures: list[str]
) -> None:
    payload = _load_json(json_path, "guard-validation JSON", failures)
    if isinstance(payload, dict):
        if payload.get("schema") != "invarlock/guard-validation-smoke-v1":
            failures.append("guard-validation JSON schema is not recognized.")
        rows = payload.get("rate_rows")
        if not isinstance(rows, list):
            failures.append("guard-validation JSON missing rate_rows list.")
        else:
            guards = {
                str(row.get("guard"))
                for row in rows
                if isinstance(row, dict) and row.get("guard")
            }
            missing = sorted(_REQUIRED_GUARDS - guards)
            if missing:
                failures.append(
                    "guard-validation JSON missing guard rows: " + ", ".join(missing)
                )
    _require_file(markdown_path, "guard-validation markdown", failures)
    if (
        markdown_path.is_file()
        and not markdown_path.read_text(encoding="utf-8").strip()
    ):
        failures.append("guard-validation markdown must not be empty.")


def _validate_sbom(path: Path, failures: list[str]) -> None:
    payload = _load_json(path, "SBOM", failures)
    if payload is not None and not isinstance(payload, dict):
        failures.append("SBOM must be a JSON object.")


def _validate_offline_bundle(offline_bundle_dir: Path, failures: list[str]) -> None:
    bundles = _existing_globs(offline_bundle_dir, ("*.tar.gz",))
    if not bundles:
        failures.append(
            f"offline release bundle missing under {offline_bundle_dir}: *.tar.gz"
        )
        return
    valid_manifest_found = False
    for bundle in bundles:
        try:
            with tarfile.open(bundle, "r:gz") as tar:
                manifest_members = [
                    member
                    for member in tar.getmembers()
                    if member.isfile()
                    and Path(member.name).name == "release_manifest.json"
                ]
                if not manifest_members:
                    failures.append(
                        f"offline release bundle manifest missing: {bundle}"
                    )
                    continue
                extracted = tar.extractfile(manifest_members[0])
                if extracted is None:
                    failures.append(
                        f"offline release bundle manifest unreadable: {bundle}"
                    )
                    continue
                manifest = json.loads(extracted.read().decode("utf-8"))
        except (
            tarfile.TarError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"offline release bundle invalid: {bundle}: {exc}")
            continue
        if not isinstance(manifest, dict):
            failures.append(
                f"offline release bundle manifest must be an object: {bundle}"
            )
            continue
        if manifest.get("schema") != "invarlock/release-offline-bundle-v1":
            failures.append(
                f"offline release bundle schema is not recognized: {bundle}"
            )
            continue
        distributions = manifest.get("distributions")
        if not isinstance(distributions, list) or not distributions:
            failures.append(f"offline release bundle has no distributions: {bundle}")
            continue
        dist_paths = {
            str(item.get("path", ""))
            for item in distributions
            if isinstance(item, dict)
        }
        if not any(path.endswith(".whl") for path in dist_paths):
            failures.append(
                f"offline release bundle missing wheel distribution: {bundle}"
            )
            continue
        if not any(path.endswith(".tar.gz") for path in dist_paths):
            failures.append(
                f"offline release bundle missing sdist distribution: {bundle}"
            )
            continue
        valid_manifest_found = True
    if not valid_manifest_found:
        failures.append("no valid offline release bundle manifest found.")


def check_release_evidence(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
) -> list[str]:
    failures: list[str] = []
    _require_any(dist_root, ("*.whl",), "wheel artifact", failures)
    _require_any(dist_root, ("*.tar.gz",), "sdist artifact", failures)
    _require_file(sbom_path, "SBOM", failures)
    _require_file(
        release_root / "wheel-sdist-hashes.txt", "wheel/sdist hashes", failures
    )
    _require_file(
        release_root / "runtime-image-digest.txt", "runtime image digest", failures
    )
    _require_file(
        release_root / "strict" / "evaluation.report.json",
        "strict example report",
        failures,
    )
    _require_file(
        release_root / "strict" / "verify.json",
        "strict verifier output",
        failures,
    )
    _validate_sbom(sbom_path, failures)
    _validate_dist_hashes(
        dist_root=dist_root,
        hash_path=release_root / "wheel-sdist-hashes.txt",
        failures=failures,
    )
    _validate_runtime_digest(release_root / "runtime-image-digest.txt", failures)
    _validate_strict_report(
        release_root / "strict" / "evaluation.report.json", failures
    )
    _validate_strict_verify(
        release_root / "strict" / "verify.json",
        release_root / "strict" / "evaluation.report.json",
        failures,
    )
    _validate_guard_validation(
        json_path=guard_validation_json,
        markdown_path=guard_validation_markdown,
        failures=failures,
    )
    _validate_offline_bundle(offline_bundle_dir, failures)
    return failures


def _build_summary(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
    failures: list[str],
) -> dict[str, object]:
    return {
        "schema": "invarlock/release-evidence-check-v1",
        "release_root": str(release_root),
        "dist_root": str(dist_root),
        "sbom_path": str(sbom_path),
        "guard_validation_json": str(guard_validation_json),
        "guard_validation_markdown": str(guard_validation_markdown),
        "offline_bundle_dir": str(offline_bundle_dir),
        "ok": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local release-evidence bundle."
    )
    parser.add_argument(
        "--root", default="artifacts/release", help="Release evidence root."
    )
    parser.add_argument(
        "--dist", default="dist", help="Distribution artifact directory."
    )
    parser.add_argument(
        "--sbom",
        default="artifacts/supply-chain/sbom.json",
        help="Install/tool surface SBOM JSON path.",
    )
    parser.add_argument(
        "--guard-validation-json",
        default="artifacts/guard-validation/guard-validation-smoke.json",
        help="Guard-validation smoke JSON artifact path.",
    )
    parser.add_argument(
        "--guard-validation-md",
        default="artifacts/guard-validation/guard-validation-smoke.md",
        help="Guard-validation smoke Markdown artifact path.",
    )
    parser.add_argument(
        "--offline-bundle-dir",
        default="artifacts/release/offline",
        help="Directory containing offline release bundle tarballs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable summary.",
    )
    args = parser.parse_args(argv)

    release_root = Path(args.root)
    dist_root = Path(args.dist)
    sbom_path = Path(args.sbom)
    guard_validation_json = Path(args.guard_validation_json)
    guard_validation_markdown = Path(args.guard_validation_md)
    offline_bundle_dir = Path(args.offline_bundle_dir)
    failures = check_release_evidence(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    summary = _build_summary(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
        failures=failures,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print("Release evidence check passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
