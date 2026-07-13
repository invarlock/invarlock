#!/usr/bin/env python3
"""Produce, pack, and strictly verify one catalog lane without publishing it."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invarlock.evidence_catalog import (  # noqa: E402
    EvidenceCatalogError,
    load_evidence_catalog,
    load_resolved_inputs,
)
from scripts.evidence_packs.python.publication_privacy_check import (  # noqa: E402
    publication_privacy_errors,
)
from scripts.model_evidence.catalog_lane_pack import (  # noqa: E402
    CatalogLaneArtifacts,
    CatalogLaneError,
    assemble_signed_catalog_pack,
)

PUBLIC_EVIDENCE_ROOT = REPO_ROOT / "public_evidence"
_COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")
_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")


def _python_command(*args: str) -> list[str]:
    return [sys.executable, "-I", "-m", "invarlock", *args]


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def validate_staging_output(out_dir: Path) -> None:
    """Keep candidate creation separate from explicit public-index publication."""

    if _path_is_within(out_dir, PUBLIC_EVIDENCE_ROOT):
        raise CatalogLaneError("staging output must remain outside public_evidence")
    if out_dir.exists() or out_dir.is_symlink():
        raise CatalogLaneError(f"staging output already exists: {out_dir}")
    receipt = out_dir.with_name(out_dir.name + ".verification.json")
    if receipt.exists() or receipt.is_symlink():
        raise CatalogLaneError(
            f"staging verification receipt already exists: {receipt}"
        )


def _lane_inputs(
    catalog_path: Path,
    lane_id: str,
    resolved_inputs_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    catalog = load_evidence_catalog(catalog_path)
    entry = catalog.entries.get(lane_id)
    if entry is None:
        raise CatalogLaneError(f"catalog lane is not present: {lane_id}")
    resolved, _digest = load_resolved_inputs(resolved_inputs_path, entry=entry)
    return entry, resolved


def build_evaluate_command(
    *,
    catalog: Path,
    lane_id: str,
    resolved_inputs: Path,
    prepared_preset: Path,
    evaluation_input_binding: Path,
    work_dir: Path,
    device: str,
    allow_network: bool,
) -> list[str]:
    """Derive the only model-evaluation command from authenticated lane inputs."""

    entry, resolved = _lane_inputs(catalog, lane_id, resolved_inputs)
    model = resolved.get("model")
    execution = entry.get("execution")
    if not isinstance(model, Mapping) or not isinstance(execution, Mapping):
        raise CatalogLaneError("catalog lane model or execution policy is invalid")
    model_id = model.get("id")
    revision = model.get("revision")
    adapter = model.get("adapter")
    required_strings = (model_id, revision, adapter)
    if not all(isinstance(value, str) and value for value in required_strings):
        raise CatalogLaneError("resolved model inputs are incomplete")
    command = _python_command(
        "evaluate",
        "--baseline",
        str(model_id),
        "--subject",
        str(model_id),
        "--baseline-revision",
        str(revision),
        "--subject-revision",
        str(revision),
        "--baseline-adapter",
        str(adapter),
        "--subject-adapter",
        str(adapter),
        "--device",
        device,
        "--profile",
        str(execution.get("profile")),
        "--tier",
        str(execution.get("tier")),
        "--preset",
        str(prepared_preset),
        "--evaluation-input-binding",
        str(evaluation_input_binding),
        "--out",
        str(work_dir / "runs"),
        "--report-out",
        str(work_dir / "report"),
        "--edit-label",
        str(execution.get("edit_name")),
        "--execution-mode",
        str(execution.get("execution_mode")),
        "--assurance",
        str(execution.get("assurance_mode")),
        "--defer-report-rendering",
        "--quiet",
        "--no-banner",
        "--no-color",
    )
    if allow_network:
        command.append("--allow-network")
    return command


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        if stdout and stderr:
            diagnostic = f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        else:
            diagnostic = stdout or stderr or "command failed"
        raise CatalogLaneError(diagnostic) from exc


def _json_stdout(
    result: subprocess.CompletedProcess[str], *, label: str
) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CatalogLaneError(f"{label} did not emit valid JSON") from exc
    if not isinstance(payload, dict):
        raise CatalogLaneError(f"{label} did not emit a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    try:
        path.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CatalogLaneError(f"could not write {path.name}: {exc}") from exc


def _prepare_lane_inputs(
    *,
    workspace: Path,
    catalog: Path,
    lane_id: str,
    resolved_inputs: Path,
    preset: Path,
    entry: Mapping[str, object],
    allow_network: bool,
    env: Mapping[str, str],
) -> Path | None:
    inputs = entry.get("inputs")
    vision = isinstance(inputs, Mapping) and inputs.get("kind") == "vision_text"
    materialization: Path | None = None
    if vision:
        if not allow_network:
            raise CatalogLaneError(
                "vision lanes require --allow-network for materialization"
            )
        materialization_dir = Path("materialization")
        result = _run(
            _python_command(
                "advanced",
                "inputs",
                "materialize",
                "--catalog",
                str(catalog),
                "--lane",
                lane_id,
                "--out",
                str(materialization_dir),
                "--allow-network",
                "--json",
            ),
            cwd=workspace,
            env=env,
        )
        if _json_stdout(result, label="input materialization").get("ok") is not True:
            raise CatalogLaneError("input materialization did not pass")
        materialization = workspace / materialization_dir / "dataset_evidence.json"

    prepare = [
        "advanced",
        "inputs",
        "prepare",
        "--catalog",
        str(catalog),
        "--lane",
        lane_id,
        "--resolved-inputs",
        str(resolved_inputs),
        "--preset",
        str(preset),
        "--out",
        "prepared-preset.yaml",
        "--json",
    ]
    if materialization is not None:
        prepare.extend(["--materialization-dir", "materialization"])
    result = _run(_python_command(*prepare), cwd=workspace, env=env)
    if _json_stdout(result, label="preset preparation").get("ok") is not True:
        raise CatalogLaneError("preset preparation did not pass")

    binding = [
        "advanced",
        "inputs",
        "binding",
        "--catalog",
        str(catalog),
        "--lane",
        lane_id,
        "--resolved-inputs",
        str(resolved_inputs),
        "--preset",
        str(preset),
        "--out",
        "evaluation-input-binding.json",
    ]
    if materialization is not None:
        binding.extend(
            ["--input-materialization", "materialization/dataset_evidence.json"]
        )
    _run(_python_command(*binding), cwd=workspace, env=env)
    return materialization


def _single_baseline_report(workspace: Path) -> Path:
    reports = sorted((workspace / "runs" / "source").glob("*/report.json"))
    if len(reports) != 1:
        raise CatalogLaneError(
            f"evaluation must produce exactly one baseline report, observed {len(reports)}"
        )
    return reports[0]


def _verify_report(
    *,
    workspace: Path,
    entry: Mapping[str, object],
    baseline: Path,
    policy_pack: Path,
    runtime_image_digest: str,
    env: Mapping[str, str],
) -> Path:
    execution = entry.get("execution")
    if not isinstance(execution, Mapping):
        raise CatalogLaneError("catalog execution policy is invalid")
    command = _python_command(
        "verify",
        "--profile",
        str(execution.get("profile")),
        "--assurance",
        "strict",
        "--baseline",
        baseline.relative_to(workspace).as_posix(),
        "--policy-pack",
        str(policy_pack),
        "--expected-runtime-image-digest",
        runtime_image_digest,
        "--json",
        "report/evaluation.report.json",
    )
    receipt = _json_stdout(
        _run(command, cwd=workspace, env=env), label="strict report verification"
    )
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping) or summary.get("ok") is not True:
        raise CatalogLaneError("strict report verification did not pass")
    path = workspace / "report-verification.json"
    _write_json(path, receipt)
    return path


def _verify_pack(
    *,
    workspace: Path,
    pack: Path,
    catalog_digest: str,
    policy_pack: Path,
    runtime_image_digest: str,
    fingerprint: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    command = _python_command(
        "advanced",
        "evidence-pack",
        "verify",
        pack.relative_to(workspace).as_posix(),
        "--strict",
        "--report-assurance",
        "strict",
        "--expected-fingerprint",
        fingerprint,
        "--expected-catalog-digest",
        catalog_digest,
        "--policy-pack",
        str(policy_pack),
        "--expected-runtime-image-digest",
        runtime_image_digest,
        "--json",
    )
    payload = _json_stdout(
        _run(command, cwd=workspace, env=env), label="strict evidence-pack verification"
    )
    if payload.get("ok") is not True:
        raise CatalogLaneError("strict evidence-pack verification did not pass")
    return payload


def _run_environment(
    *,
    runtime_image: str,
    runtime_image_digest: str,
    source_commit: str,
    source_bundle_sha256: str,
    allow_network: bool,
) -> dict[str, str]:
    if _DIGEST_RE.fullmatch(runtime_image_digest) is None:
        raise CatalogLaneError("runtime image digest must be sha256:<64 lowercase hex>")
    if _DIGEST_RE.fullmatch(source_bundle_sha256) is None:
        raise CatalogLaneError("source bundle digest must be sha256:<64 lowercase hex>")
    if _COMMIT_RE.fullmatch(source_commit) is None:
        raise CatalogLaneError(
            "source commit must be 40 lowercase hexadecimal characters"
        )
    if os.environ.get("INVARLOCK_CONTAINER_EXECUTION", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise CatalogLaneError(
            "catalog lane production must run inside the declared runtime container"
        )
    env = dict(os.environ)
    declared_digest = env.get("INVARLOCK_RUNTIME_IMAGE_DIGEST")
    if declared_digest and declared_digest != runtime_image_digest:
        raise CatalogLaneError(
            "runtime image digest disagrees with the container environment"
        )
    env.update(
        {
            "INVARLOCK_RUNTIME_IMAGE": runtime_image,
            "INVARLOCK_RUNTIME_IMAGE_DIGEST": runtime_image_digest,
            "INVARLOCK_SOURCE_COMMIT": source_commit,
            "INVARLOCK_SOURCE_BUNDLE_SHA256": source_bundle_sha256,
            "INVARLOCK_SOURCE_BUNDLE_READ_ONLY": "1",
            "INVARLOCK_ALLOW_NETWORK": "1" if allow_network else "0",
        }
    )
    return env


def run_catalog_lane(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out.resolve(strict=False)
    validate_staging_output(out_dir)
    failed_workspace_value = getattr(args, "failed_workspace_out", None)
    failed_workspace_out = (
        failed_workspace_value.resolve(strict=False)
        if isinstance(failed_workspace_value, Path)
        else None
    )
    if failed_workspace_out is not None:
        validate_staging_output(failed_workspace_out)
        if failed_workspace_out == out_dir:
            raise CatalogLaneError(
                "failed workspace output must differ from staging output"
            )
    catalog_path = args.catalog.resolve(strict=True)
    resolved_inputs = args.resolved_inputs.resolve(strict=True)
    policy_pack = args.policy_pack.resolve(strict=True)
    signing_key = args.signing_key.resolve(strict=True)
    entry, _resolved = _lane_inputs(catalog_path, args.lane, resolved_inputs)
    preset_value = entry.get("preset")
    preset_rel = preset_value.get("path") if isinstance(preset_value, Mapping) else None
    if not isinstance(preset_rel, str):
        raise CatalogLaneError("catalog preset path is invalid")
    preset = (REPO_ROOT / preset_rel).resolve(strict=True)
    env = _run_environment(
        runtime_image=args.runtime_image,
        runtime_image_digest=args.runtime_image_digest,
        source_commit=args.source_commit,
        source_bundle_sha256=args.source_bundle_sha256,
        allow_network=bool(args.allow_network),
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.run.", dir=out_dir.parent)
    )
    try:
        materialization = _prepare_lane_inputs(
            workspace=workspace,
            catalog=catalog_path,
            lane_id=args.lane,
            resolved_inputs=resolved_inputs,
            preset=preset,
            entry=entry,
            allow_network=bool(args.allow_network),
            env=env,
        )
        _run(
            build_evaluate_command(
                catalog=catalog_path,
                lane_id=args.lane,
                resolved_inputs=resolved_inputs,
                prepared_preset=Path("prepared-preset.yaml"),
                evaluation_input_binding=Path("evaluation-input-binding.json"),
                work_dir=Path("."),
                device=args.device,
                allow_network=bool(args.allow_network),
            ),
            cwd=workspace,
            env=env,
        )
        baseline = _single_baseline_report(workspace)
        verification_receipt = _verify_report(
            workspace=workspace,
            entry=entry,
            baseline=baseline,
            policy_pack=policy_pack,
            runtime_image_digest=args.runtime_image_digest,
            env=env,
        )
        artifacts = CatalogLaneArtifacts(
            catalog=catalog_path,
            lane_id=args.lane,
            evaluation_report=workspace / "report/evaluation.report.json",
            runtime_manifest=workspace / "report/runtime.manifest.json",
            baseline_report=baseline,
            policy_pack=policy_pack,
            resolved_inputs=resolved_inputs,
            resolved_config=workspace / "report/resolved-config.yaml",
            preset=preset,
            evaluation_input_binding=workspace / "evaluation-input-binding.json",
            verification_receipt=verification_receipt,
            source_commit=args.source_commit,
            source_bundle_sha256=args.source_bundle_sha256,
            input_materialization=materialization,
            expected_runtime_image_digest=args.runtime_image_digest,
            network_mode="online" if args.allow_network else "offline",
        )
        pack, fingerprint = assemble_signed_catalog_pack(
            artifacts,
            workspace / "pack",
            signing_key=signing_key,
        )
        privacy_errors = publication_privacy_errors(pack)
        if privacy_errors:
            raise CatalogLaneError("; ".join(privacy_errors))
        catalog = load_evidence_catalog(catalog_path)
        pack_receipt = _verify_pack(
            workspace=workspace,
            pack=pack,
            catalog_digest=catalog.digest,
            policy_pack=policy_pack,
            runtime_image_digest=args.runtime_image_digest,
            fingerprint=fingerprint,
            env=env,
        )
        receipt_path = workspace / "pack-verification.json"
        _write_json(receipt_path, pack_receipt)
        pack.replace(out_dir)
        final_receipt = out_dir.with_name(out_dir.name + ".verification.json")
        try:
            receipt_path.replace(final_receipt)
        except BaseException:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise
        return {
            "format_version": "catalog-lane-run-v1",
            "ok": True,
            "lane_id": args.lane,
            "catalog_digest": catalog.digest,
            "runtime_image_digest": args.runtime_image_digest,
            "signing_key_fingerprint": fingerprint,
            "staged_pack": str(out_dir),
            "verification_receipt": str(final_receipt),
            "published": False,
        }
    except Exception as exc:
        if failed_workspace_out is not None:
            try:
                failed_workspace_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(workspace), str(failed_workspace_out))
            except OSError as retain_exc:
                raise CatalogLaneError(
                    f"{exc}; could not retain failed workspace: {retain_exc}"
                ) from exc
            retained = f"failed workspace retained at {failed_workspace_out}"
            if isinstance(exc, CatalogLaneError):
                raise CatalogLaneError(f"{exc}\n{retained}") from exc
            if isinstance(exc, (EvidenceCatalogError, OSError)):
                raise CatalogLaneError(f"{exc}\n{retained}") from exc
            raise
        if isinstance(exc, (EvidenceCatalogError, OSError)):
            raise CatalogLaneError(str(exc)) from exc
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "contracts/evidence_catalog_v1.json",
    )
    parser.add_argument("--resolved-inputs", type=Path, required=True)
    parser.add_argument("--policy-pack", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failed-workspace-out",
        type=Path,
        help="Retain the private run workspace at this path when the lane fails.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = run_catalog_lane(args)
    except CatalogLaneError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CatalogLaneArtifacts",
    "CatalogLaneError",
    "assemble_signed_catalog_pack",
    "build_evaluate_command",
    "run_catalog_lane",
    "validate_staging_output",
]
