from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from invarlock.policy_pack import build_behavioral_policy_pack
from invarlock.runtime_behavior import (
    RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
    RuntimeBehaviorError,
    verify_pair,
)
from invarlock.runtime_behavioral_claim_receipt import (
    canonical_runtime_behavioral_claim_receipt_json,
)
from invarlock.runtime_security_helpers import RUNTIME_MANIFEST_FILENAME
from tests.runtime._runtime_behavior_support import (
    _baseline_identity,
    _FakeProvider,
    _rewrite_policy,
    _run,
    _strict_container_environment,  # noqa: F401
    _subject_identity,
    _write_inputs,
)


def _paired_sides(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    schedule_path, policy_path = _write_inputs(tmp_path)
    baseline = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )
    subject = _FakeProvider(
        name="llama_cpp",
        identity=_subject_identity(),
        outputs=("A", "wrong"),
    )
    _run(
        tmp_path,
        provider=baseline,
        role="baseline",
        directory_name="baseline",
        schedule_path=schedule_path,
        policy_path=policy_path,
    )
    _run(
        tmp_path,
        provider=subject,
        role="subject",
        directory_name="subject",
        schedule_path=schedule_path,
        policy_path=policy_path,
    )
    return tmp_path / "baseline", tmp_path / "subject", schedule_path, policy_path


def test_verify_pair_replays_bundles_and_publishes_digest_only_receipt(
    tmp_path: Path,
) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)
    receipt_path = tmp_path / "paired-claim.receipt.json"

    result = verify_pair(
        baseline_directory=baseline,
        subject_directory=subject,
        schedule_path=schedule_path,
        policy_pack_path=policy_path,
        receipt_path=receipt_path,
    )

    assert result.verification.ok
    assert result.receipt.baseline_score == 1.0
    assert result.receipt.subject_score == 0.5
    assert result.receipt.regression == 0.5
    assert receipt_path.read_bytes() == canonical_runtime_behavioral_claim_receipt_json(
        result.receipt
    )
    assert str(tmp_path).encode() not in receipt_path.read_bytes()


def test_verify_pair_rejects_tampered_side_without_receipt(tmp_path: Path) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)
    report_path = subject / RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["score"] = 1.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt_path = tmp_path / "should-not-exist.json"

    with pytest.raises(RuntimeBehaviorError, match="manifest v2"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=subject,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=receipt_path,
        )

    assert not receipt_path.exists()


def test_verify_pair_rejects_extra_side_file(tmp_path: Path) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)
    (subject / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(RuntimeBehaviorError, match="closed file set"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=subject,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=tmp_path / "should-not-exist.json",
        )


def test_verify_pair_rejects_manifest_image_ref_without_exact_digest(
    tmp_path: Path,
) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)
    manifest_path = subject / RUNTIME_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outer_container"]["image_ref"] = "registry.example/runtime:mutable"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeBehaviorError, match="outer image binding"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=subject,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=tmp_path / "should-not-exist.json",
        )


def test_verify_pair_rejects_role_swapped_sides(tmp_path: Path) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)

    with pytest.raises(RuntimeBehaviorError, match="baseline provider_name"):
        verify_pair(
            baseline_directory=subject,
            subject_directory=baseline,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=tmp_path / "should-not-exist.json",
        )


def test_verify_pair_allows_policy_directed_same_artifact_comparison(
    tmp_path: Path,
) -> None:
    schedule_path, policy_path = _write_inputs(tmp_path)
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    baseline_binding = original["behavioral_claim"]["baseline"]
    _rewrite_policy(
        policy_path,
        baseline=baseline_binding,
        subject=baseline_binding,
    )
    baseline_provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )
    subject_provider = _FakeProvider(
        name="hf_transformers",
        identity=_baseline_identity(),
        outputs=("A", "B"),
    )
    baseline = _run(
        tmp_path,
        provider=baseline_provider,
        role="baseline",
        directory_name="baseline-copy-source",
        schedule_path=schedule_path,
        policy_path=policy_path,
    )
    subject = _run(
        tmp_path,
        provider=subject_provider,
        role="subject",
        directory_name="subject-copy",
        schedule_path=schedule_path,
        policy_path=policy_path,
    )

    result = verify_pair(
        baseline_directory=baseline.directory,
        subject_directory=subject.directory,
        schedule_path=schedule_path,
        policy_pack_path=policy_path,
        receipt_path=tmp_path / "same-artifact-receipt.json",
    )

    assert result.verification.ok
    assert result.receipt_path.is_file()


def test_verify_pair_rejects_filesystem_copy_of_baseline_side(tmp_path: Path) -> None:
    baseline, _subject, schedule_path, policy_path = _paired_sides(tmp_path)
    copied = tmp_path / "copied-baseline"
    shutil.copytree(baseline, copied)

    with pytest.raises(RuntimeBehaviorError, match="subject provider_name"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=copied,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=tmp_path / "should-not-exist.json",
        )


def test_verify_pair_requires_the_policy_bound_by_each_side(tmp_path: Path) -> None:
    baseline, subject, schedule_path, policy_path = _paired_sides(tmp_path)
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    changed_policy = build_behavioral_policy_pack(
        tier="conservative",
        schedule_sha256=original["behavioral_claim"]["schedule_sha256"],
        baseline=original["behavioral_claim"]["baseline"],
        subject=original["behavioral_claim"]["subject"],
        metric_kind="exact_match",
        minimum_subject_score=0.4,
        maximum_regression=0.5,
        dataset_identity=original["compatibility"]["dataset_identity"],
    )
    changed_path = tmp_path / "changed-policy.json"
    changed_path.write_text(json.dumps(changed_policy), encoding="utf-8")
    receipt_path = tmp_path / "should-not-exist.json"

    with pytest.raises(RuntimeBehaviorError, match="side config"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=subject,
            schedule_path=schedule_path,
            policy_pack_path=changed_path,
            receipt_path=receipt_path,
        )

    assert not receipt_path.exists()


def test_verify_pair_rejects_aliased_side_directories(tmp_path: Path) -> None:
    baseline, _subject, schedule_path, policy_path = _paired_sides(tmp_path)

    with pytest.raises(RuntimeBehaviorError, match="must be distinct"):
        verify_pair(
            baseline_directory=baseline,
            subject_directory=baseline,
            schedule_path=schedule_path,
            policy_pack_path=policy_path,
            receipt_path=tmp_path / "should-not-exist.json",
        )
