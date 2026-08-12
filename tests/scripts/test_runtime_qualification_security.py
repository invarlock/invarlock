from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from scripts import runtime_qualification


def _arguments(
    tmp_path: Path,
    *,
    evidence: Path,
    summary: Path,
) -> argparse.Namespace:
    source_bundle = tmp_path / "source.tar.gz"
    source_bundle.write_bytes(b"authenticated fixture")
    request = tmp_path / "request.yaml"
    request.write_text("format_version: fixture\n", encoding="utf-8")
    signing_key = tmp_path / "evidence-signer.pem"
    signing_key.write_text("fixture\n", encoding="utf-8")
    trust_profile = tmp_path / "trust-profile.json"
    trust_profile.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        mode="run",
        python=sys.executable,
        request=request,
        signing_key=signing_key,
        runtime_image="sha256:" + "a" * 64,
        runtime_image_digest="sha256:" + "a" * 64,
        evidence=evidence,
        trust_profile=trust_profile,
        receipt=tmp_path / "receipt.json",
        canary_evidence=tmp_path / "canary-evidence",
        canary_receipt=tmp_path / "canary-receipt.json",
        canary_trust_profile=tmp_path / "canary-trust-profile.json",
        source_commit="b" * 40,
        source_bundle=source_bundle,
        source_bundle_sha256="sha256:" + "c" * 64,
        candidate_wheel_manifest=tmp_path / "candidate-wheels.json",
        container_engine="docker",
        runtime_device="cuda:0",
        runtime_cpus="4",
        runtime_memory_mib=8192,
        runtime_user="65532:65532",
        report=None,
        summary=summary,
    )


def test_configuration_rejects_summary_alias_that_resolves_inside_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "outputs").mkdir(parents=True)
    alias = tmp_path / "outside-looking-alias"
    alias.symlink_to(evidence, target_is_directory=True)
    summary = alias / "outputs" / "summary.json"
    monkeypatch.setattr(
        runtime_qualification,
        "_authenticate_source_bundle",
        lambda *_args, **_kwargs: "sha256:" + "d" * 64,
    )

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="evidence destination already exists",
    ):
        runtime_qualification._inputs(  # noqa: SLF001
            _arguments(tmp_path, evidence=evidence, summary=summary)
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("runtime-image", "must equal or embed"),
        ("request-parent", "request parent must be an existing directory"),
        ("duplicate-outputs", "must be distinct outputs"),
        ("inside-evidence", "outside the immutable evidence pack"),
        ("missing-engine", "container engine is unavailable"),
    ),
)
def test_configuration_rejects_invalid_output_and_runtime_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    evidence = tmp_path / "future-evidence"
    arguments = _arguments(
        tmp_path,
        evidence=evidence,
        summary=tmp_path / "summary.json",
    )
    monkeypatch.setattr(
        runtime_qualification,
        "_authenticate_source_bundle",
        lambda *_args, **_kwargs: "sha256:" + "d" * 64,
    )
    if case == "runtime-image":
        arguments.runtime_image = "mutable:latest"
    elif case == "request-parent":
        arguments.request = tmp_path / "missing" / "request.yaml"
    elif case == "duplicate-outputs":
        arguments.receipt = arguments.summary
    elif case == "inside-evidence":
        arguments.receipt = evidence / "verification-receipt.json"
    else:
        monkeypatch.setattr(runtime_qualification.shutil, "which", lambda _name: None)

    with pytest.raises(runtime_qualification.QualificationError, match=message):
        runtime_qualification._inputs(arguments)  # noqa: SLF001


def test_fresh_destination_rejects_an_ancestor_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    (real_parent / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="parent must be a real directory",
    ):
        runtime_qualification._fresh_destination(  # noqa: SLF001
            alias / "nested" / "summary.json",
            label="qualification summary",
        )

    with pytest.raises(
        runtime_qualification.QualificationError,
        match="must name a file",
    ):
        runtime_qualification._fresh_destination(  # noqa: SLF001
            Path("."),
            label="qualification summary",
        )


def test_atomic_summary_publish_cannot_contaminate_evidence_through_alias(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "nested").mkdir(parents=True)
    alias = tmp_path / "outside-looking-alias"
    alias.symlink_to(evidence, target_is_directory=True)
    aliased_summary = alias / "nested" / "summary.json"

    with pytest.raises(runtime_qualification.QualificationError):
        runtime_qualification._atomic_no_clobber(  # noqa: SLF001
            aliased_summary,
            b'{"ok":true}\n',
        )

    assert not (evidence / "nested" / "summary.json").exists()
