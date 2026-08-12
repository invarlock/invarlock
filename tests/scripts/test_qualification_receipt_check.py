from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.evidence_receipt import write_signed_verification_receipt
from scripts import qualification_receipt_check
from scripts.qualification_receipt_check import validate

ROOT = Path(__file__).resolve().parents[2]


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _provider_receipt_bytes(*, provider: str, device_kind: str) -> bytes:
    payload = json.loads(
        ROOT.joinpath(
            "examples/import/baseline/runtime-provider.receipt.json"
        ).read_text(encoding="utf-8")
    )
    payload["plugin"]["name"] = provider
    payload["capabilities"]["provider_name"] = provider
    payload["device"]["device_kind"] = device_kind
    payload["device"]["device_name"] = f"fixture-{device_kind}"
    return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _scorer_binding(
    *,
    scorer_id: str = "example.structured_f1",
    scorer_version: str = "1.0.0",
    descriptor_marker: str = "1",
    configuration: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_configuration = configuration or {"normalization": "strict"}
    configuration_bytes = json.dumps(
        selected_configuration,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {
        "configuration": selected_configuration,
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "descriptor_sha256": descriptor_marker * 64,
        "format_version": "invarlock/scorer-extension-binding-v1",
        "scorer_abi": "1",
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
    }


def _target_request(
    tmp_path: Path,
    *,
    metric: str | None = "exact_match",
    scorer_extension: dict[str, object] | None = None,
) -> Path:
    payload = yaml.safe_load(ROOT.joinpath("examples/request.yaml").read_text())
    comparison = payload["comparison"]
    comparison.pop("metric", None)
    comparison.pop("scorer_extension", None)
    if metric is not None:
        comparison["metric"] = metric
    if scorer_extension is not None:
        comparison["scorer_extension"] = scorer_extension
    path = tmp_path / "target-request.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _case(
    tmp_path: Path,
    *,
    baseline_runtime: str | None = None,
    subject_runtime: str | None = None,
    result_ok: bool = True,
    integrity_ok: bool = True,
    policy_verdict: str = "pass",
    result_status: EvidencePackStatus = EvidencePackStatus.OK,
    baseline_provider: str = "hf_transformers",
    subject_provider: str = "hf_transformers",
    baseline_receipt_provider: str | None = None,
    subject_receipt_provider: str | None = None,
    baseline_device: str = "cpu",
    subject_device: str = "cpu",
    task: str = "text_causal",
    metric: str | None = "exact_match",
    scorer_extension: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, str]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comparison: dict[str, object] = {
        "baseline": {"runtime": {"provider": baseline_provider}},
        "subject": {"runtime": {"provider": subject_provider}},
        "task": task,
    }
    if metric is not None:
        comparison["metric"] = metric
    if scorer_extension is not None:
        comparison["scorer_extension"] = scorer_extension
    request = (
        json.dumps(
            {"comparison": comparison},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    evidence.joinpath("request.json").write_bytes(request)
    payloads = {"request.json": request}
    for role, provider, device_kind in (
        (
            "baseline",
            baseline_receipt_provider or baseline_provider,
            baseline_device,
        ),
        (
            "subject",
            subject_receipt_provider or subject_provider,
            subject_device,
        ),
    ):
        relative = f"providers/{role}/runtime-provider.receipt.json"
        payload = _provider_receipt_bytes(
            provider=provider,
            device_kind=device_kind,
        )
        path = evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        payloads[relative] = payload
    checksums = "".join(
        f"{hashlib.sha256(payloads[relative]).hexdigest()}  {relative}\n"
        for relative in sorted(payloads)
    ).encode()
    evidence.joinpath("checksums.sha256").write_bytes(checksums)
    manifest = (
        json.dumps(
            {
                "checksums_sha256_digest": hashlib.sha256(checksums).hexdigest(),
                "format": "invarlock/evidence-pack-v1",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    evidence.joinpath("manifest.json").write_bytes(manifest)
    trust = tmp_path / "trust"
    trust.mkdir()
    policy = trust / "policy.json"
    policy.write_text('{"policy":"qualification"}\n', encoding="utf-8")
    verifier_key = ed25519.Ed25519PrivateKey.generate()
    verifier = trust / "verifier.pem"
    verifier.write_bytes(
        verifier_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    profile = trust / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy.json"},
                "anchors": {
                    "baseline_artifact_digest": _digest("a"),
                    "subject_artifact_digest": _digest("b"),
                    "schedule_digest": _digest("c"),
                    "baseline_runtime_digest": baseline_runtime or _digest("d"),
                    "subject_runtime_digest": subject_runtime or _digest("e"),
                    "evidence_signer_fingerprint": _digest("f"),
                },
                "verifier": {
                    "identity": "invarlock-verifier/qualification",
                    "signing_key_path": "verifier.pem",
                },
                "allow_installed_scorers": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    from invarlock.trust_inputs import load_trust_inputs

    loaded = load_trust_inputs(profile)
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    result = EvidencePackResult(
        payload={
            "ok": result_ok,
            "integrity_ok": integrity_ok,
            "policy_verdict": policy_verdict,
            "anchors": {
                "policy_digest": "sha256:"
                + hashlib.sha256(policy.read_bytes()).hexdigest(),
                "artifact_digests": dict(loaded.expected_artifact_digests),
                "schedule_digest": loaded.expected_schedule_digest,
                "runtime_digests": dict(loaded.expected_runtime_digests),
                "signer_fingerprint": loaded.expected_signer_fingerprint,
            },
        },
        status=result_status,
        manifest_digest=manifest_digest,
    )
    receipt = tmp_path / "receipt.json"
    write_signed_verification_receipt(
        evidence,
        result,
        receipt,
        policy_path=policy,
        expected_artifact_digests=dict(loaded.expected_artifact_digests),
        expected_schedule_digest=loaded.expected_schedule_digest,
        expected_runtime_digests=dict(loaded.expected_runtime_digests),
        expected_pack_signer_fingerprint=loaded.expected_signer_fingerprint,
        verifier_identity=loaded.verifier_identity,
        verifier_signing_key_path=verifier,
        trust_profile_digest=loaded.profile_digest,
    )
    return receipt, evidence, profile, manifest_digest


def test_qualification_receipt_check_binds_captured_signed_bytes(
    tmp_path: Path,
) -> None:
    receipt, evidence, profile, manifest_digest = _case(tmp_path)

    result = validate(
        receipt=receipt,
        evidence=evidence,
        trust_profile=profile,
    )

    assert result == {
        "format_version": "invarlock/qualification-receipt-check-v1",
        "ok": True,
        "pack_manifest_digest": manifest_digest,
        "receipt_sha256": "sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "verifier_fingerprint": public_key_fingerprint(
            serialization.load_pem_private_key(
                profile.parent.joinpath("verifier.pem").read_bytes(),
                password=None,
            ).public_key()
        ),
    }


def test_qualification_receipt_check_replays_after_private_key_destruction(
    tmp_path: Path,
) -> None:
    receipt, evidence, profile, manifest_digest = _case(tmp_path)
    verifier_key_path = profile.parent / "verifier.pem"
    private_key = serialization.load_pem_private_key(
        verifier_key_path.read_bytes(), password=None
    )
    assert isinstance(private_key, ed25519.Ed25519PrivateKey)
    public_key_path = profile.parent / "verifier-public.pem"
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    verifier_key_path.unlink()

    result = validate(
        receipt=receipt,
        evidence=evidence,
        trust_profile=profile,
        verifier_public_key=public_key_path,
    )

    assert result["ok"] is True
    assert result["pack_manifest_digest"] == manifest_digest
    other_key = ed25519.Ed25519PrivateKey.generate().public_key()
    public_key_path.write_bytes(
        other_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(ValueError, match="does not match caller expectation"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            verifier_public_key=public_key_path,
        )


def test_qualification_receipt_check_rejects_substituted_bytes(
    tmp_path: Path,
) -> None:
    receipt, evidence, profile, _manifest_digest = _case(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["statement"]["verdict"]["ok"] = False
    receipt.chmod(0o600)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    try:
        validate(receipt=receipt, evidence=evidence, trust_profile=profile)
    except ValueError as exc:
        assert "signature verification failed" in str(exc)
    else:  # pragma: no cover - explicit adversarial assertion
        raise AssertionError("substituted receipt was accepted")


def test_qualification_receipt_check_rejects_substituted_canary_evidence(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    evidence.joinpath("manifest.json").write_text(
        '{"format":"substituted"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not bind the supplied pack manifest"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
        )


def test_qualification_receipt_check_rejects_wrong_canary_trust_profile(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    tmp_path.joinpath("canary").mkdir()
    tmp_path.joinpath("other").mkdir()
    receipt, evidence, _profile, _manifest_digest = _case(
        tmp_path / "canary",
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    _other_receipt, _other_evidence, other_profile, _other_manifest = _case(
        tmp_path / "other",
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )

    with pytest.raises(ValueError, match="verifier key|trust profile"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=other_profile,
            expected_runtime_image_digest=runtime,
        )


def test_qualification_receipt_check_binds_both_sides_to_exact_canary_image(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )

    result = validate(
        receipt=receipt,
        evidence=evidence,
        trust_profile=profile,
        expected_runtime_image_digest=runtime,
    )

    assert result["runtime_image_digest"] == runtime


def test_qualification_receipt_check_binds_canary_provider_and_task_identity(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )

    result = validate(
        receipt=receipt,
        evidence=evidence,
        trust_profile=profile,
        expected_runtime_image_digest=runtime,
        expected_request=ROOT / "examples/request.yaml",
        expected_request_root=ROOT / "examples",
        expected_runtime_device="cpu",
    )

    assert result["compatibility"] == {
        "acceptance": {"kind": "builtin_metric", "metric": "exact_match"},
        "device_classes": {"baseline": "cpu", "subject": "cpu"},
        "providers": {
            "baseline": "hf_transformers",
            "subject": "hf_transformers",
        },
        "task": "text_causal",
    }


@pytest.mark.parametrize(
    ("canary_metric", "target_metric"),
    (
        ("exact_match", "normalized_nll_per_utf8_byte"),
        ("normalized_nll_per_utf8_byte", "exact_match"),
    ),
)
def test_qualification_receipt_check_rejects_builtin_acceptance_mismatch(
    tmp_path: Path,
    canary_metric: str,
    target_metric: str,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        metric=canary_metric,
    )
    target = _target_request(tmp_path, metric=target_metric)

    with pytest.raises(ValueError, match="acceptance/device compatibility"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=target,
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


@pytest.mark.parametrize(
    "target_binding",
    (
        _scorer_binding(scorer_id="example.execution_accuracy"),
        _scorer_binding(scorer_version="1.1.0"),
        _scorer_binding(descriptor_marker="2"),
        _scorer_binding(configuration={"normalization": "lenient"}),
    ),
)
def test_qualification_receipt_check_rejects_scorer_acceptance_mismatch(
    tmp_path: Path,
    target_binding: dict[str, object],
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        metric=None,
        scorer_extension=_scorer_binding(),
    )
    target = _target_request(
        tmp_path,
        metric=None,
        scorer_extension=target_binding,
    )

    with pytest.raises(ValueError, match="acceptance/device compatibility"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=target,
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


def test_qualification_receipt_check_binds_matching_scorer_and_cuda_class(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    binding = _scorer_binding()
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        baseline_device="cuda",
        subject_device="cuda",
        metric=None,
        scorer_extension=binding,
    )
    target = _target_request(
        tmp_path,
        metric=None,
        scorer_extension=binding,
    )

    result = validate(
        receipt=receipt,
        evidence=evidence,
        trust_profile=profile,
        expected_runtime_image_digest=runtime,
        expected_request=target,
        expected_request_root=ROOT / "examples",
        expected_runtime_device="cuda:3",
    )

    assert result["compatibility"] == {
        "acceptance": {
            "configuration_sha256": binding["configuration_sha256"],
            "descriptor_sha256": binding["descriptor_sha256"],
            "kind": "scorer_extension",
            "scorer_id": binding["scorer_id"],
            "scorer_version": binding["scorer_version"],
        },
        "device_classes": {"baseline": "cuda", "subject": "cuda"},
        "providers": {
            "baseline": "hf_transformers",
            "subject": "hf_transformers",
        },
        "task": "text_causal",
    }


@pytest.mark.parametrize(
    ("baseline_device", "subject_device", "expected_runtime_device"),
    (
        ("cpu", "cpu", "cuda"),
        ("cuda", "cuda", "cpu"),
        ("cpu", "cuda", "cuda:0"),
    ),
)
def test_qualification_receipt_check_rejects_runtime_device_class_mismatch(
    tmp_path: Path,
    baseline_device: str,
    subject_device: str,
    expected_runtime_device: str,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        baseline_device=baseline_device,
        subject_device=subject_device,
    )

    with pytest.raises(ValueError, match="acceptance/device compatibility"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device=expected_runtime_device,
        )


def test_qualification_receipt_check_rejects_provider_receipt_request_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        baseline_receipt_provider="llama_cpp",
    )

    with pytest.raises(ValueError, match="provider receipt does not match request"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


@pytest.mark.parametrize(
    ("overrides",),
    (
        ({"baseline_provider": "llama_cpp"},),
        ({"subject_provider": "llama_cpp"},),
        ({"task": "vision_text_generation"},),
    ),
)
def test_qualification_receipt_check_rejects_incompatible_canary_identity(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        **overrides,
    )

    with pytest.raises(ValueError, match="provider/task/acceptance/device"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


def test_qualification_receipt_check_rejects_unbound_canary_request_substitution(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    evidence.joinpath("request.json").write_text(
        '{"comparison":{"baseline":{"runtime":{"provider":"llama_cpp"}}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canary evidence integrity failed"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


def test_qualification_receipt_check_rejects_tampered_provider_receipt(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    provider_receipt = evidence / "providers/baseline/runtime-provider.receipt.json"
    payload = json.loads(provider_receipt.read_text(encoding="utf-8"))
    payload["device"]["device_kind"] = "cuda"
    provider_receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum|integrity"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


def test_qualification_receipt_check_rejects_provider_receipt_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    provider_receipt = evidence / "providers/baseline/runtime-provider.receipt.json"
    original_verify_checksums = qualification_receipt_check.verify_checksums
    mutated = False

    def verify_then_mutate(root: Path) -> tuple[list[str], set[str]]:
        nonlocal mutated
        result = original_verify_checksums(root)
        if not mutated:
            mutated = True
            payload = json.loads(provider_receipt.read_text(encoding="utf-8"))
            payload["device"]["device_kind"] = "cuda"
            provider_receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        qualification_receipt_check,
        "verify_checksums",
        verify_then_mutate,
    )

    with pytest.raises(ValueError, match="changed after capture"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
            expected_request=ROOT / "examples/request.yaml",
            expected_request_root=ROOT / "examples",
            expected_runtime_device="cpu",
        )


def test_qualification_receipt_check_accepts_read_only_canary_inputs(
    tmp_path: Path,
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )
    receipt.chmod(0o400)
    profile.chmod(0o400)
    receipt.parent.chmod(0o500)
    try:
        result = validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
        )
    finally:
        receipt.parent.chmod(0o700)

    assert result["runtime_image_digest"] == runtime


@pytest.mark.parametrize(
    ("baseline_runtime", "subject_runtime"),
    ((_digest("d"), _digest("e")), (_digest("e"), _digest("d"))),
)
def test_qualification_receipt_check_rejects_any_canary_image_mismatch(
    tmp_path: Path,
    baseline_runtime: str,
    subject_runtime: str,
) -> None:
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=baseline_runtime,
        subject_runtime=subject_runtime,
    )

    with pytest.raises(ValueError, match="exact qualification image"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=_digest("d"),
        )


@pytest.mark.parametrize(
    ("case_overrides",),
    (
        ({"result_ok": False},),
        ({"integrity_ok": False},),
        ({"policy_verdict": "fail"},),
        ({"result_status": EvidencePackStatus.INTEGRITY_ONLY},),
    ),
)
def test_qualification_receipt_check_rejects_signed_nonpassing_canary(
    tmp_path: Path,
    case_overrides: dict[str, object],
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
        **case_overrides,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="strict passing verdict|successful verdict is inconsistent",
    ):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest=runtime,
        )


def test_qualification_receipt_check_main_emits_bound_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt, evidence, profile, manifest_digest = _case(tmp_path)

    result = qualification_receipt_check.main(
        [
            "--receipt",
            str(receipt),
            "--evidence",
            str(evidence),
            "--trust-profile",
            str(profile),
        ]
    )

    assert result == 0
    assert (
        json.loads(capsys.readouterr().out)["pack_manifest_digest"] == manifest_digest
    )


def test_qualification_receipt_check_main_emits_exact_canary_image_binding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _digest("d")
    receipt, evidence, profile, _manifest_digest = _case(
        tmp_path,
        baseline_runtime=runtime,
        subject_runtime=runtime,
    )

    assert (
        qualification_receipt_check.main(
            [
                "--receipt",
                str(receipt),
                "--evidence",
                str(evidence),
                "--trust-profile",
                str(profile),
                "--expected-runtime-image-digest",
                runtime,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["runtime_image_digest"] == runtime


def test_qualification_receipt_check_rejects_missing_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, evidence, profile, _manifest_digest = _case(tmp_path)

    class Result:
        ok = True
        signed = True
        statement: dict[str, object] = {}
        errors: tuple[str, ...] = ()

    monkeypatch.setattr(
        qualification_receipt_check,
        "verify_signed_verification_receipt",
        lambda *_args, **_kwargs: Result(),
    )

    with pytest.raises(ValueError, match="manifest identity is missing"):
        validate(receipt=receipt, evidence=evidence, trust_profile=profile)


@pytest.mark.parametrize(
    ("metric", "scorer", "message"),
    (
        (None, None, "exactly one"),
        ("exact_match", _scorer_binding(), "exactly one"),
        ("accuracy", None, "built-in acceptance metric"),
        (None, {"format_version": "bad"}, "scorer acceptance binding"),
    ),
)
def test_acceptance_compatibility_rejects_ambiguous_or_invalid_selection(
    metric: object, scorer: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        qualification_receipt_check._acceptance_compatibility(
            metric=metric, scorer_extension=scorer
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ([], "JSON object"),
        ({}, "comparison is missing"),
        (
            {"comparison": {"task": "text_causal", "metric": "exact_match"}},
            "baseline provider",
        ),
        (
            {
                "comparison": {
                    "baseline": {"runtime": {"provider": "hf_transformers"}},
                    "subject": {"runtime": {"provider": "hf_transformers"}},
                    "metric": "exact_match",
                }
            },
            "task identity",
        ),
    ),
)
def test_request_compatibility_rejects_missing_authenticated_identity(
    payload: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        qualification_receipt_check._request_compatibility(payload, label="canary")


def test_runtime_device_class_accepts_device_indices_and_rejects_other_devices() -> (
    None
):
    assert qualification_receipt_check._runtime_device_class("cpu") == "cpu"
    assert qualification_receipt_check._runtime_device_class("cuda:7") == "cuda"
    with pytest.raises(ValueError, match="must be cpu"):
        qualification_receipt_check._runtime_device_class("mps")


def test_authenticated_canary_rejects_missing_pack_and_manifest_substitution(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="snapshot failed"):
        qualification_receipt_check._authenticated_canary_compatibility(
            tmp_path / "missing", expected_manifest_digest=_digest("a")
        )

    case = tmp_path / "case"
    case.mkdir()
    _receipt, evidence, _profile, _manifest = _case(case)
    with pytest.raises(ValueError, match="manifest changed"):
        qualification_receipt_check._authenticated_canary_compatibility(
            evidence, expected_manifest_digest=_digest("a")
        )


def test_authenticated_canary_requires_checksum_coverage_for_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, evidence, _profile, manifest_digest = _case(tmp_path)
    assert receipt.is_file()
    verify_checksums = qualification_receipt_check.verify_checksums

    def without_request(root: Path) -> tuple[list[str], set[str]]:
        errors, covered = verify_checksums(root)
        return errors, covered - {"request.json"}

    monkeypatch.setattr(
        qualification_receipt_check,
        "verify_checksums",
        without_request,
    )

    with pytest.raises(ValueError, match="request.json is not covered"):
        qualification_receipt_check._authenticated_canary_compatibility(
            evidence,
            expected_manifest_digest=manifest_digest,
        )


def test_authenticated_canary_rejects_checksum_valid_invalid_provider_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _receipt, evidence, _profile, manifest_digest = _case(tmp_path)
    monkeypatch.setattr(
        qualification_receipt_check,
        "decode_runtime_provider_receipt",
        lambda _payload: (_ for _ in ()).throw(ValueError("invalid sidecar")),
    )

    with pytest.raises(ValueError, match="runtime provider receipt is invalid"):
        qualification_receipt_check._authenticated_canary_compatibility(
            evidence,
            expected_manifest_digest=manifest_digest,
        )


@pytest.mark.parametrize("kind", ("private-rsa", "public-rsa"))
def test_receipt_check_requires_ed25519_verifier_material(
    tmp_path: Path, kind: str
) -> None:
    receipt, evidence, profile, _manifest = _case(tmp_path)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    if kind == "private-rsa":
        profile.parent.joinpath("verifier.pem").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        with pytest.raises(ValueError, match="signing key must be Ed25519"):
            validate(receipt=receipt, evidence=evidence, trust_profile=profile)
    else:
        public = tmp_path / "verifier.pub.pem"
        public.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        with pytest.raises(ValueError, match="public key must be Ed25519"):
            validate(
                receipt=receipt,
                evidence=evidence,
                trust_profile=profile,
                verifier_public_key=public,
            )


def test_receipt_check_requires_complete_compatibility_inputs_and_digest(
    tmp_path: Path,
) -> None:
    receipt, evidence, profile, _manifest = _case(tmp_path)
    with pytest.raises(ValueError, match="must be supplied together"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_request=tmp_path / "request.json",
        )
    with pytest.raises(ValueError, match="lowercase sha256"):
        validate(
            receipt=receipt,
            evidence=evidence,
            trust_profile=profile,
            expected_runtime_image_digest="latest",
        )
