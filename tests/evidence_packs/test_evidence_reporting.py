from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock import evidence_pack_integrity as integrity
from invarlock.evidence_pack_contract import (
    EVIDENCE_PATHS,
    INPUT_ROLES,
    MAX_OBSERVATION_BYTES,
    EvidenceObservation,
    build_comparison_report,
    canonical_json_bytes,
    evidence_observation_bytes,
    sha256_digest,
)
from invarlock.evidence_reporting import EvidenceReportError, render_evidence


def _report(*, extra_field: bool = False) -> dict[str, object]:
    payload = build_comparison_report(
        comparison_id="model-comparison",
        paired_records={
            "format": "invarlock/paired-records-v1",
            "metric": "exact_match",
            "schedule_sha256": "0" * 64,
            "records": [
                {
                    "record_id": "one",
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.0},
                },
                {
                    "record_id": "two",
                    "baseline": {"score": 0.0},
                    "subject": {"score": 1.0},
                },
            ],
        },
        policy={
            "resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -100.0}}}
        },
        policy_digest="sha256:" + "a" * 64,
    )
    if extra_field:
        payload["unsupported_claim"] = "trust me"
    return payload


def _signature_payload(
    manifest_bytes: bytes, *, key: ed25519.Ed25519PrivateKey
) -> bytes:
    public_key = key.public_key()
    return canonical_json_bytes(
        {
            "format": integrity.EVIDENCE_PACK_SIGNATURE_FORMAT,
            "algorithm": "ed25519",
            "signing_key_fingerprint": integrity.public_key_fingerprint(public_key),
            "public_key": {
                "encoding": "pem",
                "value": public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode("ascii"),
            },
            "signature": {
                "encoding": "base64",
                "value": base64.b64encode(key.sign(manifest_bytes)).decode("ascii"),
            },
        }
    )


def _evidence(
    tmp_path: Path,
    *,
    report_payload: dict[str, object] | None = None,
    report_extra_field: bool = False,
    manifest_format: str = "evidence-pack-v1",
    canonical_manifest: bool = True,
    checksummed_extra: bool = False,
    with_observation: bool = False,
    raw_observation_payload: dict[str, object] | None = None,
    canonical_observation: bool = True,
    duplicate_observation_request: bool = False,
) -> tuple[Path, str]:
    """Build a complete, evidence-signed minimal integrity pack."""

    root = tmp_path / "evidence"
    payloads: dict[str, bytes] = {}
    input_manifest: dict[str, object] = {}
    for index, role in enumerate(INPUT_ROLES):
        material_digest = "sha256:" + format(index + 1, "x") * 64
        identity_bytes = canonical_json_bytes(
            {
                "format": "invarlock/evidence-input-identity-v1",
                "role": role,
                "digest": material_digest,
                "locator": f"fixture://{role}",
            }
        )
        relative = f"inputs/{role}.json"
        payloads[relative] = identity_bytes
        input_manifest[role] = {
            "path": relative,
            "digest": sha256_digest(identity_bytes),
            "material_digest": material_digest,
        }

    for role, relative in EVIDENCE_PATHS.items():
        payloads[relative] = (
            b"provider: fixture\n"
            if relative.endswith(".yaml")
            else canonical_json_bytes({"fixture": role})
        )
    payloads[EVIDENCE_PATHS["evaluation_report"]] = canonical_json_bytes(
        report_payload or _report(extra_field=report_extra_field)
    )
    paired = canonical_json_bytes(
        {
            "format": "invarlock/paired-records-v1",
            "metric": "exact_match",
            "schedule_sha256": "0" * 64,
            "records": [
                {"record_id": "one"},
                {"record_id": "two"},
            ],
        }
    )
    records_path = "records/paired-records.json"
    payloads[records_path] = paired
    if checksummed_extra:
        payloads["unexpected.json"] = canonical_json_bytes({"unexpected": True})

    evidence_manifest = {
        role: {"path": relative, "digest": sha256_digest(payloads[relative])}
        for role, relative in EVIDENCE_PATHS.items()
    }
    observation_manifest: dict[str, object] | None = None
    if with_observation:
        observation_id = "spectral-summary"
        observation_payload_source = (
            raw_observation_payload
            if raw_observation_payload is not None
            else {
                "status": "observation",
                "verdict": "fail",
                "stable_rank": 1.25,
            }
        )
        if raw_observation_payload is None:
            observation_bytes = evidence_observation_bytes(
                EvidenceObservation(
                    observation_id=observation_id,
                    scope="subject",
                    kind="spectral",
                    payload=canonical_json_bytes(observation_payload_source),
                ),
                comparison_id="model-comparison",
                schedule_digest=input_manifest["dataset"]["material_digest"],
                policy_digest=input_manifest["policy"]["material_digest"],
                artifact_digests={
                    "baseline": input_manifest["baseline"]["material_digest"],
                    "subject": input_manifest["subject"]["material_digest"],
                },
            )
        else:
            # Deliberately bypass the publisher to model a separately signed pack.
            observation_bytes = canonical_json_bytes(
                {
                    "format": "invarlock/evidence-observation-v1",
                    "observation_id": observation_id,
                    "kind": "spectral",
                    "scope": "subject",
                    "authority": "observation",
                    "bindings": {
                        "comparison_id": "model-comparison",
                        "schedule_digest": input_manifest["dataset"]["material_digest"],
                        "policy_digest": input_manifest["policy"]["material_digest"],
                        "artifact_digests": {
                            "baseline": input_manifest["baseline"]["material_digest"],
                            "subject": input_manifest["subject"]["material_digest"],
                        },
                    },
                    "payload": observation_payload_source,
                }
            )
        observation_path = f"observations/{observation_id}.json"
        if not canonical_observation:
            observation_bytes = (
                json.dumps(json.loads(observation_bytes), indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
        payloads[observation_path] = observation_bytes
        observation_payload = json.loads(observation_bytes)["payload"]
        observation_descriptor = {
            "id": observation_id,
            "kind": "spectral",
            "scope": "subject",
            "payload_digest": sha256_digest(canonical_json_bytes(observation_payload)),
        }
        payloads[EVIDENCE_PATHS["request"]] = canonical_json_bytes(
            {
                "observations": [
                    observation_descriptor,
                    *(
                        [dict(observation_descriptor)]
                        if duplicate_observation_request
                        else []
                    ),
                ]
            }
        )
        evidence_manifest["request"] = {
            "path": EVIDENCE_PATHS["request"],
            "digest": sha256_digest(payloads[EVIDENCE_PATHS["request"]]),
        }
        observation_manifest = {
            observation_id: {
                "path": observation_path,
                "digest": sha256_digest(observation_bytes),
                "kind": "spectral",
                "scope": "subject",
            }
        }
    checksums = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {relative}\n"
        for relative, payload in sorted(payloads.items())
    ).encode("utf-8")
    key = ed25519.Ed25519PrivateKey.generate()
    signer = integrity.public_key_fingerprint(key.public_key())
    manifest = {
        "format": manifest_format,
        "comparison_id": "model-comparison",
        "inputs": input_manifest,
        "evidence": evidence_manifest,
        "paired_records": {
            "path": records_path,
            "digest": sha256_digest(paired),
            "count": 2,
        },
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": hashlib.sha256(checksums).hexdigest(),
        "signing_key_fingerprint": signer,
    }
    if observation_manifest is not None:
        manifest["observations"] = observation_manifest
    manifest_bytes = (
        canonical_json_bytes(manifest)
        if canonical_manifest
        else (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (root / "checksums.sha256").write_bytes(checksums)
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(
        _signature_payload(manifest_bytes, key=key)
    )
    return root, signer


def test_render_markdown_from_complete_evidence_signed_pack(
    tmp_path: Path,
) -> None:
    evidence, signer = _evidence(tmp_path)

    result = render_evidence(evidence, explain=True)

    assert "# InvarLock comparison report" in result.text
    assert "**Verdict:** **PASS**" in result.text
    assert "| Exact-match delta (pp) | 50 |" in result.text
    assert (
        "human rendering of the signature-authenticated evidence bundle" in result.text
    )
    assert "embedded evidence signature verified" in result.text
    assert "records the expected signer" in result.text
    assert "create the signed acceptance receipt" in result.text
    assert signer in result.text
    assert result.evidence_signer == signer
    assert result.html_path is None


def test_render_html_is_self_contained_and_no_clobber(tmp_path: Path) -> None:
    evidence, signer = _evidence(tmp_path)
    html = tmp_path / "reports" / "report.html"

    result = render_evidence(evidence, html_path=html)

    assert result.html_path == html.absolute()
    rendered = html.read_text(encoding="utf-8")
    assert "<h1>InvarLock comparison report</h1>" in rendered
    assert "human rendering of the signature-authenticated evidence bundle" in rendered
    assert signer in rendered
    with pytest.raises(EvidenceReportError, match="already exists"):
        render_evidence(evidence, html_path=html)


def test_render_authenticated_observations_separately_from_verdict(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path, with_observation=True)
    html = tmp_path / "observations.html"

    result = render_evidence(evidence, html_path=html)

    assert "**Verdict:** **PASS**" in result.text
    assert "## Authenticated observations" in result.text
    assert (
        "paired metric and policy remain the complete acceptance calculation"
        in result.text
    )
    assert "`spectral-summary`" in result.text
    assert '"verdict": "fail"' in result.text
    assert result.observations[0]["authority"] == "observation"
    rendered = html.read_text(encoding="utf-8")
    assert "<h2>Authenticated observations</h2>" in rendered
    assert (
        "paired metric and policy remain the complete acceptance calculation"
        in rendered
    )


def test_render_rejects_signed_observation_payload_over_publisher_limit(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(
        tmp_path,
        with_observation=True,
        raw_observation_payload={"blob": "x" * MAX_OBSERVATION_BYTES},
    )

    with pytest.raises(
        EvidenceReportError,
        match=f"payload exceeds the {MAX_OBSERVATION_BYTES}-byte limit",
    ):
        render_evidence(evidence)


def test_render_rejects_signed_noncanonical_observation_envelope(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(
        tmp_path,
        with_observation=True,
        canonical_observation=False,
    )

    with pytest.raises(EvidenceReportError, match="must use canonical JSON"):
        render_evidence(evidence)


def test_render_rejects_duplicate_signed_observation_request_entries(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(
        tmp_path,
        with_observation=True,
        duplicate_observation_request=True,
    )

    with pytest.raises(
        EvidenceReportError,
        match="normalized request observation entry is invalid",
    ):
        render_evidence(evidence)


def test_render_escapes_authenticated_observation_payload_in_html(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(
        tmp_path,
        with_observation=True,
        raw_observation_payload={"note": "<script>alert('observation')</script>"},
    )
    html = tmp_path / "observation.html"

    render_evidence(evidence, html_path=html)

    rendered = html.read_text(encoding="utf-8")
    assert "<script>alert('observation')</script>" not in rendered
    assert "&lt;script&gt;alert(&#x27;observation&#x27;)&lt;/script&gt;" in rendered


def test_render_rejects_tampered_bound_report(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path)
    report_path = evidence / EVIDENCE_PATHS["evaluation_report"]
    report_path.write_bytes(canonical_json_bytes({**_report(), "verdict": "fail"}))

    with pytest.raises(EvidenceReportError, match="checksum mismatch"):
        render_evidence(evidence)


def test_render_rejects_unchecksummed_extra_file(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path)
    (evidence / "extra.txt").write_text("not part of the pack\n", encoding="utf-8")

    with pytest.raises(EvidenceReportError, match="extra files"):
        render_evidence(evidence)


def test_render_rejects_checksummed_file_outside_manifest(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path, checksummed_extra=True)

    with pytest.raises(EvidenceReportError, match="outside the evidence manifest"):
        render_evidence(evidence)


def test_render_rejects_tampered_evidence_signature(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path)
    signature_path = evidence / integrity.MANIFEST_SIGNATURE_FILENAME
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    signature["signature"]["value"] = base64.b64encode(b"\0" * 64).decode("ascii")
    signature_path.write_bytes(canonical_json_bytes(signature))

    with pytest.raises(EvidenceReportError, match="signature verification failed"):
        render_evidence(evidence)


def test_render_rejects_incomplete_pack(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path)
    (evidence / EVIDENCE_PATHS["subject_run_report"]).unlink()

    with pytest.raises(EvidenceReportError, match="missing"):
        render_evidence(evidence)


def test_render_rejects_authenticated_nonclosed_report(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path, report_extra_field=True)

    with pytest.raises(EvidenceReportError, match="fields are invalid"):
        render_evidence(evidence)


def test_render_rejects_authenticated_semantically_false_verdict(
    tmp_path: Path,
) -> None:
    report = _report()
    report["verdict"] = "fail"
    evidence, _signer = _evidence(tmp_path, report_payload=report)

    with pytest.raises(EvidenceReportError, match="verdict does not match"):
        render_evidence(evidence)


@pytest.mark.parametrize(
    ("metric", "comparison"),
    [
        (
            "exact_match",
            {"kind": "normalized_nll_ratio", "value": 1.0, "maximum": 1.1},
        ),
        (
            "normalized_nll_per_utf8_byte",
            {"kind": "exact_match_delta_pp", "value": 0.0, "minimum": 0.0},
        ),
    ],
)
def test_render_rejects_metric_comparison_kind_contradiction(
    tmp_path: Path,
    metric: str,
    comparison: dict[str, object],
) -> None:
    report = _report()
    report["metric"] = metric
    report["comparison"] = comparison
    if metric == "normalized_nll_per_utf8_byte":
        report.pop("paired_binary")
        report["derived_measurements"] = {
            "perplexity_ratio": {
                "status": "unavailable",
                "basis": "authenticated_target_likelihood",
                "method": "target_token_weighted_perplexity_ratio_v1",
                "reason": "target_token_counts_unavailable",
            }
        }
    evidence, _signer = _evidence(tmp_path, report_payload=report)

    with pytest.raises(EvidenceReportError, match="do not agree"):
        render_evidence(evidence)


def test_render_rejects_signed_noncanonical_manifest(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path, canonical_manifest=False)

    with pytest.raises(EvidenceReportError, match="manifest is not canonical JSON"):
        render_evidence(evidence)


def test_render_rejects_non_manifest(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path, manifest_format="unsupported-pack")

    with pytest.raises(EvidenceReportError, match="manifest schema failed"):
        render_evidence(evidence)


def test_render_html_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    evidence, _signer = _evidence(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(EvidenceReportError, match="could not write HTML report"):
        render_evidence(evidence, html_path=linked_parent / "report.html")
    assert not (real_parent / "report.html").exists()
