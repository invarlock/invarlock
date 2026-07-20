from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock import evidence_reporting as reporting
from invarlock.evidence_pack_contract import build_comparison_report
from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from tests.evidence_packs.test_evidence_reporting import _evidence, _report


def _nll_report() -> dict[str, object]:
    return build_comparison_report(
        comparison_id="model-comparison",
        paired_records={
            "format": "invarlock/paired-records-v1",
            "metric": "normalized_nll_per_utf8_byte",
            "schedule_sha256": "0" * 64,
            "records": [
                {
                    "record_id": "one",
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.05},
                },
                {
                    "record_id": "two",
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.05},
                },
            ],
            "derived_measurements": {
                "perplexity_ratio": {
                    "status": "unavailable",
                    "basis": "authenticated_target_likelihood",
                    "method": "target_token_weighted_perplexity_ratio_v1",
                    "reason": "target_token_counts_unavailable",
                }
            },
        },
        policy={
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.1}}
            }
        },
        policy_digest="sha256:" + "a" * 64,
    )


def test_closed_report_renderer_covers_normalized_nll_and_html_escaping() -> None:
    report = _nll_report()
    report["comparison_id"] = "<comparison>"

    closed = reporting._closed_comparison_report(report)
    markdown = reporting._render_markdown(
        closed,
        explain=False,
        evidence_signer="sha256:" + "a" * 64,
        observations=(),
    )
    html = reporting._render_html(
        closed,
        explain=True,
        evidence_signer="<unsafe-signer>",
        observations=(),
    )

    assert "Normalized NLL ratio" in markdown
    assert "Maximum allowed ratio" in markdown
    assert "&lt;comparison&gt;" in html
    assert "&lt;unsafe-signer&gt;" in html
    assert "manifest, checksums" in html


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "fields are invalid"),
        (lambda value: value.update(format="other"), "format is invalid"),
        (lambda value: value.update(comparison_id=""), "comparison_id is invalid"),
        (lambda value: value.update(record_count=True), "record_count is invalid"),
        (lambda value: value.update(baseline=[]), "baseline is invalid"),
        (
            lambda value: value["baseline"].update(mean_score=True),
            "baseline.mean_score must be a number",
        ),
        (lambda value: value.update(comparison=[]), "comparison is invalid"),
        (
            lambda value: value["comparison"].update(kind="other"),
            "comparison is invalid",
        ),
        (
            lambda value: value["comparison"].update(value=True),
            "comparison.value must be a number",
        ),
        (
            lambda value: value["comparison"].update(value=float("inf")),
            "comparison.value must be finite",
        ),
        (lambda value: value.update(verdict="unknown"), "verdict is invalid"),
    ],
)
def test_closed_report_rejects_open_or_ambiguous_display_fields(
    mutate: object, message: str
) -> None:
    report = _report()
    assert callable(mutate)
    mutate(report)

    with pytest.raises(EvidenceReportError, match=message):
        reporting._closed_comparison_report(report)


def test_closed_report_rejects_internally_contradictory_values() -> None:
    report = _report()
    report["comparison"] = {
        "kind": "exact_match_delta_pp",
        "value": 49.0,
        "minimum": -1.0,
    }
    with pytest.raises(EvidenceReportError, match="does not match the side means"):
        reporting._closed_comparison_report(report)

    report = _report()
    report["verdict"] = "fail"
    with pytest.raises(EvidenceReportError, match="verdict does not match"):
        reporting._closed_comparison_report(report)


def test_render_rejects_unsafe_evidence_root_and_html_destination(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path)
    alias = tmp_path / "evidence-link"
    alias.symlink_to(evidence, target_is_directory=True)
    with pytest.raises(EvidenceReportError, match="real directory"):
        render_evidence(alias)

    with pytest.raises(EvidenceReportError, match="must name a regular file"):
        render_evidence(evidence, html_path=Path("/"))


def test_report_error_exposes_stable_exit_code() -> None:
    assert EvidenceReportError("invalid").exit_code == 2
    assert EvidenceReportError("write failed", exit_code=1).exit_code == 1


def _qualified_exact_report() -> dict[str, object]:
    return build_comparison_report(
        comparison_id="qualified",
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
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": -100.0,
                        "minimum_record_count": 2,
                        "maximum_interval_width_pp": 200.0,
                    }
                }
            }
        },
        policy_digest="sha256:" + "a" * 64,
    )


def _replace_nll_comparison_with_exact_kind(value: dict[str, object]) -> None:
    comparison = value["comparison"]
    assert isinstance(comparison, dict)
    comparison.pop("maximum")
    comparison.update(kind="exact_match_delta_pp", minimum=-10.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(policy_digest=""), "policy_digest is invalid"),
        (lambda value: value.update(record_count=0), "record_count is invalid"),
        (
            lambda value: value["subject"].update(mean_score=2.0),
            "score means must be between zero and one",
        ),
        (
            lambda value: value["comparison"].update(value=101.0),
            "score comparison is out of range",
        ),
        (
            lambda value: value["comparison"].update(minimum=-101.0),
            "score policy limit is out of range",
        ),
        (
            lambda value: value["uncertainty"].update(lower=-101.0),
            "score uncertainty is out of range",
        ),
        (lambda value: value.update(uncertainty=[]), "uncertainty is invalid"),
        (
            lambda value: value["uncertainty"].update(extra=True),
            "uncertainty is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(method="other"),
            "uncertainty method is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(scope="other"),
            "uncertainty scope is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(interval_mass=0.9),
            "uncertainty mass is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(lower=60.0, upper=50.0),
            "uncertainty bounds are invalid",
        ),
        (lambda value: value.update(paired_binary=[]), "paired_binary is invalid"),
        (
            lambda value: value["paired_binary"].update(discordant_pairs=0),
            "paired_binary counts are invalid",
        ),
        (
            lambda value: value["paired_binary"].update(
                mcnemar_exact_two_sided_p_value=2.0
            ),
            "paired_binary values are invalid",
        ),
        (
            lambda value: value["paired_binary"].update(
                effect_size_confidence_interval={}
            ),
            "paired_binary values are invalid",
        ),
    ],
)
def test_closed_exact_report_rejects_every_untrusted_numeric_surface(
    mutate: object, message: str
) -> None:
    report = _report()
    assert callable(mutate)
    mutate(report)
    with pytest.raises(EvidenceReportError, match=message):
        reporting._closed_comparison_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _replace_nll_comparison_with_exact_kind,
            "metric and comparison kind",
        ),
        (
            lambda value: value["uncertainty"].update(extra=True),
            "uncertainty is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(method="other"),
            "uncertainty method is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(scope="other"),
            "uncertainty scope is invalid",
        ),
        (
            lambda value: value["uncertainty"].update(replicates=1024),
            "uncertainty replicates are invalid",
        ),
        (
            lambda value: value["baseline"].update(mean_score=0.0),
            "ratio means must be non-negative",
        ),
        (
            lambda value: value["comparison"].update(maximum=0.0),
            "ratio comparison and uncertainty",
        ),
        (
            lambda value: value.update(derived_measurements={}),
            "derived measurements are invalid",
        ),
        (
            lambda value: value["comparison"].update(value=1.01),
            "does not match the side means",
        ),
    ],
)
def test_closed_nll_report_rejects_invalid_resampling_and_ratio_surfaces(
    mutate: object, message: str
) -> None:
    report = _nll_report()
    assert callable(mutate)
    mutate(report)
    with pytest.raises(EvidenceReportError, match=message):
        reporting._closed_comparison_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(sample_qualification=[]),
            "sample_qualification is invalid",
        ),
        (
            lambda value: value["sample_qualification"].update(record_count=[]),
            "record_count is invalid",
        ),
        (
            lambda value: value["sample_qualification"].update(interval_width=[]),
            "interval_width is invalid",
        ),
        (
            lambda value: value["sample_qualification"]["record_count"].update(
                minimum=0
            ),
            "values are invalid",
        ),
        (
            lambda value: value["sample_qualification"]["interval_width"].update(
                unit="ratio"
            ),
            "values are invalid",
        ),
        (
            lambda value: value["sample_qualification"]["record_count"].update(
                passed=False
            ),
            "verdict is invalid",
        ),
    ],
)
def test_closed_report_replays_sample_qualification_fields(
    mutate: object, message: str
) -> None:
    report = copy.deepcopy(_qualified_exact_report())
    assert callable(mutate)
    mutate(report)
    with pytest.raises(EvidenceReportError, match=message):
        reporting._closed_comparison_report(report)


def test_renderer_includes_available_perplexity_and_sample_qualification() -> None:
    report = _nll_report()
    report["derived_measurements"] = {
        "perplexity_ratio": {
            "status": "available",
            "basis": "authenticated_target_likelihood",
            "method": "target_token_weighted_perplexity_ratio_v1",
            "tokenizer_metadata_sha256": "b" * 64,
            "target_token_count": 2,
            "baseline_perplexity": 2.0,
            "subject_perplexity": 2.1,
            "ratio": 1.05,
        }
    }
    closed = reporting._closed_comparison_report(report)
    assert "Baseline perplexity" in reporting._render_markdown(
        closed,
        explain=True,
        evidence_signer="sha256:" + "a" * 64,
        observations=[],
    )
    assert "Baseline perplexity" in reporting._render_html(
        closed,
        explain=False,
        evidence_signer="sha256:" + "a" * 64,
        observations=[],
    )

    qualified = reporting._closed_comparison_report(_qualified_exact_report())
    assert "Sample qualification" in reporting._render_markdown(
        qualified,
        explain=False,
        evidence_signer="sha256:" + "a" * 64,
        observations=[],
    )


def _manifest(evidence: Path) -> dict[str, object]:
    payload = json.loads((evidence / "manifest.json").read_bytes())
    assert isinstance(payload, dict)
    return payload


def test_report_json_loader_and_manifest_path_inventory_fail_closed(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"[]")
    with pytest.raises(EvidenceReportError, match="must be a JSON object"):
        reporting._load_object_with_bytes(invalid, label="input", max_bytes=10)
    invalid.write_bytes(b"{")
    with pytest.raises(EvidenceReportError):
        reporting._load_object_with_bytes(invalid, label="input", max_bytes=10)

    manifest = {
        "inputs": {"skip": None, "kept": {"path": "inputs/kept.json"}},
        "evidence": [],
        "paired_records": {"path": "records/paired.json"},
        "observations": {
            "skip": None,
            "kept": {"path": "observations/kept.json"},
        },
    }
    assert reporting._manifest_payload_paths(manifest) == {
        "inputs/kept.json",
        "records/paired.json",
        "observations/kept.json",
    }


def test_observation_loader_rejects_each_untrusted_manifest_surface(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path, with_observation=True)
    manifest = _manifest(evidence)

    cases: list[tuple[dict[str, object], str]] = []
    missing_request = copy.deepcopy(manifest)
    missing_request["evidence"] = {}
    cases.append((missing_request, "request reference is invalid"))

    unsafe_request = copy.deepcopy(manifest)
    unsafe_request["evidence"]["request"]["path"] = "../request.json"  # type: ignore[index]
    cases.append((unsafe_request, "request path is unsafe"))

    invalid_references = copy.deepcopy(manifest)
    invalid_references["observations"] = []
    cases.append((invalid_references, "manifest observations are invalid"))

    mismatched = copy.deepcopy(manifest)
    mismatched["observations"] = {}
    cases.append((mismatched, "do not match normalized request"))

    invalid_inputs = copy.deepcopy(manifest)
    invalid_inputs["inputs"] = []
    cases.append((invalid_inputs, "manifest inputs are invalid"))

    unavailable_bindings = copy.deepcopy(manifest)
    unavailable_bindings["comparison_id"] = None
    cases.append((unavailable_bindings, "bindings are unavailable"))

    invalid_entry = copy.deepcopy(manifest)
    invalid_entry["observations"]["spectral-summary"] = []  # type: ignore[index]
    cases.append((invalid_entry, "entry is invalid"))

    invalid_path = copy.deepcopy(manifest)
    invalid_path["observations"]["spectral-summary"]["path"] = None  # type: ignore[index]
    cases.append((invalid_path, "path is invalid"))

    unsafe_observation = copy.deepcopy(manifest)
    unsafe_observation["observations"]["spectral-summary"]["path"] = "../x"  # type: ignore[index]
    cases.append((unsafe_observation, "path is unsafe"))

    for candidate, message in cases:
        loaded, errors = reporting._load_observations(evidence, candidate)
        assert loaded == []
        assert any(message in error for error in errors), (message, errors)

    request_path = evidence / "request.json"
    original_request = request_path.read_bytes()
    request_path.write_bytes(b"[]")
    assert reporting._load_observations(evidence, manifest)[1] == [
        "normalized request is not canonical JSON"
    ]
    request_path.write_bytes(reporting.canonical_json_bytes({"observations": {}}))
    assert (
        "observations are invalid"
        in reporting._load_observations(evidence, manifest)[1][0]
    )
    request_path.write_bytes(reporting.canonical_json_bytes({"observations": []}))
    assert (
        "do not match normalized request"
        in reporting._load_observations(evidence, manifest)[1][0]
    )
    request_path.write_bytes(original_request)


def test_observation_loader_reports_payload_binding_and_encoding_errors(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path, with_observation=True)
    manifest = _manifest(evidence)
    observation_path = evidence / "observations/spectral-summary.json"
    original = observation_path.read_bytes()

    observation_path.write_bytes(b"[]")
    assert (
        "must be a JSON object"
        in reporting._load_observations(evidence, manifest)[1][0]
    )

    observation_path.write_bytes(json.dumps(json.loads(original), indent=2).encode())
    errors = reporting._load_observations(evidence, manifest)[1]
    assert any("canonical JSON" in error for error in errors)
    assert any("manifest digest" in error for error in errors)

    payload = json.loads(original)
    payload["payload"] = {"changed": True}
    observation_path.write_bytes(reporting.canonical_json_bytes(payload))
    errors = reporting._load_observations(evidence, manifest)[1]
    assert any("does not match normalized request" in error for error in errors)
    observation_path.write_bytes(original)


def test_reference_binding_checker_covers_identity_and_record_failures(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path)
    manifest = _manifest(evidence)

    baseline = evidence / "inputs/baseline.json"
    baseline.write_bytes(b"[]")
    errors = reporting._reference_binding_errors(evidence, manifest)
    assert any("must be a JSON object" in error for error in errors)
    assert any("manifest digest" in error for error in errors)

    baseline.write_bytes(b"{")
    errors = reporting._reference_binding_errors(evidence, manifest)
    assert any("valid JSON" in error for error in errors)

    records = evidence / "records/paired-records.json"
    records.write_bytes(reporting.canonical_json_bytes({"records": []}))
    errors = reporting._reference_binding_errors(evidence, manifest)
    assert any("digest does not bind paired records" in error for error in errors)
    assert any("count does not bind paired records" in error for error in errors)

    records.write_bytes(b"{")
    assert any(
        "valid JSON" in error
        for error in reporting._reference_binding_errors(evidence, manifest)
    )
