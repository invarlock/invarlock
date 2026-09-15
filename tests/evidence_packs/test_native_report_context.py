"""Native report context stays within authenticated evidence and declared facts."""

import json
from copy import deepcopy

import pytest

from invarlock.evidence_pack_contract import (
    build_comparison_report,
    canonical_json_bytes,
)
from invarlock.evidence_reporting import (
    EvidenceReportError,
    _native_report_context,
    _report_view,
    render_evidence,
)
from tests.cli.test_import_journey import (
    _materialize_request,
    _text_scorer_registry_and_binding,
)
from tests.evidence_packs.test_evidence_pack import (
    _rebind_and_resign_pack,
    _signing_key,
)
from tests.evidence_packs.test_evidence_reporting import _evidence, _report


def _metric_report(tmp_path, metric):
    if metric == "exact_match":
        return _report()
    if metric == "extension":
        registry, binding = _text_scorer_registry_and_binding()
        material = _materialize_request(
            tmp_path, scorer_binding=binding, scorer_registry=registry
        )
        pairs = json.loads(material["records"].read_bytes())
        policy = json.loads(material["policy"].read_bytes())
    else:
        pairs = {
            "format": "invarlock/paired-records-v1",
            "metric": "normalized_nll_per_utf8_byte",
            "schedule_sha256": "0" * 64,
            "records": [
                {
                    "record_id": str(i),
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.1},
                }
                for i in range(2)
            ],
            "derived_measurements": {
                "perplexity_ratio": {
                    "status": "unavailable",
                    "basis": "authenticated_target_likelihood",
                    "method": "target_token_weighted_perplexity_ratio_v1",
                    "reason": "target_token_counts_unavailable",
                }
            },
        }
        policy = {
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.25}}
            }
        }
    return build_comparison_report(
        comparison_id="model-comparison",
        paired_records=pairs,
        policy=policy,
        policy_digest="sha256:" + "a" * 64,
    )


def _request(mode):
    return {
        "execution": {"mode": mode},
        "comparison": {
            "task": "text_causal",
            "baseline": {
                "artifact": {"model_id": "model-baseline"},
                "runtime": {"provider": "hf_transformers"},
            },
            "subject": {
                "artifact": {"model_id": "model-candidate"},
                "runtime": {"provider": "gguf"},
            },
            "dataset": {
                "name": "questions",
                "split": "validation",
                "source_sha256": "d" * 64,
                "source_format": "jsonl",
                "selected_record_count": 2,
                "limit": None,
            }
            if mode == "run"
            else "schedule/runtime-behavioral-schedule.json",
        },
    }


@pytest.mark.parametrize(
    "metric,direction,neutral",
    [("exact_match", "minimum", 0.0), ("normalized_nll", "maximum", 1.0)],
)
def test_native_interval_semantics_preserve_numeric_summary(
    tmp_path, metric, direction, neutral
):
    report = _metric_report(tmp_path, metric)
    view = _report_view(report, evidence_signer="fixture", observations=[])
    result = view.metrics[0]
    assert result.interval.threshold_direction == direction
    assert result.interval.neutral == neutral
    assert result.interval.threshold == report["comparison"][direction]
    assert result.decision == view.decision == report["verdict"]
    if metric == "exact_match":
        assert (result.baseline, result.candidate, result.change) == (
            "50%",
            "100%",
            "+50 pp",
        )
        assert (
            "This policy does not specify an absolute accuracy floor." in result.notes
        )
    else:
        assert (result.baseline, result.candidate, result.change) == (
            "1 nats / byte",
            "1.1 nats / byte",
            "1.1 ratio",
        )


def _context_pack(tmp_path, metric="exact_match", request=None):
    report = _metric_report(tmp_path, metric)
    pack, _ = _evidence(tmp_path, report_payload=report)
    if request is not None:
        (pack / "request.json").write_bytes(canonical_json_bytes(request))
        key, signer = _signing_key(tmp_path)
        manifest = json.loads((pack / "manifest.json").read_bytes())
        manifest["signing_key_fingerprint"] = signer
        (pack / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        _rebind_and_resign_pack(pack, key)
    return pack, report


@pytest.mark.parametrize("mode", ["run", "import"])
@pytest.mark.parametrize("metric", ["exact_match", "normalized_nll", "extension"])
def test_native_context_survives_metric_and_execution_modes(tmp_path, mode, metric):
    pack, report = _context_pack(tmp_path, metric, _request(mode))
    before = {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    original_report = deepcopy(report)
    output = tmp_path / "report.html"
    result = render_evidence(pack, html_path=output)
    for rendered in (result.text, output.read_text()):
        for expected in (
            "model-baseline",
            "model-candidate",
            "hf_transformers",
            "gguf",
            "text_causal",
            "Schedule digest",
        ):
            assert expected in rendered
        assert (
            "Runtime execution" if mode == "run" else "Imported runtime evidence"
        ) in rendered
        assert "different authenticated artifact digests" in rendered
        assert "Not performed by report" in rendered
        if mode == "run":
            assert "questions" in rendered and "validation" in rendered
            assert "Selected records" in rendered
        else:
            assert "fixture://dataset" in rendered
            assert "Dataset split" in rendered and "Unavailable in evidence" in rendered
    assert before == {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    assert report == original_report


def test_legacy_context_is_unavailable_without_inferred_model_or_execution(tmp_path):
    pack, _ = _context_pack(tmp_path)
    output = tmp_path / "report.html"
    rendered = render_evidence(pack, html_path=output).text
    assert "Unavailable in evidence" in rendered
    assert "Runtime execution" not in rendered
    assert "Imported runtime evidence" not in rendered
    assert "fixture://dataset" in rendered
    html = output.read_text()
    assert 'class="comparison-table"' in html
    assert (
        '<th scope="row">Artifact<span class="different-label">Differs</span></th>'
        in html
    )
    assert "<dt>Baseline artifact</dt>" not in html
    assert "<dt>Candidate artifact</dt>" not in html


def test_context_tampering_is_rejected_before_html_publication(tmp_path):
    pack, _ = _context_pack(tmp_path, request=_request("run"))
    (pack / "request.json").write_bytes(canonical_json_bytes(_request("import")))
    output = tmp_path / "report.html"
    with pytest.raises(EvidenceReportError):
        render_evidence(pack, html_path=output)
    assert not output.exists()


@pytest.mark.parametrize("variant", ["absent", "malformed", "long", "limited"])
def test_context_projection_handles_missing_and_bounded_descriptions(tmp_path, variant):
    pack, report = _context_pack(tmp_path)
    request = _request("run")
    if variant == "absent":
        (pack / "request.json").unlink()
    else:
        if variant == "malformed":
            request = {"execution": {"mode": []}, "comparison": []}
        elif variant == "long":
            request["comparison"]["task"] = "x" * 300
            request["comparison"]["dataset"]["name"] = " "
        else:
            request["comparison"]["dataset"]["limit"] = 2
        (pack / "request.json").write_bytes(canonical_json_bytes(request))
    identities = {
        side: {"digest": "sha256:" + "a" * 64} for side in ("baseline", "subject")
    }
    context, changes = _native_report_context(pack, report, identities)
    facts = dict(context)
    assert "same authenticated artifact digest" in changes[0]
    if variant == "long":
        assert facts["Task"] == "x" * 256 + "… (preview)"
        assert facts["Dataset"] == "Unavailable in evidence"
    elif variant == "limited":
        assert facts["Selection limit"] == "2"
    else:
        assert facts["Workflow"] == facts["Task"] == "Unavailable in evidence"
