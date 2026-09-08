"""Report views preserve metric methods, policy scope and recipient trust limits."""

import math
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.pipeline.contracts import PipelineError
from invarlock.pipeline.evidence import create_evidence
from invarlock.pipeline.report import _view, render_html, render_markdown
from invarlock.pipeline.templates import example_project


def evidence(kind="classification", key=None):
    baseline, candidate, policy = example_project(kind)
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    return create_evidence(baseline, candidate, policy, key)


@pytest.mark.parametrize(
    "kind", ["exact_match", "normalized_match", "numeric_tolerance", "json_exact"]
)
def test_binary_methods_have_binary_interval_and_percent_units(kind):
    baseline, candidate, policy = example_project("classification")
    metric = policy["metrics"][0]
    metric.update(
        kind=kind,
        configuration={} if kind != "normalized_match" else metric["configuration"],
    )
    policy["metrics"] = [metric]
    policy["slices"] = []
    for run in (baseline, candidate):
        for row in run["records"]:
            row["expected"] = row["output"] = "1"
    value = create_evidence(baseline, candidate, policy)
    view = _view(value["comparison"], value).metrics[0]
    assert "confidence interval" in view.interval.label
    assert view.interval.unit == "pp"
    assert view.baseline == view.candidate == "100%"
    assert "resampling interval" not in render_markdown(
        value["comparison"], evidence=value
    )


def test_fractional_fields_remain_scalar_resampling_with_original_unit():
    baseline, candidate, policy = example_project("extraction")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for row in candidate["records"]:
        row["output"]["currency"] = "CAD"
    value = create_evidence(baseline, candidate, policy)
    view = _view(value["comparison"], value).metrics[0]
    assert view.candidate == "0.5 score"
    assert view.interval.unit == "score"
    assert "resampling interval" in view.interval.label
    assert "confidence interval" not in view.interval.label


@pytest.mark.parametrize("state", ["signed", "unsigned", "unavailable"])
def test_signing_state_never_claims_recipient_acceptance(state):
    value = evidence(key=Ed25519PrivateKey.generate() if state == "signed" else None)
    bound = None if state == "unavailable" else value
    for renderer in (render_html, render_markdown):
        result = renderer(value["comparison"], evidence=bound)
        assert "Not performed by report" in result
        expected = {
            "signed": "Signature present",
            "unsigned": "Unsigned local comparison",
            "unavailable": "Signing state unavailable",
        }[state]
        assert expected in result
        assert "signature verified" not in result.lower()
        assert "recipient verification passed" not in result.lower()


def test_missing_count_is_not_observed_count_and_preview_stays_bounded():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for row in candidate["records"]:
        row["output"] = None
        row["error"] = "capture incomplete"
    value = create_evidence(baseline, candidate, policy)
    view = _view(value["comparison"], value)
    metric = view.metrics[0]
    assert metric.count == "0"
    assert "More evidence is needed" in metric.explanation
    assert "The policy was not met" not in metric.explanation
    assert "0 metric / scope results did not meet" not in view.summary
    assert metric.baseline == metric.candidate == "Unavailable"
    assert metric.interval is None
    complete = next(c for c in metric.checks if c.name == "Complete paired results")
    assert complete.observed == "0 of 40" and complete.passed is False
    assert view.technical["metrics"][0]["missing_count"] == 40
    assert len(view.technical["metrics"][0]["missing_ids_preview"]) == 20
    assert view.technical["metrics"][0]["missing_ids_preview_is_complete"] is False
    for renderer in (render_html, render_markdown):
        result = renderer(value["comparison"], evidence=value)
        assert "40 missing paired results" in result
        assert "More evidence needed" in result
        assert "case-39" not in result


def test_improving_candidate_below_floor_is_not_explained_as_deterioration():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0].update(
        maximum_regression=0.5, maximum_interval_width=2, candidate_minimum=0.9
    )
    for run, wrong in ((baseline, 10), (candidate, 5)):
        for row in run["records"][:wrong]:
            row["output"] = "wrong"
    value = create_evidence(baseline, candidate, policy)
    assert value["comparison"]["decision"] == "regression"
    metric = _view(value["comparison"], value).metrics[0]
    assert metric.change == "+12.5 pp"
    checks = {c.name: c for c in metric.checks}
    assert checks["Allowed change"].passed is True
    assert checks["Candidate minimum"].observed == "87.5%"
    assert checks["Candidate minimum"].required == ">= 90%"
    assert checks["Candidate minimum"].passed is False
    assert "Candidate minimum" in metric.explanation
    assert "deteriorat" not in metric.explanation.lower()


@pytest.mark.parametrize("mutation", ["policy", "run", "comparison", "extra"])
def test_bound_evidence_rejects_contradictions(mutation):
    value = evidence()
    comparison = deepcopy(value["comparison"])
    if mutation == "policy":
        value["policy"]["metrics"][0]["candidate_minimum"] += 0.01
    elif mutation == "run":
        value["candidate"]["records"][0]["output"] = "other"
    elif mutation == "comparison":
        comparison["metrics"][0]["reasons"] = ["different recorded reasons"]
    else:
        value["independently_verified"] = True
    for renderer in (render_html, render_markdown):
        with pytest.raises(PipelineError):
            renderer(comparison, evidence=value)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)


def test_report_escapes_external_labels_and_omits_large_run_payloads():
    baseline, candidate, policy = example_project("classification")
    attack = '<img src=x onerror="alert(1)">[link](https://evil.invalid)'
    policy["metrics"] = policy["metrics"][:1]
    policy["metrics"][0]["name"] = attack
    policy["slices"] = []
    for run in (baseline, candidate):
        run["run_id"] = attack
        run["records"][0]["input"] = "DO_NOT_RENDER_PAYLOAD" + "x" * 100_000
    value = create_evidence(baseline, candidate, policy)
    result = render_html(value["comparison"], evidence=value)
    parsed = Tags()
    parsed.feed(result)
    assert "img" not in parsed.tags and "script" not in parsed.tags
    assert not any(
        name in {"src", "href", "onerror", "onload"} for name, _ in parsed.attributes
    )
    assert "&lt;img" in result
    assert "DO_NOT_RENDER_PAYLOAD" not in result
    assert len(result.encode()) < 40_000
    markdown = render_markdown(value["comparison"], evidence=value)
    assert attack not in markdown
    assert "[link](https://evil.invalid)" not in markdown


@pytest.mark.parametrize("kind", ["judge", "classification"])
def test_extreme_finite_threshold_has_finite_svg_coordinates(kind):
    baseline, candidate, policy = example_project(kind)
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0]["maximum_regression"] = 1.7e308
    value = create_evidence(baseline, candidate, policy)
    view = _view(value["comparison"], value).metrics[0]
    assert view.interval.threshold is None or math.isfinite(view.interval.threshold)
    assert all("inf" not in check.required for check in view.checks)
    result = render_html(value["comparison"], evidence=value)
    parsed = Tags()
    parsed.feed(result)
    coordinates = [
        float(v) for k, v in parsed.attributes if k in {"x1", "x2", "cx", "cy"}
    ]
    assert coordinates and all(math.isfinite(v) for v in coordinates)


def test_combined_reports_build_one_validated_view(monkeypatch):
    from invarlock.pipeline import report

    value = evidence()
    expected_html = report.render_html(value["comparison"], evidence=value)
    expected_markdown = report.render_markdown(value["comparison"], evidence=value)
    original = report._view
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "_view", counted)
    html, markdown = report.render_reports(value["comparison"], evidence=value)
    assert calls == [1]
    assert (html, markdown) == (expected_html, expected_markdown)


def test_large_bound_configuration_is_previewed_without_source_mutation():
    from invarlock.evidence_pack_contract import canonical_json_bytes

    baseline, candidate, policy = example_project("extraction")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    key = '<img src=x onerror="alert(1)">' + "x" * 100_000
    policy["metrics"][0]["configuration"]["fields"] = ["/" + key]
    for run in (baseline, candidate):
        for row in run["records"]:
            row["expected"] = row["output"] = {key: 1}
    value = create_evidence(baseline, candidate, policy)
    before = canonical_json_bytes(value)
    view = _view(value["comparison"], value)
    displayed_policy = view.details[0][1]
    assert "preview" in view.details[0][0].lower()
    displayed_metric = displayed_policy["metrics"][0]
    assert displayed_metric["configuration"]["preview_only"] is True
    assert displayed_metric["candidate_minimum"] == 0.8
    assert displayed_metric["minimum_count"] == 10
    html = render_html(value["comparison"], evidence=value)
    assert len(html.encode()) < 40_000
    assert key not in html and "&lt;img" in html
    assert value["comparison"]["bindings"]["policy"] in html
    assert canonical_json_bytes(value) == before


def test_configuration_preview_bounds_depth_nodes_strings_and_retains_numbers():
    from invarlock.pipeline.report import _configuration_preview

    number = 0.12345678901234566
    value = {
        "number": number,
        "integer": 12345678901234567890,
        "nested": {"x": {"x": {"x": {"x": {"x": {"x": "DEEP_TAIL"}}}}}},
        "wide": list(range(1000)),
        "long": "x" * 10000,
    }
    before = deepcopy(value)
    preview = _configuration_preview(value)
    assert preview["preview_only"] is True
    assert str(number) in preview["text"]
    assert "12345678901234567890" in preview["text"]
    assert "DEEP_TAIL" not in preview["text"]
    assert "999" not in preview["text"]
    assert "x" * 1000 not in preview["text"]
    assert len(preview["text"]) < 70_000
    assert value == before
