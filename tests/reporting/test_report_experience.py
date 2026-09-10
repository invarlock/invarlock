"""Report views preserve metric methods, policy scope and recipient trust limits."""

import math
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_contracts import PAYLOADS, CapturedSnapshot
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.record_reporting import _view, render_html, render_markdown
from tests._evaluation_support import build_pack, pack_json


def evidence(kind="classification", key=None):
    baseline, candidate, policy = example_project(kind)
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    return build_pack(baseline, candidate, policy, key)


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
    value = build_pack(baseline, candidate, policy)
    view = _view(pack_json(value, "report"), value).metrics[0]
    assert "confidence interval" in view.interval.label
    assert view.interval.unit == "pp"
    assert view.baseline == view.candidate == "100%"
    assert "resampling interval" not in render_markdown(
        pack_json(value, "report"), evidence=value
    )


def test_fractional_fields_remain_scalar_resampling_with_original_unit():
    baseline, candidate, policy = example_project("extraction")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for row in candidate["records"]:
        row["output"]["currency"] = "CAD"
    value = build_pack(baseline, candidate, policy)
    view = _view(pack_json(value, "report"), value).metrics[0]
    assert view.candidate == "0.5 score"
    assert view.interval.unit == "score"
    assert "resampling interval" in view.interval.label
    assert "confidence interval" not in view.interval.label


@pytest.mark.parametrize("state", ["signed", "unsigned", "unavailable"])
def test_signing_state_never_claims_recipient_acceptance(state):
    value = evidence(key=Ed25519PrivateKey.generate() if state == "signed" else None)
    bound = None if state == "unavailable" else value
    for renderer in (render_html, render_markdown):
        result = renderer(pack_json(value, "report"), evidence=bound)
        assert "Not performed by report" in result
        expected = {
            "signed": "Signed manifest verified",
            "unsigned": "Unsigned local evidence",
            "unavailable": "Signing state unavailable",
        }[state]
        assert expected in result
        assert (
            "recipient-owned key" in result
            if state == "signed"
            else "Signed manifest verified" not in result
        )
        assert "recipient verification passed" not in result.lower()


def test_missing_count_is_not_observed_count_and_preview_stays_bounded():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for row in candidate["records"]:
        row["output"] = None
        row["error"] = "capture incomplete"
    value = build_pack(baseline, candidate, policy)
    view = _view(pack_json(value, "report"), value)
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
        result = renderer(pack_json(value, "report"), evidence=value)
        assert "40 missing paired results" in result
        assert "More evidence needed" in result
        assert "case-39" not in result


def test_improving_candidate_below_floor_is_not_explained_as_deterioration():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0].update(
        maximum_regression=0.5, maximum_interval_width=2, subject_minimum=0.9
    )
    for run, wrong in ((baseline, 10), (candidate, 5)):
        for row in run["records"][:wrong]:
            row["output"] = "wrong"
    value = build_pack(baseline, candidate, policy)
    assert pack_json(value, "report")["decision"] == "regression"
    metric = _view(pack_json(value, "report"), value).metrics[0]
    assert metric.change == "+12.5 pp"
    checks = {c.name: c for c in metric.checks}
    assert checks["Allowed change"].passed is True
    assert checks["Subject minimum"].observed == "87.5%"
    assert checks["Subject minimum"].required == ">= 90%"
    assert checks["Subject minimum"].passed is False
    assert "Subject minimum" in metric.explanation
    assert "deteriorat" not in metric.explanation.lower()


@pytest.mark.parametrize(
    "mutation", ["policy", "run", "comparison", "signature", "extra"]
)
def test_bound_evidence_rejects_contradictions(mutation):
    value = evidence(key=Ed25519PrivateKey.generate())
    comparison = pack_json(value, "report")
    files = dict(value.files)
    if mutation == "policy":
        policy = pack_json(value, "policy")
        policy["metrics"][0]["subject_minimum"] += 0.01
        files[PAYLOADS["policy"]] = canonical_json_bytes(policy)
    elif mutation == "run":
        run = pack_json(value, "subject")
        run["records"][0]["output"] = "other"
        files[PAYLOADS["subject"]] = canonical_json_bytes(run)
    elif mutation == "comparison":
        comparison["metrics"][0]["reasons"] = ["different recorded reasons"]
    elif mutation == "signature":
        files["manifest.signature.json"] = b'"not-a-signature"\n'
    else:
        files["independently_verified.json"] = b"true\n"
    value = CapturedSnapshot(files)
    for renderer in (render_html, render_markdown):
        with pytest.raises(EvaluationRecordsError):
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
    value = build_pack(baseline, candidate, policy)
    result = render_html(pack_json(value, "report"), evidence=value)
    parsed = Tags()
    parsed.feed(result)
    assert "img" not in parsed.tags and "script" not in parsed.tags
    assert not any(
        name in {"src", "href", "onerror", "onload"} for name, _ in parsed.attributes
    )
    assert "&lt;img" in result
    assert "DO_NOT_RENDER_PAYLOAD" not in result
    assert len(result.encode()) < 40_000
    markdown = render_markdown(pack_json(value, "report"), evidence=value)
    assert attack not in markdown
    assert "[link](https://evil.invalid)" not in markdown


@pytest.mark.parametrize("kind", ["judge", "classification"])
def test_extreme_finite_threshold_has_finite_svg_coordinates(kind):
    baseline, candidate, policy = example_project(kind)
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0]["maximum_regression"] = 1.7e308
    value = build_pack(baseline, candidate, policy)
    view = _view(pack_json(value, "report"), value).metrics[0]
    assert view.interval.threshold is None or math.isfinite(view.interval.threshold)
    assert all("inf" not in check.required for check in view.checks)
    result = render_html(pack_json(value, "report"), evidence=value)
    parsed = Tags()
    parsed.feed(result)
    coordinates = [
        float(v) for k, v in parsed.attributes if k in {"x1", "x2", "cx", "cy"}
    ]
    assert coordinates and all(math.isfinite(v) for v in coordinates)


def test_combined_reports_build_one_validated_view(monkeypatch):
    from invarlock import record_reporting as report

    value = evidence()
    expected_html = report.render_html(pack_json(value, "report"), evidence=value)
    expected_markdown = report.render_markdown(
        pack_json(value, "report"), evidence=value
    )
    original = report._view
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "_view", counted)
    html, markdown = report.render_reports(pack_json(value, "report"), evidence=value)
    assert calls == [1]
    assert (html, markdown) == (expected_html, expected_markdown)


def test_large_bound_configuration_is_previewed_without_source_mutation():
    baseline, candidate, policy = example_project("extraction")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    key = '<img src=x onerror="alert(1)">' + "x" * 100_000
    policy["metrics"][0]["configuration"]["fields"] = ["/" + key]
    for run in (baseline, candidate):
        for row in run["records"]:
            row["expected"] = row["output"] = {key: 1}
    value = build_pack(baseline, candidate, policy)
    before = dict(value.files)
    view = _view(pack_json(value, "report"), value)
    displayed_policy = view.details[0][1]
    assert "preview" in view.details[0][0].lower()
    displayed_metric = displayed_policy["metrics"][0]
    assert displayed_metric["configuration"]["preview_only"] is True
    assert displayed_metric["subject_minimum"] == 0.8
    assert displayed_metric["minimum_count"] == 10
    html = render_html(pack_json(value, "report"), evidence=value)
    assert len(html.encode()) < 40_000
    assert key not in html and "&lt;img" in html
    assert pack_json(value, "report")["bindings"]["policy"] in html
    assert value.files == before


def test_configuration_preview_bounds_depth_nodes_strings_and_retains_numbers():
    from invarlock.record_reporting import _configuration_preview

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


@pytest.mark.parametrize("outcome", ["regression", "missing"])
def test_multi_metric_overview_keeps_every_scope_and_adverse_check_visible(outcome):
    import re
    from html import escape

    from invarlock.report_presentation import decision_label

    baseline, candidate, policy = example_project("classification")
    if outcome == "regression":
        for row in candidate["records"][:10]:
            row["output"] = "wrong"
    else:
        candidate["records"][0]["error"] = "upstream capture missing"
        candidate["records"][0]["output"] = None
    value = build_pack(baseline, candidate, policy)
    before = dict(value.files)
    view = _view(pack_json(value, "report"), value)
    html = render_html(pack_json(value, "report"), evidence=value)

    overview = html.split('<section class="results-overview"', 1)[1].split(
        "</section>", 1
    )[0]
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", overview, re.S)[1:]
    assert len(rows) == len(view.metrics) == 4
    assert "<details" not in overview and " hidden" not in overview
    for row, metric in zip(rows, view.metrics, strict=True):
        for text in (
            metric.name,
            metric.scope,
            metric.baseline,
            metric.candidate,
            metric.change,
            metric.count,
            decision_label(metric.decision),
        ):
            assert escape(text, quote=True) in row
        for check in metric.checks:
            if check.passed is not True:
                assert escape(check.name, quote=True) in row
    assert any(m.decision != "pass" for m in view.metrics)
    assert value.files == before
    assert "Not performed by report" in html


def test_multi_metric_navigation_escapes_labels_and_authorizes_only_fixed_script():
    import re

    baseline, candidate, policy = example_project("classification")
    attack = '<img src=x onerror="alert(1)">'
    policy["metrics"][0]["name"] = attack
    value = build_pack(baseline, candidate, policy)
    html = render_html(pack_json(value, "report"), evidence=value)
    parsed = Tags()
    parsed.feed(html)
    assert parsed.tags.count("script") == 1 and "img" not in parsed.tags
    ids = [value for name, value in parsed.attributes if name == "id"]
    assert len(ids) == len(set(ids))
    links = [value for name, value in parsed.attributes if name == "href"]
    assert links and all(link.startswith("#") and link[1:] in ids for link in links)
    assert not any(name.startswith("on") for name, _ in parsed.attributes)
    assert not any(
        value in {"tab", "tablist", "tabpanel"} for _, value in parsed.attributes
    )
    navigation = re.search(r'<nav class="metric-navigation".*?</nav>', html, re.S)
    assert navigation is not None
    assert navigation.group().count('href="#metric-group-') == 2
    assert html.count('id="metric-result-') == 4
    assert "&lt;img" in navigation.group()
    assert html.index('id="metric-result-1"') < html.index('id="metric-result-3"')
    assert html.index('id="metric-result-3"') < html.index('id="metric-result-2"')
    assert html.index('id="metric-result-2"') < html.index('id="metric-result-4"')
    assert not re.search(r"<section[^>]*\bhidden(?:[\s=>])", html)

    import base64
    import hashlib

    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = next(
        value
        for name, value in parsed.attributes
        if name == "content" and "default-src" in value
    )
    assert f"script-src 'sha256-{digest}'" in csp
    assert "script-src 'unsafe-inline'" not in csp
    assert "unsafe-eval" not in csp
    assert attack not in script
    normal_baseline, normal_candidate, normal_policy = example_project("classification")
    normal = build_pack(normal_baseline, normal_candidate, normal_policy)
    normal_html = render_html(pack_json(normal, "report"), evidence=normal)
    assert re.search(r"<script>(.*?)</script>", normal_html, re.S).group(1) == script


def test_single_metric_uses_simple_detail_without_navigation():
    value = evidence()
    html = render_html(pack_json(value, "report"), evidence=value)
    assert 'class="results-overview"' not in html
    assert 'class="metric-navigation"' not in html
    assert "Decision checks" in html


def test_multiple_scopes_of_one_metric_need_no_tabs_or_script():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    value = build_pack(baseline, candidate, policy)
    html = render_html(pack_json(value, "report"), evidence=value)
    assert 'class="results-overview"' in html
    assert 'class="metric-navigation"' not in html
    assert "<script" not in html
    assert html.count('id="metric-result-') == 2
    assert "overall" in html and "exceptions" in html
