"""Shared, non-authoritative presentation of explicit comparison facts.

Builders retain each evidence family's own validation and trust rules. These
views never authorize a signer, replay a score, or change a stored decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from html import escape as html_escape
from typing import Any

_DISPLAY_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)
_TERMINAL_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)


def visible_controls(value: str) -> str:
    """Render control and bidirectional formatting characters visibly."""
    return _DISPLAY_CONTROL_RE.sub(
        lambda match: f"\\u{ord(match.group()):04x}",
        value,
    )


def terminal_text(value: str) -> str:
    """Make one dynamic terminal value unable to create or reorder lines."""
    return _TERMINAL_CONTROL_RE.sub(
        lambda match: f"\\u{ord(match.group()):04x}",
        value,
    )


def escape(value: str, quote: bool = True) -> str:
    """Show terminal control bytes literally before escaping report markup."""
    visible = visible_controls(value)
    return html_escape(visible, quote=quote)


def xml_text(value: str) -> str:
    """Render controls visibly and replace characters forbidden by XML 1.0."""
    return "".join(
        character
        if character in "\t\n\r"
        or "\u0020" <= character <= "\ud7ff"
        or "\ue000" <= character <= "\ufffd"
        or "\U00010000" <= character <= "\U0010ffff"
        else (
            f"\\u{ord(character):04x}"
            if ord(character) <= 0xFFFF
            else f"\\U{ord(character):08x}"
        )
        for character in visible_controls(value)
    )


# Paper rendition of docs/assets/invarlock-mark.svg. The bracket/signal geometry
# stays recognizable without an app tile, translucent overprint or tiny dot.
# One ink and a live-text wordmark also survive background-free printing.
_BRAND_MARK = """<svg class="mark" xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 512 512" aria-hidden="true" focusable="false">
<g transform="translate(256 256) scale(1.55) translate(-256 -256)" fill="none" stroke="currentColor" stroke-width="22" stroke-linecap="round" stroke-linejoin="round">
<path d="M174 150 H126 V362 H174"/>
<path d="M338 150 H386 V362 H338"/>
<path d="M156 292 C 198 212, 230 212, 256 252 S 314 332, 356 242"/>
</g></svg>"""


# Fixed trusted code only; report labels and evidence are never interpolated here.
_METRIC_TABS_SCRIPT = """(() => {
  const main = document.querySelector('main');
  const nav = main && main.querySelector('.metric-navigation');
  const toggle = main && main.querySelector('.metric-display-toggle');
  if (!nav || !toggle) return;
  const links = Array.from(nav.querySelectorAll('a[href^="#metric-group-"]'));
  const headings = links.map(link => document.getElementById(link.hash.slice(1)));
  const panels = headings.map(heading => heading && heading.closest('.metric-group'));
  if (links.length < 2 || panels.some(panel => !panel || !main.contains(panel))) return;
  let active = 0;
  let showingAll = false;
  const indexFor = target => panels.findIndex(panel => panel.contains(target));
  const hashTarget = () => {
    const id = window.location.hash.slice(1);
    return /^metric-(group|result)-[0-9]+$/.test(id) ? document.getElementById(id) : null;
  };
  const update = () => {
    if (showingAll) nav.removeAttribute('role');
    else nav.setAttribute('role', 'tablist');
    links.forEach((link, index) => {
      const panel = panels[index];
      panel.hidden = !showingAll && index !== active;
      if (showingAll) {
        ['role', 'aria-selected', 'aria-controls', 'tabindex'].forEach(attr => link.removeAttribute(attr));
        panel.removeAttribute('role');
        panel.setAttribute('aria-labelledby', headings[index].id);
      } else {
        link.id = 'metric-tab-' + (index + 1);
        link.setAttribute('role', 'tab');
        link.setAttribute('aria-selected', String(index === active));
        link.setAttribute('aria-controls', panel.id);
        link.tabIndex = index === active ? 0 : -1;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', link.id);
      }
    });
    toggle.textContent = showingAll ? 'Use metric tabs' : 'Show all metrics';
  };
  const select = (index, focusTab) => {
    active = index;
    update();
    if (focusTab) links[index].focus();
  };
  const remember = index => {
    // Fragment replacement remains local, including file:// reports.
    try { window.history.replaceState(null, '', links[index].hash); } catch (_) {}
  };
  links.forEach((link, index) => {
    link.addEventListener('click', event => {
      if (showingAll || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      select(index, true);
      remember(index);
    });
    link.addEventListener('keydown', event => {
      if (showingAll || event.ctrlKey || event.metaKey || event.altKey) return;
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % links.length;
      else if (event.key === 'ArrowLeft') next = (index + links.length - 1) % links.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = links.length - 1;
      else return;
      event.preventDefault();
      select(next, true);
      remember(next);
    });
  });
  const followHash = () => {
    const target = hashTarget();
    const index = target ? indexFor(target) : -1;
    if (index < 0) return;
    select(index, false);
    target.focus();
    target.scrollIntoView({block: 'start'});
  };
  main.querySelectorAll('.results-overview a[href^="#metric-result-"], .findings a[href^="#metric-result-"]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const target = document.getElementById(link.hash.slice(1));
      const index = target ? indexFor(target) : -1;
      if (index >= 0) {
        select(index, false);
        target.focus();
      }
    });
  });
  toggle.addEventListener('click', () => {
    showingAll = !showingAll;
    update();
  });
  window.addEventListener('hashchange', followHash);
  const initial = hashTarget();
  const initialIndex = initial ? indexFor(initial) : -1;
  const adverse = panels.findIndex(panel => panel.querySelector('.metric.fail, .metric.insufficient'));
  active = initialIndex >= 0 ? initialIndex : Math.max(0, adverse);
  update();
  toggle.hidden = false;
  if (initialIndex >= 0) followHash();
})();"""
_METRIC_TABS_HASH = base64.b64encode(
    hashlib.sha256(_METRIC_TABS_SCRIPT.encode("utf-8")).digest()
).decode("ascii")


@dataclass(frozen=True)
class CheckView:
    name: str
    observed: str
    required: str
    passed: bool | None
    explanation: str = ""


@dataclass(frozen=True)
class IntervalView:
    lower: float
    upper: float
    estimate: float
    threshold: float | None
    label: str
    unit: str
    threshold_direction: str | None = None
    neutral: float | None = None
    method: str = ""
    basis: tuple[str, ...] = ()


def display_label(value: str) -> str:
    """Present metric/scope identifiers only; leave authored text unchanged."""
    prefix = ""
    for component in ("Deterministic · ", "Judge · "):
        if value.startswith(component):
            prefix, value = component, value[len(component) :]
            break
    acronyms = {
        "qa": "QA",
        "nll": "NLL",
        "utf8": "UTF-8",
        "utf-8": "UTF-8",
        "f1": "F1",
        "api": "API",
        "http": "HTTP",
    }
    if value.lower() in acronyms:
        return prefix + acronyms[value.lower()]
    tokens = re.split(r"[_-]", value)
    if len(tokens) < 2 or any(
        re.fullmatch(r"[A-Za-z0-9]+", token) is None for token in tokens
    ):
        return prefix + value
    words = []
    index = 0
    while index < len(tokens):
        word = tokens[index]
        if word.lower() == "utf" and tokens[index + 1 : index + 2] == ["8"]:
            word = "utf8"
            index += 1
        words.append(acronyms.get(word.lower(), word))
        index += 1
    words[0] = words[0][0].upper() + words[0][1:]
    return prefix + " ".join(words)


@dataclass(frozen=True)
class MetricView:
    name: str
    scope: str
    decision: str
    baseline: str
    candidate: str
    change: str
    count: str
    explanation: str
    checks: tuple[CheckView, ...] = ()
    interval: IntervalView | None = None
    notes: tuple[str, ...] = ()
    baseline_detail: str = ""
    candidate_detail: str = ""
    count_detail: str = ""
    count_label: str = "Observed pairs"
    evidence_source: str = ""

    @property
    def display_name(self) -> str:
        return display_label(self.name)

    @property
    def display_scope(self) -> str:
        return display_label(self.scope)


@dataclass(frozen=True)
class AuthoredReview:
    """Separately attributed commentary; never serialized into signed evidence."""

    label: str
    byline: str
    context: tuple[tuple[str, str], ...]
    recommendation: str
    decision: str
    basis: tuple[tuple[str, str], ...]
    next_action: str


@dataclass(frozen=True)
class ReportView:
    title: str
    family: str
    decision: str
    summary: str
    metrics: tuple[MetricView, ...]
    assurance: tuple[tuple[str, str], ...]
    identity: tuple[tuple[str, str], ...] = ()
    subjects: tuple[tuple[str, str], ...] = ()
    next_steps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    details: tuple[tuple[str, Any], ...] = ()
    technical: dict[str, Any] = field(default_factory=dict)
    context: tuple[tuple[str, str], ...] = ()
    changes: tuple[str, ...] = ()
    review: AuthoredReview | None = None


def _metric_identifier_details(view: ReportView) -> tuple[tuple[str, Any], ...]:
    identifiers = [
        {
            "result": index,
            "metric": metric.name,
            "scope": metric.scope,
            "display_metric": metric.display_name,
            "display_scope": metric.display_scope,
        }
        for index, metric in enumerate(view.metrics, 1)
        if metric.display_name != metric.name or metric.display_scope != metric.scope
    ]
    return (
        (("Recorded metric and scope identifiers", identifiers),) if identifiers else ()
    )


def decision_label(value: str) -> str:
    return {
        "pass": "Policy satisfied",
        "fail": "Policy not met",
        "regression": "Policy not met",
        "insufficient_evidence": "More evidence needed",
    }.get(value, "Decision unavailable")


def number(value: float, *, signed: bool = False) -> str:
    """Round display values only; exact values stay in the source artifact."""
    return format(value, "+.4g" if signed and value != 0 else ".4g")


def _tone(value: str) -> str:
    return (
        "pass"
        if value == "pass"
        else "insufficient"
        if value == "insufficient_evidence"
        else "fail"
        if value in {"fail", "regression"}
        else "unavailable"
    )


def _status(check: CheckView) -> str:
    return (
        "Passed"
        if check.passed is True
        else "Not met"
        if check.passed is False
        else "Unavailable"
    )


def _interval_label(view: IntervalView, *, compact: bool = False) -> str:
    """Name the interval without repeating its measured bounds."""
    label = view.label
    if compact:
        label = label.replace("confidence interval", "CI")
    elif view.method:
        label += f" ({view.method})"
    return label


def _interval_summary(view: IntervalView, *, compact: bool = False) -> str:
    """Format interval values for the change tile and accessible description."""
    label = _interval_label(view, compact=compact)
    lower = number(view.lower, signed=view.neutral == 0)
    upper = number(view.upper, signed=view.neutral == 0)
    return f"{label}: {lower} to {upper} {view.unit}".strip()


def _interval_reading(view: IntervalView) -> tuple[tuple[str, str], ...]:
    """Explain geometry and endpoint rules without recalculating a verdict."""
    change = "The marker shows the observed change between subject and baseline."
    if view.unit == "pp":
        change += " A change in percentage points (pp) is the subject percentage minus the baseline percentage."
    elif view.unit == "ratio":
        change += " A ratio divides the subject value by the baseline value; 1 means no change."
    reading = [
        ("Change", change),
        (
            "Interval",
            "The bar shows uncertainty around that change. Its left endpoint is the lower bound; its right endpoint is the upper bound. The interval level does not tell you how often model runs would pass or fail the policy.",
        ),
    ]
    if view.threshold is not None and view.threshold_direction in {
        "minimum",
        "maximum",
    }:
        reading.append(
            (
                "Policy rule",
                "The paired lower bound (left endpoint) must be at least the policy minimum. The whole interval must lie on the allowed side of the threshold for this bound check to pass."
                if view.threshold_direction == "minimum"
                else "The upper bound (right endpoint) must be at most the policy maximum. The whole interval must lie on the allowed side of the threshold for this bound check to pass.",
            )
        )
    if view.neutral is not None:
        reading.append(
            (
                "No change",
                "The dashed line marks no change. An interval crossing it includes changes in either direction. A worse observed score describes this run; policy rejection alone does not prove worse performance beyond these cases.",
            )
        )
    return tuple(reading)


def _interval_reading_html(view: IntervalView, *, heading_level: int = 4) -> str:
    result = f'<div class="interval-reading"><h{heading_level}>How to read this comparison</h{heading_level}><dl>'
    result += "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(text)}</dd>"
        for label, text in _interval_reading(view)
    )
    result += "</dl></div>"
    if view.basis:
        result += (
            '<details class="interval-basis"><summary>How this interval was calculated</summary><div class="detail-content">'
            + "".join(f"<p>{escape(text)}</p>" for text in view.basis)
            + "</div></details>"
        )
    return result


def _interval(view: IntervalView) -> str:
    values = [view.lower, view.upper, view.estimate]
    if view.threshold is not None:
        values.append(view.threshold)
    if view.neutral is not None:
        values.append(view.neutral)
    # Normalize before subtraction so extreme finite bounds remain drawable.
    scale = max(1.0, *(abs(value) for value in values))
    first, last = min(values) / scale, max(values) / scale
    padding = (last - first) * 0.12 or max(abs(first) * 0.1, 0.01)
    low, high = first - padding, last + padding

    def project(normalized: float) -> float:
        return 30 + (normalized - low) / (high - low) * 580

    def x(value: float) -> float:
        return project(value / scale)

    ticks = [first] if first == last else [first, first / 2 + last / 2, last]
    desired_step = (last - first) * (scale / 4)
    if desired_step > 0:
        magnitude = 10.0 ** math.floor(math.log10(desired_step))
        if magnitude > 0:
            factor = desired_step / magnitude
            step = (
                next(value for value in (1, 2, 5, 10) if value >= factor)
                * magnitude
                / scale
            )
            first_tick, last_tick = math.floor(low / step), math.ceil(high / step)
            axis_low, axis_high = first_tick * step, last_tick * step
            if (
                math.isfinite(axis_low * scale)
                and math.isfinite(axis_high * scale)
                and axis_low < axis_high
                and axis_low <= first <= last <= axis_high
            ):
                low, high = axis_low, axis_high
                ticks = [index * step for index in range(first_tick, last_tick + 1)]
    description = f"{_interval_summary(view)}. Estimate {number(view.estimate, signed=view.neutral == 0)} {view.unit}."
    graphics = []
    legend = ""
    allowed_legend = ""
    neutral_legend = ""
    if view.threshold is not None:
        description += f" Policy threshold {number(view.threshold)} {view.unit}."
        position = x(view.threshold)
        direction = view.threshold_direction
        if direction in {"minimum", "maximum"}:
            left, right = (
                (position, 610.0) if direction == "minimum" else (30.0, position)
            )
            graphics.append(
                f'<rect class="allowed" x="{left:.2f}" y="18" width="{right - left:.2f}" height="42"/>'
            )
            allowed_legend = '<span class="chart-key"><i class="allowed-key" aria-hidden="true"></i>Allowed change region</span>'
            description += f" Shading marks values at or {'above' if direction == 'minimum' else 'below'} the threshold."
        legend = "Policy threshold"
        graphics.append(
            f'<line class="threshold" x1="{position:.2f}" x2="{position:.2f}" y1="10" y2="66"/>'
        )
    if view.neutral is not None:
        position = x(view.neutral)
        graphics.append(
            f'<line class="neutral" x1="{position:.2f}" x2="{position:.2f}" y1="18" y2="60"/>'
        )
        neutral_legend = "No change"
        description += f" No change: {number(view.neutral)} {view.unit}."
    labels = []
    for tick in ticks:
        value = tick * scale
        position = project(tick)
        graphics.append(
            f'<line class="tick" x1="{position:.2f}" x2="{position:.2f}" y1="60" y2="67"/>'
        )
        labels.append(
            f'<span style="left:{position / 640 * 100:.2f}%">{escape(number(value))}</span>'
        )

    def annotation(value: float, text: str, kind: str) -> str:
        percent = x(value) / 640 * 100
        anchor = "start" if percent < 22 else "end" if percent > 78 else "middle"
        return (
            f'<span class="chart-label {kind} {anchor}" style="left:{percent:.2f}%">'
            + escape(text)
            + "</span>"
        )

    annotations = []
    if view.threshold is not None:
        operator = {"minimum": "≥ ", "maximum": "≤ "}.get(
            view.threshold_direction or "", ""
        )
        annotations.append(
            annotation(
                view.threshold,
                f"Limit {operator}{number(view.threshold)} {view.unit}".strip(),
                "limit-label",
            )
        )
    bounds = []
    if x(view.upper) - x(view.lower) < 180:
        midpoint = view.lower / 2 + view.upper / 2
        bounds.append(
            annotation(
                midpoint, f"{number(view.lower)} to {number(view.upper)}", "bound-label"
            )
        )
    else:
        bounds.extend(
            (
                annotation(view.lower, number(view.lower), "bound-label"),
                annotation(view.upper, number(view.upper), "bound-label"),
            )
        )
    return (
        '<figure class="interval"><div class="chart-annotations" aria-hidden="true">'
        + "".join(annotations)
        + '</div><div class="interval-plot"><svg viewBox="0 0 640 72" role="img" aria-label="'
        + escape(description, quote=True)
        + '">'
        + "".join(graphics)
        + '<line class="axis" x1="30" x2="610" y1="60" y2="60"/>'
        + f'<line class="range" x1="{x(view.lower):.2f}" x2="{x(view.upper):.2f}" y1="38" y2="38"/>'
        + f'<circle class="estimate" cx="{x(view.estimate):.2f}" cy="38" r="6"/>'
        + '</svg><div class="interval-bounds" aria-hidden="true">'
        + "".join(bounds)
        + '</div></div><div class="axis-labels" aria-hidden="true">'
        + "".join(labels)
        + "</div>"
        + '<figcaption><span class="chart-key"><i class="estimate-key" aria-hidden="true"></i>Estimate'
        + '</span><span class="chart-key"><i class="interval-key" aria-hidden="true"></i>'
        + escape(_interval_label(view))
        + "</span>"
        + (
            ' <span class="chart-key"><i class="threshold-key" aria-hidden="true"></i>'
            + escape(legend.strip())
            + "</span>"
            if legend
            else ""
        )
        + (
            '<span class="chart-key"><i class="neutral-key" aria-hidden="true"></i>'
            + escape(neutral_legend)
            + "</span>"
            if neutral_legend
            else ""
        )
        + allowed_legend
        + "</figcaption></figure>"
    )


_CSS = """
.allowed-key{width:16px;height:10px;background:var(--allowed);border:1px solid var(--neutral)}
:root{color-scheme:light;--ink:#20251f;--muted:#62695e;--line:#dce0d6;--paper:#fff;--canvas:#f2f3ee;--teal:#31664f;--red:#9b3542;--amber:#826219;--range:#315b91;--allowed:#edf4ed;--neutral:#7b8779}
*{box-sizing:border-box}
body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:#145d7b}main{max-width:1120px;margin:32px auto;background:var(--paper);padding:36px 48px 48px;border-top:5px solid var(--ink);box-shadow:0 8px 40px #20251f08}
h1,h2,h3,h4,h5,p{margin-top:0}h1,h2,h3,h4,h5,p,li,dd{overflow-wrap:anywhere}h1{font-size:36px;line-height:1.15;letter-spacing:-.03em;margin-bottom:14px}h2{font-size:18px;line-height:1.3;letter-spacing:-.02em;margin-bottom:0}h3,h4{font-size:20px;line-height:1.3;margin-bottom:10px}
.brand{display:flex;align-items:center;gap:10px;font-weight:750}.mark{width:34px;height:34px;flex-shrink:0}.family{min-width:0;overflow-wrap:anywhere;margin-left:auto;color:var(--muted);font-size:13px;text-align:right}
.hero{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,1fr);gap:30px;margin:24px 0 12px;padding:26px;background:#f4f6f1;border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:4px}
.hero .verdict h2::before{content:"";display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--teal);margin-right:12px;vertical-align:middle}.hero.fail .verdict h2::before{background:var(--red)}.hero.insufficient .verdict h2::before{background:var(--amber)}
.verdict{min-width:0}.decision-summary{max-width:65ch;margin-bottom:0;overflow-wrap:anywhere;font-size:16px;color:var(--muted)}.decision-summary strong{color:var(--ink);font-weight:650}.hero .eyebrow{margin-bottom:10px}
.assurance{min-width:0}.assurance h2{font-size:13px;margin:0 0 10px;color:var(--muted)}.assurance dl{margin:0}.assurance dl>div{display:grid;grid-template-columns:minmax(90px,1fr) minmax(0,1.8fr);gap:12px;padding:8px 0;border-bottom:1px dashed var(--line);font-size:12px}.assurance dl>div:last-child{border:0}.assurance dt{color:var(--muted)}.assurance dd{margin:0;overflow-wrap:anywhere}.next-steps{margin-top:22px}
.eyebrow{margin-bottom:10px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:750;color:var(--muted)}
.summary-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.pill{font-size:13px;font-weight:650;padding:4px 10px;border-radius:6px;background:#eef3f6;color:var(--muted)}
.badge{display:inline-block;font-size:13px;font-weight:700;border:1px solid currentColor;border-radius:5px;padding:2px 8px;white-space:nowrap}
.pass .badge,.badge.pass{color:var(--teal);background:#eef8f4}.fail .badge,.badge.fail{color:var(--red);background:#fff1f2}.insufficient .badge,.badge.insufficient{color:var(--amber);background:#fff8e6}
.panel,.metric{background:var(--paper);border:1px solid var(--line);border-radius:4px;padding:24px}.panel{margin:20px 0}.panel h2{font-size:16px;margin-bottom:12px}.metric{margin-bottom:24px}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:24px 0 12px}.section-heading p{font-size:13px;color:var(--muted);margin:0}
.subjects{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0}.subjects div{min-width:0}.subjects dt{font-size:13px;font-weight:650;color:var(--muted)}.subjects dd{margin:3px 0 0;overflow-wrap:anywhere}
.comparison-table{table-layout:fixed;margin-top:0}.comparison-table th:first-child{width:23%}.comparison-table td{width:38.5%;overflow-wrap:anywhere}.comparison-table .different{background:#f3f5f0}.side-label{display:none}.different-label{display:block;font-size:12px;font-weight:400;color:#315e76}
.shared-context{border:0;border-top:1px solid var(--line);border-radius:0;margin:12px 0 0;background:transparent}.shared-context summary{padding:12px 0}.shared-context .detail-content{padding:0}
.context-list{display:grid;grid-template-columns:minmax(120px,1fr) 3fr;gap:8px 18px;margin:0}.context-list dt{font-weight:650;font-size:13px}.context-list dd{margin:0;overflow-wrap:anywhere;font-size:14px}
.change-notes{font-size:13px;color:var(--muted);margin:14px 0 0;padding-left:20px}.change-notes li+li{margin-top:6px}
.metric-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.metric-heading>div{min-width:0}.metric h3,.metric h4{overflow-wrap:anywhere}.metric-heading>.badge{flex-shrink:0}.scope{font-size:13px;color:var(--muted);margin:4px 0}.metric-explanation{margin:10px 0 18px}
.values{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:8px;background:var(--paper);overflow:hidden;margin:0}
.value{padding:12px;border-right:1px solid var(--line);min-width:0}.value:last-child{border:0}.value dt{font-size:13px;color:var(--muted)}.value dd{margin:4px 0 0;font-size:26px;font-weight:650;line-height:1.2;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.value small{display:block;margin-top:5px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);letter-spacing:0}.interval{margin:22px 0}.interval svg{display:block;width:100%;height:auto}.interval figcaption{overflow-wrap:anywhere;font-size:13px;color:var(--muted);margin-top:8px}.axis{stroke:var(--line);stroke-width:2}.range{stroke:var(--range);stroke-width:7;stroke-linecap:round}.estimate{fill:var(--paper);stroke:var(--range);stroke-width:3}.threshold{stroke:var(--red);stroke-width:2}.neutral{stroke:var(--neutral);stroke-width:1.5;stroke-dasharray:3 4}.allowed{fill:var(--allowed)}.tick{stroke:var(--neutral);stroke-width:1.5}.legend{display:block;margin-top:4px}.chart-annotations{height:24px;position:relative;font-size:12px;font-variant-numeric:tabular-nums}.chart-label{position:absolute;max-width:72%;overflow-wrap:anywhere}.chart-label.middle{transform:translateX(-50%)}.chart-label.end{transform:translateX(-100%)}.limit-label{top:0;color:var(--red)}.interval-plot{position:relative}.interval-bounds{position:absolute;inset:0;pointer-events:none;font-size:12px;font-variant-numeric:tabular-nums}.bound-label{bottom:calc(47.222222% + 10px);line-height:1.2;font-weight:650}.chart-key{display:inline-block;margin-right:18px}.chart-key i,.legend i{display:inline-block;margin-right:7px;vertical-align:middle}.estimate-key{width:10px;height:10px;border:2px solid var(--range);border-radius:50%}.interval-key{width:16px;border-top:3px solid var(--range)}.neutral-key{width:16px;border-top:2px dashed var(--neutral)}.threshold-key{height:12px;border-left:2px solid var(--red)}.axis-labels{position:relative;height:22px;font-size:13px;font-variant-numeric:tabular-nums;color:var(--muted)}.axis-labels span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin-top:16px}caption{text-align:left;font-weight:650;padding:0 0 8px}th{text-align:left;color:var(--muted);font-size:13px;font-weight:650}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}th:first-child,td:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
.interval-reading{margin:18px 0;font-size:14px}.interval-reading h4,.interval-reading h5{font-size:14px;margin:0 0 10px}.interval-reading dl{display:grid;grid-template-columns:100px minmax(0,1fr);gap:10px 16px;margin:0}.interval-reading dt{font-weight:650}.interval-reading dd{margin:0;color:var(--muted)}.interval-basis{margin:16px 0}.interval-basis p{font-size:14px;line-height:1.6}@media(max-width:600px){.interval-reading dl{grid-template-columns:1fr;gap:5px}.interval-reading dd+dt{margin-top:8px}}
.check-label{display:none}.context-heading{font-size:14px;margin:14px 0 8px}.checks-table td{white-space:normal;overflow-wrap:anywhere}.checks-table td:last-child{white-space:nowrap;overflow-wrap:normal}.checks-table th[scope="row"]{width:48%}.check-detail{display:block;font-size:13px;font-weight:400;margin-top:4px;color:var(--muted)}.check-fail{color:var(--red);font-weight:650}.check-pass{color:var(--teal)}.check-unknown{color:var(--amber)}.notes{font-size:13px;color:var(--muted);padding-left:20px;margin-bottom:0}.notes li+li{margin-top:6px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px}.panel dl{margin:0}.panel dt{font-size:13px;font-weight:650;margin-top:10px}.panel dt:first-child{margin-top:0}.panel dd{margin:3px 0 0;color:var(--muted);font-size:14px;overflow-wrap:anywhere}.panel ol,.panel ul{padding-left:20px;font-size:14px;margin:0}.panel li+li{margin-top:8px}
.limits{color:var(--muted);font-size:14px;margin:24px 0}.limits h2{font-size:18px;margin-bottom:8px}.limits li+li{margin-top:6px}
details{border:1px solid var(--line);border-radius:8px;background:var(--paper);margin:10px 0}summary{overflow-wrap:anywhere;padding:14px 18px;font-size:14px;font-weight:650;cursor:pointer}details .detail-content{padding:0 18px 18px}pre{font:13px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f2;border-radius:6px;padding:14px;max-height:480px;overflow:auto}code{overflow-wrap:anywhere}.footer{color:var(--muted);font-size:13px;margin-top:26px;border-top:1px solid var(--line);padding-top:16px}
a:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #277b91;outline-offset:4px}a,[tabindex]{scroll-margin-top:20px}.metric-group:target,.metric:target{border-color:#277b91}
.metric-group>.section-heading h3{font-size:22px;margin:0}.metric-group>.section-heading>a{font-size:13px}.results-overview{margin:18px 0}.results-overview .section-heading p{max-width:48ch}.overview-scroll{border:1px solid var(--line);border-radius:8px;background:var(--paper);padding:4px 16px}.overview-table{margin:12px 0;min-width:840px}.overview-table a,.overview-table .scope{display:block}.overview-table .badge{white-space:normal}.overview-table tbody th{font-size:14px}.overview-table td{font-variant-numeric:tabular-nums}.overview-checks{margin:0;padding-left:16px}.overview-checks li+li{margin-top:4px}
.metric-controls{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:18px 0}.metric-navigation{min-width:0;display:flex;flex-wrap:wrap;gap:8px}.metric-navigation a{max-width:100%;overflow-wrap:anywhere;display:block;padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--paper);font-weight:650;text-decoration:none}.metric-navigation a:hover{border-color:#277b91;background:#eaf3f7}.metric-navigation span{display:block;color:var(--muted);font-size:13px;font-weight:400}.metric-navigation [role="tab"][aria-selected="true"]{border-color:#145d7b;background:#e4f1f7;box-shadow:inset 0 -3px #145d7b}.metric-display-toggle{font:inherit;font-size:13px;border:1px solid var(--line);border-radius:7px;background:var(--paper);color:#145d7b;padding:9px 12px;cursor:pointer}.metric-display-toggle:focus-visible{outline:3px solid #277b91;outline-offset:4px}
.report-heading{padding:30px 0 22px;border-bottom:1px solid var(--line);margin-bottom:22px}.report-heading h1{font-size:46px;letter-spacing:-.045em;margin-bottom:8px}.report-heading>p:last-child{margin:0;color:var(--muted);font-size:15px}.review-context h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:14px}.review-subjects{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:0 0 16px}.review-subjects>div{min-width:0;border-left:2px solid var(--line);padding-left:14px}.review-subjects dt{font-size:12px;font-weight:650;color:var(--muted);margin-bottom:4px}.review-subjects dd{font-size:16px;font-weight:600;margin:0;overflow-wrap:anywhere}.context-status{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;list-style:none;padding:0;margin:0 0 12px;font-size:12px;color:var(--muted)}.context-status li{padding:10px 12px;background:#f5f6f2;border-radius:3px}.hero .verdict h2{font-size:30px;letter-spacing:-.03em;line-height:1.2;margin:0 0 14px}.hero.fail{border-left-color:var(--red);background:#fff7f6}.hero.insufficient{border-left-color:var(--amber);background:#fffbf1}.interpretation-note{font-size:12px;color:var(--muted);max-width:90ch;margin:0 0 24px}.findings{border:0;border-block:1px solid var(--line);padding:20px 0}.findings h2,.next-steps h2{font-size:20px}.findings p:last-child{margin:12px 0 0;font-size:13px}.next-steps{background:#f4f6f1;border-left:3px solid #77876b}.next-steps h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;margin-top:18px}.limits{border:0;border-top:1px solid var(--line);padding:20px 0}.section-heading>h2{font-size:24px}.metric-heading h3,.metric-heading h4{font-size:23px}.metric{border-top:3px solid var(--ink)}
@media screen and (max-width:650px){main{padding:20px 18px 40px;margin:0;border-top-width:4px}.review-subjects{grid-template-columns:1fr!important}.report-heading h1{font-size:34px!important}.context-status{grid-template-columns:1fr!important}.hero .verdict h2{font-size:26px!important}.hero{grid-template-columns:1fr;gap:22px;padding:22px 0}.assurance dl>div{grid-template-columns:1fr 1.8fr}h1{font-size:29px}.metric,.panel{padding:18px}.values{grid-template-columns:repeat(2,minmax(0,1fr))}.value:nth-child(2){border-right:0}.value:nth-child(-n+2){border-bottom:1px solid var(--line)}.value dd{font-size:20px}.columns,.subjects{grid-template-columns:1fr}.family{max-width:160px}.section-heading{display:block}.section-heading p{margin-top:5px}.metric-heading{gap:8px;flex-direction:column}th,td{padding:10px 8px}.context-list{grid-template-columns:1fr}.context-list dd{margin-bottom:6px}.comparison-table{font-size:13px}.comparison-table thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.comparison-table,.comparison-table tbody,.comparison-table tr,.comparison-table th,.comparison-table td{display:block;width:100%}.comparison-table th:first-child{width:100%}.comparison-table tbody tr{padding:10px 12px;border-bottom:1px solid var(--line)}.comparison-table tbody th,.comparison-table tbody td{padding:5px 0;border:0}.comparison-table .different-label{display:inline;margin-left:8px}.side-label{display:block;font-size:11px;color:var(--muted);font-weight:500;margin-bottom:2px}.checks-table{min-width:0}.checks-table caption{display:block;width:100%}.checks-table thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.checks-table,.checks-table tbody,.checks-table tbody tr,.checks-table tbody th,.checks-table tbody td{display:block;width:100%}.checks-table th[scope="row"]{width:100%}.checks-table tbody tr{padding:12px 0;border-bottom:1px solid var(--line)}.checks-table tbody th,.checks-table tbody td{border:0;padding:5px 0}.checks-table tbody td{display:grid;grid-template-columns:90px minmax(0,1fr);white-space:normal;overflow-wrap:anywhere}.check-label{display:block;font-weight:400;color:var(--muted)}.interval{margin-inline:0}}
.hero.unavailable{border-left-color:var(--neutral);background:#f5f6f2}.hero.unavailable .verdict h2::before{background:var(--neutral)}.unavailable .badge,.badge.unavailable{color:var(--muted);background:#f5f6f2}
.change-summary{font-size:14px;margin:14px 0 0}.review-action{font-size:14px;margin:16px 0 0;padding-left:16px;border-left:3px solid var(--ink)}.result-counts{font-size:13px;font-weight:650;font-variant-numeric:tabular-nums}.finding-status{font-size:12px;font-weight:700;display:block;margin-bottom:3px}.finding-status.fail{color:var(--red)}.finding-status.insufficient{color:var(--amber)}.finding-status.unavailable{color:var(--muted)}
@page{margin:14mm;@bottom-left{content:"InvarLock · Model change review";font:8pt sans-serif;color:#62695e}@bottom-right{content:"Page " counter(page) " of " counter(pages);font:8pt sans-serif;color:#62695e}}
.identity-list>div{margin-bottom:10px;break-inside:avoid}.identity-list dt{font-size:12px;font-weight:650}.identity-list dd{margin:3px 0 0;font-size:12px;overflow-wrap:anywhere}
@media print{p{break-inside:avoid}*:focus{outline:none!important}.report-heading{padding:16px 0 12px;margin-bottom:12px}.report-heading h1{font-size:25pt}.report-heading .eyebrow{margin-bottom:6px}.interpretation-note{font-size:9pt;margin-bottom:14px}.review-context h2{margin-bottom:10px}.review-subjects{margin-bottom:10px}.review-subjects dd{font-size:10pt}.context-status{gap:8px;margin-bottom:8px}.context-status li{padding:6px 8px}.change-summary{font-size:10pt;margin-top:8px}.decision-summary{font-size:10pt}.hero .verdict h2{font-size:19pt;margin-bottom:10px}.review-action{font-size:10pt;margin-bottom:12px}.findings{padding:12px 0}.interval-reading h5{break-after:avoid}.interval-reading dt,.interval-reading dd{break-inside:avoid}.metric-lead{break-inside:avoid}.metric-explanation,caption,thead{break-after:avoid}.metric .checks-table{break-inside:avoid;font-size:9pt}body{background:white;font-size:10pt}main{max-width:none;padding:0;margin:0;box-shadow:none;border-top:0}.hero{display:block;margin-top:12px;padding:14px;break-inside:auto}.verdict{break-inside:avoid}.assurance{margin-top:16px}.assurance dl>div{grid-template-columns:140px 1fr;padding:6px 0;break-inside:avoid}h1{font-size:25pt}.metric,.panel{padding:15px}.columns{display:block}.panel{margin:12px 0}details{break-inside:auto}details::details-content{display:block!important;content-visibility:visible!important;height:auto!important}details>.detail-content{display:block!important}details.evidence-detail:not([open])::details-content{display:none!important;content-visibility:hidden!important}details.evidence-detail:not([open])>.detail-content{display:none!important}summary{break-after:avoid}details>summary::marker{content:""}pre{max-height:none}.footer{font-size:9pt}.scroll{overflow:visible}a{color:inherit}.metric-group[hidden]{display:block!important}.metric-navigation,.metric-display-toggle,.metric-group>.section-heading>a{display:none}.overview-table,.checks-table{min-width:0;font-size:8pt;table-layout:fixed}.overview-table th{font-size:8pt}.overview-scroll{padding:0;border:0}.overview-table th,.overview-table td{padding:6px 4px}.overview-table .badge{font-size:8pt}.section-heading,.metric-heading,h2,h3,h4{break-after:avoid}tr,.review-context,.findings,.limits,.next-steps,.interval,.notes{break-inside:avoid}details.evidence-detail:not([open])>summary::after{content:" — available in HTML and original evidence";font-weight:400;font-size:9pt;color:var(--muted)}.checks-table td{white-space:normal}}
"""


def _results_overview(metrics: tuple[MetricView, ...]) -> str:
    parts = [
        '<section class="results-overview" aria-labelledby="results-overview-heading"><div class="section-heading"><h2 id="results-overview-heading" tabindex="-1">All results at a glance</h2><p>Each row is one metric and scope; overlapping scopes are not additional independent cases.</p></div>',
        '<div class="scroll overview-scroll" tabindex="0" role="region" aria-label="All metric and scope results"><table class="overview-table"><caption>Recorded results and checks needing attention</caption><thead><tr>',
        '<th scope="col">Metric / scope</th><th scope="col">Decision</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Change</th><th scope="col">Observed pairs</th><th scope="col">Checks needing attention</th></tr></thead><tbody>',
    ]
    for index, metric in enumerate(metrics, 1):
        attention = "".join(
            f'<li class="check-{"fail" if check.passed is False else "unknown"}">{escape(check.name)}: {_status(check)}</li>'
            for check in metric.checks
            if check.passed is not True
        )
        checks = (
            f'<ul class="overview-checks">{attention}</ul>'
            if attention
            else "No failed or unavailable checks"
            if metric.checks
            else "Check details unavailable"
        )
        parts.append(
            f'<tr><th scope="row"><a href="#metric-result-{index}">{escape(metric.display_name)}<span class="scope">{escape(metric.display_scope)}</span></a></th>'
            + f'<td><span class="badge {_tone(metric.decision)}">{escape(decision_label(metric.decision))}</span></td>'
            + "".join(
                f"<td>{escape(value)}</td>"
                for value in (
                    metric.baseline,
                    metric.candidate,
                    metric.change,
                    metric.count,
                )
            )
            + f"<td>{checks}</td></tr>"
        )
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


@dataclass(frozen=True)
class _ComparisonContext:
    rows: tuple[tuple[str, str, str, bool], ...]
    matching: tuple[tuple[str, str], ...]
    additional: tuple[tuple[str, str], ...]


def _comparison_view(view: ReportView) -> _ComparisonContext:
    """Align unambiguous sides and compare displayed text, not hidden full values."""
    side_names = {"Baseline": "Baseline", "Subject": "Subject", "Candidate": "Subject"}
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    shared = []
    matching = []
    for label, value in view.context:
        side, separator, name = label.partition(" ")
        if separator and name and side in side_names:
            grouped.setdefault(name, []).append((label, side_names[side], value))
        else:
            shared.append((label, value))
    paired = {}
    for name, entries in grouped.items():
        if len(entries) == 2 and {side for _, side, _ in entries} == {
            "Baseline",
            "Subject",
        }:
            paired[name] = {side: value for _, side, value in entries}
        else:
            # Repeated fields and competing Subject/Candidate labels cannot be
            # resolved by choosing whichever appeared first or last.
            shared.extend((label, value) for label, _, value in entries)
    rows = []
    subject_entries = []
    for label, value in view.subjects:
        side, _, name = label.partition(" ")
        subject_entries.append((side_names.get(side, side), name, value))
    unambiguous_subjects = (
        len(subject_entries) == 2
        and {side for side, _, _ in subject_entries} == {"Baseline", "Subject"}
        and len({name for _, name, _ in subject_entries}) == 1
    )
    subjects = (
        {side: value for side, _, value in subject_entries}
        if unambiguous_subjects
        else {}
    )
    subject_field = subject_entries[0][1] if unambiguous_subjects else ""
    if unambiguous_subjects and not subject_field:
        for field_name in ("Recorded model ID", "Recorded model key", "Recorded run"):
            prefix = field_name + ": "
            if all(
                value.startswith(prefix) and value[len(prefix) :]
                for value in subjects.values()
            ):
                subject_field = field_name
                subjects = {
                    side: value[len(prefix) :] for side, value in subjects.items()
                }
                break
    observed = paired.get("observed model", {})
    deployment = paired.get("deployment", {})
    represented = (
        unambiguous_subjects
        and not subject_field
        and all(
            side in observed
            and side in deployment
            and subjects.get(side)
            == f"Observed model: {observed[side]} · Deployment: {deployment[side]}"
            for side in ("Baseline", "Subject")
        )
    )
    if represented:
        # The complete displayed subject labels are already represented by rows.
        paired = {"observed model": paired["observed model"], **paired}
    elif unambiguous_subjects:
        rows.append(
            (
                subject_field[:1].upper() + subject_field[1:]
                if subject_field
                else "Subject identity",
                subjects["Baseline"],
                subjects["Subject"],
                subjects["Baseline"] != subjects["Subject"],
            )
        )
    else:
        shared.extend(view.subjects)
    for name, sides in paired.items():
        if sides["Baseline"] == sides["Subject"]:
            matching.append((name[:1].upper() + name[1:], sides["Baseline"]))
        else:
            rows.append(
                (name[:1].upper() + name[1:], sides["Baseline"], sides["Subject"], True)
            )
    return _ComparisonContext(tuple(rows), tuple(matching), tuple(shared))


_MATCHING_CONTEXT_NOTE = (
    "Matching previews do not establish equality of the complete retained fields."
)


def _comparison_context(view: ReportView) -> str:
    context = _comparison_view(view)
    rows = context.rows
    parts = [
        '<section class="panel" aria-labelledby="comparison-context"><h2 id="comparison-context">What was compared</h2>'
    ]
    if rows:
        parts.append(
            '<table class="comparison-table"><thead><tr><th scope="col">Recorded field</th><th scope="col">Baseline</th><th scope="col">Subject</th></tr></thead><tbody>'
        )
        for label, baseline, subject, changed in rows:
            style = ' class="different"' if changed else ""
            annotation = (
                '<span class="different-label">Differs</span>' if changed else ""
            )
            parts.append(
                f'<tr{style}><th scope="row">{escape(label)}{annotation}</th><td data-side="Baseline"><span class="side-label" aria-hidden="true">Baseline</span>{escape(baseline)}</td><td data-side="Subject"><span class="side-label" aria-hidden="true">Subject</span>{escape(subject)}</td></tr>'
            )
        parts.append("</tbody></table>")
    if context.matching or context.additional:
        if rows:
            parts.append(
                f'<details class="shared-context"><summary>Additional recorded context ({len(context.matching) + len(context.additional)})</summary><div class="detail-content">'
            )
        for label, entries in (
            ("Matching displayed fields", context.matching),
            ("Additional fields", context.additional),
        ):
            if not entries:
                continue
            parts.append(f'<h3 class="context-heading">{label}</h3>')
            if label == "Matching displayed fields":
                parts.append('<p class="scope">' + _MATCHING_CONTEXT_NOTE + "</p>")
            parts.append('<dl class="context-list">')
            parts.extend(
                f"<dt>{escape(name)}</dt><dd>{escape(value)}</dd>"
                for name, value in entries
            )
            parts.append("</dl>")
        if rows:
            parts.append("</div></details>")
    if view.changes:
        parts.append(
            '<ul class="change-notes" aria-label="Recorded changes">'
            + "".join(f"<li>{escape(note)}</li>" for note in view.changes)
            + "</ul>"
        )
    parts.append("</section>")
    return "".join(parts)


def _detail_content(data: Any) -> str:
    # Preview text is intentionally not necessarily JSON: bounded walkers may
    # insert elision markers. Escape it directly rather than parsing it.
    if (
        isinstance(data, dict)
        and data.get("preview_only") is True
        and isinstance(data.get("text"), str)
    ):
        note = str(
            data.get("note", "Bounded preview; full values remain in the evidence.")
        )
        limits = data.get("limits", {})
        return (
            "<p>"
            + escape(note)
            + "</p><pre>"
            + escape(data["text"])
            + '</pre><p class="scope">Preview limits: '
            + escape(
                json.dumps(limits, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            + "</p>"
        )
    return (
        "<pre>"
        + escape(
            json.dumps(
                data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
        )
        + "</pre>"
    )


def _summary_html(view: ReportView) -> str:
    """Emphasize supplied display values without interpreting summary markup."""
    if len(view.metrics) != 1:
        return escape(view.summary)
    metric = view.metrics[0]
    values = {metric.display_name, metric.baseline, metric.candidate, metric.change}
    if metric.interval:
        values.update(
            number(value)
            for value in (
                metric.interval.lower,
                metric.interval.upper,
                metric.interval.estimate,
                metric.interval.threshold,
            )
            if value is not None
        )
    tokens = sorted((value for value in values if value), key=len, reverse=True)
    if not tokens:
        return escape(view.summary)
    pattern = re.compile(
        r"(?<![\w.+-])("
        + "|".join(re.escape(value) for value in tokens)
        + r")(?![\w%]|\.\d)"
    )
    return "".join(
        "<strong>" + escape(part) + "</strong>" if index % 2 else escape(part)
        for index, part in enumerate(pattern.split(view.summary))
    )


_INTERPRETATION_NOTE = (
    "Generated interpretation of recorded evidence. This presentation is not signed "
    "evidence or an approval; keep any authored business commentary separately labeled."
)
_DECISION_CONTEXT = (
    "Intended use and business context: Not supplied",
    "Decision owner and recommendation: Not supplied",
    "Organizational approval: Not recorded by this report",
)
_NEXT_DECISION = (
    "The decision owner should confirm that the evaluated cases and requirements cover "
    "the intended use, review unresolved findings, and record the release decision "
    "alongside recipient verification."
)
_MISSING_LIMITATIONS = (
    "Limitations were not supplied with this view. Their absence does not establish "
    "that the evaluated scope covers the intended use."
)
_MISSING_ASSURANCE = "Reporting checks were not supplied with this view."
_MISSING_CHECKS = (
    "Decision check details were not supplied with this view. Consult the original "
    "evidence before interpreting the recorded result."
)


def _interpretation_note(view: ReportView) -> str:
    if view.review:
        return (
            "Decision context, recommendation and review rationale are separately "
            "authored commentary. Technical results come from the retained evidence."
        )
    return _INTERPRETATION_NOTE


def _review_action(view: ReportView) -> str:
    if view.decision == "pass":
        return "Confirm the intended use and complete recipient verification before recording a release decision."
    if view.decision in {"fail", "regression"}:
        return (
            "Review the unmet requirements and their causes before considering release."
        )
    if view.decision == "insufficient_evidence":
        return "Resolve the missing or insufficient evidence before drawing a release conclusion."
    return "Establish the recorded outcome and evidence coverage before drawing a release conclusion."


def _result_counts(view: ReportView) -> str:
    counts = dict.fromkeys(("pass", "fail", "insufficient", "unavailable"), 0)
    for metric in view.metrics:
        counts[_tone(metric.decision)] += 1
    return " · ".join(
        f"{counts[tone]} {label}"
        for tone, label in (
            ("pass", "satisfied"),
            ("fail", "not met"),
            ("insufficient", "need more evidence"),
            ("unavailable", "unavailable"),
        )
        if counts[tone]
    )


def _change_summary(view: ReportView) -> str:
    if not view.changes:
        return "No change description was supplied with this view."
    extra = len(view.changes) - 1
    return view.changes[0] + (
        f" {extra} additional comparison note{'s' if extra != 1 else ''} under What was compared."
        if extra
        else ""
    )


def _findings(view: ReportView) -> tuple[list[tuple[int, MetricView]], str]:
    findings = [(i, m) for i, m in enumerate(view.metrics, 1) if m.decision != "pass"]
    if findings:
        note = f"{len(findings)} of {len(view.metrics)} metric / scope results need review."
        if len(findings) > 5:
            note += " First 5 shown here; all results and checks follow below."
    elif view.metrics:
        note = "No failed or unresolved metric results are recorded. This describes only the evaluated scope and requirements."
    else:
        note = "Metric results are unavailable in this view."
    return findings[:5], note


def _review_context_html(view: ReportView) -> str:
    if view.review:
        review = view.review
        return (
            '<section class="review-context" aria-labelledby="review-context"><h2 id="review-context">Decision context</h2>'
            f'<p class="eyebrow">{escape(review.label)}</p>'
            f"<p><strong>Owner decision.</strong> {escape(review.decision)}</p>"
            f'<p class="scope">{escape(review.byline)}</p>'
            + "".join(
                f"<p><strong>{escape(label)}.</strong> {escape(value)}</p>"
                for label, value in review.context
            )
            + "</section>"
            + f'<p class="change-summary"><strong>Change context.</strong> {escape(_change_summary(view))}</p>'
        )
    return (
        '<section class="review-context" aria-labelledby="review-context"><h2 id="review-context">Decision context</h2>'
        + (
            '<dl class="review-subjects">'
            + "".join(
                f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
                for label, value in view.subjects
            )
            + "</dl>"
            if view.subjects
            else ""
        )
        + '<ul class="context-status">'
        + "".join(f"<li>{item}</li>" for item in _DECISION_CONTEXT)
        + '</ul><p class="scope">The recorded task, cases and requirements define the evaluated scope. A policy result does not supply a business recommendation.</p></section>'
        + f'<p class="change-summary"><strong>Change context.</strong> {escape(_change_summary(view))}</p>'
    )


def _findings_html(view: ReportView) -> str:
    findings, note = _findings(view)
    return (
        '<section class="panel findings"><h2>Findings to review</h2>'
        f"<p>{note}</p>"
        + (
            f'<p class="result-counts">{_result_counts(view)}</p>'
            if view.metrics
            else ""
        )
        + "<ul>"
        + "".join(
            f'<li><span class="finding-status {_tone(metric.decision)}">{escape(decision_label(metric.decision))}</span> <a href="#metric-result-{index}">Result {index}: {escape(metric.display_name)} · {escape(metric.display_scope)}</a> — {escape(metric.explanation)}</li>'
            for index, metric in findings
        )
        + '</ul><p><a href="#results-and-requirements">Supporting evidence: results and decision checks</a></p></section>'
    )


def render_html(view: ReportView) -> str:
    """Render escaped fields with optional hash-authorized local metric tabs."""
    e = escape
    multiple_results = len(view.metrics) > 1
    groups: dict[str, list[tuple[int, MetricView]]] = {}
    for result_index, metric in enumerate(view.metrics, 1):
        groups.setdefault(metric.name, []).append((result_index, metric))
    tabs_enabled = len(groups) > 1
    script_policy = f" script-src 'sha256-{_METRIC_TABS_HASH}';" if tabs_enabled else ""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none';{script_policy}\">",
        f"<title>{e(decision_label(view.decision))} · {e(view.title)}</title><style>{_CSS}</style></head><body><main>",
        f'<header class="brand">{_BRAND_MARK} InvarLock<span class="family">{e(view.family)}</span></header>',
        f'<div class="report-heading"><p class="eyebrow">Comparison / Review / Evidence</p><h1>{"Client review package" if view.review else "Model change review"}</h1><p>{e(view.title)}</p></div>',
        f'<p class="interpretation-note">{_interpretation_note(view)}</p>',
        _review_context_html(view),
        f'<section class="hero {_tone(view.decision)}" aria-labelledby="decision"><div class="verdict"><p class="eyebrow">Recorded policy result</p><h2 id="decision">{e(decision_label(view.decision))}</h2><p class="decision-summary">{_summary_html(view)}</p>',
    ]
    if multiple_results:
        parts.append(
            f'<div class="summary-row"><span class="pill">{len(view.metrics)} metric / scope results</span></div>'
        )
    parts.append(
        f'<p class="review-action"><strong>{"Owner recommendation" if view.review else "Review action"}.</strong> {e(view.review.recommendation) if view.review else _review_action(view)}</p>'
    )
    parts.append(
        '</div><aside class="assurance" aria-labelledby="assurance-heading"><h2 id="assurance-heading">What was checked</h2><dl>'
    )
    for name, value in view.assurance:
        parts.append(f"<div><dt>{e(name)}</dt><dd>{e(value)}</dd></div>")
    parts.append("</dl>")
    if not view.assurance:
        parts.append(f"<p>{_MISSING_ASSURANCE}</p>")
    parts.append("</aside></section>")
    parts.append(_findings_html(view))
    support = [
        f'<section class="panel next-steps"><h2>{"Decision follow-up" if view.review else "Next decision"}</h2><p>{e(view.review.next_action) if view.review else _NEXT_DECISION}</p>'
    ]
    if view.next_steps and not view.review:
        support.append("<h3>Next steps</h3><ol>")
        support.extend(f"<li>{e(step)}</li>" for step in view.next_steps)
        support.append("</ol>")
    support.append("</section>")
    parts.append(
        '<section class="panel limits"><h2>Scope and limitations</h2><ul>'
        + "".join(
            f"<li>{e(n)}</li>" for n in (view.limitations or (_MISSING_LIMITATIONS,))
        )
        + "</ul></section>"
    )
    parts.extend(support)
    if view.review:
        parts.append(
            '<section class="panel review-basis"><h2>Review rationale</h2>'
            + "".join(
                f"<p><strong>{e(label)}.</strong> {e(value)}</p>"
                for label, value in view.review.basis
            )
            + "</section>"
        )
    if view.subjects or view.context or view.changes:
        parts.append(_comparison_context(view))
    parts.append(
        '<div class="section-heading"><h2 id="results-and-requirements">Results and requirements</h2></div>'
    )
    if multiple_results:
        parts.append(
            '<details class="overview-disclosure"'
            + (
                " open"
                if any(metric.decision != "pass" for metric in view.metrics)
                else ""
            )
            + '><summary>Compare all metric and scope results</summary><div class="detail-content">'
            + _results_overview(view.metrics)
            + "</div></details>"
        )
        if len(groups) > 1:
            parts.append(
                '<div class="metric-controls"><nav class="metric-navigation" aria-label="Metric results">'
                + "".join(
                    f'<a href="#metric-group-{index}">{e(display_label(name))} <span>{len(group)} scope result{"s" if len(group) != 1 else ""}</span></a>'
                    for index, (name, group) in enumerate(groups.items(), 1)
                )
                + '</nav><button class="metric-display-toggle" type="button" hidden>Show all metrics</button></div>'
            )
    for group_index, (metric_name, group) in enumerate(groups.items(), 1):
        if multiple_results:
            parts.append(
                f'<section id="metric-panel-{group_index}" class="metric-group" aria-labelledby="metric-group-{group_index}"><div class="section-heading"><h3 id="metric-group-{group_index}" tabindex="-1">{e(display_label(metric_name))}</h3><a href="#results-overview-heading">Back to overview</a></div>'
            )
        for result_index, metric in group:
            heading = (
                f"<h4>{e(metric.display_scope)}</h4>"
                if multiple_results
                else f"<h3>{e(metric.display_name)}</h3>"
            )
            context = (
                f"Metric: {e(metric.display_name)}"
                if multiple_results
                else f"Scope: {e(metric.display_scope)}"
            )
            parts.append(
                f'<section id="metric-result-{result_index}" tabindex="-1" class="metric {_tone(metric.decision)}"><div class="metric-lead"><div class="metric-heading"><div>{heading}<p class="scope">{context}</p></div><span class="badge">{e(decision_label(metric.decision))}</span></div><p class="metric-explanation">{e(metric.explanation)}</p><dl class="values">'
            )
            change_detail = (
                _interval_summary(metric.interval, compact=True)
                if metric.interval
                else ""
            )
            for name, value, detail in [
                ("Baseline", metric.baseline, metric.baseline_detail),
                ("Subject", metric.candidate, metric.candidate_detail),
                ("Change", metric.change, change_detail),
                (metric.count_label, metric.count, metric.count_detail),
            ]:
                sublabel = f"<small>{e(detail)}</small>" if detail else ""
                parts.append(
                    f'<div class="value"><dt>{e(name)}</dt><dd>{e(value)}{sublabel}</dd></div>'
                )
            parts.append("</dl></div>")
            if metric.interval:
                parts.append(_interval(metric.interval))
                parts.append(
                    _interval_reading_html(
                        metric.interval, heading_level=5 if multiple_results else 4
                    )
                )
            if metric.checks:
                parts.append(
                    '<div class="scroll" tabindex="0" role="region" aria-label="Decision checks"><table class="checks-table"><caption>Decision checks</caption><thead><tr><th scope="col">Check</th><th scope="col">Observed</th><th scope="col">Required</th><th scope="col">Result</th></tr></thead><tbody>'
                )
            for check in metric.checks:
                tone = (
                    "pass"
                    if check.passed is True
                    else "fail"
                    if check.passed is False
                    else "unknown"
                )
                detail = (
                    f'<span class="check-detail">{e(check.explanation)}</span>'
                    if check.explanation
                    else ""
                )
                parts.append(
                    f'<tr><th scope="row">{e(check.name)}{detail}</th><td data-label="Observed"><span class="check-label" aria-hidden="true">Observed</span>{e(check.observed)}</td><td data-label="Required"><span class="check-label" aria-hidden="true">Required</span>{e(check.required)}</td><td data-label="Result" class="check-{tone}"><span class="check-label" aria-hidden="true">Result</span>{_status(check)}</td></tr>'
                )
            if metric.checks:
                parts.append("</tbody></table></div>")
            else:
                parts.append(f'<p class="missing-checks">{_MISSING_CHECKS}</p>')
            parts.append(
                f'<p class="scope">Result {result_index} · Supporting record: {e(metric.evidence_source or "Not supplied with this view")}</p>'
            )
            if metric.notes:
                parts.append(
                    '<ul class="notes">'
                    + "".join(f"<li>{e(n)}</li>" for n in metric.notes)
                    + "</ul>"
                )
            parts.append("</section>")
        if multiple_results:
            parts.append("</section>")
    parts.append(
        '<div class="section-heading"><h2>Evidence details</h2><p>Identities, methods and exact recorded values</p></div><p class="scope">The review and methods print automatically. Open an evidence detail below to include its contents in print; complete originals remain in the evidence bundle.</p>'
    )
    if view.identity:
        parts.append(
            '<details><summary>Comparison and policy identities</summary><div class="detail-content"><dl class="identity-list">'
        )
        parts.extend(
            f"<div><dt>{e(k)}</dt><dd><code>{e(v)}</code></dd></div>"
            for k, v in view.identity
        )
        parts.append("</dl></div></details>")
    for heading, data in (
        *_metric_identifier_details(view),
        *view.details,
        ("Exact comparison data", view.technical),
    ):
        parts.append(
            f'<details class="evidence-detail"><summary>{e(heading)}</summary><div class="detail-content">{_detail_content(data)}</div></details>'
        )
    for _, data in view.details:
        if not isinstance(data, dict) or not isinstance(data.get("metrics"), list):
            continue
        for item in data["metrics"]:
            if not isinstance(item, dict):
                continue
            preview = item.get("configuration")
            if (
                isinstance(preview, dict)
                and preview.get("preview_only") is True
                and isinstance(preview.get("text"), str)
            ):
                parts.append(
                    f'<details class="evidence-detail"><summary>{e(display_label(str(item.get("name", "Metric"))))} configuration preview</summary><div class="detail-content">{_detail_content(preview)}</div></details>'
                )
    parts.append(
        '<footer class="footer">InvarLock · Evidence report. Displayed values may be rounded; exact values are preserved in the evidence bundle. This report is not an independent acceptance receipt.</footer></main></body></html>\n'
    )
    if tabs_enabled:
        parts[-1] = parts[-1].replace(
            "</body>", f"<script>{_METRIC_TABS_SCRIPT}</script></body>"
        )
    return "".join(parts)


def render_markdown(view: ReportView, *, include_details: bool = False) -> str:
    def clean(value: str) -> str:
        value = escape(value, quote=False).replace("\n", " ").replace("\r", " ")
        for ch in "|\\`*[]()":
            value = value.replace(ch, f"&#{ord(ch)};")
        return re.sub(r"(?<!\w)_|_(?!\w)", "&#95;", value)

    lines = [
        f"# {clean(view.title)}",
        "",
        "## Decision context",
        "",
        _interpretation_note(view),
        "",
        *(f"- **{clean(label)}:** {clean(value)}" for label, value in view.subjects),
        "",
        *(
            [
                f"**{clean(view.review.label)}**",
                "",
                f"**Owner decision:** {clean(view.review.decision)}",
                "",
                clean(view.review.byline),
                "",
                *(f"- **{clean(k)}:** {clean(v)}" for k, v in view.review.context),
            ]
            if view.review
            else [f"- {item}" for item in _DECISION_CONTEXT]
        ),
        "",
        f"**Change context:** {clean(_change_summary(view))}",
        "",
        f"**Recorded policy result: {decision_label(view.decision)}**",
        "",
        clean(view.summary),
        "",
        f"**{'Owner recommendation' if view.review else 'Review action'}:** {clean(view.review.recommendation) if view.review else _review_action(view)}",
        "",
    ]
    findings, note = _findings(view)
    lines += ["## Findings to review", "", note, ""]
    if view.metrics:
        lines += [_result_counts(view), ""]
    lines.extend(
        f"- {decision_label(m.decision)} — Result {i}: {clean(m.display_name)} · {clean(m.display_scope)} — {clean(m.explanation)} Supporting record: {clean(m.evidence_source or 'Not supplied with this view')}"
        for i, m in findings
    )
    lines += [
        "",
        "Supporting evidence: the numbered results and decision checks below.",
        "",
    ]
    lines += ["", "## What was checked", ""]
    lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.assurance)
    if not view.assurance:
        lines += [_MISSING_ASSURANCE, ""]
    lines += ["", "## Scope and limitations", ""]
    lines.extend(f"- {clean(s)}" for s in (view.limitations or (_MISSING_LIMITATIONS,)))
    lines += [
        "",
        "## Decision follow-up" if view.review else "## Next decision",
        "",
        clean(view.review.next_action) if view.review else _NEXT_DECISION,
        "",
    ]
    if view.next_steps and not view.review:
        lines += ["### Next steps", ""]
        lines.extend(f"{i}. {clean(s)}" for i, s in enumerate(view.next_steps, 1))
    if view.review:
        lines += ["", "## Review rationale", ""]
        lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.review.basis)
    if view.subjects or view.context or view.changes:
        lines += ["## What was compared", ""]
        context = _comparison_view(view)
        # Paired lists retain full identifiers in narrow Markdown terminals;
        # table cells can silently ellipsize long model and artifact names.
        for label, baseline, subject, changed in context.rows:
            difference = "Differs" if changed else "Same displayed text"
            lines += [
                "",
                f"### {clean(label)}: {difference}",
                "",
                f"- **Baseline:** {clean(baseline)}",
                f"- **Subject:** {clean(subject)}",
            ]
        for heading, entries in (
            ("Matching displayed fields", context.matching),
            ("Additional fields", context.additional),
        ):
            if entries:
                lines += ["", "### " + heading, ""]
                if heading == "Matching displayed fields":
                    lines += [_MATCHING_CONTEXT_NOTE, ""]
                lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in entries)
        if view.changes:
            lines += ["", "### Recorded changes", ""]
            lines.extend(f"- {clean(change)}" for change in view.changes)
    for result_index, metric in enumerate(view.metrics, 1):
        lines += [
            "",
            f"## {clean(metric.display_name)}: {clean(metric.display_scope)}",
            "",
            f"**{decision_label(metric.decision)}**. {clean(metric.explanation)}",
            "",
            f"| Baseline | Subject | Change | {clean(metric.count_label)} |",
            "| --- | --- | --- | --- |",
            "| "
            + " | ".join(
                map(
                    clean,
                    [metric.baseline, metric.candidate, metric.change, metric.count],
                )
            )
            + " |",
            "",
        ]
        for label, detail in (
            ("Baseline", metric.baseline_detail),
            ("Subject", metric.candidate_detail),
            (metric.count_label, metric.count_detail),
        ):
            if detail:
                lines += [f"{clean(label)}: {clean(detail)}.", ""]
        if metric.interval:
            i = metric.interval
            lines += [
                f"{clean(_interval_label(i))}: [{number(i.lower)}, {number(i.upper)}] {clean(i.unit)}.",
                "",
                "### How to read this comparison",
                "",
            ]
            for label, explanation in _interval_reading(i):
                lines += [f"**{clean(label)}:** {clean(explanation)}", ""]
            if i.basis:
                lines += ["### How this interval was calculated", ""]
                for paragraph in i.basis:
                    lines += [clean(paragraph), ""]
        lines += ["### Decision checks", ""]
        if metric.checks:
            lines += [
                "| Check | Observed | Required | Result |",
                "| --- | --- | --- | --- |",
            ]
        lines.extend(
            "| "
            + " | ".join(map(clean, [c.name, c.observed, c.required, _status(c)]))
            + " |"
            for c in metric.checks
        )
        lines.append("")
        if not metric.checks:
            lines += [_MISSING_CHECKS, ""]
        lines += [
            f"Result {result_index} · Supporting record: {clean(metric.evidence_source or 'Not supplied with this view')}",
            "",
        ]
        lines.extend(f"- {clean(n)}" for n in metric.notes)
    lines += ["", "## Evidence identities", ""]
    lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.identity)
    details = _metric_identifier_details(view)
    if include_details:
        details += (*view.details, ("Exact comparison data", view.technical))
    for heading, data in details:
        lines += [
            "",
            f"## {clean(heading)}",
            "",
            "```json",
            visible_controls(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            ).replace("`", "\\u0060"),
            "```",
        ]
    return "\n".join(lines) + "\n"
