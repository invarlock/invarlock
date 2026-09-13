"""Shared, non-authoritative presentation of explicit comparison facts.

Builders retain each evidence family's own validation and trust rules. These
views never authorize a signer, replay a score, or change a stored decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
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


# Static mark from docs/assets/invarlock-app-icon.svg; the adjacent name labels it.
_BRAND_MARK = """<svg class="mark" xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 512 512" aria-hidden="true" focusable="false">
<rect x="32" y="32" width="448" height="448" rx="92" fill="#11130f"/>
<rect x="32.5" y="32.5" width="447" height="447" rx="91.5" fill="none" stroke="#3f4235"/>
<g transform="translate(256 256) scale(1.3) translate(-256 -256)">
<path d="M174 150 H126 V362 H174" fill="none" stroke="#f4efe3" stroke-width="26.15" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
<path d="M338 150 H386 V362 H338" fill="none" stroke="#f4efe3" stroke-width="26.15" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
<path d="M156 292 C 198 212, 230 212, 256 252 S 314 332, 356 242" fill="none" stroke="#9fb7ff" stroke-width="24.62" stroke-linecap="round" opacity="0.72"/>
<path d="M156 292 C 198 212, 230 212, 256 252 S 314 332, 356 242" fill="none" stroke="#f4efe3" stroke-width="10.77" stroke-linecap="round" opacity="0.52"/>
<circle cx="256" cy="252" r="6.92" fill="#f4efe3" opacity="0.64"/>
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
  main.querySelectorAll('.results-overview a[href^="#metric-result-"]').forEach(link => {
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
    )


def _status(check: CheckView) -> str:
    return (
        "Passed"
        if check.passed is True
        else "Not met"
        if check.passed is False
        else "Unavailable"
    )


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

    def x(value: float) -> float:
        return 30 + (value / scale - low) / (high - low) * 580

    description = f"{view.label}: {number(view.lower)} to {number(view.upper)} {view.unit}. Estimate {number(view.estimate)} {view.unit}."
    graphics = []
    legend = ""
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
            operator = "≥" if direction == "minimum" else "≤"
            legend = f"Change requirement: {operator} {number(view.threshold)} {view.unit}. Shading shows where this requirement is met."
        else:
            legend = f"Policy threshold: {number(view.threshold)} {view.unit}."
        graphics.append(
            f'<line class="threshold" x1="{position:.2f}" x2="{position:.2f}" y1="10" y2="66"/>'
        )
    if view.neutral is not None:
        position = x(view.neutral)
        graphics.append(
            f'<line class="neutral" x1="{position:.2f}" x2="{position:.2f}" y1="18" y2="60"/>'
        )
        legend += f" No change: {number(view.neutral)} {view.unit}."
    ticks = [first] if first == last else [first, first / 2 + last / 2, last]
    labels = []
    for tick in ticks:
        value = tick * scale
        position = x(value)
        graphics.append(
            f'<line class="tick" x1="{position:.2f}" x2="{position:.2f}" y1="60" y2="67"/>'
        )
        labels.append(
            f'<span style="left:{position / 640 * 100:.2f}%">{escape(number(value))}</span>'
        )
    return (
        '<figure class="interval"><svg viewBox="0 0 640 72" role="img" aria-label="'
        + escape(description, quote=True)
        + '">'
        + "".join(graphics)
        + '<line class="axis" x1="30" x2="610" y1="60" y2="60"/>'
        + f'<line class="range" x1="{x(view.lower):.2f}" x2="{x(view.upper):.2f}" y1="38" y2="38"/>'
        + f'<circle class="estimate" cx="{x(view.estimate):.2f}" cy="38" r="6"/>'
        + '</svg><div class="axis-labels" aria-hidden="true">'
        + "".join(labels)
        + "</div>"
        + "<figcaption>"
        + escape(description)
        + (
            ' <span class="legend">' + escape(legend.strip()) + "</span>"
            if legend
            else ""
        )
        + "</figcaption></figure>"
    )


_CSS = """
:root{color-scheme:light;--ink:#172a35;--muted:#526773;--line:#d8e3e8;--paper:#fff;--canvas:#f2f6f8;--teal:#086756;--red:#a32639;--amber:#79530b}
*{box-sizing:border-box}
body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:#145d7b}main{max-width:1120px;margin:auto;padding:30px 28px 60px}
h1,h2,h3,h4,p{margin-top:0}h1{font-size:36px;line-height:1.15;letter-spacing:-.03em;margin-bottom:14px}h2{font-size:22px;line-height:1.3;letter-spacing:-.02em;margin-bottom:0}h3,h4{font-size:20px;line-height:1.3;margin-bottom:10px}
.brand{display:flex;align-items:center;gap:10px;font-weight:750}.mark{width:34px;height:34px;flex-shrink:0}.family{margin-left:auto;color:var(--muted);font-size:13px;text-align:right}
.hero{margin:24px 0;padding:28px 30px;border:1px solid var(--line);border-radius:14px;background:var(--paper);border-top:4px solid var(--teal)}
.hero.fail{border-top-color:var(--red)}.hero.insufficient{border-top-color:var(--amber)}.hero>p{max-width:90ch;margin-bottom:0;overflow-wrap:anywhere}.hero .eyebrow{margin-bottom:10px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:750;color:var(--muted)}
.summary-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.pill{font-size:13px;font-weight:650;padding:4px 10px;border-radius:6px;background:#eef3f6;color:var(--muted)}
.badge{display:inline-block;font-size:13px;font-weight:700;border:1px solid currentColor;border-radius:5px;padding:2px 8px;white-space:nowrap}
.pass .badge,.badge.pass{color:var(--teal);background:#eef8f4}.fail .badge,.badge.fail{color:var(--red);background:#fff1f2}.insufficient .badge,.badge.insufficient{color:var(--amber);background:#fff8e6}
.panel,.metric{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px}.panel h2{font-size:18px;margin-bottom:12px}.metric{margin-bottom:18px;break-inside:avoid}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:26px 0 14px}.section-heading p{font-size:13px;color:var(--muted);margin:0}
.subjects{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0}.subjects div{min-width:0}.subjects dt{font-size:13px;font-weight:650;color:var(--muted)}.subjects dd{margin:3px 0 0;overflow-wrap:anywhere}
.comparison-table{table-layout:fixed;margin-top:0}.comparison-table th:first-child{width:23%}.comparison-table td{width:38.5%;overflow-wrap:anywhere}.comparison-table .different{background:#f0f6fa}.different-label{display:block;font-size:12px;font-weight:400;color:#315e76}
.shared-context{border:0;border-top:1px solid var(--line);border-radius:0;margin:12px 0 0;background:transparent}.shared-context summary{padding:12px 0}.shared-context .detail-content{padding:0}
.context-list{display:grid;grid-template-columns:minmax(120px,1fr) 3fr;gap:8px 18px;margin:0}.context-list dt{font-weight:650;font-size:13px}.context-list dd{margin:0;overflow-wrap:anywhere;font-size:14px}
.change-notes{font-size:13px;color:var(--muted);margin:14px 0 0;padding-left:20px}.change-notes li+li{margin-top:6px}
.metric-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.metric-heading>div{min-width:0}.metric h3,.metric h4{overflow-wrap:anywhere}.metric-heading>.badge{flex-shrink:0}.scope{font-size:13px;color:var(--muted);margin:4px 0}.metric-explanation{margin:10px 0 18px}
.values{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:8px;background:#f8fafb;overflow:hidden;margin:0}
.value{padding:14px;border-right:1px solid var(--line);min-width:0}.value:last-child{border:0}.value dt{font-size:13px;color:var(--muted)}.value dd{margin:4px 0 0;font-size:22px;font-weight:650;line-height:1.3;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.interval{margin:22px 0}.interval svg{display:block;width:100%;height:auto}.interval figcaption{overflow-wrap:anywhere;font-size:13px;color:var(--muted);margin-top:8px}.axis{stroke:var(--line);stroke-width:2}.range{stroke:#277b91;stroke-width:7;stroke-linecap:round}.estimate{fill:var(--ink);stroke:white;stroke-width:2}.threshold{stroke:var(--amber);stroke-width:2;stroke-dasharray:4 4}.neutral{stroke:#78909c;stroke-width:1.5}.allowed{fill:#e5f3ed}.tick{stroke:#78909c;stroke-width:1.5}.legend{display:block;margin-top:4px}.axis-labels{position:relative;height:22px;font-size:13px;font-variant-numeric:tabular-nums;color:var(--muted)}.axis-labels span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin-top:16px}caption{text-align:left;font-weight:650;padding:0 0 8px}th{text-align:left;color:var(--muted);font-size:13px;font-weight:650}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}th:first-child,td:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
.checks-table td{white-space:nowrap}.checks-table th[scope="row"]{width:48%}.check-detail{display:block;font-size:13px;font-weight:400;margin-top:4px;color:var(--muted)}.check-fail{color:var(--red);font-weight:650}.check-pass{color:var(--teal)}.check-unknown{color:var(--amber)}.notes{font-size:13px;color:var(--muted);padding-left:20px;margin-bottom:0}.notes li+li{margin-top:6px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px}.panel dl{margin:0}.panel dt{font-size:13px;font-weight:650;margin-top:10px}.panel dt:first-child{margin-top:0}.panel dd{margin:3px 0 0;color:var(--muted);font-size:14px;overflow-wrap:anywhere}.panel ol,.panel ul{padding-left:20px;font-size:14px;margin:0}.panel li+li{margin-top:8px}
.limits{color:var(--muted);font-size:14px;margin:24px 0}.limits h2{font-size:18px;margin-bottom:8px}.limits li+li{margin-top:6px}
details{border:1px solid var(--line);border-radius:8px;background:var(--paper);margin:10px 0}summary{overflow-wrap:anywhere;padding:14px 18px;font-size:14px;font-weight:650;cursor:pointer}details .detail-content{padding:0 18px 18px}pre{font:13px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;border-radius:6px;padding:14px;max-height:480px;overflow:auto}code{overflow-wrap:anywhere}.footer{color:var(--muted);font-size:13px;margin-top:26px;border-top:1px solid var(--line);padding-top:16px}
a:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #277b91;outline-offset:4px}a,[tabindex]{scroll-margin-top:20px}.metric-group:target,.metric:target{border-color:#277b91}
.metric-group>.section-heading h3{font-size:22px;margin:0}.metric-group>.section-heading>a{font-size:13px}.results-overview{margin:18px 0}.results-overview .section-heading p{max-width:48ch}.overview-scroll{border:1px solid var(--line);border-radius:8px;background:var(--paper);padding:4px 16px}.overview-table{margin:12px 0;min-width:840px}.overview-table a,.overview-table .scope{display:block}.overview-table .badge{white-space:normal}.overview-table tbody th{font-size:14px}.overview-table td{font-variant-numeric:tabular-nums}.overview-checks{margin:0;padding-left:16px}.overview-checks li+li{margin-top:4px}
.metric-controls{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:18px 0}.metric-navigation{display:flex;flex-wrap:wrap;gap:8px}.metric-navigation a{display:block;padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--paper);font-weight:650;text-decoration:none}.metric-navigation a:hover{border-color:#277b91;background:#eaf3f7}.metric-navigation span{display:block;color:var(--muted);font-size:13px;font-weight:400}.metric-navigation [role="tab"][aria-selected="true"]{border-color:#145d7b;background:#e4f1f7;box-shadow:inset 0 -3px #145d7b}.metric-display-toggle{font:inherit;font-size:13px;border:1px solid var(--line);border-radius:7px;background:var(--paper);color:#145d7b;padding:9px 12px;cursor:pointer}.metric-display-toggle:focus-visible{outline:3px solid #277b91;outline-offset:4px}
@media(max-width:650px){main{padding:20px 14px 40px}.hero{padding:22px 18px}h1{font-size:29px}.metric,.panel{padding:18px}.values{grid-template-columns:repeat(2,minmax(0,1fr))}.value:nth-child(2){border-right:0}.value:nth-child(-n+2){border-bottom:1px solid var(--line)}.value dd{font-size:20px}.columns,.subjects{grid-template-columns:1fr}.family{max-width:160px}.section-heading{display:block}.section-heading p{margin-top:5px}.metric-heading{gap:8px;flex-direction:column}th,td{padding:10px 8px}.context-list{grid-template-columns:1fr}.context-list dd{margin-bottom:6px}.comparison-table{font-size:13px}.comparison-table th:first-child{width:25%}.checks-table{min-width:540px}.interval{margin-inline:0}}
@media print{body{background:white;font-size:10pt}main{max-width:none;padding:0}.hero{margin-top:16px;padding:20px}h1{font-size:25pt}.metric,.panel{padding:15px}.columns{display:block}.panel{margin:12px 0}details{break-inside:avoid}details>.detail-content{display:block!important}pre{max-height:none}.footer{font-size:9pt}.scroll{overflow:visible}a{color:inherit}.metric-group[hidden]{display:block!important}.metric-navigation,.metric-display-toggle,.metric-group>.section-heading>a{display:none}.overview-table,.checks-table{min-width:0;font-size:8pt;table-layout:fixed}.overview-table th{font-size:8pt}.overview-scroll{padding:0;border:0}.overview-table th,.overview-table td{padding:6px 4px}.overview-table .badge{font-size:8pt}.metric-group>.section-heading{break-after:avoid}.checks-table td{white-space:normal}}
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
            f'<tr><th scope="row"><a href="#metric-result-{index}">{escape(metric.name)}<span class="scope">{escape(metric.scope)}</span></a></th>'
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


def _comparison_context(view: ReportView) -> str:
    """Align unambiguous sides and compare displayed text, not hidden full values."""
    side_names = {"Baseline": "Baseline", "Subject": "Subject", "Candidate": "Subject"}
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    shared = []
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
                False,
            )
        )
    else:
        shared.extend(view.subjects)
    for name, sides in paired.items():
        if sides["Baseline"] == sides["Subject"]:
            shared.append(
                (name.capitalize() + " (same displayed text)", sides["Baseline"])
            )
        else:
            rows.append((name.capitalize(), sides["Baseline"], sides["Subject"], True))
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
                f'<tr{style}><th scope="row">{escape(label)}{annotation}</th><td>{escape(baseline)}</td><td>{escape(subject)}</td></tr>'
            )
        parts.append("</tbody></table>")
    if shared:
        if rows:
            parts.append(
                f'<details class="shared-context"><summary>Additional recorded context ({len(shared)})</summary><div class="detail-content">'
            )
        parts.append('<dl class="context-list">')
        parts.extend(
            f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
            for label, value in shared
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
        f"<title>{e(view.title)}</title><style>{_CSS}</style></head><body><main>",
        f'<header class="brand">{_BRAND_MARK} InvarLock<span class="family">{e(view.family)}</span></header>',
        f'<section class="hero {_tone(view.decision)}" aria-labelledby="decision"><p class="eyebrow">Recorded policy result</p><h1 id="decision">{e(decision_label(view.decision))}</h1><p>{e(view.summary)}</p>',
    ]
    if multiple_results:
        parts.append(
            f'<div class="summary-row"><span class="pill">{len(view.metrics)} metric / scope results</span></div>'
        )
    parts.append("</section>")
    if view.subjects or view.context or view.changes:
        parts.append(_comparison_context(view))
    support = [
        '<div class="columns"><section class="panel"><h2>What was checked</h2><dl>'
    ]
    for name, value in view.assurance:
        support.append(f"<dt>{e(name)}</dt><dd>{e(value)}</dd>")
    support.append('</dl></section><section class="panel"><h2>Next steps</h2><ol>')
    support.extend(f"<li>{e(step)}</li>" for step in view.next_steps)
    support.append("</ol></section></div>")
    if multiple_results:
        parts.append(
            '<details class="overview-disclosure"><summary>Compare all metric and scope results</summary><div class="detail-content">'
            + _results_overview(view.metrics)
            + "</div></details>"
        )
        if len(groups) > 1:
            parts.append(
                '<div class="metric-controls"><nav class="metric-navigation" aria-label="Metric results">'
                + "".join(
                    f'<a href="#metric-group-{index}">{e(name)} <span>{len(group)} scope result{"s" if len(group) != 1 else ""}</span></a>'
                    for index, (name, group) in enumerate(groups.items(), 1)
                )
                + '</nav><button class="metric-display-toggle" type="button" hidden>Show all metrics</button></div>'
            )
    if not tabs_enabled:
        parts.append(
            '<div class="section-heading"><h2>Results and requirements</h2></div>'
        )
    for group_index, (metric_name, group) in enumerate(groups.items(), 1):
        if multiple_results:
            parts.append(
                f'<section id="metric-panel-{group_index}" class="metric-group" aria-labelledby="metric-group-{group_index}"><div class="section-heading"><h3 id="metric-group-{group_index}" tabindex="-1">{e(metric_name)}</h3><a href="#results-overview-heading">Back to overview</a></div>'
            )
        for result_index, metric in group:
            heading = (
                f"<h4>{e(metric.scope)}</h4>"
                if multiple_results
                else f"<h3>{e(metric.name)}</h3>"
            )
            context = (
                f"Metric: {e(metric.name)}"
                if multiple_results
                else f"Scope: {e(metric.scope)}"
            )
            parts.append(
                f'<section id="metric-result-{result_index}" tabindex="-1" class="metric {_tone(metric.decision)}"><div class="metric-heading"><div>{heading}<p class="scope">{context}</p></div><span class="badge">{e(decision_label(metric.decision))}</span></div><p class="metric-explanation">{e(metric.explanation)}</p><dl class="values">'
            )
            for name, value in [
                ("Baseline", metric.baseline),
                ("Candidate", metric.candidate),
                ("Change", metric.change),
                ("Observed pairs", metric.count),
            ]:
                parts.append(
                    f'<div class="value"><dt>{name}</dt><dd>{e(value)}</dd></div>'
                )
            parts.append("</dl>")
            if metric.interval:
                parts.append(_interval(metric.interval))
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
                    f'<tr><th scope="row">{e(check.name)}{detail}</th><td>{e(check.observed)}</td><td>{e(check.required)}</td><td class="check-{tone}">{_status(check)}</td></tr>'
                )
            parts.append("</tbody></table></div>")
            if metric.notes:
                parts.append(
                    '<ul class="notes">'
                    + "".join(f"<li>{e(n)}</li>" for n in metric.notes)
                    + "</ul>"
                )
            parts.append("</section>")
        if multiple_results:
            parts.append("</section>")
    parts.extend(support)
    if view.limitations:
        parts.append(
            '<details class="limits"><summary>Scope and limitations</summary><div class="detail-content"><ul>'
            + "".join(f"<li>{e(n)}</li>" for n in view.limitations)
            + "</ul></div></details>"
        )
    parts.append(
        '<div class="section-heading"><h2>Evidence details</h2><p>Identities, methods and exact recorded values</p></div>'
    )
    if view.identity:
        parts.append(
            '<details><summary>Comparison and policy identities</summary><div class="detail-content"><dl>'
        )
        parts.extend(
            f"<dt>{e(k)}</dt><dd><code>{e(v)}</code></dd>" for k, v in view.identity
        )
        parts.append("</dl></div></details>")
    for heading, data in (*view.details, ("Exact comparison data", view.technical)):
        parts.append(
            f'<details><summary>{e(heading)}</summary><div class="detail-content">{_detail_content(data)}</div></details>'
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
                    f'<details><summary>{e(str(item.get("name", "Metric")))} configuration preview</summary><div class="detail-content">{_detail_content(preview)}</div></details>'
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
        value = escape(value).replace("\n", " ").replace("\r", " ")
        for ch in "|\\`*[]()":
            value = value.replace(ch, f"&#{ord(ch)};")
        return re.sub(r"(?<!\w)_|_(?!\w)", "&#95;", value)

    lines = [
        f"# {clean(view.title)}",
        "",
        f"**Recorded policy result: {decision_label(view.decision)}**",
        "",
        clean(view.summary),
        "",
    ]
    if view.subjects or view.context or view.changes:
        lines += ["## What was compared", ""]
        lines.extend(
            f"- **{clean(k)}:** {clean(v)}" for k, v in (*view.subjects, *view.context)
        )
        if view.changes:
            lines += ["", "### Recorded changes", ""]
            lines.extend(f"- {clean(change)}" for change in view.changes)
    lines += ["", "## What was checked", ""]
    lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.assurance)
    for metric in view.metrics:
        lines += [
            "",
            f"## {clean(metric.name)} — {clean(metric.scope)}",
            "",
            f"**{decision_label(metric.decision)}**. {clean(metric.explanation)}",
            "",
            "| Baseline | Candidate | Change | Observed pairs |",
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
        if metric.interval:
            i = metric.interval
            lines += [
                f"{clean(i.label)}: [{number(i.lower)}, {number(i.upper)}] {clean(i.unit)}.",
                "",
            ]
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
        lines.extend(f"- {clean(n)}" for n in metric.notes)
    lines += ["", "## Next steps", ""]
    lines.extend(f"{i}. {clean(s)}" for i, s in enumerate(view.next_steps, 1))
    lines += ["", "## Scope and limitations", ""]
    lines.extend(f"- {clean(s)}" for s in view.limitations)
    lines += ["", "## Evidence identities", ""]
    lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.identity)
    if include_details:
        for heading, data in (*view.details, ("Exact comparison data", view.technical)):
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
