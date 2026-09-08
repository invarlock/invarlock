"""Shared, non-authoritative presentation of explicit comparison facts.

Builders retain each evidence family's own validation and trust rules. These
views never authorize a signer, replay a score, or change a stored decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import escape
from typing import Any


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
    # Normalize before subtraction: finite policy bounds can span nearly the
    # complete float range even when the observed interval is small.
    scale = max(1.0, *(abs(value) for value in values))
    low, high = min(values) / scale, max(values) / scale
    padding = (high - low) * 0.12 or max(abs(low) * 0.1, 0.01)
    low, high = low - padding, high + padding

    def x(value: float) -> float:
        return 30 + (value / scale - low) / (high - low) * 580

    description = f"{view.label}: {number(view.lower)} to {number(view.upper)} {view.unit}. Estimate {number(view.estimate)} {view.unit}."
    threshold = ""
    if view.threshold is not None:
        description += f" Policy threshold {number(view.threshold)} {view.unit}."
        threshold = f'<line class="threshold" x1="{x(view.threshold):.2f}" x2="{x(view.threshold):.2f}" y1="12" y2="54"/>'
    return (
        '<figure class="interval"><svg viewBox="0 0 640 68" role="img" aria-label="'
        + escape(description, quote=True)
        + '"><line class="axis" x1="30" x2="610" y1="34" y2="34"/>'
        + threshold
        + f'<line class="range" x1="{x(view.lower):.2f}" x2="{x(view.upper):.2f}" y1="34" y2="34"/>'
        + f'<circle class="estimate" cx="{x(view.estimate):.2f}" cy="34" r="6"/>'
        + "</svg><figcaption>"
        + escape(description)
        + (
            ' <span class="legend">Dashed line: policy threshold.</span>'
            if threshold
            else ""
        )
        + "</figcaption></figure>"
    )


_CSS = """
:root{color-scheme:light;--ink:#172a35;--muted:#526773;--line:#d8e3e8;--paper:#fff;--canvas:#f2f6f8;--teal:#086756;--red:#a32639;--amber:#79530b}
*{box-sizing:border-box}body{margin:0;overflow-wrap:anywhere;background:var(--canvas);color:var(--ink);font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:#145d7b}main{max-width:1120px;margin:auto;padding:36px 28px 60px}.brand{display:flex;align-items:center;gap:10px;font-weight:750;letter-spacing:.01em}.mark{display:grid;place-items:center;width:30px;height:30px;border-radius:8px;background:var(--ink);color:white;font-size:18px}.family{margin-left:auto;color:var(--muted);font-size:12px;text-align:right}
.hero{margin:26px 0 22px;padding:30px;border:1px solid var(--line);border-radius:16px;background:var(--paper);border-top:4px solid var(--teal)}.hero.fail{border-top-color:var(--red)}.hero.insufficient{border-top-color:var(--amber)}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:750;color:var(--muted);margin:0 0 7px}h1{font-size:36px;line-height:1.15;letter-spacing:-.03em;margin:0 0 14px}h2{font-size:21px;letter-spacing:-.02em;line-height:1.3;margin:0}h3{font-size:16px;margin:0 0 10px}.hero>p{max-width:80ch;margin:0 0 18px}.summary-row{display:flex;flex-wrap:wrap;gap:8px}.pill{font-size:12px;font-weight:650;padding:4px 10px;border-radius:6px;background:#eef3f6;color:var(--muted)}.badge{display:inline-block;font-size:12px;font-weight:700;border:1px solid currentColor;border-radius:5px;padding:2px 8px;white-space:nowrap}.pass .badge,.badge.pass{color:var(--teal);background:#eef8f4}.fail .badge,.badge.fail{color:var(--red);background:#fff1f2}.insufficient .badge,.badge.insufficient{color:var(--amber);background:#fff8e6}
.subjects{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0 0 24px}.subjects div{min-width:0}.subjects dt{font-size:11px;font-weight:650;color:var(--muted)}.subjects dd{margin:3px 0 0;font-size:13px;overflow-wrap:anywhere}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:30px 0 14px}.section-heading p{font-size:12px;margin:0;color:var(--muted)}.metric{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin:0 0 16px;padding:24px;break-inside:avoid}.metric-heading>div{min-width:0}.metric-heading>.badge{flex-shrink:0;max-width:45%}.metric-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.scope{font-size:12px;color:var(--muted);margin:4px 0}.metric-explanation{margin:12px 0 18px}.values{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:8px;background:#f8fafb;overflow:hidden}.value{padding:12px 16px;border-right:1px solid var(--line)}.value:last-child{border:0}.value dt{font-size:11px;color:var(--muted)}.value dd{margin:3px 0 0;font-size:22px;font-weight:650;letter-spacing:-.02em;overflow-wrap:anywhere}.interval{margin:14px 0}.interval svg{display:block;width:100%;max-height:68px}.interval figcaption{font-size:12px;color:var(--muted)}.axis{stroke:var(--line);stroke-width:2}.range{stroke:#277b91;stroke-width:7;stroke-linecap:round}.estimate{fill:var(--ink);stroke:white;stroke-width:2}.threshold{stroke:var(--amber);stroke-width:2;stroke-dasharray:4 4}.legend{display:block}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}caption{text-align:left;font-weight:650;padding:0 0 8px}th{text-align:left;color:var(--muted);font-size:11px;font-weight:650}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th:first-child,td:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}tbody tr:last-child td{border-bottom:0}.check-detail{display:block;color:var(--muted);font-size:12px}.check-fail{color:var(--red);font-weight:650}.check-pass{color:var(--teal)}.check-unknown{color:var(--amber)}.notes{font-size:12px;color:var(--muted);padding-left:18px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:22px}.panel h2{font-size:16px;margin-bottom:12px}.metric h3{font-size:21px;line-height:1.3;margin:0}.limits h2{font-size:16px;margin-bottom:8px}.panel dl{margin:0}.panel dt{font-size:12px;font-weight:650;margin-top:12px}.panel dt:first-child{margin-top:0}.panel dd{margin:3px 0 0;color:var(--muted);font-size:13px;overflow-wrap:anywhere}.panel ol,.panel ul{padding-left:20px;font-size:13px;margin:0}.panel li+li{margin-top:9px}.limits{color:var(--muted);font-size:13px;margin:22px 0}.limits li+li{margin-top:6px}details{border:1px solid var(--line);border-radius:8px;background:var(--paper);margin:10px 0}summary{padding:14px 18px;font-size:13px;font-weight:650;cursor:pointer}summary:focus-visible{outline:3px solid #277b91;outline-offset:3px}details .detail-content{padding:0 18px 18px}pre{font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;border-radius:6px;padding:14px;max-height:480px;overflow:auto}code{overflow-wrap:anywhere}.footer{color:var(--muted);font-size:11px;margin-top:26px;border-top:1px solid var(--line);padding-top:16px}
@media(max-width:650px){main{padding:20px 14px 40px}.hero{padding:22px 18px}h1{font-size:29px}.metric{padding:18px}.values{grid-template-columns:repeat(2,minmax(0,1fr))}.value:nth-child(2){border-right:0}.value:nth-child(-n+2){border-bottom:1px solid var(--line)}.value dd{font-size:20px}.columns,.subjects{grid-template-columns:1fr}.family{max-width:160px}.section-heading{display:block}.section-heading p{margin-top:5px}.metric-heading{gap:8px;flex-direction:column}.badge{white-space:normal;font-size:11px}th,td{padding:9px 7px}}
@media print{body{background:white;font-size:10pt}main{max-width:none;padding:0}.hero{margin-top:16px;padding:20px}h1{font-size:25pt}.metric,.panel{padding:15px}.columns{display:block}.panel{margin:12px 0}details{break-inside:avoid}pre{max-height:none}.footer{font-size:9pt}.scroll{overflow:visible}a{color:inherit}}
"""


def render_html(view: ReportView) -> str:
    """Render only escaped view fields; no scripts or network dependencies."""
    e = escape
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        f"<title>{e(view.title)}</title><style>{_CSS}</style></head><body><main>",
        f'<header class="brand"><span class="mark" aria-hidden="true">↔</span> InvarLock<span class="family">{e(view.family)}</span></header>',
        f'<section class="hero {_tone(view.decision)}" aria-labelledby="decision"><p class="eyebrow">Recorded policy result</p><h1 id="decision">{e(decision_label(view.decision))}</h1><p>{e(view.summary)}</p><div class="summary-row">',
        f'<span class="pill">{len(view.metrics)} metric / scope result{"s" if len(view.metrics) != 1 else ""}</span>',
        f'<span class="pill">Original decision: {e(view.decision)}</span></div></section>',
    ]
    if view.subjects:
        parts.append(
            '<dl class="subjects">'
            + "".join(
                f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in view.subjects
            )
            + "</dl>"
        )
    support = [
        '<div class="columns"><section class="panel"><h2>What was checked</h2><dl>'
    ]
    for name, value in view.assurance:
        support.append(f"<dt>{e(name)}</dt><dd>{e(value)}</dd>")
    support.append('</dl></section><section class="panel"><h2>Next steps</h2><ol>')
    support.extend(f"<li>{e(step)}</li>" for step in view.next_steps)
    support.append("</ol></section></div>")
    parts.append(
        '<div class="section-heading"><h2>Results and requirements</h2><p>Observed values alongside the configured checks</p></div>'
    )
    for metric in view.metrics:
        parts.append(
            f'<section class="metric {_tone(metric.decision)}"><div class="metric-heading"><div><h3>{e(metric.name)}</h3><p class="scope">Scope: {e(metric.scope)}</p></div><span class="badge">{e(decision_label(metric.decision))}</span></div><p class="metric-explanation">{e(metric.explanation)}</p><dl class="values">'
        )
        for name, value in [
            ("Baseline", metric.baseline),
            ("Candidate", metric.candidate),
            ("Change", metric.change),
            ("Observed pairs", metric.count),
        ]:
            parts.append(f'<div class="value"><dt>{name}</dt><dd>{e(value)}</dd></div>')
        parts.append("</dl>")
        if metric.interval:
            parts.append(_interval(metric.interval))
        parts.append(
            '<div class="scroll"><table><caption>Decision checks</caption><thead><tr><th scope="col">Check</th><th scope="col">Observed</th><th scope="col">Required</th><th scope="col">Result</th></tr></thead><tbody>'
        )
        for check in metric.checks:
            tone = (
                "pass"
                if check.passed is True
                else "fail"
                if check.passed is False
                else "unknown"
            )
            parts.append(
                f'<tr><th scope="row">{e(check.name)}<span class="check-detail">{e(check.explanation)}</span></th><td>{e(check.observed)}</td><td>{e(check.required)}</td><td class="check-{tone}">{_status(check)}</td></tr>'
            )
        parts.append("</tbody></table></div>")
        if metric.notes:
            parts.append(
                '<ul class="notes">'
                + "".join(f"<li>{e(n)}</li>" for n in metric.notes)
                + "</ul>"
            )
        parts.append("</section>")
    parts.extend(support)
    if view.limitations:
        parts.append(
            '<section class="limits"><h2>Scope and limitations</h2><ul>'
            + "".join(f"<li>{e(n)}</li>" for n in view.limitations)
            + "</ul></section>"
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
            f'<details><summary>{e(heading)}</summary><div class="detail-content"><pre>{e(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))}</pre></div></details>'
        )
    parts.append(
        '<footer class="footer">InvarLock · Human-readable evidence report. Display values are rounded; exact values remain in the evidence. This HTML is a presentation, not an independent acceptance receipt.</footer></main></body></html>\n'
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
        "## What was checked",
        "",
    ]
    lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.assurance)
    if view.subjects:
        lines += ["", "## Compared artifacts", ""]
        lines.extend(f"- **{clean(k)}:** {clean(v)}" for k, v in view.subjects)
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
                json.dumps(
                    data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
                ).replace("`", "\\u0060"),
                "```",
            ]
    return "\n".join(lines) + "\n"
