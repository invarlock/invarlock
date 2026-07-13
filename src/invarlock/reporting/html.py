"""
Structured HTML exporter for evaluation reports.

The visible structure comes from the renderer-neutral report outline. Markdown
remains available as a separate export, while this path focuses on a browser
review surface for evaluation.report.json.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from invarlock.public_contracts import REPORT_SCHEMA_VERSION

from .branding import BRAND_NAME, BRAND_TAGLINE, html_brand_mark, version_label
from .report_outline import (
    EvaluationReportOutline,
    ReportFact,
    ReportSection,
    build_evaluation_report_outline,
)
from .report_schema import validate_report

_TONE_CLASS = {
    "pass": "tone-pass",
    "fail": "tone-fail",
    "warn": "tone-warn",
    "warning": "tone-warn",
    "info": "tone-info",
}
_APPENDIX_PREVIEW_LIMIT = 1600


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tone(value: str) -> str:
    return _TONE_CLASS.get(value.lower(), "tone-info")


def _summary_fact(section: ReportSection, label: str) -> ReportFact | None:
    return section.facts_by_label.get(label)


def _fact_value(section: ReportSection, label: str, default: str = "N/A") -> str:
    fact = _summary_fact(section, label)
    return fact.value if fact is not None else default


def _section(outline: EvaluationReportOutline, key: str) -> ReportSection | None:
    for section in outline.sections:
        if section.key == key:
            return section
    return None


def _render_status(value: str, status: str) -> str:
    return f'<span class="status-pill {_tone(status)}">{escape(value)}</span>'


def _render_summary_strip(outline: EvaluationReportOutline) -> str:
    decision = _section(outline, "decision")
    primary = _section(outline, "primary_metric")
    if decision is None or primary is None:
        return ""
    warning_count = _fact_value(decision, "Guard Warnings", "0")
    cells = (
        (
            "Report-local gates",
            outline.overall_status,
            "info",
        ),
        ("Subject", _fact_value(decision, "Model"), "info"),
        ("Baseline", _fact_value(decision, "Baseline"), "info"),
        ("Metric", _fact_value(primary, "Metric"), "info"),
        (
            "Warnings",
            warning_count,
            "warn" if warning_count not in {"0", "N/A"} else "pass",
        ),
    )
    headings = "".join(
        f"<th>{escape(label)}</th>" for label, _value, _tone_name in cells
    )
    values = "".join(
        f'<td><strong class="{_tone(tone_name)}">{escape(value)}</strong></td>'
        for _label, value, tone_name in cells
    )
    return (
        '<section class="summary-strip" aria-label="Report summary">'
        '<table class="summary-table">'
        f"<thead><tr>{headings}</tr></thead>"
        f"<tbody><tr>{values}</tr></tbody>"
        "</table>"
        "</section>"
    )


def _render_nav(outline: EvaluationReportOutline) -> str:
    links: list[str] = []
    for index, section in enumerate(outline.sections):
        current = ' aria-current="true"' if index == 0 else ""
        links.append(
            f'<li><a href="#{escape(section.key)}"{current}>'
            f"{escape(section.title)}</a></li>"
        )
    items = "".join(links)
    return (
        '<nav class="report-outline" aria-label="Report sections">'
        '<p class="outline-eyebrow">Sections</p>'
        f"<ol>{items}</ol>"
        "</nav>"
    )


def _render_source_chips(section: ReportSection) -> str:
    if not section.source_blocks:
        return ""
    sources = ", ".join(
        f"<code>{escape(source)}</code>" for source in section.source_blocks
    )
    return (
        '<p class="source-line" aria-label="JSON source blocks">'
        f"Source blocks: {sources}</p>"
    )


def _render_fact_table(section: ReportSection) -> str:
    include_detail = any(bool(fact.detail) for fact in section.facts)
    rows: list[str] = []
    for fact in section.facts:
        value = _render_status(fact.value, fact.status)
        source = f"<code>{escape(fact.source)}</code>" if fact.source else ""
        detail_cell = (
            f"<td>{escape(fact.detail) if fact.detail else ''}</td>"
            if include_detail
            else ""
        )
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(fact.label)}</th>'
            f"<td>{value}</td>"
            f"{detail_cell}"
            f"<td>{source}</td>"
            "</tr>"
        )
    detail_header = "<th>Detail</th>" if include_detail else ""
    return (
        '<div class="table-wrap">'
        "<table>"
        f"<thead><tr><th>Field</th><th>Value</th>{detail_header}<th>Source</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_guard_warnings(evaluation_report: dict[str, Any]) -> str:
    block = _mapping(evaluation_report.get("guard_warnings"))
    warnings = block.get("warnings")
    if not isinstance(warnings, list) or not warnings:
        return ""
    rows: list[str] = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        location = warning.get("module") or warning.get("family") or "N/A"
        baseline = warning.get("baseline", "N/A")
        subject = warning.get("subject", "N/A")
        if not isinstance(baseline, str):
            baseline = json.dumps(baseline, sort_keys=True, allow_nan=False)
        if not isinstance(subject, str):
            subject = json.dumps(subject, sort_keys=True, allow_nan=False)
        rows.append(
            "<tr>"
            f"<td>{escape(str(warning.get('guard') or 'unknown'))}</td>"
            f"<td>{escape(str(warning.get('kind') or 'warning'))}</td>"
            f"<td>{escape(str(location))}</td>"
            f"<td>{escape(str(baseline))}</td>"
            f"<td>{escape(str(subject))}</td>"
            f"<td>{escape(str(warning.get('policy_gate') or 'N/A'))}</td>"
            f"<td>{escape(str(warning.get('message') or ''))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="detail-block">'
        "<h3>Guard Warnings</h3>"
        "<p>Warnings describe baseline-relative guard movement. They do not change "
        "the policy verdict unless verification is run with a strict warning policy.</p>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>Guard</th><th>Kind</th><th>Location</th><th>Baseline</th>"
        "<th>Subject</th><th>Policy Gate</th><th>Message</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div></div>"
    )


def _preview_json(value: Any) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)
    if len(text) <= _APPENDIX_PREVIEW_LIMIT:
        return text
    return (
        text[:_APPENDIX_PREVIEW_LIMIT].rstrip()
        + "\n...\n[truncated in HTML preview; use evaluation.report.json for full data]"
    )


def _render_appendix_previews(
    evaluation_report: dict[str, Any], section: ReportSection
) -> str:
    if section.key != "technical_appendix":
        return ""
    blocks: list[str] = []
    for source in section.source_blocks:
        value = evaluation_report.get(source)
        if value in ({}, [], None):
            continue
        blocks.append(
            "<details>"
            f"<summary>{escape(source)}</summary>"
            f"<pre>{escape(_preview_json(value))}</pre>"
            "</details>"
        )
    if not blocks:
        return ""
    return (
        '<div class="appendix-previews">'
        "<p>Appendix previews are intentionally capped. The JSON report remains "
        "the complete audit artifact.</p>"
        f"{''.join(blocks)}"
        "</div>"
    )


def _render_section(evaluation_report: dict[str, Any], section: ReportSection) -> str:
    extras = ""
    if section.key == "guard_signals":
        extras = _render_guard_warnings(evaluation_report)
    elif section.key == "technical_appendix":
        extras = _render_appendix_previews(evaluation_report, section)
    return (
        f'<section id="{escape(section.key)}" class="report-section priority-{escape(section.priority)}">'
        "<header>"
        f"<h2>{escape(section.title)}</h2>"
        f"<p>{escape(section.summary)}</p>"
        f"{_render_source_chips(section)}"
        "</header>"
        f"{_render_fact_table(section)}"
        f"{extras}"
        "</section>"
    )


def _render_sections(
    evaluation_report: dict[str, Any], outline: EvaluationReportOutline
) -> str:
    return "".join(
        _render_section(evaluation_report, section) for section in outline.sections
    )


def render_report_html(evaluation_report: dict[str, Any]) -> str:
    """Render an evaluation report as an HTML document with quick navigation."""

    if not validate_report(evaluation_report):
        raise ValueError("Invalid evaluation report structure")
    outline = build_evaluation_report_outline(evaluation_report)
    summary_strip = _render_summary_strip(outline)
    nav = _render_nav(outline)
    sections = _render_sections(evaluation_report, outline)
    shell = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(outline.title)}</title>"
        "<script>(function(){try{var t=localStorage.getItem('invarlock-report-theme');"
        "if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t;}}"
        "catch(e){}})();</script>"
        "<style>"
        ":root,:root[data-theme='light']{color-scheme:light;"
        "--sticky-offset:72px;"
        "--pass:#2f6b4f;--fail:#a02c24;--warn:#95671c;--info:#5c5950;"
        "--ink:#18150f;--muted:#5c5950;--bg:#fcfbf7;--panel:#f4f2eb;"
        "--panel-soft:#ebe8df;--border:#d8d3c5;"
        "--accent:#1f3a7a;--accent-hover:#172c5e;--accent-soft:#ebe8df;"
        "--signal:#8d2433;--color-accent:var(--accent);--brand-mark-accent:var(--accent)}"
        "@media (prefers-color-scheme: dark){"
        ":root:not([data-theme='light']){color-scheme:dark;"
        "--sticky-offset:72px;"
        "--pass:#9ad0a9;--fail:#f19a92;--warn:#e6ba72;--info:#c9c0aa;"
        "--ink:#f4efe3;--muted:#c9c0aa;--bg:#11130f;--panel:#191c16;"
        "--panel-soft:#23271e;--border:#3f4235;"
        "--accent:#9fb7ff;--accent-hover:#c0ccff;--accent-soft:#23271e;"
        "--signal:#eda1ac;--color-accent:var(--accent);--brand-mark-accent:var(--accent)}}"
        ":root[data-theme='dark']{color-scheme:dark;"
        "--sticky-offset:72px;"
        "--pass:#9ad0a9;--fail:#f19a92;--warn:#e6ba72;--info:#c9c0aa;"
        "--ink:#f4efe3;--muted:#c9c0aa;--bg:#11130f;--panel:#191c16;"
        "--panel-soft:#23271e;--border:#3f4235;"
        "--accent:#9fb7ff;--accent-hover:#c0ccff;--accent-soft:#23271e;"
        "--signal:#eda1ac;--color-accent:var(--accent);--brand-mark-accent:var(--accent)}"
        "*{box-sizing:border-box}"
        "body{margin:0;min-height:100vh;padding:34px;color:var(--ink);"
        'font-family:"Sora","Avenir Next","Segoe UI Variable","Segoe UI",ui-sans-serif,system-ui,sans-serif;'
        "line-height:1.55;background:var(--bg)}"
        ".report-shell{max-width:1180px;margin:0 auto}"
        ".report-topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;"
        "justify-content:space-between;gap:16px;margin:0 0 18px 0;padding:10px 0;"
        "background:var(--bg);border-bottom:1px solid var(--border)}"
        ".report-header{margin:0 0 18px 0;padding:0 0 22px 0;border-bottom:1px solid var(--border)}"
        ".eyebrow{margin:0;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:var(--signal)}"
        '.report-header h1{margin:10px 0 6px 0;font-family:"Newsreader",ui-serif,Georgia,serif;'
        "font-size:2rem;line-height:1.15;font-weight:600}"
        ".report-header p{margin:0;max-width:58rem;color:var(--muted)}"
        ".brand-lockup{display:flex;align-items:center;gap:12px}"
        ".brand-mark{display:inline-grid;place-items:center;width:38px;height:38px;"
        "color:var(--ink)}"
        ".brand-mark-svg{display:block;width:38px;height:38px}"
        ".brand-meta{margin-top:10px!important;font-size:0.9rem}"
        ".render-notice{margin-top:12px!important;padding:9px 11px;border-left:3px solid var(--warn);"
        "background:var(--panel);color:var(--warn)!important;font-size:0.88rem;font-weight:700}"
        ".theme-toggle{appearance:none;border:1px solid var(--border);background:transparent;"
        "color:var(--ink);padding:7px 10px;border-radius:2px;font:inherit;font-size:0.86rem;"
        "font-weight:700;line-height:1;cursor:pointer}"
        ".theme-toggle:hover,.theme-toggle:focus-visible{border-color:var(--accent);color:var(--accent);outline:none}"
        ".summary-strip{margin:0 0 24px 0;padding:0 0 16px 0;border-bottom:1px solid var(--border)}"
        ".summary-table{width:100%;border-collapse:collapse;background:transparent;table-layout:fixed}"
        ".summary-table th{padding:10px 14px 6px 0;border-top:0;color:var(--muted);"
        "font-size:0.78rem;letter-spacing:0.06em;text-transform:uppercase;text-align:left}"
        ".summary-table td{padding:0 14px 10px 0;border-top:0;vertical-align:top}"
        ".summary-table th+th,.summary-table td+td{border-left:1px solid var(--border);padding-left:14px}"
        ".summary-table strong{font-size:1rem;line-height:1.3;overflow-wrap:anywhere}"
        ".report-grid{display:grid;grid-template-columns:minmax(180px,220px) minmax(0,1fr);gap:36px;align-items:start}"
        ".report-outline{position:sticky;top:calc(var(--sticky-offset,72px) + 16px);padding:4px 16px 0 0;border-right:1px solid var(--border)}"
        ".outline-eyebrow{margin:0 0 10px 0;font-size:0.78rem;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;color:var(--signal)}"
        ".report-outline ol{list-style:none;padding:0;margin:0;display:grid;gap:6px}"
        ".report-outline a{display:block;padding:5px 0;text-decoration:none;"
        "color:var(--ink);background:transparent}"
        ".report-outline a:hover{color:var(--accent-hover)}"
        ".report-outline a[aria-current='true']{color:var(--accent);font-weight:700;"
        "border-left:2px solid var(--accent);padding-left:8px}"
        ".report-sections{display:block}"
        ".report-section{padding:26px 0;border-top:1px solid var(--border);background:transparent;scroll-margin-top:calc(var(--sticky-offset,72px) + 16px)}"
        ".report-section:first-child{border-top:0;padding-top:0}"
        ".report-section header{margin-bottom:14px}"
        '.report-section h2{margin:0 0 6px 0;font-family:"Newsreader",ui-serif,Georgia,serif;'
        "font-size:1.35rem;line-height:1.2;font-weight:600}"
        ".report-section h3{margin:18px 0 6px 0;font-size:1rem}"
        ".report-section p{margin:0;color:var(--muted)}"
        ".source-line{margin-top:8px!important;font-size:0.82rem}"
        "a{color:var(--accent)}"
        ".table-wrap{width:100%;overflow:auto}"
        "table{border-collapse:collapse;width:100%;margin:0;background:transparent;font-size:0.94rem}"
        "th,td{border-top:1px solid var(--border);padding:9px 10px;text-align:left;vertical-align:top}"
        "thead th{border-top:0;color:var(--muted);font-size:0.78rem;letter-spacing:0.05em;text-transform:uppercase}"
        "tbody th{width:22%;font-weight:700}"
        "td{overflow-wrap:anywhere}"
        'code,pre{font-family:"JetBrains Mono",ui-monospace,monospace}'
        "code{font-size:0.86em;color:var(--ink)}"
        "pre{background:var(--panel-soft)}"
        "pre{padding:12px;overflow:auto;border:1px solid var(--border);font-size:0.84rem;line-height:1.45}"
        ".status-pill{font-weight:750}"
        ".tone-pass{color:var(--pass)}.tone-fail{color:var(--fail)}"
        ".tone-warn{color:var(--warn)}.tone-info{color:var(--info)}"
        ".detail-block{margin-top:18px;padding-top:16px;border-top:1px solid var(--border)}"
        ".appendix-previews{display:grid;gap:10px;margin-top:16px}"
        "details{border-top:1px solid var(--border);background:transparent}"
        "summary{cursor:pointer;padding:10px 0;font-weight:700}"
        "details pre{margin:0 0 10px 0;border:1px solid var(--border);border-radius:0}"
        "@media (max-width:980px){.summary-table{table-layout:auto}"
        ".report-grid{grid-template-columns:1fr}.report-outline{position:static;border-right:0;border-bottom:1px solid var(--border);padding:0 0 14px 0}}"
        "@media (max-width:720px){body{padding:16px}.report-topbar{top:0}.report-header{padding:0 0 18px 0}"
        ".report-topbar{align-items:flex-start}.report-header h1{font-size:1.7rem}"
        ".summary-table thead{display:none}.summary-table,.summary-table tbody,.summary-table tr,.summary-table td{display:block;width:100%}"
        ".summary-table td{padding:8px 0;border-left:0!important;border-top:1px solid var(--border)}"
        ".summary-table td::before{display:block;margin-bottom:2px;color:var(--muted);font-size:0.72rem;"
        "font-weight:700;letter-spacing:0.06em;text-transform:uppercase}"
        ".summary-table td:nth-child(1)::before{content:'Report-local gates'}"
        ".summary-table td:nth-child(2)::before{content:'Subject'}"
        ".summary-table td:nth-child(3)::before{content:'Baseline'}"
        ".summary-table td:nth-child(4)::before{content:'Metric'}"
        ".summary-table td:nth-child(5)::before{content:'Warnings'}"
        ".report-section{padding:16px}th,td{padding:8px}tbody th{width:auto}}"
        "@media print{body{background:#fff;padding:0}.theme-toggle{display:none}.report-grid{grid-template-columns:1fr}"
        ".report-outline{display:none}a{color:inherit;text-decoration:none}}"
        "</style>"
        '</head><body><div class="report-shell">'
        '<div class="report-topbar">'
        f'<div class="brand-lockup"><span class="brand-mark">{html_brand_mark()}</span>'
        '<p class="eyebrow">InvarLock</p></div>'
        '<button class="theme-toggle" type="button" data-theme-toggle aria-pressed="false" aria-label="Toggle light and dark theme">Light/Dark</button>'
        "</div>"
        '<header class="report-header">'
        f"<h1>{escape(outline.title)}</h1>"
        f"<p>{escape(BRAND_TAGLINE)}</p>"
        f'<p class="brand-meta">{escape(BRAND_NAME)} {escape(version_label())} · schema {escape(REPORT_SCHEMA_VERSION)} · renderer outline</p>'
        '<p class="render-notice">REPORT-LOCAL / UNVERIFIED RENDER — this renderer does not independently verify report bytes, provenance, policy inputs, or declared assurance fields.</p>'
        "</header>"
        f"{summary_strip}"
        '<div class="report-grid">'
        f"{nav}"
        f'<main class="report-sections">{sections}</main>'
        "</div>"
        "</div>"
        "<script>(function(){var root=document.documentElement;"
        "var button=document.querySelector('[data-theme-toggle]');"
        "if(!button){return;}function systemTheme(){return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}"
        "function activeTheme(){return root.dataset.theme||systemTheme();}"
        "function sync(){var current=activeTheme();button.textContent=current==='dark'?'Light':'Dark';"
        "button.setAttribute('aria-pressed',current==='dark'?'true':'false');}"
        "button.addEventListener('click',function(){var next=activeTheme()==='dark'?'light':'dark';"
        "root.dataset.theme=next;try{localStorage.setItem('invarlock-report-theme',next);}catch(e){}sync();});"
        "sync();})();"
        "(function(){var root=document.documentElement;var topbar=document.querySelector('.report-topbar');"
        "function offset(){var value=topbar?Math.ceil(topbar.getBoundingClientRect().height):72;"
        "root.style.setProperty('--sticky-offset',value+'px');return value;}"
        "offset();window.addEventListener('resize',offset);})();"
        "(function(){var root=document.documentElement;var topbar=document.querySelector('.report-topbar');"
        "var links=[].slice.call(document.querySelectorAll('.report-outline a[href^=\"#\"]'));"
        "if(!links.length){return;}var items=links.map(function(link){var id=link.getAttribute('href').slice(1);"
        "return{link:link,section:document.getElementById(id),id:id};}).filter(function(item){return item.section;});"
        "function setActive(id){items.forEach(function(item){if(item.id===id){item.link.setAttribute('aria-current','true');}"
        "else{item.link.removeAttribute('aria-current');}});}function stickyOffset(){"
        "var css=parseFloat(getComputedStyle(root).getPropertyValue('--sticky-offset'));"
        "if(Number.isFinite(css)){return css;}return topbar?topbar.getBoundingClientRect().height:72;}"
        "function updateActive(){var y=window.scrollY+stickyOffset()+18;"
        "var active=items[0];items.forEach(function(item){if(item.section.offsetTop<=y){active=item;}});"
        "if(active){setActive(active.id);}}window.addEventListener('scroll',updateActive,{passive:true});"
        "window.addEventListener('resize',updateActive);window.addEventListener('hashchange',updateActive);"
        "updateActive();})();</script>"
        "</body></html>"
    )
    return shell


__all__ = ["render_report_html"]
