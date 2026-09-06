"""Static, escaped reports for people and standard CI consumers."""

from __future__ import annotations

import html
from typing import Any, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from invarlock.pipeline.contracts import validate


def _interval_text(metric: dict[str, Any]) -> str:
    interval = metric["interval"]
    if interval is None:
        return "unavailable"
    return f"[{interval['lower']:.6g}, {interval['upper']:.6g}]"


def render_markdown(comparison: dict[str, Any]) -> str:
    validate(comparison, "comparison")

    def clean(value: Any) -> str:
        escaped = (
            html.escape(str(value))
            .replace("|", "&#124;")
            .replace("\n", " ")
            .replace("\r", " ")
        )
        for character in ("\\", "`", "*", "_", "[", "]", "(", ")"):
            escaped = escaped.replace(character, f"&#{ord(character)};")
        return escaped

    lines = [
        "# Release comparison",
        "",
        f"Decision: **{comparison['decision']}**",
        "",
        "| Metric | Slice | Count | Baseline | Candidate | Delta | 95% interval | Unit | Scoring | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in comparison["metrics"]:
        lines.append(
            "| "
            + " | ".join(
                clean(_interval_text(m) if k == "interval" else m[k])
                for k in (
                    "name",
                    "slice",
                    "count",
                    "baseline_mean",
                    "candidate_mean",
                    "delta",
                    "interval",
                    "unit",
                    "scoring_assurance",
                    "decision",
                )
            )
            + " |"
        )
    lines += [
        "",
        *[
            f"- {clean(m['name'])} / {clean(m['slice'])}: {clean('; '.join(m['reasons']))}"
            for m in comparison["metrics"]
            if m["reasons"]
        ],
        "",
        *[clean(value) for value in comparison["limitations"]],
        "",
    ]
    return "\n".join(lines)


def render_html(comparison: dict[str, Any]) -> str:
    validate(comparison, "comparison")
    rows = []
    for m in comparison["metrics"]:
        cells = [
            _interval_text(m) if k == "interval" else m[k]
            for k in (
                "name",
                "slice",
                "count",
                "baseline_mean",
                "candidate_mean",
                "delta",
                "interval",
                "unit",
                "scoring_assurance",
                "decision",
            )
        ]
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in cells) + "</tr>"
        )
    details = "".join(
        f"<li>{html.escape(m['name'])} / {html.escape(m['slice'])}: {html.escape('; '.join(m['reasons']))}</li>"
        for m in comparison["metrics"]
        if m["reasons"]
    )
    limitations = "".join(
        f"<li>{html.escape(v)}</li>" for v in comparison["limitations"]
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Release comparison</title><style>body{font:16px system-ui;line-height:1.5;max-width:1100px;margin:32px auto;padding:0 20px;color:#142f44}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd6dd}th{background:#eaf0f4}.scroll{overflow:auto}code{overflow-wrap:anywhere}</style><main><h1>Release comparison</h1>"""
        + f"<p>Decision: <strong>{comparison['decision']}</strong></p><p>This report presents a comparison. Signature authentication requires the separate verify command.</p><div class=scroll><table><caption>Policy results</caption><thead><tr>"
        + "".join(
            f"<th scope=col>{v}</th>"
            for v in (
                "Metric",
                "Slice",
                "Count",
                "Baseline",
                "Candidate",
                "Delta",
                "95% interval",
                "Unit",
                "Scoring",
                "Decision",
            )
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + f"</tbody></table></div><ul>{details}</ul><h2>Interpretation</h2><ul>{limitations}</ul></main></html>"
    )


def render_junit(comparison: dict[str, Any]) -> bytes:
    validate(comparison, "comparison")
    metrics = comparison["metrics"]
    root = Element(
        "testsuite",
        name="InvarLock release comparison",
        tests=str(len(metrics)),
        failures=str(sum(m["decision"] == "regression" for m in metrics)),
        errors=str(sum(m["decision"] == "insufficient_evidence" for m in metrics)),
    )
    for metric in metrics:
        case = SubElement(
            root, "testcase", name=metric["name"], classname=metric["slice"]
        )
        if metric["decision"] != "pass":
            SubElement(
                case,
                "failure" if metric["decision"] == "regression" else "error",
                message="; ".join(metric["reasons"]),
            )
    return cast(bytes, tostring(root, encoding="utf-8", xml_declaration=True))
