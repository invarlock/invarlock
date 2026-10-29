"""Decision briefs and recipient instructions kept outside signed evidence."""

from __future__ import annotations

import re

from invarlock.report_presentation import (
    MetricView,
    ReportView,
    decision_label,
    escape,
)
from invarlock.report_reading import (
    observed_change,
    range_reading,
    score_caption,
    score_scale,
    suggested_action,
)

_STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f1f3ed;color:#20251f;font:16px/1.5 system-ui,sans-serif}main{max-width:960px;margin:24px auto;padding:32px;background:white;border-top:4px solid #20251f}h1{font-size:30px;line-height:1.15;margin:8px 0 16px}h2{font-size:18px;margin:18px 0 8px}p{margin:8px 0}.brand,.scope,footer{font-size:12px;color:#586054}.brand{font-weight:700}a{color:#145d7b}a:focus-visible{outline:3px solid #145d7b;outline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;margin:14px 0}.decision{border-left:4px solid #586054;padding:8px 14px;background:#f3f5f0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}table{border-collapse:collapse;width:100%;font-size:13px;table-layout:fixed}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #d9dfd3;padding:7px;overflow-wrap:anywhere}pre{padding:14px;background:#f3f5f0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}p,li,code{overflow-wrap:anywhere}.identity{font:11px/1.4 ui-monospace,monospace;overflow-wrap:anywhere}footer{margin-top:20px;border-top:1px solid #d9dfd3;padding-top:10px}.context{font-size:14px}.scope{font-size:12px}.scroll{overflow-x:auto}
.score-card{border-top:1px solid #d9dfd3;padding:10px 0}.score-card h3{font-size:16px;margin:0 0 5px;overflow-wrap:anywhere}.score-row{display:grid;grid-template-columns:75px minmax(50px,1fr) 230px;gap:10px;align-items:center;font-size:14px}.score-row strong{text-align:right}.track{height:12px;background:#f1f3ed;border-bottom:1px solid #c5cfc0}.bar{display:block;height:100%;background:#73816d}.bar.candidate{background:#185c78}.score-detail{font-size:12px;font-weight:400;color:#586054}.score-card .scope{margin:3px 0}.result-line{margin-top:8px}summary{cursor:pointer}summary:focus-visible{outline:3px solid #145d7b}
@media(max-width:600px){main{padding:18px;margin:0}.grid{display:block}.score-row{grid-template-columns:65px minmax(20px,1fr) 95px;font-size:12px;gap:6px}.score-detail{display:block;font-size:11px}h1{font-size:25px}table{font-size:12px}th,td{padding:5px}}
@page{margin:14mm;@bottom-right{content:"Page " counter(page) " of " counter(pages);font:8pt sans-serif}}
@media print{.grid{break-inside:avoid}.score-card{padding:5px 0;break-inside:avoid}.score-card h3{font-size:10pt}.score-row{font-size:8pt}.score-detail{font-size:8pt;display:inline}.track{height:9px}.bar{print-color-adjust:exact;-webkit-print-color-adjust:exact}.score-card p{margin:3px 0}body{background:white;font-size:9pt;line-height:1.35}p{margin:5px 0}main{padding:0;margin:0;max-width:none;border:0}h1{font-size:20pt;margin-bottom:8px}h2{font-size:11pt;margin-top:9px;margin-bottom:5px}.context{font-size:9pt}.scope,.brand,footer{font-size:8pt}nav{font-size:8pt;margin:8px 0}table{font-size:8pt}th,td{padding:5px}.decision{padding:7px 10px}h1,h2,thead{break-after:avoid}tr{break-inside:avoid}a{color:inherit}.identity{font-size:7pt}footer{margin-top:10px;padding-top:6px}.scroll{overflow:visible}}
"""


def _document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body><main>{body}</main></body></html>\n"
    )


def _excerpt(value: str, limit: int = 360) -> str:
    return (
        value
        if len(value) <= limit
        else value[:limit].rsplit(" ", 1)[0] + "… (full text in review)"
    )


def _score_chart(metric: MetricView) -> str:
    scale = score_scale(metric)
    if scale is None:
        return ""
    scores = metric.scores
    assert scores is not None
    rows = []
    for label, raw, displayed, detail in (
        ("Baseline", scores.baseline, metric.baseline, metric.baseline_detail),
        ("Candidate", scores.candidate, metric.candidate, metric.candidate_detail),
    ):
        assert raw is not None
        rows.append(
            f'<div class="score-row"><span>{label}</span><span class="track" aria-hidden="true"><span class="bar {label.lower()}" style="width:{raw / scale * 100:.6f}%"></span></span>'
            f"<strong>{escape(displayed)}"
            + (
                f' <span class="score-detail">· {escape(detail)}</span>'
                if detail
                else ""
            )
            + "</strong></div>"
        )
    axis = (
        "0–100%"
        if scores.kind == "accuracy"
        else "0–1"
        if scores.kind == "rubric"
        else f"0–{scale:.4g} nats / byte"
    )
    return (
        '<div class="score-chart">'
        + "".join(rows)
        + f'<p class="scope">Observed scores · shared bar scale: {axis}.</p></div>'
    )


def brief(view: ReportView, *, digest: str, receipt: bool, html: bool) -> str:
    """One ordered content model, with full conditions and stable result indices."""
    from invarlock.report_facts import generated_facts

    review = view.review
    assert review is not None
    facts = view.critical_facts or generated_facts(view)
    recipient = dict(facts).get(
        "Recipient verification", "Not performed for this recipient."
    )
    authenticated = not recipient.startswith("Not performed")
    receipt_note = (
        "The retained receipt snapshot was checked against independent expectations. See verification-snapshot.json."
        if authenticated
        else "Not authenticated by packaging. The attached historical receipt has not been checked for this recipient."
        if receipt
        else "No historical receipt supplied."
    )
    indexed = list(enumerate(view.metrics, 1))
    blockers = [
        (i, m)
        for i, m in indexed
        if m.decision != "pass" or any(c.passed is not True for c in m.checks)
    ]
    blockers.sort(key=lambda pair: not pair[1].required)
    others = [(i, m) for i, m in indexed if (i, m) not in blockers]
    metrics = blockers + others[: max(0, 6 - len(blockers))]
    more = (
        f"Showing {len(metrics)} of {len(view.metrics)} measure / test-group results. All results and checks are in the technical appendix."
        if len(metrics) < len(view.metrics)
        else ""
    )
    intro = "The review author supplies the decision and interpretation. The recorded test result is separate from release approval."
    summary = review.summary or (
        (
            "How to read the result",
            "Baseline is the reference version; candidate is the version being assessed. Scores, requirements and uncertainty are explained below.",
        ),
        (
            "Application meaning",
            "An application-specific interpretation was not supplied. Ask the review owner how these results apply to the intended use.",
        ),
    )
    attribution = (
        "Summary by the review author."
        if review.summary
        else "Reading guide generated from the recorded comparison."
    )
    # Owner rows already appear in structured facts; keep legacy context intact below.
    fact_keys = {k for k, _ in facts}
    context = tuple((k, v) for k, v in review.context if k not in fact_keys)
    limits = list(view.limitations)
    limits += [
        "Results cover the recorded tests. Workload performance and safety need separate evidence."
    ]
    if any(m.scores and m.scores.kind == "rubric" for m in view.metrics):
        limits.append(
            "Judge ratings do not establish correctness. Repeated ratings are not extra independent test units."
        )
    if len(view.metrics) > 1:
        limits.append(
            "Overlapping groups cannot be added; separate ranges provide no combined confidence guarantee."
        )
    # Put trust, blockers and requirements before unbounded authored narrative.
    essentials = tuple((k, v) for k, v in facts if k != "Evidence limit")
    owner = (
        ("Owner decision", review.decision),
        ("Recommendation", review.recommendation),
        ("Next action", review.next_action),
    ) + context

    def clean(value: str) -> str:
        value = escape(value, quote=False).replace("\n", " ").replace("\r", " ")
        for ch in "|\\`*[]()":
            value = value.replace(ch, f"&#{ord(ch)};")
        return re.sub(r"(?<!\w)_|_(?!\w)", "&#95;", value)

    if not html:
        lines = [
            f"# {clean(view.title)}",
            "",
            "Decision brief",
            "",
            clean(review.label),
            "",
            clean(review.byline),
            "",
            intro,
            "",
            "## Generated facts",
            "",
        ]
        lines += [f"- **{clean(k)}:** {clean(v)}" for k, v in essentials]
        if view.decision_scope:
            lines += ["", clean(view.decision_scope)]
        lines += ["", "## Intended use and owner decision", ""]
        lines += [f"- **{clean(k)}:** {clean(v)}" for k, v in owner]
        lines += ["", "## Results at a glance", "", more, ""]
        for index, m in metrics:
            lines += [
                f"### {clean(m.display_name)} / {clean(m.display_scope)}",
                "",
                clean(score_caption(m)),
                "",
                f"- Baseline: {clean(m.baseline)}; {clean(m.baseline_detail)}",
                f"- Candidate: {clean(m.candidate)}; {clean(m.candidate_detail)}",
                f"- Change: {clean(observed_change(m))}; {clean(decision_label(m.decision))}.",
                f"- {clean(m.count_label)}: {clean(m.count)}. {clean(m.count_detail)}",
                "",
                clean(range_reading(m)),
                "",
            ]
            if m.decision != "pass":
                lines += [f"**Suggested follow-up:** {clean(suggested_action(m))}", ""]
            lines += [
                f"[See result {index}: evidence and checks](technical-report.html#metric-result-{index})",
                "",
            ]
        lines += ["## Decision limits", ""] + [clean(v) for v in limits]
        lines += [
            "",
            "## Recipient check",
            "",
            clean(recipient),
            "",
            receipt_note,
            "",
            "## In plain language",
            "",
            attribution,
            "",
        ]
        lines += [f"- **{clean(k)}:** {clean(v)}" for k, v in summary]
        lines += [
            "",
            f"Evidence statement identity (not approval): {digest}",
            "",
            " · ".join(f"[{k}]({v})" for k, v in view.handoff_links),
            "",
        ]
        return "\n".join(lines)
    e = escape

    def rows(values: tuple[tuple[str, str], ...]) -> str:
        return "".join(
            f'<p class="context"><strong>{e(k)}.</strong> {e(v)}</p>' for k, v in values
        )

    navigation = "".join(
        f'<a href="{e(path)}">{e(label)}</a>' for label, path in view.handoff_links
    )
    cards = []
    for index, m in metrics:
        chart = (
            _score_chart(m)
            or f"<p><strong>Baseline:</strong> {e(m.baseline)} · <strong>Candidate:</strong> {e(m.candidate)}</p>"
        )
        cards.append(
            '<article class="score-card">'
            f'<h3>{e(m.display_name)} <span class="scope">· {e(m.display_scope)}</span></h3>'
            f'<p class="scope">{e(score_caption(m))}</p>'
            + chart
            + f"<p><strong>{e(decision_label(m.decision))}.</strong> Change: {e(observed_change(m))} · {e(m.count_label)}: {e(m.count)}. {e(m.count_detail)}</p>"
            f"<p>{e(range_reading(m))}</p>"
            + (
                f"<p><strong>Suggested follow-up.</strong> {e(suggested_action(m))}</p>"
                if m.decision != "pass"
                else ""
            )
            + f'<p class="scope"><a href="technical-report.html#metric-result-{index}">See result {index}: evidence and checks</a></p></article>'
        )
    body = (
        f'<div class="brand">InvarLock · Decision brief</div><h1>{e(view.title)}</h1>'
        f'<p class="scope">{e(review.label)} · {e(review.byline)}</p><p class="scope">{intro}</p>'
        f'<nav aria-label="Review package">{navigation}</nav>'
        "<section><h2>Generated facts</h2>"
        + rows(essentials)
        + (f"<p>{e(view.decision_scope)}</p>" if view.decision_scope else "")
        + "</section><section><h2>Intended use and owner decision</h2>"
        + rows(owner)
        + "</section><section><h2>Results at a glance</h2>"
        + (f'<p class="scope">{more}</p>' if more else "")
        + "".join(cards)
        + "</section><section><h2>Decision limits</h2>"
        + "".join(f"<p>{e(v)}</p>" for v in limits)
        + f"</section><section><h2>Recipient check</h2><p>{e(recipient)}</p><p>{e(receipt_note)}</p></section>"
        '<section class="reader-summary"><h2>In plain language</h2>'
        + f'<p class="scope">{attribution}</p>'
        + rows(summary)
        + "</section><footer>Evidence statement identity (not approval): "
        + f'<span class="identity">{e(digest)}</span></footer>'
    )
    return _document(view.title, body)


def verification_guide(kind: str, *, version: str, decision: str, html: bool) -> str:
    profiles = {
        "runtime": "Native trust-inputs profile: approved policy, evidence signer, baseline and subject artifact identities, schedule, runtime identities, and request digest when required. The profile contains your verifier identity and private-key path.",
        "captured": "Captured trust-inputs-v2 profile: approved policy, evidence signer, complete baseline and subject run digests and request digest. The profile contains your verifier identity and private-key path.",
        "judge": "Judge recipient policy: independently approved signer, intended subject and all required plan, measurement, run and policy bindings. Supply your own verifier signing key and identity separately below.",
        "evidence_set": "Composition recipient policy: approved index digest, shared-input bindings, and independently maintained captured and judge recipient profiles. The captured profile supplies the verifier key and identity for the component receipts.",
    }
    setup = f'python -m pip install "invarlock=={version}"\ninvarlock --version'
    command = """(
  : "${TRUST_PROFILE:?Set an absolute path to your independently approved trust profile}"
  : "${RESULT_DIR:?Set a new absolute result directory outside this package}"
"""
    if kind == "judge":
        command += """  : "${VERIFIER_KEY:?Set the absolute path to your private verifier key outside this package}"
  : "${VERIFIER_IDENTITY:?Set your approved verifier identity}"
"""
    command += '''  mkdir "$RESULT_DIR" || exit 1
  verify_status=0
  invarlock verify ./evidence --trust-profile "$TRUST_PROFILE" \\
    --receipt "$RESULT_DIR/verification.receipt.json"'''
    if kind == "judge":
        command += ''' \\
    --verifier-signing-key "$VERIFIER_KEY" --verifier-identity "$VERIFIER_IDENTITY"'''
    command += """ \\
    --json > "$RESULT_DIR/verification.result.json" || verify_status=$?
  printf 'Verification exit code: %s\\n' "$verify_status"
  exit "$verify_status"
)"""
    sections = [
        (
            "1. Install the matching core",
            "Use Python 3.12 or later. Install the matching released core using the command below, or an independently approved candidate wheel. A development version string alone does not identify a build: retain the supplied wheel digest separately. No model weights, evaluator SDK, provider account or source checkout is required for this retained-evidence check.",
            setup,
        ),
        (
            "2. Obtain independent inputs",
            "Keep trust inputs and private keys outside this package. Obtain them from your approved source, not by copying assertions from the submitted evidence or its unsigned inventories. "
            + profiles[kind]
            + " Set TRUST_PROFILE to that profile's absolute path and RESULT_DIR to a fresh absolute directory outside this package. Do not use an evidence signing key as the verifier key.",
            None,
        ),
        (
            "3. Verify from the extracted directory",
            "Run the following from the directory containing brief.html and evidence/. It writes a new result and receipt outside the evidence. Existing result directories are refused. Copy the result files into your review record after inspecting them; packaging does not discover or authenticate them automatically.",
            command,
        ),
        (
            "4. Interpret the result",
            f"This package displays the recorded policy result: {decision_label(decision)}. Recipient policy and trust may differ. Exit 0 means the verifier accepted the scoped evidence under your supplied inputs; it does not authorize deployment. Exit 7 means the required policy or recipient acceptance was not established. Inspect the JSON to distinguish a verified rejection or insufficient evidence from an integrity/replay failure. The exit code alone is not proof of successful replay. Other nonzero exits require diagnosis; do not bypass missing anchors, invalid signatures or incompatible inputs.",
            None,
        ),
        (
            "5. Retain the decision record",
            "Keep the unchanged evidence, authored context, brief, full review, technical appendix, new result and receipt, and the identity of your separately approved trust inputs. Record the owner's release decision separately. A supplied historical verification.receipt.json is an unverified attachment; a matching evidence reference does not authenticate its signature or authorize its signer. "
            + (
                "The evidence-set result is not a new signed attestation; embedded component receipts retain their distinct scopes."
                if kind == "evidence_set"
                else "A signed verification receipt and an organizational approval are different records."
            ),
            None,
        ),
        (
            "Troubleshooting",
            "Missing trust profile or key: contact your technical owner for independently approved inputs. Wrong expected identity or signature: stop and resolve the mismatch; never replace approved pins with values from the submitted package. Wrong format/version: use the compatible approved core and the profile for this evidence family. Existing result directory: choose a fresh directory. Unsupported scorer or runtime replay requirement: follow the verifier's diagnostic with your technical owner; do not install unapproved code or rerun inference just to make a rejection disappear.",
            None,
        ),
    ]
    if not html:
        return (
            "# Verify this client package\n\n"
            + "\n\n".join(
                f"## {title}\n\n{text}" + (f"\n\n```sh\n{code}\n```" if code else "")
                for title, text, code in sections
            )
            + "\n"
        )
    body = '<div class="brand">InvarLock · Recipient guide</div><h1>Verify this client package</h1><nav><a href="brief.html">Decision brief</a><a href="review.html">Full review</a><a href="verification.md">Plain-text commands</a></nav>'
    body += "".join(
        f"<section><h2>{escape(title)}</h2><p>{escape(text)}</p>"
        + (f"<pre><code>{escape(code)}</code></pre>" if code else "")
        + "</section>"
        for title, text, code in sections
    )
    return _document("Verify this client package", body)
