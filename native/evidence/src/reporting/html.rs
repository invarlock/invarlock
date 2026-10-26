use super::presentation::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
const CSS: &str = include_str!("assets/style.css");
const BRAND: &str = include_str!("assets/brand.svg");
const SCRIPT: &str = include_str!("assets/metric_tabs.js");
fn interval_reading_html(v: &Value, level: usize) -> String {
    let mut out = format!(
        "<div class=\"interval-reading\"><h{level}>How to read this comparison</h{level}><dl>"
    );
    for (label, text) in interval_reading(v) {
        out += &format!("<dt>{}</dt><dd>{}</dd>", e(label), e(&text))
    }
    out += "</dl></div>";
    if !a(v, "basis").is_empty() {
        out+="<details class=\"interval-basis\"><summary>How this interval was calculated</summary><div class=\"detail-content\">";
        for paragraph in a(v, "basis") {
            out += &format!("<p>{}</p>", e(s(paragraph)))
        }
        out += "</div></details>"
    }
    out
}
fn interval(v: &Value) -> String {
    let lower = v["lower"].as_f64().unwrap();
    let upper = v["upper"].as_f64().unwrap();
    let estimate = v["estimate"].as_f64().unwrap();
    let threshold = v["threshold"].as_f64();
    let neutral = v["neutral"].as_f64();
    let unit = t(v, "unit");
    let mut values = vec![lower, upper, estimate];
    values.extend(threshold);
    values.extend(neutral);
    let scale = values.iter().map(|v| v.abs()).fold(1.0, f64::max);
    let first = values.iter().copied().fold(f64::INFINITY, f64::min) / scale;
    let last = values.iter().copied().fold(f64::NEG_INFINITY, f64::max) / scale;
    let mut padding = (last - first) * 0.12;
    if padding == 0.0 {
        padding = (first.abs() * 0.1).max(0.01)
    }
    let (mut low, mut high) = (first - padding, last + padding);
    let mut ticks = if first == last {
        vec![first]
    } else {
        vec![first, first / 2.0 + last / 2.0, last]
    };
    let desired = (last - first) * (scale / 4.0);
    if desired > 0.0 {
        let magnitude = 10f64.powf(desired.log10().floor());
        if magnitude > 0.0 {
            let factor = desired / magnitude;
            let step = [1.0, 2.0, 5.0, 10.0]
                .into_iter()
                .find(|v| *v >= factor)
                .unwrap()
                * magnitude
                / scale;
            let (ft, lt) = ((low / step).floor(), (high / step).ceil());
            let (al, ah) = (ft * step, lt * step);
            if (al * scale).is_finite()
                && (ah * scale).is_finite()
                && al < ah
                && al <= first
                && last <= ah
            {
                low = al;
                high = ah;
                ticks = (ft as i64..=lt as i64).map(|i| i as f64 * step).collect()
            }
        }
    }
    let project = |n: f64| 30.0 + (n - low) / (high - low) * 580.0;
    let x = |n: f64| project(n / scale);
    let mut description = format!(
        "{}. Estimate {} {unit}.",
        interval_summary(v, false),
        number(estimate, neutral == Some(0.0))
    );
    let mut graphics = String::new();
    let mut legend = "";
    let mut allowed_legend = "";
    let mut neutral_legend = "";
    if let Some(threshold) = threshold {
        description += &format!(" Policy threshold {} {unit}.", number(threshold, false));
        let position = x(threshold);
        let direction = t(v, "threshold_direction");
        if ["minimum", "maximum"].contains(&direction) {
            let (left, right) = if direction == "minimum" {
                (position, 610.0)
            } else {
                (30.0, position)
            };
            graphics += &format!(
                "<rect class=\"allowed\" x=\"{left:.2}\" y=\"18\" width=\"{:.2}\" height=\"42\"/>",
                right - left
            );
            allowed_legend="<span class=\"chart-key\"><i class=\"allowed-key\" aria-hidden=\"true\"></i>Allowed change region</span>";
            description += &format!(
                " Shading marks values at or {} the threshold.",
                if direction == "minimum" {
                    "above"
                } else {
                    "below"
                }
            )
        }
        legend = "Policy threshold";
        graphics+=&format!("<line class=\"threshold\" x1=\"{position:.2}\" x2=\"{position:.2}\" y1=\"10\" y2=\"66\"/>")
    }
    if let Some(neutral) = neutral {
        let position = x(neutral);
        graphics+=&format!("<line class=\"neutral\" x1=\"{position:.2}\" x2=\"{position:.2}\" y1=\"18\" y2=\"60\"/>");
        neutral_legend = "No change";
        description += &format!(" No change: {} {unit}.", number(neutral, false))
    }
    let mut labels = String::new();
    for tick in ticks {
        let position = project(tick);
        graphics += &format!(
            "<line class=\"tick\" x1=\"{position:.2}\" x2=\"{position:.2}\" y1=\"60\" y2=\"67\"/>"
        );
        labels += &format!(
            "<span style=\"left:{:.2}%\">{}</span>",
            position / 640.0 * 100.0,
            e(&number(tick * scale, false))
        )
    }
    let annotation = |value: f64, text: &str, kind: &str| {
        let percent = x(value) / 640.0 * 100.0;
        let anchor = if percent < 22.0 {
            "start"
        } else if percent > 78.0 {
            "end"
        } else {
            "middle"
        };
        format!(
            "<span class=\"chart-label {kind} {anchor}\" style=\"left:{percent:.2}%\">{}</span>",
            e(text)
        )
    };
    let mut annotations = String::new();
    if let Some(threshold) = threshold {
        let operator = match t(v, "threshold_direction") {
            "minimum" => "≥ ",
            "maximum" => "≤ ",
            _ => "",
        };
        annotations += &annotation(
            threshold,
            format!("Limit {operator}{} {unit}", number(threshold, false)).trim(),
            "limit-label",
        )
    }
    let bounds = if x(upper) - x(lower) < 180.0 {
        annotation(
            lower / 2.0 + upper / 2.0,
            &format!("{} to {}", number(lower, false), number(upper, false)),
            "bound-label",
        )
    } else {
        annotation(lower, &number(lower, false), "bound-label")
            + &annotation(upper, &number(upper, false), "bound-label")
    };
    format!("<figure class=\"interval\"><div class=\"chart-annotations\" aria-hidden=\"true\">{annotations}</div><div class=\"interval-plot\"><svg viewBox=\"0 0 640 72\" role=\"img\" aria-label=\"{}\">{graphics}<line class=\"axis\" x1=\"30\" x2=\"610\" y1=\"60\" y2=\"60\"/><line class=\"range\" x1=\"{:.2}\" x2=\"{:.2}\" y1=\"38\" y2=\"38\"/><circle class=\"estimate\" cx=\"{:.2}\" cy=\"38\" r=\"6\"/></svg><div class=\"interval-bounds\" aria-hidden=\"true\">{bounds}</div></div><div class=\"axis-labels\" aria-hidden=\"true\">{labels}</div><figcaption><span class=\"chart-key\"><i class=\"estimate-key\" aria-hidden=\"true\"></i>Estimate</span><span class=\"chart-key\"><i class=\"interval-key\" aria-hidden=\"true\"></i>{}</span>{}{}{allowed_legend}</figcaption></figure>",e(&description),x(lower),x(upper),x(estimate),e(&interval_label(v,false)),if legend.is_empty(){String::new()}else{format!(" <span class=\"chart-key\"><i class=\"threshold-key\" aria-hidden=\"true\"></i>{}</span>",e(legend))},if neutral_legend.is_empty(){String::new()}else{format!("<span class=\"chart-key\"><i class=\"neutral-key\" aria-hidden=\"true\"></i>{}</span>",e(neutral_legend))})
}
fn overview(metrics: &[Value]) -> String {
    let mut out="<section class=\"results-overview\" aria-labelledby=\"results-overview-heading\"><div class=\"section-heading\"><h2 id=\"results-overview-heading\" tabindex=\"-1\">All results at a glance</h2><p>Each row is one metric and scope; overlapping scopes are not additional independent cases.</p></div><div class=\"scroll overview-scroll\" tabindex=\"0\" role=\"region\" aria-label=\"All metric and scope results\"><table class=\"overview-table\"><caption>Recorded results and checks needing attention</caption><thead><tr><th scope=\"col\">Metric / scope</th><th scope=\"col\">Decision</th><th scope=\"col\">Baseline</th><th scope=\"col\">Candidate</th><th scope=\"col\">Change</th><th scope=\"col\">Observed pairs</th><th scope=\"col\">Checks needing attention</th></tr></thead><tbody>".to_owned();
    for (index, m) in metrics.iter().enumerate() {
        let attention = a(m, "checks")
            .iter()
            .filter(|c| c["passed"] != true)
            .map(|c| {
                format!(
                    "<li class=\"check-{}\">{}: {}</li>",
                    if c["passed"] == false {
                        "fail"
                    } else {
                        "unknown"
                    },
                    e(t(c, "name")),
                    status(c)
                )
            })
            .collect::<String>();
        let checks = if !attention.is_empty() {
            format!("<ul class=\"overview-checks\">{attention}</ul>")
        } else if !a(m, "checks").is_empty() {
            "No failed or unavailable checks".into()
        } else {
            "Check details unavailable".into()
        };
        out+=&format!("<tr><th scope=\"row\"><a href=\"#metric-result-{}\">{}<span class=\"scope\">{}</span></a></th><td><span class=\"badge {}\">{}</span></td>",index+1,e(&display_label(t(m,"name"))),e(&display_label(t(m,"scope"))),tone(t(m,"decision")),e(decision(t(m,"decision"))));
        for k in ["baseline", "candidate", "change", "count"] {
            out += &format!("<td>{}</td>", e(t(m, k)))
        }
        out += &format!("<td>{checks}</td></tr>")
    }
    out + "</tbody></table></div></section>"
}
fn context_html(v: &Value) -> String {
    let c = comparison_context(v);
    let mut out="<section class=\"panel\" aria-labelledby=\"comparison-context\"><h2 id=\"comparison-context\">What was compared</h2>".to_owned();
    if !c.rows.is_empty() {
        out+="<table class=\"comparison-table\"><thead><tr><th scope=\"col\">Recorded field</th><th scope=\"col\">Baseline</th><th scope=\"col\">Subject</th></tr></thead><tbody>";
        for (label, baseline, subject, changed) in &c.rows {
            out+=&format!("<tr{}><th scope=\"row\">{}{}</th><td data-side=\"Baseline\"><span class=\"side-label\" aria-hidden=\"true\">Baseline</span>{}</td><td data-side=\"Subject\"><span class=\"side-label\" aria-hidden=\"true\">Subject</span>{}</td></tr>",if *changed{" class=\"different\""}else{""},e(label),if *changed{"<span class=\"different-label\">Differs</span>"}else{""},e(baseline),e(subject))
        }
        out += "</tbody></table>"
    }
    if !c.matching.is_empty() || !c.additional.is_empty() {
        if !c.rows.is_empty() {
            out+=&format!("<details class=\"shared-context\"><summary>Additional recorded context ({})</summary><div class=\"detail-content\">",c.matching.len()+c.additional.len())
        }
        for (label, entries) in [
            ("Matching displayed fields", &c.matching),
            ("Additional fields", &c.additional),
        ] {
            if entries.is_empty() {
                continue;
            }
            out += &format!("<h3 class=\"context-heading\">{label}</h3>");
            if label == "Matching displayed fields" {
                out += &format!("<p class=\"scope\">{}</p>", msg("_MATCHING_CONTEXT_NOTE"))
            }
            out += "<dl class=\"context-list\">";
            for (k, v) in entries {
                out += &format!("<dt>{}</dt><dd>{}</dd>", e(k), e(v))
            }
            out += "</dl>"
        }
        if !c.rows.is_empty() {
            out += "</div></details>"
        }
    }
    if !a(v, "changes").is_empty() {
        out += "<ul class=\"change-notes\" aria-label=\"Recorded changes\">";
        for note in a(v, "changes") {
            out += &format!("<li>{}</li>", e(s(note)))
        }
        out += "</ul>"
    }
    out + "</section>"
}
fn detail(v: &Value) -> String {
    if v["preview_only"] == true && v["text"].is_string() {
        return format!(
            "<p>{}</p><pre>{}</pre><p class=\"scope\">Preview limits: {}</p>",
            e(v.get("note")
                .and_then(Value::as_str)
                .unwrap_or("Bounded preview; full values remain in the evidence.")),
            e(t(v, "text")),
            e(&pretty(v.get("limits").unwrap_or(&json!({})), 0, false))
        );
    }
    format!("<pre>{}</pre>", e(&pretty(v, 0, true)))
}
fn summary(v: &Value) -> String {
    let metrics = a(v, "metrics");
    if metrics.len() != 1 {
        return e(t(v, "summary"));
    }
    let m = &metrics[0];
    let mut tokens = vec![
        display_label(t(m, "name")),
        t(m, "baseline").into(),
        t(m, "candidate").into(),
        t(m, "change").into(),
    ];
    if m["interval"].is_object() {
        for k in ["lower", "upper", "estimate", "threshold"] {
            if let Some(n) = m["interval"][k].as_f64() {
                tokens.push(number(n, false))
            }
        }
    }
    tokens.retain(|s| !s.is_empty());
    tokens.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    tokens.dedup();
    let text = t(v, "summary");
    let mut out = String::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let before = text[..cursor].chars().next_back();
        let word = |c: char| c.is_alphanumeric() || c == '_';
        let candidate = if before.is_none_or(|c| !word(c) && !".+-".contains(c)) {
            tokens.iter().find(|token| {
                text[cursor..].starts_with(token.as_str()) && {
                    let mut after = text[cursor + token.len()..].chars();
                    let c = after.next();
                    c.is_none_or(|c| {
                        !word(c)
                            && c != '%'
                            && !(c == '.' && after.next().is_some_and(|c| c.is_numeric()))
                    })
                }
            })
        } else {
            None
        };
        if let Some(token) = candidate {
            out += &format!("<strong>{}</strong>", e(token));
            cursor += token.len()
        } else {
            let c = text[cursor..].chars().next().unwrap();
            out += &e(&c.to_string());
            cursor += c.len_utf8()
        }
    }
    out
}
fn review_context(v: &Value) -> String {
    let r = &v["review"];
    let mut out="<section class=\"review-context\" aria-labelledby=\"review-context\"><h2 id=\"review-context\">Decision context</h2>".to_owned();
    if r.is_object() {
        out+=&format!("<p class=\"eyebrow\">{}</p><p><strong>Owner decision.</strong> {}</p><p class=\"scope\">{}</p>",e(t(r,"label")),e(t(r,"decision")),e(t(r,"byline")));
        for p in a(r, "context") {
            out += &format!("<p><strong>{}.</strong> {}</p>", e(s(&p[0])), e(s(&p[1])))
        }
        out += "</section>"
    } else {
        if !a(v, "subjects").is_empty() {
            out += "<dl class=\"review-subjects\">";
            for p in a(v, "subjects") {
                out += &format!(
                    "<div><dt>{}</dt><dd>{}</dd></div>",
                    e(s(&p[0])),
                    e(s(&p[1]))
                )
            }
            out += "</dl>"
        }
        out += "<ul class=\"context-status\">";
        for item in decision_context() {
            out += &format!("<li>{item}</li>")
        }
        out+="</ul><p class=\"scope\">The recorded task, cases and requirements define the evaluated scope. A policy result does not supply a business recommendation.</p></section>"
    }
    out + &format!(
        "<p class=\"change-summary\"><strong>Change context.</strong> {}</p>",
        e(&change_summary(v))
    )
}
fn findings_html(v: &Value) -> String {
    let (found, note) = findings(v);
    let mut out =
        format!("<section class=\"panel findings\"><h2>Findings to review</h2><p>{note}</p>");
    if !a(v, "metrics").is_empty() {
        out += &format!("<p class=\"result-counts\">{}</p>", result_counts(v))
    }
    out += "<ul>";
    for (i, m) in found {
        out+=&format!("<li><span class=\"finding-status {}\">{}</span> <a href=\"#metric-result-{i}\">Result {i}: {} · {}</a> — {}</li>",tone(t(m,"decision")),e(decision(t(m,"decision"))),e(&display_label(t(m,"name"))),e(&display_label(t(m,"scope"))),e(t(m,"explanation")))
    }
    out+"</ul><p><a href=\"#results-and-requirements\">Supporting evidence: results and decision checks</a></p></section>"
}
/// Render escaped presentation fields with a hash-authorized local metric-tab script.
pub fn render_html(v: &Value) -> String {
    let metrics = a(v, "metrics");
    let multiple = metrics.len() > 1;
    let review = &v["review"];
    let authored = review.is_object();
    let mut groups = Vec::<(&str, Vec<(usize, &Value)>)>::new();
    for (i, m) in metrics.iter().enumerate() {
        let name = t(m, "name");
        if let Some((_, g)) = groups.iter_mut().find(|(n, _)| *n == name) {
            g.push((i + 1, m))
        } else {
            groups.push((name, vec![(i + 1, m)]))
        }
    }
    let tabs = groups.len() > 1;
    let script_policy = if tabs {
        format!(
            " script-src 'sha256-{}';",
            STANDARD.encode(Sha256::digest(SCRIPT.as_bytes()))
        )
    } else {
        String::new()
    };
    let mut out=format!("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none';{script_policy}\"><title>{} · {}</title><style>{CSS}</style></head><body><main><header class=\"brand\">{BRAND} InvarLock<span class=\"family\">{}</span></header><div class=\"report-heading\"><p class=\"eyebrow\">Comparison / Review / Evidence</p><h1>{}</h1><p>{}</p></div><p class=\"interpretation-note\">{}</p>{}<section class=\"hero {}\" aria-labelledby=\"decision\"><div class=\"verdict\"><p class=\"eyebrow\">Recorded policy result</p><h2 id=\"decision\">{}</h2><p class=\"decision-summary\">{}</p>",e(decision(t(v,"decision"))),e(t(v,"title")),e(t(v,"family")),if authored{"Client review package"}else{"Model change review"},e(t(v,"title")),interpretation(v),review_context(v),tone(t(v,"decision")),e(decision(t(v,"decision"))),summary(v));
    if multiple {
        out+=&format!("<div class=\"summary-row\"><span class=\"pill\">{} metric / scope results</span></div>",metrics.len())
    }
    out+=&format!("<p class=\"review-action\"><strong>{}.</strong> {}</p></div><aside class=\"assurance\" aria-labelledby=\"assurance-heading\"><h2 id=\"assurance-heading\">What was checked</h2><dl>",if authored{"Owner recommendation"}else{"Review action"},if authored{e(t(review,"recommendation"))}else{review_action(v).into()});
    for p in a(v, "assurance") {
        out += &format!(
            "<div><dt>{}</dt><dd>{}</dd></div>",
            e(s(&p[0])),
            e(s(&p[1]))
        )
    }
    out += "</dl>";
    if a(v, "assurance").is_empty() {
        out += &format!("<p>{}</p>", msg("_MISSING_ASSURANCE"))
    }
    out += "</aside></section>";
    out += &findings_html(v);
    out += "<section class=\"panel limits\"><h2>Scope and limitations</h2><ul>";
    if a(v, "limitations").is_empty() {
        out += &format!("<li>{}</li>", e(msg("_MISSING_LIMITATIONS")))
    } else {
        for n in a(v, "limitations") {
            out += &format!("<li>{}</li>", e(s(n)))
        }
    }
    out += "</ul></section>";
    out += &format!(
        "<section class=\"panel next-steps\"><h2>{}</h2><p>{}</p>",
        if authored {
            "Decision follow-up"
        } else {
            "Next decision"
        },
        if authored {
            e(t(review, "next_action"))
        } else {
            msg("_NEXT_DECISION").into()
        }
    );
    if !authored && !a(v, "next_steps").is_empty() {
        out += "<h3>Next steps</h3><ol>";
        for step in a(v, "next_steps") {
            out += &format!("<li>{}</li>", e(s(step)))
        }
        out += "</ol>"
    }
    out += "</section>";
    if authored {
        out += "<section class=\"panel review-basis\"><h2>Review rationale</h2>";
        for p in a(review, "basis") {
            out += &format!("<p><strong>{}.</strong> {}</p>", e(s(&p[0])), e(s(&p[1])))
        }
        out += "</section>"
    }
    if ["subjects", "context", "changes"]
        .iter()
        .any(|k| !a(v, k).is_empty())
    {
        out += &context_html(v)
    }
    out+="<div class=\"section-heading\"><h2 id=\"results-and-requirements\">Results and requirements</h2></div>";
    if multiple {
        out+=&format!("<details class=\"overview-disclosure\"{}><summary>Compare all metric and scope results</summary><div class=\"detail-content\">{}</div></details>",if metrics.iter().any(|m|t(m,"decision")!="pass"){" open"}else{""},overview(metrics));
        if tabs {
            out+="<div class=\"metric-controls\"><nav class=\"metric-navigation\" aria-label=\"Metric results\">";
            for (i, (name, group)) in groups.iter().enumerate() {
                out += &format!(
                    "<a href=\"#metric-group-{}\">{} <span>{} scope result{}</span></a>",
                    i + 1,
                    e(&display_label(name)),
                    group.len(),
                    if group.len() == 1 { "" } else { "s" }
                )
            }
            out+="</nav><button class=\"metric-display-toggle\" type=\"button\" hidden>Show all metrics</button></div>"
        }
    }
    for (group_index, (name, group)) in groups.iter().enumerate() {
        let group_index = group_index + 1;
        if multiple {
            out+=&format!("<section id=\"metric-panel-{group_index}\" class=\"metric-group\" aria-labelledby=\"metric-group-{group_index}\"><div class=\"section-heading\"><h3 id=\"metric-group-{group_index}\" tabindex=\"-1\">{}</h3><a href=\"#results-overview-heading\">Back to overview</a></div>",e(&display_label(name)))
        }
        for (index, m) in group {
            let heading = if multiple {
                format!("<h4>{}</h4>", e(&display_label(t(m, "scope"))))
            } else {
                format!("<h3>{}</h3>", e(&display_label(t(m, "name"))))
            };
            let context = if multiple {
                format!("Metric: {}", e(&display_label(t(m, "name"))))
            } else {
                format!("Scope: {}", e(&display_label(t(m, "scope"))))
            };
            out+=&format!("<section id=\"metric-result-{index}\" tabindex=\"-1\" class=\"metric {}\"><div class=\"metric-lead\"><div class=\"metric-heading\"><div>{heading}<p class=\"scope\">{context}</p></div><span class=\"badge\">{}</span></div><p class=\"metric-explanation\">{}</p><dl class=\"values\">",tone(t(m,"decision")),e(decision(t(m,"decision"))),e(t(m,"explanation")));
            let change_detail = if m["interval"].is_object() {
                interval_summary(&m["interval"], true)
            } else {
                String::new()
            };
            let count_label = m
                .get("count_label")
                .and_then(Value::as_str)
                .unwrap_or("Observed pairs");
            for (label, value, detail) in [
                ("Baseline", t(m, "baseline"), t(m, "baseline_detail")),
                ("Subject", t(m, "candidate"), t(m, "candidate_detail")),
                ("Change", t(m, "change"), change_detail.as_str()),
                (count_label, t(m, "count"), t(m, "count_detail")),
            ] {
                let detail = if detail.is_empty() {
                    String::new()
                } else {
                    format!("<small>{}</small>", e(detail))
                };
                out += &format!(
                    "<div class=\"value\"><dt>{}</dt><dd>{}{detail}</dd></div>",
                    e(label),
                    e(value)
                )
            }
            out += "</dl></div>";
            if m["interval"].is_object() {
                out += &interval(&m["interval"]);
                out += &interval_reading_html(&m["interval"], if multiple { 5 } else { 4 })
            }
            let checks = a(m, "checks");
            if !checks.is_empty() {
                out+="<div class=\"scroll\" tabindex=\"0\" role=\"region\" aria-label=\"Decision checks\"><table class=\"checks-table\"><caption>Decision checks</caption><thead><tr><th scope=\"col\">Check</th><th scope=\"col\">Observed</th><th scope=\"col\">Required</th><th scope=\"col\">Result</th></tr></thead><tbody>"
            }
            for check in checks {
                let tone = match check["passed"].as_bool() {
                    Some(true) => "pass",
                    Some(false) => "fail",
                    None => "unknown",
                };
                let detail = if t(check, "explanation").is_empty() {
                    String::new()
                } else {
                    format!(
                        "<span class=\"check-detail\">{}</span>",
                        e(t(check, "explanation"))
                    )
                };
                out+=&format!("<tr><th scope=\"row\">{}{detail}</th><td data-label=\"Observed\"><span class=\"check-label\" aria-hidden=\"true\">Observed</span>{}</td><td data-label=\"Required\"><span class=\"check-label\" aria-hidden=\"true\">Required</span>{}</td><td data-label=\"Result\" class=\"check-{tone}\"><span class=\"check-label\" aria-hidden=\"true\">Result</span>{}</td></tr>",e(t(check,"name")),e(t(check,"observed")),e(t(check,"required")),status(check))
            }
            if !checks.is_empty() {
                out += "</tbody></table></div>"
            } else {
                out += &format!("<p class=\"missing-checks\">{}</p>", msg("_MISSING_CHECKS"))
            }
            let source = t(m, "evidence_source");
            out += &format!(
                "<p class=\"scope\">Result {index} · Supporting record: {}</p>",
                e(if source.is_empty() {
                    "Not supplied with this view"
                } else {
                    source
                })
            );
            if !a(m, "notes").is_empty() {
                out += "<ul class=\"notes\">";
                for note in a(m, "notes") {
                    out += &format!("<li>{}</li>", e(s(note)))
                }
                out += "</ul>"
            }
            out += "</section>";
        }
        if multiple {
            out += "</section>"
        }
    }
    out+="<div class=\"section-heading\"><h2>Evidence details</h2><p>Identities, methods and exact recorded values</p></div><p class=\"scope\">The review and methods print automatically. Open an evidence detail below to include its contents in print; complete originals remain in the evidence bundle.</p>";
    if !a(v, "identity").is_empty() {
        out+="<details><summary>Comparison and policy identities</summary><div class=\"detail-content\"><dl class=\"identity-list\">";
        for p in a(v, "identity") {
            out += &format!(
                "<div><dt>{}</dt><dd><code>{}</code></dd></div>",
                e(s(&p[0])),
                e(s(&p[1]))
            )
        }
        out += "</dl></div></details>"
    }
    let mut details = identifier_details(v);
    details.extend(a(v, "details").iter().cloned());
    details.push(json!([
        "Exact comparison data",
        v.get("technical").cloned().unwrap_or(json!({}))
    ]));
    for p in details {
        out+=&format!("<details class=\"evidence-detail\"><summary>{}</summary><div class=\"detail-content\">{}</div></details>",e(s(&p[0])),detail(&p[1]))
    }
    for p in a(v, "details") {
        for m in a(&p[1], "metrics") {
            let preview = &m["configuration"];
            if preview["preview_only"] == true && preview["text"].is_string() {
                out+=&format!("<details class=\"evidence-detail\"><summary>{} configuration preview</summary><div class=\"detail-content\">{}</div></details>",e(&display_label(m.get("name").and_then(Value::as_str).unwrap_or("Metric"))),detail(preview))
            }
        }
    }
    out+="<footer class=\"footer\">InvarLock · Evidence report. Displayed values may be rounded; exact values are preserved in the evidence bundle. This report is not an independent acceptance receipt.</footer></main>";
    if tabs {
        out += &format!("<script>{SCRIPT}</script>")
    }
    out + "</body></html>\n"
}
