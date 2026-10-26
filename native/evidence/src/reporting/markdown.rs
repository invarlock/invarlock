use super::presentation::*;
use serde_json::{json, Value};
fn clean(value: &str) -> String {
    let mut value = escape(value, false).replace(['\n', '\r'], " ");
    for c in "|\\`*[]()".chars() {
        value = value.replace(c, &format!("&#{};", c as u32))
    }
    let chars = value.chars().collect::<Vec<_>>();
    chars
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let word = |c: &char| c.is_alphanumeric() || *c == '_';
            if *c == '_'
                && (i == 0 || !word(&chars[i - 1]) || i + 1 == chars.len() || !word(&chars[i + 1]))
            {
                "&#95;".into()
            } else {
                c.to_string()
            }
        })
        .collect()
}
fn pairs(lines: &mut Vec<String>, pairs: &[Value]) {
    lines.extend(
        pairs
            .iter()
            .map(|p| format!("- **{}:** {}", clean(s(&p[0])), clean(s(&p[1])))),
    )
}
fn push(lines: &mut Vec<String>, values: &[&str]) {
    lines.extend(values.iter().map(|s| s.to_string()))
}
/// Render a presentation view; this does not verify evidence or approve a result.
pub fn render_markdown(v: &Value, include_details: bool) -> String {
    let review = &v["review"];
    let authored = review.is_object();
    let mut lines = vec![
        format!("# {}", clean(t(v, "title"))),
        String::new(),
        "## Decision context".into(),
        String::new(),
        interpretation(v).into(),
        String::new(),
    ];
    pairs(&mut lines, a(v, "subjects"));
    lines.push(String::new());
    if authored {
        lines.extend([
            format!("**{}**", clean(t(review, "label"))),
            String::new(),
            format!("**Owner decision:** {}", clean(t(review, "decision"))),
            String::new(),
            clean(t(review, "byline")),
            String::new(),
        ]);
        pairs(&mut lines, a(review, "context"))
    } else {
        lines.extend(decision_context().iter().map(|s| format!("- {s}")))
    }
    lines.extend([
        String::new(),
        format!("**Change context:** {}", clean(&change_summary(v))),
        String::new(),
        format!("**Recorded policy result: {}**", decision(t(v, "decision"))),
        String::new(),
        clean(t(v, "summary")),
        String::new(),
        format!(
            "**{}:** {}",
            if authored {
                "Owner recommendation"
            } else {
                "Review action"
            },
            if authored {
                clean(t(review, "recommendation"))
            } else {
                review_action(v).into()
            }
        ),
        String::new(),
    ]);
    let (found, note) = findings(v);
    push(&mut lines, &["## Findings to review", "", &note, ""]);
    if !a(v, "metrics").is_empty() {
        push(&mut lines, &[&result_counts(v), ""])
    }
    for (i, m) in found {
        let source = t(m, "evidence_source");
        lines.push(format!(
            "- {} — Result {i}: {} · {} — {} Supporting record: {}",
            decision(t(m, "decision")),
            clean(&display_label(t(m, "name"))),
            clean(&display_label(t(m, "scope"))),
            clean(t(m, "explanation")),
            clean(if source.is_empty() {
                "Not supplied with this view"
            } else {
                source
            })
        ))
    }
    push(
        &mut lines,
        &[
            "",
            "Supporting evidence: the numbered results and decision checks below.",
            "",
            "",
            "## What was checked",
            "",
        ],
    );
    pairs(&mut lines, a(v, "assurance"));
    if a(v, "assurance").is_empty() {
        push(&mut lines, &[msg("_MISSING_ASSURANCE"), ""])
    }
    push(&mut lines, &["", "## Scope and limitations", ""]);
    if a(v, "limitations").is_empty() {
        lines.push(format!("- {}", clean(msg("_MISSING_LIMITATIONS"))))
    } else {
        lines.extend(
            a(v, "limitations")
                .iter()
                .map(|s| format!("- {}", clean(self::s(s)))),
        )
    }
    push(
        &mut lines,
        &[
            "",
            if authored {
                "## Decision follow-up"
            } else {
                "## Next decision"
            },
            "",
            &if authored {
                clean(t(review, "next_action"))
            } else {
                msg("_NEXT_DECISION").into()
            },
            "",
        ],
    );
    if !authored && !a(v, "next_steps").is_empty() {
        push(&mut lines, &["### Next steps", ""]);
        lines.extend(
            a(v, "next_steps")
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, clean(self::s(s)))),
        )
    }
    if authored {
        push(&mut lines, &["", "## Review rationale", ""]);
        pairs(&mut lines, a(review, "basis"))
    }
    if ["subjects", "context", "changes"]
        .iter()
        .any(|k| !a(v, k).is_empty())
    {
        push(&mut lines, &["## What was compared", ""]);
        let context = comparison_context(v);
        for (label, baseline, subject, changed) in context.rows {
            lines.extend([
                String::new(),
                format!(
                    "### {}: {}",
                    clean(&label),
                    if changed {
                        "Differs"
                    } else {
                        "Same displayed text"
                    }
                ),
                String::new(),
                format!("- **Baseline:** {}", clean(&baseline)),
                format!("- **Subject:** {}", clean(&subject)),
            ])
        }
        for (heading, entries) in [
            ("Matching displayed fields", context.matching),
            ("Additional fields", context.additional),
        ] {
            if !entries.is_empty() {
                push(&mut lines, &["", &format!("### {heading}"), ""]);
                if heading == "Matching displayed fields" {
                    push(&mut lines, &[msg("_MATCHING_CONTEXT_NOTE"), ""])
                }
                lines.extend(
                    entries
                        .into_iter()
                        .map(|(k, v)| format!("- **{}:** {}", clean(&k), clean(&v))),
                )
            }
        }
        if !a(v, "changes").is_empty() {
            push(&mut lines, &["", "### Recorded changes", ""]);
            lines.extend(
                a(v, "changes")
                    .iter()
                    .map(|s| format!("- {}", clean(self::s(s)))),
            )
        }
    }
    for (index, m) in a(v, "metrics").iter().enumerate() {
        let count_label = m
            .get("count_label")
            .and_then(Value::as_str)
            .unwrap_or("Observed pairs");
        lines.extend([
            String::new(),
            format!(
                "## {}: {}",
                clean(&display_label(t(m, "name"))),
                clean(&display_label(t(m, "scope")))
            ),
            String::new(),
            format!(
                "**{}**. {}",
                decision(t(m, "decision")),
                clean(t(m, "explanation"))
            ),
            String::new(),
            format!("| Baseline | Subject | Change | {} |", clean(count_label)),
            "| --- | --- | --- | --- |".into(),
            format!(
                "| {} |",
                ["baseline", "candidate", "change", "count"]
                    .iter()
                    .map(|k| clean(t(m, k)))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            String::new(),
        ]);
        for (label, key) in [
            ("Baseline", "baseline_detail"),
            ("Subject", "candidate_detail"),
            (count_label, "count_detail"),
        ] {
            if !t(m, key).is_empty() {
                push(
                    &mut lines,
                    &[&format!("{}: {}.", clean(label), clean(t(m, key))), ""],
                )
            }
        }
        if m["interval"].is_object() {
            let i = &m["interval"];
            push(
                &mut lines,
                &[
                    &format!(
                        "{}: [{}, {}] {}.",
                        clean(&interval_label(i, false)),
                        number(i["lower"].as_f64().unwrap(), false),
                        number(i["upper"].as_f64().unwrap(), false),
                        clean(t(i, "unit"))
                    ),
                    "",
                    "### How to read this comparison",
                    "",
                ],
            );
            for (label, explanation) in interval_reading(i) {
                push(
                    &mut lines,
                    &[
                        &format!("**{}:** {}", clean(label), clean(&explanation)),
                        "",
                    ],
                )
            }
            if !a(i, "basis").is_empty() {
                push(&mut lines, &["### How this interval was calculated", ""]);
                for p in a(i, "basis") {
                    push(&mut lines, &[&clean(s(p)), ""])
                }
            }
        }
        push(&mut lines, &["### Decision checks", ""]);
        if !a(m, "checks").is_empty() {
            push(
                &mut lines,
                &[
                    "| Check | Observed | Required | Result |",
                    "| --- | --- | --- | --- |",
                ],
            )
        }
        for c in a(m, "checks") {
            lines.push(format!(
                "| {} |",
                [t(c, "name"), t(c, "observed"), t(c, "required"), status(c)]
                    .iter()
                    .map(|s| clean(s))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ))
        }
        lines.push(String::new());
        if a(m, "checks").is_empty() {
            push(&mut lines, &[msg("_MISSING_CHECKS"), ""])
        }
        let source = t(m, "evidence_source");
        push(
            &mut lines,
            &[
                &format!(
                    "Result {} · Supporting record: {}",
                    index + 1,
                    clean(if source.is_empty() {
                        "Not supplied with this view"
                    } else {
                        source
                    })
                ),
                "",
            ],
        );
        lines.extend(
            a(m, "notes")
                .iter()
                .map(|s| format!("- {}", clean(self::s(s)))),
        );
    }
    push(&mut lines, &["", "## Evidence identities", ""]);
    pairs(&mut lines, a(v, "identity"));
    let mut details = identifier_details(v);
    if include_details {
        details.extend(a(v, "details").iter().cloned());
        details.push(json!([
            "Exact comparison data",
            v.get("technical").cloned().unwrap_or(json!({}))
        ]))
    }
    for pair in details {
        push(
            &mut lines,
            &[
                "",
                &format!("## {}", clean(s(&pair[0]))),
                "",
                "```json",
                &visible_controls(&pretty(&pair[1], 0, true)).replace('`', "\\u0060"),
                "```",
            ],
        );
    }
    lines.join("\n") + "\n"
}
