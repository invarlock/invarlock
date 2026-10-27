use invarlock_native_contracts::canonical_json_no_newline;
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::OnceLock};
pub(super) fn t<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("")
}
pub(super) fn a<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v[k].as_array().map(Vec::as_slice).unwrap_or(&[])
}
pub(super) fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
pub(super) fn msg(k: &str) -> &'static str {
    static M: OnceLock<Value> = OnceLock::new();
    M.get_or_init(|| serde_json::from_str(include_str!("assets/messages.json")).unwrap())[k]
        .as_str()
        .unwrap()
}
pub(super) fn decision_context() -> [&'static str; 3] {
    [
        "Intended use and business context: Not supplied",
        "Decision owner and recommendation: Not supplied",
        "Organizational approval: Not recorded by this report",
    ]
}
fn controls(c: char, terminal: bool) -> bool {
    matches!(c,'\u{0000}'..='\u{0008}'|'\u{000b}'..='\u{001f}'|'\u{007f}'..='\u{009f}'|'\u{061c}'|'\u{200e}'|'\u{200f}'|'\u{2028}'..='\u{202e}'|'\u{2066}'..='\u{2069}')
        || terminal && c <= '\u{001f}'
}
pub fn visible_controls(s: &str) -> String {
    s.chars()
        .map(|c| {
            if controls(c, false) {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}
pub fn terminal_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            if controls(c, true) {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}
pub(super) fn escape(s: &str, quote: bool) -> String {
    let mut out = visible_controls(s)
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    if quote {
        out = out.replace('"', "&quot;").replace('\'', "&#x27;")
    }
    out
}
pub(super) fn e(s: &str) -> String {
    escape(s, true)
}
fn acronym(s: &str) -> Option<&'static str> {
    match s.to_lowercase().as_str() {
        "qa" => Some("QA"),
        "nll" => Some("NLL"),
        "utf8" | "utf-8" => Some("UTF-8"),
        "f1" => Some("F1"),
        "api" => Some("API"),
        "http" => Some("HTTP"),
        _ => None,
    }
}
pub fn display_label(value: &str) -> String {
    let (prefix, value) = ["Deterministic · ", "Judge · "]
        .iter()
        .find_map(|p| value.strip_prefix(p).map(|s| (*p, s)))
        .unwrap_or(("", value));
    if let Some(a) = acronym(value) {
        return format!("{prefix}{a}");
    }
    let tokens = value.split(['_', '-']).collect::<Vec<_>>();
    if tokens.len() < 2
        || tokens
            .iter()
            .any(|t| t.is_empty() || !t.bytes().all(|c| c.is_ascii_alphanumeric()))
    {
        return format!("{prefix}{value}");
    }
    let mut words = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let mut word = tokens[i];
        if word.eq_ignore_ascii_case("utf") && tokens.get(i + 1) == Some(&"8") {
            word = "utf8";
            i += 1
        }
        words.push(acronym(word).unwrap_or(word).to_string());
        i += 1
    }
    words[0] = capitalize(&words[0]);
    format!("{prefix}{}", words.join(" "))
}
pub(super) fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}
pub(super) fn decision(s: &str) -> &'static str {
    match s {
        "pass" => "Policy satisfied",
        "fail" | "regression" => "Policy not met",
        "insufficient_evidence" => "More evidence needed",
        _ => "Decision unavailable",
    }
}
pub(super) fn tone(s: &str) -> &'static str {
    match s {
        "pass" => "pass",
        "fail" | "regression" => "fail",
        "insufficient_evidence" => "insufficient",
        _ => "unavailable",
    }
}
pub(super) fn status(check: &Value) -> &'static str {
    match check["passed"].as_bool() {
        Some(true) => "Passed",
        Some(false) => "Not met",
        None => "Unavailable",
    }
}
pub fn number(value: f64, signed: bool) -> String {
    number_precision(value, signed, 4)
}
pub(super) fn number_precision(value: f64, signed: bool, precision: i32) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let sign = if value.is_sign_negative() {
        "-"
    } else if signed && value != 0.0 {
        "+"
    } else {
        ""
    };
    let value = value.abs();
    if value == 0.0 {
        return format!("{sign}0");
    }
    let scientific = format!("{value:.*e}", (precision - 1) as usize);
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent = exponent.parse::<i32>().unwrap();
    if !(-4..precision).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!(
            "{sign}{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        )
    } else {
        let text = format!("{:.*}", (precision - 1 - exponent).max(0) as usize, value);
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.')
        } else {
            &text
        };
        format!("{sign}{text}")
    }
}
pub(super) fn interval_label(v: &Value, compact: bool) -> String {
    let mut label = t(v, "label").to_owned();
    if compact {
        label = label.replace("confidence interval", "CI")
    } else if !t(v, "method").is_empty() {
        label += &format!(" ({})", t(v, "method"))
    }
    label
}
pub(super) fn interval_summary(v: &Value, compact: bool) -> String {
    format!(
        "{}: {} to {} {}",
        interval_label(v, compact),
        number(
            v["lower"].as_f64().unwrap(),
            v["neutral"].as_f64() == Some(0.0)
        ),
        number(
            v["upper"].as_f64().unwrap(),
            v["neutral"].as_f64() == Some(0.0)
        ),
        t(v, "unit")
    )
    .trim()
    .to_owned()
}
pub(super) fn interval_reading(v: &Value) -> Vec<(&'static str, String)> {
    let mut change =
        "The marker shows the observed change between subject and baseline.".to_owned();
    if t(v, "unit") == "pp" {
        change+=" A change in percentage points (pp) is the subject percentage minus the baseline percentage."
    } else if t(v, "unit") == "ratio" {
        change += " A ratio divides the subject value by the baseline value; 1 means no change."
    }
    let mut out=vec![("Change",change),("Interval","The bar shows uncertainty around that change. Its left endpoint is the lower bound; its right endpoint is the upper bound. The interval level does not tell you how often model runs would pass or fail the policy.".into())];
    if !v["threshold"].is_null() && ["minimum", "maximum"].contains(&t(v, "threshold_direction")) {
        out.push(("Policy rule",if t(v,"threshold_direction")=="minimum"{"The paired lower bound (left endpoint) must be at least the policy minimum. The whole interval must lie on the allowed side of the threshold for this bound check to pass."}else{"The upper bound (right endpoint) must be at most the policy maximum. The whole interval must lie on the allowed side of the threshold for this bound check to pass."}.into()))
    }
    if !v["neutral"].is_null() {
        out.push(("No change","The dashed line marks no change. An interval crossing it includes changes in either direction. A worse observed score describes this run; policy rejection alone does not prove worse performance beyond these cases.".into()))
    }
    out
}
pub(super) fn interpretation(v: &Value) -> &'static str {
    if v["review"].is_object() {
        "Decision context, recommendation and review rationale are separately authored commentary. Technical results come from the retained evidence."
    } else {
        msg("_INTERPRETATION_NOTE")
    }
}
pub(super) fn review_action(v: &Value) -> &'static str {
    match t(v,"decision"){"pass"=>"Confirm the intended use and complete recipient verification before recording a release decision.","fail"|"regression"=>"Review the unmet requirements and their causes before considering release.","insufficient_evidence"=>"Resolve the missing or insufficient evidence before drawing a release conclusion.",_=>"Establish the recorded outcome and evidence coverage before drawing a release conclusion."}
}
pub(super) fn result_counts(v: &Value) -> String {
    let mut counts = BTreeMap::new();
    for m in a(v, "metrics") {
        *counts.entry(tone(t(m, "decision"))).or_insert(0) += 1;
    }
    [
        ("pass", "satisfied"),
        ("fail", "not met"),
        ("insufficient", "need more evidence"),
        ("unavailable", "unavailable"),
    ]
    .iter()
    .filter_map(|(k, label)| counts.get(k).map(|n| format!("{n} {label}")))
    .collect::<Vec<_>>()
    .join(" · ")
}
pub(super) fn change_summary(v: &Value) -> String {
    let changes = a(v, "changes");
    if changes.is_empty() {
        return "No change description was supplied with this view.".into();
    }
    let extra = changes.len() - 1;
    format!(
        "{}{}",
        s(&changes[0]),
        if extra > 0 {
            format!(
                " {extra} additional comparison note{} under What was compared.",
                if extra == 1 { "" } else { "s" }
            )
        } else {
            String::new()
        }
    )
}
pub(super) fn findings(v: &Value) -> (Vec<(usize, &Value)>, String) {
    let metrics = a(v, "metrics");
    let findings = metrics
        .iter()
        .enumerate()
        .filter(|(_, m)| t(m, "decision") != "pass")
        .map(|(i, m)| (i + 1, m))
        .collect::<Vec<_>>();
    let note = if !findings.is_empty() {
        format!(
            "{} of {} metric / scope results need review.{}",
            findings.len(),
            metrics.len(),
            if findings.len() > 5 {
                " First 5 shown here; all results and checks follow below."
            } else {
                ""
            }
        )
    } else if !metrics.is_empty() {
        "No failed or unresolved metric results are recorded. This describes only the evaluated scope and requirements.".into()
    } else {
        "Metric results are unavailable in this view.".into()
    };
    (findings.into_iter().take(5).collect(), note)
}
pub(super) fn identifier_details(v: &Value) -> Vec<Value> {
    let entries=a(v,"metrics").iter().enumerate().filter_map(|(i,m)|{let name=display_label(t(m,"name"));let scope=display_label(t(m,"scope"));(name!=t(m,"name")||scope!=t(m,"scope")).then(||json!({"result":i+1,"metric":m["name"],"scope":m["scope"],"display_metric":name,"display_scope":scope}))}).collect::<Vec<_>>();
    if entries.is_empty() {
        vec![]
    } else {
        vec![json!(["Recorded metric and scope identifiers", entries])]
    }
}
pub(super) fn pretty(v: &Value, depth: usize, indent: bool) -> String {
    let (open, close, parts) = if let Some(obj) = v.as_object() {
        let mut keys = obj.keys().collect::<Vec<_>>();
        keys.sort();
        (
            "{",
            "}",
            keys.into_iter()
                .map(|k| {
                    format!(
                        "{}: {}",
                        String::from_utf8(canonical_json_no_newline(&json!(k)).unwrap()).unwrap(),
                        pretty(&obj[k], depth + 1, indent)
                    )
                })
                .collect::<Vec<_>>(),
        )
    } else if let Some(array) = v.as_array() {
        (
            "[",
            "]",
            array.iter().map(|v| pretty(v, depth + 1, indent)).collect(),
        )
    } else {
        return String::from_utf8(canonical_json_no_newline(v).unwrap()).unwrap();
    };
    if parts.is_empty() {
        return format!("{open}{close}");
    }
    if !indent {
        return format!("{open}{}{close}", parts.join(", "));
    }
    let pad = "  ".repeat(depth + 1);
    format!(
        "{open}\n{pad}{}\n{}{close}",
        parts.join(&format!(",\n{pad}")),
        "  ".repeat(depth)
    )
}
#[derive(Default)]
pub(super) struct Context {
    pub rows: Vec<(String, String, String, bool)>,
    pub matching: Vec<(String, String)>,
    pub additional: Vec<(String, String)>,
}
fn side(s: &str) -> &str {
    match s {
        "Candidate" => "Subject",
        s => s,
    }
}
pub(super) fn comparison_context(v: &Value) -> Context {
    let mut grouped = Vec::<(String, Vec<(String, String, String)>)>::new();
    let mut out = Context::default();
    for pair in a(v, "context") {
        let label = s(&pair[0]);
        let value = s(&pair[1]);
        if let Some((who, name)) = label.split_once(' ') {
            if !name.is_empty() && ["Baseline", "Subject", "Candidate"].contains(&who) {
                if let Some((_, rows)) = grouped.iter_mut().find(|(k, _)| k == name) {
                    rows.push((label.into(), side(who).into(), value.into()))
                } else {
                    grouped.push((
                        name.into(),
                        vec![(label.into(), side(who).into(), value.into())],
                    ))
                }
                continue;
            }
        }
        out.additional.push((label.into(), value.into()))
    }
    let mut paired = Vec::<(String, BTreeMap<String, String>)>::new();
    for (name, rows) in grouped {
        let sides = rows
            .iter()
            .map(|r| r.1.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if rows.len() == 2 && sides == ["Baseline", "Subject"].into_iter().collect() {
            paired.push((name, rows.into_iter().map(|(_, s, v)| (s, v)).collect()))
        } else {
            out.additional
                .extend(rows.into_iter().map(|(l, _, v)| (l, v)))
        }
    }
    let subjects = a(v, "subjects")
        .iter()
        .map(|p| {
            let (who, name) = s(&p[0]).split_once(' ').unwrap_or((s(&p[0]), ""));
            (side(who), name, s(&p[1]).to_owned())
        })
        .collect::<Vec<_>>();
    let unambiguous = subjects.len() == 2
        && subjects
            .iter()
            .map(|s| s.0)
            .collect::<std::collections::BTreeSet<_>>()
            == ["Baseline", "Subject"].into_iter().collect()
        && subjects[0].1 == subjects[1].1;
    if unambiguous {
        let mut field = subjects[0].1.to_owned();
        let mut sides = subjects
            .iter()
            .map(|s| (s.0, s.2.clone()))
            .collect::<BTreeMap<_, _>>();
        if field.is_empty() {
            for f in ["Recorded model ID", "Recorded model key", "Recorded run"] {
                let prefix = format!("{f}: ");
                if sides
                    .values()
                    .all(|s| s.starts_with(&prefix) && s.len() > prefix.len())
                {
                    field = f.into();
                    for s in sides.values_mut() {
                        *s = s[prefix.len()..].into()
                    }
                    break;
                }
            }
        }
        let observed = paired.iter().find(|(n, _)| n == "observed model");
        let deployment = paired.iter().find(|(n, _)| n == "deployment");
        let represented = field.is_empty()
            && observed.is_some()
            && deployment.is_some()
            && ["Baseline", "Subject"].iter().all(|side| {
                sides[*side]
                    == format!(
                        "Observed model: {} · Deployment: {}",
                        observed.unwrap().1[*side],
                        deployment.unwrap().1[*side]
                    )
            });
        if represented {
            let index = paired
                .iter()
                .position(|(n, _)| n == "observed model")
                .unwrap();
            let row = paired.remove(index);
            paired.insert(0, row)
        } else {
            out.rows.push((
                if field.is_empty() {
                    "Subject identity".into()
                } else {
                    capitalize(&field)
                },
                sides["Baseline"].clone(),
                sides["Subject"].clone(),
                sides["Baseline"] != sides["Subject"],
            ))
        }
    } else {
        out.additional.extend(
            a(v, "subjects")
                .iter()
                .map(|p| (s(&p[0]).into(), s(&p[1]).into())),
        )
    }
    for (name, sides) in paired {
        if sides["Baseline"] == sides["Subject"] {
            out.matching
                .push((capitalize(&name), sides["Baseline"].clone()))
        } else {
            out.rows.push((
                capitalize(&name),
                sides["Baseline"].clone(),
                sides["Subject"].clone(),
                true,
            ))
        }
    }
    out
}

pub(super) fn comma(n: u64) -> String {
    let text = n.to_string();
    text.chars()
        .enumerate()
        .flat_map(|(i, c)| {
            let mut out = String::new();
            if i > 0 && (text.len() - i).is_multiple_of(3) {
                out.push(',')
            }
            out.push(c);
            out.chars().collect::<Vec<_>>()
        })
        .collect()
}
pub(super) fn check(
    name: &str,
    observed: String,
    required: String,
    passed: Value,
    explanation: &str,
) -> Value {
    json!({"name":name,"observed":observed,"required":required,"passed":passed,"explanation":explanation})
}
pub(super) fn empty_view() -> Value {
    json!({"title":"","family":"","decision":"","summary":"","metrics":[],"assurance":[],"identity":[],"subjects":[],"next_steps":[],"limitations":[],"details":[],"technical":{},"context":[],"changes":[],"review":null,"handoff_links":[]})
}
