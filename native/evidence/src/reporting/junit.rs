use super::presentation::{a, s, t, visible_controls};
use crate::Result;
use serde_json::Value;
pub(super) fn xml(s: &str) -> String {
    visible_controls(s)
        .chars()
        .map(|c| {
            if c == '\u{fffe}' || c == '\u{ffff}' {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect::<String>()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\t', "&#09;")
        .replace('\n', "&#10;")
        .replace('\r', "&#13;")
}
pub fn render_junit(comparison: &Value) -> Result<Vec<u8>> {
    invarlock_native_contracts::validate_schema(comparison, "comparison")?;
    let metrics = a(comparison, "metrics");
    let mut out=format!("<?xml version='1.0' encoding='utf-8'?>\n<testsuite name=\"InvarLock release comparison\" tests=\"{}\" failures=\"{}\" errors=\"{}\"",metrics.len(),metrics.iter().filter(|m|t(m,"decision")=="regression").count(),metrics.iter().filter(|m|t(m,"decision")=="insufficient_evidence").count());
    if metrics.is_empty() {
        out += " />"
    } else {
        out.push('>');
        for m in metrics {
            out += &format!(
                "<testcase name=\"{}\" classname=\"{}\"",
                xml(t(m, "name")),
                xml(t(m, "slice"))
            );
            if t(m, "decision") == "pass" {
                out += " />"
            } else {
                out += &format!(
                    "><{} message=\"{}\" /></testcase>",
                    if t(m, "decision") == "regression" {
                        "failure"
                    } else {
                        "error"
                    },
                    xml(&a(m, "reasons").iter().map(s).collect::<Vec<_>>().join("; "))
                )
            }
        }
        out += "</testsuite>"
    }
    Ok(out.into_bytes())
}

pub(super) fn recorded(view: &Value) -> Vec<u8> {
    let metrics = a(view, "metrics");
    let mut out=format!("<?xml version='1.0' encoding='utf-8'?>\n<testsuite name=\"InvarLock recorded policy checks\" tests=\"{}\" failures=\"{}\" errors=\"{}\"",metrics.len(),metrics.iter().filter(|m|["fail","regression"].contains(&t(m,"decision"))).count(),metrics.iter().filter(|m|m["decision"]=="insufficient_evidence").count());
    if metrics.is_empty() {
        out += " />";
    } else {
        out.push('>');
        for m in metrics {
            out += &format!(
                "<testcase name=\"{}\" classname=\"{}\"",
                xml(t(m, "name")),
                xml(t(m, "scope"))
            );
            if m["decision"] == "pass" {
                out += " />";
            } else {
                out += &format!(
                    "><{} message=\"{}\" /></testcase>",
                    if m["decision"] == "insufficient_evidence" {
                        "error"
                    } else {
                        "failure"
                    },
                    xml(t(m, "explanation"))
                );
            }
        }
        out += "</testsuite>";
    }
    out.into_bytes()
}
pub(super) fn judge(view: &Value, facts: &Value) -> Vec<u8> {
    let required = facts["assurance"]["decision_role"] == "required";
    let decision = t(view, "decision");
    let mut out=format!("<?xml version='1.0' encoding='utf-8'?>\n<testsuite name=\"InvarLock bounded judge policy\" tests=\"1\" failures=\"{}\" errors=\"{}\" skipped=\"{}\"><properties><property name=\"decision_role\" value=\"{}\" /><property name=\"policy_decision\" value=\"{}\" /></properties><testcase name=\"{}\" classname=\"bounded-judge-fixed-benchmark-v1\">",u8::from(required&&decision=="regression"),u8::from(required&&decision=="insufficient_evidence"),u8::from(!required),if required{"required"}else{"advisory"},xml(decision),xml(t(&view["metrics"][0],"name")));
    if !required {
        out+=&format!("<skipped message=\"Advisory metric: {decision}; required decisions are not gated.\" />")
    } else if decision != "pass" {
        out += &format!(
            "<{} message=\"{}\" />",
            if decision == "regression" {
                "failure"
            } else {
                "error"
            },
            xml(t(&view["metrics"][0], "explanation"))
        )
    }
    out+="<system-out>Offline replay completed. Recipient signer authorization and acceptance were not performed.</system-out></testcase></testsuite>";
    out.into_bytes()
}
