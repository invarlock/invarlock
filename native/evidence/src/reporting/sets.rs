//! Combined component presentation; this view does not authorize recipients.
use super::presentation::{a, decision, display_label, empty_view, t, visible_controls};
use crate::{evidence_sets, EvidenceError, Result};
use invarlock_native_contracts::sha256_digest;
use serde_json::{json, Value};
use std::path::Path;
const STATISTICAL_SCOPE: &str = "component-methods-no-joint-confidence";

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    index: &Value,
    raw: &[u8],
    first: &Value,
    second: &Value,
    shared: &Value,
    manifest: &Value,
    envelope: &Value,
    judge_facts: &Value,
) -> (Value, Value, String) {
    let verdict = if [first, second]
        .iter()
        .any(|v| v["decision"] == "regression")
    {
        "regression"
    } else if [first, second]
        .iter()
        .any(|v| v["decision"] == "insufficient_evidence")
    {
        "insufficient_evidence"
    } else {
        "pass"
    };
    let mut metrics = Vec::new();
    for (label, member, component) in [
        ("Deterministic", "deterministic", first),
        ("Judge", "judge", second),
    ] {
        for metric in a(component, "metrics") {
            let mut metric = metric.clone();
            metric["name"] = json!(format!("{label} · {}", t(&metric, "name")));
            metric["evidence_source"] = json!(format!(
                "{}/{}",
                t(&index["members"][member], "path"),
                t(&metric, "evidence_source")
            ));
            if member == "deterministic" {
                let mut notes = a(&metric, "notes").to_vec();
                notes.push(json!(
                    "Original cases; each interval applies to this component only."
                ));
                metric["notes"] = json!(notes);
            }
            metrics.push(metric);
        }
    }
    let mut causes = Vec::new();
    for (label, component) in [("Deterministic", first), ("Judge", second)] {
        if component["decision"] == "pass" {
            continue;
        }
        if let Some(m) = a(component, "metrics")
            .iter()
            .find(|m| m["decision"] == component["decision"])
        {
            if !t(m, "explanation").is_empty() {
                causes.push(format!(
                    "{label} finding for {} ({}): {}",
                    display_label(t(m, "name")),
                    display_label(t(m, "scope")),
                    t(m, "explanation")
                ));
            }
        }
    }
    let findings = if causes.is_empty() {
        String::new()
    } else {
        causes.join(" ") + " "
    };
    let mut view = empty_view();
    view["title"] = json!("InvarLock combined comparison report");
    view["family"] = json!("Deterministic and bounded judge evidence");
    view["decision"] = json!(verdict);
    view["summary"] = json!(format!("Deterministic comparison: {}. Judge comparison: {}. {findings}Both components must satisfy their recorded policies for the combined result to pass. They compare the same frozen baseline and subject answers using separate statistical methods.",decision(t(first,"decision")).to_lowercase(),decision(t(second,"decision")).to_lowercase()));
    view["metrics"] = json!(metrics);
    view["assurance"] = json!([
        ["Deterministic signature",if manifest["authentication"] == "signed" { "Signed manifest verified; no recipient-owned key was supplied." } else { "Unsigned local evidence." }],
        ["Deterministic scoring replay","Not performed by report; recorded component result shown."],
        ["Judge signature",if !envelope["signature"].is_null() { "Signature present; recipient authorization not performed." } else { "Unsigned local evidence." }],
        ["Judge measurement replay","Recomputed offline from retained measurements."],
        ["Shared inputs","Same complete baseline and subject runs, original case set, and evaluated subject identity."],
        ["Recipient acceptance","Not performed by report."]
    ]);
    view["subjects"] = first["subjects"].clone();
    view["changes"] = first["changes"].clone();
    let mut context = Vec::new();
    for value in a(first, "context").iter().chain(a(second, "context")) {
        if !context.contains(value) {
            context.push(value.clone());
        }
    }
    view["context"] = json!(context);
    let subject_field = if shared.get("subject_service_identity_sha256").is_some() {
        "subject_service_identity_sha256"
    } else {
        "subject_artifact_sha256"
    };
    let digest = sha256_digest(raw);
    view["identity"] = json!([
        ["Evidence set", digest],
        ["Baseline run", shared["baseline_run_sha256"]],
        ["Subject run", shared["subject_run_sha256"]],
        ["Original case set", shared["case_set_sha256"]],
        [
            if subject_field == "subject_service_identity_sha256" {
                "Subject service identity"
            } else {
                "Subject artifact"
            },
            shared[subject_field]
        ]
    ]);
    view["next_steps"] = json!(["Before accepting the combined result, verify both components with your own evidence-set recipient policy and trust inputs."]);
    let mut limitations = vec![json!("Both required components must pass. Their separate intervals provide no joint confidence guarantee."),json!("Deterministic metrics count original cases. Judge repetitions do not increase that count or the number of independent units.")];
    limitations.extend(
        a(first, "limitations")
            .iter()
            .chain(a(second, "limitations"))
            .cloned(),
    );
    view["limitations"] = json!(limitations);
    let details = [("Deterministic", first), ("Judge", second)]
        .iter()
        .flat_map(|(label, v)| {
            a(v, "details")
                .iter()
                .map(move |d| json!([format!("{label} · {}", d[0].as_str().unwrap_or("")), d[1]]))
        })
        .collect::<Vec<_>>();
    view["details"] = json!(details);
    view["technical"] = json!({"statistical_scope":STATISTICAL_SCOPE,"deterministic":first["technical"],"judge":second["technical"]});
    let facts = json!({"decision":verdict,"statistical_scope":STATISTICAL_SCOPE,"recipient_acceptance":"not_performed","shared_inputs":shared,"component_decisions":{"deterministic":first["decision"],"judge":second["decision"]},"judge":judge_facts["judge"]});
    (view, facts, digest)
}

fn xml(value: &str) -> String {
    visible_controls(value)
        .chars()
        .map(|c| {
            if matches!(c, '\u{fffe}' | '\u{ffff}') {
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
pub fn render_junit(view: &Value) -> Vec<u8> {
    let metrics = a(view, "metrics");
    let mut out = format!("<?xml version='1.0' encoding='utf-8'?>\n<testsuite name=\"InvarLock combined component policy\" tests=\"{}\" failures=\"{}\" errors=\"{}\"",metrics.len(),metrics.iter().filter(|m|m["decision"]=="regression").count(),metrics.iter().filter(|m|m["decision"]=="insufficient_evidence").count());
    if metrics.is_empty() {
        out += " />";
    } else {
        out += ">";
        for metric in metrics {
            out += &format!(
                "<testcase name=\"{}\" classname=\"same-answer-component-conjunction-v1\">",
                xml(t(metric, "name"))
            );
            if metric["decision"] != "pass" {
                out += &format!(
                    "<{} message=\"{}\" />",
                    if metric["decision"] == "regression" {
                        "failure"
                    } else {
                        "error"
                    },
                    xml(t(metric, "explanation"))
                );
            }
            out += "<system-out>Component policy result only; recipient acceptance not performed. No joint confidence guarantee.</system-out></testcase>";
        }
        out += "</testsuite>";
    }
    out.into_bytes()
}

/// Build from retained, authenticated component snapshots without recipient authorization.
pub fn load(path: &Path, case_ids: Option<&[String]>) -> Result<Value> {
    let root = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| EvidenceError::contract(e.to_string()))?
            .join(path)
    };
    let pinned = invarlock_native_filesystem::PinnedDirectory::open(&root, false)
        .map_err(|e| EvidenceError::contract(e.to_string()))?;
    let (index, raw) = evidence_sets::read_object(&root.join("evidence-set.json"), 256 * 1024)?;
    evidence_sets::validate(&index, "evidence_set", "evidence set index")?;
    let paths = evidence_sets::index_paths(&root, &index)?;
    evidence_sets::check_statements(&root, &index)?;
    let captured = super::captured_view::load(&paths["deterministic"])?;
    let manifest = &captured["manifest"];
    if sha256_digest(&crate::canonical(manifest)?)
        != index["members"]["deterministic"]["statement_sha256"]
    {
        return Err(EvidenceError::contract(
            "captured statement changed during report preparation",
        ));
    }
    let payloads = captured["payloads"]
        .as_object()
        .ok_or_else(|| EvidenceError::contract("captured report snapshot is invalid"))?;
    evidence_sets::require_deterministic_policy(&payloads["policy"])?;
    let shared = evidence_sets::shared_captured_inputs(payloads)?;
    let (expected_envelope, envelope_raw) =
        evidence_sets::read_object(&paths["judge"].join("envelope.json"), 256 * 1024)?;
    if sha256_digest(&envelope_raw) != index["members"]["judge"]["statement_sha256"] {
        return Err(EvidenceError::contract(
            "judge statement changed before rendering",
        ));
    }
    let judge = super::judge_view::load(&paths["judge"], case_ids)?;
    let envelope = &judge["envelope"];
    if crate::canonical(envelope)? != crate::canonical(&expected_envelope)? {
        return Err(EvidenceError::contract(
            "judge statement changed during rendering",
        ));
    }
    let fields = [
        "baseline_run_sha256",
        "subject_run_sha256",
        "case_set_sha256",
    ];
    let observed = fields
        .iter()
        .map(|f| (f.to_string(), envelope["bindings"][f].clone()))
        .collect::<serde_json::Map<_, _>>();
    let expected = fields
        .iter()
        .map(|f| (f.to_string(), shared[f].clone()))
        .collect::<serde_json::Map<_, _>>();
    evidence_sets::require_shared(
        &json!(observed),
        &json!(expected),
        "component runs or case sets differ",
    )?;
    let field = if shared.get("subject_service_identity_sha256").is_some() {
        "subject_service_identity_sha256"
    } else {
        "subject_artifact_sha256"
    };
    evidence_sets::require_shared(
        &json!({field:envelope["intended_subject"]}),
        &json!({field:shared[field]}),
        "component subject identities differ",
    )?;
    if judge["artifacts"]["analysis_policy"]["decision_role"] != "required" {
        return Err(EvidenceError::contract(
            "evidence set requires a required judge policy",
        ));
    }
    evidence_sets::check_statements(&root, &index)?;
    if evidence_sets::read(&root.join("evidence-set.json"), 256 * 1024)? != raw {
        return Err(EvidenceError::contract(
            "evidence set index changed during rendering",
        ));
    }
    pinned
        .verify()
        .map_err(|e| EvidenceError::contract(e.to_string()))?;
    let (view, facts, digest) = assemble(
        &index,
        &raw,
        &captured["view"],
        &judge["view"],
        &shared,
        manifest,
        envelope,
        &judge["facts"],
    );
    Ok(json!({"view":view,"facts":facts,"index_sha256":digest}))
}

/// Render all requested formats from one checked view, then publish without replacement.
pub fn render(config: &Value) -> super::publication::ReportOutput {
    use super::publication::{self, PublicationError, ReportOutput};
    let requested = [
        ("html", "html_path"),
        ("markdown", "markdown_path"),
        ("junit", "junit_path"),
    ]
    .into_iter()
    .filter_map(|(name, key)| {
        config[key]
            .as_str()
            .map(|p| (name.to_string(), std::path::PathBuf::from(p)))
    })
    .collect::<Vec<_>>();
    let requested_value = requested
        .iter()
        .map(|(name, path)| (name.clone(), json!(path.to_string_lossy())))
        .collect::<serde_json::Map<_, _>>();
    let run = || -> std::result::Result<ReportOutput, PublicationError> {
        let evidence = Path::new(
            config["evidence_path"]
                .as_str()
                .ok_or_else(|| PublicationError::new("evidence_path is required"))?,
        );
        publication::validate(&requested, Some(evidence))?;
        let ids = config["case_ids"]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let loaded = load(evidence, Some(&ids)).map_err(|e| PublicationError::new(e.message))?;
        let view = &loaded["view"];
        let text = super::render_markdown(view, config["explain"].as_bool().unwrap_or(false));
        let rendered = requested
            .iter()
            .map(|(name, _)| match name.as_str() {
                "html" => super::render_html(view).into_bytes(),
                "markdown" => text.as_bytes().to_vec(),
                "junit" => render_junit(view),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let written = publication::publish(&requested, &rendered, Some(evidence))?;
        let mut payload = json!({"format_version":"invarlock/evidence-set-report-v1","kind":"evidence_set","ok":true,"index_sha256":loaded["index_sha256"],"requested_outputs":requested_value,"written_outputs":written,"errors":[]});
        if let Some(facts) = loaded["facts"].as_object() {
            payload.as_object_mut().unwrap().extend(facts.clone());
        }
        Ok(ReportOutput {
            text,
            payload,
            status: 0,
        })
    };
    match run() {
        Ok(output) => output,
        Err(error) => ReportOutput {
            text: String::new(),
            status: 2,
            payload: json!({"format_version":"invarlock/evidence-set-report-v1","kind":"evidence_set","ok":false,"errors":[error.message.chars().take(1024).collect::<String>()],"requested_outputs":requested_value,"written_outputs":error.written_outputs,"failed_output":error.failed_output}),
        },
    }
}
