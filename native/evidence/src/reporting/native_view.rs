//! Presentation of already authenticated, semantically checked native report facts.
use super::presentation::*;
use crate::{EvidenceError, Result};
use serde_json::{json, Value};
fn n(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn count(v: &Value) -> String {
    comma(v.as_u64().unwrap())
}
fn checks(r: &Value) -> Result<Vec<Value>> {
    let c = &r["comparison"];
    let u = &r["uncertainty"];
    let f = |v: &Value| number_precision(n(v), false, 6);
    let mut out = match t(c, "kind") {
        "exact_match_delta_pp" | "scorer_extension_delta_pp" => vec![check(
            if u["scope"] == "authenticated_schedule" {
                "Finite-schedule lower bound"
            } else {
                "Paired lower bound"
            },
            format!("{} pp", f(&u["lower"])),
            format!(">= {} pp", f(&c["minimum"])),
            json!(n(&u["lower"]) >= n(&c["minimum"])),
            "The lower bound must meet the allowed change, not just the point estimate.",
        )],
        "normalized_nll_ratio" => vec![check(
            "Finite-schedule upper ratio bound",
            f(&u["upper"]),
            format!("<= {}", f(&c["maximum"])),
            json!(n(&u["upper"]) <= n(&c["maximum"])),
            "The upper ratio bound must stay within the allowed loss increase.",
        )],
        _ => {
            return Err(EvidenceError::contract(
                "unsupported canonical comparison kind",
            ))
        }
    };
    if r["sample_qualification"].is_object() {
        let sample = &r["sample_qualification"];
        let count = &sample["record_count"];
        let w = &sample["interval_width"];
        let unit = if w["unit"] == "percentage_points" {
            " pp"
        } else {
            " ratio"
        };
        out.push(check(
            "Record count",
            count["observed"].to_string(),
            format!(">= {}", count["minimum"]),
            count["passed"].clone(),
            "The comparison must include the configured minimum number of paired records.",
        ));
        out.push(check(
            "Interval width",
            format!("{}{unit}", f(&w["observed"])),
            format!("<= {}{unit}", f(&w["maximum"])),
            w["passed"].clone(),
            "The interval must be narrow enough for the configured precision requirement.",
        ))
    }
    if r["side_accuracy"].is_object() {
        let acc = &r["side_accuracy"];
        for (side, label) in [
            ("baseline", "Baseline accuracy"),
            ("subject", "Candidate accuracy"),
        ] {
            out.push(check(
                label,
                format!(
                    "{}%",
                    number_precision(n(&acc[side]["observed"]) * 100.0, false, 6)
                ),
                format!(
                    ">= {}%",
                    number_precision(n(&acc["minimum"]) * 100.0, false, 6)
                ),
                acc[side]["passed"].clone(),
                "Each side must meet the configured absolute accuracy floor.",
            ))
        }
    }
    Ok(out)
}
fn summary(r: &Value, checks: &[Value]) -> String {
    if r["verdict"] == "pass" {
        return "The uncertainty interval is within the required range, and the other configured checks passed.".into();
    }
    let c = &r["comparison"];
    let value = n(&c["value"]);
    if checks[0]["passed"] != true {
        match t(c,"kind"){"exact_match_delta_pp"=>{let min=n(&c["minimum"]);return if min<=value&&value<0.0{format!("The observed loss was within the allowance, but a loss greater than {} percentage points could not be ruled out.",number(-min,false))}else if min<=0.0&&value>=0.0{"The observed score did not decline, but a loss beyond the policy allowance could not be ruled out.".into()}else if value<min&&min<=0.0{"The observed loss exceeded the policy allowance.".into()}else{"The comparison did not establish the improvement required by the policy.".into()}},"normalized_nll_ratio"=>return if value<=n(&c["maximum"]){"The observed subject-to-baseline NLL ratio was within the limit, but its uncertainty interval extended beyond it."}else{"The observed subject-to-baseline NLL ratio exceeded the policy limit."}.into(),_=>return "The comparison did not establish the score change required by the policy.".into()}
    }
    if let Some(check) = checks.iter().skip(1).find(|c| c["passed"] == false) {
        return match t(check, "name") {
            "Record count" => "There were too few paired records to meet the policy requirement.",
            "Interval width" => "The uncertainty interval was wider than the policy permits.",
            "Baseline accuracy" => "The baseline accuracy was below the required minimum.",
            "Candidate accuracy" => "The subject accuracy was below the required minimum.",
            _ => "",
        }
        .into();
    }
    String::new()
}
pub fn build_native_view(
    r: &Value,
    evidence_signer: &str,
    observations: &Value,
    subjects: &Value,
    context: &Value,
    changes: &Value,
) -> Result<Value> {
    let c = &r["comparison"];
    let u = &r["uncertainty"];
    let exact = c["kind"] == "exact_match_delta_pp";
    let ratio = c["kind"] == "normalized_nll_ratio";
    let checks = checks(r)?;
    let mut summary = summary(r, &checks);
    let mut notes = vec![
        json!(if ratio {
            "Lower scores are better; the change is the candidate-to-baseline ratio."
        } else {
            "Higher scores are better; the change is candidate minus baseline in percentage points."
        }),
        json!("The policy tests the interval bound, not just the observed change."),
    ];
    if r.get("sample_qualification").is_none() {
        notes.push(json!(
            "This policy does not specify a minimum sample count or interval-width requirement."
        ))
    }
    if exact && r.get("side_accuracy").is_none() {
        notes.push(json!(
            "This policy does not specify an absolute accuracy floor."
        ))
    }
    let mut details = Vec::new();
    if let Some(p) = r.get("paired_binary") {
        details.push(json!(["Paired outcome analysis", p]))
    }
    if let Some(d) = r.get("derived_measurements") {
        details.push(json!(["Derived likelihood interpretation", d]));
        let p = &d["perplexity_ratio"];
        if p["status"] == "available" {
            notes.extend([
                json!(format!(
                    "Baseline perplexity: {}.",
                    number(n(&p["baseline_perplexity"]), false)
                )),
                json!(format!(
                    "Candidate perplexity: {}.",
                    number(n(&p["subject_perplexity"]), false)
                )),
                json!(format!(
                    "Perplexity ratio: {}. These derived values do not affect acceptance.",
                    number(n(&p["ratio"]), false)
                )),
            ])
        } else {
            notes.push(json!(format!("Perplexity interpretation unavailable: {}. Acceptance uses normalized NLL per expected UTF-8 byte.",t(p,"reason").replace('_'," "))))
        }
    }
    if observations.as_array().is_some_and(|v| !v.is_empty()) {
        details.push(json!(["Authenticated observations", observations]));
        notes.push(json!("Authenticated observations are supplementary; the paired metric and policy remain the complete acceptance calculation."))
    }
    let count = count(&r["record_count"]);
    let paired = &r["paired_binary"];
    let baseline_matches = if exact {
        Some(
            paired["both_pass"].as_u64().unwrap()
                + paired["baseline_pass_subject_fail"].as_u64().unwrap(),
        )
    } else {
        None
    };
    let subject_matches = if exact {
        Some(
            paired["both_pass"].as_u64().unwrap()
                + paired["baseline_fail_subject_pass"].as_u64().unwrap(),
        )
    } else {
        None
    };
    let basis = if exact {
        json!([format!("Baseline and subject are paired on the same {count} cases. Newcombe hybrid score uses their paired match outcomes to form a nominal 95% confidence interval for the accuracy change."),format!("Both matched: {}; baseline only: {}; subject only: {}; neither matched: {}.",self::count(&paired["both_pass"]),self::count(&paired["baseline_pass_subject_fail"]),self::count(&paired["baseline_fail_subject_pass"]),self::count(&paired["both_fail"])),"Change is subject accuracy minus baseline accuracy in percentage points, not relative percent change. The 95% confidence level describes the interval method, not the share of correct answers.","This exact-match method uses a fixed 95% confidence level; the allowed-loss threshold is a separate policy choice. Under suitable sampling and independent-pair assumptions, the method aims for intervals to cover the underlying accuracy difference in about 95% of repeated studies. This is not the probability that the subject exceeds the allowed loss, and these cases are not automatically representative of production."])
    } else {
        json!([format!("The declared percentile method resamples the same {count} paired records with replacement, keeping each baseline and subject result together. Its retained resampling count is {}.",self::count(&u["replicates"])),if ratio{"The interval covers the central 95% of the paired resampling distribution for the subject-to-baseline mean NLL ratio. Each side's NLL uses nats per expected UTF-8 byte."}else{"The interval covers the central 95% of the paired resampling distribution for subject-minus-baseline mean score change, in the displayed change units."},"This describes resampling of the fixed recorded schedule; it is not a population confidence interval or a claim that the sample represents production traffic."])
    };
    let unit = if ratio { "ratio" } else { "pp" };
    let suffix = if exact {
        "%"
    } else if ratio {
        " nats / byte"
    } else {
        " score"
    };
    let scale = if exact { 100.0 } else { 1.0 };
    let baseline = format!(
        "{}{suffix}",
        number(n(&r["baseline"]["mean_score"]) * scale, false)
    );
    let subject = format!(
        "{}{suffix}",
        number(n(&r["subject"]["mean_score"]) * scale, false)
    );
    let change = format!("{} {unit}", number(n(&c["value"]), !ratio));
    let metric = json!({"name":match t(r,"metric"){"exact_match"=>"Exact-match accuracy","normalized_nll_per_utf8_byte"=>"Normalized negative log-likelihood",s=>s},"scope":"All paired records","decision":r["verdict"],"baseline":baseline,"candidate":subject,"change":change,"count":count,"explanation":summary,"checks":checks,"interval":{"lower":u["lower"],"upper":u["upper"],"estimate":c["value"],"threshold":c[if ratio{"maximum"}else{"minimum"}],"label":if u["scope"]=="paired_binary_outcomes"{"Paired 95% confidence interval"}else{"95% finite-schedule resampling interval"},"unit":unit,"threshold_direction":if ratio{"maximum"}else{"minimum"},"neutral":if ratio{1.0}else{0.0},"method":if exact{"Newcombe hybrid score"}else{"Paired percentile resampling"},"basis":basis},"notes":notes,"baseline_detail":baseline_matches.map(|m|format!("{} of {count} matched",comma(m))).unwrap_or_default(),"candidate_detail":subject_matches.map(|m|format!("{} of {count} matched",comma(m))).unwrap_or_default(),"count_detail":"","count_label":"Observed pairs","evidence_source":"reports/evaluation.report.json#"});
    if exact {
        summary+=&format!(" The subject matched the expected answer in {} of {count} cases, compared with {} for the baseline. Accuracy changed from {baseline} to {subject}, a change of {change}.",comma(subject_matches.unwrap()),comma(baseline_matches.unwrap()))
    } else {
        summary+=&format!(" Across {count} paired records, the subject scored {subject}, compared with {baseline} for the baseline, {}.",if ratio{format!("a subject-to-baseline ratio of {}",number(n(&c["value"]),false))}else{format!("a change of {change}")})
    }
    let mut view = empty_view();
    view["title"] = json!("InvarLock comparison report");
    view["family"] = json!("Runtime evaluation evidence");
    view["decision"] = r["verdict"].clone();
    view["summary"] = json!(summary);
    view["metrics"] = json!([metric]);
    view["assurance"] = json!([
        [
            "Bundle integrity",
            "Inventory, checksums and embedded evidence signature verified."
        ],
        [
            "Recorded policy result",
            "Read from the authenticated canonical report."
        ],
        [
            "Independent recipient acceptance",
            "Not performed by report. An embedded signer is not a recipient-owned trust anchor."
        ]
    ]);
    view["identity"] = json!([
        ["Comparison", r["comparison_id"]],
        ["Metric", r["metric"]],
        ["Policy", r["policy_digest"]],
        ["Evidence signer", evidence_signer]
    ]);
    view["subjects"] = subjects.clone();
    view["context"] = context.clone();
    view["changes"] = changes.clone();
    view["next_steps"]=json!(["Review the decision checks and their requirements.","Run invarlock verify with your independently supplied trust profile and receipt destination to create the signed acceptance or rejection receipt.","Keep this report alongside the original immutable evidence and the separate verification receipt."]);
    view["limitations"]=json!(["This report summarizes the evidence bundle. Rendering checks its integrity and embedded signature; independent acceptance requires verification with your own trust inputs.","Results apply to the recorded cases, metric and policy. A pass does not establish general model quality, safety or representative production performance.",if exact{"The interval describes paired binary outcomes under its stated method. Population interpretation requires an appropriate sampling design."}else{"The resampling interval describes stability on the authenticated schedule, not population uncertainty or representativeness."}]);
    view["details"] = json!(details);
    view["technical"] = r.clone();
    Ok(view)
}
