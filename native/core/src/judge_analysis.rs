//! Analysis of authenticated retained judge measurements.
use crate::judge_contracts::{digest, validate_judge_analysis_policy_prechecked};
use crate::scoring::decimal;
use crate::{judge_statistics, validate_measurements, AnalysisError};
use num_rational::BigRational as Rat;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap()
}
const ASSUMPTIONS: [&str; 6] = [
    "Declared units are independent; distributions may differ between units.",
    "Scores remain within the declared rating scale under the frozen judge protocol.",
    "Cases, answers, unit grouping, schedule, and decision policy are fixed before judging.",
    "Units have equal weight; cases within units and repetitions within cases have equal weight.",
    "Repetitions and cases within a unit do not increase the number of independent units.",
    "The declared comparison family includes both published intervals and all enclosing claims.",
];
const ESTIMAND:&str="Equal-unit average expected judge score on the fixed scheduled benchmark; the paired effect is subject minus baseline. No population generalization is asserted.";
fn fraction(v: &Value) -> Result<Rat, AnalysisError> {
    Ok(Rat::new(
        text(v, "numerator")
            .parse()
            .map_err(|_| err("invalid fraction numerator"))?,
        text(v, "denominator")
            .parse()
            .map_err(|_| err("invalid fraction denominator"))?,
    ))
}
fn fraction_json(v: Rat) -> Value {
    json!({"numerator":v.numer().to_string(),"denominator":v.denom().to_string()})
}
fn gate(name: &str, interval: &Value, policy: &Value) -> Result<Value, AnalysisError> {
    let mut reasons = Vec::new();
    if crate::analysis::py_number(&interval["unit_count"])
        < crate::analysis::py_number(&policy["minimum_units"])
    {
        reasons.push("minimum_units_not_met")
    }
    if decimal(text(interval, "upper"))? - decimal(text(interval, "lower"))?
        > decimal(text(policy, "maximum_interval_width"))?
    {
        reasons.push("maximum_interval_width_exceeded")
    }
    if !reasons.is_empty() {
        return Ok(json!({"name":name,"decision":"insufficient_evidence","reasons":reasons}));
    }
    let mut request = json!({"statistic":if name=="paired_effect"{"decide_effect"}else{"decide_subject_bound"},"interval":interval,"direction":policy["direction"]});
    if name == "paired_effect" {
        request["margin"] = policy["allowed_degradation"].clone()
    } else {
        request["bound"] = policy["subject_bound"].clone()
    }
    let decision = judge_statistics(&request)?;
    Ok(
        json!({"name":name,"decision":decision,"reasons":if decision=="insufficient_evidence"{vec!["interval_crosses_decision_threshold"]}else{vec![]}}),
    )
}
/// Replay every retained observation and frozen-answer binding before inference.
pub fn analyze_measurements(
    plan: &Value,
    data: &Value,
    policy: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<Value, AnalysisError> {
    validate_measurements(data, plan, baseline, subject)?;
    validate_judge_analysis_policy_prechecked(policy, plan)?;
    let cases = plan["sampling"]["case_units"].as_array().unwrap();
    let mapping = cases
        .iter()
        .map(|c| (text(c, "case_id"), text(c, "unit_id")))
        .collect::<BTreeMap<_, _>>();
    let mut expected = BTreeMap::<&str, usize>::new();
    for unit in mapping.values() {
        *expected.entry(unit).or_default() += 1
    }
    let mut completed = BTreeMap::<&str, usize>::new();
    let trials = data["trials"].as_array().unwrap();
    for t in trials {
        if t["status"] == "complete" && t["parse"]["status"] == "ok" {
            *completed.entry(mapping[text(t, "case_id")]).or_default() += 1
        }
    }
    let repetitions = plan["schedule"]["repetitions"].as_u64().unwrap();
    let complete_units = expected
        .iter()
        .filter(|(u, n)| completed.get(*u).copied().unwrap_or(0) == **n * repetitions as usize * 2)
        .count();
    let completed_trials = completed.values().sum::<usize>();
    let expected_trials = plan["schedule"]["expected_trials"].as_u64().unwrap();
    let incomplete = expected_trials - completed_trials as u64;
    let counts = json!({"expected_trials":expected_trials,"recorded_trials":trials.len(),"completed_trials":completed_trials,"incomplete_trials":incomplete,"scheduled_cases":cases.len(),"scheduled_units":expected.len(),"complete_units":complete_units,"repetitions":repetitions});
    let mut effect = Value::Null;
    let mut subject_interval = Value::Null;
    let mut gates = Vec::new();
    let mut reasons = BTreeSet::new();
    let decision;
    if incomplete > 0 {
        decision = json!("insufficient_evidence");
        reasons.insert("incomplete_planned_schedule".to_owned());
    } else {
        let ratings = plan["scale"]["ratings"].as_array().unwrap();
        let mut bounds = ratings
            .iter()
            .map(|r| Ok((decimal(text(r, "value"))?, text(r, "value"))))
            .collect::<Result<Vec<_>, AnalysisError>>()?;
        bounds.sort_by(|a, b| a.0.cmp(&b.0));
        let lower = bounds.first().unwrap();
        let upper = bounds.last().unwrap();
        let width = &upper.0 - &lower.0;
        let collapsed = judge_statistics(
            &json!({"statistic":"collapse_scores","cases":cases,"scores":trials.iter().map(|t|json!({"case_id":t["case_id"],"side":t["side"],"repetition":t["repetition"].as_u64().unwrap()-1,"value":t["parse"]["value"]})).collect::<Vec<_>>(),"repetitions":repetitions,"lower_bound":lower.1,"upper_bound":upper.1}),
        )?;
        let units = collapsed["units"].as_array().unwrap();
        let effects = units
            .iter()
            .map(|u| {
                Ok(fraction_json(
                    fraction(&u["subject"])? - fraction(&u["baseline"])?,
                ))
            })
            .collect::<Result<Vec<_>, AnalysisError>>()?;
        // Scale endpoints have at most 15 fractional digits in the wire contract.
        let width_text = crate::judge::terminating_string(&width);
        effect = judge_statistics(
            &json!({"statistic":"hoeffding_interval","values":effects,"lower_bound":format!("-{width_text}"),"upper_bound":width_text,"alpha":policy["alpha"],"comparisons":policy["comparison_family_size"]}),
        )?;
        subject_interval = judge_statistics(
            &json!({"statistic":"hoeffding_interval","values":units.iter().map(|u|u["subject"].clone()).collect::<Vec<_>>(),"lower_bound":lower.1,"upper_bound":upper.1,"alpha":policy["alpha"],"comparisons":policy["comparison_family_size"]}),
        )?;
        gates.push(gate("paired_effect", &effect, policy)?);
        if !policy["subject_bound"].is_null() {
            gates.push(gate("subject_bound", &subject_interval, policy)?)
        }
        decision = judge_statistics(
            &json!({"statistic":"combine_decisions","gates":gates.iter().map(|g|json!({"decision":g["decision"],"required":true})).collect::<Vec<_>>()}),
        )?;
        for g in &gates {
            for reason in g["reasons"].as_array().unwrap() {
                reasons.insert(reason.as_str().unwrap().to_owned());
            }
        }
    }
    let decoded_policy = json!({"direction":policy["direction"],"allowed_degradation":policy["allowed_degradation"],"alpha":policy["alpha"],"comparison_family_size":policy["comparison_family_size"],"required":policy["decision_role"]=="required","minimum_units":policy["minimum_units"],"maximum_interval_width":policy["maximum_interval_width"],"subject_bound":policy["subject_bound"],"plan_sha256":policy["plan_sha256"],"metric_name":policy["metric_name"]});
    Ok(
        json!({"plan_sha256":data["plan_sha256"],"measurements_sha256":digest(data)?,"policy":decoded_policy,"sampling_basis":plan["sampling"]["basis"],"counts":counts,"effect_interval":effect,"subject_interval":subject_interval,"subject_interval_role":if policy["subject_bound"].is_null(){"descriptive"}else{"decision"},"gates":gates,"decision":decision,"reasons":reasons.into_iter().collect::<Vec<_>>(),"method":"fixed-benchmark-hoeffding-v1","estimand":ESTIMAND,"assumptions":ASSUMPTIONS}),
    )
}
pub fn combine_judge_analysis_results(results: &Value) -> Result<Value, AnalysisError> {
    let family = results
        .as_array()
        .ok_or_else(|| err("analysis results must be an array"))?;
    if let Some(first) = family.first() {
        let alpha = decimal(text(&first["policy"], "alpha"))?;
        for result in family {
            if decimal(text(&result["policy"], "alpha"))? != alpha {
                return Err(err("combined results must declare the same family alpha"));
            }
            if crate::analysis::py_number(&result["policy"]["comparison_family_size"])
                < crate::analysis::py_number(&json!(2 * family.len()))
            {
                return Err(err("comparison family is too small for combined intervals"));
            }
        }
    }
    judge_statistics(
        &json!({"statistic":"combine_decisions","gates":family.iter().map(|r|json!({"decision":r["decision"],"required":r["policy"]["required"]})).collect::<Vec<_>>()}),
    )
}
