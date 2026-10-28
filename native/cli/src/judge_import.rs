//! Publish and preflight retained judge measurements without a collection runtime.
use super::{err, outcome, Args, Result};
use invarlock_native_contracts as contracts;
use invarlock_native_core as core;
use invarlock_native_evidence as evidence;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub(super) fn run(
    a: &Args,
    authored: &Value,
    root: &Path,
) -> Result<evidence::transaction::Outcome> {
    if authored["execution"]["mode"] != "judge_import" {
        return Err(err(
            "judge collection requires an explicitly selected Python compatibility worker",
        ));
    }
    if a.get("max-bootstrap-draws")
        .is_some_and(|v| v != "102400000")
    {
        return Err(err("max-bootstrap-draws is not a judge request control"));
    }
    let root = root.canonicalize().map_err(|e| err(e.to_string()))?;
    let mut request = authored.clone();
    for (flag, field) in [
        ("baseline-run", "baseline_run"),
        ("subject-run", "subject_run"),
        ("output", "evidence"),
    ] {
        if let Some(value) = a.get(flag) {
            let path = super::absolute(value)?;
            let relative = path
                .strip_prefix(&root)
                .map_err(|_| err(format!("{flag} must be inside request root")))?;
            if field == "evidence" {
                request["output"][field] = json!(relative);
            } else {
                request["comparison"][field] = json!(relative);
            }
        }
    }
    contracts::request::validate_authored(&request)?;
    let destination =
        evidence::preparation::runtime_destination(&root, &request["output"]["evidence"])?;
    let pinned = fs::PinnedDirectory::open(&root, false).map_err(|e| err(e.to_string()))?;
    let mut values = json!({});
    let mut missing = Vec::new();
    let mut total = 0usize;
    for (name, reference) in request["comparison"].as_object().unwrap() {
        let parts = contracts::request::reference_parts(reference.as_str().unwrap(), name)?;
        let relative = parts.iter().collect::<PathBuf>();
        if root.join(&relative).starts_with(&destination) {
            return Err(err(
                "input files must remain outside the evidence destination",
            ));
        }
        let limit = match name.as_str() {
            "plan" => 64 * 1024 * 1024,
            "measurements" => 384 * 1024 * 1024,
            "policy" => 65536,
            _ => 128 * 1024 * 1024,
        };
        let raw = match pinned.read_regular(&relative, limit) {
            Ok((raw, _)) => raw,
            Err(fs::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {
                missing.push(name.clone());
                continue;
            }
            Err(e) => return Err(err(e.to_string())),
        };
        total = total
            .checked_add(raw.len())
            .ok_or_else(|| err("combined judge inputs exceed the 384 MiB allowance"))?;
        if total > 384 * 1024 * 1024 {
            return Err(err("combined judge inputs exceed the 384 MiB allowance"));
        }
        let value = contracts::parse_json(&raw, &format!("judge {name}"))?;
        if !value.is_object() {
            return Err(err(format!("judge {name} must be a JSON object")));
        }
        values[name] = value;
    }
    pinned.verify().map_err(|e| err(e.to_string()))?;
    let mut result = json!({"format_version":"invarlock/judge-evaluation-preflight-v1","kind":"judge","execution_mode":"judge_import","ok":false,"ready":false,"evidence":destination,"signer_identity":request["output"]["signer_identity"],"missing_inputs":missing,"cases":null,"independent_units":null,"planned_trials":null,"maximum_attempts":null,"measurements":null,"judge":null,"rubric":null,"scale":null,"budgets":null,"budget_capacity":null,"collection_available":false,"network_calls":0,"errors":[]});
    if let Some(plan) = values.get("plan") {
        core::validate_measurement_plan(plan).map_err(|e| err(e.to_string()))?;
        result["cases"] = json!(plan["answer_bindings"].as_array().unwrap().len());
        result["independent_units"] = json!(plan["sampling"]["case_units"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["unit_id"].as_str().unwrap())
            .collect::<BTreeSet<_>>()
            .len());
        result["planned_trials"] = plan["schedule"]["expected_trials"].clone();
        result["maximum_attempts"] = json!(
            plan["schedule"]["expected_trials"].as_u64().unwrap()
                * plan["schedule"]["max_attempts"].as_u64().unwrap()
        );
        result["judge"] = plan["judge"].clone();
        result["rubric"] = json!({"sha256":plan["rubric"]["sha256"],"excerpt":plan["rubric"]["text"].as_str().unwrap().chars().take(4096).collect::<String>()});
        result["scale"] = plan["scale"].clone();
        if let Some(policy) = values.get("policy") {
            core::validate_judge_analysis_policy(policy, plan).map_err(|e| err(e.to_string()))?;
            result["metric"] = json!({"name":policy["metric_name"],"direction":policy["direction"],"allowed_degradation":policy["allowed_degradation"],"minimum_units":policy["minimum_units"],"maximum_interval_width":policy["maximum_interval_width"]});
            result["precision"] =
                core::judge_plan_precision(plan, policy).map_err(|e| err(e.to_string()))?;
        }
        if let (Some(baseline), Some(subject)) =
            (values.get("baseline_run"), values.get("subject_run"))
        {
            if [baseline, subject]
                .iter()
                .any(|v| v["source"]["name"] == "invarlock-native-judge")
            {
                return Err(err("Native judge answers require their runtime capture; resume the native metric: judge request and workspace"));
            }
            core::validate_frozen_answers(plan, baseline, subject)
                .map_err(|e| err(e.to_string()))?;
            result["baseline"] = baseline["run_id"].clone();
            result["subject"] = subject["run_id"].clone();
            if let Some(measurements) = values.get("measurements") {
                core::validate_measurements(measurements, plan, baseline, subject)
                    .map_err(|e| err(e.to_string()))?;
                result["measurements"] = measurements["completeness"].clone();
            }
        }
    }
    if !missing.is_empty() {
        result["errors"] = json!([format!(
            "Supply the missing request-relative inputs: {}",
            missing.join(", ")
        )]);
    }
    let ready = missing.is_empty();
    result["ok"] = json!(ready);
    result["ready"] = json!(ready);
    if a.flag("preflight") || !ready {
        return outcome(result, if ready { 0 } else { 2 });
    }
    if a.flag("unsigned") == a.get("signing-key").is_some() {
        return Err(err(
            "Choose either --signing-key or explicit --unsigned for judge evidence",
        ));
    }
    let key = a
        .get("signing-key")
        .map(|path| fs::read_file(Path::new(path), 65536).map_err(|e| err(e.to_string())))
        .transpose()?;
    if let Some(raw) = &key {
        evidence::crypto::private_key(raw)?;
    }
    let mut config = values;
    config["analysis_policy"] = config.as_object_mut().unwrap().remove("policy").unwrap();
    config["destination"] = json!(destination);
    config["signer_identity"] = if key.is_some() {
        request["output"]["signer_identity"].clone()
    } else {
        Value::Null
    };
    let publication = evidence::judge::publish_with_key(&config, key.as_deref())?;
    let analysis = &publication["analysis_result"];
    outcome(
        json!({"format_version":"invarlock/judge-evaluation-result-v1","kind":"judge","ok":true,"evidence":destination,"authentication":if key.is_some(){"signed"}else{"unsigned_local"},"signer_identity":config["signer_identity"],"independent_verification":"not_performed","decision":analysis["decision"],"analysis":analysis,"errors":[]}),
        if a.flag("fail-on-policy") && analysis["decision"] != "pass" {
            7
        } else {
            0
        },
    )
}
