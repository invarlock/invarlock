//! Closed evaluator export qualification with independent deterministic scoring.
use crate::{canonical, required_string, EvidenceError, Result};
use invarlock_native_contracts::{
    canonical_json_no_newline, first_schema_error, parse_json, python_repr, sha256_digest,
    SchemaErrorOrder,
};
use invarlock_native_core as analysis;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::path::Path;
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn check(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(error(message))
    }
}
fn read(path: &Path, label: &str, limit: usize) -> Result<Vec<u8>> {
    fs::read_file(path, limit).map_err(|e| match e {
        fs::Error::Limit(_) => error(format!("{label} exceeds the {limit}-byte size limit")),
        _ => error(format!("{label} is unavailable or unsafe")),
    })
}
fn validate(value: &Value, schema: &str, label: &str) -> Result<()> {
    if let Some(e) = first_schema_error(value, schema, SchemaErrorOrder::Instance)? {
        let detail = if e.path.is_empty() {
            e.message
        } else {
            format!("{}: {}", e.path.join("."), e.message)
        };
        return Err(error(format!("{label} is invalid: {detail}")));
    }
    Ok(())
}
fn document(config: &Value, field: &str, schema: &str, label: &str) -> Result<(Value, Vec<u8>)> {
    let raw = read(
        Path::new(required_string(config, field)?),
        label,
        16 * 1024 * 1024,
    )?;
    let value = parse_json(&raw, label)?;
    check(value.is_object(), format!("{label} must be a JSON object"))?;
    check(
        raw == canonical(&value)?,
        format!("{label} must use canonical JSON"),
    )?;
    validate(&value, schema, label)?;
    Ok((value, raw))
}
// The qualified CPython 3.12 sum operation uses compensated floating addition.
// Numerical operation order follows Python/bltinmodule.c in CPython v3.12.13;
// see THIRD_PARTY_NOTICES.md for the retained PSF license.
fn sum(values: &[f64]) -> f64 {
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    for &value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    if correction != 0.0 && correction.is_finite() {
        total + correction
    } else {
        total
    }
}
pub fn qualify(config: &Value) -> Result<Value> {
    let (profile, profile_raw) = document(
        config,
        "profile_path",
        "evaluator_qualification_profile",
        "evaluator qualification profile",
    )?;
    let (schedule, schedule_raw) = document(
        config,
        "schedule_path",
        "evaluator_qualification_schedule",
        "evaluator qualification schedule",
    )?;
    let (export, export_raw) = document(
        config,
        "export_path",
        "evaluator_qualification_export",
        "evaluator qualification export",
    )?;
    let raw = read(
        Path::new(required_string(config, "raw_output_path")?),
        "raw upstream output",
        64 * 1024 * 1024,
    )?;
    check(
        export["profile_id"] == profile["profile_id"],
        "export profile_id does not match profile",
    )?;
    check(
        export["upstream"] == profile["upstream"]["package"],
        "upstream package identity does not match profile",
    )?;
    let bindings = json!({"profile_sha256":sha256_digest(&profile_raw),"schedule_sha256":sha256_digest(&schedule_raw),"export_sha256":sha256_digest(&export_raw),"raw_output_sha256":sha256_digest(&raw),"runner_sha256":profile["execution"]["runner_sha256"],"dependency_lock_sha256":profile["execution"]["dependency_lock_sha256"]});
    for field in [
        "profile_sha256",
        "schedule_sha256",
        "raw_output_sha256",
        "runner_sha256",
        "dependency_lock_sha256",
    ] {
        check(
            export["bindings"][field] == bindings[field],
            if field == "raw_output_sha256" {
                "raw upstream output digest does not match export".to_string()
            } else {
                format!("{} does not match", field.replace('_', " "))
            },
        )?;
    }
    let mut result = json!({"format":"invarlock/evaluator-qualification-result-v1","profile_id":profile["profile_id"],"bindings":bindings,
        "outcome":"observation_only","authority":"observation_only","reason_codes":[],"record_count":0,"scores":[],"mean_score":null,"records_sha256":null});
    let mut runtime_records = vec![];
    let metric = profile["authority"]["metric"].clone();
    if profile["authority"]["mode"] == "observation_only" {
        check(
            export["summary"].is_object(),
            "observation-only export must bind one upstream summary",
        )?;
        check(
            export["summary"]["sha256"] == bindings["raw_output_sha256"],
            "observation-only summary digest does not match raw upstream output",
        )?;
        result["reason_codes"] = json!([profile["authority"]["reason"]]);
    } else {
        let kind = metric["kind"].as_str().unwrap();
        let empty = json!({});
        let configuration = metric.get("configuration").unwrap_or(&empty);
        analysis::validate_configuration(kind, configuration).map_err(|e| error(e.to_string()))?;
        check(
            export["summary"].is_null(),
            "deterministic export must not substitute an aggregate summary",
        )?;
        let expected = schedule["records"].as_array().unwrap();
        let observed = export["records"].as_array().unwrap();
        check(
            expected.len() == observed.len()
                && expected.iter().zip(observed).all(|(a, b)| {
                    a["record_id"] == b["record_id"] && a["input_sha256"] == b["input_sha256"]
                }),
            "export record order and input identities must exactly match schedule",
        )?;
        let mut scores = vec![];
        for (expected, observed) in expected.iter().zip(observed) {
            let id = python_repr(&observed["record_id"]);
            check(
                observed["status"] == "ok",
                format!("export record {id} is not successful"),
            )?;
            let output = observed["output_text"]
                .as_str()
                .ok_or_else(|| error(format!("record {id} output_text must be a string")))?;
            check(
                observed["output_sha256"] == sha256_digest(output.as_bytes()),
                format!("export record {id} output digest is invalid"),
            )?;
            let mut score = if observed["output_sha256"] == expected["reference_output_sha256"] {
                1.0
            } else {
                0.0
            };
            if let Some(reference) = expected.get("reference_output") {
                check(
                    sha256_digest(reference.as_str().unwrap().as_bytes())
                        == expected["reference_output_sha256"],
                    "reference text differs from its independent digest",
                )?;
            }
            if kind != "exact_match" {
                check(
                    expected.get("reference_output").is_some(),
                    "non-exact replay requires independently supplied reference_output text",
                )?;
                score = analysis::score(
                    kind,
                    &expected["reference_output"],
                    &observed["output_text"],
                    configuration,
                )
                .map_err(|e| error(e.to_string()))?;
            }
            check(
                observed["reported_score"].as_f64() == Some(score),
                format!(
                    "export record {id} reported score does not match independent {kind} replay"
                ),
            )?;
            runtime_records.push(json!({"record_id":observed["record_id"],"input_sha256":observed["input_sha256"].as_str().unwrap().trim_start_matches("sha256:"),"status":"ok","output_text":output,"output_sha256":observed["output_sha256"].as_str().unwrap().trim_start_matches("sha256:"),"logprob_sum":null,"token_count":null,"utf8_byte_count":null,"error_code":null}));
            scores.push(score);
        }
        result["outcome"] = json!("qualified_for_import");
        result["authority"] = json!("verdict_authority");
        result["record_count"] = json!(scores.len());
        result["mean_score"] = json!(sum(&scores) / scores.len() as f64);
        result["scores"] = json!(scores);
        result["records_sha256"] = json!(sha256_digest(&canonical_json_no_newline(&json!(
            runtime_records
        ))?));
    }
    validate(
        &result,
        "evaluator_qualification_result",
        "qualification result",
    )?;
    if let Some(destination) = config["destination"].as_str() {
        fs::write_file_no_replace(Path::new(destination), &canonical(&result)?, 0o600, true)
            .map_err(|e| error(format!("could not write qualification result: {e}")))?;
    }
    Ok(json!({"result":result,"runtime_records":runtime_records,"metric":metric}))
}
