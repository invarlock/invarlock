//! Owned request inputs and structural preflight before any scoring or publication.
use crate::{captured, crypto, required_string, EvidenceError, Result};
use invarlock_native_contracts::{canonical_json, parse_json, request, sha256_digest};
use invarlock_native_core as core;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn io_error(e: impl std::fmt::Display) -> EvidenceError {
    error(e.to_string())
}
fn reference(root: &Path, value: &Value, label: &str) -> Result<PathBuf> {
    let parts = request::reference_parts(
        value
            .as_str()
            .ok_or_else(|| error(format!("{label} must be a safe relative reference")))?,
        label,
    )?;
    Ok(parts.iter().fold(root.to_path_buf(), |p, c| p.join(c)))
}
fn read(
    root: &fs::PinnedDirectory,
    reference: &Value,
    label: &str,
    maximum: usize,
) -> Result<Vec<u8>> {
    let parts = request::reference_parts(reference.as_str().unwrap_or(""), "captured input")?;
    let relative = parts.iter().collect::<PathBuf>();
    let (bytes, _) = root
        .read_regular(&relative, maximum)
        .map_err(|e| error(format!("captured input could not be read safely: {e}")))?;
    if bytes.len() > maximum {
        return Err(error(format!("{label} exceeds the {maximum}-byte limit")));
    }
    Ok(bytes)
}
fn output(root: &Path, value: &Value) -> Result<PathBuf> {
    let path = reference(root, value, "output.evidence")?;
    let parts = request::reference_parts(value.as_str().unwrap(), "output.evidence")?;
    let mut current = root.to_path_buf();
    for (i, part) in parts.iter().enumerate() {
        current.push(part);
        match std::fs::symlink_metadata(&current) {
            Ok(meta) => {
                if i == parts.len() - 1 {
                    return Err(error("output.evidence already exists"));
                }
                if meta.file_type().is_symlink() {
                    return Err(error("output.evidence traverses a symlink"));
                }
                if !meta.is_dir() {
                    return Err(error("output.evidence is not a root-confined regular path"));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => return Err(io_error(e)),
        }
    }
    Ok(path)
}
fn run(root: &fs::PinnedDirectory, authored: &Value, config: &Value, side: &str) -> Result<Value> {
    let source = &authored["comparison"][side];
    let raw = read(
        root,
        &source["path"],
        "evaluation export",
        128 * 1024 * 1024,
    )?;
    let result = (|| -> Result<Value> {
        let imported = source["adapter"] != "invarlock";
        let value = if imported {
            let value=config["imported_runs"].get(side).ok_or_else(||error("external evaluator format requires its explicitly supplied Python import adapter"))?.clone();
            if value["source_digest"] != sha256_digest(&raw) {
                return Err(error(
                    "imported run source digest does not match the captured input bytes",
                ));
            }
            value
        } else {
            parse_json(&raw, "evaluation export")?
        };
        core::check_run(&value).map_err(|e| error(e.to_string()))?;
        if !imported
            && [
                "source",
                "run_id",
                "artifact_digest",
                "service_identity",
                "score_provenance",
                "input_projection",
            ]
            .iter()
            .any(|key| source.get(*key).is_some_and(|v| !v.is_null()))
        {
            return Err(error(
                "canonical run identities cannot be overridden at import",
            ));
        }
        Ok(value)
    })();
    result.map_err(|e| error(format!("{side} run could not be loaded: {}", e.message)))
}
pub fn prepare_captured(config: &Value) -> Result<Value> {
    let unsigned = config.get("unsigned").unwrap_or(&Value::Bool(false));
    let unsigned = unsigned
        .as_bool()
        .ok_or_else(|| error("unsigned must be a boolean"))?;
    let default_budget = json!(core::DEFAULT_MAX_BOOTSTRAP_DRAWS);
    let budget = config.get("max_bootstrap_draws").unwrap_or(&default_budget);
    if !budget.is_null()
        && (!budget.is_number()
            || budget.to_string().contains(['.', 'e', 'E'])
            || budget.to_string().starts_with('-'))
    {
        return Err(error("max_bootstrap_draws must be a non-negative integer"));
    }
    let key = config["signing_key_path"].as_str();
    if unsigned && key.is_some() {
        return Err(error("unsigned mode cannot be combined with a signing key"));
    }
    if !unsigned && key.is_none() {
        return Err(error("captured evaluation requires an Ed25519 signing key"));
    }
    let root = Path::new(required_string(config, "root")?)
        .canonicalize()
        .map_err(io_error)?;
    let pinned = fs::PinnedDirectory::open(&root, false).map_err(io_error)?;
    let authored = &config["authored_request"];
    request::validate_schema(authored, true)?;
    let destination = output(&root, &authored["output"]["evidence"])?;
    let baseline = run(&pinned, authored, config, "baseline")?;
    let subject = run(&pinned, authored, config, "subject")?;
    let raw = read(
        &pinned,
        &authored["comparison"]["policy"],
        "captured policy",
        128 * 1024 * 1024,
    )?;
    let policy = parse_json(&raw, "captured policy")
        .map_err(|e| error(format!("captured policy is invalid: {e}")))?;
    if !policy.is_object() {
        return Err(error("captured policy must be a JSON object"));
    }
    let (normalized, capacity) = (|| -> Result<(Value, Value)> {
        core::check_policy(&policy).map_err(|e|error(e.to_string()))?;
        for (side,run) in [("baseline",&baseline),("subject",&subject)] {
            if let Some(expected)=policy.get("expected_case_set_digest") {
                let cases=run["records"].as_array().unwrap().iter().map(|r|json!({"id":r["id"],"input":r["input"],"expected":r["expected"],"metadata":r["metadata"]})).collect::<Vec<_>>();
                if crate::judge::case_set_digest(&json!({"format":"invarlock/evaluation-case-set-v1","cases":cases}))?!=expected.as_str().unwrap() {
                    return Err(error(format!("run {} does not match the planned case set",invarlock_native_contracts::python_repr(&run["run_id"]))));
                }
            }
            let actual=sha256_digest(&canonical_json(run)?);
            if authored["comparison"][side].get("expected_run_digest").is_some_and(|expected|expected.as_str()!=Some(actual.as_str())) {
                return Err(error(format!("{side} run digest differs from expected pin")));
            }
        }
        let normalized = captured::normalize_request(authored, &baseline, &subject, &policy)?;
        let capacity =
            core::captured_preflight_capacity(&baseline, &subject, &policy, budget.as_u64())
                .map_err(|e| error(e.to_string()))?;
        Ok((normalized, capacity))
    })()
    .map_err(|e| error(format!("captured preflight failed: {}", e.message)))?;
    if let Some(key) = key {
        crypto::private_key(&fs::read_file(Path::new(key), 65536).map_err(io_error)?)?;
    }
    pinned.verify().map_err(io_error)?;
    let preflight = json!({"format_version":"invarlock/evaluation-preflight-v3","kind":"captured","ok":true,"execution_mode":"captured","requested_authentication":if unsigned{"unsigned_local"}else{"signed"},"baseline_run_digest":normalized["comparison"]["baseline"]["run_digest"],"subject_run_digest":normalized["comparison"]["subject"]["run_digest"],"policy_digest":normalized["comparison"]["policy_digest"],"record_count":capacity["record_count"],"required_bootstrap_draws":capacity["required_bootstrap_draws"],"missing_id_bytes":capacity["missing_id_bytes"],"checks":["request","run_bindings","policy","paired_capacity","output_destination","signing_mode"],"request_digest":sha256_digest(&canonical_json(&normalized)?),"output":destination.to_string_lossy(),"max_bootstrap_draws":budget,"scope_count":capacity["scope_count"]});
    Ok(
        json!({"baseline":baseline,"subject":subject,"policy":policy,"normalized_request":normalized,"preflight":preflight}),
    )
}

fn runtime_read(
    root: &fs::PinnedDirectory,
    reference: &Value,
    label: &str,
    maximum: usize,
) -> Result<Vec<u8>> {
    let parts = request::reference_parts(reference.as_str().unwrap_or(""), label)?;
    root.read_regular(&parts.iter().collect::<PathBuf>(), maximum)
        .map(|(raw, _)| raw)
        .map_err(io_error)
}
fn runtime_object(raw: &[u8], label: &str) -> Result<Value> {
    let value = parse_json(raw, label)?;
    if !value.is_object() {
        return Err(error(format!("{label} must be a JSON object")));
    }
    Ok(value)
}
/// Read and authenticate the portable schedule, metric policy, and observations.
/// Provider discovery, capability negotiation, and artifact loading remain explicit host adapters.
pub fn prepare_runtime_inputs(config: &Value) -> Result<Value> {
    use base64::Engine;
    use invarlock_native_contracts::schedule::Schedule;
    let authored = &config["authored_request"];
    request::validate_schema(authored, false)?;
    let root = Path::new(required_string(config, "root")?)
        .canonicalize()
        .map_err(io_error)?;
    let pinned = fs::PinnedDirectory::open(&root, false).map_err(io_error)?;
    let comparison = &authored["comparison"];
    let mode = authored["execution"]["mode"].as_str().unwrap_or("");
    let task = comparison["task"].as_str().unwrap();
    if mode == "run" {
        for side in ["baseline", "subject"] {
            if !invarlock_native_contracts::runtime::json_equal(
                &comparison[side]["runtime"]["settings"]["batch_size"],
                &json!(1),
            ) {
                return Err(error(format!(
                    "{side} strict paired execution requires batch_size=1"
                )));
            }
        }
    }
    let schedule = if mode == "import" {
        let raw = runtime_read(
            &pinned,
            &authored["execution"]["schedule"],
            "canonical schedule",
            64 * 1024 * 1024,
        )?;
        let schedule = Schedule::build(&runtime_object(&raw, "canonical schedule")?)?;
        if schedule.value()["task"] != task {
            return Err(error(
                "imported schedule task does not match comparison.task",
            ));
        }
        if schedule.bytes() != raw {
            return Err(error("canonical schedule bytes are not canonical"));
        }
        let dataset = runtime_read(
            &pinned,
            &comparison["dataset"],
            "dataset input",
            64 * 1024 * 1024,
        )?;
        if schedule.bytes() != dataset {
            return Err(error(
                "dataset input must be the exact canonical imported schedule",
            ));
        }
        schedule
    } else {
        let source = &comparison["dataset"];
        let raw = runtime_read(
            &pinned,
            &source["path"],
            "local evaluation dataset",
            64 * 1024 * 1024,
        )?;
        invarlock_native_contracts::schedule::prepare_local(source, &raw, task)?
    };
    let raw = runtime_read(
        &pinned,
        &comparison["policy"],
        "policy input",
        4 * 1024 * 1024,
    )?;
    let policy = runtime_object(&raw, "policy input")?;
    let policy_digest = sha256_digest(&raw);
    let extension = comparison.get("scorer_extension");
    let metric = extension
        .and_then(|v| v["scorer_id"].as_str())
        .or_else(|| comparison["metric"].as_str())
        .ok_or_else(|| error("comparison metric selection is invalid"))?;
    let mut requirements = json!({});
    let judge = if metric == "judge" {
        Some(crate::judge_native::prepare(&raw, schedule.value())?)
    } else {
        let mut paired = json!({"format":"invarlock/paired-records-v1","metric":metric,"schedule_sha256":"0".repeat(64),"records":[{"record_id":"policy-preflight","input_sha256":"0".repeat(64),"baseline":{"score":1.0},"subject":{"score":1.0}}]});
        if metric == "normalized_nll_per_utf8_byte" {
            paired["derived_measurements"] = json!({"perplexity_ratio":{"status":"available","basis":"authenticated_target_likelihood","method":"target_token_weighted_perplexity_ratio_v1","tokenizer_metadata_sha256":"0".repeat(64),"target_token_count":1,"baseline_perplexity":1.0,"subject_perplexity":1.0,"ratio":1.0}});
        }
        if let Some(binding) = extension {
            paired["scorer_extension"] = binding.clone();
            paired["scorer_replay"] = json!({"baseline":{},"subject":{}});
        }
        core::build_comparison_report(
            "policy-preflight",
            &paired,
            &policy,
            &policy_digest,
            "invarlock/comparison-report-v3",
        )
        .map_err(io_error)?;
        let selected = &policy["resolved_policy"]["metrics"][if extension.is_some() {
            "scorer_extension"
        } else {
            metric
        }];
        if let Some(minimum) = selected.get("minimum_record_count") {
            let width = if metric == "normalized_nll_per_utf8_byte" && extension.is_none() {
                "maximum_interval_width_ratio"
            } else {
                "maximum_interval_width_pp"
            };
            requirements["minimum_record_count"] = minimum.clone();
            requirements[width] = selected[width].clone();
            let count = schedule.value()["records"].as_array().unwrap().len();
            if minimum.as_u64().is_some_and(|n| (count as u64) < n) {
                return Err(error(format!(
                    "schedule has {count} records but policy requires at least {minimum}"
                )));
            }
        }
        None
    };
    let mut observations = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for observation in authored["observations"].as_array().into_iter().flatten() {
        let id = observation["id"].as_str().unwrap();
        if !ids.insert(id) {
            return Err(error(format!(
                "duplicate observation id: {}",
                invarlock_native_contracts::python_repr(&observation["id"])
            )));
        }
        let bytes = runtime_read(
            &pinned,
            &observation["path"],
            &format!("observation {id:?}"),
            1024 * 1024,
        )?;
        if bytes.is_empty() {
            return Err(error(format!(
                "observation {id:?} payload must be non-empty bytes"
            )));
        }
        let payload = parse_json(&bytes, &format!("observation {id} payload"))?;
        if !payload.is_object() {
            return Err(error(format!(
                "observation {id:?} payload must be a JSON object"
            )));
        }
        if canonical_json(&payload)? != bytes {
            return Err(error(format!(
                "observation {id:?} payload must use canonical JSON"
            )));
        }
        observations.push(json!({"id":id,"kind":observation["kind"],"scope":observation["scope"],"payload_base64":base64::engine::general_purpose::STANDARD.encode(&bytes),"payload_digest":sha256_digest(&bytes)}));
    }
    observations.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    pinned.verify().map_err(io_error)?;
    Ok(
        json!({"schedule":schedule.value(),"canonical_schedule":String::from_utf8(schedule.bytes().to_vec()).unwrap(),"schedule_digest":schedule.digest(),"policy":policy,"policy_base64":base64::engine::general_purpose::STANDARD.encode(&raw),"policy_digest":policy_digest,"selected_metric":metric,"sample_requirements":requirements,"observations":observations,"judge":judge}),
    )
}
fn normalized_side(value: &Value) -> Result<Value> {
    let model = value["artifact"]["model_id"].as_str().unwrap_or("");
    let locator = value["artifact"]["locator"]
        .as_str()
        .unwrap_or("")
        .trim_matches(invarlock_native_contracts::schedule::python_whitespace);
    let bytes = locator.as_bytes();
    if locator.is_empty()
        || locator.starts_with(['/', '~', '\\'])
        || (bytes.len() > 1 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || locator.to_lowercase().starts_with("file://")
    {
        return Err(error(format!(
            "{model} artifact locator must be a stable non-file locator"
        )));
    }
    Ok(json!({"artifact":{"model_id":model,"locator":locator},"runtime":value["runtime"]}))
}
pub fn normalize_runtime_request(config: &Value) -> Result<Value> {
    let authored = &config["authored_request"];
    let schedule = invarlock_native_contracts::schedule::Schedule::build(&config["schedule"])?;
    let comparison = &authored["comparison"];
    let run = authored["execution"]["mode"] == "run";
    let mut execution = json!({"mode":if run{"run"}else{"import"}});
    let dataset = if run {
        invarlock_native_contracts::schedule::preparation_descriptor(
            &comparison["dataset"],
            &schedule,
        )?
    } else {
        execution["records"] = json!("records/paired-records.json");
        execution["schedule"] = json!("schedule/runtime-behavioral-schedule.json");
        for side in ["baseline", "subject"] {
            let mut side_value = json!({});
            for (field, role) in [
                ("identity", "provider_identity"),
                ("receipt", "provider_receipt"),
                ("observation", "scoring_observation"),
                ("run_report", "run_report"),
                ("runtime_manifest", "runtime_manifest"),
                ("runtime_config", "runtime_config"),
            ] {
                let key = format!("{side}_{role}");
                let path = crate::native_pack::EVIDENCE
                    .iter()
                    .find(|(name, _)| *name == key)
                    .unwrap()
                    .1;
                side_value[field] = json!(path);
            }
            execution[side] = side_value;
        }
        json!("schedule/runtime-behavioral-schedule.json")
    };
    let mut normalized = json!({"format_version":authored["format_version"],"comparison":{"baseline":normalized_side(&comparison["baseline"] )?,"subject":normalized_side(&comparison["subject"] )?,"dataset":dataset,"policy":"inputs/policy.json","task":comparison["task"]},"execution":execution,"output":{"evidence":"evidence"}});
    let key = if comparison.get("scorer_extension").is_some() {
        "scorer_extension"
    } else {
        "metric"
    };
    normalized["comparison"][key] = comparison[key].clone();
    if let Some(observations) = config["observations"].as_array().filter(|a| !a.is_empty()) {
        normalized["observations"]=json!(observations.iter().map(|o|json!({"id":o["id"],"kind":o["kind"],"scope":o["scope"],"payload_digest":o["payload_digest"]})).collect::<Vec<_>>());
    }
    if let Some(judge) = comparison.get("judge") {
        normalized["comparison"]["judge"] =
            json!({"workspace":"judge-workspace","signer_identity":judge["signer_identity"]});
        if let Some(model) = judge.get("model") {
            normalized["comparison"]["judge"]["model"] = normalized_side(model)?;
        }
    }
    Ok(normalized)
}
