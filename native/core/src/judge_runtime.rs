//! Offline replay of first-party retained runtime-provider judge sources.
use crate::{
    judge_contracts::{bare_digest, bytes, digest, render_judge_runtime_prompt},
    AnalysisError,
};
use invarlock_native_contracts::{runtime, Value};
use serde_json::json;
fn err(message: impl Into<String>) -> AnalysisError {
    AnalysisError::new(message)
}
fn check(ok: bool, message: &str) -> Result<(), AnalysisError> {
    if ok {
        Ok(())
    } else {
        Err(err(message))
    }
}
fn exact(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)))
}
fn typed(
    v: &Value,
    schema: &str,
    label: &str,
    validate: fn(&Value) -> invarlock_native_contracts::Result<()>,
) -> Result<(), AnalysisError> {
    if bytes(v)?.len() > 64 * 1024 * 1024 {
        return Err(err(format!("{label} exceeds the 67108864-byte size limit")));
    }
    if let Some(error) = invarlock_native_contracts::first_schema_error(
        v,
        schema,
        invarlock_native_contracts::SchemaErrorOrder::Instance,
    )
    .map_err(|e| err(e.to_string()))?
    {
        let path = if error.path.is_empty() {
            "<root>".to_owned()
        } else {
            error.path.join(".")
        };
        return Err(err(format!(
            "{label} schema validation failed at {path}: {}",
            error.message
        )));
    }

    validate(v).map_err(|e| err(format!("{label} could not be reconstructed: {e}")))
}
pub(crate) fn replay(decoded: &Value, _plan: &Value) -> Result<Vec<Value>, AnalysisError> {
    check(
        exact(
            decoded,
            &[
                "format",
                "runtime_spec",
                "artifact_identity",
                "scoring_observation",
                "provider_receipt",
                "trials",
            ],
        ) && decoded["format"] == "retained-runtime-provider-judge-v1"
            && decoded["trials"].is_array(),
        "source has an unsupported runtime-provider profile",
    )?;
    let trials = decoded["trials"].as_array().unwrap();
    check(
        trials.iter().all(|trial| {
            trial.is_object()
                && trial["attempts"]
                    .as_array()
                    .is_some_and(|a| a.iter().all(Value::is_object))
        }),
        "retained runtime-provider trials must contain attempt objects",
    )?;
    Ok(trials.clone())
}
pub(crate) fn family(sources: &[Value], plan: &Value) -> Result<(), AnalysisError> {
    if sources.is_empty() {
        return Ok(());
    }
    let judge = &plan["judge"];
    let config = &judge["config"];
    let schedule = &plan["schedule"];
    check(
        judge["model_identity"]["kind"] == "local_weights"
            && matches!(
                judge["provider"].as_str(),
                Some("hf_transformers" | "llama_cpp")
            )
            && judge["approved_resolved_models"] == json!([judge["requested_model"]])
            && config["temperature"] == "0"
            && config["top_p"] == "1"
            && !config["seed"].is_null()
            && config["reasoning_effort"].is_null()
            && schedule["max_attempts"].as_u64() == Some(1)
            && schedule["retry_on"].as_array().is_some_and(Vec::is_empty)
            && schedule["cache"] == "forbid",
        "runtime judge plan does not satisfy the strict local profile",
    )?;
    let first = &sources[0];
    let mut expected_receipt = first["provider_receipt"].clone();
    if let Some(o) = expected_receipt.as_object_mut() {
        o.remove("scoring_observation_sha256");
    }
    for source in sources {
        let mut receipt = source["provider_receipt"].clone();
        if let Some(o) = receipt.as_object_mut() {
            o.remove("scoring_observation_sha256");
        }
        check(
            runtime::json_equal(&source["runtime_spec"], &first["runtime_spec"])
                && runtime::json_equal(&source["artifact_identity"], &first["artifact_identity"])
                && runtime::json_equal(&receipt, &expected_receipt),
            "retained runtime judge shards have inconsistent runtime identity",
        )?;
        source_check(source, plan)?;
    }
    Ok(())
}
fn source_check(source: &Value, plan: &Value) -> Result<(), AnalysisError> {
    let spec = &source["runtime_spec"];
    check(
        exact(spec, &["provider_name", "model_id", "settings"]),
        "retained runtime judge spec has an unsupported shape",
    )?;
    let artifact = &source["artifact_identity"];
    let observation = &source["scoring_observation"];
    let receipt = &source["provider_receipt"];
    let validation = (|| {
        runtime::validate_spec(spec).map_err(|e| err(e.to_string()))?;
        typed(
            artifact,
            "model_artifact_identity",
            "model artifact identity",
            runtime::validate_artifact,
        )?;
        typed(
            observation,
            "runtime_scoring_observation",
            "runtime scoring observation",
            runtime::validate_observation,
        )?;
        typed(
            receipt,
            "runtime_provider_receipt",
            "runtime provider receipt",
            runtime::validate_receipt,
        )?;
        Ok::<_, AnalysisError>(())
    })();
    if let Err(e) = validation {
        return Err(err(format!(
            "retained runtime judge evidence is invalid: {}",
            e.to_string().chars().take(240).collect::<String>()
        )));
    }
    let judge = &plan["judge"];
    let provider = spec["provider_name"].as_str().unwrap_or("");
    check(
        ["hf_transformers", "llama_cpp"].contains(&provider),
        "retained runtime judge provider is unsupported",
    )?;
    check(
        spec["provider_name"] == judge["provider"],
        "retained runtime judge provider differs from the plan",
    )?;
    check(
        spec["model_id"] == judge["requested_model"]
            && judge["approved_resolved_models"] == json!([spec["model_id"]]),
        "retained runtime judge model differs from the plan",
    )?;
    check(
        judge["model_identity"]["kind"] == "local_weights"
            && judge["model_identity"]["weights_sha256"] == digest(artifact)?,
        "retained runtime judge artifact identity differs from the plan",
    )?;
    check(
        receipt["plugin"]["distribution"] == "invarlock",
        "retained runtime judge provider is not first-party",
    )?;
    check(
        !receipt["outer_image_digest"].is_null(),
        "retained runtime judge lacks a pinned outer image",
    )?;
    runtime::provider_bindings(
        artifact,
        observation,
        receipt,
        &bytes(observation)?,
        receipt["outer_image_digest"].as_str(),
    )
    .map_err(|e| err(e.to_string()))?;
    let request = json!({"runtime":{"provider":provider,"settings":spec["settings"]}});
    runtime::request_bindings(&request, artifact, receipt).map_err(|e| err(e.to_string()))?;
    check(
        receipt["capabilities"]["tasks"]
            .as_array()
            .is_some_and(|v| v.contains(&json!("text_causal")))
            && receipt["capabilities"]["metrics"]
                .as_array()
                .is_some_and(|v| v.contains(&json!("exact_match"))),
        "retained runtime judge lacks required text capabilities",
    )?;
    let execution = &receipt["execution_settings"];
    let config = &judge["config"];
    let settings = &spec["settings"];
    check(
        execution["allow_network"] == false
            && runtime::json_equal(&execution["seed"], &config["seed"])
            && runtime::json_equal(
                &execution["max_output_tokens"],
                &config["max_output_tokens"],
            )
            && config["temperature"] == "0"
            && config["top_p"] == "1"
            && config["reasoning_effort"].is_null(),
        "retained runtime judge generation settings differ from the plan",
    )?;
    check(
        settings["seed"].as_u64().is_some()
            && runtime::json_equal(&settings["seed"], &config["seed"])
            && settings["max_output_tokens"].as_u64().is_some()
            && runtime::json_equal(&settings["max_output_tokens"], &config["max_output_tokens"])
            && settings["batch_size"].as_u64() == Some(1)
            && settings["context_length"]
                .as_u64()
                .is_some_and(|n| (1..=1024 * 1024).contains(&n))
            && settings["max_output_tokens"].as_u64() <= settings["context_length"].as_u64()
            && settings["timeout_seconds"]
                .as_u64()
                .is_some_and(|n| (1..=604800).contains(&n)),
        "retained runtime judge spec violates strict execution bounds",
    )?;
    check(
        observation["schedule_sha256"] == digest(plan)?,
        "retained runtime judge observation differs from the plan",
    )?;
    check(
        observation["aggregate_source_sha256"] == digest(&observation["records"])?,
        "retained runtime judge observation aggregate is invalid",
    )?;
    let records = observation["records"].as_array().unwrap();
    let trials = source["trials"]
        .as_array()
        .ok_or_else(|| err("retained runtime-provider trials must contain attempt objects"))?;
    check(
        records.len() == trials.len(),
        "retained runtime judge observation coverage is incomplete",
    )?;
    for (index, (record, trial)) in records.iter().zip(trials).enumerate() {
        let attempts = trial["attempts"]
            .as_array()
            .ok_or_else(|| err("retained runtime judge trial must contain one attempt"))?;
        check(
            attempts.len() == 1,
            "retained runtime judge trial must contain one attempt",
        )?;
        let attempt = &attempts[0];
        let prompt = attempt["request"]["text"]
            .as_str()
            .ok_or_else(|| err("retained runtime judge request is invalid"))?;
        let prompt = render_judge_runtime_prompt(plan, prompt.as_bytes()).map_err(|e| {
            err(format!(
                "retained runtime judge prompt is invalid: {}",
                e.to_string().chars().take(240).collect::<String>()
            ))
        })?;
        let part = json!([{"kind":"text","role":"prompt","text":std::str::from_utf8(&prompt).map_err(|e|err(e.to_string()))?,"sha256":bare_digest(&prompt)}]);
        check(
            record["record_id"] == trial["trial_id"]
                && record["input_sha256"] == digest(&part)?
                && attempt["source"]["record_index"].as_u64() == Some(index as u64)
                && attempt["source"]["model_event_id"] == record["record_id"],
            "retained runtime judge record does not match its trial",
        )?;
        if record["status"] == "ok" {
            check(
                attempt["status"] == "completed"
                    && record["output_text"].is_string()
                    && attempt["response"].is_object()
                    && attempt["response"]["text"] == record["output_text"]
                    && attempt["response"]["sha256"] == record["output_sha256"],
                "retained runtime judge output differs from its attempt",
            )?;
        } else {
            check(
                attempt["status"] == "cancelled"
                    && attempt["response"].is_null()
                    && attempt["error"].is_object(),
                "retained runtime judge error differs from its attempt",
            )?;
        }
    }
    Ok(())
}
