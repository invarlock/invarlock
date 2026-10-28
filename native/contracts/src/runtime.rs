//! Portable typed runtime contracts and request/receipt identity bindings.
use crate::{
    canonical_json_no_newline, sha256_digest, validate_schema, ContractError, Result, Value,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
fn integrity(message: impl Into<String>) -> ContractError {
    ContractError::new("runtime_contract", message)
}
fn check(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(integrity(message))
    }
}
fn is_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}
pub fn canonical_digest(value: &Value) -> Result<String> {
    Ok(sha256_digest(&canonical_json_no_newline(value)?)[7..].to_owned())
}
pub fn validate_spec(value: &Value) -> Result<()> {
    let provider = value["provider_name"].as_str().unwrap_or("");
    nonempty_string(&value["provider_name"], "provider_name")?;
    check(
        canonical_name(provider),
        "provider_name must be a canonical provider name",
    )?;
    nonempty_string(&value["model_id"], "model_id")?;
    let settings = value["settings"]
        .as_object()
        .ok_or_else(|| integrity("settings must be a mapping of JSON scalar values"))?;
    for (key, v) in settings {
        check(canonical_name(key), "settings keys must be canonical names")?;
        check(
            !v.is_array() && !v.is_object(),
            "settings values must be JSON scalar values",
        )?;
        if v.as_number()
            .is_some_and(|n| n.to_string().contains(['.', 'e', 'E']))
        {
            check(
                v.as_f64().is_some_and(f64::is_finite),
                "settings values must be finite JSON scalar values",
            )?;
        }
    }
    Ok(())
}
fn nonempty_string(value: &Value, name: &str) -> Result<()> {
    let text = value.as_str().unwrap_or("");
    check(
        !text.is_empty()
            && text
                == text.trim_matches(|c: char| {
                    c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c)
                }),
        format!("{name} must be a non-empty trimmed string"),
    )?;
    check(
        text.chars().all(|c| c as u32 >= 32),
        format!("{name} must not contain control characters"),
    )
}
fn canonical_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.as_bytes()[0].is_ascii_lowercase()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}
fn nonnegative_integer(value: &Value) -> bool {
    value
        .as_number()
        .is_some_and(|n| n.to_string().bytes().all(|b| b.is_ascii_digit()))
}
fn positive_integer(value: &Value) -> bool {
    nonnegative_integer(value) && value.as_number().is_some_and(|n| n.to_string() != "0")
}
pub fn validate_artifact(value: &Value) -> Result<()> {
    validate_schema(value, "model_artifact_identity")?;
    if value["artifact_format"] == "gguf" {
        check(
            positive_integer(&value["byte_length"]),
            "byte_length must be a positive integer",
        )?;
    }
    Ok(())
}
pub fn validate_observation(value: &Value) -> Result<()> {
    validate_schema(value, "runtime_scoring_observation")?;
    let records = value["records"]
        .as_array()
        .ok_or_else(|| integrity("records must be a non-empty tuple"))?;
    let mut ids = BTreeSet::new();
    for record in records {
        check(
            ids.insert(record["record_id"].as_str().unwrap_or("")),
            "record IDs must be unique within an observation",
        )?;
        check(
            record["output_text"].is_null() == record["output_sha256"].is_null(),
            "output_text and output_sha256 must be present together",
        )?;
        for name in ["token_count", "utf8_byte_count"] {
            if !record[name].is_null() {
                check(
                    nonnegative_integer(&record[name]),
                    format!("{name} must be a non-negative integer"),
                )?;
            }
        }
        if !record["logprob_sum"].is_null() {
            check(
                record["logprob_sum"].as_f64().is_some_and(f64::is_finite),
                "logprob_sum must be finite",
            )?;
            check(
                positive_integer(&record["token_count"])
                    && positive_integer(&record["utf8_byte_count"]),
                "logprob_sum requires positive token_count and utf8_byte_count",
            )?;
        }
    }
    Ok(())
}
pub fn validate_receipt(value: &Value) -> Result<()> {
    validate_schema(value, "runtime_provider_receipt")?;
    validate_artifact(&value["artifact_identity"])?;
    for name in [
        "seed",
        "context_length",
        "batch_size",
        "max_output_tokens",
        "timeout_seconds",
    ] {
        check(
            if name == "seed" {
                nonnegative_integer(&value["execution_settings"][name])
            } else {
                positive_integer(&value["execution_settings"][name])
            },
            format!(
                "{name} must be a {} integer",
                if name == "seed" {
                    "non-negative"
                } else {
                    "positive"
                }
            ),
        )?;
    }
    Ok(())
}
pub fn provider_bindings(
    artifact: &Value,
    observation: &Value,
    receipt: &Value,
    observation_bytes: &[u8],
    expected_runtime: Option<&str>,
) -> Result<()> {
    check(
        json_equal(&receipt["artifact_identity"], artifact),
        "receipt artifact_identity does not match the bound artifact",
    )?;
    check(
        receipt["scoring_observation_sha256"] == sha256_digest(observation_bytes)[7..],
        "receipt scoring_observation_sha256 does not match observation bytes",
    )?;
    check(
        receipt["plugin"]["name"] == receipt["capabilities"]["provider_name"]
            && receipt["plugin"]["name"] == observation["provider_name"],
        "receipt and observation provider names do not agree",
    )?;
    check(
        observation["artifact_identity_sha256"] == canonical_digest(artifact)?,
        "observation artifact_identity_sha256 does not match bound artifact",
    )?;
    check(
        receipt["capabilities"]["artifact_formats"]
            .as_array()
            .is_some_and(|a| a.contains(&artifact["artifact_format"])),
        "bound artifact format is not declared by provider capabilities",
    )?;
    if let Some(expected) = expected_runtime {
        check(
            is_digest(expected),
            "expected outer image digest is not canonical",
        )?;
        check(
            receipt["outer_image_digest"] == expected,
            "receipt outer_image_digest does not match expected runtime image",
        )?;
    }
    Ok(())
}
pub fn request_bindings(request: &Value, artifact: &Value, receipt: &Value) -> Result<()> {
    let provider = request["runtime"]["provider"].as_str().unwrap_or("");
    let settings = request["runtime"]["settings"]
        .as_object()
        .ok_or_else(|| integrity("normalized request runtime settings are invalid"))?;
    check(
        receipt["plugin"]["name"] == provider,
        "request provider does not match provider receipt",
    )?;
    let execution = [
        "batch_size",
        "context_length",
        "max_output_tokens",
        "seed",
        "timeout_seconds",
    ];
    for name in execution {
        if let Some(v) = settings.get(name) {
            check(
                json_equal(v, &receipt["execution_settings"][name]),
                format!("provider receipt does not match request runtime setting '{name}'"),
            )?;
        }
        check(
            if name == "seed" {
                nonnegative_integer(&receipt["execution_settings"][name])
            } else {
                positive_integer(&receipt["execution_settings"][name])
            },
            format!("provider receipt execution setting '{name}' must be an integer"),
        )?;
    }
    let names: BTreeSet<_> = settings.keys().map(String::as_str).collect();
    let mut expected: BTreeMap<&str, Value> = BTreeMap::new();
    match provider {
        "llama_cpp" => {
            check(
                artifact["artifact_format"] == "gguf",
                "llama_cpp request and GGUF artifact identity do not agree",
            )?;
            for (r, a) in [
                ("artifact_byte_length", "byte_length"),
                ("artifact_sha256", "sha256"),
                ("gguf_metadata_sha256", "gguf_metadata_sha256"),
                ("tensor_inventory_sha256", "tensor_inventory_sha256"),
                ("tokenizer_metadata_sha256", "tokenizer_metadata_sha256"),
            ] {
                expected.insert(r, artifact[a].clone());
            }
            for (r, a) in [
                ("backend_binary_sha256", "binary_sha256"),
                ("backend_source_sha256", "source_sha256"),
                ("backend_version", "version"),
            ] {
                expected.insert(r, receipt["backend"][a].clone());
            }
            for name in execution {
                expected.insert(name, receipt["execution_settings"][name].clone());
            }
            let v1: BTreeSet<_> = expected.keys().copied().collect();
            let v2: BTreeSet<_> = v1
                .iter()
                .copied()
                .chain(["cpu_threads", "prompt_batch_size", "prompt_microbatch_size"])
                .collect();
            check(
                names == v1 || names == v2,
                "llama_cpp request settings are not the closed supported set",
            )?;
            if names == v2 {
                for name in [
                    "context_length",
                    "cpu_threads",
                    "prompt_batch_size",
                    "prompt_microbatch_size",
                ] {
                    check(
                        settings[name].as_u64().is_some_and(|n| n > 0),
                        format!("llama_cpp request setting '{name}' must be a positive integer"),
                    )?;
                }
                check(
                    settings["cpu_threads"].as_u64().unwrap() <= 256,
                    "llama_cpp request cpu_threads exceeds the supported limit",
                )?;
                check(settings["prompt_batch_size"].as_u64().unwrap()<=settings["context_length"].as_u64().unwrap().min(4096),"llama_cpp request prompt_batch_size must not exceed context_length or the supported limit")?;
                check(
                    settings["prompt_microbatch_size"].as_u64().unwrap()
                        <= settings["prompt_batch_size"].as_u64().unwrap(),
                    "llama_cpp request prompt_microbatch_size must not exceed prompt_batch_size",
                )?;
            }
            check(
                receipt["backend"]["name"] == "llama.cpp",
                "llama_cpp provider receipt backend name is invalid",
            )?;
            check(
                receipt["backend"]["build_sha256"].is_null(),
                "llama_cpp provider receipt backend build digest must be null",
            )?;
        }
        "hf_transformers" | "hf_vision_text" => {
            check(
                artifact["artifact_format"] == "hf_snapshot",
                format!("{provider} request and HF artifact identity do not agree"),
            )?;
            let mut allowed: BTreeSet<_> = execution
                .into_iter()
                .chain([
                    "checkpoint_tree_sha256",
                    "immutable_revision",
                    "tokenizer_metadata_sha256",
                    "offline",
                ])
                .collect();
            let mut required: BTreeSet<_> = execution
                .into_iter()
                .chain(["offline", "tokenizer_metadata_sha256"])
                .collect();
            if provider == "hf_vision_text" {
                allowed.insert("processor_metadata_sha256");
                required.extend(["checkpoint_tree_sha256", "processor_metadata_sha256"]);
            }
            for name in [
                "checkpoint_tree_sha256",
                "immutable_revision",
                "tokenizer_metadata_sha256",
            ] {
                if !artifact[name].is_null() {
                    required.insert(name);
                }
                if let Some(observed) = settings.get(name) {
                    let observed = observed
                        .as_str()
                        .map(|s| json!(s.strip_prefix("sha256:").unwrap_or(s)))
                        .unwrap_or_else(|| observed.clone());
                    check(
                        observed == artifact[name],
                        format!(
                            "{provider} provider receipt does not match request setting '{name}'"
                        ),
                    )?;
                }
            }
            check(
                names.is_subset(&allowed) && required.is_subset(&names),
                format!("{provider} request settings are not the closed supported set"),
            )?;
            check(
                settings["offline"] == true,
                format!("{provider} request must require offline execution"),
            )?;
            if provider == "hf_vision_text" {
                check(
                    settings["processor_metadata_sha256"]
                        .as_str()
                        .is_some_and(|s| {
                            is_digest(&format!(
                                "sha256:{}",
                                s.strip_prefix("sha256:").unwrap_or(s)
                            ))
                        }),
                    "hf_vision_text request processor metadata digest is invalid",
                )?;
            }
        }
        "tensorrt_llm" => {
            check(
                artifact["artifact_format"] == "tensorrt_llm_engine",
                "tensorrt_llm request and TensorRT-LLM artifact identity do not agree",
            )?;
            for name in [
                "builder_config_sha256",
                "engine_bundle_tree_sha256",
                "engine_metadata_sha256",
                "file_inventory_sha256",
                "target_compute_capability",
                "tokenizer_metadata_sha256",
            ] {
                expected.insert(name, artifact[name].clone());
            }
            for (r, a) in [
                ("backend_build_sha256", "build_sha256"),
                ("backend_version", "version"),
                ("runner_binary_sha256", "binary_sha256"),
            ] {
                expected.insert(r, receipt["backend"][a].clone());
            }
            for name in execution {
                expected.insert(name, receipt["execution_settings"][name].clone());
            }
            check(
                names == expected.keys().copied().collect(),
                "tensorrt_llm request settings are not the closed supported set",
            )?;
            check(
                receipt["backend"]["name"] == "TensorRT-LLM"
                    && receipt["backend"]["source_sha256"].is_null(),
                "tensorrt_llm provider receipt backend identity is invalid",
            )?;
            check(receipt["device"]["compute_capability"]==artifact["target_compute_capability"],"tensorrt_llm provider receipt device compute capability does not match the artifact target")?;
        }
        _ => {
            check(
                artifact["artifact_format"] != "gguf",
                "llama_cpp request and GGUF artifact identity do not agree",
            )?;
        }
    }
    for (name, wanted) in expected {
        check(
            settings.get(name).is_some_and(|v| json_equal(v, &wanted)),
            format!("{provider} provider receipt does not match request setting '{name}'"),
        )?;
    }
    if [
        "llama_cpp",
        "hf_transformers",
        "hf_vision_text",
        "tensorrt_llm",
    ]
    .contains(&provider)
    {
        check(
            receipt["execution_settings"]["allow_network"] == false,
            format!("{provider} provider receipt must disable network access"),
        )?;
    }
    Ok(())
}
pub fn json_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Bool(value), Value::Number(_)) => json_equal(&json!(if *value { 1 } else { 0 }), b),
        (Value::Number(_), Value::Bool(value)) => json_equal(a, &json!(if *value { 1 } else { 0 })),
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| json_equal(v, w)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(v, w)| json_equal(v, w))
        }
        (Value::Number(a), Value::Number(b)) => {
            let sa = a.to_string();
            let sb = b.to_string();
            let ia = !sa.contains(['.', 'e', 'E']);
            let ib = !sb.contains(['.', 'e', 'E']);
            match (ia, ib) {
                (true, true) => sa == sb,
                (false, false) => a.as_f64() == b.as_f64(),
                (true, false) => b
                    .as_f64()
                    .is_some_and(|f| f.fract() == 0.0 && format!("{f:.0}") == sa),
                (false, true) => a
                    .as_f64()
                    .is_some_and(|f| f.fract() == 0.0 && format!("{f:.0}") == sb),
            }
        }
        _ => a == b,
    }
}

/// Validate the closed local judge collection reserves before provider loading.
pub fn validate_runtime_provider_collection(configuration: &Value, plan: &Value) -> Result<Value> {
    let names = configuration
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect::<BTreeSet<_>>());
    check(
        names
            == Some(BTreeSet::from([
                "profile",
                "max_calls",
                "max_output_tokens",
            ])),
        "runtime judge collection must contain profile, max_calls, and max_output_tokens",
    )?;
    check(
        configuration["profile"] == "runtime-provider-text-frozen-answer-v1",
        "unsupported runtime judge collection profile",
    )?;
    for (name, maximum) in [
        ("max_calls", 200_000u64),
        ("max_output_tokens", 1_000_000_000_000),
    ] {
        check(
            configuration[name]
                .as_u64()
                .is_some_and(|n| n > 0 && n <= maximum),
            format!("runtime judge {name} must be a bounded positive integer"),
        )?;
    }
    let expected = plan["schedule"]["expected_trials"]
        .as_u64()
        .ok_or_else(|| integrity("runtime judge plan expected_trials must be an integer"))?;
    let per_call = plan["judge"]["config"]["max_output_tokens"]
        .as_u64()
        .ok_or_else(|| integrity("runtime judge plan max_output_tokens must be an integer"))?;
    check(
        configuration["max_calls"].as_u64().unwrap() >= expected
            && expected.checked_mul(per_call).is_some_and(|reserve| {
                configuration["max_output_tokens"].as_u64().unwrap() >= reserve
            }),
        "runtime judge collection must reserve every planned inference",
    )?;
    Ok(
        json!({"max_calls": configuration["max_calls"], "max_output_tokens": configuration["max_output_tokens"]}),
    )
}

#[cfg(test)]
mod retained_request_tests {
    use super::*;
    #[test]
    fn tensorrt_import_binds_the_engine_artifact_format() {
        let hash = "a".repeat(64);
        let artifact = json!({"artifact_format":"tensorrt_llm_engine","builder_config_sha256":hash,"engine_bundle_tree_sha256":hash,"engine_metadata_sha256":hash,"file_inventory_sha256":hash,"target_compute_capability":"9.0","tokenizer_metadata_sha256":hash});
        let receipt = json!({"plugin":{"name":"tensorrt_llm"},"backend":{"name":"TensorRT-LLM","build_sha256":hash,"version":"1.0","binary_sha256":hash,"source_sha256":null},"execution_settings":{"allow_network":false,"batch_size":1,"context_length":128,"max_output_tokens":8,"seed":0,"timeout_seconds":30},"device":{"compute_capability":"9.0"}});
        let mut settings = artifact.as_object().unwrap().clone();
        settings.remove("artifact_format");
        settings.extend(receipt["execution_settings"].as_object().unwrap().clone());
        settings.remove("allow_network");
        for (field, value) in [
            ("backend_build_sha256", json!(hash)),
            ("backend_version", json!("1.0")),
            ("runner_binary_sha256", json!(hash)),
        ] {
            settings.insert(field.into(), value);
        }
        let side = json!({"runtime":{"provider":"tensorrt_llm","settings":settings}});
        request_bindings(&side, &artifact, &receipt).unwrap();
        let mut wrong = artifact.clone();
        wrong["artifact_format"] = json!("tensorrt_llm");
        assert!(request_bindings(&side, &wrong, &receipt).is_err());
        let mut wrong = receipt.clone();
        wrong["device"]["compute_capability"] = json!("8.0");
        assert!(request_bindings(&side, &artifact, &wrong).is_err());
    }
}
