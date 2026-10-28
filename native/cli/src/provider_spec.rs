//! Built-in provider admission for retained evidence, without model SDK loading.
use super::{err, Result};
use invarlock_native_contracts as contracts;
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path, sync::OnceLock};
fn profiles() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| {
        serde_json::from_str(include_str!("assets/provider-profiles.json"))
            .expect("embedded provider profiles")
    })
}
fn text(settings: &Value, key: &str, optional: bool) -> Result<Value> {
    let v = &settings[key];
    if optional && v.is_null() {
        return Ok(Value::Null);
    }
    let s = v
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.trim_matches(contracts::schedule::python_whitespace) == *s
                && s.chars().all(|c| c >= ' ')
        })
        .ok_or_else(|| err(format!("{key} must be non-empty trimmed text")))?;
    Ok(json!(s))
}
fn digest(settings: &Value, key: &str, optional: bool, prefix: bool) -> Result<Value> {
    let v = text(settings, key, optional)?;
    if v.is_null() {
        return Ok(v);
    }
    let s = v.as_str().unwrap();
    let d = if prefix {
        s.strip_prefix("sha256:").unwrap_or(s)
    } else {
        s
    };
    if d.len() != 64
        || !d
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(err(format!("{key} must be a sha256 digest")));
    }
    Ok(json!(d))
}
fn integer(v: &Value, key: &str, positive: bool) -> Result<()> {
    if !v.is_number() || !v.to_string().bytes().all(|b| b.is_ascii_digit()) || (positive && *v == 0)
    {
        return Err(err(format!(
            "{key} must be a {} integer",
            if positive { "positive" } else { "non-negative" }
        )));
    }
    Ok(())
}
pub(super) fn identity(side: &Value) -> Result<Value> {
    let provider = side["runtime"]["provider"].as_str().unwrap_or("");
    let profile = profiles().get(provider).ok_or_else(|| {
        err(format!(
            "provider {provider:?} requires an explicitly selected Python compatibility worker"
        ))
    })?;
    let settings = &side["runtime"]["settings"];
    let fields = settings
        .as_object()
        .ok_or_else(|| err("runtime settings must be an object"))?;
    let names: BTreeSet<_> = fields.keys().map(String::as_str).collect();
    let allowed: BTreeSet<_> = profile["allowed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let required: BTreeSet<_> = profile["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let unknown: Vec<_> = names.difference(&allowed).copied().collect();
    if !unknown.is_empty() {
        return Err(err(format!(
            "unsupported {provider} setting(s): {}",
            unknown.join(", ")
        )));
    }
    let missing: Vec<_> = required.difference(&names).copied().collect();
    if !missing.is_empty() {
        return Err(err(format!(
            "missing {provider} setting(s): {}",
            missing.join(", ")
        )));
    }
    for name in profile["positive"].as_array().unwrap() {
        let k = name.as_str().unwrap();
        if provider != "hf_transformers" || !settings[k].is_null() {
            integer(&settings[k], k, true)?;
        }
    }
    if provider != "hf_transformers" || !settings["seed"].is_null() {
        integer(&settings["seed"], "seed", false)?;
    }
    for name in profile["digests"].as_array().unwrap() {
        digest(settings, name.as_str().unwrap(), false, false)?;
    }
    let model = side["artifact"]["model_id"]
        .as_str()
        .ok_or_else(|| err("artifact model_id is required"))?;
    let format = "invarlock/model-artifact-identity-v1";
    let value = match provider {
        "hf_transformers" | "hf_vision_text" => {
            if provider == "hf_vision_text" {
                if settings["batch_size"] != 1 {
                    return Err(err("hf_vision_text currently requires batch_size=1"));
                }
                if settings["offline"] != true {
                    return Err(err("hf_vision_text requires offline=true"));
                }
                digest(settings, "processor_metadata_sha256", false, true)?;
            } else if !settings["offline"].is_null() && !settings["offline"].is_boolean() {
                return Err(err("offline must be boolean"));
            }
            let revision = text(settings, "immutable_revision", true)?;
            let tree = digest(
                settings,
                "checkpoint_tree_sha256",
                provider == "hf_transformers",
                true,
            )?;
            let tokenizer = digest(settings, "tokenizer_metadata_sha256", false, true)?;
            if revision.is_null() && tree.is_null() {
                return Err(err(
                    "hf_transformers requires an immutable identity revision or tree digest",
                ));
            }
            let mut model = model.to_owned();
            if provider == "hf_transformers" {
                let local = Path::new(&model).is_absolute()
                    || Path::new(&model).try_exists().unwrap_or(true)
                    || ["./", "../", "~/"]
                        .iter()
                        .any(|prefix| model.starts_with(prefix))
                    || model.contains('\\')
                    || (model.len() > 2
                        && model.as_bytes()[0].is_ascii_alphabetic()
                        && model.as_bytes()[1] == b':'
                        && [b'/', b'\\'].contains(&model.as_bytes()[2]));
                if local {
                    let t = tree.as_str().ok_or_else(|| {
                        err("local hf_transformers paths require checkpoint_tree_sha256")
                    })?;
                    model = format!("local-checkpoint-{}", &t[..12]);
                }
            }
            json!({"format_version":format,"artifact_format":"hf_snapshot","model_id":model,"immutable_revision":revision,"checkpoint_tree_sha256":tree,"tokenizer_metadata_sha256":tokenizer})
        }
        "llama_cpp" => {
            if settings["cpu_threads"].as_u64().unwrap_or(u64::MAX) > 256 {
                return Err(err(
                    "llama_cpp request cpu_threads exceeds the supported limit",
                ));
            }
            let context = settings["context_length"].as_u64().unwrap_or(u64::MAX);
            let batch = settings["prompt_batch_size"].as_u64().unwrap_or(u64::MAX);
            let micro = settings["prompt_microbatch_size"]
                .as_u64()
                .unwrap_or(u64::MAX);
            if batch > context.min(4096) {
                return Err(err("llama_cpp request prompt_batch_size must not exceed context_length or the supported limit"));
            }
            if micro > batch {
                return Err(err(
                    "llama_cpp request prompt_microbatch_size must not exceed prompt_batch_size",
                ));
            }
            let version = text(settings, "backend_version", false)?;
            let version = version.as_str().unwrap();
            if !version.is_ascii()
                || version.split_whitespace().collect::<Vec<_>>().join(" ") != version
                || !regex::Regex::new(profile["version_pattern"].as_str().unwrap())
                    .unwrap()
                    .is_match(version)
            {
                return Err(err(
                    "llama.cpp backend version does not match the closed grammar",
                ));
            }
            if regex::Regex::new(profile["sensitive_pattern"].as_str().unwrap())
                .unwrap()
                .is_match(version)
            {
                return Err(err(
                    "llama.cpp backend version contains sensitive-looking text",
                ));
            }
            let hash = settings["artifact_sha256"].as_str().unwrap();
            if model != format!("gguf-sha256-{hash}.gguf") {
                return Err(err(
                    "llama_cpp model_id must be the privacy-safe full GGUF digest name",
                ));
            }
            json!({"format_version":format,"artifact_format":"gguf","artifact_name":model,"sha256":hash,"byte_length":settings["artifact_byte_length"],"gguf_metadata_sha256":settings["gguf_metadata_sha256"],"tensor_inventory_sha256":settings["tensor_inventory_sha256"],"tokenizer_metadata_sha256":settings["tokenizer_metadata_sha256"]})
        }
        "tensorrt_llm" => {
            let version = text(settings, "backend_version", false)?;
            if version
                .as_str()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                != version.as_str().unwrap()
            {
                return Err(err("backend_version must use canonical single spacing"));
            }
            let capability = text(settings, "target_compute_capability", false)?;
            if !regex::Regex::new(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)\z")
                .unwrap()
                .is_match(capability.as_str().unwrap())
            {
                return Err(err(
                    "target_compute_capability must use major.minor notation",
                ));
            }
            if model
                != format!(
                    "tensorrt-llm-sha256-{}",
                    settings["engine_bundle_tree_sha256"].as_str().unwrap()
                )
            {
                return Err(err(
                    "tensorrt_llm model_id must be the privacy-safe full engine digest name",
                ));
            }
            json!({"format_version":format,"artifact_format":"tensorrt_llm_engine","bundle_name":model,"engine_bundle_tree_sha256":settings["engine_bundle_tree_sha256"],"file_inventory_sha256":settings["file_inventory_sha256"],"builder_config_sha256":settings["builder_config_sha256"],"tokenizer_metadata_sha256":settings["tokenizer_metadata_sha256"],"engine_metadata_sha256":settings["engine_metadata_sha256"],"target_compute_capability":capability})
        }
        _ => unreachable!(),
    };
    contracts::runtime::validate_artifact(&value)?;
    Ok(value)
}
pub(super) fn admit(
    side: &Value,
    artifact: &Value,
    receipt: &Value,
    task: &Value,
    metric: &Value,
    role: &str,
) -> Result<()> {
    let expected = identity(side)?;
    contracts::runtime::validate_artifact(artifact)?;
    contracts::runtime::validate_receipt(receipt)?;
    let provider = &side["runtime"]["provider"];
    if receipt["plugin"]["name"] != *provider
        || receipt["capabilities"]["provider_name"] != *provider
    {
        return Err(err(format!(
            "{role} provider evidence does not match the requested runtime"
        )));
    }
    let caps = &profiles()[provider.as_str().unwrap()]["capabilities"];
    for (field, selected, label) in [("tasks", task, "task"), ("metrics", metric, "metric")] {
        if !caps[field].as_array().unwrap().contains(selected)
            || !receipt["capabilities"][field]
                .as_array()
                .is_some_and(|a| a.contains(selected))
        {
            return Err(err(format!(
                "{role} provider evidence does not declare requested {label} {}",
                contracts::python_repr(selected)
            )));
        }
    }
    if expected != *artifact {
        return Err(err(format!(
            "{role} artifact identity does not match the requested runtime spec"
        )));
    }
    if receipt["outer_image_digest"].is_null() {
        return Err(err(format!(
            "{role} provider receipt lacks a strict outer runtime image digest"
        )));
    }
    contracts::runtime::request_bindings(side, artifact, receipt)
        .map_err(|e| err(format!("{role} {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_provider_identity_contracts() {
        let cases: Value =
            serde_json::from_str(include_str!("assets/provider-identity-corpus.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let actual = identity(&case["side"]);
            if let Some(expected) = case.get("result") {
                assert_eq!(
                    actual.unwrap_or_else(|e| panic!("{}: {e}", case["label"])),
                    *expected,
                    "{}",
                    case["label"]
                );
            } else {
                assert!(
                    actual.is_err(),
                    "{}: unexpected acceptance {actual:?}",
                    case["label"]
                );
            }
        }
    }
}
