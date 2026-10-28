#[path = "provider_spec.rs"]
mod provider_spec;
use super::{err, outcome, Args, Result};
use invarlock_native_contracts as contracts;
use invarlock_native_evidence as evidence;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
fn read(
    root: &fs::PinnedDirectory,
    reference: &Value,
    label: &str,
    limit: usize,
) -> Result<Vec<u8>> {
    let parts = contracts::request::reference_parts(reference.as_str().unwrap_or(""), label)?;
    root.read_regular(&parts.iter().collect::<PathBuf>(), limit)
        .map(|(raw, _)| raw)
        .map_err(|e| err(e.to_string()))
}
pub(super) fn run(
    a: &Args,
    request: &Value,
    root: &Path,
) -> Result<evidence::transaction::Outcome> {
    use base64::Engine;
    if request["execution"]["mode"] != "import" {
        return Err(err(
            "model execution requires an explicitly selected Python compatibility worker",
        ));
    }
    if a.flag("unsigned")
        || a.get("max-bootstrap-draws")
            .is_some_and(|s| s != "102400000")
    {
        return Err(err("captured controls are not valid for runtime requests"));
    }
    if request["comparison"].get("scorer_extension").is_some() {
        return Err(err("request requires an explicitly authorized scorer registry through the Python compatibility worker"));
    }
    if request["comparison"]["metric"] == "judge" {
        return Err(err(
            "runtime judge collection requires the explicit compatibility worker",
        ));
    }
    let mut enabled = Vec::new();
    for (env, label) in [
        ("INVARLOCK_ALLOW_NETWORK", "network access"),
        ("INVARLOCK_ALLOW_REMOTE_CODE", "remote code"),
        (
            "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
            "third-party provider discovery",
        ),
    ] {
        if std::env::var(env)
            .is_ok_and(|v| ["1", "true", "yes", "on"].contains(&v.trim().to_lowercase().as_str()))
        {
            enabled.push(label);
        }
    }
    if !enabled.is_empty() {
        return Err(err(format!(
            "strict evaluation requires disabled runtime opt-ins: {}",
            enabled.join(", ")
        )));
    }
    let key_path = a.path("signing-key")?;
    let key_path = key_path
        .as_str()
        .ok_or_else(|| err("an Ed25519 evidence-signing key is required"))?;
    let key_raw = fs::read_file(Path::new(key_path), 65536).map_err(|e| err(e.to_string()))?;
    let key = evidence::crypto::private_key(&key_raw)?;
    for side in ["baseline", "subject"] {
        provider_spec::identity(&request["comparison"][side])?;
    }
    let prepared = evidence::preparation::prepare_runtime_inputs(
        &json!({"authored_request":request,"root":root}),
    )?;
    let normalized = evidence::preparation::normalize_runtime_request(
        &json!({"authored_request":request,"schedule":prepared["schedule"],"observations":prepared["observations"]}),
    )?;
    let pinned = fs::PinnedDirectory::open(root, false).map_err(|e| err(e.to_string()))?;
    let destination =
        evidence::preparation::runtime_destination(root, &request["output"]["evidence"])?;
    let mut config = json!({"normalized_request":normalized,"schedule_base64":base64::engine::general_purpose::STANDARD.encode(prepared["canonical_schedule"].as_str().unwrap().as_bytes()),"policy_base64":prepared["policy_base64"],"observations":prepared["observations"],"destination":destination});
    let mut artifacts = json!({});
    let mut providers = json!({});
    for side in ["baseline", "subject"] {
        let mut material = json!({});
        let mut parsed = json!({});
        for (field, role) in [
            ("identity", "provider_identity"),
            ("receipt", "provider_receipt"),
            ("observation", "scoring_observation"),
            ("run_report", "run_report"),
            ("runtime_manifest", "runtime_manifest"),
            ("runtime_config", "runtime_config"),
        ] {
            let raw = read(
                &pinned,
                &request["execution"][side][field],
                role,
                64 * 1024 * 1024,
            )?;
            parsed[role] = if role == "runtime_config" {
                crate::request::parse(&raw)?
            } else {
                contracts::parse_json(&raw, role)?
            };
            material[role] = json!(base64::engine::general_purpose::STANDARD.encode(&raw));
            if role == "provider_identity" {
                artifacts[side] = json!(contracts::sha256_digest(&raw));
            }
        }
        provider_spec::admit(
            &request["comparison"][side],
            &parsed["provider_identity"],
            &parsed["provider_receipt"],
            &request["comparison"]["task"],
            &request["comparison"]["metric"],
            side,
        )?;
        let expected = json!({"format":"invarlock/runtime-side-config-v1","role":side,"provider":request["comparison"][side]["runtime"]["provider"],"artifact_identity_sha256":contracts::canonical_json_no_newline(&parsed["provider_identity"]).map(|r|contracts::sha256_digest(&r)[7..].to_owned())?,"schedule_sha256":prepared["schedule_digest"].as_str().unwrap().trim_start_matches("sha256:"),"policy_digest":prepared["policy_digest"]});
        if !contracts::runtime::json_equal(&parsed["runtime_config"], &expected) {
            return Err(err(format!("{side} runtime config does not bind role, provider, artifact, schedule, and policy")));
        }
        config[format!("{side}_material")] = material;
        providers[side] = request["comparison"][side]["runtime"]["provider"].clone();
    }
    let raw = read(
        &pinned,
        &request["execution"]["records"],
        "imported paired records",
        64 * 1024 * 1024,
    )?;
    let pairs = contracts::parse_json(&raw, "imported paired records")?;
    if contracts::canonical_json(&pairs)? != raw {
        return Err(err("imported paired records must use canonical JSON"));
    }
    pinned.verify().map_err(|e| err(e.to_string()))?;
    if a.flag("preflight") {
        let mut checks = vec![
            "request",
            "artifacts",
            "dataset_schedule",
            "policy",
            "provider_capabilities",
            "signing_key",
            "output_destination",
        ];
        let mut preflight = json!({"format_version":"invarlock/evaluation-preflight-v2","ok":true,"execution_mode":"import","output":request["output"]["evidence"],"schedule_digest":format!("sha256:{}",prepared["schedule_digest"].as_str().unwrap().trim_start_matches("sha256:")),"policy_digest":prepared["policy_digest"],"artifact_digests":artifacts,"evidence_signer_fingerprint":evidence::crypto::fingerprint(&key.verifying_key()),"request_digest":contracts::canonical_json_digest(&normalized)?,"record_count":prepared["schedule"]["records"].as_array().unwrap().len(),"providers":providers});
        if let Some(requirements) = prepared["sample_requirements"]
            .as_object()
            .filter(|m| !m.is_empty())
        {
            checks.push("sample_record_count");
            let width = requirements
                .keys()
                .find(|k| k.as_str() != "minimum_record_count")
                .unwrap();
            preflight["sample_qualification"] = json!({"record_count":{"minimum":requirements["minimum_record_count"],"observed":prepared["schedule"]["records"].as_array().unwrap().len(),"status":"pass"},"interval_width":{"maximum":requirements[width],"unit":if width.ends_with("_ratio"){"ratio"}else{"percentage_points"},"status":"pending_execution"}});
        }
        preflight["checks"] = json!(checks);
        return outcome(preflight, 0);
    }
    let derived = evidence::native_publication::derive_transaction(&config, None)?;
    if !contracts::runtime::json_equal(&derived["paired_records"], &pairs) {
        return Err(err(
            "imported paired records do not equal verifier-derived pairs",
        ));
    }
    config["comparison_id"] = derived["comparison_id"].clone();
    config["expected_paired_records"] = pairs;
    let published = evidence::native_publication::publish_with_key(&config, None, Some(&key_raw))?;
    let mut output = outcome(
        json!({"format_version":"invarlock/evaluation-result-v1","ok":true,"comparison_id":derived["comparison_id"],"evidence":destination,"pack_manifest_digest":published["pack_manifest_digest"]}),
        if a.flag("fail-on-policy") && published["policy_verdict"] != "pass" {
            7
        } else {
            0
        },
    )?;
    output.terminal = published["policy_verdict"].as_str().map(str::to_owned);
    Ok(output)
}
