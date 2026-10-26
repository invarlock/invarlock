//! Closed caller-owned trust profiles and immutable verification inputs.
use crate::{canonical, EvidenceError, Result};
use invarlock_native_contracts::{first_schema_error, parse_json, sha256_digest, SchemaErrorOrder};
use invarlock_native_core as analysis;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Secret material is deliberately excluded from the public configuration.
pub struct TrustSnapshot {
    pub config: Value,
    pub policy_bytes: Vec<u8>,
    pub signing_key_bytes: Vec<u8>,
    pub captured: bool,
}
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn relative(value: &Value, label: &str) -> Result<PathBuf> {
    let text = value
        .as_str()
        .ok_or_else(|| error(format!("{label} path is invalid")))?;
    if text.starts_with('/') || text.split('/').any(|p| matches!(p, "" | "." | "..")) {
        return Err(error(format!("{label} path is unsafe")));
    }
    Ok(PathBuf::from(text))
}
fn read(root: &fs::PinnedDirectory, path: &Path, label: &str, limit: usize) -> Result<Vec<u8>> {
    root.read_regular(path, limit)
        .map(|(raw, _)| raw)
        .map_err(|e| match e {
            fs::Error::Limit(_) => error(format!("{label} exceeds the {limit}-byte size limit")),
            fs::Error::Changed(_) => error(format!("{label} changed while being read")),
            _ => error(format!(
                "{label} could not be opened without following symlinks"
            )),
        })
}
pub fn load(path: &Path, key_override: Option<&[u8]>) -> Result<TrustSnapshot> {
    let profile = std::path::absolute(path)
        .map_err(|_| error("trust-input profile parent path is invalid"))?;
    let parent = profile
        .parent()
        .ok_or_else(|| error("trust-input profile parent path is invalid"))?;
    let root = fs::PinnedDirectory::open(parent, false).map_err(|_| {
        error("trust-input profile parent must be an existing non-symlink directory")
    })?;
    let raw = read(
        &root,
        Path::new(
            profile
                .file_name()
                .ok_or_else(|| error("trust-input profile path is invalid"))?,
        ),
        "trust-input profile",
        256 * 1024,
    )?;
    let payload = parse_json(&raw, "trust-input profile")?;
    if !payload.is_object() {
        return Err(error("trust-input profile must decode to a JSON object"));
    }
    let captured = payload["format"] == "invarlock/trust-inputs-v2";
    if let Some(violation) = first_schema_error(
        &payload,
        if captured {
            "trust_inputs_v2"
        } else {
            "trust_inputs"
        },
        SchemaErrorOrder::Instance,
    )? {
        let path = if violation.path.is_empty() {
            "<root>".to_string()
        } else {
            violation.path.join(".")
        };
        return Err(error(format!(
            "trust-input profile schema failed at {path}: {}",
            violation.message
        )));
    }
    let policy = relative(&payload["policy"]["path"], "policy")?;
    let key = relative(
        &payload["verifier"]["signing_key_path"],
        "verifier signing key",
    )?;
    let policy_bytes = read(
        &root,
        &policy,
        "policy",
        if captured {
            128 * 1024 * 1024
        } else {
            4 * 1024 * 1024
        },
    )?;
    let signing_key_bytes = match key_override {
        Some(bytes) => {
            if bytes.len() > 65536 {
                return Err(error(
                    "verifier key override exceeds the 65536-byte size limit",
                ));
            }
            bytes.to_vec()
        }
        None => read(&root, &key, "verifier signing key", 65536)?,
    };
    root.verify()
        .map_err(|_| error("trust-input profile parent changed while being read"))?;
    if captured {
        let policy = parse_json(&policy_bytes, "captured policy")?;
        if !policy.is_object() {
            return Err(error("captured policy must be a JSON object"));
        }
        analysis::validate_policy(&policy).map_err(|e| error(e.to_string()))?;
    }
    let anchors = &payload["anchors"];
    let mut config = json!({"policy_path":parent.join(policy),"verifier_signing_key_path":parent.join(key),
        "expected_request_digest":anchors["request_digest"],"expected_signer":anchors["evidence_signer_fingerprint"],
        "verifier_identity":payload["verifier"]["identity"],"trust_profile_digest":sha256_digest(&canonical(&payload)?)});
    if captured {
        config["expected_baseline_run"] = anchors["baseline_run_digest"].clone();
        config["expected_subject_run"] = anchors["subject_run_digest"].clone();
    } else {
        for side in ["baseline", "subject"] {
            config[format!("expected_{side}_artifact")] =
                anchors[format!("{side}_artifact_digest")].clone();
            config[format!("expected_{side}_runtime")] =
                anchors[format!("{side}_runtime_digest")].clone();
        }
        config["expected_schedule"] = anchors["schedule_digest"].clone();
        config["allow_installed_scorers"] = payload["allow_installed_scorers"].clone();
    }
    Ok(TrustSnapshot {
        config,
        policy_bytes,
        signing_key_bytes,
        captured,
    })
}
