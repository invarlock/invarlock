//! Native authoring of independently replayable runtime-import evidence.
use crate::native_pack::{replay, ScorerReplay, EVIDENCE, ROLES};
use crate::{canonical, crypto, digest, is_digest, required_string, EvidenceError, Result};
use invarlock_native_contracts::{
    canonical_json_no_newline, first_schema_error, parse_json, python_repr, runtime, sha256_digest,
    validate_schema, SchemaErrorOrder,
};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
const MAX_BYTES: usize = 64 * 1024 * 1024;
const SIDE_FILES: [(&str, &str); 6] = [
    ("run_report", "report.json"),
    ("runtime_manifest", "runtime.manifest.json"),
    ("runtime_config", "run.yaml"),
    ("provider_identity", "model-artifact.identity.json"),
    ("provider_receipt", "runtime-provider.receipt.json"),
    ("scoring_observation", "runtime-scoring.observation.json"),
];
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn io_error(e: impl std::fmt::Display) -> EvidenceError {
    error(e.to_string())
}
fn hex(bytes: &[u8]) -> String {
    sha256_digest(bytes)[7..].into()
}
fn bare(value: &Value) -> Result<String> {
    Ok(hex(&canonical_json_no_newline(value)?))
}
fn object(raw: &[u8], label: &str) -> Result<Value> {
    let value = parse_json(raw, label)?;
    if !value.is_object() {
        return Err(error(format!("{label} must be a JSON object")));
    }
    Ok(value)
}
fn material(config: &Value, name: &str, label: &str) -> Result<Vec<u8>> {
    if let Some(encoded) = config.get(format!("{name}_base64")).and_then(Value::as_str) {
        if encoded.len() > MAX_BYTES.div_ceil(3) * 4 {
            return Err(error(format!("{label} exceeds its byte limit")));
        }
        let raw = crate::decode_base64(encoded)
            .map_err(|_| error(format!("{label} is not valid base64")))?;
        if raw.len() > MAX_BYTES {
            return Err(error(format!("{label} exceeds its byte limit")));
        }
        Ok(raw)
    } else {
        fs::read_file(
            Path::new(required_string(config, &format!("{name}_path"))?),
            MAX_BYTES,
        )
        .map_err(io_error)
    }
}
fn side_material(config: &Value, side: &str) -> Result<BTreeMap<String, Vec<u8>>> {
    if let Some(values) = config.get(format!("{side}_material")) {
        let values = values
            .as_object()
            .ok_or_else(|| error("side material must be an object"))?;
        if values.len() != SIDE_FILES.len()
            || SIDE_FILES
                .iter()
                .any(|(role, _)| !values.contains_key(*role))
        {
            return Err(error(
                "side material must contain exactly the six runtime files",
            ));
        }
        let mut files = BTreeMap::new();
        for (role, _) in SIDE_FILES {
            let item = json!({format!("{role}_base64"):values[role]});
            files.insert(format!("{side}_{role}"), material(&item, role, role)?);
        }
        Ok(files)
    } else {
        side_bytes(
            Path::new(required_string(config, &format!("{side}_directory"))?),
            side,
        )
    }
}
fn normalized(value: &Value, label: &str) -> Result<String> {
    let value = value.as_str().unwrap_or("").trim().to_ascii_lowercase();
    let value = if value.starts_with("sha256:") {
        value
    } else {
        format!("sha256:{value}")
    };
    if !is_digest(&value) {
        return Err(error(format!("{label} must be a sha256:... digest")));
    }
    Ok(value)
}
fn side_bytes(directory: &Path, side: &str) -> Result<BTreeMap<String, Vec<u8>>> {
    if directory.is_symlink() || !directory.is_dir() {
        return Err(error("runtime import side must be a real directory"));
    }
    let root = fs::PinnedDirectory::open(directory, false).map_err(io_error)?;
    let mut files = BTreeMap::new();
    let mut identities = Vec::new();
    for (role, file) in SIDE_FILES {
        let (raw, identity) = root
            .read_regular(Path::new(file), MAX_BYTES)
            .map_err(io_error)?;
        identities.push((file, identity));
        files.insert(format!("{side}_{role}"), raw);
    }
    for (file, identity) in identities {
        root.verify_regular(Path::new(file), &identity)
            .map_err(io_error)?;
    }
    root.verify().map_err(io_error)?;
    Ok(files)
}
fn new_destination(path: &Path, label: &str) -> Result<()> {
    if path.file_name().is_none() {
        return Err(error(format!("{label} destination must name a directory")));
    }
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::PinnedDirectory::open(parent, true).map_err(io_error)?;
    if std::fs::symlink_metadata(path).is_ok() {
        return Err(error(format!(
            "{label} destination already exists: {}",
            path.display()
        )));
    }
    Ok(())
}
fn stage_files(
    destination: &Path,
    files: &BTreeMap<String, Vec<u8>>,
    read_only: bool,
) -> Result<()> {
    let mut stage = fs::StagedDirectory::new(destination).map_err(io_error)?;
    let mut directories = BTreeSet::new();
    for path in files.keys() {
        let mut parent = Path::new(path).parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            directories.insert(path.to_owned());
            parent = path.parent();
        }
    }
    let mut ordered: Vec<_> = directories.into_iter().collect();
    ordered.sort_by_key(|path| path.components().count());
    for path in ordered {
        stage.create_directory(&path).map_err(io_error)?;
    }
    for (path, raw) in files {
        stage
            .write(Path::new(path), raw, if read_only { 0o444 } else { 0o600 })
            .map_err(io_error)?;
    }
    if read_only {
        stage.commit_read_only().map_err(io_error)
    } else {
        stage.commit().map_err(io_error)
    }
}
type PublicationInputs = (
    Value,
    Value,
    Vec<u8>,
    BTreeMap<&'static str, Value>,
    Value,
    BTreeMap<String, Vec<u8>>,
);
fn publication_inputs(config: &Value) -> Result<PublicationInputs> {
    let request = config["normalized_request"].clone();
    if let Some(e) = first_schema_error(&request, "evaluation_request", SchemaErrorOrder::Instance)?
    {
        let path = if e.path.is_empty() {
            "<root>".into()
        } else {
            e.path.join(".")
        };
        return Err(error(format!(
            "normalized request schema failed at {path}: {}",
            e.message
        )));
    }
    let schedule_raw = material(config, "schedule", "runtime behavioral schedule")?;
    let schedule = object(&schedule_raw, "runtime behavioral schedule")?;
    replay::schedule(&schedule, &schedule_raw, &request)?;
    let policy_raw = material(config, "policy", "policy")?;
    object(&policy_raw, "policy")?;
    let mut identities = BTreeMap::new();
    for role in ROLES {
        let supplied = &config["identities"][role];
        let mut value = json!({"format":"invarlock/evidence-input-identity-v1","role":role,"digest":normalized(&supplied["digest"],&format!("{role} digest"))?});
        for (name, max) in [("locator", 4096), ("media_type", 255)] {
            if let Some(field) = supplied.get(name).filter(|v| !v.is_null()) {
                let text = field.as_str().unwrap_or("").trim();
                if text.is_empty()
                    || if name == "locator" {
                        text.len() > max
                    } else {
                        text.chars().count() > max
                    }
                {
                    return Err(error(format!("{role} {name} is invalid")));
                }
                value[name] = json!(text);
            }
        }
        identities.insert(role, value);
    }
    if identities["dataset"]["digest"] != sha256_digest(&schedule_raw) {
        return Err(error(
            "dataset identity must equal the canonical evaluation schedule digest",
        ));
    }
    if identities["policy"]["digest"] != sha256_digest(&policy_raw) {
        return Err(error("policy identity does not match policy bytes"));
    }
    if request["comparison"]["policy"] != "inputs/policy.json" {
        return Err(error(
            "normalized request policy must name the canonical policy identity",
        ));
    }
    let anchors = json!({"policy_digest":identities["policy"]["digest"],"artifact_digests":{"baseline":identities["baseline"]["digest"],"subject":identities["subject"]["digest"]},"runtime_digests":{"baseline":identities["baseline_runtime"]["digest"],"subject":identities["subject_runtime"]["digest"]},"request_digest":digest(&request)?});
    let mut files = BTreeMap::from([
        ("request".into(), canonical(&request)?),
        ("schedule".into(), schedule_raw),
    ]);
    for side in ["baseline", "subject"] {
        files.extend(side_material(config, side)?);
    }
    Ok((request, schedule, policy_raw, identities, anchors, files))
}
/// Reconstruct identities and paired records from one owned worker/import snapshot.
pub fn derive_transaction(config: &Value, bridge: Option<&dyn ScorerReplay>) -> Result<Value> {
    let configured = transaction_config(config)?;
    let (request, schedule, policy, identities, anchors, held) = publication_inputs(&configured)?;
    let loaded = held
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    // Judge capture compares the same frozen answers using exact-match pairs.
    // Keep the original validated request and its identity for every binding.
    let mut scoring_request = request.clone();
    if scoring_request["comparison"]["metric"] == "judge" {
        scoring_request["comparison"]["metric"] = json!("exact_match");
    }
    let paired = replay::derive(
        &loaded,
        &scoring_request,
        &schedule,
        &identities,
        &anchors,
        bridge,
    )?;
    let failures = replay::runtime_checks(&loaded, &anchors);
    if let Some(failure) = failures.first() {
        return Err(error(failure.clone()));
    }
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    let canonical_pairs = canonical(&paired)?;
    for raw in [
        held["request"].as_slice(),
        held["baseline_provider_identity"].as_slice(),
        held["subject_provider_identity"].as_slice(),
        held["schedule"].as_slice(),
        policy.as_slice(),
        anchors["runtime_digests"]["baseline"]
            .as_str()
            .unwrap()
            .as_bytes(),
        anchors["runtime_digests"]["subject"]
            .as_str()
            .unwrap()
            .as_bytes(),
        canonical_pairs.as_slice(),
    ] {
        digest.update((raw.len() as u64).to_be_bytes());
        digest.update(raw);
    }
    Ok(
        json!({"paired_records":paired,"comparison_id":format!("comparison-{:x}",digest.finalize())[..43].to_string(),"artifact_digests":anchors["artifact_digests"],"runtime_digests":anchors["runtime_digests"]}),
    )
}
fn transaction_config(config: &Value) -> Result<Value> {
    if config.get("identities").is_some() {
        return Ok(config.clone());
    }
    let mut config = config.clone();
    let schedule = material(&config, "schedule", "runtime behavioral schedule")?;
    let policy = material(&config, "policy", "policy")?;
    let mut identities = json!({"dataset":{"digest":sha256_digest(&schedule),"locator":"schedule/runtime-behavioral-schedule.json","media_type":"application/json"},"policy":{"digest":sha256_digest(&policy),"locator":"inputs/policy.json","media_type":"application/json"}});
    for side in ["baseline", "subject"] {
        let files = side_material(&config, side)?;
        let receipt = replay::typed_sidecar(
            &files[&format!("{side}_provider_receipt")],
            "runtime provider receipt",
            "runtime_provider_receipt",
        )?;
        let runtime_digest = receipt["outer_image_digest"].as_str().ok_or_else(|| {
            error(format!(
                "{side} provider receipt lacks a strict outer runtime image digest"
            ))
        })?;
        let locator =
            config["normalized_request"]["comparison"][side]["artifact"]["locator"].clone();
        identities[side] = json!({"digest":sha256_digest(&files[&format!("{side}_provider_identity")]),"locator":locator,"media_type":"application/json"});
        identities[format!("{side}_runtime")] =
            json!({"digest":runtime_digest,"locator":format!("runtime:{runtime_digest}")});
    }
    config["identities"] = identities;
    Ok(config)
}
pub fn publish(config: &Value, bridge: Option<&dyn ScorerReplay>) -> Result<Value> {
    publish_with_key(config, bridge, None)
}
pub fn publish_with_key(
    config: &Value,
    bridge: Option<&dyn ScorerReplay>,
    signing_key_bytes: Option<&[u8]>,
) -> Result<Value> {
    let configured = transaction_config(config)?;
    let config = &configured;
    let destination = Path::new(required_string(config, "destination")?);
    new_destination(destination, "evidence")?;
    let (request, schedule, policy_raw, identities, anchors, mut held) =
        publication_inputs(config)?;
    let loaded: BTreeMap<&str, &[u8]> = held
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let paired = replay::derive(&loaded, &request, &schedule, &identities, &anchors, bridge)?;
    if config
        .get("expected_paired_records")
        .is_some_and(|v| !v.is_null() && !runtime::json_equal(v, &paired))
    {
        return Err(error(
            "publication paired records do not match transaction-derived records",
        ));
    }
    let policy = object(&policy_raw, "policy")?;
    let comparison_id = required_string(config, "comparison_id")?;
    let report = invarlock_native_core::build_comparison_report(
        comparison_id,
        &paired,
        &policy,
        identities["policy"]["digest"].as_str().unwrap_or(""),
        "invarlock/comparison-report-v3",
    )
    .map_err(io_error)?;
    let runtime_errors = replay::runtime_checks(&loaded, &anchors);
    if let Some(error_message) = runtime_errors.first() {
        return Err(error(error_message.clone()));
    }
    held.insert("evaluation_report".into(), canonical(&report)?);
    let mut payloads = BTreeMap::new();
    let mut input_manifest = json!({});
    let mut evidence_manifest = json!({});
    for role in ROLES {
        let relative = format!("inputs/{role}.json");
        let bytes = canonical(&identities[role])?;
        input_manifest[role] = json!({"path":relative,"digest":sha256_digest(&bytes),"material_digest":identities[role]["digest"]});
        payloads.insert(relative, bytes);
    }
    for (role, relative) in EVIDENCE {
        let raw = held
            .remove(role)
            .ok_or_else(|| error(format!("missing publication {role}")))?;
        evidence_manifest[role] = json!({"path":relative,"digest":sha256_digest(&raw)});
        payloads.insert(relative.into(), raw);
    }
    let observations = config["observations"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if observations.len() > 64 {
        return Err(error("evidence pack supports at most 64 observations"));
    }
    let mut requested = Vec::new();
    let mut observation_manifest = json!({});
    for observation in observations {
        let id = required_string(&observation, "id")?;
        let raw = material(
            &observation,
            "payload",
            &format!("observation {id} payload"),
        )?;
        let payload = object(&raw, &format!("observation {id} payload"))?;
        if canonical(&payload)? != raw {
            return Err(error(format!(
                "observation {} payload must use canonical JSON",
                python_repr(&json!(id))
            )));
        }
        let envelope = json!({"format":"invarlock/evidence-observation-v1","observation_id":id,"kind":observation["kind"],"scope":observation["scope"],"authority":"observation","bindings":{"comparison_id":comparison_id,"schedule_digest":identities["dataset"]["digest"],"policy_digest":identities["policy"]["digest"],"artifact_digests":{"baseline":identities["baseline"]["digest"],"subject":identities["subject"]["digest"]}},"payload":payload});
        validate_schema(&envelope, "evidence_observation")?;
        if observation_manifest.get(id).is_some() {
            return Err(error(format!(
                "duplicate observation_id: {}",
                python_repr(&json!(id))
            )));
        }
        let relative = format!("observations/{id}.json");
        let bytes = canonical(&envelope)?;
        observation_manifest[id] = json!({"path":relative,"digest":sha256_digest(&bytes),"kind":observation["kind"],"scope":observation["scope"]});
        payloads.insert(relative, bytes);
        requested.push(json!({"id":id,"kind":observation["kind"],"scope":observation["scope"],"payload_digest":sha256_digest(&raw)}));
    }
    requested.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    if request.get("observations").cloned().unwrap_or(json!([])) != json!(requested) {
        return Err(error(
            "normalized request observations do not match publication inputs",
        ));
    }
    for (relative, raw) in &payloads {
        if raw.is_empty() {
            return Err(error(format!("{relative} must contain non-empty bytes")));
        }
        if raw.len() > MAX_BYTES {
            return Err(error(format!(
                "{relative} exceeds the {MAX_BYTES}-byte limit"
            )));
        }
    }
    let paired_bytes = canonical(&paired)?;
    let paired_digest = sha256_digest(&paired_bytes);
    payloads.insert("records/paired-records.json".into(), paired_bytes);
    let disk_key;
    let key_raw = match signing_key_bytes {
        Some(raw) if raw.len() <= 65536 => raw,
        Some(_) => return Err(error("signing key exceeds its byte limit")),
        None => {
            disk_key = fs::read_file(
                Path::new(required_string(config, "signing_key_path")?),
                65536,
            )
            .map_err(io_error)?;
            &disk_key
        }
    };
    let key = crypto::private_key(key_raw)?;
    let fingerprint = crypto::fingerprint(&key.verifying_key());
    let checksums = payloads
        .iter()
        .map(|(path, raw)| format!("{}  {path}\n", hex(raw)))
        .collect::<String>()
        .into_bytes();
    let mut manifest = json!({"format":"invarlock/evidence-pack-v1","comparison_id":comparison_id,"inputs":input_manifest,"evidence":evidence_manifest,"paired_records":{"path":"records/paired-records.json","digest":paired_digest,"count":paired["records"].as_array().map(Vec::len).unwrap_or(0)},"checksums_sha256":"checksums.sha256","checksums_sha256_digest":hex(&checksums),"signing_key_fingerprint":fingerprint});
    if observation_manifest
        .as_object()
        .is_some_and(|o| !o.is_empty())
    {
        manifest["observations"] = observation_manifest;
    }
    validate_schema(&manifest, "evidence_pack")?;
    let manifest_bytes = canonical(&manifest)?;
    let manifest_digest = sha256_digest(&manifest_bytes);
    let signature = json!({"format":"invarlock/evidence-pack-signature-v1","algorithm":"ed25519","signing_key_fingerprint":fingerprint,"public_key":{"encoding":"pem","value":crypto::public_pem(&key.verifying_key())?},"signature":{"encoding":"base64","value":crypto::signature_value(&key,&manifest_bytes)}});
    payloads.insert("manifest.json".into(), manifest_bytes);
    payloads.insert("manifest.signature.json".into(), canonical(&signature)?);
    payloads.insert("checksums.sha256".into(), checksums);
    stage_files(destination, &payloads, true)?;
    Ok(
        json!({"evidence_path":destination,"pack_manifest_digest":manifest_digest,"policy_verdict":report["verdict"]}),
    )
}
fn schedule_file(config: &Value) -> Result<Value> {
    let raw = material(config, "schedule", "runtime behavioral schedule")?;
    let schedule = object(&raw, "runtime behavioral schedule")?;
    let request = json!({"execution":{"mode":"import"},"comparison":{"task":schedule["task"],"dataset":"schedule/runtime-behavioral-schedule.json"}});
    replay::schedule(&schedule, &raw, &request)?;
    Ok(schedule)
}
fn side_bindings(config: &Value) -> Result<&str> {
    let role = config["role"].as_str().unwrap_or("");
    if !["baseline", "subject"].contains(&role) {
        return Err(error("role must be baseline or subject"));
    }
    if !config["policy_digest"].as_str().is_some_and(is_digest) {
        return Err(error("policy_digest must be a sha256 digest"));
    }
    Ok(role)
}
fn bind_records(records: &Value, schedule: &Value) -> Result<()> {
    let rows = records
        .as_array()
        .filter(|r| !r.is_empty())
        .ok_or_else(|| error("records must be a non-empty tuple"))?;
    if rows.len() > 10000 {
        return Err(error("records exceed the 10000-record limit"));
    }
    let scheduled = schedule["records"]
        .as_array()
        .ok_or_else(|| error("schedule records are invalid"))?;
    if rows.len() != scheduled.len()
        || rows.iter().zip(scheduled).any(|(r, s)| {
            r["record_id"] != s["record_id"] || r["input_sha256"] != s["input_sha256"]
        })
    {
        return Err(error(
            "external record order and input identities must exactly match the schedule",
        ));
    }
    for row in rows {
        let label = python_repr(&row["record_id"]);
        if row["status"] != "ok" {
            return Err(error(format!("external record {label} is not successful")));
        }
        if let Some(text) = row["output_text"].as_str() {
            if row["output_sha256"] != hex(text.as_bytes()) {
                return Err(error(format!(
                    "external record {label} output digest is invalid"
                )));
            }
        }
    }
    Ok(())
}
fn records(config: &Value, schedule: &Value) -> Result<Value> {
    let mut rows = Vec::new();
    if let Some(path) = config.get("records_path").and_then(Value::as_str) {
        let raw = fs::read_file(Path::new(path), MAX_BYTES).map_err(io_error)?;
        if raw.is_empty() || raw.last() != Some(&b'\n') {
            return Err(error(
                "external scoring records must be non-empty newline-terminated JSONL",
            ));
        }
        for (index, line) in raw.split(|c| *c == b'\n').enumerate() {
            if index > 0 && line.is_empty() && index == raw.iter().filter(|b| **b == b'\n').count()
            {
                break;
            }
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.is_empty() {
                return Err(error(format!(
                    "external record line {} must not be blank",
                    index + 1
                )));
            }
            let value = object(line, &format!("external record line {}", index + 1))?;
            rows.push(value);
        }
        if rows.len() > 10000 {
            return Err(error(
                "external scoring records exceed the 10000-record limit",
            ));
        }
    } else {
        rows = config["records"]
            .as_array()
            .ok_or_else(|| error("records must be a non-empty tuple"))?
            .clone();
    }
    for (index, row) in rows.iter_mut().enumerate() {
        let fields = row.as_object_mut().ok_or_else(|| {
            error(format!(
                "external record line {} must be a JSON object",
                index + 1
            ))
        })?;
        for name in ["input_sha256", "record_id", "status"] {
            if !fields.contains_key(name) {
                return Err(error(format!(
                    "external record line {} is missing {}",
                    index + 1,
                    python_repr(&json!(name))
                )));
            }
        }
        let allowed = BTreeSet::from([
            "record_id",
            "input_sha256",
            "status",
            "output_text",
            "output_sha256",
            "logprob_sum",
            "token_count",
            "utf8_byte_count",
            "error_code",
        ]);
        if let Some(name) = fields
            .keys()
            .filter(|k| !allowed.contains(k.as_str()))
            .min()
        {
            return Err(error(format!("external record line {} has unsupported field {}; aggregate summaries are not accepted",index+1,python_repr(&json!(name)))));
        }
        for name in allowed {
            fields.entry(name).or_insert(Value::Null);
        }
    }
    let value = json!(rows);
    bind_records(&value, schedule)?;
    Ok(value)
}
fn observation(
    provider: &Value,
    artifact: &Value,
    schedule: &Value,
    records: Value,
) -> Result<Value> {
    let value = json!({"format_version":"invarlock/runtime-scoring-observation-v1","provider_name":provider,"artifact_identity_sha256":bare(artifact)? ,"schedule_sha256":bare(schedule)? ,"aggregate_source_sha256":bare(&records)? ,"records":records});
    runtime::validate_observation(&value)?;
    Ok(value)
}
fn report(observation: &Value, bytes: &[u8]) -> Value {
    json!({"format":"invarlock/runtime-side-report-v1","provider":observation["provider_name"],"artifact_identity_sha256":observation["artifact_identity_sha256"],"scoring_observation_sha256":hex(bytes),"schedule_sha256":observation["schedule_sha256"],"record_count":observation["records"].as_array().map(Vec::len).unwrap_or(0)})
}
fn side_config(role: &str, observation: &Value, policy: &Value) -> Value {
    json!({"format":"invarlock/runtime-side-config-v1","role":role,"provider":observation["provider_name"],"artifact_identity_sha256":observation["artifact_identity_sha256"],"schedule_sha256":observation["schedule_sha256"],"policy_digest":policy})
}
fn pretty(value: &Value) -> Result<Vec<u8>> {
    let sorted: Value =
        serde_json::from_slice(&canonical_json_no_newline(value)?).map_err(io_error)?;
    let text = serde_json::to_string_pretty(&sorted).map_err(io_error)?;
    let mut out = String::new();
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out.push('\n');
    Ok(out.into_bytes())
}
fn image_ref(reference: &str, digest: &str) -> Result<String> {
    if !is_digest(digest) {
        return Err(error(
            "runtime provenance requires a lowercase image digest",
        ));
    }
    if reference.is_empty()
        || reference != reference.trim()
        || reference.starts_with(['/', '\\'])
        || reference.as_bytes().get(1) == Some(&b':')
            && reference.as_bytes()[0].is_ascii_alphabetic()
        || reference.chars().any(|c| (c as u32) < 32)
        || reference.matches('@').count() > 1
    {
        return Err(error(
            "runtime image reference must be a portable reference",
        ));
    }
    if let Some((name, embedded)) = reference.split_once('@') {
        if name.is_empty() {
            return Err(error("runtime image reference must name an image"));
        }
        if !is_digest(embedded) {
            return Err(error("runtime image reference contains an invalid digest"));
        }
        if embedded != digest {
            return Err(error("runtime image reference and digest do not agree"));
        }
        return Ok(reference.into());
    }
    Ok(if reference == digest {
        reference.into()
    } else {
        format!("{reference}@{digest}")
    })
}
fn manifest_timestamp(text: &str) -> Result<()> {
    let invalid = || error("generated_at_utc must be an ISO 8601 timestamp");
    let raw = text.as_bytes();
    if raw.len() < 4 || !raw[..4].iter().all(u8::is_ascii_digit) {
        return Err(invalid());
    }
    let year = text[..4].parse::<u32>().map_err(|_| invalid())?;
    if year == 0 {
        return Err(invalid());
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let digit = |s: &str| -> Option<u32> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            None
        } else {
            s.parse().ok()
        }
    };
    let date_len =
        if raw.get(4) == Some(&b'-') && raw.get(5) == Some(&b'W') || raw.get(4) == Some(&b'W') {
            let extended = raw.get(4) == Some(&b'-');
            let start = if extended { 6 } else { 5 };
            let week = text
                .get(start..start + 2)
                .and_then(digit)
                .ok_or_else(invalid)?;
            let mut end = start + 2;
            let day = if extended && raw.get(end) == Some(&b'-') {
                end += 2;
                text.get(end - 1..end).and_then(digit).ok_or_else(invalid)?
            } else if !extended && raw.get(end).is_some_and(u8::is_ascii_digit) {
                end += 1;
                text.get(end - 1..end).and_then(digit).ok_or_else(invalid)?
            } else {
                1
            };
            let prior = year - 1;
            let weekday = (prior + prior / 4 - prior / 100 + prior / 400) % 7;
            let maximum = if weekday == 3 || weekday == 2 && leap {
                53
            } else {
                52
            };
            if week == 0 || week > maximum || !(1..=7).contains(&day) {
                return Err(invalid());
            }
            end
        } else {
            let extended = raw.get(4) == Some(&b'-');
            let (month, day, end) = if extended {
                if raw.get(7) != Some(&b'-') {
                    return Err(invalid());
                }
                (
                    text.get(5..7).and_then(digit),
                    text.get(8..10).and_then(digit),
                    10,
                )
            } else {
                (
                    text.get(4..6).and_then(digit),
                    text.get(6..8).and_then(digit),
                    8,
                )
            };
            let month = month.ok_or_else(invalid)?;
            let day = day.ok_or_else(invalid)?;
            let maximum = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => {
                    if leap {
                        29
                    } else {
                        28
                    }
                }
                _ => return Err(invalid()),
            };
            if day == 0 || day > maximum {
                return Err(invalid());
            }
            end
        };
    if date_len == text.len() {
        return Err(error("generated_at_utc must include a UTC offset"));
    }
    let after = text.get(date_len..).ok_or_else(invalid)?;
    let separator = after.chars().next().ok_or_else(invalid)?;
    let time = &after[separator.len_utf8()..];
    let offset_index = time.find(['+', '-', 'Z']);
    let (time, offset) = if let Some(index) = offset_index {
        (&time[..index], Some(&time[index..]))
    } else {
        (time, None)
    };
    let parse_time = |clock: &str| -> bool {
        let mut pieces = clock.split(['.', ',']);
        let body = pieces.next().unwrap_or("");
        if let Some(frac) = pieces.next() {
            if frac.is_empty()
                || !frac.bytes().all(|b| b.is_ascii_digit())
                || pieces.next().is_some()
            {
                return false;
            }
        }
        let units: Vec<_> = if body.contains(':') {
            body.split(':').collect()
        } else {
            if ![2, 4, 6].contains(&body.len()) || !body.is_ascii() {
                return false;
            }
            (0..body.len())
                .step_by(2)
                .map(|i| &body[i..i + 2])
                .collect()
        };
        !units.is_empty()
            && units.len() <= 3
            && units.iter().enumerate().all(|(i, s)| {
                s.len() == 2 && digit(s).is_some_and(|n| n < if i == 0 { 24 } else { 60 })
            })
    };
    if !parse_time(time) {
        return Err(invalid());
    }
    let offset = offset.ok_or_else(|| error("generated_at_utc must include a UTC offset"))?;
    if offset == "Z" {
        return Ok(());
    }
    if !parse_time(&offset[1..]) {
        return Err(invalid());
    }
    if offset[1..].bytes().any(|b| b.is_ascii_digit() && b != b'0') {
        return Err(error("generated_at_utc must use UTC"));
    }
    Ok(())
}
pub fn write_runtime_side(config: &Value) -> Result<Value> {
    let role = side_bindings(config)?;
    let schedule = schedule_file(config)?;
    let artifact = &config["artifact_identity"];
    runtime::validate_artifact(artifact)?;
    let records = records(config, &schedule)?;
    let observation = observation(&config["plugin"]["name"], artifact, &schedule, records)?;
    if config["plugin"]["name"] != config["capabilities"]["provider_name"] {
        return Err(error(
            "plugin, capabilities, and observation provider names must agree",
        ));
    }
    if !config["capabilities"]["artifact_formats"]
        .as_array()
        .is_some_and(|a| a.contains(&artifact["artifact_format"]))
    {
        return Err(error(
            "artifact format is not declared by provider capabilities",
        ));
    }
    if config["execution_settings"]["allow_network"] == true {
        return Err(error(
            "strict runtime import requires offline execution settings",
        ));
    }
    let observation_bytes = canonical_json_no_newline(&observation)?;
    let receipt = json!({"format_version":"invarlock/runtime-provider-receipt-v1","plugin":config["plugin"],"backend":config["backend"],"capabilities":config["capabilities"],"artifact_identity":artifact,"execution_settings":config["execution_settings"],"device":config["device"],"outer_image_digest":config["runtime_image_digest"],"scoring_observation_sha256":hex(&observation_bytes)});
    runtime::validate_receipt(&receipt)?;
    let observed_pair = match (
        config.get("observed_observation"),
        config.get("observed_receipt"),
    ) {
        (None, None) => false,
        (Some(original_observation), Some(original_receipt)) => {
            runtime::validate_observation(original_observation)?;
            runtime::validate_receipt(original_receipt)?;
            let original_bytes = canonical_json_no_newline(original_observation)?;
            runtime::provider_bindings(
                artifact,
                original_observation,
                original_receipt,
                &original_bytes,
                Some(required_string(config, "runtime_image_digest")?),
            )?;
            if original_bytes != observation_bytes {
                return Err(error(
                    "observed scoring observation does not match the reconstructed runtime evidence",
                ));
            }
            if canonical_json_no_newline(original_receipt)? != canonical_json_no_newline(&receipt)?
            {
                return Err(error(
                    "observed provider receipt does not match the reconstructed runtime evidence",
                ));
            }
            true
        }
        _ => {
            return Err(error(
                "observed_observation and observed_receipt must be supplied together",
            ))
        }
    };
    let report_bytes = canonical(&report(&observation, &observation_bytes))?;
    let config_bytes = canonical(&side_config(role, &observation, &config["policy_digest"]))?;
    let artifact_bytes = canonical_json_no_newline(artifact)?;
    let receipt_bytes = canonical_json_no_newline(&receipt)?;
    let image = required_string(config, "runtime_image_digest")?;
    let reference = image_ref(required_string(config, "runtime_image_ref")?, image)?;
    let timestamp = required_string(config, "generated_at_utc")?;
    manifest_timestamp(timestamp)?;
    let manifest = json!({"manifest_version":1,"generated_at_utc":timestamp,"verifier_contract_version":"runtime-manifest-v1","report":{"path":"report.json","filename":"report.json","sha256":hex(&report_bytes)},"config":{"path":"run.yaml","sha256":hex(&config_bytes),"source":"file"},"execution_mode":"container","outer_container":{"image_ref":reference,"image_digest":image,"container_execution":true,"allow_network":false,"allow_remote_code":false,"allow_third_party_plugins":false},"runtime_provider":{"artifact_identity":{"filename":"model-artifact.identity.json","sha256":hex(&artifact_bytes)},"scoring_observation":{"filename":"runtime-scoring.observation.json","sha256":hex(&observation_bytes)},"receipt":{"filename":"runtime-provider.receipt.json","sha256":hex(&receipt_bytes)}}});
    let files = BTreeMap::from([
        ("report.json".into(), report_bytes),
        ("run.yaml".into(), config_bytes),
        ("model-artifact.identity.json".into(), artifact_bytes),
        ("runtime-scoring.observation.json".into(), observation_bytes),
        ("runtime-provider.receipt.json".into(), receipt_bytes),
        ("runtime.manifest.json".into(), pretty(&manifest)?),
    ]);
    validate_schema(&manifest, "runtime_manifest")?;
    runtime::provider_bindings(
        artifact,
        &observation,
        &receipt,
        &files["runtime-scoring.observation.json"],
        Some(image),
    )?;
    let lookup = |name: &str| -> Result<&[u8]> {
        SIDE_FILES
            .iter()
            .find(|(role, _)| *role == name)
            .and_then(|(_, file)| files.get(*file))
            .map(Vec::as_slice)
            .ok_or_else(|| error("runtime side file is missing"))
    };
    replay::runtime_manifest(
        role,
        &manifest,
        &receipt,
        &config["runtime_image_digest"],
        &lookup,
    )?;
    let destination = Path::new(required_string(config, "directory")?);
    new_destination(destination, "runtime import")?;
    stage_files(destination, &files, false)?;
    let mut reload = config.clone();
    reload["expected_runtime_image_digest"] = config["runtime_image_digest"].clone();
    let mut result = load_runtime_side(&reload)?;
    if observed_pair {
        let mut material = serde_json::Map::new();
        for name in [
            "model-artifact.identity.json",
            "runtime-scoring.observation.json",
            "runtime-provider.receipt.json",
        ] {
            material.insert(
                name.into(),
                json!(base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &files[name],
                )),
            );
        }
        result["material"] = Value::Object(material);
        result["receipt_sha256"] = json!(hex(&files["runtime-provider.receipt.json"]));
    }
    Ok(result)
}
pub fn load_runtime_side(config: &Value) -> Result<Value> {
    let role = side_bindings(config)?;
    let schedule = schedule_file(config)?;
    let directory = Path::new(required_string(config, "directory")?);
    let held = side_bytes(directory, role)?;
    let get = |name: &str| -> Result<&[u8]> {
        held.get(&format!("{role}_{name}"))
            .map(Vec::as_slice)
            .ok_or_else(|| error("runtime side file is missing"))
    };
    let artifact = replay::typed_sidecar(
        get("provider_identity")?,
        "model artifact identity",
        "model_artifact_identity",
    )?;
    let observation = replay::typed_sidecar(
        get("scoring_observation")?,
        "runtime scoring observation",
        "runtime_scoring_observation",
    )?;
    let receipt = replay::typed_sidecar(
        get("provider_receipt")?,
        "runtime provider receipt",
        "runtime_provider_receipt",
    )?;
    let runtime_digest = required_string(config, "expected_runtime_image_digest")?;
    runtime::provider_bindings(
        &artifact,
        &observation,
        &receipt,
        get("scoring_observation")?,
        Some(runtime_digest),
    )?;
    bind_records(&observation["records"], &schedule)?;
    if observation["schedule_sha256"] != bare(&schedule)? {
        return Err(error(
            "runtime import observation does not bind the supplied schedule",
        ));
    }
    let observed_report = object(get("run_report")?, "runtime import report")?;
    if canonical(&observed_report)? != get("run_report")? {
        return Err(error("runtime import report must use canonical JSON"));
    }
    if observed_report != report(&observation, get("scoring_observation")?) {
        return Err(error("runtime import report does not bind its observation"));
    }
    let observed_config = object(get("runtime_config")?, "runtime import config")?;
    if canonical(&observed_config)? != get("runtime_config")? {
        return Err(error("runtime import config must use canonical JSON"));
    }
    if observed_config != side_config(role, &observation, &config["policy_digest"]) {
        return Err(error(
            "runtime import config does not bind role, schedule, artifact, and policy",
        ));
    }
    let manifest = object(get("runtime_manifest")?, "runtime import manifest")?;
    replay::runtime_manifest(
        role,
        &manifest,
        &receipt,
        &config["expected_runtime_image_digest"],
        &get,
    )?;
    Ok(
        json!({"role":role,"directory":directory.canonicalize().map_err(io_error)?,"runtime_image_digest":runtime_digest,"artifact_identity_sha256":bare(&artifact)? ,"scoring_observation_sha256":hex(get("scoring_observation")?),"record_count":observation["records"].as_array().map(Vec::len).unwrap_or(0)}),
    )
}
pub fn write_paired(config: &Value, bridge: Option<&dyn ScorerReplay>) -> Result<Value> {
    let (request, schedule, _, identities, anchors, held) = publication_inputs(config)?;
    let loaded = held
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let paired = replay::derive(&loaded, &request, &schedule, &identities, &anchors, bridge)?;
    let output = Path::new(required_string(config, "path")?);
    let bytes = canonical(&paired)?;
    fs::write_file_no_replace(output, &bytes, 0o600, false)
        .map_err(|_| error("runtime import paired-record destination must be new and writable"))?;
    Ok(json!({"path":output,"payload":paired,"sha256":hex(&bytes)}))
}
