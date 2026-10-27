//! Complete captured evidence replay over one owned immutable input snapshot.
use crate::{
    bounded, canonical, crypto, digest, is_digest, required_string, EvidenceError, Result,
};
use ed25519_dalek::SigningKey;
use invarlock_native_contracts::{
    canonical_json_matches, parse_json, sha256_digest, validate_schema, SchemaCheckedValue,
};
use invarlock_native_core as analysis;
use invarlock_native_filesystem as fs;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const CONTROL: usize = 64 * 1024;
const DETECTOR: usize = 256 * 1024;
const PAYLOAD: usize = 128 * 1024 * 1024;
const TOTAL: usize = 384 * 1024 * 1024;
const REQUEST: usize = 1024 * 1024;
const RECEIPT: usize = 1024 * 1024;
const PACK_FORMAT: &str = "invarlock/evidence-pack-v2";
const RECEIPT_FORMAT: &str = "invarlock/evidence-verification-receipt-v3";
const PAYLOADS: [(&str, &str); 5] = [
    ("request", "request.json"),
    ("policy", "inputs/policy.json"),
    ("baseline", "records/baseline.json"),
    ("subject", "records/subject.json"),
    ("report", "reports/evaluation.report.json"),
];

fn io_error(error: impl std::fmt::Display) -> EvidenceError {
    EvidenceError::incomplete(format!(
        "verification_io_error: {}",
        bounded(&error.to_string())
    ))
}
pub fn object(raw: &[u8], label: &str, require_canonical: bool) -> Result<Value> {
    let value = parse_json(raw, label)
        .map_err(|_| EvidenceError::contract(format!("{label} is not strict JSON")))?;
    if !value.is_object() {
        return Err(EvidenceError::contract(format!(
            "{label} must be a JSON object"
        )));
    }
    if require_canonical && !canonical_json_matches(&value, raw)? {
        return Err(EvidenceError::contract(format!(
            "{label} must be canonical JSON"
        )));
    }
    Ok(value)
}
fn validate_manifest(value: &Value) -> Result<()> {
    let keys: BTreeSet<_> = value["files"]
        .as_object()
        .map(|v| v.keys().map(String::as_str).collect())
        .unwrap_or_default();
    if keys != PAYLOADS.iter().map(|(r, _)| *r).collect() {
        return Err(EvidenceError::contract(
            "captured manifest role inventory is invalid",
        ));
    }
    validate_schema(value, "evidence_pack_v2").map_err(|e| {
        EvidenceError::contract(format!(
            "captured contract is invalid: {}",
            bounded(&e.to_string())
        ))
    })
}
fn detect(raw: &[u8]) -> Result<Value> {
    if raw.len() > DETECTOR {
        return Err(EvidenceError::contract(
            "manifest exceeds detector byte limit",
        ));
    }
    let value = object(raw, "evidence manifest", false)?;
    if value["format"] != PACK_FORMAT || value["kind"] != "captured" {
        return Err(EvidenceError::contract(
            "captured manifest format or kind is invalid",
        ));
    }
    if raw.len() > CONTROL {
        return Err(EvidenceError::contract(
            "captured manifest exceeds control byte limit",
        ));
    }
    Ok(value)
}
pub fn comparison_id(request: &str, baseline: &str, subject: &str, policy: &str) -> Result<String> {
    digest(
        &json!({"kind":"captured","request_digest":request,"baseline_run_digest":baseline,
        "subject_run_digest":subject,"policy_digest":policy}),
    )
}
fn authenticate(manifest: &Value, raw: &[u8], signature_raw: &[u8]) -> Result<String> {
    let signature = object(signature_raw, "manifest signature", true)?;
    let expected: BTreeSet<&str> = [
        "format",
        "algorithm",
        "signing_key_fingerprint",
        "public_key",
        "signature",
    ]
    .into_iter()
    .collect();
    let fields: BTreeSet<&str> = signature
        .as_object()
        .expect("object checked")
        .keys()
        .map(String::as_str)
        .collect();
    if fields != expected {
        return Err(EvidenceError::contract(format!(
            "captured signature fields are invalid: {}",
            bounded(&fields.into_iter().collect::<Vec<_>>().join(", "))
        )));
    }
    if signature["format"] != "invarlock/evidence-pack-signature-v1"
        || signature["algorithm"] != "ed25519"
    {
        return Err(EvidenceError::contract(
            "captured signature format or algorithm is invalid",
        ));
    }
    for (field, encoding) in [("public_key", "pem"), ("signature", "base64")] {
        if signature[field].as_object().map(|v| v.len()) != Some(2)
            || signature[field]["encoding"] != encoding
            || !signature[field]["value"].is_string()
        {
            return Err(EvidenceError::contract(
                "captured signature encoding is invalid",
            ));
        }
    }
    let result = (|| {
        let key = crypto::public_key(
            signature["public_key"]["value"]
                .as_str()
                .expect("string checked"),
        )?;
        crypto::verify(
            &key,
            raw,
            signature["signature"]["value"]
                .as_str()
                .expect("string checked"),
        )?;
        Ok::<_, EvidenceError>(crypto::fingerprint(&key))
    })();
    let fingerprint =
        result.map_err(|_| EvidenceError::integrity("captured manifest signature is invalid"))?;
    if signature["signing_key_fingerprint"] != fingerprint
        || manifest["signing_key_fingerprint"] != fingerprint
    {
        return Err(EvidenceError::integrity(
            "captured manifest signer binding is invalid",
        ));
    }
    Ok(fingerprint)
}
fn validate_request_strings(value: &Value, path: &mut Vec<String>) -> Result<()> {
    if path.len() == 4
        && path[0] == "comparison"
        && (path[1] == "baseline" || path[1] == "subject")
        && path[2] == "service_identity"
        && path[3] == "configuration"
    {
        return Ok(());
    }
    match value {
        Value::String(text) if text.chars().any(|c| c < ' ' || c == '\u{7f}') => {
            return Err(EvidenceError::contract(
                "captured request strings cannot contain control characters",
            ));
        }
        Value::Object(map) => {
            for (key, child) in map {
                validate_request_strings(&Value::String(key.clone()), path)?;
                path.push(key.clone());
                validate_request_strings(child, path)?;
                path.pop();
            }
        }
        Value::Array(values) => {
            for child in values {
                validate_request_strings(child, path)?;
            }
        }
        _ => {}
    }
    Ok(())
}
pub fn validate_normalized_request(
    request: &Value,
    baseline: &Value,
    subject: &Value,
    policy: &Value,
) -> Result<()> {
    if canonical(request)?.len() > REQUEST {
        return Err(EvidenceError::contract(
            "captured request exceeds the 1 MiB byte limit",
        ));
    }
    validate_request_strings(request, &mut vec![])?;
    validate_schema(request, "normalized_captured_request")?;
    let comparison = &request["comparison"];
    if comparison["policy_digest"] != digest(policy)? {
        return Err(EvidenceError::integrity(
            "captured request contradicts its bound inputs",
        ));
    }
    if let Some(selected) = comparison.get("metric") {
        let metrics = policy["metrics"]
            .as_array()
            .ok_or_else(|| EvidenceError::contract("policy metrics are invalid"))?;
        if metrics.len() != 1 || metrics[0]["kind"] != *selected {
            return Err(EvidenceError::integrity(
                "selected metric must match the single policy metric",
            ));
        }
    }
    for (side, run) in [("baseline", baseline), ("subject", subject)] {
        let source = &comparison[side];
        let actual = digest(run)?;
        if source["run_digest"] != actual {
            return Err(EvidenceError::integrity(
                "captured request contradicts its bound inputs",
            ));
        }
        if source
            .get("expected_run_digest")
            .is_some_and(|expected| *expected != source["run_digest"])
        {
            return Err(EvidenceError::contract(format!(
                "{side} normalized run pin is inconsistent"
            )));
        }
        let identities = [
            "source",
            "run_id",
            "artifact_digest",
            "score_provenance",
            "service_identity",
        ];
        if source["adapter"] == "invarlock" {
            if identities
                .iter()
                .chain(std::iter::once(&"input_projection"))
                .any(|key| source.get(*key).is_some())
            {
                return Err(EvidenceError::integrity(
                    "canonical run identities cannot be overridden at import",
                ));
            }
        } else {
            if identities[..3].iter().any(|key| source.get(*key).is_none())
                || run["source_digest"].is_null()
            {
                return Err(EvidenceError::integrity("native import requires source, run and artifact identities and physical source digest"));
            }
            for key in identities {
                let empty = json!({});
                let default = if key == "score_provenance" {
                    &empty
                } else {
                    &Value::Null
                };
                if source.get(key).unwrap_or(default) != run.get(key).unwrap_or(&Value::Null) {
                    return Err(EvidenceError::integrity(format!(
                        "{side} {key} differs from declared source intent"
                    )));
                }
            }
            if let Some(projection) = source.get("input_projection") {
                for row in run["records"]
                    .as_array()
                    .ok_or_else(|| EvidenceError::contract("records are invalid"))?
                {
                    if row["context"]["input_projection"]["configuration"] != *projection {
                        return Err(EvidenceError::integrity(format!(
                            "{side} input projection differs from declared intent"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}
fn checksums(files: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (name, raw) in files {
        bytes.extend_from_slice(
            format!(
                "{}  {name}\n",
                sha256_digest(raw).trim_start_matches("sha256:")
            )
            .as_bytes(),
        );
    }
    bytes
}
/// Authenticated payload values cannot be changed while retaining their proofs.
/// ```compile_fail
/// use invarlock_native_evidence::captured::CapturedInputs;
/// use serde_json::Value;
/// fn mutate(loaded: &mut CapturedInputs) {
///     loaded.payloads["baseline"]["records"] = Value::Null;
/// }
/// ```
pub struct CapturedPayloads {
    values: BTreeMap<String, SchemaCheckedValue>,
}
impl CapturedPayloads {
    pub fn get(&self, role: &str) -> Option<&Value> {
        self.values.get(role).map(SchemaCheckedValue::value)
    }
    pub fn schema_checked(&self, role: &str) -> Option<&SchemaCheckedValue> {
        self.values.get(role)
    }
    pub fn into_values(self) -> Map<String, Value> {
        self.values
            .into_iter()
            .map(|(role, checked)| (role, checked.into_value()))
            .collect()
    }
}
impl std::ops::Index<&str> for CapturedPayloads {
    type Output = Value;
    fn index(&self, role: &str) -> &Value {
        self.values[role].value()
    }
}
pub struct CapturedInputs {
    pub manifest: Value,
    pub payloads: CapturedPayloads,
    pub signer: String,
}
enum CapturedEntry<'a> {
    Borrowed(&'a fs::SnapshotEntry),
    Owned(fs::SnapshotEntry),
}
impl CapturedEntry<'_> {
    fn entry(&self) -> &fs::SnapshotEntry {
        match self {
            Self::Borrowed(entry) => entry,
            Self::Owned(entry) => entry,
        }
    }
}
fn required_entry<'a>(
    entries: &'a BTreeMap<String, CapturedEntry<'_>>,
    name: &str,
) -> Result<&'a fs::SnapshotEntry> {
    entries
        .get(name)
        .map(CapturedEntry::entry)
        .ok_or_else(|| EvidenceError::contract("captured evidence file inventory is incomplete"))
}
pub fn load_inputs(entries: &BTreeMap<String, fs::SnapshotEntry>) -> Result<CapturedInputs> {
    load_input_entries(
        entries
            .iter()
            .map(|(name, entry)| (name.clone(), CapturedEntry::Borrowed(entry)))
            .collect(),
    )
}
pub(crate) fn load_owned_inputs(
    entries: BTreeMap<String, fs::SnapshotEntry>,
) -> Result<CapturedInputs> {
    load_input_entries(
        entries
            .into_iter()
            .map(|(name, entry)| (name, CapturedEntry::Owned(entry)))
            .collect(),
    )
}
fn load_input_entries(mut entries: BTreeMap<String, CapturedEntry<'_>>) -> Result<CapturedInputs> {
    let manifest_raw = required_entry(&entries, "manifest.json")?.bytes();
    let manifest = object(manifest_raw, "captured manifest", true)?;
    validate_manifest(&manifest)?;
    let signer = if manifest["authentication"] == "signed" {
        authenticate(
            &manifest,
            manifest_raw,
            required_entry(&entries, "manifest.signature.json")?.bytes(),
        )?
    } else {
        "Unsigned local evidence".to_string()
    };
    let mut payloads = Map::new();
    let mut payload_hashes = BTreeMap::new();
    let mut canonical_checks = BTreeMap::new();
    for (role, name) in PAYLOADS {
        let entry = entries.remove(name).ok_or_else(|| {
            EvidenceError::contract("captured evidence file inventory is incomplete")
        })?;
        let raw = entry.entry().bytes();
        let hash = entry.entry().sha256();
        if manifest["files"][role]["digest"] != format!("sha256:{hash}") {
            return Err(EvidenceError::integrity(format!(
                "captured {role} binding is invalid"
            )));
        }
        let value = object(raw, role, false)?;
        // Retain the result, not the bytes. Syntax and record-limit errors from
        // any payload still precede canonical/schema errors at the phase below.
        canonical_checks.insert(
            role,
            canonical_json_matches(&value, raw).map_err(EvidenceError::from),
        );
        payload_hashes.insert(name, hash.to_owned());
        payloads.insert(role.into(), value);
        drop(entry);
    }
    for role in ["baseline", "subject"] {
        if payloads[role]["records"]
            .as_array()
            .is_some_and(|rows| rows.len() > 50_000)
        {
            return Err(EvidenceError::contract(
                "run exceeds the 50,000-record limit",
            ));
        }
    }
    let mut checked = BTreeMap::new();
    for (role, _) in PAYLOADS {
        if !canonical_checks
            .remove(role)
            .expect("checked parsed payload")?
        {
            return Err(EvidenceError::contract(format!(
                "{role} must be canonical JSON"
            )));
        }
        if role != "request" {
            let value = payloads.remove(role).expect("parsed payload");
            let value = SchemaCheckedValue::new(
                value,
                match role {
                    "baseline" | "subject" => "run",
                    "report" => "comparison",
                    _ => "policy",
                },
            )
            .map_err(|e| EvidenceError::contract(bounded(&e.to_string())))?;
            checked.insert(role.to_owned(), value);
        }
    }
    let ledger_entry = required_entry(&entries, "checksums.sha256")?;
    let ledger = ledger_entry.bytes();
    if manifest["checksums_sha256_digest"] != ledger_entry.sha256().trim_start_matches("sha256:") {
        return Err(EvidenceError::integrity(
            "captured checksum ledger binding is invalid",
        ));
    }
    let sorted: BTreeSet<_> = PAYLOADS.iter().map(|(_, name)| *name).collect();
    let mut expected_ledger = String::new();
    for name in sorted {
        expected_ledger.push_str(&format!(
            "{}  {name}\n",
            payload_hashes[name].trim_start_matches("sha256:")
        ));
    }
    if ledger != expected_ledger.as_bytes() {
        return Err(EvidenceError::integrity(
            "captured checksum ledger inventory is invalid",
        ));
    }
    let refs = &manifest["files"];
    if manifest["request_digest"] != refs["request"]["digest"]
        || manifest["comparison_id"]
            != comparison_id(
                manifest["request_digest"].as_str().unwrap_or(""),
                refs["baseline"]["digest"].as_str().unwrap_or(""),
                refs["subject"]["digest"].as_str().unwrap_or(""),
                refs["policy"]["digest"].as_str().unwrap_or(""),
            )?
    {
        return Err(EvidenceError::integrity(
            "captured request or comparison identity is invalid",
        ));
    }
    validate_normalized_request(
        &payloads["request"],
        checked["baseline"].value(),
        checked["subject"].value(),
        checked["policy"].value(),
    )?;
    let request = SchemaCheckedValue::new(
        payloads.remove("request").expect("parsed request"),
        "normalized_captured_request",
    )?;
    checked.insert("request".into(), request);
    let payloads = CapturedPayloads { values: checked };
    let bindings = json!({"baseline":format!("sha256:{}",payload_hashes["records/baseline.json"]),
        "subject":format!("sha256:{}",payload_hashes["records/subject.json"]),"policy":format!("sha256:{}",payload_hashes["inputs/policy.json"])});
    if payloads["report"]["bindings"] != bindings {
        return Err(EvidenceError::integrity(
            "captured comparison contradicts its bound inputs",
        ));
    }
    Ok(CapturedInputs {
        manifest,
        payloads,
        signer,
    })
}

struct VerificationInputs {
    evidence: PathBuf,
    receipt: PathBuf,
    policy: SchemaCheckedValue,
    anchors: Value,
    identity: String,
    profile: Value,
    key: SigningKey,
    budget: Option<u64>,
}
fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir().map_err(io_error)?.join(path))
    }
}
pub(crate) fn require_outside(path: &Path, pack: &Path) -> Result<()> {
    let path = absolute(path)?;
    let pack = absolute(pack)?;
    let resolve = |path: &Path| -> Result<PathBuf> {
        let mut ancestor = path;
        let mut suffix = Vec::new();
        loop {
            match ancestor.canonicalize() {
                Ok(mut resolved) => {
                    for name in suffix.iter().rev() {
                        resolved.push(name);
                    }
                    let mut normalized = PathBuf::new();
                    for component in resolved.components() {
                        match component {
                            std::path::Component::ParentDir => {
                                normalized.pop();
                            }
                            std::path::Component::CurDir => {}
                            other => normalized.push(other.as_os_str()),
                        }
                    }
                    return Ok(normalized);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    suffix.push(
                        ancestor
                            .file_name()
                            .ok_or_else(|| io_error(&e))?
                            .to_os_string(),
                    );
                    ancestor = ancestor.parent().ok_or_else(|| io_error(e))?;
                }
                Err(e) => return Err(io_error(e)),
            }
        }
    };
    let pack_resolved = resolve(&pack)?;
    let resolved = resolve(&path)?;
    if path.starts_with(&pack) || resolved.starts_with(pack_resolved) {
        return Err(EvidenceError::contract(
            "destination must be outside the evidence pack",
        ));
    }
    Ok(())
}
fn inputs(
    config: &Value,
    policy_bytes: Option<&[u8]>,
    key_bytes: Option<&[u8]>,
) -> Result<VerificationInputs> {
    let anchor_names = [
        "expected_baseline_run",
        "expected_subject_run",
        "expected_request_digest",
        "expected_signer",
    ];
    if anchor_names
        .iter()
        .any(|key| !config[*key].as_str().is_some_and(is_digest))
    {
        return Err(EvidenceError::incomplete(
            "captured run, request, and signer anchors are required",
        ));
    }
    let identity = config["verifier_identity"].as_str().unwrap_or("");
    if identity.is_empty()
        || identity.len() > 256
        || !identity.as_bytes()[0].is_ascii_alphanumeric()
        || !identity
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:/-".contains(&b))
    {
        return Err(EvidenceError::incomplete(
            "valid verifier identity is required",
        ));
    }
    let profile = config.get("trust_profile_digest").unwrap_or(&Value::Null);
    if !profile.is_null() && !profile.as_str().is_some_and(is_digest) {
        return Err(EvidenceError::incomplete("trust profile digest is invalid"));
    }
    if !config["receipt_path"].is_string()
        || (!config["verifier_signing_key_path"].is_string() && key_bytes.is_none())
    {
        return Err(EvidenceError::incomplete(
            "receipt and verifier signing key are required",
        ));
    }
    let budget = invarlock_native_contracts::bootstrap_budget(
        config.get("max_bootstrap_draws"),
        102_400_000,
    )
    .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
    let evidence = PathBuf::from(required_string(config, "evidence_path")?);
    let receipt = PathBuf::from(required_string(config, "receipt_path")?);
    let prepared = (|| {
        require_outside(&receipt, &evidence)?;
        let disk;
        let raw = match policy_bytes {
            Some(raw) => raw,
            None => {
                disk = fs::read_file(Path::new(required_string(config, "policy_path")?), PAYLOAD)
                    .map_err(io_error)?;
                &disk
            }
        };
        if raw.len() > PAYLOAD {
            return Err(EvidenceError::contract(
                "independent policy exceeds byte limit",
            ));
        }
        let policy = object(raw, "independent policy", false)?;
        let policy = SchemaCheckedValue::new(policy, "policy")?;
        analysis::ValidatedPolicy::from_schema_checked(&policy)
            .map_err(|e| EvidenceError::contract(e.to_string()))?;
        let policy_digest = policy.canonical_digest()?.to_owned();
        let key_disk;
        let key_raw = match key_bytes {
            Some(raw) => raw,
            None => {
                key_disk = fs::read_file(
                    Path::new(required_string(config, "verifier_signing_key_path")?),
                    CONTROL,
                )
                .map_err(io_error)?;
                &key_disk
            }
        };
        if key_raw.len() > CONTROL {
            return Err(EvidenceError::incomplete(
                "verifier key could not be loaded",
            ));
        }
        let key = crypto::private_key(key_raw)?;
        Ok::<_, EvidenceError>((policy, policy_digest, key))
    })()
    .map_err(|e| {
        EvidenceError::incomplete(format!(
            "invalid verification inputs: {}",
            bounded(&e.to_string())
        ))
    })?;
    let (policy, policy_digest, key) = prepared;
    let anchors = json!({"baseline_run_digest":config[anchor_names[0]],"subject_run_digest":config[anchor_names[1]],
        "request_digest":config[anchor_names[2]],"signer_fingerprint":config[anchor_names[3]],"policy_digest":policy_digest});
    Ok(VerificationInputs {
        evidence,
        receipt,
        policy,
        anchors,
        identity: identity.into(),
        profile: profile.clone(),
        key,
        budget,
    })
}
fn assurance(policy: &Value) -> Value {
    let mut values = vec![];
    let mut scopes = vec!["overall"];
    if let Some(slices) = policy["slices"].as_array() {
        for slice in slices {
            scopes.push(slice["name"].as_str().unwrap_or(""));
        }
    }
    if let Some(metrics) = policy["metrics"].as_array() {
        for scope in scopes {
            for metric in metrics {
                values.push(json!({"name":metric["name"],"slice":scope,"kind":metric["kind"],
                "scoring_assurance":if metric["kind"]=="recorded" {"recorded"} else {"recomputed"}}));
            }
        }
    }
    Value::Array(values)
}
fn write_receipt(
    input: &VerificationInputs,
    manifest_raw: &[u8],
    replay: &str,
    completed: Option<&Value>,
    failure: Option<&EvidenceError>,
) -> Result<()> {
    let decision = completed
        .map(|c| c["decision"].clone())
        .unwrap_or(Value::Null);
    let good = failure.is_none();
    let ok = good && decision == "pass";
    let status = failure
        .map(|e| e.exit_code)
        .unwrap_or(if ok { 0 } else { 7 });
    let statement = json!({"format":RECEIPT_FORMAT,"kind":"captured","pack_format":PACK_FORMAT,
        "pack_manifest_digest":sha256_digest(manifest_raw),"anchors":input.anchors,
        "subject":{"kind":"captured_run","run_digest":input.anchors["subject_run_digest"]},
        "verifier":{"identity":input.identity,"signing_key_fingerprint":crypto::fingerprint(&input.key.verifying_key()),
            "trust_profile_digest":input.profile},"verification_scope":"captured_comparison","replay_status":replay,
        "scoring_assurance":if good {assurance(input.policy.value())} else {Value::Null},
        "verdict":{"ok":ok,"integrity_ok":good,"decision":decision,
            "policy_verdict":if good {json!(if ok {"pass"} else {"fail"})} else {Value::Null},"verification_status":status}});
    let receipt = crypto::signed_receipt(&input.key, statement)?;
    validate_schema(&receipt, "evidence_verification_receipt_v3")?;
    let bytes = canonical(&receipt)?;
    if bytes.len() > RECEIPT {
        return Err(EvidenceError::incomplete("receipt exceeds byte limit"));
    }
    fs::write_file_no_replace(&input.receipt, &bytes, 0o600, true)
        .map_err(|_| EvidenceError::incomplete("receipt_publication_failed"))
}

pub fn verify(config: &Value) -> Result<Value> {
    verify_with_inputs(config, None, None)
}
/// Caller-owned immutable trust bytes; paths remain subject to normal policy.
pub fn verify_with_inputs(
    config: &Value,
    policy_bytes: Option<&[u8]>,
    key_bytes: Option<&[u8]>,
) -> Result<Value> {
    let input = inputs(config, policy_bytes, key_bytes)?;
    let root = fs::PinnedDirectory::open(&input.evidence, false).map_err(io_error)?;
    let (manifest_raw, manifest_identity) = root
        .read_regular(Path::new("manifest.json"), DETECTOR)
        .map_err(io_error)?;
    detect(&manifest_raw)?;
    let mut replay = "not_started";
    let result = (|| {
        let manifest = object(&manifest_raw, "captured manifest", true)?;
        validate_manifest(&manifest)?;
        let mut names: BTreeMap<String, usize> = PAYLOADS
            .iter()
            .map(|(_, name)| {
                (
                    name.to_string(),
                    if *name == "request.json" {
                        REQUEST
                    } else {
                        PAYLOAD
                    },
                )
            })
            .collect();
        names.insert("manifest.json".into(), CONTROL);
        names.insert("checksums.sha256".into(), CONTROL);
        if manifest["authentication"] == "signed" {
            names.insert("manifest.signature.json".into(), CONTROL);
        }
        let directories: BTreeSet<String> = ["inputs", "records", "reports"]
            .into_iter()
            .map(String::from)
            .collect();
        let limits = fs::SnapshotLimits {
            max_file_bytes: PAYLOAD,
            max_total_bytes: TOTAL,
            ..Default::default()
        };
        let snapshot = root
            .capture_selected(limits, &names, &directories)
            .map_err(snapshot_error)?;
        if snapshot
            .entries()
            .get("manifest.json")
            .is_none_or(|entry| entry.bytes() != manifest_raw)
        {
            return Err(EvidenceError::integrity("manifest changed after detection"));
        }
        let policy_matches = snapshot
            .entries()
            .get("inputs/policy.json")
            .map(|entry| canonical_json_matches(input.policy.value(), entry.bytes()));
        let stored_report = snapshot
            .entries()
            .get("reports/evaluation.report.json")
            .map(|entry| entry.bytes().to_vec());
        let (entries, source_guard) = snapshot.into_parts();
        let loaded = load_owned_inputs(entries)?;
        if loaded.manifest["authentication"] != "signed" {
            return Err(EvidenceError::integrity(
                "captured verification requires signed evidence",
            ));
        }
        if loaded.signer != input.anchors["signer_fingerprint"] {
            return Err(EvidenceError::integrity(
                "captured evidence signer does not match expected anchor",
            ));
        }
        for (role, key) in [
            ("baseline", "baseline_run_digest"),
            ("subject", "subject_run_digest"),
            ("request", "request_digest"),
            ("policy", "policy_digest"),
        ] {
            if loaded.manifest["files"][role]["digest"] != input.anchors[key] {
                return Err(EvidenceError::integrity(format!(
                    "captured {role} differs from expected anchor"
                )));
            }
        }
        if !policy_matches.expect("loaded policy entry")? {
            return Err(EvidenceError::integrity(
                "captured policy differs from independent policy",
            ));
        }
        let stored_report = stored_report.expect("loaded report entry");
        replay = "failed";
        let comparison = (|| {
            let baseline = analysis::ValidatedRun::from_schema_checked(
                loaded.payloads.schema_checked("baseline").unwrap(),
            )?;
            let subject = analysis::ValidatedRun::from_schema_checked(
                loaded.payloads.schema_checked("subject").unwrap(),
            )?;
            let policy = analysis::ValidatedPolicy::from_schema_checked(&input.policy)?;
            analysis::compare_validated_runs(&baseline, &subject, &policy, input.budget)
        })()
        .map_err(|e| {
            if e.reason == "local_work_budget_exceeded"
                || e.reason == "unsupported_scoring_environment"
            {
                EvidenceError::incomplete(e.reason)
            } else {
                EvidenceError::integrity(format!(
                    "captured replay failed: {}",
                    bounded(&e.to_string())
                ))
            }
        })?;
        replay = "completed";
        if !canonical_json_matches(&comparison, &stored_report)? {
            return Err(EvidenceError::integrity(
                "captured replay disagrees with stored comparison",
            ));
        }
        source_guard
            .verify_source()
            .map_err(|_| EvidenceError::integrity("captured inventory changed after capture"))?;
        Ok(comparison)
    })();
    // A source change prevents attributing even a rejected receipt to old bytes.
    root.verify().map_err(io_error)?;
    root.verify_regular(Path::new("manifest.json"), &manifest_identity)
        .map_err(|_| EvidenceError::integrity("manifest changed after detection"))?;
    match result {
        Err(mut failure) => {
            if failure.exit_code != 2 {
                write_receipt(&input, &manifest_raw, replay, None, Some(&failure))
                    .map_err(|_| EvidenceError::incomplete("receipt_publication_failed"))?;
                failure.receipt_written = true;
            }
            Err(failure)
        }
        Ok(comparison) => {
            write_receipt(&input, &manifest_raw, replay, Some(&comparison), None)
                .map_err(|_| EvidenceError::incomplete("receipt_publication_failed"))?;
            let summaries:Vec<_>=comparison["metrics"].as_array().ok_or_else(||EvidenceError::incomplete("verification_internal_error"))?.iter().map(|metric|{
                let missing=metric["missing_ids"].as_array().map_or(0,Vec::len);
                json!({"name":metric["name"],"slice":metric["slice"],"decision":metric["decision"],
                    "usable_count":metric["count"].as_u64().unwrap_or(0).saturating_sub(missing as u64),
                    "missing_count":missing,"reasons":metric["reasons"]})
            }).collect();
            let pass = comparison["decision"] == "pass";
            Ok(
                json!({"format_version":"invarlock/evidence-pack-verify-v2","kind":"captured","ok":pass,"integrity_ok":true,
                "policy_verdict":if pass {"pass"}else{"fail"},"decision":comparison["decision"],"replay_status":"completed",
                "signed_receipt":input.receipt.file_name().and_then(|v|v.to_str()),"pack_manifest_digest":sha256_digest(&manifest_raw),"metric_summaries":summaries}),
            )
        }
    }
}

/// Authenticate a receipt against caller-selected anchors, separately from its verdict.
pub fn verify_receipt(config: &Value) -> Result<Value> {
    let checked = (|| {
        let receipt_raw =
            fs::read_file(Path::new(required_string(config, "receipt_path")?), RECEIPT)
                .map_err(io_error)?;
        let receipt = object(&receipt_raw, "verification receipt", false)?;
        let policy_raw = fs::read_file(Path::new(required_string(config, "policy_path")?), PAYLOAD)
            .map_err(io_error)?;
        let policy = object(&policy_raw, "independent policy", false)?;
        analysis::validate_policy(&policy).map_err(|e| EvidenceError::contract(e.to_string()))?;
        let policy_digest = digest(&policy)?;
        validate_schema(&receipt, "evidence_verification_receipt_v3")?;
        let statement = &receipt["statement"];
        if statement["subject"]["run_digest"] != statement["anchors"]["subject_run_digest"] {
            return Err(EvidenceError::contract(
                "captured receipt subject differs from expected anchor",
            ));
        }
        if statement["verdict"]["integrity_ok"] == true
            && statement["scoring_assurance"] != assurance(&policy)
        {
            return Err(EvidenceError::contract(
                "captured receipt assurance is incomplete or contradicts policy",
            ));
        }
        let evidence = Path::new(required_string(config, "evidence_path")?);
        let manifest_raw =
            fs::read_file(&evidence.join("manifest.json"), DETECTOR).map_err(io_error)?;
        let manifest = detect(&manifest_raw)?;
        if statement["pack_manifest_digest"] != sha256_digest(&manifest_raw) {
            return Err(EvidenceError::integrity(
                "receipt does not bind the examined manifest",
            ));
        }
        let anchors = json!({"baseline_run_digest":config["expected_baseline_run"],"subject_run_digest":config["expected_subject_run"],
            "policy_digest":policy_digest,"request_digest":config["expected_request_digest"],"signer_fingerprint":config["expected_signer"]});
        if statement["anchors"] != anchors {
            return Err(EvidenceError::integrity(
                "captured receipt anchors do not match caller expectations",
            ));
        }
        let verifier = &statement["verifier"];
        if verifier["identity"] != config["expected_verifier_identity"]
            || verifier["trust_profile_digest"] != config["expected_profile_digest"]
        {
            return Err(EvidenceError::integrity(
                "captured receipt verifier does not match caller expectations",
            ));
        }
        let public = crypto::public_key(
            receipt["signature"]["public_key"]["value"]
                .as_str()
                .unwrap_or(""),
        )
        .map_err(|_| EvidenceError::integrity("captured receipt verifier key must be Ed25519"))?;
        let fingerprint = crypto::fingerprint(&public);
        if config["expected_verifier_fingerprint"] != fingerprint
            || verifier["signing_key_fingerprint"] != fingerprint
        {
            return Err(EvidenceError::integrity(
                "captured receipt verifier key is unauthorized",
            ));
        }
        crypto::verify(
            &public,
            &canonical(statement)?,
            receipt["signature"]["value"].as_str().unwrap_or(""),
        )
        .map_err(|_| EvidenceError::integrity("captured receipt signature is invalid"))?;
        if statement["verdict"]["integrity_ok"] == true {
            validate_manifest(&manifest)?;
            if !canonical_json_matches(&manifest, &manifest_raw)?
                || manifest["authentication"] != "signed"
                || manifest["signing_key_fingerprint"] != config["expected_signer"]
            {
                return Err(EvidenceError::integrity(
                    "completed captured receipt contradicts manifest signer",
                ));
            }
            for (role, anchor) in [
                ("baseline", "baseline_run_digest"),
                ("subject", "subject_run_digest"),
                ("policy", "policy_digest"),
                ("request", "request_digest"),
            ] {
                if manifest["files"][role]["digest"] != anchors[anchor] {
                    return Err(EvidenceError::integrity(
                        "completed captured receipt contradicts manifest references",
                    ));
                }
            }
            let id = comparison_id(
                anchors["request_digest"].as_str().unwrap_or(""),
                anchors["baseline_run_digest"].as_str().unwrap_or(""),
                anchors["subject_run_digest"].as_str().unwrap_or(""),
                &policy_digest,
            )?;
            if manifest["request_digest"] != config["expected_request_digest"]
                || manifest["comparison_id"] != id
            {
                return Err(EvidenceError::integrity(
                    "completed captured receipt contradicts manifest identity",
                ));
            }
        }
        Ok((statement.clone(), fingerprint))
    })();
    match checked {
        Ok((statement, fingerprint)) => Ok(
            json!({"ok":true,"signed":true,"statement":statement,"verifier_fingerprint":fingerprint,"errors":[]}),
        ),
        Err(error) => Ok(
            json!({"ok":false,"signed":true,"statement":null,"verifier_fingerprint":null,"errors":[bounded(&error.to_string())]}),
        ),
    }
}

fn snapshot_error(error: fs::Error) -> EvidenceError {
    match error {
        fs::Error::Changed(_) => EvidenceError::integrity("captured inventory changed during scan"),
        fs::Error::Io(_) => EvidenceError::incomplete("verification_io_error"),
        fs::Error::UnsafePath(message) if message.starts_with("unexpected evidence entry:") => {
            EvidenceError::contract("captured evidence inventory has an unknown entry")
        }
        fs::Error::UnsafePath(message)
            if message == "evidence inventory does not match required entries" =>
        {
            EvidenceError::contract("captured evidence file inventory is incomplete")
        }
        fs::Error::UnsafePath(_) => {
            EvidenceError::contract("captured inventory contains an unsafe entry")
        }
        fs::Error::Limit(message) => EvidenceError::contract(message),
    }
}

pub fn normalize_request(
    authored: &Value,
    baseline: &Value,
    subject: &Value,
    policy: &Value,
) -> Result<Value> {
    validate_schema(authored, "evaluation_request_v2")?;
    let mut comparison = Map::new();
    comparison.insert("policy_digest".into(), json!(digest(policy)?));
    if let Some(metric) = authored["comparison"].get("metric") {
        comparison.insert("metric".into(), metric.clone());
    }
    for (side, run) in [("baseline", baseline), ("subject", subject)] {
        let mut source = authored["comparison"][side]
            .as_object()
            .ok_or_else(|| EvidenceError::contract("captured side must be an object"))?
            .clone();
        source.remove("path");
        source.insert("run_digest".into(), json!(digest(run)?));
        comparison.insert(side.into(), Value::Object(source));
    }
    let normalized = json!({"format":authored["format_version"],"execution":{"mode":"captured"},"comparison":comparison});
    validate_normalized_request(&normalized, baseline, subject, policy)?;
    Ok(normalized)
}

/// Evaluate and publish one captured transaction. All portable computation is native.
pub fn publish(config: &Value) -> Result<Value> {
    let baseline = &config["baseline"];
    let subject = &config["subject"];
    let policy = &config["policy"];
    let maximum = invarlock_native_contracts::bootstrap_budget(
        config.get("max_bootstrap_draws"),
        102_400_000,
    )
    .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
    let comparison = analysis::compare_runs(baseline, subject, policy, maximum).map_err(|e| {
        EvidenceError::new(
            if e.reason == "local_work_budget_exceeded" {
                2
            } else {
                4
            },
            format!("captured comparison failed: {e}"),
        )
    })?;
    let request = if let Some(value) = config.get("normalized_request") {
        validate_normalized_request(value, baseline, subject, policy)?;
        value.clone()
    } else {
        normalize_request(&config["request"], baseline, subject, policy)?
    };
    let unsigned = config["unsigned"].as_bool().unwrap_or(false);
    let key_path = config.get("signing_key_path").and_then(Value::as_str);
    if unsigned && key_path.is_some() {
        return Err(EvidenceError::incomplete(
            "--unsigned cannot be combined with a signing key",
        ));
    }
    if !unsigned && key_path.is_none() {
        return Err(EvidenceError::incomplete(
            "captured evaluation requires a signing key unless unsigned mode is explicit",
        ));
    }
    let key = key_path
        .map(|path| {
            fs::read_file(Path::new(path), CONTROL)
                .map_err(io_error)
                .and_then(|raw| crypto::private_key(&raw))
        })
        .transpose()?;
    let mut files = BTreeMap::new();
    for (name, value) in [
        ("request.json", &request),
        ("inputs/policy.json", policy),
        ("records/baseline.json", baseline),
        ("records/subject.json", subject),
        ("reports/evaluation.report.json", &comparison),
    ] {
        let bytes = canonical(value)?;
        if bytes.len()
            > if name == "request.json" {
                REQUEST
            } else {
                PAYLOAD
            }
        {
            return Err(EvidenceError::contract(format!(
                "{name} exceeds its byte limit"
            )));
        }
        files.insert(name.to_string(), bytes);
    }
    let request_digest = sha256_digest(&files["request.json"]);
    let mut references = Map::new();
    for (role, path) in PAYLOADS {
        references.insert(
            role.into(),
            json!({"path":path,"digest":sha256_digest(&files[path])}),
        );
    }
    let id = comparison_id(
        &request_digest,
        references["baseline"]["digest"].as_str().unwrap(),
        references["subject"]["digest"].as_str().unwrap(),
        references["policy"]["digest"].as_str().unwrap(),
    )?;
    let ledger = checksums(&files);
    let mut manifest = json!({"format":PACK_FORMAT,"kind":"captured","authentication":if unsigned {"unsigned_local"}else{"signed"},
        "comparison_id":id,"request_digest":request_digest,"signing_key_fingerprint":null,
        "files":references,"checksums":"checksums.sha256","checksums_sha256_digest":sha256_digest(&ledger).trim_start_matches("sha256:")});
    if let Some(key) = &key {
        manifest["signing_key_fingerprint"] = json!(crypto::fingerprint(&key.verifying_key()));
        manifest["signature"] = json!("manifest.signature.json");
    }
    validate_manifest(&manifest)?;
    let manifest_raw = canonical(&manifest)?;
    if let Some(key) = &key {
        files.insert(
            "manifest.signature.json".into(),
            canonical(&crypto::manifest_signature(key, &manifest_raw)?)?,
        );
    }
    files.insert("checksums.sha256".into(), ledger);
    files.insert("manifest.json".into(), manifest_raw);
    if files.values().map(Vec::len).sum::<usize>() > TOTAL {
        return Err(EvidenceError::contract(
            "captured inventory exceeds total byte limit",
        ));
    }
    let destination = PathBuf::from(required_string(config, "destination")?);
    let mut stage = fs::StagedDirectory::new(&destination).map_err(io_error)?;
    for directory in ["inputs", "records", "reports"] {
        stage
            .create_directory(Path::new(directory))
            .map_err(io_error)?;
    }
    for (path, raw) in &files {
        stage.write(Path::new(path), raw, 0o444).map_err(io_error)?;
    }
    stage.commit().map_err(io_error)?;
    Ok(
        json!({"evidence_path":destination,"comparison":comparison,"manifest":manifest,"pack_manifest_digest":sha256_digest(&files["manifest.json"])}),
    )
}

#[cfg(test)]
mod immutable_input_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn load_fixture(
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<u8>>),
        owned: bool,
        mutate_source: bool,
    ) -> Result<CapturedInputs> {
        let run = |side: &str| {
            json!({
                "format":"invarlock/evaluation-run-v1", "run_id":side,
                "artifact_digest":digest(&json!(side)).unwrap(), "source_digest":null,
                "source":{"name":"fixture","version":"1"}, "score_provenance":{},
                "records":[{"id":"one","input":"question","expected":"yes","output":"yes",
                    "scores":{},"metadata":{},"error":null,"context":{}}]
            })
        };
        let baseline = run("baseline");
        let subject = run("subject");
        let policy = json!({
            "format":"invarlock/comparison-policy-v1", "slices":[],
            "metrics":[{"name":"exact","kind":"exact_match","configuration":{},
                "direction":"higher","unit":"score","aggregation":"mean",
                "minimum_count":1,"maximum_regression":0,"maximum_interval_width":100}]
        });
        let request = json!({
            "format":"invarlock/evaluation-request-v2", "execution":{"mode":"captured"},
            "comparison":{"baseline":{"adapter":"invarlock","run_digest":digest(&baseline).unwrap()},
                "subject":{"adapter":"invarlock","run_digest":digest(&subject).unwrap()},
                "policy_digest":digest(&policy).unwrap()}
        });
        let report = analysis::compare_runs(&baseline, &subject, &policy, None).unwrap();
        let mut files = BTreeMap::from([
            ("request.json".into(), canonical(&request).unwrap()),
            ("inputs/policy.json".into(), canonical(&policy).unwrap()),
            (
                "records/baseline.json".into(),
                canonical(&baseline).unwrap(),
            ),
            ("records/subject.json".into(), canonical(&subject).unwrap()),
            (
                "reports/evaluation.report.json".into(),
                canonical(&report).unwrap(),
            ),
        ]);
        let mut references = Map::new();
        for (role, name) in PAYLOADS {
            references.insert(
                role.into(),
                json!({"path":name,"digest":sha256_digest(&files[name])}),
            );
        }
        let request_digest = references["request"]["digest"].as_str().unwrap();
        let mut manifest = json!({
            "format":PACK_FORMAT,"kind":"captured","authentication":"unsigned_local",
            "signing_key_fingerprint":null,"checksums":"checksums.sha256",
            "comparison_id":comparison_id(request_digest,
                references["baseline"]["digest"].as_str().unwrap(),
                references["subject"]["digest"].as_str().unwrap(),
                references["policy"]["digest"].as_str().unwrap()).unwrap(),
            "request_digest":request_digest,"files":references
        });
        mutate(&mut files);
        for (role, name) in PAYLOADS {
            manifest["files"][role]["digest"] = json!(sha256_digest(&files[name]));
        }
        let ledger = checksums(&files);
        manifest["checksums_sha256_digest"] =
            json!(sha256_digest(&ledger).trim_start_matches("sha256:"));
        files.insert("checksums.sha256".into(), ledger);
        files.insert("manifest.json".into(), canonical(&manifest).unwrap());
        let temporary = Temporary(std::env::temp_dir().canonicalize().unwrap().join(format!(
            "captured-immutable-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        std::fs::create_dir(&temporary.0).unwrap();
        for (name, bytes) in &files {
            let path = temporary.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        let root = fs::PinnedDirectory::open(&temporary.0, false).unwrap();
        let names = files.keys().map(|name| (name.clone(), PAYLOAD)).collect();
        let dirs = ["inputs", "records", "reports"]
            .into_iter()
            .map(String::from)
            .collect();
        drop(files);
        let snapshot = root
            .capture_selected(fs::SnapshotLimits::default(), &names, &dirs)
            .unwrap();
        let (entries, guard) = snapshot.into_parts();
        let loaded = if owned {
            load_owned_inputs(entries)?
        } else {
            load_inputs(&entries)?
        };
        if mutate_source {
            std::fs::write(temporary.0.join("records/baseline.json"), b"{}\n").unwrap();
        }
        guard
            .verify_source()
            .map_err(|_| EvidenceError::integrity("captured inventory changed after capture"))?;
        Ok(loaded)
    }
    #[test]
    fn later_syntax_failure_precedes_earlier_canonical_or_schema_failure() {
        for owned in [false, true] {
            for earlier in [b"{ }\n".as_slice(), b"{}\n"] {
                let failure = load_fixture(
                    |files| {
                        files.insert("inputs/policy.json".into(), earlier.to_vec());
                        files.insert("reports/evaluation.report.json".into(), b"{".to_vec());
                    },
                    owned,
                    false,
                )
                .err()
                .unwrap();
                assert_eq!(failure.message, "report is not strict JSON");
                assert_eq!(failure.exit_code, 4);
            }
        }
    }
    #[test]
    fn loaded_schema_evidence_owns_values_after_snapshot_release() {
        let loaded = load_fixture(|_| {}, true, false).unwrap();
        let baseline = loaded.payloads.schema_checked("baseline").unwrap();
        let digest = baseline.canonical_digest().unwrap().to_owned();
        let mut copy = loaded.payloads["baseline"].clone();
        copy["records"] = Value::Null;
        assert!(SchemaCheckedValue::new(copy, "run").is_err());
        assert_eq!(baseline.canonical_digest().unwrap(), digest);
        analysis::ValidatedRun::from_schema_checked(baseline).unwrap();
        assert_eq!(loaded.payloads.get("baseline"), Some(baseline.value()));
        let shared =
            crate::evidence_sets::shared_checked_captured_inputs(&loaded.payloads).unwrap();
        let mut values = loaded.payloads.into_values();
        assert_eq!(
            shared,
            crate::evidence_sets::shared_captured_inputs(&values).unwrap()
        );
        values["baseline"]["records"] = Value::Null;
        assert!(SchemaCheckedValue::new(values.remove("baseline").unwrap(), "run").is_err());
    }
    #[test]
    fn owned_and_borrowed_loads_match_and_reject_source_changes() {
        let borrowed = load_fixture(|_| {}, false, false).unwrap();
        let owned = load_fixture(|_| {}, true, false).unwrap();
        assert_eq!(
            canonical(&borrowed.manifest).unwrap(),
            canonical(&owned.manifest).unwrap()
        );
        assert_eq!(borrowed.signer, owned.signer);
        assert_eq!(
            canonical(&Value::Object(borrowed.payloads.into_values())).unwrap(),
            canonical(&Value::Object(owned.payloads.into_values())).unwrap()
        );
        for owned in [false, true] {
            let error = load_fixture(|_| {}, owned, true).err().unwrap();
            assert_eq!(error.message, "captured inventory changed after capture");
            assert_eq!(error.exit_code, 6);
        }
    }
    #[test]
    fn record_limits_precede_deferred_canonical_results() {
        for owned in [false, true] {
            let error = load_fixture(
                |files| {
                    files.insert("inputs/policy.json".into(), b"{ }\n".to_vec());
                    let mut run = object(&files["records/baseline.json"], "run", false).unwrap();
                    run["records"] = Value::Array(vec![Value::Null; 50_001]);
                    files.insert("records/baseline.json".into(), canonical(&run).unwrap());
                },
                owned,
                false,
            )
            .err()
            .unwrap();
            assert_eq!(error.message, "run exceeds the 50,000-record limit");
            assert_eq!(error.exit_code, 4);
        }
    }
    #[test]
    fn deferred_canonical_and_schema_errors_keep_payload_order() {
        for owned in [false, true] {
            for (policy_raw, canonical_error) in [(b"{ }\n".as_slice(), true), (b"{}\n", false)] {
                let error = load_fixture(
                    |files| {
                        files.insert("inputs/policy.json".into(), policy_raw.to_vec());
                        files.insert("records/baseline.json".into(), b"{ }\n".to_vec());
                    },
                    owned,
                    false,
                )
                .err()
                .unwrap();
                if canonical_error {
                    assert_eq!(error.message, "policy must be canonical JSON");
                } else {
                    let expected = validate_schema(&json!({}), "policy").unwrap_err();
                    assert_eq!(error.message, bounded(&expected.to_string()));
                }
                assert_eq!(error.exit_code, 4);
            }
        }
    }
}
