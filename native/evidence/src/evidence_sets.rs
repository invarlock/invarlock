//! Fresh conjunction of independently authorized captured and judge evidence.
use crate::{
    canonical, captured, digest, judge, required_string, trust_inputs, EvidenceError, Result,
};
use invarlock_native_contracts::{
    first_schema_error, parse_json, python_repr, sha256_digest, SchemaErrorOrder,
};
use invarlock_native_filesystem as fs;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    os::unix::fs::DirBuilderExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
const CONTROL: usize = 256 * 1024;
const RESULT_LIMIT: usize = 1024 * 1024;
const INDEX: &str = "evidence-set.json";
const MEMBERS: [&str; 2] = ["deterministic", "judge"];
const SCOPE: &str = "same-answer-component-conjunction-v1";
const STATISTICAL_SCOPE: &str = "component-methods-no-joint-confidence";
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn io_error(e: impl std::fmt::Display) -> EvidenceError {
    error(e.to_string())
}
fn truncate(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}
fn exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}
fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.into())
    } else {
        Ok(std::env::current_dir().map_err(io_error)?.join(path))
    }
}
pub(crate) fn read(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let root = fs::PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), false)
        .map_err(io_error)?;
    let name = path
        .file_name()
        .ok_or_else(|| error("file name is required"))?;
    let relative = Path::new(name);
    let label = name.to_string_lossy();
    let meta = root.inspect(relative).map_err(|e| match e {
        fs::Error::UnsafePath(ref message)
            if message == "entry must be a regular file or directory" =>
        {
            error(format!("{label} must be a non-symlink regular file"))
        }
        _ => io_error(e),
    })?;
    if meta.identity.mode & 0o170000 != 0o100000 {
        return Err(error(format!("{label} must be a non-symlink regular file")));
    }
    if meta.identity.size > maximum as u64 {
        return Err(error(format!("{label} exceeds the {maximum}-byte limit")));
    }
    let (raw, _) = root.read_regular(relative, maximum).map_err(io_error)?;
    root.verify().map_err(io_error)?;
    Ok(raw)
}
pub(crate) fn read_object(path: &Path, maximum: usize) -> Result<(Value, Vec<u8>)> {
    let raw = read(path, maximum)?;
    let label = path.file_name().unwrap_or_default().to_string_lossy();
    let value = parse_json(&raw, &label)?;
    if !value.is_object() {
        return Err(error(format!("{label} must contain a JSON object")));
    }
    Ok((value, raw))
}
pub(crate) fn validate(value: &Value, schema: &str, label: &str) -> Result<()> {
    if let Some(e) = first_schema_error(value, schema, SchemaErrorOrder::Schema)? {
        return Err(error(format!(
            "{label} is invalid: {}",
            truncate(&e.message, 240)
        )));
    }
    Ok(())
}
pub(crate) fn member_path(root: &Path, reference: &Value) -> Result<PathBuf> {
    let text = reference.as_str().unwrap_or("");
    if text.is_empty()
        || text.starts_with(['/', '~', '\\'])
        || text.contains("://")
        || text.contains('\\')
        || text.chars().any(|c| (c as u32) < 32 || c == '\u{007f}')
        || text.as_bytes().get(1) == Some(&b':')
        || text.split('/').any(|p| p.is_empty() || p == "..")
        || text.ends_with('/')
    {
        return Err(error("member path must be a safe relative reference"));
    }
    let mut path = absolute(root)?;
    for part in text.split('/').filter(|p| *p != ".") {
        path.push(part);
    }
    Ok(path)
}
fn external(path: &Path, root: &Path) -> Result<()> {
    captured::require_outside(path, root)
        .map_err(|_| error("recipient inputs and outputs must remain outside the evidence set"))
}
pub(crate) fn index_paths(root: &Path, index: &Value) -> Result<BTreeMap<&'static str, PathBuf>> {
    let paths = MEMBERS
        .into_iter()
        .map(|name| Ok((name, member_path(root, &index["members"][name]["path"])?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    if paths["deterministic"].starts_with(&paths["judge"])
        || paths["judge"].starts_with(&paths["deterministic"])
    {
        return Err(error("evidence set member paths overlap"));
    }
    if ["manifest.json", "envelope.json"]
        .iter()
        .any(|name| exists(&root.join(name)))
    {
        return Err(error(
            "evidence set contains conflicting format discriminators",
        ));
    }
    Ok(paths)
}
pub(crate) fn check_statements(root: &Path, index: &Value) -> Result<()> {
    for (name, statement, other) in [
        ("deterministic", "manifest.json", "envelope.json"),
        ("judge", "envelope.json", "manifest.json"),
    ] {
        let path = member_path(root, &index["members"][name]["path"])?;
        if exists(&path.join(INDEX)) {
            return Err(error("nested evidence sets are unsupported"));
        }
        if exists(&path.join(other)) {
            return Err(error("member contains conflicting format discriminators"));
        }
        if sha256_digest(&read(&path.join(statement), CONTROL)?)
            != index["members"][name]["statement_sha256"]
        {
            return Err(error(format!("{name} statement differs from index pin")));
        }
    }
    Ok(())
}
fn empty_member(kind: &str) -> Value {
    json!({"kind":kind,"verified":false,"authenticated":false,"replayed":false,"accepted":false,"decision":null,"statement_sha256":null,"trust_profile_sha256":null,"receipt":null,"errors":[]})
}
fn empty_result() -> Value {
    json!({"format_version":"invarlock/evidence-set-verification-v1","kind":"evidence_set","scope":SCOPE,"statistical_scope":STATISTICAL_SCOPE,"ok":false,"verified":false,"authenticated":false,"replayed":false,"accepted":false,"decision":null,"index_sha256":null,"recipient_policy_sha256":null,"shared_inputs":null,"shared_inputs_verified":false,"members":{"deterministic":empty_member("captured"),"judge":empty_member("judge")},"errors":[]})
}
pub fn shared_captured_inputs(payloads: &Map<String, Value>) -> Result<Value> {
    let mut sets = Vec::new();
    for role in ["baseline", "subject"] {
        let run = payloads
            .get(role)
            .ok_or_else(|| error("component run is missing"))?;
        let rows = run["records"]
            .as_array()
            .ok_or_else(|| error("component records must be an array"))?;
        let cases:Vec<_>=rows.iter().map(|row|json!({"id":row["id"],"input":row["input"],"expected":row["expected"],"metadata":row["metadata"]})).collect();
        sets.push(judge::case_set_digest(
            &json!({"format":"invarlock/evaluation-case-set-v1","cases":cases}),
        )?);
    }
    if sets[0] != sets[1] {
        return Err(error("component baseline and subject case sets differ"));
    }
    let subject = &payloads["subject"];
    let mut value = json!({"baseline_run_sha256":digest(&payloads["baseline"])? ,"subject_run_sha256":digest(subject)? ,"case_set_sha256":sets[0],"subject_artifact_sha256":subject["artifact_digest"]});
    if let Some(service) = subject.get("service_identity") {
        value["subject_service_identity_sha256"] = json!(digest(service)?);
    }
    Ok(value)
}
pub(crate) fn require_shared(observed: &Value, expected: &Value, message: &str) -> Result<()> {
    if observed == expected {
        return Ok(());
    }
    for field in [
        "baseline_run_sha256",
        "subject_run_sha256",
        "case_set_sha256",
        "subject_artifact_sha256",
        "subject_service_identity_sha256",
    ] {
        if observed.get(field) != expected.get(field) {
            let repr = |v: Option<&Value>| {
                v.map(|v| truncate(&python_repr(v), 80))
                    .unwrap_or("<missing>".into())
            };
            return Err(error(format!(
                "{message}: {field}; expected {}, observed {}",
                repr(expected.get(field)),
                repr(observed.get(field))
            )));
        }
    }
    Err(error(format!("{message}: unexpected shared input fields")))
}
pub fn require_deterministic_policy(policy: &Value) -> Result<()> {
    invarlock_native_core::validate_policy(policy).map_err(io_error)?;
    if policy["metrics"].as_array().is_some_and(|a| {
        a.iter().any(|m| {
            !matches!(
                m["kind"].as_str(),
                Some(
                    "exact_match"
                        | "normalized_match"
                        | "numeric_tolerance"
                        | "json_fields"
                        | "json_exact"
                        | "token_f1"
                )
            )
        })
    }) {
        return Err(error("evidence set requires recomputed deterministic captured metrics; recorded scores are unsupported"));
    }
    Ok(())
}
pub(crate) fn load_captured(path: &Path) -> Result<captured::CapturedInputs> {
    let root = fs::PinnedDirectory::open(path, false).map_err(io_error)?;
    let (raw, _) = root
        .read_regular(Path::new("manifest.json"), CONTROL)
        .map_err(io_error)?;
    let manifest = captured::object(&raw, "captured manifest", true)?;
    let mut names: BTreeMap<String, usize> = [
        ("manifest.json", CONTROL),
        ("checksums.sha256", CONTROL),
        ("request.json", 1024 * 1024),
        ("inputs/policy.json", 128 * 1024 * 1024),
        ("records/baseline.json", 128 * 1024 * 1024),
        ("records/subject.json", 128 * 1024 * 1024),
        ("reports/evaluation.report.json", 128 * 1024 * 1024),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    if manifest["authentication"] == "signed" {
        names.insert("manifest.signature.json".into(), CONTROL);
    }
    let dirs = BTreeSet::from(["inputs".into(), "records".into(), "reports".into()]);
    let snapshot = root
        .capture_selected(
            fs::SnapshotLimits {
                max_file_bytes: 128 * 1024 * 1024,
                max_total_bytes: 384 * 1024 * 1024,
                ..Default::default()
            },
            &names,
            &dirs,
        )
        .map_err(io_error)?;
    let loaded = captured::load_inputs(snapshot.entries())?;
    snapshot.verify_source().map_err(io_error)?;
    Ok(loaded)
}
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Result<Self> {
        let root = std::env::temp_dir().canonicalize().map_err(io_error)?;
        for _ in 0..100 {
            let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(io_error)?
                .as_nanos();
            let path = root.join(format!(
                "invarlock-evidence-set-{}-{time}-{sequence}",
                std::process::id()
            ));
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_error(e)),
            }
        }
        Err(error("unable to allocate private receipt directory"))
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn verify_inner(
    root: &Path,
    policy_path: &Path,
    result: &mut Value,
    maximum: &Value,
) -> Result<()> {
    let (index, index_raw) = read_object(&root.join(INDEX), CONTROL)?;
    validate(&index, "evidence_set", "evidence set index")?;
    let paths = index_paths(root, &index)?;
    external(policy_path, root)?;
    let (policy, policy_raw) = read_object(policy_path, CONTROL)?;
    validate(
        &policy,
        "evidence_set_recipient_policy",
        "evidence set recipient policy",
    )?;
    result["index_sha256"] = json!(sha256_digest(&index_raw));
    result["recipient_policy_sha256"] = json!(sha256_digest(&policy_raw));
    result["shared_inputs"] = policy["shared_inputs"].clone();
    if sha256_digest(&index_raw) != policy["index_sha256"] {
        return Err(error("evidence set index differs from recipient pin"));
    }
    check_statements(root, &index)?;
    let mut profiles = BTreeMap::new();
    let mut profile_values = BTreeMap::new();
    let mut profile_raws = BTreeMap::new();
    for name in MEMBERS {
        let path = member_path(
            policy_path.parent().unwrap_or(Path::new(".")),
            &policy["members"][name]["trust_profile"],
        )?;
        external(&path, root)?;
        let (value, raw) = read_object(&path, CONTROL)?;
        let expected = &policy["members"][name]["trust_profile_sha256"];
        if sha256_digest(&raw) != *expected {
            return Err(error(format!(
                "{name} trust profile differs from recipient pin"
            )));
        }
        result["members"][name]["statement_sha256"] =
            index["members"][name]["statement_sha256"].clone();
        result["members"][name]["trust_profile_sha256"] = expected.clone();
        profiles.insert(name, path);
        profile_values.insert(name, value);
        profile_raws.insert(name, raw);
    }
    let trust = trust_inputs::load(&profiles["deterministic"], None)?;
    if !trust.captured {
        return Err(error(
            "deterministic member requires a captured trust profile",
        ));
    }
    if trust.config["trust_profile_digest"] != digest(&profile_values["deterministic"])? {
        return Err(error("captured trust profile changed while loading"));
    }
    for key in ["policy_path", "verifier_signing_key_path"] {
        external(Path::new(required_string(&trust.config, key)?), root)?;
    }
    let deterministic_policy = parse_json(&trust.policy_bytes, "deterministic policy")?;
    if !deterministic_policy.is_object() {
        return Err(error("deterministic policy must be an object"));
    }
    require_deterministic_policy(&deterministic_policy)?;
    let shared = &policy["shared_inputs"];
    for role in ["baseline", "subject"] {
        if trust.config[format!("expected_{role}_run")] != shared[format!("{role}_run_sha256")] {
            return Err(error(format!(
                "captured {role} differs from the shared recipient pin"
            )));
        }
    }
    let loaded = load_captured(&paths["deterministic"])?;
    if digest(&loaded.manifest)? != index["members"]["deterministic"]["statement_sha256"] {
        return Err(error("captured statement changed while loading"));
    }
    require_shared(
        &shared_captured_inputs(&loaded.payloads)?,
        shared,
        "captured runs, cases, or subject artifact differ from shared recipient pins",
    )?;
    let (expected_envelope, expected_envelope_raw) =
        read_object(&paths["judge"].join("envelope.json"), CONTROL)?;
    if sha256_digest(&expected_envelope_raw) != index["members"]["judge"]["statement_sha256"] {
        return Err(error("judge statement changed before verification"));
    }
    let judge_policy = &profile_values["judge"];
    validate(
        judge_policy,
        "judge_measurement_recipient_policy",
        "judge recipient policy",
    )?;
    let intended = shared
        .get("subject_service_identity_sha256")
        .unwrap_or(&shared["subject_artifact_sha256"]);
    if judge_policy["intended_subject"] != *intended {
        return Err(error(
            "judge intended subject differs from shared recipient pin",
        ));
    }
    for field in [
        "baseline_run_sha256",
        "subject_run_sha256",
        "case_set_sha256",
    ] {
        if judge_policy["bindings"][field] != shared[field] {
            return Err(error(format!(
                "judge {field} differs from shared recipient pin"
            )));
        }
    }
    let temporary = Temporary::new()?;
    let captured_receipt_path = temporary.0.join("captured.json");
    let mut config = trust.config.clone();
    config["evidence_path"] = json!(paths["deterministic"]);
    config["receipt_path"] = json!(captured_receipt_path);
    config["max_bootstrap_draws"] = maximum.clone();
    let captured_result = match captured::verify_with_inputs(
        &config,
        Some(&trust.policy_bytes),
        Some(&trust.signing_key_bytes),
    ) {
        Ok(value) => value,
        Err(e) => {
            result["members"]["deterministic"]["errors"] = json!([truncate(&e.message, 1024)]);
            if exists(&captured_receipt_path) {
                result["members"]["deterministic"]["receipt"] =
                    json!(String::from_utf8(read(&captured_receipt_path, CONTROL)?)
                        .map_err(io_error)?);
            }
            return Err(e);
        }
    };
    let captured_receipt =
        String::from_utf8(read(&captured_receipt_path, CONTROL)?).map_err(io_error)?;
    if captured_result["pack_manifest_digest"]
        != index["members"]["deterministic"]["statement_sha256"]
    {
        return Err(error("verified captured statement differs from index pin"));
    }
    for key in ["verified", "authenticated", "replayed"] {
        result["members"]["deterministic"][key] = json!(true);
    }
    result["members"]["deterministic"]["accepted"] = captured_result["ok"].clone();
    result["members"]["deterministic"]["decision"] = captured_result["decision"].clone();
    result["members"]["deterministic"]["receipt"] = json!(captured_receipt);
    let judge_config = json!({"evidence_path":paths["judge"],"recipient_policy_path":profiles["judge"],"receipt_path":temporary.0.join("judge.json"),"verifier_identity":trust.config["verifier_identity"],"verifier_signing_key_path":trust.config["verifier_signing_key_path"]});
    let judged = judge::verify(&judge_config)?;
    judge::write_receipt_with_key(&judge_config, &judged, Some(&trust.signing_key_bytes))?;
    let judge_receipt =
        String::from_utf8(read(&temporary.0.join("judge.json"), CONTROL)?).map_err(io_error)?;
    if judged["recipient_policy_sha256"] != judge::object_digest(judge_policy)? {
        return Err(error("judge trust profile changed during verification"));
    }
    if !judged["envelope_sha256"].is_null()
        && judged["envelope_sha256"] != judge::object_digest(&expected_envelope)?
    {
        return Err(error("verified judge statement differs from index pin"));
    }
    for key in [
        "verified",
        "authenticated",
        "replayed",
        "accepted",
        "decision",
        "errors",
    ] {
        result["members"]["judge"][key] = judged[key].clone();
    }
    result["members"]["judge"]["receipt"] = json!(judge_receipt);
    if judged["verified"] != true {
        return Err(error(format!(
            "judge component did not authenticate and replay: {}",
            judged["errors"]
                .as_array()
                .map(|a| a
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("; "))
                .unwrap_or_default()
        )));
    }
    if judged["bindings"].is_null()
        || [
            "baseline_run_sha256",
            "subject_run_sha256",
            "case_set_sha256",
        ]
        .iter()
        .any(|field| judged["bindings"][*field] != shared[*field])
    {
        return Err(error(
            "verified judge shared inputs differ from recipient pins",
        ));
    }
    if judged["intended_subject"] != *intended {
        return Err(error(
            "verified judge subject identity differs from recipient pin",
        ));
    }
    let (analysis_policy, _) = read_object(&paths["judge"].join("analysis_policy.json"), CONTROL)?;
    if judge::object_digest(&analysis_policy)? != judged["bindings"]["analysis_policy_sha256"] {
        return Err(error("judge analysis policy changed during composition"));
    }
    if analysis_policy["decision_role"] != "required" {
        return Err(error(
            "evidence set requires a required judge policy; advisory promotion is forbidden",
        ));
    }
    check_statements(root, &index)?;
    if read(&root.join(INDEX), CONTROL)? != index_raw || read(policy_path, CONTROL)? != policy_raw {
        return Err(error("composition inputs changed during verification"));
    }
    for name in MEMBERS {
        if read(&profiles[name], CONTROL)? != profile_raws[name] {
            return Err(error("component trust profile changed during verification"));
        }
    }
    let decisions: Vec<_> = MEMBERS
        .iter()
        .map(|name| &result["members"][*name]["decision"])
        .collect();
    let decision = if decisions.iter().any(|v| **v == "regression") {
        "regression"
    } else if decisions.iter().any(|v| **v == "insufficient_evidence") {
        "insufficient_evidence"
    } else {
        "pass"
    };
    let accepted = MEMBERS
        .iter()
        .all(|name| result["members"][*name]["accepted"] == true);
    for key in [
        "verified",
        "authenticated",
        "replayed",
        "shared_inputs_verified",
    ] {
        result[key] = json!(true);
    }
    result["decision"] = json!(decision);
    result["accepted"] = json!(accepted);
    result["ok"] = json!(accepted);
    Ok(())
}
pub fn verify(config: &Value) -> Result<Value> {
    let mut result = empty_result();
    let root = absolute(Path::new(required_string(config, "evidence_path")?))?;
    let policy = absolute(Path::new(required_string(config, "recipient_policy_path")?))?;
    let receipt = config
        .get("receipt_path")
        .filter(|v| !v.is_null())
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let maximum = config
        .get("max_bootstrap_draws")
        .cloned()
        .unwrap_or(json!(102_400_000));
    let checked: Result<()> = (|| {
        invarlock_native_contracts::bootstrap_budget(Some(&maximum), 102_400_000)?;
        if let Some(path) = &receipt {
            external(path, &root)?;
            if exists(path) {
                return Err(error("composition receipt destination already exists"));
            }
        }
        let root_guard = fs::PinnedDirectory::open(&root, false).map_err(io_error)?;
        let policy_guard =
            fs::PinnedDirectory::open(policy.parent().unwrap_or(Path::new(".")), false)
                .map_err(io_error)?;
        verify_inner(&root, &policy, &mut result, &maximum)?;
        root_guard.verify().map_err(io_error)?;
        policy_guard.verify().map_err(io_error)?;
        Ok(())
    })();
    if let Err(e) = checked {
        for key in ["ok", "verified", "accepted", "shared_inputs_verified"] {
            result[key] = json!(false);
        }
        result["decision"] = Value::Null;
        result["errors"] = json!([truncate(&e.message, 1024)]);
    }
    validate(
        &result,
        "evidence_set_verification",
        "evidence set verification",
    )?;
    let raw = canonical(&result)?;
    if raw.len() > RESULT_LIMIT {
        return Err(error("composition verification result exceeds byte limit"));
    }
    if let Some(path) = receipt {
        external(&path, &root)?;
        fs::write_file_no_replace(&path, &raw, 0o600, true).map_err(io_error)?;
    }
    Ok(result)
}
pub fn verify_stored(config: &Value) -> Result<Value> {
    let mut fresh = config.clone();
    fresh
        .as_object_mut()
        .ok_or_else(|| error("configuration must be an object"))?
        .remove("receipt_path");
    let mut actual = verify(&fresh)?;
    let checked: Result<()> = (|| {
        let (claimed, _) = read_object(
            Path::new(required_string(config, "stored_result_path")?),
            RESULT_LIMIT,
        )?;
        validate(
            &claimed,
            "evidence_set_verification",
            "stored composition result",
        )?;
        if canonical(&claimed)? != canonical(&actual)? {
            return Err(error(
                "stored composition result differs from fresh verification",
            ));
        }
        Ok(())
    })();
    if let Err(e) = checked {
        for key in ["ok", "verified", "accepted"] {
            actual[key] = json!(false);
        }
        actual["errors"] = json!([truncate(&e.message, 1024)]);
    }
    Ok(actual)
}
pub fn write_index(config: &Value) -> Result<Value> {
    let root = absolute(Path::new(required_string(config, "evidence_path")?))?;
    let mut index = json!({"format":"invarlock/evidence-set-v1","members":{}});
    for (name, kind, file) in [
        ("deterministic", "captured", "manifest.json"),
        ("judge", "judge", "envelope.json"),
    ] {
        let reference = Value::String(required_string(config, name)?.into());
        let path = member_path(&root, &reference)?;
        index["members"][name] = json!({"kind":kind,"path":reference,"statement_sha256":sha256_digest(&read(&path.join(file),CONTROL)?)});
    }
    validate(&index, "evidence_set", "evidence set index")?;
    index_paths(&root, &index)?;
    check_statements(&root, &index)?;
    fs::write_file_no_replace(&root.join(INDEX), &canonical(&index)?, 0o600, true)
        .map_err(io_error)?;
    Ok(json!({"index_path":root.join(INDEX)}))
}
