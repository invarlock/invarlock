//! Independent native evidence verification over one descriptor-bound snapshot.
mod receipt;
pub(crate) mod replay;
use crate::{canonical, crypto, is_digest, required_string, EvidenceError, Result};
use invarlock_native_contracts::{
    canonical_json_no_newline, parse_json, sha256_digest, validate_schema,
};
use invarlock_native_filesystem as fs;
pub use receipt::verify_receipt;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
pub(crate) const ROLES: [&str; 6] = [
    "baseline",
    "subject",
    "dataset",
    "baseline_runtime",
    "subject_runtime",
    "policy",
];
pub(crate) const EVIDENCE: [(&str, &str); 15] = [
    ("request", "request.json"),
    ("schedule", "schedule/runtime-behavioral-schedule.json"),
    ("evaluation_report", "reports/evaluation.report.json"),
    ("baseline_run_report", "runs/baseline/report.json"),
    ("subject_run_report", "runs/subject/report.json"),
    (
        "baseline_runtime_manifest",
        "providers/baseline/runtime.manifest.json",
    ),
    (
        "subject_runtime_manifest",
        "providers/subject/runtime.manifest.json",
    ),
    ("baseline_runtime_config", "providers/baseline/run.yaml"),
    ("subject_runtime_config", "providers/subject/run.yaml"),
    (
        "baseline_provider_identity",
        "providers/baseline/model-artifact.identity.json",
    ),
    (
        "subject_provider_identity",
        "providers/subject/model-artifact.identity.json",
    ),
    (
        "baseline_provider_receipt",
        "providers/baseline/runtime-provider.receipt.json",
    ),
    (
        "subject_provider_receipt",
        "providers/subject/runtime-provider.receipt.json",
    ),
    (
        "baseline_scoring_observation",
        "providers/baseline/runtime-scoring.observation.json",
    ),
    (
        "subject_scoring_observation",
        "providers/subject/runtime-scoring.observation.json",
    ),
];
const PACK_FORMAT: &str = "invarlock/evidence-pack-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;
fn integrity(message: impl Into<String>) -> EvidenceError {
    EvidenceError::integrity(message)
}
fn hex(raw: &[u8]) -> String {
    sha256_digest(raw)[7..].to_owned()
}
pub(crate) fn object(raw: &[u8], label: &str) -> Result<Value> {
    let v = parse_json(raw, label).map_err(|e| integrity(e.to_string()))?;
    if !v.is_object() {
        return Err(integrity(format!("{label} must be a JSON object")));
    }
    Ok(v)
}
fn exact_keys(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)))
}
fn normalized(v: &Value, label: &str) -> Result<String> {
    let s = v.as_str().unwrap_or("").trim().to_lowercase();
    if !is_digest(&s) {
        return Err(EvidenceError::incomplete(format!(
            "{label} must be a sha256:... digest"
        )));
    }
    Ok(s)
}
pub(crate) fn raw<'a>(
    snapshot: &'a fs::PackSnapshot,
    path: &str,
    limit: usize,
) -> Result<&'a [u8]> {
    let entry = snapshot
        .entries()
        .get(path)
        .ok_or_else(|| integrity(format!("{path} is unavailable")))?;
    if entry.bytes().len() > limit {
        return Err(integrity(format!(
            "{path} exceeds the {limit}-byte size limit"
        )));
    }
    Ok(entry.bytes())
}
fn outside(candidate: &Path, pack: &Path, label: &str) -> Result<()> {
    let pack = pack
        .canonicalize()
        .map_err(|_| EvidenceError::incomplete("evidence must be a real directory"))?;
    let mut ancestor = candidate.to_path_buf();
    let mut tail = Vec::new();
    while !ancestor.exists() {
        let leaf = ancestor
            .file_name()
            .ok_or_else(|| {
                EvidenceError::incomplete(format!("{label} path could not be resolved safely"))
            })?
            .to_owned();
        tail.push(leaf);
        ancestor = ancestor.parent().unwrap_or(Path::new(".")).to_path_buf();
        if ancestor.as_os_str().is_empty() {
            ancestor = PathBuf::from(".");
        }
    }
    let mut resolved = ancestor.canonicalize().map_err(|_| {
        EvidenceError::incomplete(format!("{label} path could not be resolved safely"))
    })?;
    for leaf in tail.iter().rev() {
        resolved.push(leaf);
    }

    if resolved.starts_with(pack) {
        return Err(EvidenceError::incomplete(format!(
            "{label} must remain outside the submitted evidence pack"
        )));
    }
    Ok(())
}
struct Inputs {
    pack: PathBuf,
    policy: Value,
    anchors: Value,
    receipt: PathBuf,
    key: ed25519_dalek::SigningKey,
    identity: String,
    profile: Value,
}
fn inputs(
    config: &Value,
    policy_override: Option<&[u8]>,
    key_override: Option<&[u8]>,
) -> Result<Inputs> {
    if config
        .get("expected_baseline_run")
        .is_some_and(|v| !v.is_null())
        || config
            .get("expected_subject_run")
            .is_some_and(|v| !v.is_null())
        || config
            .get("max_bootstrap_draws")
            .is_some_and(|v| v.as_u64() != Some(102_400_000))
    {
        return Err(EvidenceError::incomplete(
            "captured anchors/work controls require captured evidence",
        ));
    }
    let pack = PathBuf::from(required_string(config, "evidence_path")?);
    if !pack.is_dir() || pack.is_symlink() {
        return Err(EvidenceError::incomplete(
            "evidence must be a real directory",
        ));
    }
    let policy_path = PathBuf::from(required_string(config, "policy_path")?);
    outside(&policy_path, &pack, "independent policy")?;
    let policy_bytes = match policy_override {
        Some(bytes) => bytes.to_vec(),
        None => fs::read_file(&policy_path, 4 * 1024 * 1024)
            .map_err(|e| EvidenceError::incomplete(e.to_string()))?,
    };
    if policy_bytes.len() > 4 * 1024 * 1024 {
        return Err(EvidenceError::incomplete(
            "independent policy exceeds byte limit",
        ));
    }
    let policy = object(&policy_bytes, "independent policy anchor")?;
    let mut anchors = json!({"policy_digest":sha256_digest(&policy_bytes),"artifact_digests":{},"runtime_digests":{}});
    for side in ["baseline", "subject"] {
        for kind in ["artifact", "runtime"] {
            let name = format!("expected_{side}_{kind}");
            anchors[format!("{kind}_digests")][side] = json!(normalized(
                &config[&name],
                &format!("independent {side} {kind} anchor")
            )?);
        }
    }
    anchors["schedule_digest"] = json!(normalized(
        &config["expected_schedule"],
        "independent schedule anchor"
    )?);
    anchors["signer_fingerprint"] = json!(normalized(
        &config["expected_signer"],
        "independent signer anchor"
    )?);
    if config
        .get("expected_request_digest")
        .is_some_and(|v| !v.is_null())
    {
        anchors["request_digest"] = json!(normalized(
            &config["expected_request_digest"],
            "independent request anchor"
        )?);
    }
    let receipt = PathBuf::from(required_string(config, "receipt_path")?);
    outside(&receipt, &pack, "verification receipt")?;
    let keypath = PathBuf::from(required_string(config, "verifier_signing_key_path")?);
    outside(&keypath, &pack, "verifier Ed25519 signing key")?;
    let key_bytes = match key_override {
        Some(bytes) => bytes.to_vec(),
        None => fs::read_file(&keypath, 64 * 1024)
            .map_err(|e| EvidenceError::incomplete(e.to_string()))?,
    };
    if key_bytes.len() > 64 * 1024 {
        return Err(EvidenceError::incomplete("verifier key exceeds byte limit"));
    }
    let key = crypto::private_key(&key_bytes)?;
    let identity = config["verifier_identity"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_owned();
    if identity.is_empty()
        || identity.len() > 256
        || !identity.as_bytes()[0].is_ascii_alphanumeric()
        || !identity
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:/-".contains(&b))
    {
        return Err(EvidenceError::incomplete("verifier identity is invalid"));
    }
    let profile = config
        .get("trust_profile_digest")
        .filter(|v| !v.is_null())
        .map(|v| normalized(v, "trust profile digest"))
        .transpose()?
        .map(Value::String)
        .unwrap_or(Value::Null);
    Ok(Inputs {
        pack,
        policy,
        anchors,
        receipt,
        key,
        identity,
        profile,
    })
}

/// Host-owned scorer authority. Implementations must use an explicitly permitted
/// registry and return its validated result for exactly this authenticated input.
/// No dynamic code, subprocess, or network is loaded by the verifier itself.
pub trait ScorerReplay {
    fn replay(&self, request: &Value) -> Result<Value>;
}
pub struct NativeVerification {
    pub payload: Value,
    pub status: u8,
    pub receipt_written: bool,
    pub verified_report: Option<Value>,
}
pub fn verify(config: &Value) -> Result<Value> {
    Ok(verify_transaction(config, None)?.payload)
}
pub fn verify_with_scorer(config: &Value, scorer: Option<&dyn ScorerReplay>) -> Result<Value> {
    Ok(verify_transaction(config, scorer)?.payload)
}
pub fn verify_transaction(
    config: &Value,
    scorer: Option<&dyn ScorerReplay>,
) -> Result<NativeVerification> {
    verify_transaction_inner(config, scorer, None, None)
}
/// Verify against immutable caller-owned trust bytes. JSON cannot supply these overrides.
pub fn verify_with_inputs(
    config: &Value,
    scorer: Option<&dyn ScorerReplay>,
    policy_bytes: &[u8],
    key_bytes: &[u8],
) -> Result<NativeVerification> {
    verify_transaction_inner(config, scorer, Some(policy_bytes), Some(key_bytes))
}
fn verify_transaction_inner(
    config: &Value,
    scorer: Option<&dyn ScorerReplay>,
    policy: Option<&[u8]>,
    key: Option<&[u8]>,
) -> Result<NativeVerification> {
    let input = inputs(config, policy, key)?;
    let snapshot = fs::PackSnapshot::capture(
        &input.pack,
        fs::SnapshotLimits {
            max_entries: 256,
            max_files: 128,
            max_file_bytes: MAX_BYTES,
            max_total_bytes: 512 * 1024 * 1024,
        },
    )
    .map_err(|_| {
        EvidenceError::incomplete("verification result manifest digest must be a sha256:... digest")
    })?;
    let manifest_raw = raw(&snapshot, "manifest.json", 256 * 1024).map_err(|_| {
        EvidenceError::incomplete("verification result manifest digest must be a sha256:... digest")
    })?;
    let mut manifest = Value::Null;
    let mut signer = Value::Null;
    let mut request_digest = Value::Null;
    let mut report = None;
    let mut observations = vec![];
    let mut semantic_errors = vec![];
    let checked = (|| {
        manifest = object(manifest_raw, "manifest.json")?;
        if manifest["format"] != PACK_FORMAT {
            return Err(EvidenceError::contract(
                "unsupported manifest format; expected 'invarlock/evidence-pack-v1'",
            ));
        }
        validate_schema(&manifest, "evidence_pack")
            .map_err(|e| EvidenceError::contract(e.to_string()))?;
        if canonical(&manifest)? != manifest_raw {
            return Err(EvidenceError::contract(
                "manifest.json is not canonical JSON",
            ));
        }
        if let Ok(request_raw) = raw(&snapshot, "request.json", MAX_BYTES) {
            if object(request_raw, "normalized request").is_ok() {
                request_digest = json!(sha256_digest(request_raw));
            }
        }
        if let Ok(signature_raw) = raw(&snapshot, "manifest.signature.json", 64 * 1024) {
            if let Ok(bundle) = object(signature_raw, "manifest signature") {
                if let Ok(key) =
                    crypto::public_key(bundle["public_key"]["value"].as_str().unwrap_or(""))
                {
                    signer = json!(crypto::fingerprint(&key));
                }
            }
        }
        signer = json!(authenticate(
            &snapshot,
            &manifest,
            manifest_raw,
            &input.anchors
        )?);
        checksums(&snapshot, &manifest)?;
        let mut identities = BTreeMap::new();
        for role in ROLES {
            let path = format!("inputs/{role}.json");
            let bytes = raw(&snapshot, &path, 64 * 1024)?;
            let identity = object(bytes, &format!("{role} identity"))?;
            let fields = identity.as_object().expect("checked");
            if !["format", "role", "digest"]
                .iter()
                .all(|k| fields.contains_key(*k))
                || fields.keys().any(|k| {
                    !["format", "role", "digest", "locator", "media_type"].contains(&k.as_str())
                })
            {
                return Err(integrity(format!("{role} identity fields are invalid")));
            }
            if identity["format"] != "invarlock/evidence-input-identity-v1"
                || identity["role"] != role
                || !identity["digest"].as_str().is_some_and(is_digest)
            {
                return Err(integrity(format!("{role} identity is invalid")));
            }
            if canonical(&identity)? != bytes {
                return Err(integrity(format!("{role} identity is not canonical JSON")));
            }
            if manifest["inputs"][role]["path"] != path
                || manifest["inputs"][role]["digest"] != sha256_digest(bytes)
                || manifest["inputs"][role]["material_digest"] != identity["digest"]
            {
                return Err(integrity(format!(
                    "{role} identity digest does not match manifest"
                )));
            }
            identities.insert(role, identity);
        }
        let mut loaded = BTreeMap::new();
        for (role, path) in EVIDENCE {
            let bytes = raw(&snapshot, path, MAX_BYTES)?;
            if manifest["evidence"][role]["path"] != path
                || manifest["evidence"][role]["digest"] != sha256_digest(bytes)
            {
                return Err(integrity(format!("{role} digest does not match manifest")));
            }
            loaded.insert(role, bytes);
        }
        let request = object(loaded["request"], "normalized request")?;
        if canonical(&request)? != loaded["request"] {
            return Err(integrity("normalized request is not canonical JSON"));
        }
        request_digest = json!(sha256_digest(loaded["request"]));
        if input
            .anchors
            .get("request_digest")
            .is_some_and(|v| v != &request_digest)
        {
            semantic_errors
                .push("normalized request does not match independent request anchor".into());
        }
        if let Some(error) = invarlock_native_contracts::first_schema_error(
            &request,
            "evaluation_request",
            invarlock_native_contracts::SchemaErrorOrder::Instance,
        )? {
            let path = if error.path.is_empty() {
                "<root>".into()
            } else {
                error.path.join(".")
            };
            semantic_errors.push(format!(
                "normalized request schema failed at {path}: {}",
                error.message
            ));
        }
        if request["comparison"].get("scorer_extension").is_none()
            && !matches!(
                request["comparison"]["metric"].as_str(),
                Some("exact_match" | "normalized_nll_per_utf8_byte")
            )
        {
            return Err(integrity("request comparison.metric is unsupported"));
        }
        if !semantic_errors.is_empty() {
            return Ok(());
        }
        let schedule = object(loaded["schedule"], "runtime behavioral schedule")?;
        replay::schedule(&schedule, loaded["schedule"], &request)?;
        let schedule_digest = sha256_digest(loaded["schedule"]);
        for side in ["baseline", "subject"] {
            if identities[side]["digest"] != input.anchors["artifact_digests"][side] {
                return Err(integrity(format!(
                    "embedded {side} artifact identity does not match caller artifact anchor"
                )));
            }
            if identities[format!("{side}_runtime").as_str()]["digest"]
                != input.anchors["runtime_digests"][side]
            {
                return Err(integrity(format!(
                    "embedded {side} runtime identity does not match caller runtime anchor"
                )));
            }
        }
        if identities["dataset"]["digest"] != schedule_digest {
            return Err(integrity(
                "dataset identity does not match the canonical schedule",
            ));
        }
        if identities["dataset"]["digest"] != input.anchors["schedule_digest"] {
            return Err(integrity(
                "embedded canonical schedule identity does not match caller schedule anchor",
            ));
        }
        if identities["policy"]["digest"] != input.anchors["policy_digest"] {
            return Err(integrity(
                "embedded policy identity does not match caller policy anchor",
            ));
        }
        observations = replay::observations(&snapshot, &manifest, &request, &identities)?;
        let derived = replay::derive(
            &loaded,
            &request,
            &schedule,
            &identities,
            &input.anchors,
            scorer,
        )?;
        semantic_errors.extend(replay::runtime_checks(&loaded, &input.anchors));
        let paired_raw = raw(&snapshot, "records/paired-records.json", MAX_BYTES)?;
        let paired = object(paired_raw, "paired records")?;
        if canonical(&paired)? != paired_raw {
            semantic_errors.push("paired records are not canonical JSON".into());
        }
        if manifest["paired_records"]["digest"] != sha256_digest(paired_raw) {
            semantic_errors.push("paired records digest does not match manifest".into());
        }
        if paired["records"].as_array().map(Vec::len)
            != manifest["paired_records"]["count"]
                .as_u64()
                .map(|n| n as usize)
        {
            semantic_errors.push("paired records count does not match manifest".into());
        }
        if !json_equal(&paired, &derived) {
            semantic_errors.push(
                "paired records do not match scores derived from provider observations".into(),
            );
        }
        let stored = object(loaded["evaluation_report"], "comparison report")?;
        if canonical(&stored)? != loaded["evaluation_report"] {
            semantic_errors.push("comparison report is not canonical JSON".into());
        }
        if !stored["format"].is_string() {
            return Err(integrity("comparison report format is invalid"));
        }
        let expected = invarlock_native_core::build_comparison_report(
            manifest["comparison_id"].as_str().unwrap_or(""),
            &derived,
            &input.policy,
            input.anchors["policy_digest"].as_str().unwrap_or(""),
            stored["format"].as_str().unwrap_or(""),
        )
        .map_err(|e| integrity(e.to_string()))?;
        if !json_equal(&expected, &stored) {
            semantic_errors.push("comparison report does not match verifier replay".into());
        } else {
            report = Some(expected);
        }
        Ok(())
    })();
    let source_check = snapshot
        .verify_source()
        .map_err(|e| integrity(e.to_string()));
    let mut errors = semantic_errors;
    if let Err(e) = &checked {
        errors.push(e.message.clone());
    }
    if let Err(e) = source_check {
        errors.push(e.message);
    }
    let integrity_ok = errors.is_empty();
    let verdict = report
        .as_ref()
        .map(|v| v["verdict"].clone())
        .unwrap_or(Value::Null);
    let ok = integrity_ok && verdict != "fail";
    let status = checked
        .as_ref()
        .err()
        .map(|e| e.exit_code)
        .unwrap_or(if !integrity_ok {
            6
        } else if ok {
            0
        } else {
            7
        });
    let fingerprint = crypto::fingerprint(&input.key.verifying_key());
    let mut payload = json!({"format_version":"invarlock/evidence-pack-verify-v1","pack":input.pack.file_name().unwrap_or_default().to_string_lossy(),"pack_format":PACK_FORMAT,"comparison_id":manifest.get("comparison_id").unwrap_or(&Value::Null),"request_digest":request_digest,"ok":ok,"integrity_ok":integrity_ok,"reports_verified":integrity_ok,"verification_scope":if integrity_ok{"paired_comparison"}else{"not_verified"},"assurance_status":if integrity_ok{"verified"}else{"not_verified"},"policy_verdict":verdict,"observations":observations,"authenticity":if signer==input.anchors["signer_fingerprint"]{"pinned"}else{"mismatch"},"signer_fingerprint":signer,"anchors":input.anchors,"warnings":[],"errors":errors});
    receipt::write(&input, &payload, manifest_raw, status)?;
    payload["signed_receipt"] = json!(input
        .receipt
        .file_name()
        .unwrap_or_default()
        .to_string_lossy());
    payload["pack_manifest_digest"] = json!(sha256_digest(manifest_raw));
    payload["verifier_identity"] = json!(input.identity);
    payload["verifier_fingerprint"] = json!(fingerprint);
    if !input.profile.is_null() {
        payload["trust_profile_digest"] = input.profile;
    }
    // The result retains the failed receipt and full public verdict. The host
    // selects process exit status from these fields without discarding evidence.
    Ok(NativeVerification {
        payload,
        status,
        receipt_written: true,
        verified_report: report,
    })
}
pub(crate) fn authenticate(
    snapshot: &fs::PackSnapshot,
    manifest: &Value,
    bytes: &[u8],
    anchors: &Value,
) -> Result<String> {
    if !snapshot.entries().contains_key("manifest.signature.json") {
        return Err(integrity("manifest.signature.json missing"));
    }
    let bundle = object(
        raw(snapshot, "manifest.signature.json", 64 * 1024)?,
        "manifest.signature.json",
    )?;
    if !exact_keys(
        &bundle,
        &[
            "format",
            "algorithm",
            "signing_key_fingerprint",
            "public_key",
            "signature",
        ],
    ) {
        return Err(integrity("manifest.signature.json fields are invalid"));
    }
    if bundle["format"] != "invarlock/evidence-pack-signature-v1"
        || bundle["algorithm"] != "ed25519"
    {
        return Err(integrity(
            "manifest.signature.json format or algorithm is invalid",
        ));
    }
    for (field, encoding) in [("public_key", "pem"), ("signature", "base64")] {
        if !exact_keys(&bundle[field], &["encoding", "value"])
            || bundle[field]["encoding"] != encoding
            || !bundle[field]["value"].is_string()
        {
            return Err(integrity(format!(
                "manifest.signature.json {field} is invalid"
            )));
        }
    }
    let key = crypto::public_key(bundle["public_key"]["value"].as_str().unwrap())?;
    let fingerprint = crypto::fingerprint(&key);
    if bundle["signing_key_fingerprint"] != fingerprint
        || manifest["signing_key_fingerprint"] != fingerprint
    {
        return Err(integrity(
            "manifest signing key fingerprint does not match public key",
        ));
    }
    if anchors["signer_fingerprint"] != fingerprint {
        return Err(integrity(format!(
            "manifest signature signer mismatch: expected one of [{}], got {fingerprint}",
            anchors["signer_fingerprint"].as_str().unwrap_or("")
        )));
    }
    crypto::verify(&key, bytes, bundle["signature"]["value"].as_str().unwrap())
        .map_err(|_| integrity("manifest signature verification failed"))?;
    Ok(fingerprint)
}
pub(crate) fn checksums(snapshot: &fs::PackSnapshot, manifest: &Value) -> Result<()> {
    let bytes = raw(snapshot, "checksums.sha256", 1024 * 1024)?;
    if manifest["checksums_sha256_digest"] != hex(bytes) {
        return Err(integrity("checksums.sha256 digest mismatch"));
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| integrity("checksums.sha256 is not UTF-8"))?;
    let mut seen = BTreeSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let b = line.as_bytes();
        if b.len() < 67
            || b[64] != b' '
            || !matches!(b[65], b' ' | b'*')
            || !b[..64].iter().all(u8::is_ascii_hexdigit)
        {
            return Err(integrity(format!(
                "checksums.sha256 line {} is not a valid sha256 entry",
                i + 1
            )));
        }
        let rel = line[66..].replace('\\', "/");
        let rel = rel.trim_start_matches("./");
        if rel.is_empty()
            || rel
                .split('/')
                .any(|s| s.is_empty() || s == "." || s == "..")
            || !seen.insert(rel.to_owned())
        {
            return Err(integrity(format!(
                "checksums.sha256 line {} has an unsafe or duplicate path",
                i + 1
            )));
        }
        let entry = snapshot
            .entries()
            .get(rel)
            .ok_or_else(|| integrity(format!("checksums entry missing from pack: {rel}")))?;
        if entry.sha256() != line[..64].to_lowercase() {
            return Err(integrity(format!("checksum mismatch for {rel}")));
        }
    }
    let extras: Vec<_> = snapshot
        .entries()
        .keys()
        .filter(|p| {
            !seen.contains(p.as_str())
                && ![
                    "checksums.sha256",
                    "manifest.json",
                    "manifest.signature.json",
                ]
                .contains(&p.as_str())
        })
        .cloned()
        .collect();
    if !extras.is_empty() {
        return Err(integrity(format!(
            "Pack contains extra files not covered by checksums.sha256: {}",
            extras.join(", ")
        )));
    }
    Ok(())
}
fn json_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
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
