//! Independent retained judge replay, recipient acceptance and signed transport.
use crate::{
    bounded, canonical, crypto, digest, is_digest, required_string, EvidenceError, Result,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use invarlock_native_contracts::{
    canonical_json_no_newline, parse_json, sha256_digest, validate_schema,
};
use invarlock_native_core as analysis;
use invarlock_native_filesystem as fs;
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const SCOPE: &str = "bounded-judge-fixed-benchmark-v1";
const FORMAT: &str = "invarlock/judge-measurement-evidence-v1";
const RESULT_FORMAT: &str = "invarlock/judge-verification-result-v1";
const RECEIPT_FORMAT: &str = "invarlock/judge-measurement-verification-receipt-v1";
const RECEIPT_DOMAIN: &[u8] = b"invarlock:judge-measurement-verification-receipt:v1\0";
const ARTIFACTS: [&str; 7] = [
    "plan.json",
    "measurements.json",
    "baseline_run.json",
    "subject_run.json",
    "case_set.json",
    "analysis_policy.json",
    "analysis_result.json",
];
fn limit(name: &str) -> usize {
    match name {
        "plan.json" => 64 * 1024 * 1024,
        "measurements.json" => 384 * 1024 * 1024,
        "analysis_policy.json" | "envelope.json" => 65536,
        "analysis_result.json" => 1024 * 1024,
        _ => 128 * 1024 * 1024,
    }
}
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn io_error(e: impl std::fmt::Display) -> EvidenceError {
    error(e.to_string())
}
fn object(raw: &[u8], label: &str) -> Result<Value> {
    let value = parse_json(raw, label)?;
    if !value.is_object() {
        return Err(error(format!("{label} must contain a JSON object")));
    }
    Ok(value)
}
pub fn object_digest(value: &Value) -> Result<String> {
    Ok(sha256_digest(&canonical_json_no_newline(value)?)
        .trim_start_matches("sha256:")
        .to_owned())
}
fn envelope_signed(envelope: &Value) -> Result<Vec<u8>> {
    let mut statement = envelope.clone();
    statement
        .as_object_mut()
        .ok_or_else(|| error("judge envelope must be an object"))?
        .remove("signature");
    let mut bytes = FORMAT.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend(canonical_json_no_newline(&statement)?);
    Ok(bytes)
}
fn identity(value: &Value, label: &str) -> Result<String> {
    let text = value.as_str().unwrap_or("").trim();
    if text.is_empty()
        || text.len() > 256
        || !text.as_bytes()[0].is_ascii_alphanumeric()
        || !text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:/-".contains(&b))
    {
        return Err(error(format!("{label} is invalid")));
    }
    Ok(text.to_string())
}
fn normalized_digest(value: &Value, label: &str, prefix: bool) -> Result<String> {
    let text = value.as_str().unwrap_or("").trim().to_lowercase();
    let valid = if prefix {
        is_digest(&text)
    } else {
        text.len() == 64
            && text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if !valid {
        return Err(error(format!(
            "{label} must be {}",
            if prefix {
                "a sha256:... fingerprint"
            } else {
                "a lowercase SHA-256 digest"
            }
        )));
    }
    Ok(text)
}
fn external(path: &Path, evidence: &Path, label: &str) -> Result<()> {
    crate::captured::require_outside(path, evidence).map_err(|_| {
        error(format!(
            "{label} must be supplied outside submitted judge evidence"
        ))
    })
}
fn validate_envelope(value: &Value) -> Result<()> {
    validate_schema(value, "judge_measurement_evidence").map_err(|e| {
        error(format!(
            "judge evidence envelope is invalid: {}",
            bounded(&e.to_string())
        ))
    })
}
pub fn load_policy(path: &Path, evidence: &Path) -> Result<Value> {
    external(path, evidence, "recipient policy")?;
    let value = object(
        &fs::read_file(path, 65536).map_err(io_error)?,
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("recipient policy"),
    )?;
    validate_schema(&value, "judge_measurement_recipient_policy").map_err(|e| {
        error(format!(
            "judge recipient policy is invalid: {}",
            bounded(&e.to_string())
        ))
    })?;
    Ok(value)
}
pub fn case_set_digest(value: &Value) -> Result<String> {
    validate_schema(value, "case_set")?;
    let mut rows = value["cases"]
        .as_array()
        .ok_or_else(|| error("case set requires cases"))?
        .clone();
    rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    if rows.windows(2).any(|w| w[0]["id"] == w[1]["id"]) {
        return Err(error("planned case set has duplicate case IDs"));
    }
    let mut ordered = value.clone();
    ordered["cases"] = Value::Array(rows);
    digest(&ordered)
}
fn evaluated_subject(run: &Value) -> Result<String> {
    if let Some(service) = run.get("service_identity") {
        digest(service)
    } else {
        run["artifact_digest"]
            .as_str()
            .filter(|s| is_digest(s))
            .map(str::to_string)
            .ok_or_else(|| error("run requires artifact or hosted service identity"))
    }
}
fn bindings(artifacts: &Map<String, Value>) -> Result<Value> {
    let mut bindings = json!({"baseline_run_sha256":digest(&artifacts["baseline_run.json"])? ,
        "subject_run_sha256":digest(&artifacts["subject_run.json"])? ,
        "case_set_sha256":case_set_digest(&artifacts["case_set.json"])? ,
        "plan_sha256":object_digest(&artifacts["plan.json"])? ,
        "measurements_sha256":object_digest(&artifacts["measurements.json"])? ,
        "analysis_policy_sha256":object_digest(&artifacts["analysis_policy.json"])? ,
        "analysis_result_sha256":object_digest(&artifacts["analysis_result.json"])?});
    if let Some(capture) = artifacts.get("native_capture.json") {
        bindings["native_capture_sha256"] = json!(object_digest(capture)?);
    }
    Ok(bindings)
}
fn analyze(artifacts: &Map<String, Value>) -> Result<Value> {
    let native = ["baseline_run.json", "subject_run.json"]
        .iter()
        .any(|side| artifacts[*side]["source"]["name"] == "invarlock-native-judge");
    if let Some(capture) = artifacts.get("native_capture.json") {
        let (baseline, subject) = crate::judge_native::validate_capture(capture)?;
        for (name, expected) in [
            ("baseline_run.json", &baseline),
            ("subject_run.json", &subject),
        ] {
            if canonical_json_no_newline(&artifacts[name])? != canonical_json_no_newline(expected)?
            {
                return Err(error(
                    "native capture differs from the frozen judge answers",
                ));
            }
        }
        let (plan, policy) =
            crate::judge_native::finalize_plan(&capture["recipe"], &baseline, &subject)?;
        if canonical_json_no_newline(&plan)? != canonical_json_no_newline(&artifacts["plan.json"])?
        {
            return Err(error("native judge plan differs from the captured recipe"));
        }
        if canonical_json_no_newline(&policy)?
            != canonical_json_no_newline(&artifacts["analysis_policy.json"])?
        {
            return Err(error(
                "native judge analysis policy differs from the captured recipe",
            ));
        }
    } else if native {
        return Err(error("native judge answers require their original capture"));
    }
    analysis::analyze_measurements(
        &artifacts["plan.json"],
        &artifacts["measurements.json"],
        &artifacts["analysis_policy.json"],
        &artifacts["baseline_run.json"],
        &artifacts["subject_run.json"],
    )
    .map_err(|e| error(e.to_string()))
}
pub fn replay(path: &Path, expected_envelope: Option<&str>) -> Result<Value> {
    let snapshot = replay_snapshot(path, expected_envelope)?;
    Ok(json!({"envelope":snapshot["envelope"],"analysis_result":snapshot["analysis_result"]}))
}
/// One independently replayed retained snapshot for presentation and composition.
pub fn replay_snapshot(path: &Path, expected_envelope: Option<&str>) -> Result<Value> {
    let root = fs::PinnedDirectory::open(path, false).map_err(io_error)?;
    let (raw, envelope_identity) = root
        .read_regular(Path::new("envelope.json"), 65536)
        .map_err(io_error)?;
    let envelope = object(&raw, "envelope.json")?;
    validate_envelope(&envelope)?;
    if expected_envelope
        .is_some_and(|expected| object_digest(&envelope).map_or(true, |actual| actual != expected))
    {
        return Err(error("judge evidence envelope changed before replay"));
    }
    let mut names = ARTIFACTS.to_vec();
    if envelope["bindings"].get("native_capture_sha256").is_some() {
        names.push("native_capture.json");
    } else if std::fs::symlink_metadata(path.join("native_capture.json")).is_ok() {
        return Err(error("native capture is not bound by the judge envelope"));
    }
    let mut identities = BTreeMap::new();
    let mut artifacts = Map::new();
    for name in names {
        let (bytes, identity) = root
            .read_regular(Path::new(name), limit(name))
            .map_err(io_error)?;
        artifacts.insert(name.to_string(), object(&bytes, name)?);
        identities.insert(name, identity);
    }
    let actual = bindings(&artifacts)?;
    if envelope["bindings"] != actual {
        return Err(error(
            "judge evidence artifact bindings do not match the envelope",
        ));
    }
    if actual["case_set_sha256"] != artifacts["plan.json"]["case_set_sha256"] {
        return Err(error(
            "judge evidence case set does not match the approved plan",
        ));
    }
    if envelope["intended_subject"] != evaluated_subject(&artifacts["subject_run.json"])? {
        return Err(error(
            "judge evidence intended subject does not match the frozen run",
        ));
    }
    let analysis = analyze(&artifacts)?;
    if canonical_json_no_newline(&analysis)?
        != canonical_json_no_newline(&artifacts["analysis_result.json"])?
    {
        return Err(error(
            "judge evidence analysis result differs from independent replay",
        ));
    }
    root.verify_regular(Path::new("envelope.json"), &envelope_identity)
        .map_err(io_error)?;
    for (name, identity) in identities {
        root.verify_regular(Path::new(name), &identity)
            .map_err(io_error)?;
    }
    root.verify().map_err(io_error)?;
    let artifacts = artifacts
        .into_iter()
        .map(|(name, value)| (name.trim_end_matches(".json").to_owned(), value))
        .collect::<Map<_, _>>();
    Ok(json!({"envelope":envelope,"analysis_result":analysis,"artifacts":artifacts}))
}
fn empty_result() -> Value {
    json!({"verified":false,"authenticated":false,"replayed":false,"accepted":false,"decision":null,
        "intended_subject":null,"signer_identity":null,"signer_public_key_sha256":null,"envelope_sha256":null,
        "recipient_policy_sha256":null,"bindings":null,"errors":[],"format":RESULT_FORMAT,"decision_scope":SCOPE})
}
pub fn verify(config: &Value) -> Result<Value> {
    let mut result = empty_result();
    let checked: Result<()> = (|| {
        let evidence = Path::new(required_string(config, "evidence_path")?);
        let policy = load_policy(
            Path::new(required_string(config, "recipient_policy_path")?),
            evidence,
        )?;
        result["recipient_policy_sha256"] = json!(object_digest(&policy)?);
        let envelope = object(
            &fs::read_file(&evidence.join("envelope.json"), 65536).map_err(io_error)?,
            "envelope.json",
        )?;
        validate_envelope(&envelope)?;
        result["envelope_sha256"] = json!(object_digest(&envelope)?);
        result["intended_subject"] = envelope["intended_subject"].clone();
        result["bindings"] = envelope["bindings"].clone();
        let signer = &envelope["signer"];
        if signer.is_null() || envelope["signature"].is_null() {
            return Err(error(
                "unsigned judge evidence cannot be independently accepted",
            ));
        }
        result["signer_identity"] = signer["identity"].clone();
        result["signer_public_key_sha256"] = signer["public_key_sha256"].clone();
        let trusted = &policy["trusted_signer"];
        if signer["identity"] != trusted["identity"]
            || signer["public_key_sha256"] != trusted["public_key_sha256"]
        {
            return Err(error(
                "judge evidence signer is not authorized by recipient policy",
            ));
        }
        let raw =
            crate::decode_base64(signer["public_key"].as_str().unwrap_or("")).map_err(io_error)?;
        let public = VerifyingKey::from_bytes(
            raw.as_slice()
                .try_into()
                .map_err(|_| error("Ed25519 public key must contain 32 bytes"))?,
        )
        .map_err(io_error)?;
        let fingerprint = crypto::fingerprint(&public);
        if trusted["public_key_sha256"] != fingerprint {
            return Err(error(
                "judge evidence signer key does not match recipient fingerprint",
            ));
        }
        if let Some(keys) = config.get("trusted_public_keys").filter(|v| !v.is_null()) {
            let key = keys
                .get(trusted["identity"].as_str().unwrap_or(""))
                .ok_or_else(|| {
                    error("judge signer differs from the additional recipient key anchor")
                })?;
            let path = PathBuf::from(
                key.as_str()
                    .ok_or_else(|| error("trusted signer key must be an explicit path"))?,
            );
            external(&path, evidence, "trusted signer key")?;
            let raw = fs::read_file(&path, 65536).map_err(io_error)?;
            let extra = if raw.len() == 32 {
                VerifyingKey::from_bytes(raw.as_slice().try_into().unwrap()).map_err(io_error)?
            } else {
                crypto::public_key(std::str::from_utf8(&raw).map_err(io_error)?)?
            };
            if crypto::fingerprint(&extra) != fingerprint {
                return Err(error(
                    "judge signer differs from the additional recipient key anchor",
                ));
            }
        }
        crypto::verify(
            &public,
            &envelope_signed(&envelope)?,
            envelope["signature"].as_str().unwrap_or(""),
        )
        .map_err(|_| error("judge evidence signature is invalid"))?;
        result["authenticated"] = json!(true);
        if envelope["decision_scope"] != policy["decision_scope"] {
            return Err(error("judge evidence scope differs from recipient policy"));
        }
        if envelope["intended_subject"] != policy["intended_subject"] {
            return Err(error(
                "judge evidence intended subject differs from recipient policy",
            ));
        }
        for (name, expected) in policy["bindings"]
            .as_object()
            .ok_or_else(|| error("recipient bindings must be an object"))?
        {
            if envelope["bindings"].get(name) != Some(expected) {
                return Err(error(format!(
                    "judge evidence {name} differs from recipient policy"
                )));
            }
        }
        let replay = replay(evidence, result["envelope_sha256"].as_str())?;
        let analysis = &replay["analysis_result"];
        result["replayed"] = json!(true);
        result["decision"] = analysis["decision"].clone();
        if analysis["policy"]["metric_name"] != policy["required_metric_name"] {
            return Err(error(
                "judge metric differs from the recipient's required metric",
            ));
        }
        result["verified"] = json!(true);
        if analysis["policy"]["required"] != true {
            return Err(error(
                "advisory judge evidence cannot satisfy a required recipient decision",
            ));
        }
        if analysis["decision"] != "pass" {
            return Err(error(format!(
                "required judge decision is {}",
                analysis["decision"].as_str().unwrap_or("None")
            )));
        }
        result["accepted"] = json!(true);
        Ok(())
    })();
    if let Err(failure) = checked {
        result["accepted"] = json!(false);
        result["errors"] = json!([failure.message.chars().take(1000).collect::<String>()]);
    }
    validate_schema(&result, "judge_verification_result")?;
    Ok(result)
}
fn signed_receipt_bytes(statement: &Value) -> Result<Vec<u8>> {
    let mut bytes = RECEIPT_DOMAIN.to_vec();
    bytes.extend(canonical_json_no_newline(statement)?);
    Ok(bytes)
}
pub fn write_receipt(config: &Value, result: &Value) -> Result<String> {
    write_receipt_with_key(config, result, None)
}
pub fn write_receipt_with_key(
    config: &Value,
    result: &Value,
    key_bytes: Option<&[u8]>,
) -> Result<String> {
    let evidence = Path::new(required_string(config, "evidence_path")?);
    let receipt = Path::new(required_string(config, "receipt_path")?);
    external(receipt, evidence, "signed judge receipt")?;
    let policy = load_policy(
        Path::new(required_string(config, "recipient_policy_path")?),
        evidence,
    )?;
    let policy_digest = object_digest(&policy)?;
    validate_schema(result, "judge_verification_result")?;
    if result["recipient_policy_sha256"] != policy_digest {
        return Err(error(
            "judge verification result does not bind the supplied recipient policy",
        ));
    }
    let key_path = Path::new(required_string(config, "verifier_signing_key_path")?);
    external(key_path, evidence, "verifier signing key")?;
    let disk;
    let raw = match key_bytes {
        Some(raw) => raw,
        None => {
            disk = fs::read_file(key_path, 65536).map_err(io_error)?;
            &disk
        }
    };
    let key = crypto::private_key(raw)?;
    let fingerprint = crypto::fingerprint(&key.verifying_key());
    let statement = json!({"format":RECEIPT_FORMAT,"kind":"judge","decision_scope":SCOPE,"recipient_policy_sha256":policy_digest,
        "verifier":{"identity":identity(&config["verifier_identity"],"verifier identity")?,"signing_key_fingerprint":fingerprint},"result":result});
    let signature = crypto::signature_value(&key, &signed_receipt_bytes(&statement)?);
    let receipt_value = json!({"statement":statement,"signature":{"format":"invarlock/judge-measurement-verification-receipt-signature-v1",
        "algorithm":"ed25519","public_key":{"encoding":"pem","value":crypto::public_pem(&key.verifying_key())?},"value":signature}});
    validate_schema(&receipt_value, "judge_measurement_verification_receipt")?;
    let raw = canonical(&receipt_value)?;
    if raw.len() > 1024 * 1024 {
        return Err(error("signed judge receipt exceeds the 1048576-byte limit"));
    }
    fs::write_file_no_replace(receipt, &raw, 0o600, true).map_err(io_error)?;
    Ok(fingerprint)
}
pub fn verify_receipt(config: &Value) -> Result<Value> {
    let mut signed = false;
    let checked: Result<(Value, String)> = (|| {
        let raw = fs::read_file(
            Path::new(required_string(config, "receipt_path")?),
            1024 * 1024,
        )
        .map_err(io_error)?;
        let receipt = object(&raw, "judge receipt")?;
        signed = receipt["statement"].is_object() && receipt["signature"].is_object();
        validate_schema(&receipt, "judge_measurement_verification_receipt")?;
        let statement = &receipt["statement"];
        let signature = &receipt["signature"];
        let expected_identity = identity(
            &config["expected_verifier_identity"],
            "expected verifier identity",
        )?;
        let expected_fingerprint = normalized_digest(
            &config["expected_verifier_fingerprint"],
            "expected verifier fingerprint",
            true,
        )?;
        let expected_policy = normalized_digest(
            &config["expected_recipient_policy_sha256"],
            "expected recipient policy digest",
            false,
        )?;
        if statement["recipient_policy_sha256"] != expected_policy {
            return Err(error(
                "judge receipt recipient policy does not match caller expectation",
            ));
        }
        if statement["verifier"]["identity"] != expected_identity {
            return Err(error(
                "judge receipt verifier identity does not match caller expectation",
            ));
        }
        let public = crypto::public_key(signature["public_key"]["value"].as_str().unwrap_or(""))?;
        let fingerprint = crypto::fingerprint(&public);
        if fingerprint != expected_fingerprint
            || statement["verifier"]["signing_key_fingerprint"] != fingerprint
        {
            return Err(error("judge receipt verifier key is unauthorized"));
        }
        crypto::verify(
            &public,
            &signed_receipt_bytes(statement)?,
            signature["value"].as_str().unwrap_or(""),
        )
        .map_err(|_| error("judge receipt signature is invalid"))?;
        if statement["recipient_policy_sha256"] != statement["result"]["recipient_policy_sha256"] {
            return Err(error(
                "judge receipt policy digest differs from its verification result",
            ));
        }
        validate_schema(&statement["result"], "judge_verification_result")?;
        Ok((statement.clone(), fingerprint))
    })();
    match checked {
        Ok((statement, fingerprint)) => Ok(
            json!({"ok":true,"signed":signed,"result":statement["result"],"statement":statement,"verifier_fingerprint":fingerprint,"errors":[]}),
        ),
        Err(failure) => Ok(
            json!({"ok":false,"signed":signed,"result":null,"statement":null,"verifier_fingerprint":null,"errors":[failure.message.chars().take(1000).collect::<String>()]}),
        ),
    }
}
pub fn publish(config: &Value) -> Result<Value> {
    publish_with_key(config, None)
}
pub fn publish_with_key(config: &Value, key_bytes: Option<&[u8]>) -> Result<Value> {
    let mut artifacts = Map::new();
    for (field, name) in [
        ("plan", "plan.json"),
        ("measurements", "measurements.json"),
        ("baseline_run", "baseline_run.json"),
        ("subject_run", "subject_run.json"),
        ("analysis_policy", "analysis_policy.json"),
    ] {
        artifacts.insert(name.into(), config[field].clone());
    }
    let rows = config["baseline_run"]["records"]
        .as_array()
        .ok_or_else(|| error("baseline run requires records"))?;
    let cases:Vec<Value>=rows.iter().map(|row|json!({"id":row["id"],"input":row["input"],"expected":row["expected"],"metadata":row["metadata"]})).collect();
    artifacts.insert(
        "case_set.json".into(),
        json!({"format":"invarlock/evaluation-case-set-v1","cases":cases}),
    );
    if let Some(capture) = config.get("native_capture").filter(|c| !c.is_null()) {
        artifacts.insert("native_capture.json".into(), capture.clone());
    }
    let analysis = analyze(&artifacts)?;
    artifacts.insert("analysis_result.json".into(), analysis.clone());
    let mut envelope = json!({"format":FORMAT,"decision_scope":SCOPE,"intended_subject":evaluated_subject(&artifacts["subject_run.json"])? ,
        "bindings":bindings(&artifacts)?,"signature_algorithm":"ed25519","signer":null,"signature":null});
    let key_path = config.get("signing_key_path").and_then(Value::as_str);
    let signer = config.get("signer_identity").filter(|v| !v.is_null());
    if (key_path.is_some() || key_bytes.is_some()) != signer.is_some() {
        return Err(error(
            "signing_key and signer_identity must be supplied together",
        ));
    }
    if key_path.is_some() || key_bytes.is_some() {
        let disk;
        let raw = match key_bytes {
            Some(raw) if raw.len() <= 65536 => raw,
            Some(_) => return Err(error("signing key exceeds its byte limit")),
            None => {
                disk = fs::read_file(Path::new(key_path.unwrap()), 65536).map_err(io_error)?;
                &disk
            }
        };
        let key = crypto::private_key(raw)?;
        envelope["signer"] = json!({"identity":signer,"public_key_sha256":crypto::fingerprint(&key.verifying_key()),"public_key":STANDARD.encode(key.verifying_key().as_bytes())});
        envelope["signature"] = json!(crypto::signature_value(&key, &envelope_signed(&envelope)?));
    }
    validate_envelope(&envelope)?;
    let mut files = BTreeMap::new();
    for (name, value) in artifacts {
        let raw = canonical(&value)?;
        if raw.len() > limit(&name) {
            return Err(error(format!(
                "{name} exceeds the {}-byte size limit",
                limit(&name)
            )));
        }
        files.insert(name, raw);
    }
    files.insert("envelope.json".into(), canonical(&envelope)?);
    let destination = Path::new(required_string(config, "destination")?);
    let mut stage = fs::StagedDirectory::new(destination).map_err(io_error)?;
    for (name, raw) in files {
        stage
            .write(Path::new(&name), &raw, 0o600)
            .map_err(io_error)?;
    }
    stage.commit().map_err(io_error)?;
    Ok(json!({"path":destination,"envelope":envelope,"analysis_result":analysis}))
}

/// Authenticate a stored receipt/result and require a fresh recipient replay.
pub fn verify_stored(config: &Value, signed: bool) -> Result<Value> {
    let evidence = Path::new(required_string(config, "evidence_path")?);
    let stored = Path::new(required_string(
        config,
        if signed {
            "receipt_path"
        } else {
            "stored_result_path"
        },
    )?);
    if let Err(failure) = external(
        stored,
        evidence,
        if signed {
            "signed judge receipt"
        } else {
            "stored judge result"
        },
    ) {
        let mut rejected = empty_result();
        rejected["errors"] = json!([failure.message.chars().take(1000).collect::<String>()]);
        return Ok(rejected);
    }
    let mut actual = verify(config)?;
    let checked: Result<()> = (|| {
        let claimed = if signed {
            let policy = load_policy(
                Path::new(required_string(config, "recipient_policy_path")?),
                evidence,
            )?;
            let mut args = config.clone();
            args["expected_recipient_policy_sha256"] = json!(object_digest(&policy)?);
            let verification = verify_receipt(&args)?;
            if verification["ok"] != true || verification["result"].is_null() {
                let errors = verification["errors"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("; ")
                    })
                    .unwrap_or_default();
                return Err(error(if errors.is_empty() {
                    "judge receipt is not authentic".to_string()
                } else {
                    errors
                }));
            }
            verification["result"].clone()
        } else {
            let value = object(
                &fs::read_file(stored, 65536).map_err(io_error)?,
                stored
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("stored judge result"),
            )?;
            validate_schema(&value, "judge_verification_result")?;
            value
        };
        if object_digest(&claimed)? != object_digest(&actual)? {
            return Err(error(if signed {
                "signed judge receipt differs from fresh recipient verification"
            } else {
                "stored judge result differs from independent recipient verification"
            }));
        }
        Ok(())
    })();
    if let Err(failure) = checked {
        actual["accepted"] = json!(false);
        actual["verified"] = json!(false);
        actual["errors"] = json!([failure.message.chars().take(1000).collect::<String>()]);
    }
    Ok(actual)
}
