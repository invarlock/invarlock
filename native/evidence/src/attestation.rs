//! Offline in-toto/DSSE transport and independently supplied recipient policy.
use crate::{canonical, crypto, EvidenceError, Result};
use base64::{
    engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig, STANDARD},
    Engine,
};
use invarlock_native_contracts::{
    self as contracts, first_schema_error, parse_json, sha256_digest, SchemaErrorOrder,
};
use invarlock_native_filesystem as fs;
const DECODE: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true),
);
use ed25519_dalek::{Signature, VerifyingKey};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::{Component, Path},
    time::{SystemTime, UNIX_EPOCH},
};
const PAYLOAD_TYPE: &str = "application/vnd.in-toto+json";
const PREDICATE_TYPE: &str = "https://invarlock.dev/attestations/acceptance/v2";
const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
const LIMIT: usize = 4 * 1024 * 1024;
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn fields(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("")
}
fn normalized_digest(v: &Value, label: &str) -> Result<String> {
    let s = v.as_str().unwrap_or("").trim().to_lowercase();
    if !crate::is_digest(&s) {
        return Err(err(format!("{label} must be a sha256:... digest")));
    }
    Ok(s)
}
fn identity(v: &Value, label: &str) -> Result<String> {
    let s = v.as_str().unwrap_or("").trim();
    if s.is_empty()
        || s.len() > 256
        || !s.as_bytes()[0].is_ascii_alphanumeric()
        || !s
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._:/-".contains(&c))
    {
        return Err(err(format!("{label} is invalid")));
    }
    Ok(s.into())
}
fn read(path: &Path, label: &str, limit: usize) -> Result<Vec<u8>> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|_| err(format!("{label} is unavailable")))?;
    if metadata.file_type().is_symlink() {
        return Err(err(format!("{label} must not be a symlink")));
    }
    if !metadata.is_file() {
        return Err(err(format!("{label} must be a regular file")));
    }
    if metadata.len() > limit as u64 {
        return Err(err(format!("{label} exceeds the {limit}-byte size limit")));
    }
    fs::read_file(path, limit).map_err(|e| {
        err(match e {
            fs::Error::Changed(_) => format!("{label} changed while being read"),
            _ => format!("{label} could not be opened safely"),
        })
    })
}
fn load(path: &Path, label: &str) -> Result<(Vec<u8>, Value)> {
    let raw = read(path, label, LIMIT)?;
    let value = parse_json(&raw, label)?;
    if !value.is_object() {
        return Err(err(format!("{label} must be a JSON object")));
    }
    Ok((raw, value))
}
fn path<'a>(v: &'a Value, k: &str) -> Result<&'a Path> {
    v[k].as_str()
        .map(Path::new)
        .ok_or_else(|| err(format!("{k} is required")))
}
fn schema(v: &Value, name: &str, prefix: &str) -> Result<()> {
    if let Some(e) = first_schema_error(v, name, SchemaErrorOrder::Schema)? {
        return Err(err(format!("{prefix}{}", e.message)));
    }
    Ok(())
}
fn pae(payload: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "DSSEv1 {} {PAYLOAD_TYPE} {} ",
        PAYLOAD_TYPE.len(),
        payload.len()
    )
    .into_bytes();
    out.extend(payload);
    out
}
fn receipt_format(v: &Value) -> bool {
    matches!(
        v.as_str(),
        Some(
            "invarlock/evidence-verification-receipt-v1"
                | "invarlock/evidence-verification-receipt-v2"
        )
    )
}
fn other_pem_algorithm(pem: &str) -> bool {
    let body = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect::<String>();
    DECODE.decode(body).is_ok_and(|raw| {
        raw.first() == Some(&0x30) && !raw.windows(5).any(|w| w == [6, 3, 43, 101, 112])
    })
}
fn receipt_key(s: &Value) -> Result<VerifyingKey> {
    if !s.is_object() {
        return Err(err("receipt signature is invalid"));
    }
    if !fields(s, &["format", "algorithm", "public_key", "value"]) {
        return Err(err("receipt signature fields are invalid"));
    }
    if s["format"] != "invarlock/evidence-verification-receipt-signature-v1" {
        return Err(err("receipt signature format is invalid"));
    }
    if s["algorithm"] != "ed25519" {
        return Err(err("receipt signature algorithm is invalid"));
    }
    if s["public_key"]["encoding"] != "pem" || !s["public_key"]["value"].is_string() {
        return Err(err("receipt public key is invalid: missing PEM value"));
    }
    if other_pem_algorithm(text(&s["public_key"], "value")) {
        return Err(err("receipt public key must be Ed25519"));
    }
    crypto::public_key(text(&s["public_key"], "value"))
        .map_err(|_| err("receipt public key is invalid: Could not deserialize key data"))
}
fn authenticate_receipt(r: &Value) -> Result<(Value, String, String)> {
    if !fields(r, &["statement", "signature"]) {
        return Err(err("embedded receipt fields are invalid"));
    }
    let s = &r["statement"];
    if !s.is_object() {
        return Err(err("embedded receipt statement is invalid"));
    }
    if !fields(
        s,
        &[
            "format",
            "pack_manifest_digest",
            "anchors",
            "verifier",
            "verdict",
        ],
    ) {
        return Err(err("embedded receipt statement fields are invalid"));
    }
    if !receipt_format(&s["format"]) {
        return Err(err("embedded receipt format is unsupported"));
    }
    let v = &s["verifier"];
    if !fields(
        v,
        &[
            "identity",
            "signing_key_fingerprint",
            "trust_profile_digest",
        ],
    ) {
        return Err(err("embedded receipt verifier is invalid"));
    }
    let id = identity(&v["identity"], "embedded receipt verifier identity")?;
    let recorded = normalized_digest(
        &v["signing_key_fingerprint"],
        "embedded receipt verifier fingerprint",
    )?;
    if !v["trust_profile_digest"].is_null() {
        normalized_digest(
            &v["trust_profile_digest"],
            "embedded receipt trust-profile digest",
        )?;
    }
    let verdict = &s["verdict"];
    if !fields(
        verdict,
        &[
            "ok",
            "integrity_ok",
            "policy_verdict",
            "verification_status",
        ],
    ) {
        return Err(err("embedded receipt verdict is invalid"));
    }
    let status = &verdict["verification_status"];
    if !verdict["ok"].is_boolean()
        || !verdict["integrity_ok"].is_boolean()
        || !(verdict["policy_verdict"].is_null()
            || [json!("pass"), json!("fail")].contains(&verdict["policy_verdict"]))
        || !status.is_number()
        || status.to_string().contains(['.', 'e', 'E', '-'])
    {
        return Err(err("embedded receipt verdict fields are invalid"));
    }
    if verdict["ok"] == true
        && (verdict["integrity_ok"] != true || verdict["policy_verdict"] != "pass" || *status != 0)
    {
        return Err(err("embedded receipt successful verdict is inconsistent"));
    }
    let key = receipt_key(&r["signature"])?;
    let fingerprint = crypto::fingerprint(&key);
    if fingerprint != recorded {
        return Err(err(
            "embedded receipt fingerprint does not match its public key",
        ));
    }
    let signature = DECODE
        .decode(text(&r["signature"], "value"))
        .ok()
        .and_then(|b| Signature::from_slice(&b).ok())
        .ok_or_else(|| err("embedded receipt signature verification failed"))?;
    key.verify_strict(&canonical(s)?, &signature)
        .map_err(|_| err("embedded receipt signature verification failed"))?;
    Ok((s.clone(), id, fingerprint))
}
fn bound(evidence: &Path, reference: &Value, label: &str) -> Result<(Vec<u8>, Value)> {
    if !fields(reference, &["path", "digest"]) {
        return Err(err(format!("{label} reference is invalid")));
    }
    let relative = reference["path"]
        .as_str()
        .ok_or_else(|| err(format!("{label} path is invalid")))?;
    let relative = Path::new(relative);
    if relative.is_absolute() || relative.components().any(|c| c == Component::ParentDir) {
        return Err(err(format!("{label} path is invalid")));
    }
    let (raw, v) = load(&evidence.join(relative), label)?;
    if reference["digest"] != sha256_digest(&raw) {
        return Err(err(format!("{label} digest does not match manifest")));
    }
    Ok((raw, v))
}
fn artifact(raw: &[u8], v: &Value, anchor: &Value) -> Result<Value> {
    let identity_digest = sha256_digest(raw);
    if *anchor != identity_digest {
        return Err(err("artifact identity does not match receipt anchor"));
    }
    schema(
        v,
        "model_artifact_identity",
        "artifact identity is invalid: ",
    )?;
    let (exact, name, kind) = match text(v, "artifact_format") {
        "hf_snapshot" => (
            &v["checkpoint_tree_sha256"],
            &v["model_id"],
            "hf_snapshot_tree_sha256",
        ),
        "gguf" => (&v["sha256"], &v["artifact_name"], "file_sha256"),
        "tensorrt_llm_engine" => (
            &v["engine_bundle_tree_sha256"],
            &v["bundle_name"],
            "tensorrt_llm_engine_tree_sha256",
        ),
        _ => return Err(err("artifact format is unsupported")),
    };
    let exact = exact
        .as_str()
        .filter(|s| crate::is_digest(&format!("sha256:{s}")))
        .ok_or_else(|| err("artifact identity lacks an exact content digest"))?;
    if name.as_str().is_none_or(|s| s.is_empty()) {
        return Err(err("artifact identity name is invalid"));
    }
    Ok(
        json!({"name":name,"artifact_format":v["artifact_format"],"artifact_digest":format!("sha256:{exact}"),"digest_kind":kind,"artifact_identity_digest":identity_digest,"artifact_identity":v,"artifact_identity_payload":STANDARD.encode(raw)}),
    )
}
fn metric(request: &Value, report: &Value) -> Result<Value> {
    let c = &request["comparison"];
    if !c.is_object() {
        return Err(err("evidence request comparison is invalid"));
    }
    if !report["metric"].is_string() {
        return Err(err("comparison report metric is invalid"));
    }
    let scorer = &c["scorer_extension"];
    if scorer.is_null() {
        if c["metric"] != report["metric"] {
            return Err(err("request metric disagrees with comparison report"));
        }
        return Ok(json!({"kind":"built_in_metric","name":report["metric"],"scorer":null}));
    }
    if !scorer.is_object() {
        return Err(err("scorer extension binding is invalid"));
    }
    let mut out = json!({});
    for k in [
        "scorer_id",
        "scorer_version",
        "descriptor_sha256",
        "configuration_sha256",
    ] {
        let s = scorer[k]
            .as_str()
            .ok_or_else(|| err("scorer extension identity is incomplete"))?;
        out[k] = json!(if k.ends_with("_sha256") && !s.starts_with("sha256:") {
            format!("sha256:{s}")
        } else {
            s.into()
        })
    }
    Ok(json!({"kind":"scorer_extension","name":report["metric"],"scorer":out}))
}
fn contract_release(receipt: &Value, manifest: &Value, report: &Value) -> Result<&'static str> {
    if receipt_format(receipt) && *manifest == "invarlock/evidence-pack-v1" {
        match report.as_str() {
            Some("invarlock/comparison-report-v1" | "invarlock/comparison-report-v2") => {
                return Ok("0.13.0")
            }
            Some("invarlock/comparison-report-v3") => return Ok("0.15.0"),
            _ => {}
        }
    }
    Err(err(
        "receipt, evidence, and report versions are not a supported contract set",
    ))
}
fn timestamp(v: &Value, label: &str) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    let s = v
        .as_str()
        .ok_or_else(|| err(format!("{label} is invalid")))?;
    Ok(json!(super::attestation_time::format(
        super::attestation_time::parse(s, label)?
    )))
}
pub fn write_acceptance_attestation(request: &Value) -> Result<Value> {
    let evidence = path(request, "evidence_path")?;
    if !evidence.is_dir() || evidence.is_symlink() {
        return Err(err("evidence must be a real directory"));
    }
    let pinned = fs::PinnedDirectory::open(evidence, false)
        .map_err(|_| err("evidence must be a real directory"))?;
    let (receipt_raw, receipt) = load(path(request, "receipt_path")?, "verification receipt")?;
    let (rs, ri, rf) = authenticate_receipt(&receipt)?;
    let (manifest_raw, manifest) = load(&evidence.join("manifest.json"), "evidence manifest")?;
    if rs["pack_manifest_digest"] != sha256_digest(&manifest_raw) {
        return Err(err("receipt does not bind the supplied evidence manifest"));
    }
    let refs = &manifest["evidence"];
    if !refs.is_object() {
        return Err(err("evidence manifest references are invalid"));
    }
    let (br, b) = bound(
        evidence,
        &refs["baseline_provider_identity"],
        "baseline artifact identity",
    )?;
    let (sr, s) = bound(
        evidence,
        &refs["subject_provider_identity"],
        "subject artifact identity",
    )?;
    let (schedule_raw, schedule) = bound(evidence, &refs["schedule"], "evaluation schedule")?;
    let (_, report) = bound(evidence, &refs["evaluation_report"], "comparison report")?;
    let (_, evaluation) = bound(evidence, &refs["request"], "evaluation request")?;
    let anchors = &rs["anchors"];
    if !anchors.is_object() {
        return Err(err("receipt anchors are invalid"));
    }
    if !anchors["artifact_digests"].is_object() {
        return Err(err("receipt artifact anchors are invalid"));
    }
    let baseline = artifact(&br, &b, &anchors["artifact_digests"]["baseline"])?;
    let subject = artifact(&sr, &s, &anchors["artifact_digests"]["subject"])?;
    let schedule_digest = sha256_digest(&schedule_raw);
    if anchors["schedule_digest"] != schedule_digest {
        return Err(err("evaluation schedule does not match receipt anchor"));
    }
    let dataset = &schedule["dataset_identity"];
    if dataset.as_object().is_none_or(|o| o.is_empty()) {
        return Err(err("evaluation schedule source identity is invalid"));
    }
    let verdict = &rs["verdict"];
    if verdict["integrity_ok"] != true {
        return Err(err("only integrity-verified receipts can be transported"));
    }
    if ![json!("pass"), json!("fail")].contains(&verdict["policy_verdict"]) {
        return Err(err("receipt has no completed technical policy verdict"));
    }
    if report["verdict"] != verdict["policy_verdict"] {
        return Err(err("comparison report verdict disagrees with receipt"));
    }
    if report["policy_digest"] != anchors["policy_digest"] {
        return Err(err("comparison report policy disagrees with receipt"));
    }
    let raw = read(
        path(request, "signing_key_path")?,
        "attestation signing key",
        65536,
    )
    .map_err(|e| err(format!("could not load attestation signing key: {e}")))?;
    if std::str::from_utf8(&raw).is_ok_and(other_pem_algorithm) {
        return Err(err("attestation signing key must be Ed25519"));
    }
    let key = crypto::private_key(&raw).map_err(|_| {
        err("could not load attestation signing key: Could not deserialize key data")
    })?;
    let ef = crypto::fingerprint(&key.verifying_key());
    let ei = identity(&request["signer_identity"], "attestation signer identity")?;
    let relationship = if ef == rf && ei == ri {
        "same_signer"
    } else {
        "countersigned"
    };
    let predicate = json!({"format":"invarlock/acceptance-predicate-v2","subject":subject,"baseline":baseline,"contracts":{"invarlock_release":contract_release(&rs["format"],&manifest["format"],&report["format"])? ,"evidence_pack":manifest["format"],"comparison_report":report["format"],"receipt":rs["format"]},"evaluation_source":{"schedule_format":schedule["format_version"],"schedule_digest":schedule_digest,"identity_digest":sha256_digest(&canonical(dataset)?),"identity":dataset},"metric":metric(&evaluation,&report)?,"policy":{"identity":identity(&request["policy_identity"],"evaluated policy identity")?,"digest":anchors["policy_digest"]},"technical_verdict":verdict,"timestamps":{"evaluation_completed_at":timestamp(&request["evaluation_completed_at"],"evaluation completion timestamp")?,"receipt_issued_at":null,"attestation_issued_at":timestamp(&request["issued_at"],"attestation issuance timestamp")?},"signers":{"receipt":{"identity":ri,"fingerprint":rf},"envelope":{"identity":ei,"fingerprint":ef},"relationship":relationship},"receipt":{"representation":"embedded","media_type":"application/vnd.invarlock.verification-receipt+json","digest":sha256_digest(&receipt_raw),"raw_base64":STANDARD.encode(receipt_raw),"content":receipt}});
    schema(
        &predicate,
        "acceptance_predicate",
        "acceptance predicate is invalid: ",
    )?;
    let statement = json!({"_type":STATEMENT_TYPE,"subject":[{"name":predicate["subject"]["name"],"digest":{"sha256":text(&predicate["subject"],"artifact_digest").trim_start_matches("sha256:")}}],"predicateType":PREDICATE_TYPE,"predicate":predicate});
    let payload = canonical(&statement)?;
    let envelope = json!({"payloadType":PAYLOAD_TYPE,"payload":STANDARD.encode(&payload),"signatures":[{"keyid":ef,"sig":crypto::signature_value(&key,&pae(&payload))}]});
    pinned.verify().map_err(|e| err(e.to_string()))?;
    let output = path(request, "output_path")?;
    fs::write_file_no_replace(output, &canonical(&envelope)?, 0o444, false).map_err(
        |e| match e {
            fs::Error::Io(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => err(format!(
                "attestation destination already exists: {}",
                output.file_name().unwrap_or_default().to_string_lossy()
            )),
            _ => err("could not write acceptance attestation"),
        },
    )?;
    Ok(
        json!({"path":output.canonicalize().map_err(|e|err(e.to_string()))?.to_string_lossy(),"statement":statement,"signer_fingerprint":ef}),
    )
}
fn authenticated_statement(request: &Value) -> Result<(Value, String)> {
    let (_, envelope) = load(path(request, "envelope_path")?, "DSSE envelope")?;
    if !fields(&envelope, &["payloadType", "payload", "signatures"]) {
        return Err(err("DSSE envelope fields are invalid"));
    }
    if envelope["payloadType"] != PAYLOAD_TYPE {
        return Err(err("DSSE payload type is invalid"));
    }
    let signatures = envelope["signatures"]
        .as_array()
        .filter(|a| a.len() == 1)
        .ok_or_else(|| err("DSSE envelope must contain exactly one signature"))?;
    let s = &signatures[0];
    if !fields(s, &["keyid", "sig"]) {
        return Err(err("DSSE signature fields are invalid"));
    }
    let keyid = normalized_digest(&s["keyid"], "DSSE signer key ID")?;
    let decode = |v: &Value| {
        v.as_str()
            .and_then(|s| DECODE.decode(s).ok())
            .ok_or_else(|| err("DSSE base64 fields are invalid"))
    };
    let payload = decode(&envelope["payload"])?;
    let signature = decode(&s["sig"])?;
    let statement = parse_json(&payload, "in-toto Statement")?;
    if !statement.is_object() {
        return Err(err("in-toto Statement must be a JSON object"));
    }
    if payload != canonical(&statement)? {
        return Err(err("in-toto Statement must use canonical JSON"));
    }
    let material = request["trusted_public_keys"]
        .get(&keyid)
        .filter(|v| !v.is_null())
        .ok_or_else(|| err("DSSE signer key is unavailable from recipient trust anchors"))?;
    let pem = if let Some(s) = material.as_str() {
        s.to_owned()
    } else {
        String::from_utf8(
            read(path(material, "path")?, "recipient public key", 65536)
                .map_err(|e| err(format!("could not load recipient public key: {e}")))?,
        )
        .map_err(|_| err("could not load recipient public key: invalid PEM"))?
    };
    if other_pem_algorithm(&pem) {
        return Err(err("recipient public key must be Ed25519"));
    }
    let key = crypto::public_key(&pem)
        .map_err(|_| err("could not load recipient public key: Could not deserialize key data"))?;
    if crypto::fingerprint(&key) != keyid {
        return Err(err("recipient public key does not match DSSE key ID"));
    }
    Signature::from_slice(&signature)
        .ok()
        .filter(|s| key.verify_strict(&pae(&payload), s).is_ok())
        .ok_or_else(|| err("DSSE signature verification failed"))?;
    Ok((statement, keyid))
}
fn equal(a: &Value, b: &Value) -> bool {
    contracts::runtime::json_equal(a, b)
}
fn receipt_consistency(p: &Value) -> (bool, Vec<String>) {
    let mut errors = Vec::new();
    let r = &p["receipt"];
    let mut raw_matches = false;
    match DECODE.decode(text(r, "raw_base64")) {
        Ok(raw) => {
            let digest_matches = r["digest"] == sha256_digest(&raw);
            if !digest_matches {
                errors.push("embedded receipt digest over raw bytes is invalid".into());
            }
            match parse_json(&raw, "embedded receipt raw bytes") {
                Ok(parsed) => {
                    if !equal(&parsed, &r["content"]) {
                        errors.push("embedded receipt raw bytes disagree with content".into());
                    } else {
                        raw_matches = digest_matches
                    }
                }
                Err(_) => errors.push("embedded receipt raw bytes are invalid".into()),
            }
        }
        Err(_) => errors.push("embedded receipt raw bytes are invalid".into()),
    }
    let (s, id, fingerprint, authenticated) = match authenticate_receipt(&r["content"]) {
        Ok((s, id, f)) => (s, id, f, raw_matches),
        Err(e) => {
            errors.push(e.message.replacen("embedded ", "", 1));
            (json!({}), String::new(), String::new(), false)
        }
    };
    if p["signers"]["receipt"] != json!({"identity":id,"fingerprint":fingerprint}) {
        errors.push("predicate receipt signer disagrees with embedded receipt".into())
    }
    if p["contracts"]["receipt"] != s["format"] {
        errors.push("predicate receipt contract disagrees with embedded receipt".into())
    }
    if !equal(&p["technical_verdict"], &s["verdict"]) {
        errors.push("technical verdict disagrees with embedded receipt".into())
    }
    if receipt_format(&s["format"]) && !p["timestamps"]["receipt_issued_at"].is_null() {
        errors.push(
            "receipt issuance timestamp is not authenticated by the embedded v0.13 receipt".into(),
        )
    }
    let anchors = &s["anchors"];
    if !anchors.is_object() {
        return (authenticated, errors);
    }
    if !anchors["artifact_digests"].is_object() {
        errors.push("embedded receipt artifact anchors are invalid".into())
    } else {
        for role in ["baseline", "subject"] {
            let transported = &p[role];
            let replay = (|| -> Result<Value> {
                let raw = DECODE
                    .decode(text(transported, "artifact_identity_payload"))
                    .map_err(|e| err(e.to_string()))?;
                let parsed = parse_json(&raw, &format!("{role} transported artifact identity"))?;
                if !equal(&parsed, &transported["artifact_identity"]) {
                    return Err(err(format!(
                        "{role} artifact identity payload disagrees with its object"
                    )));
                }
                artifact(
                    &raw,
                    &transported["artifact_identity"],
                    &anchors["artifact_digests"][role],
                )
            })();
            match replay {
                Err(e) => errors.push(e.message),
                Ok(v) if !equal(transported, &v) => {
                    errors.push(format!("{role} artifact disagrees with embedded identity"))
                }
                _ => {}
            }
        }
    }
    let source = &p["evaluation_source"];
    if canonical(&source["identity"])
        .map(|b| source["identity_digest"] != sha256_digest(&b))
        .unwrap_or(true)
    {
        errors.push("evaluation source identity digest does not match its identity".into())
    }
    if source["schedule_digest"] != anchors["schedule_digest"] {
        errors.push("evaluation source digest disagrees with embedded receipt".into())
    }
    if p["policy"]["digest"] != anchors["policy_digest"] {
        errors.push("policy digest disagrees with embedded receipt".into())
    }
    (authenticated, errors)
}
fn recipient_errors(
    s: &Value,
    p: &Value,
    policy: &Value,
    keyid: &str,
    now: Option<&str>,
) -> Result<Vec<String>> {
    let mut errors = Vec::new();
    if s["_type"] != STATEMENT_TYPE {
        errors.push("in-toto Statement type is invalid".into())
    }
    if s["predicateType"] != PREDICATE_TYPE {
        errors.push("acceptance predicate type is invalid".into())
    }
    let outer = &p["signers"]["envelope"];
    if outer["fingerprint"] != keyid {
        errors.push("predicate envelope signer disagrees with DSSE key ID".into())
    }
    for (registry, signer, label) in [
        ("trusted_signers", outer, "envelope signer"),
        (
            "trusted_receipt_verifiers",
            &p["signers"]["receipt"],
            "receipt verifier",
        ),
    ] {
        let fingerprint = if registry == "trusted_signers" {
            json!(keyid)
        } else {
            signer["fingerprint"].clone()
        };
        let matches = policy[registry]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["fingerprint"] == fingerprint && r["identity"] == signer["identity"])
            .collect::<Vec<_>>();
        if matches.is_empty() {
            errors.push(format!("{label} is not trusted by recipient policy"))
        } else if matches.len() != 1 {
            errors.push(format!(
                "{label} has multiple matching recipient trust records"
            ))
        } else if matches[0]["status"] == "revoked" {
            errors.push(format!("{label} is revoked by recipient policy"))
        }
    }
    if !policy["expected_receipt_trust_profile_digest"].is_null()
        && p["receipt"]["content"]["statement"]["verifier"]["trust_profile_digest"]
            != policy["expected_receipt_trust_profile_digest"]
    {
        errors.push("receipt trust-profile digest does not satisfy recipient policy".into())
    }
    if policy["expected_predicate_type"] != s["predicateType"] {
        errors.push("predicate type is not allowed by recipient policy".into())
    }
    if !policy["allowed_contract_versions"]
        .as_array()
        .unwrap()
        .contains(&p["contracts"]["invarlock_release"])
    {
        errors.push("contract version is not allowed by recipient policy".into())
    }
    let technical = &p["technical_verdict"];
    if technical["policy_verdict"] != policy["required_technical_verdict"]
        || technical["ok"] != true
        || technical["integrity_ok"] != true
    {
        errors.push("technical verdict does not satisfy recipient policy".into())
    }
    let relationship = &p["signers"]["relationship"];
    let actual = if p["signers"]["receipt"] == p["signers"]["envelope"] {
        "same_signer"
    } else {
        "countersigned"
    };
    if *relationship != actual {
        errors.push("declared signer relationship is inconsistent".into())
    }
    if *relationship == "countersigned" && policy["allow_countersigned_receipts"] == false {
        errors.push("countersigned receipts are not allowed by recipient policy".into())
    }
    let issued = super::attestation_time::parse(
        text(&p["timestamps"], "attestation_issued_at"),
        "attestation issuance timestamp",
    )?;
    let now = match now {
        Some(s) => super::attestation_time::parse(s, "recipient current time")?,
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| err(e.to_string()))?
            .as_micros() as i64,
    };
    let freshness = &policy["freshness"];
    let skew = freshness["clock_skew_seconds"].as_f64().unwrap();
    let age = (now - issued) as f64 / 1_000_000.0;
    if age < -skew {
        errors.push("attestation issuance timestamp is in the future".into())
    }
    if age > freshness["max_envelope_age_seconds"].as_f64().unwrap() + skew {
        errors.push("attestation envelope is stale under recipient policy".into())
    }
    if !freshness["max_evidence_age_seconds"].is_null() {
        if p["timestamps"]["receipt_issued_at"].is_null() {
            errors.push(
                "authoritative evidence timestamp is unavailable under recipient policy".into(),
            )
        } else {
            let issued = super::attestation_time::parse(
                text(&p["timestamps"], "receipt_issued_at"),
                "authoritative evidence timestamp",
            )?;
            let age = (now - issued) as f64 / 1_000_000.0;
            if age < -skew {
                errors.push("authoritative evidence timestamp is in the future".into())
            }
            if age > freshness["max_evidence_age_seconds"].as_f64().unwrap() + skew {
                errors.push("technical evidence is stale under recipient policy".into())
            }
        }
    }
    Ok(errors)
}
/// Authentication results remain separate from the independently supplied current policy.
pub fn verify_acceptance_attestation(request: &Value) -> Value {
    let (mut envelope_authenticated, mut receipt_authenticated, mut subject_bound) =
        (false, false, false);
    let mut statement = Value::Null;
    let mut predicate = Value::Null;
    let mut errors = Vec::<String>::new();
    let checked = (|| -> Result<()> {
        let (s, keyid) = authenticated_statement(request)?;
        statement = s;
        envelope_authenticated = true;
        let raw_policy = &request["recipient_policy"];
        let policy = if fields(raw_policy, &["path"]) {
            load(path(raw_policy, "path")?, "recipient acceptance policy")?.1
        } else if raw_policy.is_object() {
            raw_policy.clone()
        } else {
            return Err(err("recipient acceptance policy must be an object"));
        };
        schema(&policy, "recipient_acceptance_policy", "")?;
        for registry in ["trusted_signers", "trusted_receipt_verifiers"] {
            let mut seen = BTreeSet::new();
            for r in policy[registry].as_array().unwrap() {
                if !seen.insert((text(r, "identity"), text(r, "fingerprint"))) {
                    return Err(err(format!(
                        "{registry} contains a duplicate identity/fingerprint pair"
                    )));
                }
            }
        }
        if !fields(
            &statement,
            &["_type", "subject", "predicateType", "predicate"],
        ) {
            return Err(err("in-toto Statement fields are invalid"));
        }
        if !statement["predicate"].is_object() {
            return Err(err("acceptance predicate is invalid"));
        }
        predicate = statement["predicate"].clone();
        schema(&predicate, "acceptance_predicate", "")?;
        let (authenticated, e) = receipt_consistency(&predicate);
        receipt_authenticated = authenticated;
        errors.extend(e);
        let subject = &predicate["subject"];
        let expected = json!([{"name":subject["name"],"digest":{"sha256":text(subject,"artifact_digest").trim_start_matches("sha256:")}}]);
        let mut subject_errors = Vec::new();
        if !equal(&statement["subject"], &expected) {
            subject_errors.push("in-toto subject disagrees with acceptance predicate".to_owned())
        }
        if request["expected_subject_digest"].is_null()
            == request["subject_artifact_path"].is_null()
        {
            return Err(err(
                "provide exactly one expected subject digest or subject artifact path",
            ));
        }
        let observed = if !request["expected_subject_digest"].is_null() {
            normalized_digest(
                &request["expected_subject_digest"],
                "expected subject digest",
            )?
        } else {
            super::attestation_artifact::artifact_path_digest(
                path(request, "subject_artifact_path")?,
                text(subject, "digest_kind"),
            )?
        };
        subject_bound = subject["artifact_digest"] == observed;
        if !subject_bound {
            subject_errors.push("subject digest does not match recipient artifact".into())
        }
        errors.extend(subject_errors);
        errors.extend(recipient_errors(
            &statement,
            &predicate,
            &policy,
            &keyid,
            request["now"].as_str(),
        )?);
        Ok(())
    })();
    if let Err(e) = checked {
        errors.push(e.message)
    }
    let verdict = if [json!("pass"), json!("fail")]
        .contains(&predicate["technical_verdict"]["policy_verdict"])
    {
        predicate["technical_verdict"]["policy_verdict"].clone()
    } else {
        Value::Null
    };
    json!({"envelope_authenticated":envelope_authenticated,"receipt_authenticated":receipt_authenticated,"subject_bound":subject_bound,"historical_technical_verdict":verdict,"accepted":envelope_authenticated&&receipt_authenticated&&subject_bound&&errors.is_empty(),"statement":statement,"errors":errors})
}
