use super::*;
pub(super) fn write(input: &Inputs, payload: &Value, manifest: &[u8], status: u8) -> Result<()> {
    let mut anchors = input.anchors.clone();
    let signer = anchors
        .as_object_mut()
        .unwrap()
        .remove("signer_fingerprint")
        .unwrap();
    anchors["pack_signer_fingerprint"] = signer;
    let statement = json!({"format":if anchors.get("request_digest").is_some(){"invarlock/evidence-verification-receipt-v2"}else{"invarlock/evidence-verification-receipt-v1"},"pack_manifest_digest":sha256_digest(manifest),"anchors":anchors,"verifier":{"identity":input.identity,"signing_key_fingerprint":crypto::fingerprint(&input.key.verifying_key()),"trust_profile_digest":input.profile},"verdict":{"ok":payload["ok"],"integrity_ok":payload["integrity_ok"],"policy_verdict":payload["policy_verdict"],"verification_status":status}});
    let receipt = crypto::signed_receipt(&input.key, statement)?;
    let bytes = canonical(&receipt)?;
    fs::write_file_no_replace(&input.receipt, &bytes, 0o444, true).map_err(|_| {
        EvidenceError::incomplete(if input.receipt.exists() {
            format!(
                "receipt destination already exists: {}",
                input
                    .receipt
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            )
        } else {
            "could not write signed receipt".into()
        })
    })
}
/// Authenticate a historical v1/v2 receipt against independent caller anchors.
/// Receipt signatures attest to the recorded verdict, including rejection; they
/// do not re-evaluate the pack or turn a negative verdict into a positive one.
pub fn verify_receipt(config: &Value) -> Result<Value> {
    let pack = PathBuf::from(required_string(config, "evidence_path")?);
    let path = PathBuf::from(required_string(config, "receipt_path")?);
    let mut errors = vec![];
    if [
        "expected_run_digests",
        "expected_baseline_run",
        "expected_subject_run",
    ]
    .iter()
    .any(|k| config.get(*k).is_some_and(|v| !v.is_null()))
    {
        errors.push("run anchors require a captured receipt".into());
    }
    if outside(&path, &pack, "verification receipt").is_err() {
        errors.push("verification receipt is inside the evidence pack".to_string());
    }
    let mut statement = Value::Null;
    let mut fingerprint = Value::Null;
    let mut signed = false;
    let result: Result<()> = (|| {
        let raw = fs::read_file(&path, 1024 * 1024).map_err(|e| integrity(e.to_string()))?;
        let receipt = object(&raw, "verification receipt")?;
        statement = receipt["statement"].clone();
        let signature = &receipt["signature"];
        signed = statement.is_object() && signature.is_object();
        if !signed {
            if config["require_signed"] != false {
                errors.push("signed verification receipt is required".into());
            }
            return Ok(());
        }
        let mut require = |ok: bool, msg: &str| {
            if !ok {
                errors.push(msg.to_owned());
            }
        };
        require(
            exact_keys(
                &statement,
                &[
                    "format",
                    "pack_manifest_digest",
                    "anchors",
                    "verifier",
                    "verdict",
                ],
            ),
            "signed receipt statement fields are invalid",
        );
        let v1 = statement["format"] == "invarlock/evidence-verification-receipt-v1";
        let v2 = statement["format"] == "invarlock/evidence-verification-receipt-v2";
        require(v1 || v2, "signed receipt format is invalid");
        require(
            statement["pack_manifest_digest"]
                .as_str()
                .is_some_and(is_digest),
            "signed receipt pack manifest digest is invalid",
        );
        require(
            exact_keys(signature, &["format", "algorithm", "public_key", "value"]),
            "signed receipt signature fields are invalid",
        );
        require(
            signature["format"] == "invarlock/evidence-verification-receipt-signature-v1",
            "signed receipt signature format is invalid",
        );
        require(
            signature["algorithm"] == "ed25519",
            "signed receipt algorithm must be ed25519",
        );
        let verifier = &statement["verifier"];
        let identity = config["expected_verifier_identity"]
            .as_str()
            .unwrap_or("")
            .trim();
        require(
            !identity.is_empty()
                && identity.len() <= 256
                && identity.as_bytes()[0].is_ascii_alphanumeric()
                && identity
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._:/-".contains(&b)),
            "expected verifier identity is invalid",
        );
        require(
            exact_keys(
                verifier,
                &[
                    "identity",
                    "signing_key_fingerprint",
                    "trust_profile_digest",
                ],
            ),
            "signed receipt verifier fields are invalid",
        );
        require(
            verifier["identity"].as_str()
                == config["expected_verifier_identity"].as_str().map(str::trim),
            "receipt verifier identity does not match caller expectation",
        );
        let profile = config
            .get("expected_trust_profile_digest")
            .or_else(|| config.get("expected_profile_digest"))
            .unwrap_or(&Value::Null);
        let profile = if profile.is_null() {
            Value::Null
        } else {
            json!(normalized(profile, "expected trust profile digest")?)
        };
        require(
            verifier["trust_profile_digest"] == profile,
            "receipt trust profile does not match caller expectation",
        );
        let verdict = &statement["verdict"];
        require(
            exact_keys(
                verdict,
                &[
                    "ok",
                    "integrity_ok",
                    "policy_verdict",
                    "verification_status",
                ],
            ),
            "signed receipt verdict fields are invalid",
        );
        require(
            verdict["ok"].is_boolean() && verdict["integrity_ok"].is_boolean(),
            "signed receipt verdict booleans are invalid",
        );
        require(
            verdict["policy_verdict"].is_null()
                || verdict["policy_verdict"] == "pass"
                || verdict["policy_verdict"] == "fail",
            "signed receipt policy verdict is invalid",
        );
        require(
            verdict["verification_status"].as_u64().is_some(),
            "signed receipt verification status is invalid",
        );
        require(
            verdict["ok"] != true
                || (verdict["integrity_ok"] == true
                    && verdict["policy_verdict"] != "fail"
                    && verdict["verification_status"] == 0),
            "signed receipt successful verdict is inconsistent",
        );
        let key = crypto::public_key(signature["public_key"]["value"].as_str().unwrap_or(""));
        if let Ok(key) = key {
            fingerprint = json!(crypto::fingerprint(&key));
            require(
                signature["public_key"]["encoding"] == "pem",
                "signed receipt public key is invalid",
            );
            require(
                verifier["signing_key_fingerprint"] == fingerprint,
                "receipt verifier fingerprint does not match its public key",
            );
            require(
                fingerprint
                    == normalized(
                        &config["expected_verifier_fingerprint"],
                        "expected verifier fingerprint",
                    )?,
                "receipt verifier key does not match caller expectation",
            );
            require(
                crypto::verify(
                    &key,
                    &canonical(&statement)?,
                    signature["value"].as_str().unwrap_or(""),
                )
                .is_ok(),
                "signed receipt signature verification failed",
            );
        } else {
            require(false, "signed receipt public key is invalid");
        }
        let manifest_bytes = fs::read_file(&pack.join("manifest.json"), 256 * 1024)
            .map_err(|e| integrity(e.to_string()))?;
        let manifest = object(&manifest_bytes, "pack manifest")?;
        let request_bytes = fs::read_file(&pack.join("request.json"), 4 * 1024 * 1024)
            .map_err(|e| integrity(e.to_string()))?;
        let request = object(&request_bytes, "pack request")?;
        require(
            manifest["evidence"]["request"]["path"] == "request.json",
            "pack request reference path is invalid",
        );
        require(
            manifest["evidence"]["request"]["digest"] == sha256_digest(&request_bytes),
            "pack request digest does not match manifest",
        );
        let expected_request = config
            .get("expected_request_digest")
            .filter(|v| !v.is_null());
        let gguf = ["baseline", "subject"]
            .iter()
            .any(|side| request["comparison"][side]["runtime"]["provider"] == "llama_cpp");
        require(
            !gguf || expected_request.is_some(),
            "independent request anchor is required for llama_cpp evidence",
        );
        require(
            !v1 || expected_request.is_none(),
            "request anchors require signed receipt format v2",
        );
        require(
            !v2 || expected_request.is_some(),
            "signed receipt v2 requires an independent request anchor",
        );
        let policy_bytes = fs::read_file(
            Path::new(required_string(config, "policy_path")?),
            4 * 1024 * 1024,
        )
        .map_err(|e| integrity(e.to_string()))?;
        let mut anchors = json!({"policy_digest":sha256_digest(&policy_bytes),"artifact_digests":{},"runtime_digests":{},"schedule_digest":normalized(&config["expected_schedule"],"schedule anchor")?,"pack_signer_fingerprint":normalized(config.get("expected_pack_signer_fingerprint").unwrap_or(&config["expected_signer"]),"pack signer anchor")?});
        for side in ["baseline", "subject"] {
            for kind in ["artifact", "runtime"] {
                anchors[format!("{kind}_digests")][side] = json!(normalized(
                    &config[format!("expected_{side}_{kind}")],
                    &format!("{side} {kind} anchor")
                )?);
            }
        }
        if let Some(expected) = expected_request {
            let expected = normalized(expected, "expected request anchor")?;
            require(
                expected == sha256_digest(&request_bytes),
                "request anchor does not match the supplied pack request",
            );
            if !v1 {
                anchors["request_digest"] = json!(expected);
            }
        }
        require(
            statement["pack_manifest_digest"] == sha256_digest(&manifest_bytes),
            "receipt does not bind the supplied pack manifest",
        );
        require(
            statement["anchors"] == anchors,
            "receipt anchors do not match caller-supplied anchors",
        );
        Ok(())
    })();
    if let Err(e) = result {
        errors.push(e.message);
    }
    Ok(
        json!({"ok":errors.is_empty(),"signed":signed,"statement":if signed{statement}else{Value::Null},"verifier_fingerprint":fingerprint,"errors":errors}),
    )
}
