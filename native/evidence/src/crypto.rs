//! Ed25519 wire profiles shared by evidence families.
use crate::{canonical, EvidenceError, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::pkcs8::{DecodePrivateKey, DecodePublicKey, EncodePublicKey};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use invarlock_native_contracts::sha256_digest;
use serde_json::{json, Value};

pub fn private_key(raw: &[u8]) -> Result<SigningKey> {
    let pem = std::str::from_utf8(raw)
        .map_err(|_| EvidenceError::incomplete("verifier key could not be loaded"))?;
    SigningKey::from_pkcs8_pem(pem)
        .map_err(|_| EvidenceError::incomplete("verifier key could not be loaded"))
}
pub fn public_key(pem: &str) -> Result<VerifyingKey> {
    VerifyingKey::from_public_key_pem(pem)
        .map_err(|_| EvidenceError::integrity("Ed25519 public key is invalid"))
}
pub fn public_pem(key: &VerifyingKey) -> Result<String> {
    key.to_public_key_pem(Default::default())
        .map_err(|_| EvidenceError::incomplete("Ed25519 public key could not be encoded"))
}
pub fn fingerprint(key: &VerifyingKey) -> String {
    sha256_digest(key.as_bytes())
}
pub fn verify(key: &VerifyingKey, payload: &[u8], signature: &str) -> Result<()> {
    let raw = crate::decode_base64(signature)
        .map_err(|_| EvidenceError::integrity("Ed25519 signature is invalid"))?;
    let signature = Signature::from_slice(&raw)
        .map_err(|_| EvidenceError::integrity("Ed25519 signature is invalid"))?;
    // Match the signature verification operation, without choosing trust here.
    key.verify_strict(payload, &signature)
        .map_err(|_| EvidenceError::integrity("Ed25519 signature is invalid"))
}
pub fn signature_value(key: &SigningKey, payload: &[u8]) -> String {
    STANDARD.encode(key.sign(payload).to_bytes())
}
pub fn manifest_signature(key: &SigningKey, raw: &[u8]) -> Result<Value> {
    let public = key.verifying_key();
    Ok(
        json!({"format":"invarlock/evidence-pack-signature-v1", "algorithm":"ed25519",
        "signing_key_fingerprint":fingerprint(&public),
        "public_key":{"encoding":"pem","value":public_pem(&public)?},
        "signature":{"encoding":"base64","value":signature_value(key,raw)}}),
    )
}
pub fn signed_receipt(key: &SigningKey, statement: Value) -> Result<Value> {
    let signed = canonical(&statement)?;
    Ok(json!({"statement":statement,"signature":{
        "format":"invarlock/evidence-verification-receipt-signature-v1",
        "algorithm":"ed25519","public_key":{"encoding":"pem",
        "value":public_pem(&key.verifying_key())?},"value":signature_value(key,&signed)}}))
}
