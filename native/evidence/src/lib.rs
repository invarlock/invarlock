//! Offline evidence transactions. No provider, network or process dependencies.
pub mod captured;
pub mod crypto;
pub mod evidence_sets;
pub mod gguf;
pub mod judge;
pub mod judge_native;
pub mod native_pack;
pub mod native_publication;
pub mod qualification;
pub mod transaction;
pub mod trust_inputs;

use invarlock_native_contracts as contracts;
use serde_json::{json, Value};
use std::fmt;

#[derive(Debug, Clone)]
pub struct EvidenceError {
    pub message: String,
    pub exit_code: u8,
    pub receipt_written: bool,
}
pub type Result<T> = std::result::Result<T, EvidenceError>;
impl EvidenceError {
    pub fn new(code: u8, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: code,
            receipt_written: false,
        }
    }
    pub fn contract(message: impl Into<String>) -> Self {
        Self::new(4, message)
    }
    pub fn integrity(message: impl Into<String>) -> Self {
        Self::new(6, message)
    }
    pub fn incomplete(message: impl Into<String>) -> Self {
        Self::new(2, message)
    }
    pub fn payload(&self) -> Value {
        json!({"error": self.message, "exit_code": self.exit_code,
               "receipt_written": self.receipt_written})
    }
}
impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for EvidenceError {}
impl From<contracts::ContractError> for EvidenceError {
    fn from(error: contracts::ContractError) -> Self {
        Self::contract(error.to_string())
    }
}
pub(crate) fn bounded(message: &str) -> String {
    message.chars().take(240).collect()
}
pub(crate) fn canonical(value: &Value) -> Result<Vec<u8>> {
    contracts::canonical_json(value).map_err(Into::into)
}
pub(crate) fn digest(value: &Value) -> Result<String> {
    Ok(contracts::sha256_digest(&canonical(value)?))
}
pub(crate) fn is_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}
pub(crate) fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| EvidenceError::incomplete(format!("{key} is required")))
}

pub mod attestation;
pub mod attestation_artifact;
mod attestation_time;

/// Python strict base64 accepts nonzero unused trailing bits while enforcing padding.
pub fn decode_base64(value: impl AsRef<[u8]>) -> std::result::Result<Vec<u8>, base64::DecodeError> {
    use base64::{
        engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig},
        Engine,
    };
    const DECODE: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true),
    );
    DECODE.decode(value)
}

pub mod reporting;

pub mod preparation;
