//! Strict JSON and immutable contract transactions independent of Python.
//!
//! Canonical encoding follows the package's Python JSON profile: sorted Unicode
//! keys, literal UTF-8, compact separators, finite binary64 floats, and a final
//! newline. Arbitrary precision integers remain integers. Lone UTF-16 surrogates
//! are rejected because they have no UTF-8 canonical representation; Python's
//! parser alone accepts these, but its canonical encoder also rejects them.

mod canonical;
mod diagnostic;
mod numeric;
mod parser;
mod printable;
mod records;
pub mod request;
pub mod runtime;
pub mod schedule;
mod schemas;

pub use canonical::{
    canonical_json, canonical_json_digest, canonical_json_matches, canonical_json_no_newline,
    canonical_json_to_writer,
};
pub use parser::parse_json;
pub use records::{
    parse_run, CompactLikelihood, CompactRecord, LikelihoodRef, ParsedRun, RecordRef, RunView,
    SchemaCheckedRun,
};
pub use schemas::{
    first_schema_error, first_schema_error_details, first_schema_error_fragment,
    first_schema_error_value, packaged_schema, schema_names, validate_schema, SchemaErrorOrder,
    SchemaViolation, MAX_INPUT_BYTES, MAX_RECORDS,
};
pub use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fmt, sync::Arc};

pub type Result<T> = std::result::Result<T, ContractError>;

/// Stable native categories with source-facing messages. Schema diagnostics use
/// the bounded source-compatible renderer rather than the validator display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError {
    pub code: &'static str,
    pub message: String,
}
impl ContractError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for ContractError {}

pub fn sha256_digest(raw: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(raw))
}

/// One authenticated byte snapshot and its parsed value. Clones share immutable
/// ownership; no accessor can mutate the value independently from its identity.
#[derive(Debug, Clone)]
pub struct ParsedDocument(Arc<ParsedInner>);
#[derive(Debug)]
struct ParsedInner {
    raw: Arc<[u8]>,
    value: Value,
    digest: String,
}
impl ParsedDocument {
    pub fn parse(raw: impl Into<Arc<[u8]>>, label: &str) -> Result<Self> {
        let raw = raw.into();
        let value = parse_json(&raw, label)?;
        let digest = sha256_digest(&raw);
        Ok(Self(Arc::new(ParsedInner { raw, value, digest })))
    }
    pub fn raw(&self) -> &[u8] {
        &self.0.raw
    }
    pub fn value(&self) -> &Value {
        &self.0.value
    }
    pub fn digest(&self) -> &str {
        &self.0.digest
    }
    pub fn require_canonical(&self) -> Result<()> {
        if !canonical_json_matches(self.value(), self.raw())? {
            return Err(ContractError::new(
                "noncanonical_json",
                "payload must use canonical JSON",
            ));
        }
        Ok(())
    }
    pub fn validate(&self, name: &str) -> Result<ValidatedDocument> {
        validate_schema(self.value(), name)?;
        Ok(ValidatedDocument {
            parsed: self.clone(),
            schema: name.into(),
        })
    }
}

/// Owned schema evidence tied to an immutable value. Consuming the wrapper
/// releases the value but also consumes its validation and cached identity.
///
/// ```compile_fail
/// use invarlock_native_contracts::{SchemaCheckedValue, Value};
/// fn mutate(checked: &SchemaCheckedValue) {
///     checked.value()["records"] = Value::Null;
/// }
/// ```
#[derive(Debug)]
pub struct SchemaCheckedValue {
    value: Value,
    schema: String,
    canonical_digest: std::sync::OnceLock<String>,
}
impl SchemaCheckedValue {
    pub fn new(value: Value, schema: &str) -> Result<Self> {
        validate_schema(&value, schema)?;
        Ok(Self {
            value,
            schema: schema.into(),
            canonical_digest: std::sync::OnceLock::new(),
        })
    }
    pub fn value(&self) -> &Value {
        &self.value
    }
    pub fn schema(&self) -> &str {
        &self.schema
    }
    pub fn into_value(self) -> Value {
        self.value
    }
    pub fn canonical_digest(&self) -> Result<&str> {
        if self.canonical_digest.get().is_none() {
            let digest = canonical_json_digest(&self.value)?;
            let _ = self.canonical_digest.set(digest);
        }
        Ok(self
            .canonical_digest
            .get()
            .expect("initialized canonical digest"))
    }
}

/// Evidence that one immutable document passed one named packaged schema.
/// This proves structural validation, not cross-document or policy validity.
#[derive(Debug, Clone)]
pub struct ValidatedDocument {
    parsed: ParsedDocument,
    schema: Arc<str>,
}
impl ValidatedDocument {
    pub fn parsed(&self) -> &ParsedDocument {
        &self.parsed
    }
    pub fn value(&self) -> &Value {
        self.parsed.value()
    }
    pub fn raw(&self) -> &[u8] {
        self.parsed.raw()
    }
    pub fn digest(&self) -> &str {
        self.parsed.digest()
    }
    pub fn schema(&self) -> &str {
        &self.schema
    }
}

/// CPython-compatible diagnostic repr for JSON values (Unicode 15 semantics).
pub fn python_repr(value: &Value) -> String {
    diagnostic::repr(value)
}

/// Parse Python's nonnegative integer work allowance. Positive allowances above
/// u64 cannot constrain the bounded native workload, so they use the unbounded
/// internal representation without narrowing or rejecting the caller's integer.
pub fn bootstrap_budget(value: Option<&Value>, default: u64) -> Result<Option<u64>> {
    match value {
        None => Ok(Some(default)),
        Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) if !number.to_string().contains(['.', 'e', 'E', '-']) => {
            Ok(number.as_u64())
        }
        _ => Err(ContractError::new(
            "invalid_evaluation_records",
            "max_bootstrap_draws must be a non-negative integer",
        )),
    }
}
