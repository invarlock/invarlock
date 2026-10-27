//! Immutable compact run ownership. Fast-path failure always reparses the original
//! strict JSON so callers retain the established validation and diagnostic phases.
use crate::{canonical_json_to_writer, parse_json, ContractError, Result, Value};
use serde_json::Map;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Write},
    ops::Index,
    sync::OnceLock,
};

#[derive(Debug)]
pub struct CompactRecord {
    id: Value,
    input: Value,
    expected: Value,
    output: Value,
    scores: Value,
    metadata: Value,
    error: Value,
    context: Value,
    likelihood: Option<Box<CompactLikelihood>>,
}
impl CompactRecord {
    fn from_value(value: Value) -> Option<Self> {
        let Value::Object(mut object) = value else {
            return None;
        };
        let result = Self {
            id: object.remove("id")?,
            input: object.remove("input")?,
            expected: object.remove("expected")?,
            output: object.remove("output")?,
            scores: object.remove("scores")?,
            metadata: object.remove("metadata")?,
            error: object.remove("error")?,
            context: object.remove("context")?,
            likelihood: match object.remove("likelihood") {
                Some(value) => Some(Box::new(CompactLikelihood::from_value(value)?)),
                None => None,
            },
        };
        object.is_empty().then_some(result)
    }
    fn get(&self, key: &str) -> Option<&Value> {
        match key {
            "id" => Some(&self.id),
            "input" => Some(&self.input),
            "expected" => Some(&self.expected),
            "output" => Some(&self.output),
            "scores" => Some(&self.scores),
            "metadata" => Some(&self.metadata),
            "error" => Some(&self.error),
            "context" => Some(&self.context),
            "likelihood" => self.likelihood.as_deref().map(CompactLikelihood::as_value),
            _ => None,
        }
    }
    fn into_value(self) -> Value {
        let mut object = Map::with_capacity(8 + usize::from(self.likelihood.is_some()));
        for (key, value) in [
            ("id", self.id),
            ("input", self.input),
            ("expected", self.expected),
            ("output", self.output),
            ("scores", self.scores),
            ("metadata", self.metadata),
            ("error", self.error),
            ("context", self.context),
        ] {
            object.insert(key.into(), value);
        }
        if let Some(value) = self.likelihood {
            object.insert("likelihood".into(), (*value).into_value());
        }
        object.sort_keys();
        Value::Object(object)
    }
}

/// Exact likelihood facts without a retained JSON object map or field keys.
#[derive(Debug)]
pub struct CompactLikelihood {
    basis: Value,
    logprob_sum: Value,
    token_count: Value,
    utf8_byte_count: Value,
    input_digest: Value,
    reference_digest: Value,
    artifact_digest: Value,
    configuration_digest: Value,
    tokenizer_digest: Value,
    source: Value,
    service_identity_digest: Option<Value>,
    materialized: OnceLock<Value>,
}
impl CompactLikelihood {
    fn from_value(value: Value) -> Option<Self> {
        let Value::Object(mut object) = value else {
            return None;
        };
        let result = Self {
            basis: object.remove("basis")?,
            logprob_sum: object.remove("logprob_sum")?,
            token_count: object.remove("token_count")?,
            utf8_byte_count: object.remove("utf8_byte_count")?,
            input_digest: object.remove("input_digest")?,
            reference_digest: object.remove("reference_digest")?,
            artifact_digest: object.remove("artifact_digest")?,
            configuration_digest: object.remove("configuration_digest")?,
            tokenizer_digest: object.remove("tokenizer_digest")?,
            source: object.remove("source")?,
            service_identity_digest: object.remove("service_identity_digest"),
            materialized: OnceLock::new(),
        };
        object.is_empty().then_some(result)
    }
    fn get(&self, key: &str) -> Option<&Value> {
        match key {
            "basis" => Some(&self.basis),
            "logprob_sum" => Some(&self.logprob_sum),
            "token_count" => Some(&self.token_count),
            "utf8_byte_count" => Some(&self.utf8_byte_count),
            "input_digest" => Some(&self.input_digest),
            "reference_digest" => Some(&self.reference_digest),
            "artifact_digest" => Some(&self.artifact_digest),
            "configuration_digest" => Some(&self.configuration_digest),
            "tokenizer_digest" => Some(&self.tokenizer_digest),
            "source" => Some(&self.source),
            "service_identity_digest" => self.service_identity_digest.as_ref(),
            _ => None,
        }
    }
    fn as_value(&self) -> &Value {
        self.materialized
            .get_or_init(|| LikelihoodRef::Compact(self).to_value())
    }
    fn into_value(self) -> Value {
        if let Some(value) = self.materialized.into_inner() {
            return value;
        }
        let mut object =
            Map::with_capacity(10 + usize::from(self.service_identity_digest.is_some()));
        for (key, value) in [
            ("basis", self.basis),
            ("logprob_sum", self.logprob_sum),
            ("token_count", self.token_count),
            ("utf8_byte_count", self.utf8_byte_count),
            ("input_digest", self.input_digest),
            ("reference_digest", self.reference_digest),
            ("artifact_digest", self.artifact_digest),
            ("configuration_digest", self.configuration_digest),
            ("tokenizer_digest", self.tokenizer_digest),
            ("source", self.source),
        ] {
            object.insert(key.into(), value);
        }
        if let Some(value) = self.service_identity_digest {
            object.insert("service_identity_digest".into(), value);
        }
        object.sort_keys();
        Value::Object(object)
    }
}

/// Schema proof and canonical identity bound to immutable compact ownership.
/// ```compile_fail
/// use invarlock_native_contracts::{SchemaCheckedRun, Value};
/// fn mutate(run: &SchemaCheckedRun) { run.view().record(0)["scores"] = Value::Null; }
/// ```
#[derive(Debug)]
pub struct SchemaCheckedRun {
    header: Value,
    records: Box<[CompactRecord]>,
    digest: OnceLock<String>,
}
#[derive(Debug)]
pub enum ParsedRun {
    Compact(SchemaCheckedRun),
    Json(Value),
}

/// Decode a canonical, schema-valid run row by row. Every unsupported, invalid,
/// or noncanonical fast-path attempt is discarded before strict generic fallback.
pub fn parse_run(raw: &[u8], label: &str) -> Result<ParsedRun> {
    if let Some(run) = try_compact_run(raw, label) {
        return Ok(ParsedRun::Compact(run));
    }
    parse_json(raw, label).map(ParsedRun::Json)
}
fn try_compact_run(raw: &[u8], label: &str) -> Option<SchemaCheckedRun> {
    if raw.len() > crate::MAX_INPUT_BYTES {
        return None;
    }
    let validators = crate::schemas::run_validators().ok()?;
    let (mut header, records) = crate::parser::parse_root_records(raw, label, |value| {
        if !validators.row.is_valid(&value) {
            return None;
        }
        CompactRecord::from_value(value)
    })
    .ok()??;
    if records.is_empty() {
        return None;
    }
    header.insert("records".into(), Value::Null);
    // A hosted-service configuration (or an unsupported header field) can
    // contain a document-sized graph. Validate the owned header by reference.
    let mut header = Value::Object(header);
    if !validators.header.is_valid(&header) {
        return None;
    }
    header
        .as_object_mut()
        .expect("parsed header object")
        .remove("records");
    let run = SchemaCheckedRun {
        header,
        records: records.into_boxed_slice(),
        digest: OnceLock::new(),
    };
    if !run.view().canonical_matches(raw).ok()? {
        return None;
    }
    // This identity is computed here only after the exact canonical comparison
    // binds these bytes to the privately owned typed representation.
    let _ = run.digest.set(crate::sha256_digest(raw));
    Some(run)
}
impl SchemaCheckedRun {
    pub fn schema(&self) -> &'static str {
        "run"
    }
    pub fn view(&self) -> RunView<'_> {
        RunView::Compact(self)
    }
    pub fn canonical_digest(&self) -> Result<&str> {
        if self.digest.get().is_none() {
            let digest = digest_to_writer(|writer| self.view().canonical_to_writer(writer, true))?;
            let _ = self.digest.set(digest);
        }
        Ok(self.digest.get().expect("initialized run digest"))
    }
    pub fn into_value(self) -> Value {
        let Value::Object(mut header) = self.header else {
            unreachable!()
        };
        header.insert(
            "records".into(),
            Value::Array(
                self.records
                    .into_vec()
                    .into_iter()
                    .map(CompactRecord::into_value)
                    .collect(),
            ),
        );
        header.sort_keys();
        Value::Object(header)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RunView<'a> {
    Json(&'a Value),
    Compact(&'a SchemaCheckedRun),
}
#[derive(Clone, Copy, Debug)]
pub enum RecordRef<'a> {
    Json(&'a Value),
    Compact(&'a CompactRecord),
}
#[derive(Clone, Copy, Debug)]
pub enum LikelihoodRef<'a> {
    Json(&'a Value),
    Compact(&'a CompactLikelihood),
}
const NULL: Value = Value::Null;
impl<'a> RunView<'a> {
    /// Header access. Use records()/record() for the separately owned row array.
    pub fn get(self, key: &str) -> Option<&'a Value> {
        match self {
            Self::Json(value) => value.get(key),
            Self::Compact(run) => run.header.get(key),
        }
    }
    pub fn record_count(self) -> usize {
        match self {
            Self::Json(value) => value
                .get("records")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
            Self::Compact(run) => run.records.len(),
        }
    }
    pub fn record(self, index: usize) -> RecordRef<'a> {
        match self {
            Self::Json(value) => RecordRef::Json(&value["records"][index]),
            Self::Compact(run) => RecordRef::Compact(&run.records[index]),
        }
    }
    pub fn records(
        self,
    ) -> impl ExactSizeIterator<Item = RecordRef<'a>> + DoubleEndedIterator + 'a {
        (0..self.record_count()).map(move |index| self.record(index))
    }
    pub fn canonical_digest(self) -> Result<String> {
        match self {
            Self::Json(value) => crate::canonical_json_digest(value),
            Self::Compact(run) => run.canonical_digest().map(str::to_owned),
        }
    }
    pub fn canonical_matches(self, raw: &[u8]) -> Result<bool> {
        struct Match<'a> {
            remaining: &'a [u8],
            matches: bool,
        }
        impl Write for Match<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.matches {
                    if let Some(rest) = self.remaining.strip_prefix(bytes) {
                        self.remaining = rest;
                    } else {
                        self.matches = false;
                    }
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut writer = Match {
            remaining: raw,
            matches: true,
        };
        self.canonical_to_writer(&mut writer, true)?;
        Ok(writer.matches && writer.remaining.is_empty())
    }
    pub fn canonical_to_writer(self, writer: &mut impl Write, newline: bool) -> Result<()> {
        let Self::Compact(run) = self else {
            let Self::Json(value) = self else {
                unreachable!()
            };
            return canonical_json_to_writer(value, writer, newline);
        };
        let header = run.header.as_object().expect("run header object");
        let mut keys: Vec<_> = header
            .keys()
            .map(String::as_str)
            .chain(std::iter::once("records"))
            .collect();
        keys.sort_unstable();
        put(writer, b"{")?;
        for (index, key) in keys.into_iter().enumerate() {
            if index != 0 {
                put(writer, b",")?;
            }
            serde_json::to_writer(&mut *writer, key)
                .map_err(|error| ContractError::new("canonical_io", error.to_string()))?;
            put(writer, b":")?;
            if key == "records" {
                put(writer, b"[")?;
                for (index, row) in self.records().enumerate() {
                    if index != 0 {
                        put(writer, b",")?;
                    }
                    row.canonical_to_writer(writer, false)?;
                }
                put(writer, b"]")?;
            } else {
                canonical_json_to_writer(&header[key], writer, false)?;
            }
        }
        put(writer, if newline { b"}\n" } else { b"}" })
    }
    pub fn to_value(self) -> Value {
        match self {
            Self::Json(value) => value.clone(),
            Self::Compact(run) => {
                let mut header = run.header.as_object().unwrap().clone();
                header.insert(
                    "records".into(),
                    Value::Array(self.records().map(RecordRef::to_value).collect()),
                );
                header.sort_keys();
                Value::Object(header)
            }
        }
    }
}
impl Index<&str> for RunView<'_> {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}
impl<'a> RecordRef<'a> {
    /// Borrow likelihood facts without populating the generic compatibility cache.
    pub fn likelihood(self) -> Option<LikelihoodRef<'a>> {
        match self {
            Self::Json(value) => value.get("likelihood").map(LikelihoodRef::Json),
            Self::Compact(record) => record.likelihood.as_deref().map(LikelihoodRef::Compact),
        }
    }
    pub fn get(self, key: &str) -> Option<&'a Value> {
        match self {
            Self::Json(value) => value.get(key),
            Self::Compact(record) => record.get(key),
        }
    }
    pub fn id(self) -> &'a str {
        self.get("id")
            .and_then(Value::as_str)
            .expect("validated record ID")
    }
    pub fn to_value(self) -> Value {
        match self {
            Self::Json(value) => value.clone(),
            Self::Compact(_) => {
                let mut object = Map::with_capacity(8 + usize::from(self.likelihood().is_some()));
                for key in [
                    "id", "input", "expected", "output", "scores", "metadata", "error", "context",
                ] {
                    if let Some(value) = self.get(key) {
                        object.insert(key.into(), value.clone());
                    }
                }
                if let Some(likelihood) = self.likelihood() {
                    object.insert("likelihood".into(), likelihood.to_value());
                }
                object.sort_keys();
                Value::Object(object)
            }
        }
    }
    pub fn canonical_to_writer(self, writer: &mut impl Write, newline: bool) -> Result<()> {
        if let Self::Json(value) = self {
            return canonical_json_to_writer(value, writer, newline);
        }
        put(writer, b"{")?;
        let mut first = true;
        for key in [
            "context",
            "error",
            "expected",
            "id",
            "input",
            "likelihood",
            "metadata",
            "output",
            "scores",
        ] {
            let present = if key == "likelihood" {
                self.likelihood().is_some()
            } else {
                self.get(key).is_some()
            };
            if !present {
                continue;
            }
            if !first {
                put(writer, b",")?;
            }
            first = false;
            serde_json::to_writer(&mut *writer, key)
                .map_err(|error| ContractError::new("canonical_io", error.to_string()))?;
            put(writer, b":")?;
            if key == "likelihood" {
                self.likelihood()
                    .expect("present likelihood")
                    .canonical_to_writer(writer, false)?;
            } else {
                canonical_json_to_writer(
                    self.get(key).expect("present record field"),
                    writer,
                    false,
                )?;
            }
        }
        put(writer, if newline { b"}\n" } else { b"}" })
    }
}
impl Index<&str> for RecordRef<'_> {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}
impl<'a> LikelihoodRef<'a> {
    pub fn get(self, key: &str) -> Option<&'a Value> {
        match self {
            Self::Json(value) => value.get(key),
            Self::Compact(value) => value.get(key),
        }
    }
    pub fn to_value(self) -> Value {
        if let Self::Json(value) = self {
            return value.clone();
        }
        let mut object =
            Map::with_capacity(10 + usize::from(self.get("service_identity_digest").is_some()));
        for key in [
            "artifact_digest",
            "basis",
            "configuration_digest",
            "input_digest",
            "logprob_sum",
            "reference_digest",
            "service_identity_digest",
            "source",
            "token_count",
            "tokenizer_digest",
            "utf8_byte_count",
        ] {
            if let Some(value) = self.get(key) {
                object.insert(key.into(), value.clone());
            }
        }
        Value::Object(object)
    }
    pub fn canonical_to_writer(self, writer: &mut impl Write, newline: bool) -> Result<()> {
        if let Self::Json(value) = self {
            return canonical_json_to_writer(value, writer, newline);
        }
        put(writer, b"{")?;
        let mut first = true;
        for key in [
            "artifact_digest",
            "basis",
            "configuration_digest",
            "input_digest",
            "logprob_sum",
            "reference_digest",
            "service_identity_digest",
            "source",
            "token_count",
            "tokenizer_digest",
            "utf8_byte_count",
        ] {
            if let Some(value) = self.get(key) {
                if !first {
                    put(writer, b",")?;
                }
                first = false;
                serde_json::to_writer(&mut *writer, key)
                    .map_err(|error| ContractError::new("canonical_io", error.to_string()))?;
                put(writer, b":")?;
                canonical_json_to_writer(value, writer, false)?;
            }
        }
        put(writer, if newline { b"}\n" } else { b"}" })
    }
}
impl Index<&str> for LikelihoodRef<'_> {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}
fn put(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes).map_err(|error| {
        ContractError::new(
            "canonical_io",
            format!("value is not canonical JSON: {error}"),
        )
    })
}
fn digest_to_writer(
    encode: impl FnOnce(&mut io::BufWriter<HashWriter>) -> Result<()>,
) -> Result<String> {
    let mut writer = io::BufWriter::with_capacity(64 * 1024, HashWriter(Sha256::new()));
    encode(&mut writer)?;
    writer
        .flush()
        .map_err(|error| ContractError::new("canonical_io", error.to_string()))?;
    Ok(format!(
        "sha256:{:x}",
        writer.get_ref().0.clone().finalize()
    ))
}
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn row_conversion_moves_open_field_allocations_and_owns_no_row_map() {
        let row = json!({"id":"row","input":"allocated open input text","expected":null,
            "output":null,"scores":{},"metadata":{},"error":null,"context":{}});
        let pointer = row["input"].as_str().unwrap().as_ptr();
        let compact = CompactRecord::from_value(row).unwrap();
        assert_eq!(
            compact.get("input").unwrap().as_str().unwrap().as_ptr(),
            pointer
        );
        assert_eq!(
            std::mem::size_of::<CompactRecord>(),
            8 * std::mem::size_of::<Value>()
                + std::mem::size_of::<Option<Box<CompactLikelihood>>>()
        );
        let rebuilt = compact.into_value();
        assert_eq!(rebuilt["input"].as_str().unwrap().as_ptr(), pointer);
    }
    #[test]
    fn likelihood_streaming_and_typed_access_never_fill_the_cold_generic_cache() {
        for service in [false, true] {
            let mut facts = json!({"basis":"reference_continuation","logprob_sum":-0.0,
                "token_count":crate::parse_json(b"123456789012345678901234567890","integer").unwrap(),
                "utf8_byte_count":3,"input_digest":crate::sha256_digest(b"input"),
                "reference_digest":crate::sha256_digest(b"reference"),"artifact_digest":crate::sha256_digest(b"artifact"),
                "configuration_digest":crate::sha256_digest(b"configuration"),"tokenizer_digest":crate::sha256_digest(b"tokenizer"),
                "source":{"name":"source é","version":"1"}});
            if service {
                facts["artifact_digest"] = Value::Null;
                facts["service_identity_digest"] = json!(crate::sha256_digest(b"service"));
                facts["logprob_sum"] = crate::parse_json(b"-1e-7", "number").unwrap();
            }
            let row = json!({"id":"row","input":"input","expected":"yes","output":null,"scores":{},
                "metadata":{},"error":null,"context":{},"likelihood":facts});
            let value = json!({"format":"invarlock/evaluation-run-v1","run_id":"run",
                "artifact_digest":crate::sha256_digest(b"artifact"),"source_digest":null,
                "source":{"name":"fixture","version":"1"},"score_provenance":{},"records":[row]});
            let raw = crate::canonical_json(&value).unwrap();
            let ParsedRun::Compact(compact) = parse_run(&raw, "run").unwrap() else {
                panic!("valid exact likelihood representation")
            };
            let record = compact.view().record(0);
            let likelihood = compact.records[0].likelihood.as_ref().unwrap();
            assert!(likelihood.materialized.get().is_none());
            let typed = record.likelihood().unwrap();
            assert!(matches!(typed, LikelihoodRef::Compact(_)));
            for key in [
                "basis",
                "logprob_sum",
                "token_count",
                "utf8_byte_count",
                "input_digest",
                "reference_digest",
                "artifact_digest",
                "configuration_digest",
                "tokenizer_digest",
                "source",
                "service_identity_digest",
            ] {
                assert_eq!(typed.get(key), facts.get(key));
                assert_eq!(typed[key], facts[key]);
            }
            assert_eq!(
                typed["token_count"].to_string(),
                "123456789012345678901234567890"
            );
            assert_eq!(
                typed["logprob_sum"].to_string(),
                facts["logprob_sum"].to_string()
            );
            let mut bytes = Vec::new();
            typed.canonical_to_writer(&mut bytes, true).unwrap();
            assert_eq!(bytes, crate::canonical_json(&facts).unwrap());
            let mut bytes = Vec::new();
            compact
                .view()
                .canonical_to_writer(&mut bytes, true)
                .unwrap();
            assert_eq!(bytes, raw);
            assert_eq!(
                compact.canonical_digest().unwrap(),
                crate::canonical_json_digest(&value).unwrap()
            );
            assert_eq!(
                crate::canonical_json(&record.to_value()).unwrap(),
                crate::canonical_json(&value["records"][0]).unwrap()
            );
            let ordinary = crate::parse_json(&raw, "run").unwrap();
            assert_eq!(
                serde_json::to_vec(&typed.to_value()).unwrap(),
                serde_json::to_vec(&ordinary["records"][0]["likelihood"]).unwrap()
            );
            assert!(
                likelihood.materialized.get().is_none(),
                "typed access and encoders must retain compact ownership"
            );
            let generic = record.get("likelihood").unwrap();
            assert!(likelihood.materialized.get().is_some());
            assert_eq!(
                serde_json::to_vec(generic).unwrap(),
                serde_json::to_vec(&ordinary["records"][0]["likelihood"]).unwrap()
            );
            assert!(std::ptr::eq(generic, record.get("likelihood").unwrap()));
            assert_eq!(
                serde_json::to_vec(&compact.into_value()).unwrap(),
                serde_json::to_vec(&ordinary).unwrap()
            );
        }
    }
}
