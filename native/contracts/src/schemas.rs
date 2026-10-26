use crate::{canonical_json_to_writer, ContractError, Result, Value};
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::{Arc, Mutex, OnceLock},
};

pub const MAX_INPUT_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_RECORDS: usize = 50_000;
const SCHEMAS: &[(&str, &str)] = &[
    (
        "runtime_provider_capabilities",
        include_str!("../schemas/runtime_provider_capabilities.json"),
    ),
    (
        "acceptance_predicate",
        include_str!("../schemas/acceptance_predicate.schema.json"),
    ),
    (
        "comparison_policy",
        include_str!("../schemas/comparison_policy.schema.json"),
    ),
    (
        "evaluation_case_set",
        include_str!("../schemas/evaluation_case_set.schema.json"),
    ),
    (
        "evaluation_request",
        include_str!("../schemas/evaluation_request.schema.json"),
    ),
    (
        "evaluation_request_v2",
        include_str!("../schemas/evaluation_request_v2.schema.json"),
    ),
    (
        "evaluation_request_v3",
        include_str!("../schemas/evaluation_request_v3.schema.json"),
    ),
    (
        "evaluation_run",
        include_str!("../schemas/evaluation_run.schema.json"),
    ),
    (
        "evaluation_setup_result",
        include_str!("../schemas/evaluation_setup_result.schema.json"),
    ),
    (
        "evaluator_export",
        include_str!("../schemas/evaluator_export.schema.json"),
    ),
    (
        "evaluator_qualification_export",
        include_str!("../schemas/evaluator_qualification_export.schema.json"),
    ),
    (
        "evaluator_qualification_profile",
        include_str!("../schemas/evaluator_qualification_profile.schema.json"),
    ),
    (
        "evaluator_qualification_result",
        include_str!("../schemas/evaluator_qualification_result.schema.json"),
    ),
    (
        "evaluator_qualification_schedule",
        include_str!("../schemas/evaluator_qualification_schedule.schema.json"),
    ),
    (
        "evidence_observation",
        include_str!("../schemas/evidence_observation.schema.json"),
    ),
    (
        "evidence_pack",
        include_str!("../schemas/evidence_pack.schema.json"),
    ),
    (
        "evidence_pack_v2",
        include_str!("../schemas/evidence_pack_v2.schema.json"),
    ),
    (
        "evidence_set",
        include_str!("../schemas/evidence_set.schema.json"),
    ),
    (
        "evidence_set_recipient_policy",
        include_str!("../schemas/evidence_set_recipient_policy.schema.json"),
    ),
    (
        "evidence_set_verification",
        include_str!("../schemas/evidence_set_verification.schema.json"),
    ),
    (
        "evidence_verification_receipt_v3",
        include_str!("../schemas/evidence_verification_receipt_v3.schema.json"),
    ),
    (
        "judge_analysis_policy",
        include_str!("../schemas/judge_analysis_policy.schema.json"),
    ),
    (
        "judge_measurement_evidence",
        include_str!("../schemas/judge_measurement_evidence.schema.json"),
    ),
    (
        "judge_measurement_plan",
        include_str!("../schemas/judge_measurement_plan.schema.json"),
    ),
    (
        "judge_measurement_recipient_policy",
        include_str!("../schemas/judge_measurement_recipient_policy.schema.json"),
    ),
    (
        "judge_measurement_verification_receipt",
        include_str!("../schemas/judge_measurement_verification_receipt.schema.json"),
    ),
    (
        "judge_measurements",
        include_str!("../schemas/judge_measurements.schema.json"),
    ),
    (
        "judge_verification_result",
        include_str!("../schemas/judge_verification_result.schema.json"),
    ),
    (
        "model_artifact_identity",
        include_str!("../schemas/model_artifact_identity.schema.json"),
    ),
    (
        "multi_metric_comparison",
        include_str!("../schemas/multi_metric_comparison.schema.json"),
    ),
    (
        "normalized_captured_request",
        include_str!("../schemas/normalized_captured_request.schema.json"),
    ),
    (
        "recipient_acceptance_policy",
        include_str!("../schemas/recipient_acceptance_policy.schema.json"),
    ),
    (
        "runtime_behavioral_schedule",
        include_str!("../schemas/runtime_behavioral_schedule.schema.json"),
    ),
    (
        "runtime_manifest",
        include_str!("../schemas/runtime_manifest.schema.json"),
    ),
    (
        "runtime_provider_receipt",
        include_str!("../schemas/runtime_provider_receipt.schema.json"),
    ),
    (
        "runtime_scoring_observation",
        include_str!("../schemas/runtime_scoring_observation.schema.json"),
    ),
    (
        "scorer_extension_binding",
        include_str!("../schemas/scorer_extension_binding.schema.json"),
    ),
    (
        "scorer_extension_descriptor",
        include_str!("../schemas/scorer_extension_descriptor.schema.json"),
    ),
    (
        "scorer_extension_result",
        include_str!("../schemas/scorer_extension_result.schema.json"),
    ),
    (
        "trust_inputs",
        include_str!("../schemas/trust_inputs.schema.json"),
    ),
    (
        "trust_inputs_v2",
        include_str!("../schemas/trust_inputs_v2.schema.json"),
    ),
];
pub fn schema_names() -> impl Iterator<Item = &'static str> {
    SCHEMAS.iter().map(|(name, _)| *name)
}
fn canonical_name(name: &str) -> &str {
    match name {
        "run" => "evaluation_run",
        "case_set" => "evaluation_case_set",
        "policy" => "comparison_policy",
        "comparison" => "multi_metric_comparison",
        _ => name.strip_suffix(".schema.json").unwrap_or(name),
    }
}
struct Compiled {
    validator: jsonschema::Validator,
    schema: Value,
}
type Cache = Mutex<HashMap<String, Arc<Compiled>>>;
static VALIDATORS: OnceLock<Cache> = OnceLock::new();
fn validator(name: &str) -> Result<Arc<Compiled>> {
    let name = canonical_name(name);
    let cache = VALIDATORS.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let locked = cache
            .lock()
            .map_err(|_| ContractError::new("schema_internal", "schema cache is unavailable"))?;
        if let Some(value) = locked.get(name) {
            return Ok(value.clone());
        }
    }
    let raw = SCHEMAS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, raw)| raw)
        .ok_or_else(|| {
            ContractError::new(
                "unknown_schema",
                format!("unknown contract schema {name:?}"),
            )
        })?;
    let schema: Value = serde_json::from_str(raw)
        .map_err(|e| ContractError::new("schema_internal", e.to_string()))?;
    // Python Draft202012Validator does not assert formats without an explicit
    // format checker. External resolution is disabled at dependency level.
    let compiled = crate::numeric::options().build(&schema).map_err(|e| {
        ContractError::new(
            "schema_internal",
            format!("invalid packaged schema {name}: {e}"),
        )
    })?;
    let compiled = Arc::new(Compiled {
        validator: compiled,
        schema,
    });
    cache
        .lock()
        .map_err(|_| ContractError::new("schema_internal", "schema cache is unavailable"))?
        .insert(name.into(), compiled.clone());
    Ok(compiled)
}
struct SizeLimit {
    size: usize,
}
impl Write for SizeLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.size = self.size.saturating_add(bytes.len());
        if self.size > MAX_INPUT_BYTES {
            return Err(io::Error::other("canonical size limit exceeded"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn abbreviate(text: &str, maximum: usize) -> String {
    if text.chars().count() <= maximum {
        return text.into();
    }
    let marker = " ... [truncated] ... ";
    let head = (maximum - marker.len()) / 2;
    let tail = maximum - marker.len() - head;
    let chars: Vec<_> = text.chars().collect();
    chars[..head]
        .iter()
        .chain(marker.chars().collect::<Vec<_>>().iter())
        .chain(chars[chars.len() - tail..].iter())
        .collect()
}
/// Validate a packaged Draft 2020-12 contract. Record aliases include the same
/// record-count and encoded-size preflight as evaluation_record_contracts.
/// Diagnostics follow the source validator ordering and bounded renderer.
pub fn validate_schema(value: &Value, name: &str) -> Result<()> {
    let resolved = canonical_name(name);
    if matches!(
        resolved,
        "evaluation_run"
            | "evaluation_case_set"
            | "comparison_policy"
            | "multi_metric_comparison"
            | "evaluator_export"
    ) {
        let key = match resolved {
            "evaluation_run" => Some("records"),
            "evaluation_case_set" => Some("cases"),
            _ => None,
        };
        if let Some((key, records)) =
            key.and_then(|key| value.get(key).and_then(Value::as_array).map(|v| (key, v)))
        {
            if records.len() > MAX_RECORDS {
                return Err(ContractError::new(
                    "record_limit",
                    format!(
                        "invalid {name}: {key} is too long: {} records; maximum {MAX_RECORDS}",
                        records.len()
                    ),
                ));
            }
        }
        let mut sink = SizeLimit { size: 0 };
        if let Err(error) = canonical_json_to_writer(value, &mut sink, true) {
            if sink.size > MAX_INPUT_BYTES {
                return Err(ContractError::new(
                    "input_limit",
                    format!("invalid {name}: {name} exceeds the {MAX_INPUT_BYTES} byte limit"),
                ));
            }
            return Err(error);
        }
    }
    let validator = validator(name)?;
    if let Some(error) = validator
        .validator
        .iter_errors(value)
        .min_by_key(|error| crate::diagnostic::priority(error, &validator.schema))
    {
        let path = instance_path(&error.instance_path.to_string()).join(".");
        let prefix = abbreviate(
            &format!("invalid {name} {}: ", abbreviate(&path, 800)),
            1200,
        );
        let reason = abbreviate(
            &crate::diagnostic::message(&error, &validator.schema),
            2400 - prefix.chars().count(),
        );
        return Err(ContractError::new(
            "schema_validation",
            format!("{prefix}{reason}"),
        ));
    }
    Ok(())
}

/// Source-facing diagnostic without a caller-specific error envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaViolation {
    pub path: Vec<String>,
    pub message: String,
}
#[derive(Debug, Clone, Copy)]
pub enum SchemaErrorOrder {
    Schema,
    Instance,
}
pub fn first_schema_error(
    value: &Value,
    name: &str,
    order: SchemaErrorOrder,
) -> Result<Option<SchemaViolation>> {
    let compiled = validator(name)?;
    let errors = compiled.validator.iter_errors(value);
    let selected = match order {
        SchemaErrorOrder::Schema => {
            errors.min_by_key(|e| crate::diagnostic::priority(e, &compiled.schema))
        }
        SchemaErrorOrder::Instance => {
            errors.min_by_key(|e| instance_path(&e.instance_path.to_string()))
        }
    };
    Ok(selected.map(|e| SchemaViolation {
        path: instance_path(&e.instance_path.to_string()),
        message: crate::diagnostic::message(&e, &compiled.schema),
    }))
}
/// Validate an immutable packaged subschema with its root definitions retained.
/// Compiled fragments are shared across calls and cannot resolve remote schemas.
pub fn first_schema_error_fragment(
    value: &Value,
    name: &str,
    pointer: &str,
    order: SchemaErrorOrder,
) -> Result<Option<SchemaViolation>> {
    let key = format!("fragment:{}:{pointer}", canonical_name(name));
    let cache = VALIDATORS.get_or_init(|| Mutex::new(HashMap::new()));
    let existing = cache
        .lock()
        .map_err(|_| ContractError::new("schema_internal", "schema cache is unavailable"))?
        .get(&key)
        .cloned();
    let compiled = match existing {
        Some(compiled) => compiled,
        None => {
            let root = packaged_schema(name)?;
            let mut schema = root.pointer(pointer).cloned().ok_or_else(|| {
                ContractError::new(
                    "unknown_schema",
                    format!("unknown contract fragment {name:?} {pointer:?}"),
                )
            })?;
            let object = schema.as_object_mut().ok_or_else(|| {
                ContractError::new(
                    "schema_internal",
                    "contract fragment must be a schema object",
                )
            })?;
            if let Some(defs) = root.get("$defs") {
                object.insert("$defs".into(), defs.clone());
            }
            let validator = crate::numeric::options()
                .build(&schema)
                .map_err(|e| ContractError::new("schema_internal", e.to_string()))?;
            let compiled = Arc::new(Compiled { validator, schema });
            cache
                .lock()
                .map_err(|_| ContractError::new("schema_internal", "schema cache is unavailable"))?
                .insert(key, compiled.clone());
            compiled
        }
    };
    let errors = compiled.validator.iter_errors(value);
    let selected = match order {
        SchemaErrorOrder::Schema => {
            errors.min_by_key(|e| crate::diagnostic::priority(e, &compiled.schema))
        }
        SchemaErrorOrder::Instance => {
            errors.min_by_key(|e| instance_path(&e.instance_path.to_string()))
        }
    };
    Ok(selected.map(|e| SchemaViolation {
        path: instance_path(&e.instance_path.to_string()),
        message: crate::diagnostic::message(&e, &compiled.schema),
    }))
}
fn instance_path(pointer: &str) -> Vec<String> {
    if pointer.is_empty() {
        Vec::new()
    } else {
        pointer
            .trim_start_matches('/')
            .split('/')
            .map(|s| s.replace("~1", "/").replace("~0", "~"))
            .collect()
    }
}

/// Return an owned copy of an embedded authoritative contract for a trusted
/// internal template projection. Mutating it cannot alter cached validators.
pub fn packaged_schema(name: &str) -> Result<Value> {
    let name = canonical_name(name);
    let raw = SCHEMAS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, raw)| raw)
        .ok_or_else(|| {
            ContractError::new(
                "unknown_schema",
                format!("unknown contract schema {name:?}"),
            )
        })?;
    serde_json::from_str(raw).map_err(|e| ContractError::new("schema_internal", e.to_string()))
}
/// Validate a trusted internal schema projection without enabling file or HTTP
/// reference retrieval. Returns the same source-facing diagnostic representation.
pub fn first_schema_error_value(
    value: &Value,
    schema: &Value,
    order: SchemaErrorOrder,
) -> Result<Option<SchemaViolation>> {
    let compiled = crate::numeric::options()
        .build(schema)
        .map_err(|e| ContractError::new("schema_internal", e.to_string()))?;
    let errors = compiled.iter_errors(value);
    let selected = match order {
        SchemaErrorOrder::Schema => errors.min_by_key(|e| crate::diagnostic::priority(e, schema)),
        SchemaErrorOrder::Instance => {
            errors.min_by_key(|e| instance_path(&e.instance_path.to_string()))
        }
    };
    Ok(selected.map(|e| SchemaViolation {
        path: instance_path(&e.instance_path.to_string()),
        message: crate::diagnostic::message(&e, schema),
    }))
}

/// Python's breadth-first context diagnostic for authored request schemas.
pub fn first_schema_error_details(value: &Value, name: &str) -> Result<Option<String>> {
    let compiled = validator(name)?;
    let selected = compiled.validator.iter_errors(value).min_by_key(|e| {
        (
            instance_path(&e.instance_path.to_string()),
            crate::diagnostic::priority(e, &compiled.schema),
        )
    });
    let Some(error) = selected else {
        return Ok(None);
    };
    let mut pending = std::collections::VecDeque::from([&error]);
    let mut details = Vec::new();
    while let Some(error) = pending.pop_front() {
        let path = instance_path(&error.instance_path.to_string()).join(".");
        let message = crate::diagnostic::message(error, &compiled.schema);
        let message = if path.is_empty() {
            message
        } else {
            format!("{path}: {message}")
        };
        if !details.contains(&message) {
            details.push(message);
        }
        match &error.kind {
            jsonschema::error::ValidationErrorKind::AnyOf { context }
            | jsonschema::error::ValidationErrorKind::OneOfNotValid { context } => {
                for branch in context {
                    let mut errors = branch.iter().collect::<Vec<_>>();
                    errors.sort_by_key(|e| crate::diagnostic::priority(e, &compiled.schema));
                    pending.extend(errors);
                }
            }
            _ => {}
        }
    }
    Ok(Some(details.join("; ")))
}
