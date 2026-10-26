//! Python jsonschema's public reason wording for packaged contract keywords.
use crate::{canonical::python_float, parser::python_repr, Value};
use jsonschema::{
    error::{TypeKind, ValidationErrorKind as K},
    ValidationError,
};
pub(crate) fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => python_repr(s),
        Value::Number(n) => {
            let raw = n.to_string();
            if raw.contains(['.', 'e', 'E']) {
                n.as_f64().and_then(|v| python_float(v).ok()).unwrap_or(raw)
            } else {
                raw
            }
        }
        Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", python_repr(k), repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
pub(crate) fn message(error: &ValidationError<'_>, schema: &Value) -> String {
    let instance = || repr(&error.instance);
    match &error.kind {
        K::Required { property } => format!("{} is a required property", repr(property)),
        K::AdditionalProperties { unexpected } => {
            let mut keys = unexpected.clone();
            keys.sort();
            format!(
                "Additional properties are not allowed ({} {} unexpected)",
                keys.iter()
                    .map(|s| python_repr(s))
                    .collect::<Vec<_>>()
                    .join(", "),
                if keys.len() == 1 { "was" } else { "were" }
            )
        }
        K::Constant { expected_value } => format!("{} was expected", repr(expected_value)),
        K::Enum { options } => format!("{} is not one of {}", instance(), repr(options)),
        K::Type { kind } => {
            let types = if let Some(types) = schema
                .pointer(error.schema_path.as_str())
                .and_then(Value::as_array)
            {
                types
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            } else {
                match kind {
                    TypeKind::Single(t) => vec![t.to_string()],
                    TypeKind::Multiple(t) => t.iter().map(|t| t.to_string()).collect(),
                }
            };
            format!(
                "{} is not of type {}",
                instance(),
                types
                    .iter()
                    .map(|s| python_repr(s))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        K::AnyOf { .. } | K::OneOfNotValid { .. } => {
            format!("{} is not valid under any of the given schemas", instance())
        }
        K::Minimum { limit } => {
            format!("{} is less than the minimum of {}", instance(), repr(limit))
        }
        K::Maximum { limit } => format!(
            "{} is greater than the maximum of {}",
            instance(),
            repr(limit)
        ),
        K::ExclusiveMinimum { limit } => format!(
            "{} is less than or equal to the minimum of {}",
            instance(),
            repr(limit)
        ),
        K::ExclusiveMaximum { limit } => format!(
            "{} is greater than or equal to the maximum of {}",
            instance(),
            repr(limit)
        ),
        K::MinLength { limit } | K::MinItems { limit } => format!(
            "{} {}",
            instance(),
            if *limit == 1 {
                "should be non-empty"
            } else {
                "is too short"
            }
        ),
        K::MaxLength { .. } | K::MaxItems { .. } => format!("{} is too long", instance()),
        K::MinProperties { limit } => format!(
            "{} {}",
            instance(),
            if *limit == 1 {
                "should be non-empty"
            } else {
                "does not have enough properties"
            }
        ),
        K::MaxProperties { .. } => format!("{} has too many properties", instance()),
        K::Pattern { pattern } => format!("{} does not match {}", instance(), python_repr(pattern)),
        K::Not { schema } => format!("{} should not be valid under {}", instance(), repr(schema)),
        K::UniqueItems => format!("{} has non-unique elements", instance()),
        K::PropertyNames { error } => message(error, schema),
        K::FalseSchema => format!("False schema does not allow {}", instance()),
        K::MultipleOf { multiple_of } => format!(
            "{} is not a multiple of {}",
            instance(),
            python_float(*multiple_of).unwrap_or_else(|_| multiple_of.to_string())
        ),
        _ => error.to_string(),
    }
}

/// Recover Python's schema insertion order when the compiled validator batches
/// object property validators in instance order. Array indices precede their
/// child schema positions, so an earlier record still owns the first error.
pub(crate) fn priority(error: &ValidationError<'_>, root: &Value) -> Vec<usize> {
    let schema_path = error.schema_path.to_string();
    let instance_path = error.instance_path.to_string();
    let parts: Vec<_> = schema_path.trim_start_matches('/').split('/').collect();
    let instance: Vec<_> = instance_path.trim_start_matches('/').split('/').collect();
    let mut node = root;
    let mut key = Vec::new();
    let mut consumed = 0;
    let mut previous = "";
    for part in parts {
        let decoded = part.replace("~1", "/").replace("~0", "~");
        if let Some(reference) = node
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|s| s.strip_prefix('#'))
        {
            if let Some(target) = root.pointer(reference) {
                node = target;
            }
        }
        if let Some(object) = node.as_object() {
            key.push(
                object
                    .keys()
                    .position(|k| k == &decoded)
                    .unwrap_or(usize::MAX),
            );
            node = object.get(&decoded).unwrap_or(&Value::Null);
        } else if let Some(array) = node.as_array() {
            let index = decoded.parse::<usize>().unwrap_or(usize::MAX);
            key.push(index);
            node = array.get(index).unwrap_or(&Value::Null);
        }
        if previous == "properties" {
            consumed += 1;
        }
        if part == "items" || previous == "prefixItems" {
            key.push(
                instance
                    .get(consumed)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
            );
            consumed += 1;
        }
        if part == "additionalProperties" && consumed < instance.len() {
            consumed += 1;
        }
        previous = part;
    }
    key
}
