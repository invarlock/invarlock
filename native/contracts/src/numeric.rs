//! JSON Schema numeric keywords with Python's arbitrary-integer semantics.
//! The upstream validator assumes every non-machine integer fits binary64.
use crate::{diagnostic::repr, Value};
use jsonschema::{Keyword, ValidationError};
use std::{cmp::Ordering, collections::HashSet};
fn integer(n: &serde_json::Number) -> bool {
    !n.to_string().contains(['.', 'e', 'E'])
}
fn int_cmp(left: &str, right: &str) -> Ordering {
    let ln = left.starts_with('-') && left != "-0";
    let rn = right.starts_with('-') && right != "-0";
    if ln != rn {
        return if ln {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let l = left.trim_start_matches('-').trim_start_matches('0');
    let r = right.trim_start_matches('-').trim_start_matches('0');
    let result = l.len().cmp(&r.len()).then_with(|| l.cmp(r));
    if ln {
        result.reverse()
    } else {
        result
    }
}
fn int_float_cmp(i: &str, f: f64) -> Ordering {
    if f == f64::INFINITY {
        return Ordering::Less;
    }
    if f == f64::NEG_INFINITY {
        return Ordering::Greater;
    }
    let truncated = format!("{:.0}", f.trunc());
    let result = int_cmp(i, &truncated);
    if result != Ordering::Equal {
        return result;
    }
    if f.fract() > 0.0 {
        Ordering::Less
    } else if f.fract() < 0.0 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}
pub(crate) fn number_cmp(
    left: &serde_json::Number,
    right: &serde_json::Number,
) -> Option<Ordering> {
    match (integer(left), integer(right)) {
        (true, true) => Some(int_cmp(&left.to_string(), &right.to_string())),
        (true, false) => right.as_f64().map(|f| int_float_cmp(&left.to_string(), f)),
        (false, true) => left
            .as_f64()
            .map(|f| int_float_cmp(&right.to_string(), f).reverse()),
        (false, false) => left.as_f64()?.partial_cmp(&right.as_f64()?),
    }
}
fn equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(l), Value::Number(r)) => number_cmp(l, r) == Some(Ordering::Equal),
        (Value::Array(l), Value::Array(r)) => {
            l.len() == r.len() && l.iter().zip(r).all(|(a, b)| equal(a, b))
        }
        (Value::Object(l), Value::Object(r)) => {
            l.len() == r.len() && l.iter().all(|(k, v)| r.get(k).is_some_and(|r| equal(v, r)))
        }
        _ => left == right,
    }
}
fn key(value: &Value, out: &mut String) {
    match value {
        Value::Number(n) => {
            out.push('n');
            if integer(n) {
                out.push_str(&n.to_string());
            } else if let Some(f) = n.as_f64() {
                if f == 0.0 {
                    out.push('0');
                } else if f.fract() == 0.0 {
                    out.push_str(&format!("{f:.0}"));
                } else {
                    out.push_str(&f.to_string());
                }
            } else {
                out.push_str(&n.to_string());
            }
            out.push(';');
        }
        Value::Array(a) => {
            out.push('[');
            for v in a {
                key(v, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            let mut entries: Vec<_> = o.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (k, v) in entries {
                out.push_str(&serde_json::to_string(k).unwrap());
                key(v, out);
            }
            out.push('}');
        }
        _ => {
            out.push_str(&serde_json::to_string(value).unwrap());
            out.push(';');
        }
    }
}
fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "integer" => value.as_number().is_some_and(|n| {
            integer(n)
                || n.as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }),
        "number" => value.is_number(),
        "string" => value.is_string(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}
struct NumericKeyword {
    name: &'static str,
    schema: Value,
}
impl NumericKeyword {
    fn message(&self, value: &Value) -> String {
        match self.name {
            "type" => {
                let types = if let Some(a) = self.schema.as_array() {
                    a.iter().map(repr).collect::<Vec<_>>().join(", ")
                } else {
                    repr(&self.schema)
                };
                format!("{} is not of type {types}", repr(value))
            }
            "minimum" => format!(
                "{} is less than the minimum of {}",
                repr(value),
                repr(&self.schema)
            ),
            "maximum" => format!(
                "{} is greater than the maximum of {}",
                repr(value),
                repr(&self.schema)
            ),
            "exclusiveMinimum" => format!(
                "{} is less than or equal to the minimum of {}",
                repr(value),
                repr(&self.schema)
            ),
            "exclusiveMaximum" => format!(
                "{} is greater than or equal to the maximum of {}",
                repr(value),
                repr(&self.schema)
            ),
            "const" => format!("{} was expected", repr(&self.schema)),
            "enum" => format!("{} is not one of {}", repr(value), repr(&self.schema)),
            "uniqueItems" => format!("{} has non-unique elements", repr(value)),
            _ => unreachable!(),
        }
    }
}
impl Keyword for NumericKeyword {
    fn validate<'i>(&self, instance: &'i Value) -> Result<(), ValidationError<'i>> {
        if self.is_valid(instance) {
            Ok(())
        } else {
            Err(ValidationError::custom(self.message(instance)))
        }
    }
    fn is_valid(&self, instance: &Value) -> bool {
        match self.name {
            "type" => {
                if let Some(k) = self.schema.as_str() {
                    matches_type(instance, k)
                } else {
                    self.schema.as_array().is_some_and(|types| {
                        types
                            .iter()
                            .any(|k| k.as_str().is_some_and(|k| matches_type(instance, k)))
                    })
                }
            }
            "const" => equal(instance, &self.schema),
            "enum" => self
                .schema
                .as_array()
                .is_some_and(|a| a.iter().any(|v| equal(v, instance))),
            "uniqueItems" => {
                if self.schema != true {
                    return true;
                }
                let Some(items) = instance.as_array() else {
                    return true;
                };
                let mut seen = HashSet::with_capacity(items.len());
                items.iter().all(|value| {
                    let mut encoded = String::new();
                    key(value, &mut encoded);
                    seen.insert(encoded)
                })
            }
            name => {
                let Some(value) = instance.as_number() else {
                    return true;
                };
                let Some(limit) = self.schema.as_number() else {
                    return false;
                };
                matches!(
                    (name, number_cmp(value, limit)),
                    ("minimum", Some(Ordering::Greater | Ordering::Equal))
                        | ("maximum", Some(Ordering::Less | Ordering::Equal))
                        | ("exclusiveMinimum", Some(Ordering::Greater))
                        | ("exclusiveMaximum", Some(Ordering::Less))
                )
            }
        }
    }
}
// The dependency factory requires its unboxed ValidationError result type.
#[allow(clippy::result_large_err)]
pub(crate) fn options() -> jsonschema::ValidationOptions {
    let mut options = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .should_validate_formats(false);
    for name in [
        "type",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "const",
        "enum",
        "uniqueItems",
    ] {
        options = options.with_keyword(name, move |_, schema, _| {
            Ok(Box::new(NumericKeyword {
                name,
                schema: schema.clone(),
            }))
        });
    }
    options
}
