//! Portable authored request contracts; YAML decoding and provider discovery are adapters.
use crate::{first_schema_error_details, python_repr, ContractError, Result, Value};
const CAPTURED: &str = "invarlock/evaluation-request-v2";
const RUNTIME: &str = "invarlock/evaluation-request-v1";
fn error(message: impl Into<String>) -> ContractError {
    ContractError::new("evaluation_request", message)
}
pub fn reject_include_directives(value: &Value) -> Result<()> {
    let captured = value.get("format_version").is_some_and(|v| v == CAPTURED);
    let mut pending = vec![(Vec::<String>::new(), value)];
    while let Some((path, current)) = pending.pop() {
        if captured
            && path.len() == 4
            && path[0] == "comparison"
            && ["baseline", "subject"].contains(&path[1].as_str())
            && path[2] == "service_identity"
            && path[3] == "configuration"
        {
            continue;
        }
        if let Some(object) = current.as_object() {
            for (key, child) in object {
                if [
                    "include",
                    "includes",
                    "include_file",
                    "include_files",
                    "extends",
                    "ref",
                ]
                .contains(&key.to_lowercase().as_str())
                {
                    return Err(error("request YAML include directives are not allowed"));
                }
                let mut child_path = path.clone();
                child_path.push(key.clone());
                pending.push((child_path, child));
            }
        } else if let Some(array) = current.as_array() {
            pending.extend(array.iter().map(|child| (path.clone(), child)));
        }
    }
    Ok(())
}
pub fn validate_schema(value: &Value, captured: bool) -> Result<()> {
    let name = if captured {
        "evaluation_request_v2"
    } else {
        "evaluation_request"
    };
    if let Some(message) = first_schema_error_details(value, name)? {
        return Err(error(format!(
            "request does not match {name}.schema.json: {message}"
        )));
    }
    Ok(())
}
pub fn validate_authored(value: &Value) -> Result<()> {
    reject_include_directives(value)?;
    if !value.is_object() {
        return Err(error("request must be a YAML object"));
    }
    let format = value.get("format_version").unwrap_or(&Value::Null);
    match format.as_str() {
        Some(CAPTURED) => validate_schema(value, true),
        Some(RUNTIME) => validate_schema(value, false),
        Some("invarlock/evaluation-request-v3") => {
            if let Some(message) = first_schema_error_details(value, "evaluation_request_v3")? {
                return Err(error(format!(
                    "judge request is invalid: {}",
                    message.chars().take(240).collect::<String>()
                )));
            }
            Ok(())
        }
        _ => Err(error(format!(
            "unsupported evaluation request format {}",
            python_repr(format)
        ))),
    }
}
pub fn mode(value: &Value) -> Result<&str> {
    if !value.is_object() {
        return Err(error("request must be a YAML object"));
    }
    let format = value.get("format_version").unwrap_or(&Value::Null);
    let (modes, label) = match format.as_str() {
        Some(CAPTURED) => (["captured", "captured"], "captured"),
        Some(RUNTIME) => (["run", "import"], "runtime"),
        Some("invarlock/evaluation-request-v3") => (["judge_import", "judge_collect"], "judge"),
        _ => {
            return Err(error(format!(
                "unsupported evaluation request format {}",
                python_repr(format)
            )))
        }
    };
    let execution = value.get("execution").and_then(Value::as_object);
    let mode = execution
        .and_then(|v| v.get("mode"))
        .and_then(Value::as_str);
    mode.filter(|mode| modes.contains(mode))
        .ok_or_else(|| error(format!("{label} request execution mode is invalid")))
}
pub fn reference_parts<'a>(reference: &'a str, label: &str) -> Result<Vec<&'a str>> {
    let bytes = reference.as_bytes();
    let parts = reference
        .split('/')
        .filter(|p| *p != ".")
        .collect::<Vec<_>>();
    if reference.is_empty()
        || reference.starts_with(['/', '~', '\\'])
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || reference.contains("://")
        || reference.contains('\\')
        || reference.chars().any(|c| c < ' ' || c == '\u{7f}')
        || parts.is_empty()
        || parts.iter().any(|p| p.is_empty() || *p == "..")
        || reference.contains("//")
        || reference.ends_with('/')
    {
        return Err(error(format!("{label} must be a safe relative reference")));
    }
    Ok(parts)
}
