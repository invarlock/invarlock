use crate::{ContractError, Result, Value};
use std::io::Write;

/// Python-compatible finite float representation, including exponent sign and
/// zero padding, fixed/scientific cutovers, integral floats, and negative zero.
pub(crate) fn python_float(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(ContractError::new(
            "nonfinite_number",
            "value is not canonical JSON: Out of range float values are not JSON compliant",
        ));
    }
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(value);
    let (sign, text) = text.strip_prefix('-').map_or(("", text), |s| ("-", s));
    let (mantissa, exp) = text.split_once('e').map_or((text, 0), |(m, e)| {
        (m, e.parse::<i32>().expect("ryu exponent"))
    });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let leading = digits.bytes().take_while(|b| *b == b'0').count();
    if leading == digits.len() {
        return Ok(format!("{sign}0.0"));
    }
    digits.drain(..leading);
    let exponent = decimal + exp - leading as i32 - 1;
    while digits.ends_with('0') && digits.len() > 1 {
        digits.pop();
    }
    if !(-4..16).contains(&exponent) {
        let mut result = format!("{sign}{}", &digits[..1]);
        if digits.len() > 1 {
            result.push('.');
            result.push_str(&digits[1..]);
        }
        result.push('e');
        result.push(if exponent < 0 { '-' } else { '+' });
        result.push_str(&format!("{:02}", exponent.abs()));
        return Ok(result);
    }
    let point = exponent + 1;
    if point <= 0 {
        Ok(format!(
            "{sign}0.{}{}",
            "0".repeat((-point) as usize),
            digits
        ))
    } else if point as usize >= digits.len() {
        Ok(format!(
            "{sign}{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        ))
    } else {
        let p = point as usize;
        Ok(format!("{sign}{}.{}", &digits[..p], &digits[p..]))
    }
}

pub fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    canonical_json_to_writer(value, &mut out, true)?;
    Ok(out)
}
pub fn canonical_json_no_newline(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    canonical_json_to_writer(value, &mut out, false)?;
    Ok(out)
}

/// Encode directly to a bounded sink or hash writer without a document-sized
/// temporary allocation. Object sorting allocates references only.
pub fn canonical_json_to_writer(
    value: &Value,
    writer: &mut impl Write,
    newline: bool,
) -> Result<()> {
    fn put(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
        writer.write_all(bytes).map_err(|e| {
            ContractError::new("canonical_io", format!("value is not canonical JSON: {e}"))
        })
    }
    enum Action<'a> {
        Value(&'a Value),
        Bytes(&'static [u8]),
        Key(&'a str),
    }
    let mut stack = vec![Action::Value(value)];
    while let Some(action) = stack.pop() {
        match action {
            Action::Bytes(bytes) => put(writer, bytes)?,
            Action::Key(key) => serde_json::to_writer(&mut *writer, key)
                .map_err(|e| ContractError::new("canonical_io", e.to_string()))?,
            Action::Value(value) => match value {
                Value::Null => put(writer, b"null")?,
                Value::Bool(v) => put(writer, if *v { b"true" } else { b"false" })?,
                Value::String(v) => serde_json::to_writer(&mut *writer, v)
                    .map_err(|e| ContractError::new("canonical_io", e.to_string()))?,
                Value::Number(n) => {
                    let raw = n.to_string();
                    if raw.contains(['.', 'e', 'E']) {
                        let f = raw.parse::<f64>().map_err(|_| {
                            ContractError::new(
                                "nonfinite_number",
                                "value is not canonical JSON: invalid number",
                            )
                        })?;
                        put(writer, python_float(f)?.as_bytes())?;
                    } else {
                        put(writer, if raw == "-0" { b"0" } else { raw.as_bytes() })?;
                    }
                }
                Value::Array(values) => {
                    put(writer, b"[")?;
                    stack.push(Action::Bytes(b"]"));
                    for (i, v) in values.iter().enumerate().rev() {
                        stack.push(Action::Value(v));
                        if i != 0 {
                            stack.push(Action::Bytes(b","));
                        }
                    }
                }
                Value::Object(values) => {
                    put(writer, b"{")?;
                    stack.push(Action::Bytes(b"}"));
                    let mut keys: Vec<_> = values.keys().collect();
                    keys.sort_unstable(); // UTF-8 lexical order is Unicode scalar order.
                    for (i, key) in keys.iter().enumerate().rev() {
                        stack.push(Action::Value(&values[*key]));
                        stack.push(Action::Bytes(b":"));
                        stack.push(Action::Key(key));
                        if i != 0 {
                            stack.push(Action::Bytes(b","));
                        }
                    }
                }
            },
        }
    }
    if newline {
        put(writer, b"\n")?;
    }
    Ok(())
}
