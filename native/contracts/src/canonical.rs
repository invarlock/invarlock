use crate::{ContractError, Result, Value};
use sha2::{Digest, Sha256};
use std::io::{self, BufWriter, Write};

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

/// Compare the complete canonical encoding, including its final newline, with
/// existing bytes without allocating a second document-sized buffer. A mismatch
/// never stops traversal: invalid numbers retain precedence over inequality.
pub fn canonical_json_matches(value: &Value, raw: &[u8]) -> Result<bool> {
    struct MatchWriter<'a> {
        remaining: &'a [u8],
        matched: bool,
    }
    impl Write for MatchWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.matched {
                if let Some(remaining) = self.remaining.strip_prefix(bytes) {
                    self.remaining = remaining;
                } else {
                    self.matched = false;
                }
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = MatchWriter {
        remaining: raw,
        matched: true,
    };
    canonical_json_to_writer(value, &mut writer, true)?;
    Ok(writer.matched && writer.remaining.is_empty())
}

/// SHA-256 identity of the complete canonical encoding, including its final
/// newline. Buffer small encoding writes before updating the digest.
pub fn canonical_json_digest(value: &Value) -> Result<String> {
    struct DigestWriter<'a>(&'a mut Sha256);
    impl Write for DigestWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut digest = Sha256::new();
    {
        let mut writer = BufWriter::with_capacity(64 * 1024, DigestWriter(&mut digest));
        canonical_json_to_writer(value, &mut writer, true)?;
        writer.flush().map_err(|e| {
            ContractError::new("canonical_io", format!("value is not canonical JSON: {e}"))
        })?;
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

/// Encode directly to a bounded sink or hash writer without a document-sized
/// temporary allocation. Pending traversal frames are bounded by nesting depth;
/// each active object additionally retains its sorted entry references.
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
        Array {
            values: &'a [Value],
            index: usize,
        },
        Object {
            entries: std::vec::IntoIter<(&'a str, &'a Value)>,
            first: bool,
        },
    }
    let mut stack = vec![Action::Value(value)];
    while let Some(action) = stack.pop() {
        match action {
            Action::Array { values, index } => {
                if let Some(value) = values.get(index) {
                    if index != 0 {
                        put(writer, b",")?;
                    }
                    stack.push(Action::Array {
                        values,
                        index: index + 1,
                    });
                    stack.push(Action::Value(value));
                } else {
                    put(writer, b"]")?;
                }
            }
            Action::Object { mut entries, first } => {
                if let Some((key, value)) = entries.next() {
                    if !first {
                        put(writer, b",")?;
                    }
                    serde_json::to_writer(&mut *writer, key)
                        .map_err(|e| ContractError::new("canonical_io", e.to_string()))?;
                    put(writer, b":")?;
                    stack.push(Action::Object {
                        entries,
                        first: false,
                    });
                    stack.push(Action::Value(value));
                } else {
                    put(writer, b"}")?;
                }
            }
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
                    stack.push(Action::Array { values, index: 0 });
                }
                Value::Object(values) => {
                    put(writer, b"{")?;
                    let mut entries: Vec<_> = values
                        .iter()
                        .map(|(key, value)| (key.as_str(), value))
                        .collect();
                    // UTF-8 lexical order is Unicode scalar order.
                    entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
                    stack.push(Action::Object {
                        entries: entries.into_iter(),
                        first: true,
                    });
                }
            },
        }
    }
    if newline {
        put(writer, b"\n")?;
    }
    Ok(())
}
