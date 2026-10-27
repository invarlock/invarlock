use crate::{canonical::python_float, ContractError, Result, Value};
use serde_json::{map::Entry, Map, Number};
use std::str::FromStr;

// Bounded independently of process stack settings. Python's recursion ceiling is
// environment-dependent; this public native boundary supports 900 containers.
pub const MAX_JSON_DEPTH: usize = 900;

pub fn parse_json(raw: &[u8], label: &str) -> Result<Value> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| ContractError::new("invalid_utf8", format!("{label} is not UTF-8 JSON")))?;
    let mut parser = Parser {
        text,
        offset: 0,
        label,
    };
    let value = parser.value(0)?;
    parser.space();
    if parser.offset != text.len() {
        return Err(parser.invalid());
    }
    Ok(value)
}
// Most contract objects have only a handful of fields. Growing IndexMap's entry
// vector geometrically retains unused Value-sized slots in every such object.
// Buffer a bounded number of fields so these maps can reserve their exact size;
// a wide object switches to the ordinary map before buffering can grow large.
const BUFFERED_OBJECT_FIELDS: usize = 32;
enum ObjectFields {
    Buffered(Vec<(String, Value)>),
    Wide(Map<String, Value>),
}
impl ObjectFields {
    fn new() -> Self {
        // Start below Vec's four-slot first growth for one- and two-field objects.
        Self::Buffered(Vec::with_capacity(3))
    }
    fn insert(&mut self, key: String, value: Value, duplicate: &mut Option<String>) {
        if let Self::Buffered(fields) = self {
            if fields.len() < BUFFERED_OBJECT_FIELDS {
                fields.push((key, value));
                return;
            }
            let mut map = Map::with_capacity(fields.len() + 1);
            for (key, value) in std::mem::take(fields) {
                insert_field(&mut map, key, value, duplicate);
            }
            *self = Self::Wide(map);
        }
        let Self::Wide(map) = self else {
            unreachable!()
        };
        insert_field(map, key, value, duplicate);
    }
    fn finish(self, duplicate: &mut Option<String>) -> Map<String, Value> {
        match self {
            Self::Wide(map) => map,
            Self::Buffered(fields) => {
                let mut map = Map::with_capacity(fields.len());
                for (key, value) in fields {
                    insert_field(&mut map, key, value, duplicate);
                }
                map
            }
        }
    }
}
fn insert_field(
    map: &mut Map<String, Value>,
    key: String,
    value: Value,
    duplicate: &mut Option<String>,
) {
    match map.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(value);
        }
        Entry::Occupied(mut entry) => {
            if duplicate.is_none() {
                *duplicate = Some(entry.key().clone());
            }
            entry.insert(value);
        }
    }
}

struct Parser<'a> {
    text: &'a str,
    offset: usize,
    label: &'a str,
}
impl Parser<'_> {
    fn invalid(&self) -> ContractError {
        ContractError::new("invalid_json", format!("{} is not valid JSON", self.label))
    }
    fn space(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.offset += 1;
        }
    }
    fn consume(&mut self, byte: u8) -> bool {
        if self.text.as_bytes().get(self.offset) == Some(&byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn value(&mut self, _depth: usize) -> Result<Value> {
        enum Frame {
            Array(Vec<Value>),
            Object {
                values: ObjectFields,
                key: String,
                duplicate: Option<String>,
            },
        }
        let mut frames = Vec::new();
        'next_value: loop {
            self.space();
            if frames.len() > MAX_JSON_DEPTH {
                return Err(ContractError::new(
                    "json_depth",
                    format!(
                        "{} is not valid JSON: nesting exceeds {MAX_JSON_DEPTH}",
                        self.label
                    ),
                ));
            }
            let rest = &self.text[self.offset..];
            for constant in ["NaN", "Infinity", "-Infinity"] {
                if rest.starts_with(constant) {
                    return Err(ContractError::new(
                        "nonstandard_constant",
                        format!("JSON contains non-standard constant '{constant}'"),
                    ));
                }
            }
            let mut value = match rest.as_bytes().first() {
                Some(b'"') => Value::String(self.string()?),
                Some(b'{') => {
                    self.offset += 1;
                    self.space();
                    if self.consume(b'}') {
                        Value::Object(Map::new())
                    } else {
                        let key = self.string()?;
                        self.space();
                        if !self.consume(b':') {
                            return Err(self.invalid());
                        }
                        frames.push(Frame::Object {
                            values: ObjectFields::new(),
                            key,
                            duplicate: None,
                        });
                        continue 'next_value;
                    }
                }
                Some(b'[') => {
                    self.offset += 1;
                    self.space();
                    if self.consume(b']') {
                        Value::Array(Vec::new())
                    } else {
                        frames.push(Frame::Array(Vec::new()));
                        continue 'next_value;
                    }
                }
                Some(b't') if rest.starts_with("true") => {
                    self.offset += 4;
                    Value::Bool(true)
                }
                Some(b'f') if rest.starts_with("false") => {
                    self.offset += 5;
                    Value::Bool(false)
                }
                Some(b'n') if rest.starts_with("null") => {
                    self.offset += 4;
                    Value::Null
                }
                Some(b'-' | b'0'..=b'9') => self.number()?,
                _ => return Err(self.invalid()),
            };
            loop {
                let Some(frame) = frames.last_mut() else {
                    return Ok(value);
                };
                self.space();
                match frame {
                    Frame::Array(values) => {
                        values.push(value);
                        if self.consume(b']') {
                            let Some(Frame::Array(values)) = frames.pop() else {
                                unreachable!()
                            };
                            value = Value::Array(values.into_boxed_slice().into_vec());
                        } else if self.consume(b',') {
                            continue 'next_value;
                        } else {
                            return Err(self.invalid());
                        }
                    }
                    Frame::Object {
                        values,
                        key,
                        duplicate,
                    } => {
                        values.insert(std::mem::take(key), value, duplicate);
                        if self.consume(b'}') {
                            let Some(Frame::Object {
                                values,
                                mut duplicate,
                                ..
                            }) = frames.pop()
                            else {
                                unreachable!()
                            };
                            let values = values.finish(&mut duplicate);
                            if let Some(key) = duplicate {
                                return Err(ContractError::new(
                                    "duplicate_key",
                                    format!("JSON object has duplicate key {}", python_repr(&key)),
                                ));
                            }
                            value = Value::Object(values);
                        } else if self.consume(b',') {
                            self.space();
                            *key = self.string()?;
                            self.space();
                            if !self.consume(b':') {
                                return Err(self.invalid());
                            }
                            continue 'next_value;
                        } else {
                            return Err(self.invalid());
                        }
                    }
                }
            }
        }
    }
    fn string(&mut self) -> Result<String> {
        let start = self.offset;
        if !self.consume(b'"') {
            return Err(self.invalid());
        }
        let bytes = self.text.as_bytes();
        let mut escaped = false;
        while self.offset < bytes.len() {
            let b = bytes[self.offset];
            self.offset += 1;
            match b {
                b'"' => {
                    if !escaped {
                        return Ok(self.text[start + 1..self.offset - 1].into());
                    }
                    let raw = &self.text[start..self.offset];
                    if unpaired_surrogate(raw) {
                        return Err(ContractError::new("unpaired_surrogate", format!("{} has an unpaired surrogate with no canonical UTF-8 representation", self.label)));
                    }
                    return serde_json::from_str(raw).map_err(|error| {
                        if error.to_string().contains("surrogate") {
                            ContractError::new("unpaired_surrogate", format!("{} has an unpaired surrogate with no canonical UTF-8 representation", self.label))
                        } else { self.invalid() }
                    });
                }
                b'\\' => {
                    escaped = true;
                    if self.offset >= bytes.len() {
                        return Err(self.invalid());
                    }
                    self.offset += 1;
                }
                0..=31 => return Err(self.invalid()),
                _ => {}
            }
        }
        Err(self.invalid())
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.offset;
        self.consume(b'-');
        if !self.consume(b'0') {
            if !self
                .text
                .as_bytes()
                .get(self.offset)
                .is_some_and(|b| matches!(b, b'1'..=b'9'))
            {
                return Err(self.invalid());
            }
            self.digits();
        }
        let mut float = false;
        if self.consume(b'.') {
            float = true;
            let before = self.offset;
            self.digits();
            if before == self.offset {
                return Err(self.invalid());
            }
        }
        if self.consume(b'e') || self.consume(b'E') {
            float = true;
            if !self.consume(b'+') {
                self.consume(b'-');
            }
            let before = self.offset;
            self.digits();
            if before == self.offset {
                return Err(self.invalid());
            }
        }
        let raw = &self.text[start..self.offset];
        let normalized;
        let token = if float {
            let value = raw.parse::<f64>().map_err(|_| self.invalid())?;
            if !value.is_finite() {
                return Err(ContractError::new(
                    "nonfinite_number",
                    format!("JSON contains non-finite number '{raw}'"),
                ));
            }
            normalized = python_float(value)?;
            normalized.as_str()
        } else {
            // CPython's standard integer string conversion limit is 4300 digits.
            if raw.trim_start_matches('-').len() > 4300 {
                return Err(self.invalid());
            }
            if raw == "-0" {
                "0"
            } else {
                raw
            }
        };
        Number::from_str(token)
            .map(Value::Number)
            .map_err(|_| self.invalid())
    }
    fn digits(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
        }
    }
}
pub(crate) fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::new();
    result.push(quote);
    for c in value.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c == quote => {
                result.push('\\');
                result.push(c);
            }
            c if !crate::printable::is_printable(c) => {
                let code = c as u32;
                result.push_str(&if code <= 255 {
                    format!("\\x{code:02x}")
                } else if code <= 0xffff {
                    format!("\\u{code:04x}")
                } else {
                    format!("\\U{code:08x}")
                });
            }
            _ => result.push(c),
        }
    }
    result.push(quote);
    result
}

fn unpaired_surrogate(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    let mut i = 1;
    while i + 1 < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        if bytes.get(i + 1) != Some(&b'u') {
            i += 2;
            continue;
        }
        let Some(code) = bytes
            .get(i + 2..i + 6)
            .and_then(|s| std::str::from_utf8(s).ok())
            .and_then(|s| u16::from_str_radix(s, 16).ok())
        else {
            return false;
        };
        i += 6;
        if (0xdc00..=0xdfff).contains(&code) {
            return true;
        }
        if (0xd800..=0xdbff).contains(&code) {
            if bytes.get(i..i + 2) != Some(b"\\u") {
                return true;
            }
            let low = bytes
                .get(i + 2..i + 6)
                .and_then(|s| std::str::from_utf8(s).ok())
                .and_then(|s| u16::from_str_radix(s, 16).ok());
            if !low.is_some_and(|n| (0xdc00..=0xdfff).contains(&n)) {
                return true;
            }
            i += 6;
        }
    }
    false
}
