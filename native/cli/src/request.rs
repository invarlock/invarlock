//! Bounded YAML ingestion, with the Python request loader's JSON scalar policy.
use invarlock_native_contracts as contracts;
use invarlock_native_evidence::{EvidenceError, Result};
use serde_json::{Map, Value};
use yaml_rust2::{
    parser::{Event, Parser},
    scanner::{Scanner, TScalarStyle, TokenType},
};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::incomplete(s)
}
fn scalar(s: String, style: TScalarStyle) -> Result<Value> {
    if style != TScalarStyle::Plain {
        return Ok(Value::String(s));
    }
    if ["true", "false", "null"].contains(&s.as_str()) {
        return Ok(serde_json::from_str(&s).unwrap());
    }
    if [
        "yes", "Yes", "YES", "no", "No", "NO", "on", "On", "ON", "off", "Off", "OFF", "True",
        "TRUE", "False", "FALSE",
    ]
    .contains(&s.as_str())
    {
        return Err(err(
            "request YAML booleans must use lowercase true or false",
        ));
    }
    if ["", "~", "Null", "NULL"].contains(&s.as_str()) {
        return Err(err("request YAML nulls must use lowercase null"));
    }
    // Use the reference resolver patterns; plain scalars that it leaves as text
    // (notably 1e3) must remain text here too.
    static PATTERNS: std::sync::OnceLock<Vec<(&str, regex::Regex, regex::Regex)>> =
        std::sync::OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        let source: Value = serde_json::from_str(include_str!("yaml-resolvers.json")).unwrap();
        ["int", "float", "timestamp"]
            .into_iter()
            .map(|tag| {
                let valid = if tag == "int" {
                    r"^-?(?:0|[1-9][0-9]*)$"
                } else {
                    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
                };
                (
                    tag,
                    regex::Regex::new(source[tag].as_str().unwrap()).unwrap(),
                    regex::Regex::new(valid).unwrap(),
                )
            })
            .collect()
    });
    for &(tag, ref pattern, ref valid) in patterns {
        if pattern.is_match(&s) {
            if tag == "timestamp" {
                return Err(err("request values must be JSON-compatible"));
            }
            if !valid.is_match(&s) {
                return Err(err(if tag == "int" {
                    "request YAML integers must use canonical JSON syntax"
                } else {
                    "request YAML numbers must use finite JSON syntax"
                }));
            }
            if tag == "float" {
                let f = s
                    .parse::<f64>()
                    .map_err(|_| err("request YAML numbers must be finite"))?;
                return serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .ok_or_else(|| err("request YAML numbers must be finite"));
            }
            return contracts::parse_json(s.as_bytes(), "request integer")
                .map_err(|e| err(e.to_string()));
        }
    }
    Ok(Value::String(s))
}
fn value<I: Iterator<Item = char>>(p: &mut Parser<I>, depth: usize) -> Result<Value> {
    if depth > 64 {
        return Err(err("request exceeds the 64-level nesting limit"));
    }
    let (event, _) = p.next_token().map_err(|e| err(e.to_string()))?;
    match event {
        Event::Scalar(s, style, 0, None) => scalar(s, style),
        Event::SequenceStart(0, None) => {
            let mut items = Vec::new();
            while p.peek().map_err(|e| err(e.to_string()))?.0 != Event::SequenceEnd {
                items.push(value(p, depth + 1)?);
            }
            p.next_token().map_err(|e| err(e.to_string()))?;
            Ok(Value::Array(items))
        }
        Event::MappingStart(0, None) => {
            let mut map = Map::new();
            while p.peek().map_err(|e| err(e.to_string()))?.0 != Event::MappingEnd {
                let key = value(p, depth + 1)?;
                let key = key
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| err("request YAML object keys must be non-empty strings"))?;
                if key == "<<" {
                    return Err(err("request YAML merge keys are not allowed"));
                }
                if map.contains_key(key) {
                    return Err(err(format!(
                        "request YAML has duplicate key {}",
                        contracts::python_repr(&Value::String(key.into()))
                    )));
                }
                let v = value(p, depth + 1)?;
                map.insert(key.into(), v);
            }
            p.next_token().map_err(|e| err(e.to_string()))?;
            Ok(Value::Object(map))
        }
        _ => Err(err(
            "request YAML aliases, anchors and explicit tags are not allowed",
        )),
    }
}
pub fn parse(bytes: &[u8]) -> Result<Value> {
    if bytes.len() > 1024 * 1024 {
        return Err(err("request exceeds the 1048576-byte limit"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| err("request must be UTF-8"))?;
    let mut scanner = Scanner::new(text.chars());
    let mut count = 0;
    let mut depth = 0usize;
    while let Some(token) = scanner
        .next_token()
        .map_err(|e| err(format!("request YAML could not be scanned: {e}")))?
    {
        match token.1 {
            TokenType::Anchor(_) | TokenType::Alias(_) => {
                return Err(err("request YAML aliases and anchors are not allowed"))
            }
            TokenType::Tag(_, _)
            | TokenType::TagDirective(_, _)
            | TokenType::VersionDirective(_, _) => {
                return Err(err(
                    "request YAML explicit tags and directives are not allowed",
                ))
            }
            TokenType::BlockMappingStart
            | TokenType::BlockSequenceStart
            | TokenType::FlowMappingStart
            | TokenType::FlowSequenceStart => {
                count += 1;
                depth += 1;
            }
            TokenType::BlockEnd | TokenType::FlowMappingEnd | TokenType::FlowSequenceEnd => {
                depth = depth.saturating_sub(1)
            }
            TokenType::Scalar(_, _) => count += 1,
            _ => (),
        }
        if depth > 64 {
            return Err(err("request exceeds the 64-level nesting limit"));
        }
        if count > 10_000 {
            return Err(err("request exceeds the 10,000-node limit"));
        }
    }
    let mut p = Parser::new_from_str(text);
    if p.next_token().map_err(|e| err(e.to_string()))?.0 != Event::StreamStart
        || p.next_token().map_err(|e| err(e.to_string()))?.0 != Event::DocumentStart
    {
        return Err(err("request must contain one document"));
    }
    let result = value(&mut p, 0)?;
    if p.next_token().map_err(|e| err(e.to_string()))?.0 != Event::DocumentEnd
        || p.next_token().map_err(|e| err(e.to_string()))?.0 != Event::StreamEnd
    {
        return Err(err("request must contain one document"));
    }
    Ok(result)
}
