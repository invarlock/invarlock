use crate::analysis::AnalysisError;
use crate::unicode_tables::*;
use invarlock_native_contracts::{canonical_json, parse_json};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use serde_json::Value;
use std::collections::BTreeMap;

fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn combining(c: u32) -> u8 {
    COMBINING
        .binary_search_by_key(&c, |x| x.0)
        .ok()
        .map(|i| COMBINING[i].1)
        .unwrap_or(0)
}
fn compose(a: u32, b: u32) -> Option<u32> {
    if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
        return Some(0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28);
    }
    if (0xac00..=0xd7a3).contains(&a)
        && (a - 0xac00).is_multiple_of(28)
        && (0x11a8..0x11c3).contains(&b)
    {
        return Some(a + b - 0x11a7);
    }
    COMPOSITION
        .binary_search_by_key(&(a, b), |x| x.0)
        .ok()
        .map(|i| COMPOSITION[i].1)
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c)
}
fn normalize(s: &str, casefold: bool) -> String {
    let mut decomposed = Vec::new();
    for c in s.chars().map(|c| c as u32) {
        if (0xac00..=0xd7a3).contains(&c) {
            let x = c - 0xac00;
            decomposed.push(0x1100 + x / 588);
            decomposed.push(0x1161 + (x % 588) / 28);
            if x % 28 != 0 {
                decomposed.push(0x11a7 + x % 28)
            }
        } else if let Ok(i) = DECOMPOSITION.binary_search_by_key(&c, |x| x.0) {
            decomposed.extend(DECOMPOSITION[i].1)
        } else {
            decomposed.push(c)
        }
    }
    // Stable canonical ordering, including leading non-starters.
    let mut start = 0;
    for i in 0..decomposed.len() {
        if combining(decomposed[i]) == 0 {
            decomposed[start..i].sort_by_key(|c| combining(*c));
            start = i + 1
        }
    }
    decomposed[start..].sort_by_key(|c| combining(*c));
    let mut out: Vec<u32> = Vec::new();
    let mut starter = 0;
    let mut previous_class = 0;
    for c in decomposed {
        let cls = combining(c);
        let combined = if !out.is_empty() && (previous_class == 0 || previous_class < cls) {
            compose(out[starter], c)
        } else {
            None
        };
        if let Some(combined) = combined {
            out[starter] = combined
        } else {
            if cls == 0 {
                starter = out.len()
            }
            out.push(c);
            previous_class = cls
        }
    }
    let mut text = String::new();
    for c in out {
        if casefold {
            if let Ok(i) = CASEFOLD.binary_search_by_key(&c, |x| x.0) {
                for x in CASEFOLD[i].1 {
                    text.push(char::from_u32(*x).unwrap())
                }
                continue;
            }
        }
        text.push(char::from_u32(c).unwrap())
    }
    text.split(whitespace)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn number(value: &Value) -> Result<BigRational, AnalysisError> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::Number(_) => {
            String::from_utf8(canonical_json(value).map_err(|e| err(e.to_string()))?)
                .unwrap()
                .trim()
                .to_string()
        }
        _ => return Err(err("not a numeric value")),
    };
    decimal(&raw)
}
pub(crate) fn decimal(raw: &str) -> Result<BigRational, AnalysisError> {
    let s = raw
        .trim_matches(whitespace)
        .chars()
        .filter(|c| *c != '_')
        .map(|c| {
            DECIMAL_DIGITS
                .binary_search_by_key(&(c as u32), |x| x.0)
                .ok()
                .map(|i| (b'0' + DECIMAL_DIGITS[i].1) as char)
                .unwrap_or(c)
        })
        .collect::<String>();
    let (negative, s) = if let Some(s) = s.strip_prefix('-') {
        (true, s)
    } else {
        (false, s.strip_prefix('+').unwrap_or(&s))
    };
    let mut pieces = s.split(['e', 'E']);
    let mantissa = pieces.next().unwrap_or("");
    let exponent = pieces
        .next()
        .map(|s| s.parse::<i32>())
        .transpose()
        .map_err(|_| err("not a decimal number"))?
        .unwrap_or(0);
    if pieces.next().is_some() {
        return Err(err("not a decimal number"));
    }
    let mut parts = mantissa.split('.');
    let whole = parts.next().unwrap_or("");
    let fraction = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return Err(err("not a decimal number"));
    }
    let digits = format!("{whole}{fraction}");
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return Err(err("not a decimal number"));
    }
    let significant = digits.trim_start_matches('0');
    let count = significant.len().max(1);
    let scale = i64::from(exponent) - fraction.len() as i64;
    let adjusted = scale + count as i64 - 1;
    if count > 256 || adjusted.abs() > 1000 {
        return Err(err(
            "numeric values require at most 256 digits and an adjusted exponent within ±1000",
        ));
    }
    let mut numerator = digits
        .parse::<BigInt>()
        .map_err(|_| err("not a decimal number"))?;
    if negative {
        numerator = -numerator
    }
    let ten = BigInt::from(10);
    Ok(if scale >= 0 {
        BigRational::from_integer(numerator * ten.pow(scale as u32))
    } else {
        BigRational::new(numerator, ten.pow((-scale) as u32))
    })
}

pub fn validate_configuration(kind: &str, configuration: &Value) -> Result<(), AnalysisError> {
    let config = configuration.as_object().ok_or_else(|| {
        err(format!(
            "unsupported configuration for {}",
            invarlock_native_contracts::python_repr(&json_string(kind))
        ))
    })?;
    let allowed: &[&str] = match kind {
        "exact_match" | "json_exact" => &[],
        "normalized_match" | "token_f1" => &["casefold", "unicode_version"],
        "numeric_tolerance" => &["absolute", "relative"],
        "json_fields" => &["fields"],
        _ => {
            return Err(err(format!(
                "unsupported configuration for {}",
                invarlock_native_contracts::python_repr(&json_string(kind))
            )))
        }
    };
    if config.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err(err(format!(
            "unsupported configuration for {}",
            invarlock_native_contracts::python_repr(&json_string(kind))
        )));
    }
    if matches!(kind, "normalized_match" | "token_f1") {
        let version = config
            .get("unicode_version")
            .ok_or_else(|| err(format!("{kind} requires an explicit unicode_version")))?;
        if version.as_str() != Some(UNICODE_VERSION) {
            let repr = invarlock_native_contracts::python_repr(version);
            return Err(AnalysisError{reason:"unsupported_scoring_environment",message:format!("{kind} requires Unicode {repr}; this runtime provides {UNICODE_VERSION}. Use a runtime with the policy's Unicode version.")});
        }
    }
    if config.get("casefold").is_some_and(|v| !v.is_boolean()) {
        return Err(err("casefold must be a boolean"));
    }
    for key in ["absolute", "relative"] {
        if let Some(v) = config.get(key) {
            if !v.is_number() || number(v).map(|v| v < BigRational::zero()).unwrap_or(true) {
                return Err(err(format!(
                    "{key} tolerance must be a finite nonnegative number"
                )));
            }
        }
    }
    if kind == "json_fields" {
        let fields = config
            .get("fields")
            .and_then(Value::as_array)
            .ok_or_else(|| err("json_fields requires 1–100 JSON pointer fields"))?;
        if fields.is_empty()
            || fields.len() > 100
            || fields
                .iter()
                .any(|v| !v.as_str().is_some_and(|s| s.starts_with('/')))
        {
            return Err(err("json_fields requires 1–100 JSON pointer fields"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for field in fields {
            let s = field.as_str().unwrap();
            if !seen.insert(s) {
                return Err(err("JSON pointer fields must be unique"));
            }
            if s.split('/')
                .skip(1)
                .any(|s| s.replace("~0", "").replace("~1", "").contains('~'))
            {
                return Err(err("invalid JSON pointer escape"));
            }
        }
    }
    Ok(())
}
fn structured(v: &Value) -> Result<Value, AnalysisError> {
    if let Some(s) = v.as_str() {
        parse_json(s.as_bytes(), "structured output").map_err(|e| err(e.to_string()))
    } else {
        Ok(v.clone())
    }
}
pub(crate) fn pointer<'a>(v: &'a Value, p: &str) -> Option<&'a Value> {
    let mut v = v;
    for token in p.split('/').skip(1) {
        let key = token.replace("~1", "/").replace("~0", "~");
        v = if let Some(obj) = v.as_object() {
            obj.get(&key)?
        } else if let Some(a) = v.as_array() {
            if key.is_empty()
                || !key.bytes().all(|b| b.is_ascii_digit())
                || key.len() > 1 && key.starts_with('0')
            {
                return None;
            }
            a.get(key.parse::<usize>().ok()?)?
        } else {
            return None;
        }
    }
    Some(v)
}
pub fn score(
    kind: &str,
    expected: &Value,
    output: &Value,
    configuration: &Value,
) -> Result<f64, AnalysisError> {
    let empty = serde_json::json!({});
    let configuration = if configuration.is_null()
        || configuration == false
        || configuration == 0
        || configuration.as_str().is_some_and(str::is_empty)
        || configuration.as_array().is_some_and(Vec::is_empty)
    {
        &empty
    } else {
        configuration
    };
    validate_configuration(kind, configuration)?;
    if kind == "numeric_tolerance" {
        let target = number(expected).map_err(|_| err("numeric reference must be finite"))?;
        let actual = match number(output) {
            Ok(x) => x,
            Err(_) => return Ok(0.0),
        };
        let absolute = number(
            &configuration
                .get("absolute")
                .cloned()
                .unwrap_or(Value::from(0)),
        )?;
        let relative = number(
            &configuration
                .get("relative")
                .cloned()
                .unwrap_or(Value::from(0)),
        )?;
        return Ok(
            if (actual - &target).abs() <= absolute.max(relative * target.abs()) {
                1.0
            } else {
                0.0
            },
        );
    }
    if kind == "json_exact" {
        let target = structured(expected)
            .and_then(|v| canonical_json(&v).map_err(|e| err(e.to_string())))
            .map_err(|_| err("JSON reference must be a valid finite JSON value"))?;
        return Ok(
            if structured(output)
                .and_then(|v| canonical_json(&v).map_err(|e| err(e.to_string())))
                .is_ok_and(|v| v == target)
            {
                1.0
            } else {
                0.0
            },
        );
    }
    if kind == "json_fields" {
        let reference = structured(expected)
            .map_err(|_| err("reference must contain every configured JSON field"))?;
        let fields = configuration["fields"].as_array().unwrap();
        let targets = fields
            .iter()
            .map(|p| {
                pointer(&reference, p.as_str().unwrap())
                    .ok_or_else(|| err("reference must contain every configured JSON field"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let actual = match structured(output) {
            Ok(x) => x,
            Err(_) => return Ok(0.0),
        };
        let mut matched = 0;
        for (field, target) in fields.iter().zip(targets) {
            if let Some(v) = pointer(&actual, field.as_str().unwrap()) {
                if canonical_json(v).map_err(|e| err(e.to_string()))?
                    == canonical_json(target).map_err(|e| err(e.to_string()))?
                {
                    matched += 1
                }
            }
        }
        return Ok(matched as f64 / fields.len() as f64);
    }
    let expected = expected
        .as_str()
        .ok_or_else(|| err("text scorer requires string expected and output values"))?;
    let Some(output) = output.as_str() else {
        return Ok(0.0);
    };
    if kind == "exact_match" {
        return Ok(if expected == output { 1.0 } else { 0.0 });
    }
    let casefold = configuration
        .get("casefold")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let expected = normalize(expected, casefold);
    let output = normalize(output, casefold);
    if kind == "normalized_match" {
        return Ok(if expected == output { 1.0 } else { 0.0 });
    }
    let counts = |s: &str| {
        let mut result = BTreeMap::<String, usize>::new();
        for token in s.split_whitespace() {
            *result.entry(token.to_owned()).or_default() += 1
        }
        result
    };
    let wanted = counts(&expected);
    let observed = counts(&output);
    let denominator: usize = wanted.values().sum::<usize>() + observed.values().sum::<usize>();
    let shared: usize = wanted
        .iter()
        .map(|(k, n)| (*n).min(*observed.get(k).unwrap_or(&0)))
        .sum();
    Ok(if denominator == 0 {
        1.0
    } else {
        (2 * shared) as f64 / denominator as f64
    })
}

pub fn casefold_text(value: &str) -> String {
    let mut text = String::new();
    for c in value.chars() {
        if let Ok(i) = CASEFOLD.binary_search_by_key(&(c as u32), |x| x.0) {
            for cp in CASEFOLD[i].1 {
                text.push(char::from_u32(*cp).unwrap())
            }
        } else {
            text.push(c)
        }
    }
    text
}

fn json_string(value: &str) -> Value {
    Value::String(value.to_owned())
}
