//! Typed behavioral schedules and deterministic preparation of pinned local JSONL.
use crate::{
    canonical_json_no_newline, parse_json, python_repr, sha256_digest, ContractError, Result, Value,
};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};
pub const MAX_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_RECORDS: usize = 10_000;
const FORMAT: &str = "invarlock/runtime-behavioral-schedule-v1";
const CONTENT_FIELDS: [&str; 4] = [
    "content_id_field",
    "content_sha256_field",
    "content_byte_length_field",
    "content_media_type_field",
];
const DESCRIPTOR_FIELDS: [&str; 15] = [
    "format_version",
    "source_sha256",
    "source_format",
    "name",
    "split",
    "input_field",
    "expected_output_field",
    "id_field",
    "content_role",
    "content_id_field",
    "content_sha256_field",
    "content_byte_length_field",
    "content_media_type_field",
    "limit",
    "selected_record_count",
];
fn err(message: impl Into<String>) -> ContractError {
    ContractError::new("schedule_contract", message)
}
fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(err(message))
    }
}
pub fn python_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c)
}
fn text<'a>(value: &'a Value, label: &str, max: usize) -> Result<&'a str> {
    let s = value.as_str().unwrap_or("");
    require(
        !s.is_empty() && s == s.trim_matches(python_whitespace),
        format!("{label} must be a non-empty trimmed string"),
    )?;
    require(
        s.chars().all(|c| (c as u32) >= 32),
        format!("{label} must not contain control characters"),
    )?;
    require(
        s.chars().count() <= max,
        format!("{label} must not exceed {max} characters"),
    )?;
    Ok(s)
}
fn canonical_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.as_bytes()[0].is_ascii_lowercase()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}
fn name(value: &Value, label: &str) -> Result<()> {
    let s = text(value, label, usize::MAX)?;
    require(
        canonical_name(s),
        format!(
            "{label} must be a canonical {}",
            if label == "schedule.task" {
                "task identifier"
            } else {
                "provider name"
            }
        ),
    )
}
fn logical(value: &Value, label: &str, max: usize) -> Result<()> {
    let s = text(value, label, max)?.replace('\\', "/");
    require(
        !s.starts_with('/')
            && !s.split('/').any(|p| p.is_empty() || p == "." || p == "..")
            && !s.contains(":/")
            && s.chars().nth(1) != Some(':'),
        format!("{label} must not be an absolute or traversal path"),
    )
}
fn sha(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}
fn positive(value: &Value) -> bool {
    value.as_number().is_some_and(|n| {
        let s = n.to_string();
        s != "0" && s.bytes().all(|b| b.is_ascii_digit())
    })
}
fn exact(value: &Value, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| err(format!("{label} must be an object")))?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual == expected {
        return Ok(());
    }
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let unknown = actual.difference(&expected).copied().collect::<Vec<_>>();
    let mut details = Vec::new();
    if !missing.is_empty() {
        details.push(format!("missing {}", missing.join(", ")));
    }
    if !unknown.is_empty() {
        details.push(format!("unknown {}", unknown.join(", ")));
    }
    Err(err(format!(
        "{label} must contain exactly the public fields ({})",
        details.join("; ")
    )))
}
pub fn validate_dataset_identity(value: &Value) -> Result<()> {
    exact(
        value,
        &[
            "provider",
            "dataset_name",
            "config_name",
            "revision",
            "split",
        ],
        "dataset_identity",
    )?;
    let provider = text(&value["provider"], "dataset_identity.provider", 64)?;
    require(
        canonical_name(provider),
        "dataset_identity.provider must be a canonical provider name",
    )?;
    for key in ["dataset_name", "config_name"] {
        if !value[key].is_null() {
            logical(&value[key], &format!("dataset_identity.{key}"), 512)?;
        }
    }
    if !value["revision"].is_null() {
        require(
            value["revision"].as_str().is_some_and(|s| {
                (40..=64).contains(&s.len())
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }),
            "dataset_identity.revision must be an immutable lowercase revision",
        )?;
    }
    logical(&value["split"], "dataset_identity.split", 128)?;
    require(
        !["hf_text", "hf_seq2seq", "wikitext2"].contains(&provider)
            || ["dataset_name", "config_name", "revision"]
                .iter()
                .all(|k| !value[*k].is_null()),
        "hosted dataset_identity requires dataset_name, config_name, and revision",
    )
}
fn media_type(s: &str) -> bool {
    let Some((a, b)) = s.split_once('/') else {
        return false;
    };
    [a, b].iter().all(|part| {
        !part.is_empty()
            && part.len() <= 127
            && (part.as_bytes()[0].is_ascii_lowercase() || part.as_bytes()[0].is_ascii_digit())
            && part
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"!#$&^_.+-".contains(&b))
    })
}
fn input_part(value: &Value, label: &str) -> Result<()> {
    require(value.is_object(), format!("{label} must be an object"))?;
    match value["kind"].as_str() {
        Some("text") => {
            exact(value, &["kind", "role", "text", "sha256"], label)?;
            if let Some(t) = value["text"].as_str() {
                require(
                    t.chars().count() <= 65536,
                    format!("{label}.text must not exceed 65536 characters"),
                )?;
            }
        }
        Some("content") => exact(
            value,
            &[
                "kind",
                "role",
                "content_id",
                "media_type",
                "byte_length",
                "sha256",
            ],
            label,
        )?,
        _ => return Err(err(format!("{label}.kind must be text or content"))),
    }
    name(&value["role"], "input part role")?;
    require(
        sha(&value["sha256"]),
        "input part sha256 must be a lowercase sha256 digest",
    )?;
    if value["kind"] == "text" {
        let t = value["text"].as_str().unwrap_or("");
        require(
            !t.trim_matches(python_whitespace).is_empty(),
            "text input part must contain non-empty text",
        )?;
        require(
            value["sha256"] == sha256_digest(t.as_bytes())[7..],
            "text input part sha256 does not match text",
        )?;
    } else {
        name(&value["content_id"], "content input part content_id")?;
        let media = text(
            &value["media_type"],
            "content input part media_type",
            usize::MAX,
        )?;
        require(
            media_type(media),
            "content input part media_type must be canonical",
        )?;
        require(
            positive(&value["byte_length"]),
            "content input part byte_length must be a positive integer",
        )?;
    }
    Ok(())
}
fn parts(value: &Value, label: &str) -> Result<()> {
    let parts = value
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| err(format!("{label} must be a non-empty array")))?;
    require(
        parts.len() <= 64,
        format!("{label} must not exceed 64 parts"),
    )?;
    for (index, part) in parts.iter().enumerate() {
        input_part(part, &format!("{label}[{index}]"))?;
    }
    Ok(())
}
fn bare(value: &Value) -> Result<String> {
    Ok(sha256_digest(&canonical_json_no_newline(value)?)[7..].into())
}
#[derive(Clone, Debug)]
pub struct Schedule(Arc<ScheduleInner>);
#[derive(Debug)]
struct ScheduleInner {
    value: Value,
    bytes: Vec<u8>,
    digest: String,
}
impl Schedule {
    pub fn value(&self) -> &Value {
        &self.0.value
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0.bytes
    }
    pub fn digest(&self) -> &str {
        &self.0.digest
    }
    pub fn parse(raw: &[u8]) -> Result<Self> {
        require(
            raw.len() <= MAX_BYTES,
            format!("runtime behavioral schedule exceeds the {MAX_BYTES}-byte size limit"),
        )?;
        Self::build(&parse_json(raw, "runtime behavioral schedule")?)
    }
    pub fn build(value: &Value) -> Result<Self> {
        require(
            value.is_object(),
            "runtime behavioral schedule must be an object",
        )?;
        exact(
            value,
            &["format_version", "task", "dataset_identity", "records"],
            "schedule",
        )?;
        require(
            value["format_version"] == FORMAT,
            "runtime behavioral schedule format_version is unsupported",
        )?;
        name(&value["task"], "schedule.task")?;
        validate_dataset_identity(&value["dataset_identity"])?;
        let rows = value["records"]
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| err("records must be a non-empty array"))?;
        require(
            rows.len() <= MAX_RECORDS,
            format!("records must not exceed {MAX_RECORDS} entries"),
        )?;
        let mut ids = BTreeSet::new();
        for (index, row) in rows.iter().enumerate() {
            let label = format!("records[{index}]");
            exact(
                row,
                &[
                    "record_id",
                    "input_parts",
                    "input_sha256",
                    "expected_output",
                ],
                &label,
            )?;
            logical(&row["record_id"], &format!("{label}.record_id"), 256)?;
            let output = row["expected_output"].as_str().unwrap_or("");
            require(
                !output.trim_matches(python_whitespace).is_empty(),
                format!("{label}.expected_output must be non-empty text"),
            )?;
            require(
                output.chars().count() <= 65536,
                format!("{label}.expected_output must not exceed 65536 characters"),
            )?;
            parts(&row["input_parts"], &format!("{label}.input_parts"))?;
            require(
                row["input_sha256"] == bare(&row["input_parts"])?,
                format!("{label}.input_sha256 does not match ordered input_parts"),
            )?;
            let parts = row["input_parts"].as_array().unwrap();
            require(
                parts.iter().any(|p| p["kind"] == "text"),
                format!("{label}.input_parts requires at least one text part"),
            )?;
            let mut roles = BTreeSet::new();
            require(
                parts
                    .iter()
                    .all(|p| roles.insert(p["role"].as_str().unwrap())),
                "input part roles must be unique within a record",
            )?;
            ids.insert(row["record_id"].as_str().unwrap());
        }
        require(
            ids.len() == rows.len(),
            "record IDs must be unique within a schedule",
        )?;
        let bytes = canonical_json_no_newline(value)?;
        require(
            bytes.len() <= MAX_BYTES,
            format!("runtime behavioral schedule exceeds the {MAX_BYTES}-byte size limit"),
        )?;
        let digest = sha256_digest(&bytes)[7..].into();
        Ok(Self(Arc::new(ScheduleInner {
            value: value.clone(),
            bytes,
            digest,
        })))
    }
    pub fn from_material(identity: &Value, records: &Value, task: &str) -> Result<Self> {
        let records = records
            .as_array()
            .ok_or_else(|| err("records must be an array"))?;
        let mut built = Vec::with_capacity(records.len());
        for (index, row) in records.iter().enumerate() {
            let label = format!("records[{index}]");
            exact(
                row,
                &["record_id", "input_parts", "expected_output"],
                &label,
            )?;
            parts(&row["input_parts"], &format!("{label}.input_parts"))?;
            let mut row = row.clone();
            row["input_sha256"] = json!(bare(&row["input_parts"])?);
            built.push(row);
        }
        Self::build(
            &json!({"format_version":FORMAT,"task":task,"dataset_identity":identity,"records":built}),
        )
    }
}
fn field_name(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        !s.is_empty()
            && s.len() <= 128
            && (s.as_bytes()[0].is_ascii_alphabetic() || s.as_bytes()[0] == b'_')
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    })
}
fn python_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        Value::Number(n) => {
            if n.to_string().contains(['.', 'e', 'E']) {
                "float"
            } else {
                "int"
            }
        }
    }
}
fn regex_string(value: &Value) -> Result<()> {
    require(
        value.is_string(),
        format!(
            "expected string or bytes-like object, got '{}'",
            python_type(value)
        ),
    )
}
pub fn validate_local_request(source: &Value) -> Result<()> {
    require(
        source.get("format").is_none_or(|v| v == "jsonl"),
        "dataset format must be 'jsonl'",
    )?;
    regex_string(&source["sha256"])?;
    require(
        sha(&source["sha256"]),
        "dataset sha256 must be a lowercase SHA-256 digest",
    )?;
    let any = CONTENT_FIELDS.iter().any(|k| !source[*k].is_null());
    let all = CONTENT_FIELDS.iter().all(|k| !source[*k].is_null());
    require(
        !any || all,
        "dataset content field mappings must be provided together",
    )?;
    require(
        all != source["content_role"].is_null(),
        "dataset content_role and content field mappings must be provided together",
    )?;
    if !source["content_role"].is_null() {
        regex_string(&source["content_role"])?;
        require(
            source["content_role"].as_str().is_some_and(canonical_name),
            "dataset content_role must be a canonical input role",
        )?;
    }
    let mut mapped = BTreeSet::new();
    let mut count = 0;
    for key in ["input_field", "expected_output_field", "id_field"]
        .into_iter()
        .chain(CONTENT_FIELDS)
    {
        if key != "input_field" && key != "expected_output_field" && source[key].is_null() {
            continue;
        }
        regex_string(&source[key])?;
        require(
            field_name(&source[key]),
            format!("dataset {key} must be a simple top-level JSON field name"),
        )?;
        mapped.insert(source[key].as_str().unwrap());
        count += 1;
    }
    require(
        mapped.len() == count,
        "dataset field mappings must reference distinct fields",
    )?;
    if !source["limit"].is_null() {
        require(
            source["limit"].is_number() || source["limit"].is_boolean(),
            format!(
                "'<=' not supported between instances of 'int' and '{}'",
                python_type(&source["limit"])
            ),
        )?;
        let limit = if source["limit"].is_boolean() {
            source["limit"].as_bool().map(|b| if b { 1.0 } else { 0.0 })
        } else {
            source["limit"].as_f64()
        };
        require(
            limit.is_some_and(|n| (1.0..=MAX_RECORDS as f64).contains(&n)),
            format!("dataset limit must be between 1 and {MAX_RECORDS}"),
        )?;
    }
    validate_dataset_identity(
        &json!({"provider":"local","dataset_name":source["name"],"config_name":"evaluation-records-jsonl-v1","revision":source["sha256"],"split":source["split"]}),
    )
}
fn lines(raw: &[u8]) -> Vec<&[u8]> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == b'\r' || raw[index] == b'\n' {
            result.push(&raw[start..index]);
            if raw[index] == b'\r' && raw.get(index + 1) == Some(&b'\n') {
                index += 1;
            }
            start = index + 1;
        }
        index += 1;
    }
    if start < raw.len() {
        result.push(&raw[start..]);
    }
    result
}
fn mapped<'a>(value: &'a Value, field: &Value, line: usize) -> Result<&'a Value> {
    let key = field.as_str().unwrap_or("");
    let v = &value[key];
    require(
        v.is_string(),
        format!(
            "dataset line {line} field {} must be text",
            python_repr(field)
        ),
    )?;
    Ok(v)
}
fn modality(task: &str, has: bool, role: &Value, prefix: &str) -> Result<()> {
    if task == "vision_text_generation" {
        require(
            has,
            format!("vision_text_generation requires {prefix}dataset content field mappings"),
        )?;
        require(
            role == "image",
            "vision_text_generation requires content_role 'image'",
        )?;
    }
    if ["text_causal", "text_seq2seq", "masked_language"].contains(&task) {
        require(
            !has,
            format!("{task} does not accept {prefix}dataset content field mappings"),
        )?;
    }
    Ok(())
}
pub fn prepare_local(source: &Value, raw: &[u8], task: &str) -> Result<Schedule> {
    validate_local_request(source)?;
    require(
        raw.len() <= MAX_BYTES,
        format!("local evaluation dataset exceeds the {MAX_BYTES}-byte size limit"),
    )?;
    let actual = sha256_digest(raw)[7..].to_owned();
    require(
        source["sha256"] == actual,
        format!(
            "local evaluation dataset sha256 mismatch: expected {}, got {actual}",
            source["sha256"].as_str().unwrap()
        ),
    )?;
    let has = !source["content_id_field"].is_null();
    modality(task, has, &source["content_role"], "")?;
    let lines = lines(raw);
    require(
        !lines.is_empty(),
        "local evaluation dataset must contain at least one record",
    )?;
    let limit = if source["limit"].is_null() {
        lines.len()
    } else if source["limit"] == true {
        1
    } else {
        let limit = source["limit"].as_f64().unwrap_or(0.0);
        require(
            lines.len() as f64 >= limit,
            "local evaluation dataset contains fewer records than the requested limit",
        )?;
        source["limit"]
            .as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| {
                err("slice indices must be integers or None or have an __index__ method")
            })?
    };
    require(
        lines.len() >= limit,
        "local evaluation dataset contains fewer records than the requested limit",
    )?;
    require(
        limit <= MAX_RECORDS,
        format!("local evaluation dataset selection exceeds the {MAX_RECORDS}-record limit"),
    )?;
    let mut records = Vec::with_capacity(limit);
    for (index, line) in lines.into_iter().take(limit).enumerate() {
        let line_number = index + 1;
        require(
            line.iter().any(|b| !b.is_ascii_whitespace()),
            format!("local evaluation dataset line {line_number} is blank"),
        )?;
        let value = parse_json(
            line,
            &format!("local evaluation dataset line {line_number}"),
        )?;
        require(
            value.is_object(),
            format!("local evaluation dataset line {line_number} must be an object"),
        )?;
        let prompt = mapped(&value, &source["input_field"], line_number)?;
        let id = if source["id_field"].is_null() {
            json!(format!("record/{index:08}"))
        } else {
            mapped(&value, &source["id_field"], line_number)?.clone()
        };
        let output = mapped(&value, &source["expected_output_field"], line_number)?;
        let prompt_part = json!({"kind":"text","role":"prompt","text":prompt,"sha256":sha256_digest(prompt.as_str().unwrap().as_bytes())[7..]});
        let mut input_parts = Vec::new();
        if has {
            let content_id = mapped(&value, &source["content_id_field"], line_number)?;
            let content_sha = mapped(&value, &source["content_sha256_field"], line_number)?;
            let media = mapped(&value, &source["content_media_type_field"], line_number)?;
            let field = &source["content_byte_length_field"];
            let length = &value[field.as_str().unwrap()];
            require(
                positive(length),
                format!(
                    "dataset line {line_number} field {} must be a positive integer",
                    python_repr(field)
                ),
            )?;
            input_parts.push(json!({"kind":"content","role":source["content_role"],"content_id":content_id,"media_type":media,"byte_length":length,"sha256":content_sha}));
        }
        input_parts.push(prompt_part);
        records.push(json!({"record_id":id,"input_parts":input_parts,"expected_output":output}));
    }
    Schedule::from_material(
        &json!({"provider":"local","dataset_name":source["name"],"config_name":"evaluation-records-jsonl-v1","revision":source["sha256"],"split":source["split"]}),
        &json!(records),
        task,
    )
}
pub fn preparation_descriptor(source: &Value, schedule: &Schedule) -> Result<Value> {
    let mut value = json!({"format_version":"invarlock/local-jsonl-preparation-v1","source_sha256":source["sha256"],"source_format":source.get("format").unwrap_or(&json!("jsonl")),"selected_record_count":schedule.value()["records"].as_array().unwrap().len()});
    for key in [
        "name",
        "split",
        "input_field",
        "expected_output_field",
        "id_field",
        "content_role",
        "content_id_field",
        "content_sha256_field",
        "content_byte_length_field",
        "content_media_type_field",
        "limit",
    ] {
        value[key] = source[key].clone();
    }
    validate_preparation(&value, schedule)?;
    Ok(value)
}
pub fn validate_preparation(value: &Value, schedule: &Schedule) -> Result<()> {
    require(
        value.as_object().is_some_and(|o| {
            o.len() == DESCRIPTOR_FIELDS.len()
                && DESCRIPTOR_FIELDS.iter().all(|k| o.contains_key(*k))
        }),
        "run dataset preparation fields are invalid",
    )?;
    require(
        value["format_version"] == "invarlock/local-jsonl-preparation-v1",
        "run dataset preparation format_version is invalid",
    )?;
    require(
        value["source_format"] == "jsonl",
        "run dataset preparation source_format is invalid",
    )?;
    require(
        sha(&value["source_sha256"]),
        "run dataset preparation source_sha256 is invalid",
    )?;
    let any = CONTENT_FIELDS.iter().any(|k| !value[*k].is_null());
    let all = CONTENT_FIELDS.iter().all(|k| !value[*k].is_null());
    require(
        !any || all,
        "run dataset preparation content field mappings must be provided together",
    )?;
    require(
        all != value["content_role"].is_null(),
        "run dataset preparation content_role and content field mappings must be provided together",
    )?;
    if !value["content_role"].is_null() {
        require(
            value["content_role"].as_str().is_some_and(canonical_name),
            "run dataset preparation content_role is invalid",
        )?;
    }
    let mut names = BTreeSet::new();
    let mut count = 0;
    for key in ["input_field", "expected_output_field", "id_field"]
        .into_iter()
        .chain(CONTENT_FIELDS)
    {
        if key != "input_field" && key != "expected_output_field" && value[key].is_null() {
            continue;
        }
        require(
            field_name(&value[key]),
            format!("run dataset preparation {key} is invalid"),
        )?;
        names.insert(value[key].as_str().unwrap());
        count += 1;
    }
    require(
        names.len() == count,
        "run dataset preparation field mappings must be distinct",
    )?;
    let expected = json!({"provider":"local","dataset_name":value["name"],"config_name":"evaluation-records-jsonl-v1","revision":value["source_sha256"],"split":value["split"]});
    require(
        schedule.value()["dataset_identity"] == expected,
        "run dataset preparation does not match the canonical schedule identity",
    )?;
    let rows = schedule.value()["records"].as_array().unwrap();
    let count = rows.len() as u64;
    require(
        value["selected_record_count"].as_u64() == Some(count),
        "run dataset preparation selected_record_count does not match the schedule",
    )?;
    require(
        value["limit"].is_null() || value["limit"].as_u64() == Some(count),
        "run dataset preparation limit does not match the selection",
    )?;
    if value["id_field"].is_null() {
        require(
            rows.iter()
                .enumerate()
                .all(|(i, r)| r["record_id"] == format!("record/{i:08}")),
            "run dataset preparation positional IDs do not match the schedule",
        )?;
    }
    require(
        all == rows.iter().all(|r| {
            r["input_parts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["kind"] == "content")
        }),
        "run dataset preparation content mappings do not match the schedule",
    )?;
    modality(
        schedule.value()["task"].as_str().unwrap(),
        all,
        &value["content_role"],
        "run ",
    )
}
