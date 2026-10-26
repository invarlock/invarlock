//! Offline verification of retained OpenAI-compatible HTTP request/response facts.
use crate::analysis::py_equal;
use crate::judge_contracts::{bare_digest, bytes};
use crate::AnalysisError;
use base64::Engine;
use invarlock_native_contracts::parse_json;
use serde_json::{json, Value};
use std::net::IpAddr;
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn fields(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
}
fn strict(v: &Value) -> bool {
    v.is_number() && !v.to_string().contains(['.', 'e', 'E'])
}
fn credentials(v: &Value) -> bool {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        if let Some(o) = v.as_object() {
            for (k, v) in o {
                if [
                    "authorization",
                    "api-key",
                    "api_key",
                    "apikey",
                    "access_token",
                    "secret",
                ]
                .contains(&crate::scoring::casefold_text(k).as_str())
                {
                    return true;
                }
                stack.push(v)
            }
        } else if let Some(a) = v.as_array() {
            stack.extend(a)
        }
    }
    false
}
pub fn normalized_compatible_base_url(value: &Value) -> Result<String, AnalysisError> {
    let s = value
        .as_str()
        .ok_or_else(|| err("base_url must be bounded text"))?;
    if s.is_empty()
        || s.chars().count() > 2048
        || s.chars()
            .any(|c| c == '\\' || c <= ' ' || ('\u{7f}'..='\u{9f}').contains(&c))
    {
        return Err(err("base_url must be bounded text"));
    }
    let malformed = || err("base_url is malformed");
    let explicit = || err("base_url must be an explicit credential-free HTTP(S) URL");
    let (scheme, rest) = s.split_once("://").ok_or_else(explicit)?;
    let scheme = scheme.to_ascii_lowercase();
    let (rest, fragment) = rest.split_once('#').unwrap_or((rest, ""));
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, path) = rest
        .split_once('/')
        .map(|(a, p)| (a, format!("/{p}")))
        .unwrap_or((rest, String::new()));
    // urlsplit rejects delimiters introduced by NFKC in the authority.
    for c in authority.chars().filter(|c| !"@:#?".contains(*c)) {
        if let Ok(i) =
            crate::unicode_tables::DECOMPOSITION.binary_search_by_key(&(c as u32), |x| x.0)
        {
            if crate::unicode_tables::DECOMPOSITION[i]
                .1
                .iter()
                .any(|c| [47, 63, 35, 64, 58].contains(c))
            {
                return Err(malformed());
            }
        }
    }
    let host_authority = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    let (raw_host, port) = if let Some(bracketed) = host_authority.strip_prefix('[') {
        let (host, suffix) = bracketed.split_once(']').ok_or_else(malformed)?;
        let ipfuture = host.starts_with('v')
            && host[1..].split_once('.').is_some_and(|(version, suffix)| {
                !version.is_empty()
                    && version.bytes().all(|c| c.is_ascii_hexdigit())
                    && !suffix.is_empty()
            });
        if !ipfuture && !matches!(host.parse::<IpAddr>(), Ok(IpAddr::V6(_))) {
            return Err(malformed());
        }
        let port = if suffix.is_empty() {
            None
        } else {
            Some(suffix.strip_prefix(':').ok_or_else(malformed)?)
        };
        (host, port)
    } else {
        if host_authority.contains(['[', ']']) {
            return Err(malformed());
        }
        host_authority
            .split_once(':')
            .map(|(h, p)| (h, Some(p)))
            .unwrap_or((host_authority, None))
    };
    let port = port
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<u16>())
        .transpose()
        .map_err(|_| malformed())?;
    if !["http", "https"].contains(&scheme.as_str())
        || raw_host.is_empty()
        || authority.contains('@')
        || !query.is_empty()
        || !fragment.is_empty()
    {
        return Err(explicit());
    }
    let host_error = || err("base_url host is malformed");
    if raw_host.contains('%') {
        return Err(host_error());
    }
    let host = match raw_host.parse::<IpAddr>() {
        Ok(ip) => ip.to_string(),
        Err(_) => crate::idna::to_ascii(raw_host.trim_end_matches('.')).ok_or_else(host_error)?,
    };
    if [
        "api.openai.com",
        "openrouter.ai",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
    ]
    .contains(&host.as_str())
    {
        return Err(err(
            "official hosted APIs must use their qualified provider integration",
        ));
    }
    let path = path.trim_end_matches('/');
    if !path.ends_with("/v1")
        || path.contains('%')
        || path.split('/').any(|p| [".", ".."].contains(&p))
        || path.split('/').skip(1).any(str::is_empty)
    {
        return Err(err("base_url path must end in /v1"));
    }
    let port = if scheme == "http" && port == Some(80) || scheme == "https" && port == Some(443) {
        None
    } else {
        port
    };
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host
    };
    Ok(format!(
        "{scheme}://{host}{}{path}/",
        port.map(|p| format!(":{p}")).unwrap_or_default()
    ))
}
pub fn normalize_openai_compatible_collection(c: &Value) -> Result<Value, AnalysisError> {
    let mut keys = vec![
        "profile",
        "service",
        "base_url",
        "model",
        "authentication",
        "request_timeout_seconds",
        "max_calls",
        "max_input_bytes",
        "max_output_tokens",
    ];
    if c.get("response_format").is_some() {
        keys.push("response_format")
    }
    if !fields(c, &keys) {
        return Err(err("OpenAI-compatible collection has unsupported fields"));
    }
    if c["profile"] != "openai-compatible-text-frozen-answer-v1" {
        return Err(err("unsupported OpenAI-compatible collection profile"));
    }
    if !c["service"]
        .as_str()
        .is_some_and(|s| ["vllm", "ollama", "lm_studio", "openai_compatible"].contains(&s))
    {
        return Err(err("unsupported compatible service family"));
    }
    if !c["model"].as_str().is_some_and(|s| {
        !s.is_empty()
            && s.chars().count() <= 256
            && !s
                .chars()
                .any(|c| c < '!' || ('\u{7f}'..='\u{9f}').contains(&c))
    }) {
        return Err(err("compatible model must be explicit bounded text"));
    }
    if !c["authentication"]
        .as_str()
        .is_some_and(|s| ["none", "bearer_env"].contains(&s))
    {
        return Err(err("unsupported compatible authentication mode"));
    }
    if c.get("response_format").is_some_and(|v| {
        !v.as_str()
            .is_some_and(|s| ["json_object", "json_schema"].contains(&s))
    }) {
        return Err(err("unsupported compatible response format"));
    }
    for (k, max) in [
        ("request_timeout_seconds", 3600),
        ("max_calls", 200000),
        ("max_input_bytes", 384 * 1024 * 1024),
        ("max_output_tokens", 1000000000000),
    ] {
        if !strict(&c[k]) || !c[k].as_u64().is_some_and(|n| n >= 1 && n <= max) {
            return Err(err(format!(
                "OpenAI-compatible {k} must be a bounded positive integer"
            )));
        }
    }
    let base = normalized_compatible_base_url(&c["base_url"])?;
    if c["authentication"] == "bearer_env" && base.starts_with("http://") {
        let authority = base
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap();
        let host = if authority.starts_with('[') {
            authority.trim_start_matches('[').split(']').next().unwrap()
        } else {
            authority.split(':').next().unwrap()
        };
        if host != "localhost" && !host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
            return Err(err(
                "bearer authentication requires HTTPS or an explicit loopback endpoint",
            ));
        }
    }
    let mut out = c.clone();
    out["base_url"] = json!(base);
    Ok(out)
}
fn collection(c: &Value) -> Result<Value, AnalysisError> {
    normalize_openai_compatible_collection(c)
        .map_err(|_| err("retained compatible endpoint configuration is invalid"))
}
pub fn openai_compatible_service_identity(c: &Value) -> Result<Value, AnalysisError> {
    Ok(identity(&normalize_openai_compatible_collection(c)?))
}
pub fn validate_openai_compatible_collection(
    c: &Value,
    plan: &Value,
) -> Result<Value, AnalysisError> {
    crate::validate_measurement_plan(plan)?;
    let c = normalize_openai_compatible_collection(c)?;
    let judge = &plan["judge"];
    let config = &judge["config"];
    let schedule = &plan["schedule"];
    if judge["provider"] != "openai_compatible"
        || judge["requested_model"] != c["model"]
        || !judge["approved_resolved_models"]
            .as_array()
            .unwrap()
            .contains(&c["model"])
        || judge["model_identity"] != json!({"kind":"hosted_api","weights_sha256":null})
        || judge["service_identity"] != identity(&c)
    {
        return Err(err(
            "compatible service identity differs from the approved plan",
        ));
    }
    if !config["reasoning_effort"].is_null() || config["seed"].is_null() {
        return Err(err(
            "compatible collection requires reasoning_effort=null and an explicit seed",
        ));
    }
    if schedule["max_attempts"] != 1
        || !schedule["retry_on"].as_array().unwrap().is_empty()
        || schedule["cache"] != "forbid"
    {
        return Err(err(
            "compatible collection requires one uncached attempt without retries",
        ));
    }
    let expected = schedule["expected_trials"].as_u64().unwrap();
    if expected > 1000 {
        return Err(err(
            "compatible collection exceeds the retained source-count limit",
        ));
    }
    if c["max_calls"].as_u64().unwrap() < expected
        || c["max_output_tokens"].as_u64().unwrap()
            < expected * config["max_output_tokens"].as_u64().unwrap()
    {
        return Err(err(
            "compatible collection must reserve every planned inference",
        ));
    }
    Ok(
        json!({"max_calls":c["max_calls"],"max_input_bytes":c["max_input_bytes"],"max_output_tokens":c["max_output_tokens"]}),
    )
}
fn identity(c: &Value) -> Value {
    let mut i = json!({"service":c["service"],"endpoint_sha256":bare_digest(c["base_url"].as_str().unwrap().as_bytes())});
    if let Some(format) = c.get("response_format") {
        i["response_format"] = format.clone()
    }
    i
}
fn wire(c: &Value, plan: &Value, messages: &Value) -> Value {
    let config = &plan["judge"]["config"];
    let response = if c["response_format"] == "json_schema" {
        json!({"type":"json_schema","json_schema":{"name":"invarlock_judge_rating","strict":true,"schema":{"type":"object","additionalProperties":false,"properties":{"rating":{"type":"string","enum":plan["scale"]["ratings"].as_array().unwrap().iter().map(|r|r["label"].clone()).collect::<Vec<_>>()}},"required":["rating"]}}})
    } else {
        json!({"type":"json_object"})
    };
    json!({"model":c["model"],"messages":messages,"temperature":config["temperature"].as_str().unwrap().parse::<f64>().unwrap(),"top_p":config["top_p"].as_str().unwrap().parse::<f64>().unwrap(),"max_tokens":config["max_output_tokens"],"seed":config["seed"],"n":1,"stream":false,"response_format":response})
}
fn response_facts(v: &Value, plan: &Value) -> Result<Value, AnalysisError> {
    let invalid = || err("retained compatible success response is invalid");
    if !v.is_object() || credentials(v) {
        return Err(invalid());
    }
    let choices = v["choices"].as_array().ok_or_else(invalid)?;
    if choices.len() != 1 || !choices[0].is_object() {
        return Err(invalid());
    }
    let choice = &choices[0];
    let message = &choice["message"];
    let model = v["model"].as_str().ok_or_else(invalid)?;
    if !message.is_object()
        || message["role"] != "assistant"
        || !message["content"].is_string()
        || model.is_empty()
        || model.chars().count() > 256
        || model
            .chars()
            .any(|c| c < '!' || ('\u{7f}'..='\u{9f}').contains(&c))
        || !plan["judge"]["approved_resolved_models"]
            .as_array()
            .unwrap()
            .contains(&v["model"])
    {
        return Err(invalid());
    }
    if choice
        .get("index")
        .is_some_and(|i| !strict(i) || i.as_u64() != Some(0))
        || !message["tool_calls"].is_null() && message["tool_calls"] != json!([])
        || !message["function_call"].is_null()
        || choice["finish_reason"] != "stop"
    {
        return Err(invalid());
    }
    for k in ["id", "system_fingerprint"] {
        if !v[k].is_null()
            && !v[k].as_str().is_some_and(|s| {
                s.chars().count() <= 256
                    && !s
                        .chars()
                        .any(|c| c < ' ' || ('\u{7f}'..='\u{9f}').contains(&c))
            })
        {
            return Err(invalid());
        }
    }
    let usage = if v["usage"].is_null() {
        Value::Null
    } else {
        let usage = &v["usage"];
        if !usage.is_object()
            || ["prompt_tokens", "completion_tokens"]
                .iter()
                .any(|k| !strict(&usage[*k]) || usage[*k].as_u64().is_none_or(|n| n > 1000000000))
            || usage["completion_tokens"].as_u64().unwrap()
                > plan["judge"]["config"]["max_output_tokens"]
                    .as_u64()
                    .unwrap()
        {
            return Err(invalid());
        }
        json!({"input_tokens":usage["prompt_tokens"],"output_tokens":usage["completion_tokens"]})
    };
    Ok(
        json!({"content":message["content"],"model":v["model"],"request_id":v["id"],"system_fingerprint":v["system_fingerprint"],"finish_reason":"stop","usage":usage}),
    )
}
fn response_blob(blob: &Value) -> Result<Vec<u8>, AnalysisError> {
    let invalid = || err("retained compatible response body is invalid");
    if !fields(blob, &["media_type", "encoding", "text", "sha256"])
        || !blob["media_type"]
            .as_str()
            .is_some_and(|s| !s.is_empty() && s.chars().count() <= 128)
    {
        return Err(invalid());
    }
    let text = blob["text"].as_str().ok_or_else(invalid)?;
    let raw = match blob["encoding"].as_str() {
        Some("utf-8") => text.as_bytes().to_vec(),
        Some("base64") => base64::engine::general_purpose::STANDARD
            .decode(text)
            .map_err(|_| invalid())?,
        _ => return Err(invalid()),
    };
    if bare_digest(&raw) != blob["sha256"] {
        return Err(invalid());
    }
    Ok(raw)
}
fn source(source: &Value, plan: &Value) -> Result<(), AnalysisError> {
    let c = &source["collection"];
    let http = &source["http"];
    let ident = &source["service_identity"];
    let trials = source["trials"].as_array();
    if !c.is_object()
        || !fields(
            http,
            &[
                "outcome",
                "request",
                "response_status",
                "response_headers",
                "response_body",
            ],
        )
        || !fields(
            ident,
            &[
                "service",
                "endpoint_sha256",
                "requested_model",
                "response_model",
                "request_id",
                "system_fingerprint",
            ],
        )
        || trials.is_none()
        || trials.unwrap().len() != 1
        || !trials.unwrap()[0].is_object()
    {
        return Err(err(
            "retained OpenAI-compatible source has an unsupported shape",
        ));
    }
    let checked = collection(c)?;
    if checked != *c {
        return Err(err("retained compatible endpoint is not canonical"));
    }
    let judge = &plan["judge"];
    let config = &judge["config"];
    let schedule = &plan["schedule"];
    let expected_identity = identity(&checked);
    if ident["service"] != expected_identity["service"]
        || ident["endpoint_sha256"] != expected_identity["endpoint_sha256"]
        || ident["requested_model"] != c["model"]
    {
        return Err(err("retained compatible endpoint identity is invalid"));
    }
    if judge["provider"] != "openai_compatible"
        || judge["requested_model"] != c["model"]
        || !judge["approved_resolved_models"]
            .as_array()
            .unwrap()
            .contains(&c["model"])
        || judge["model_identity"] != json!({"kind":"hosted_api","weights_sha256":null})
        || judge["service_identity"] != expected_identity
        || !config["reasoning_effort"].is_null()
        || config["seed"].is_null()
        || schedule["max_attempts"].as_u64() != Some(1)
        || schedule["retry_on"] != json!([])
        || schedule["cache"] != "forbid"
    {
        return Err(err(
            "retained compatible source differs from the strict plan",
        ));
    }
    let trial = &trials.unwrap()[0];
    let attempts = trial["attempts"]
        .as_array()
        .filter(|a| a.len() == 1)
        .ok_or_else(|| err("retained compatible trial must contain one attempt"))?;
    let attempt = &attempts[0];
    let request = &attempt["request"];
    if !request.is_object() || !request["text"].is_string() {
        return Err(err("retained compatible request material is invalid"));
    }
    let normalized = parse_json(
        request["text"].as_str().unwrap().as_bytes(),
        "retained compatible request",
    )
    .map_err(|e| {
        err(format!(
            "retained compatible request material is invalid: {}",
            e.message.chars().take(240).collect::<String>()
        ))
    })?;
    if !normalized.is_object() {
        return Err(err(
            "retained compatible normalized request must be an object",
        ));
    }
    if bytes(&http["request"])? != bytes(&wire(c, plan, &normalized["messages"]))?
        || credentials(&http["request"])
    {
        return Err(err(
            "retained compatible wire request differs from the approved request",
        ));
    }
    let headers = &http["response_headers"];
    if !headers.as_object().is_some_and(|h| {
        h.iter().all(|(k, v)| {
            ["content-type", "server", "x-request-id"].contains(&k.as_str())
                && v.as_str().is_some_and(|s| {
                    s.chars().count() <= 512
                        && !s
                            .chars()
                            .any(|c| c < ' ' || ('\u{7f}'..='\u{9f}').contains(&c))
                })
        })
    }) {
        return Err(err("retained compatible response headers are invalid"));
    }
    let status = &http["response_status"];
    let outcome = http["outcome"].as_str().unwrap_or("");
    if ![
        "success",
        "http_error",
        "transport_error",
        "malformed_response",
        "response_too_large",
        "credential_echo",
    ]
    .contains(&outcome)
    {
        return Err(err("retained compatible HTTP outcome is invalid"));
    }
    if if outcome == "transport_error" {
        !status.is_null()
    } else {
        !strict(status) || !status.as_u64().is_some_and(|s| (100..=599).contains(&s))
    } {
        return Err(err("retained compatible HTTP status is invalid"));
    }
    let body = &http["response_body"];
    let mut decoded = Value::Null;
    let mut raw = None;
    if !body.is_null() {
        let b = response_blob(body)?;
        if b.len() > 2 * 1024 * 1024 {
            return Err(err(
                "retained compatible response body exceeds its byte limit",
            ));
        }
        decoded = parse_json(&b, "retained compatible response").unwrap_or(Value::Null);
        if credentials(&decoded) {
            return Err(err(
                "retained compatible response contains credential material",
            ));
        }
        raw = Some(b)
    }
    if outcome == "success" {
        if status.as_u64() != Some(200) || raw.is_none() {
            return Err(err("retained compatible success response is incomplete"));
        }
        let facts = response_facts(&decoded, plan)?;
        let response = &attempt["response"];
        if attempt["status"] != "completed"
            || !response.is_object()
            || response["text"] != facts["content"]
            || response["sha256"] != bare_digest(facts["content"].as_str().unwrap().as_bytes())
            || attempt["resolved_model"] != facts["model"]
            || attempt["request_id"] != facts["request_id"]
            || attempt["finish_reason"] != facts["finish_reason"]
            || !py_equal(&attempt["usage"], &facts["usage"])
            || ident["response_model"] != facts["model"]
            || ident["request_id"] != facts["request_id"]
            || ident["system_fingerprint"] != facts["system_fingerprint"]
        {
            return Err(err("retained compatible completion identity is invalid"));
        }
    } else {
        if outcome == "http_error"
            && (status.as_u64() == Some(200)
                || body.is_null() && !status.as_u64().is_some_and(|s| (300..=399).contains(&s)))
        {
            return Err(err("retained compatible HTTP error outcome is invalid"));
        }
        if ["transport_error", "response_too_large", "credential_echo"].contains(&outcome)
            && !body.is_null()
        {
            return Err(err(
                "retained compatible unsafe failure retained a response body",
            ));
        }
        if ["transport_error", "credential_echo"].contains(&outcome)
            && headers.as_object().is_some_and(|h| !h.is_empty())
        {
            return Err(err(
                "retained compatible unsafe failure retained response headers",
            ));
        }
        let (code, message) = match outcome {
            "http_error" => (
                "compatible-http-error",
                format!(
                    "compatible service returned HTTP {}",
                    status.as_u64().unwrap()
                ),
            ),
            "transport_error" => (
                "compatible-transport-error",
                "compatible service transport failed or exceeded its deadline".into(),
            ),
            "malformed_response" => (
                "compatible-malformed-response",
                "compatible service returned an unsupported response".into(),
            ),
            "response_too_large" => (
                "compatible-response-too-large",
                "compatible service response exceeded the retained byte limit".into(),
            ),
            _ => (
                "compatible-credential-echo",
                "compatible service response could not be retained safely".into(),
            ),
        };
        if attempt["status"] != "cancelled"
            || !attempt["response"].is_null()
            || !attempt["error"].is_object()
            || attempt["error"]["code"] != code
            || attempt["error"]["message"] != message
            || ["resolved_model", "request_id", "finish_reason", "usage"]
                .iter()
                .any(|k| !attempt[*k].is_null())
            || ["response_model", "request_id", "system_fingerprint"]
                .iter()
                .any(|k| !ident[*k].is_null())
        {
            return Err(err("retained compatible failed outcome is invalid"));
        }
    }
    Ok(())
}
pub(crate) fn replay(decoded: &Value, plan: &Value) -> Result<Vec<Value>, AnalysisError> {
    if !fields(
        decoded,
        &["format", "collection", "http", "service_identity", "trials"],
    ) || decoded["format"] != "invarlock/retained-openai-compatible-judge-v1"
        || !decoded["trials"].is_array()
    {
        return Err(err(
            "retained source has an unsupported OpenAI-compatible profile",
        ));
    }
    let trials = decoded["trials"].as_array().unwrap();
    if trials.iter().any(|t| {
        !t.is_object()
            || !t["attempts"]
                .as_array()
                .is_some_and(|a| a.iter().all(Value::is_object))
    }) {
        return Err(err(
            "retained OpenAI-compatible trials must contain attempt objects",
        ));
    }
    source(decoded, plan)?;
    Ok(trials.clone())
}
pub(crate) fn family(sources: &[Value], plan: &Value) -> Result<(), AnalysisError> {
    if sources.is_empty() {
        return Ok(());
    }
    let c = &sources[0]["collection"];
    let mut spent = 0;
    for s in sources {
        if s["collection"] != *c {
            return Err(err(
                "retained compatible shards have inconsistent endpoint configuration",
            ));
        }
        spent += bytes(&s["http"]["request"])?.len();
    }
    let calls = sources.len() as u64;
    if calls > c["max_calls"].as_u64().unwrap()
        || spent as u64 > c["max_input_bytes"].as_u64().unwrap()
        || calls
            * plan["judge"]["config"]["max_output_tokens"]
                .as_u64()
                .unwrap()
            > c["max_output_tokens"].as_u64().unwrap()
    {
        return Err(err(
            "retained compatible shards exceed their aggregate resource reservations",
        ));
    }
    Ok(())
}

/// Construct the exact request body reserved by a validated compatible collection.
pub fn render_compatible_judge_request(
    configuration: &Value,
    plan: &Value,
    messages: &Value,
) -> Result<Value, AnalysisError> {
    crate::validate_measurement_plan(plan)?;
    validate_openai_compatible_collection(configuration, plan)?;
    if !messages.as_array().is_some_and(|m| {
        m.iter()
            .all(|v| v.is_object() && v["role"].is_string() && v["content"].is_string())
    }) {
        return Err(err(
            "normalized judge messages must be role/content objects",
        ));
    }
    Ok(wire(configuration, plan, messages))
}
