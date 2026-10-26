//! Offline replay of the pinned retained Inspect event projections.
use crate::analysis::py_equal;
use crate::judge_contracts::bytes;
use crate::AnalysisError;
use invarlock_native_contracts::parse_json;
use serde_json::{json, Value};
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn fields(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|v| v.len() == keys.len() && keys.iter().all(|k| v.contains_key(*k)))
}
fn strict(v: &Value) -> bool {
    v.is_number() && !v.to_string().contains(['.', 'e', 'E'])
}
fn decode(v: &Value) -> Value {
    v.as_str()
        .and_then(|s| parse_json(s.as_bytes(), "provider completion").ok())
        .unwrap_or_else(|| v.clone())
}
fn float(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.parse().ok())
}
fn secrets(value: &Value) -> Result<(), AnalysisError> {
    let mut stack = vec![value];
    while let Some(v) = stack.pop() {
        if let Some(o) = v.as_object() {
            for (k, v) in o {
                let key = crate::scoring::casefold_text(k);
                if [
                    "authorization",
                    "api-key",
                    "api_key",
                    "apikey",
                    "access_token",
                    "secret",
                ]
                .contains(&key.as_str())
                {
                    return Err(err(
                        "retained Inspect provider call contains a credential field",
                    ));
                }
                if ["headers", "extra_headers", "http_headers"].contains(&key.as_str())
                    && !v.as_object().is_some_and(|v| {
                        v.iter().all(|(k, v)| {
                            crate::scoring::casefold_text(k) == "x-irid"
                                && v.as_str()
                                    .is_some_and(|s| !s.is_empty() && s.chars().count() <= 128)
                        })
                    })
                {
                    return Err(err(
                        "retained Inspect provider call contains unsupported headers",
                    ));
                }
                stack.push(v)
            }
        } else if let Some(a) = v.as_array() {
            stack.extend(a)
        }
    }
    Ok(())
}
fn provider_completion(response: &Value) -> Result<Value, AnalysisError> {
    if response["format"] == "invarlock/judge-provider-response-v1" {
        if !fields(
            response,
            &["format", "content", "model", "id", "finish_reason", "usage"],
        ) || !response["content"].is_string()
        {
            return Err(err(
                "retained normalized provider response has an unsupported shape",
            ));
        }
        return Ok(decode(&response["content"]));
    }
    if fields(response, &["rating"]) {
        return Ok(response.clone());
    }
    if let Some(choices) = response["choices"].as_array() {
        if choices.len() == 1
            && choices[0].is_object()
            && choices[0]["message"].is_object()
            && choices[0]["message"]["content"].is_string()
        {
            let m = &choices[0]["message"];
            if m.get("role").is_some_and(|v| v != "assistant")
                || !m["refusal"].is_null() && m["refusal"] != ""
                || !m["tool_calls"].is_null() && m["tool_calls"] != json!([])
                || !m["function_call"].is_null()
            {
                return Err(err("retained Inspect provider response must be a tool-free assistant message without refusal"));
            }
            return Ok(decode(&m["content"]));
        }
    }
    Err(err(
        "retained Inspect provider response uses an unsupported shape",
    ))
}
fn provider_response(
    response: &Value,
    output: &Value,
    request: &Value,
) -> Result<(), AnalysisError> {
    if !response.is_object() {
        return Err(err(
            "completed retained Inspect provider response must be an object",
        ));
    }
    let normalized = response["format"] == "invarlock/judge-provider-response-v1";
    if request.get("service_tier").is_some() && response["service_tier"] != "default" {
        return Err(err(
            "retained Inspect provider response contradicts its service tier",
        ));
    }
    let completion = provider_completion(response)?;
    if !output["completion"].is_string() {
        return Err(err("retained Inspect output completion must be text"));
    }
    if bytes(&completion)? != bytes(&decode(&output["completion"]))? {
        return Err(err(
            "retained Inspect provider response contradicts its completion",
        ));
    }
    for (r, o, msg) in [
        ("model", "model", "resolved model"),
        ("id", "request_id", "request ID"),
    ] {
        if (normalized || response[r].is_string()) && !py_equal(&response[r], &output[o]) {
            return Err(err(format!(
                "retained Inspect provider response contradicts its {msg}"
            )));
        }
    }
    if normalized {
        if !py_equal(&response["finish_reason"], &output["finish_reason"]) {
            return Err(err(
                "retained Inspect provider response contradicts its finish reason",
            ));
        }
        if !py_equal(&response["usage"], &output["usage"]) {
            return Err(err(
                "retained Inspect provider usage contradicts its output",
            ));
        }
    }
    if let Some(choices) = response["choices"].as_array() {
        if !choices.is_empty() {
            let finish = match choices[0]["finish_reason"].as_str() {
                Some("stop") => "stop",
                Some("length") => "max_tokens",
                Some("content_filter") => "content_filter",
                _ => {
                    return Err(err(
                        "retained Inspect provider response contradicts its finish reason",
                    ))
                }
            };
            if output["finish_reason"] != finish {
                return Err(err(
                    "retained Inspect provider response contradicts its finish reason",
                ));
            }
        }
    }
    if response["usage"].is_object() && output["usage"].is_object() {
        let usage = &response["usage"];
        if !py_equal(
            usage.get("prompt_tokens").unwrap_or(&usage["input_tokens"]),
            &output["usage"]["input_tokens"],
        ) || !py_equal(
            usage
                .get("completion_tokens")
                .unwrap_or(&usage["output_tokens"]),
            &output["usage"]["output_tokens"],
        ) {
            return Err(err(
                "retained Inspect provider usage contradicts its output",
            ));
        }
    }
    Ok(())
}
fn provider_projection(
    call: &Value,
    output: &Value,
    normalized_request: &Value,
    completed: bool,
    version: &Value,
) -> Result<(), AnalysisError> {
    let request = &call["request"];
    let normalized = request["format"] == "invarlock/judge-request-v1";
    let gpt56 = !normalized
        && version == "0.3.263"
        && matches!(
            normalized_request["model"].as_str(),
            Some("openai/gpt-5.6-sol" | "openai/gpt-5.6-luna")
        );
    if normalized {
        if bytes(request)? != bytes(normalized_request)? {
            return Err(err(
                "retained Inspect normalized provider request differs from the request",
            ));
        }
    } else {
        let allowed = [
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "reasoning_effort",
            "n",
            "tools",
            "tool_choice",
            "extra_headers",
            "service_tier",
        ];
        if request
            .as_object()
            .unwrap()
            .keys()
            .any(|k| !allowed.contains(&k.as_str()))
        {
            return Err(err(
                "retained Inspect provider request contains unsupported controls",
            ));
        }
        if request.get("service_tier").is_some() && request["service_tier"] != "default" {
            return Err(err(
                "retained Inspect provider request has an unsupported service tier",
            ));
        }
    }
    let messages = request["messages"]
        .as_array()
        .ok_or_else(|| err("retained Inspect provider request must contain chat messages"))?;
    if messages.iter().any(|m| {
        !fields(m, &["role", "content"]) || !m["role"].is_string() || !m["content"].is_string()
    }) {
        return Err(err("retained Inspect provider messages must be plain text"));
    }
    let mut expected = normalized_request["messages"].clone();
    if gpt56 {
        for m in expected.as_array_mut().unwrap() {
            if m["role"] == "system" {
                m["role"] = json!("developer")
            }
        }
    }
    if request["messages"] != expected {
        return Err(err(
            "retained Inspect provider messages differ from the approved request",
        ));
    }
    let requested = normalized_request["model"].as_str().unwrap_or("");
    if !request["model"].as_str().is_some_and(|m| {
        m == requested
            || m == requested
                .split_once('/')
                .map(|(_, s)| s)
                .unwrap_or(requested)
    }) {
        return Err(err(
            "retained Inspect provider model differs from the approved request",
        ));
    }
    if !normalized {
        let expected = &normalized_request["config"];
        if gpt56
            && (request.get("temperature").is_some()
                || crate::scoring::number(&expected["temperature"])?
                    != crate::scoring::decimal("1")?
                || request.get("max_completion_tokens").is_none())
        {
            return Err(err(
                "retained Inspect GPT-5.6 controls differ from its approved projection",
            ));
        }
        for k in ["temperature", "top_p"] {
            if gpt56 && k == "temperature" {
                continue;
            }
            let Some(v) = request.get(k) else {
                return Err(err("retained Inspect provider config is incomplete"));
            };
            if !v.is_number() {
                return Err(err("retained Inspect provider controls must be numeric"));
            }
            if crate::scoring::number(v)? != crate::scoring::number(&expected[k])? {
                return Err(err(
                    "retained Inspect provider config differs from the request",
                ));
            }
        }
        if !py_equal(&request["seed"], &expected["seed"])
            || !request["seed"].is_null() && !strict(&request["seed"])
        {
            return Err(err(
                "retained Inspect provider seed differs from the request",
            ));
        }
        let limits = ["max_tokens", "max_completion_tokens"]
            .iter()
            .filter(|k| request.get(**k).is_some())
            .count();
        if limits != 1 {
            return Err(err(
                "retained Inspect provider request must use one token limit",
            ));
        }
        let limit = request
            .get("max_tokens")
            .unwrap_or(&request["max_completion_tokens"]);
        if !strict(limit) || !py_equal(limit, &expected["max_output_tokens"]) {
            return Err(err(
                "retained Inspect provider token limit differs from the request",
            ));
        }
        if request
            .get("n")
            .is_some_and(|v| !strict(v) || v.as_u64() != Some(1))
        {
            return Err(err(
                "retained Inspect provider request must select one completion",
            ));
        }
        if request["reasoning_effort"] != expected["reasoning_effort"] {
            return Err(err(
                "retained Inspect provider reasoning effort differs from the request",
            ));
        }
    }
    if !request["tools"].is_null() && request["tools"] != json!([]) {
        return Err(err("retained Inspect provider request contains tools"));
    }
    if !request["tool_choice"].is_null() && request["tool_choice"] != "none" {
        return Err(err(
            "retained Inspect provider request contains an unsupported tool choice",
        ));
    }
    if completed {
        provider_response(&call["response"], output, request)?
    }
    Ok(())
}
fn event(attempt: &Value, event: &Value, collection: &Value) -> Result<(), AnalysisError> {
    if !attempt.is_object() || !event.is_object() {
        return Err(err("retained Inspect attempts and events must be objects"));
    }
    if !fields(
        event,
        &[
            "event",
            "uuid",
            "role",
            "model",
            "input",
            "tools",
            "tool_choice",
            "config",
            "retries",
            "cache",
            "call",
            "output",
            "error",
        ],
    ) {
        return Err(err("retained Inspect model event has unsupported fields"));
    }
    let mapping = &attempt["source"];
    let output = &event["output"];
    let call = &event["call"];
    let request = &attempt["request"];
    if !mapping.is_object()
        || event["event"] != "model"
        || event["uuid"] != mapping["model_event_id"]
        || event["role"] != "grader"
        || !output.is_object()
        || !call.is_object()
        || !request.is_object()
        || event["model"].is_null()
    {
        return Err(err(
            "retained Inspect model event does not match its attempt",
        ));
    }
    if !fields(
        output,
        &[
            "model",
            "request_id",
            "finish_reason",
            "usage",
            "completion",
        ],
    ) {
        return Err(err("retained Inspect model output has unsupported fields"));
    }
    if !event["cache"].is_null()
        || !strict(&event["retries"])
        || event["retries"].as_u64() != Some(0)
    {
        return Err(err(
            "retained Inspect model events cannot use cache or SDK retries",
        ));
    }
    if event["tools"] != json!([]) || event["tool_choice"] != "none" {
        return Err(err("retained Inspect model events cannot use tools"));
    }
    let normalized = request["text"]
        .as_str()
        .and_then(|s| parse_json(s.as_bytes(), "retained Inspect normalized request").ok())
        .ok_or_else(|| err("retained Inspect normalized request is invalid"))?;
    if !normalized.is_object()
        || event["model"] != normalized["model"]
        || !py_equal(&event["input"], &normalized["messages"])
    {
        return Err(err(
            "retained Inspect input differs from its normalized request",
        ));
    }
    let config = &normalized["config"];
    if !fields(
        config,
        &[
            "temperature",
            "top_p",
            "max_output_tokens",
            "seed",
            "reasoning_effort",
        ],
    ) {
        return Err(err("retained Inspect normalized config must be an object"));
    }
    let temperature = float(&config["temperature"])
        .ok_or_else(|| err("retained Inspect normalized config is invalid"))?;
    let top_p = float(&config["top_p"])
        .ok_or_else(|| err("retained Inspect normalized config is invalid"))?;
    let observed = json!({"temperature":temperature,"top_p":top_p,"max_tokens":config["max_output_tokens"],"seed":config["seed"],"reasoning_effort":config["reasoning_effort"],"max_retries":0,"timeout":collection["request_timeout_seconds"],"attempt_timeout":collection["request_timeout_seconds"],"max_connections":collection["concurrency"],"adaptive_connections":false,"num_choices":1,"internal_tools":false,"parallel_tool_calls":false,"reasoning_summary":"none","reasoning_history":"none","cache":false,"batch":false});
    if !py_equal(&event["config"], &observed) {
        return Err(err(
            "retained Inspect generation config differs from its request",
        ));
    }
    if !fields(call, &["request", "response", "error"]) || !call["request"].is_object() {
        return Err(err("retained Inspect provider call is incomplete"));
    }
    if bytes(call)?.len() > 1024 * 1024 {
        return Err(err(
            "retained Inspect provider call exceeds the 1048576-byte limit",
        ));
    }
    secrets(call)?;
    let completed = attempt["status"] == "completed";
    provider_projection(
        call,
        output,
        &normalized,
        completed,
        &collection["inspect_version"],
    )?;
    if output["model"] != attempt["resolved_model"] {
        return Err(err(
            "retained Inspect resolved model does not match its attempt",
        ));
    }
    let usage = &output["usage"];
    if !py_equal(usage, &attempt["usage"]) {
        return Err(err(
            "retained Inspect token usage does not match its attempt",
        ));
    }
    if completed && usage.is_null() {
        return Err(err("completed retained Inspect call requires token usage"));
    }
    if !usage.is_null() {
        if !fields(usage, &["input_tokens", "output_tokens"]) {
            return Err(err("retained Inspect token usage is invalid"));
        }
        for (k, max) in [
            (
                "input_tokens",
                collection["input_tokens_per_call"].as_u64().unwrap(),
            ),
            (
                "output_tokens",
                config["max_output_tokens"].as_u64().unwrap(),
            ),
        ] {
            if !strict(&usage[k]) || usage[k].as_u64().is_none_or(|v| v > max) {
                return Err(err(format!(
                    "retained Inspect {} exceeds its per-call reservation",
                    k.replace('_', " ")
                )));
            }
        }
    }
    if completed
        && (!event["error"].is_null()
            || !call["error"].is_null() && call["error"] != false
            || !call["response"].is_object())
    {
        return Err(err(
            "completed retained Inspect call has an inconsistent outcome",
        ));
    }
    if !completed
        && (event["error"].is_null()
            || call["error"] != true
            || !call["response"].is_null() && !call["response"].is_object())
    {
        return Err(err(
            "failed retained Inspect call has an inconsistent outcome",
        ));
    }
    if !completed {
        let e = &event["error"];
        if !fields(e, &["status", "code", "message"])
            || e["status"] != attempt["status"]
            || !attempt["error"].is_object()
            || json!({"code":e["code"],"message":e["message"]}) != attempt["error"]
        {
            return Err(err("retained Inspect error differs from its attempt"));
        }
    }
    if !attempt["response"].is_null() {
        let completion = decode(&output["completion"]);
        if String::from_utf8(bytes(&completion)?).unwrap() != attempt["response"]["text"] {
            return Err(err(
                "retained Inspect completion does not match its attempt",
            ));
        }
    }
    if output["request_id"] != attempt["request_id"]
        || output["finish_reason"] != attempt["finish_reason"]
    {
        return Err(err(
            "retained Inspect output metadata does not match its attempt",
        ));
    }
    if event["error"].is_null() != completed {
        return Err(err(
            "retained Inspect error state does not match its attempt",
        ));
    }
    Ok(())
}
pub(crate) fn replay(decoded: &Value) -> Result<Vec<Value>, AnalysisError> {
    if !fields(
        decoded,
        &["format", "inspect_version", "collection", "records"],
    ) {
        return Err(err("retained Inspect source has an unsupported shape"));
    }
    if decoded["format"] != "invarlock/retained-inspect-model-events-v1"
        || !matches!(
            decoded["inspect_version"].as_str(),
            Some("0.3.254" | "0.3.263")
        )
        || !decoded["records"].is_array()
    {
        return Err(err("retained Inspect source has an unsupported profile"));
    }
    let c = &decoded["collection"];
    if !fields(
        c,
        &[
            "grader",
            "inspect_version",
            "profile",
            "epochs",
            "log_model_api",
            "log_samples",
            "sdk_max_retries",
            "tools",
            "concurrency",
            "requests_per_minute",
            "request_timeout_seconds",
            "max_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_cost_microusd",
            "input_tokens_per_call",
            "cost_microusd_per_call",
        ],
    ) {
        return Err(err("retained Inspect collection options must be an object"));
    }
    let grader = c["grader"].as_str().unwrap_or("");
    if grader.is_empty()
        || grader.chars().count() > 256
        || grader.chars().any(|ch| ch < '!' || ":@?#\\".contains(ch))
        || c["inspect_version"] != decoded["inspect_version"]
        || c["profile"] != "inspect-text-frozen-answer-v1"
        || !strict(&c["epochs"])
        || c["epochs"].as_u64() != Some(1)
        || c["log_model_api"] != true
        || c["log_samples"] != true
        || !strict(&c["sdk_max_retries"])
        || c["sdk_max_retries"].as_u64() != Some(0)
        || c["tools"] != false
    {
        return Err(err(
            "retained Inspect collection configuration is unsupported",
        ));
    }
    for (k, max) in [
        ("concurrency", 32),
        ("requests_per_minute", 10000),
        ("request_timeout_seconds", 3600),
        ("max_calls", 600000),
        ("max_input_tokens", 1000000000000),
        ("max_output_tokens", 1000000000000),
        ("max_cost_microusd", 1000000000000),
        ("input_tokens_per_call", 1048576),
        ("cost_microusd_per_call", 1000000000),
    ] {
        if !strict(&c[k]) || !c[k].as_u64().is_some_and(|n| n >= 1 && n <= max) {
            return Err(err(format!(
                "retained Inspect {k} is outside its supported range"
            )));
        }
    }
    let mut trials = Vec::new();
    let mut spent = 0u64;
    let mut output = 0u64;
    for record in decoded["records"].as_array().unwrap() {
        if !fields(record, &["trial", "events"]) {
            return Err(err(
                "retained Inspect records must contain one trial and its events",
            ));
        }
        let trial = &record["trial"];
        let aa = trial["attempts"].as_array();
        let events = record["events"].as_array();
        if !trial.is_object()
            || aa.is_none()
            || events.is_none()
            || aa.unwrap().len() != events.unwrap().len()
        {
            return Err(err(
                "retained Inspect event count must match the trial attempts",
            ));
        }
        for (a, e) in aa.unwrap().iter().zip(events.unwrap()) {
            event(a, e, c)?;
            spent += 1;
            output += e["config"]["max_tokens"].as_u64().unwrap()
        }
        trials.push(trial.clone())
    }
    if spent > c["max_calls"].as_u64().unwrap()
        || spent * c["input_tokens_per_call"].as_u64().unwrap()
            > c["max_input_tokens"].as_u64().unwrap()
        || output > c["max_output_tokens"].as_u64().unwrap()
        || spent * c["cost_microusd_per_call"].as_u64().unwrap()
            > c["max_cost_microusd"].as_u64().unwrap()
    {
        return Err(err(
            "retained Inspect calls exceed their declared resource reservations",
        ));
    }
    Ok(trials)
}
pub(crate) fn family(sources: &[Value], plan: &Value, spent: usize) -> Result<(), AnalysisError> {
    if sources.is_empty() {
        return Ok(());
    }
    let c = &sources[0]["collection"];
    if sources.iter().any(|s| !py_equal(&s["collection"], c)) {
        return Err(err(
            "retained Inspect shards must bind the same collection options",
        ));
    }
    let grader = c["grader"].as_str().unwrap();
    if c["grader"] != plan["judge"]["requested_model"] {
        return Err(err("grader differs from approved plan"));
    }
    if plan["judge"]["model_identity"]["kind"] != "hosted_api" {
        return Err(err(
            "local weight execution is not qualified by this adapter",
        ));
    }
    let provider = grader.split('/').next().unwrap();
    let config = &plan["judge"]["config"];
    if ["anthropic", "google"].contains(&provider) && !config["seed"].is_null() {
        return Err(err(format!(
            "the pinned Inspect {provider} adapter does not support seed"
        )));
    }
    if provider == "anthropic"
        && crate::scoring::number(&config["top_p"])? != crate::scoring::decimal("1")?
    {
        return Err(err("the configured Anthropic profile requires top_p=1"));
    }
    if ["anthropic", "google"].contains(&provider)
        && !config["reasoning_effort"].is_null()
        && config["reasoning_effort"] != "none"
    {
        let model = grader.split_once('/').map(|(_, s)| s).unwrap_or("");
        let supported = if provider == "anthropic" {
            ["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"]
                .iter()
                .any(|s| model.starts_with(s))
        } else {
            model.starts_with("gemini-2.5-")
        };
        let budget = match config["reasoning_effort"].as_str() {
            Some("minimal") => 2048,
            Some("low") => 4096,
            Some("medium") => 10000,
            Some("high") => 16000,
            Some("xhigh" | "max") => 32000,
            _ => 0,
        };
        if !supported || budget == 0 {
            return Err(err(format!(
                "the pinned Inspect {provider} reasoning effort is not qualified for this model"
            )));
        }
        if config["max_output_tokens"].as_u64().unwrap() <= budget {
            return Err(err(format!(
                "the pinned Inspect {provider} output limit must exceed its reasoning budget"
            )));
        }
    }
    if ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"].contains(&grader)
        && (c["inspect_version"] != "0.3.263"
            || crate::scoring::number(&config["temperature"])? != crate::scoring::decimal("1")?
            || !config["reasoning_effort"]
                .as_str()
                .is_some_and(|s| ["none", "low", "medium", "high", "xhigh", "max"].contains(&s)))
    {
        return Err(err(format!("{grader} requires Inspect 0.3.263, approved temperature 1, and a supported explicit reasoning_effort")));
    }
    let spent = spent as u64;
    if spent > c["max_calls"].as_u64().unwrap()
        || spent * c["input_tokens_per_call"].as_u64().unwrap()
            > c["max_input_tokens"].as_u64().unwrap()
        || spent * config["max_output_tokens"].as_u64().unwrap()
            > c["max_output_tokens"].as_u64().unwrap()
        || spent * c["cost_microusd_per_call"].as_u64().unwrap()
            > c["max_cost_microusd"].as_u64().unwrap()
    {
        return Err(err(
            "retained Inspect shards exceed their aggregate resource reservations",
        ));
    }
    Ok(())
}
pub fn validate_retained_inspect_event(
    attempt: &Value,
    event_value: &Value,
    collection: &Value,
) -> Result<(), AnalysisError> {
    event(attempt, event_value, collection)
}
pub fn replay_retained_inspect_source(decoded: &Value) -> Result<Vec<Value>, AnalysisError> {
    replay(decoded)
}
