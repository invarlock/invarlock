//! Expanded Inspect export normalization and deterministic retained-source sharding.
use crate::judge_contracts::{bare_digest, bytes, expected_judge_trial_id};
use crate::{measurement_plan_digest, render_judge_request, validate_measurements, AnalysisError};
use invarlock_native_contracts::parse_json;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
const SOURCE_LIMIT: usize = 16 * 1024 * 1024;
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn object(v: &Value, keys: &[&str], label: &str) -> Result<(), AnalysisError> {
    if !v
        .as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
    {
        return Err(err(format!("invalid {label} fields")));
    }
    Ok(())
}
fn integer(v: &Value, low: u64, high: u64, label: &str) -> Result<u64, AnalysisError> {
    v.as_u64()
        .filter(|v| *v >= low && *v <= high)
        .ok_or_else(|| err(format!("invalid {label}")))
}
fn text<'a>(v: &'a Value, max: usize, label: &str) -> Result<&'a str, AnalysisError> {
    v.as_str()
        .filter(|s| !s.is_empty() && s.chars().count() <= max)
        .ok_or_else(|| err(format!("invalid {label}")))
}
pub fn validate_inspect_collection_options(
    options: &Value,
    plan: &Value,
) -> Result<String, AnalysisError> {
    let digest = measurement_plan_digest(plan)?;
    validate_options(options)?;
    crate::judge_inspect::family(&[json!({"collection":options})], plan, 0)?;
    Ok(digest)
}
pub fn validate_inspect_collection_budgets(
    c: &Value,
    plan: &Value,
) -> Result<Value, AnalysisError> {
    let keys = [
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
    ];
    if !c
        .as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
    {
        return Err(err(
            "collection configuration must contain exactly the supported Inspect judge fields",
        ));
    }
    if c["inspect_version"] != "0.3.263"
        || c["profile"] != "inspect-text-frozen-answer-v1"
        || c["epochs"].as_u64() != Some(1)
        || c["log_model_api"] != true
        || c["log_samples"] != true
        || c["sdk_max_retries"].as_u64() != Some(0)
        || c["tools"] != false
    {
        return Err(err("collection identity, logging, retry, epoch, or tool settings differ from the supported Inspect judge profile"));
    }
    let mut budgets = json!({});
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
        if !c[k].as_u64().is_some_and(|n| n >= 1 && n <= max) {
            return Err(err(format!(
                "collection {k} must be an explicit bounded positive integer"
            )));
        }
        budgets[k] = c[k].clone();
    }
    if plan["schedule"]["max_attempts"] != 1 {
        return Err(err(
            "live Inspect collection currently requires one attempt per trial",
        ));
    }
    let grader = c["grader"]
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.chars().count() <= 256
                && !s.chars().any(|c| c < '!' || ":@?#\\".contains(c))
        })
        .ok_or_else(|| err("grader must be an explicit model identity without URL credentials"))?;
    crate::judge_inspect::family(&[json!({"collection":c})], plan, 0)?;
    let (provider, model) = grader.split_once('/').unwrap_or((grader, ""));
    if provider == "anthropic" || provider == "google" {
        let prefix = if provider == "anthropic" {
            "claude-"
        } else {
            "gemini-"
        };
        let mut rest = model.strip_prefix(prefix).unwrap_or("");
        if provider == "anthropic" {
            let alpha = rest.bytes().take_while(u8::is_ascii_lowercase).count();
            if alpha > 0 && rest.as_bytes().get(alpha) == Some(&b'-') {
                rest = &rest[alpha + 1..]
            }
        }
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let mut major = rest[..digits].parse::<u64>().ok();
        rest = &rest[digits..];
        let mut minor = 0;
        let separator = if provider == "anthropic" { '-' } else { '.' };
        if let Some(tail) = rest.strip_prefix(separator) {
            let n = tail.bytes().take_while(u8::is_ascii_digit).count();
            if n > 0 {
                minor = tail[..n].parse::<u64>().unwrap_or(u64::MAX);
                rest = &tail[n..]
            }
        }
        if !rest.is_empty() && !rest.starts_with('-') {
            major = None
        }
        let qualified = major.is_some_and(|major| {
            if provider == "anthropic" {
                (major, minor) < (4, 6)
            } else {
                major < 3
            }
        });
        if !qualified {
            return Err(err(format!(
                "the pinned Inspect {provider} live wire shape is not qualified for this model"
            )));
        }
    }
    Ok(budgets)
}
fn validate_options(o: &Value) -> Result<(), AnalysisError> {
    if !o.is_object() {
        return Err(err("collection options must be an object"));
    }
    let grader = text(&o["grader"], 256, "explicit grader")?;
    object(
        o,
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
        "collection options",
    )?;
    if grader.chars().any(|c| c < '!' || ":@?#\\".contains(c)) {
        return Err(err(
            "grader must be an explicit model identity without URL credentials",
        ));
    }
    if !matches!(o["inspect_version"].as_str(), Some("0.3.254" | "0.3.263")) {
        return Err(err("unsupported Inspect version"));
    }
    if o["profile"] != "inspect-text-frozen-answer-v1" {
        return Err(err("unsupported Inspect profile"));
    }
    if o["epochs"].as_u64() != Some(1) {
        return Err(err("only epoch one is supported"));
    }
    if o["log_model_api"] != true || o["log_samples"] != true {
        return Err(err("complete API and sample logging is required"));
    }
    if o["sdk_max_retries"].as_u64() != Some(0) {
        return Err(err("SDK retries must be disabled"));
    }
    if o["tools"] != false {
        return Err(err("tools are unsupported"));
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
        integer(&o[k], 1, max, k)?;
    }
    Ok(())
}
fn blob(v: &Value) -> Result<Value, AnalysisError> {
    let content = bytes(v)?;
    if content.len() > 1048576 {
        return Err(err("request or response exceeds byte allowance"));
    }
    Ok(
        json!({"media_type":"application/json","text":String::from_utf8(content.clone()).unwrap(),"sha256":bare_digest(&content)}),
    )
}
fn completion(v: &Value) -> Result<Value, AnalysisError> {
    let value = v
        .as_str()
        .ok_or_else(|| err("model completion must be text"))?;
    if value.len() > 1048576 {
        return Err(err("model completion exceeds byte allowance"));
    }
    Ok(parse_json(value.as_bytes(), "judge completion").unwrap_or_else(|_| v.clone()))
}
fn rating(v: &Value, plan: &Value) -> Value {
    if v.as_object()
        .is_some_and(|o| o.len() == 1 && o.contains_key("rating"))
        && v["rating"].is_string()
    {
        if let Some(r) = plan["scale"]["ratings"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["label"] == v["rating"])
        {
            return json!({"status":"ok","rating":r["label"],"value":r["value"]});
        }
    }
    json!({"status":"invalid","rating":null,"value":null})
}
fn credentials(v: &Value) -> Result<(), AnalysisError> {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        if let Some(o) = v.as_object() {
            for (k, v) in o {
                let k = crate::scoring::casefold_text(k);
                if [
                    "authorization",
                    "api-key",
                    "api_key",
                    "apikey",
                    "access_token",
                    "secret",
                ]
                .contains(&k.as_str())
                {
                    return Err(err("provider request contains a credential field"));
                }
                if ["headers", "extra_headers", "http_headers"].contains(&k.as_str())
                    && !v.as_object().is_some_and(|o| {
                        o.iter().all(|(k, v)| {
                            crate::scoring::casefold_text(k) == "x-irid"
                                && v.as_str()
                                    .is_some_and(|s| !s.is_empty() && s.chars().count() <= 128)
                        })
                    })
                {
                    return Err(err("provider request contains unsupported headers"));
                }
                stack.push(v)
            }
        } else if let Some(a) = v.as_array() {
            stack.extend(a)
        }
    }
    Ok(())
}
/// Immutable preparation reused for incremental event import. The plan, options,
/// frozen runs and request bindings are validated when this context is built.
pub struct InspectImportContext {
    plan: Value,
    options: Value,
    digest: String,
    frozen: BTreeMap<String, Value>,
    bindings: BTreeMap<String, Value>,
}
impl InspectImportContext {
    pub fn new(
        plan: &Value,
        options: &Value,
        baseline: &Value,
        subject: &Value,
    ) -> Result<Self, AnalysisError> {
        let digest = validate_inspect_collection_options(options, plan)?;
        Self::prepared(plan, options, baseline, subject, digest)
    }
    fn prepared(
        plan: &Value,
        options: &Value,
        baseline: &Value,
        subject: &Value,
        digest: String,
    ) -> Result<Self, AnalysisError> {
        if !baseline.is_object() || !subject.is_object() {
            return Err(err("import requires both frozen runs"));
        }
        let mut frozen = BTreeMap::<String, Value>::new();
        for (side, run) in [("baseline", baseline), ("subject", subject)] {
            let rows = run["records"]
                .as_array()
                .ok_or_else(|| err("frozen run requires records"))?;
            let mut seen = BTreeSet::new();
            for record in rows {
                let id = record["id"]
                    .as_str()
                    .filter(|_| record.is_object())
                    .ok_or_else(|| err("invalid frozen record"))?;
                if !seen.insert(id) {
                    return Err(err("duplicate frozen case"));
                }
                if !record["input"].is_string()
                    || !record["output"].is_string()
                    || !record["error"].is_null()
                {
                    return Err(err("frozen records require successful text answers"));
                }
                let row = frozen
                    .entry(id.to_owned())
                    .or_insert_with(|| json!({"input":record["input"]}));
                if row["input"] != record["input"] {
                    return Err(err("frozen paired inputs differ"));
                }
                row[side] = record["output"].clone();
                if plan["prompt"]["reference_mode"] == "per_case" {
                    if !record["expected"].is_string() {
                        return Err(err("per-case judging requires a string reference"));
                    }
                    if row
                        .get("expected")
                        .is_some_and(|v| v != &record["expected"])
                    {
                        return Err(err("frozen paired references differ"));
                    }
                    row["expected"] = record["expected"].clone()
                }
            }
        }
        let bindings = plan["answer_bindings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| (b["case_id"].as_str().unwrap().to_owned(), b.clone()))
            .collect::<BTreeMap<_, _>>();
        if frozen.keys().ne(bindings.keys()) {
            return Err(err("frozen input membership differs from plan"));
        }
        for (id, row) in &frozen {
            let mut keys = vec!["input", "baseline", "subject"];
            if plan["prompt"]["reference_mode"] == "per_case" {
                keys.push("expected")
            }
            object(row, &keys, "frozen input")?;
            for side in ["baseline", "subject"] {
                if !row[side].is_string()
                    || bare_digest(row[side].as_str().unwrap().as_bytes())
                        != bindings[id][format!("{side}_answer_sha256")]
                {
                    return Err(err("frozen answer digest mismatch"));
                }
            }
        }
        crate::judge_contracts::validate_frozen_answers(plan, baseline, subject)?;
        Ok(Self {
            plan: plan.clone(),
            options: options.clone(),
            digest,
            frozen,
            bindings,
        })
    }
    pub fn normalize_sample(
        &self,
        sample: &Value,
        record_index: usize,
    ) -> Result<Value, AnalysisError> {
        self.sample(sample, record_index, &mut BTreeSet::new())
    }
    fn sample(
        &self,
        sample: &Value,
        record_index: usize,
        event_ids: &mut BTreeSet<String>,
    ) -> Result<Value, AnalysisError> {
        let plan = &self.plan;
        let options = &self.options;
        object(sample, &["id", "epoch", "metadata", "events"], "sample")?;
        if sample["epoch"].as_u64() != Some(1) {
            return Err(err("only epoch one is supported"));
        }
        let m = &sample["metadata"];
        object(
            m,
            &[
                "case_id",
                "side",
                "repetition",
                "plan_sha256",
                "answer_sha256",
                "scorer_id",
            ],
            "sample metadata",
        )?;
        let id = m["case_id"].as_str().unwrap_or("");
        let side = m["side"].as_str().unwrap_or("");
        if !self.bindings.contains_key(id) || !["baseline", "subject"].contains(&side) {
            return Err(err("unknown sample binding"));
        }
        let repetition = integer(
            &m["repetition"],
            1,
            plan["schedule"]["repetitions"].as_u64().unwrap(),
            "repetition",
        )?;
        let trial_id = expected_judge_trial_id(&self.digest, id, side, repetition)?;
        if sample["id"] != trial_id
            || m["plan_sha256"] != self.digest
            || m["answer_sha256"] != self.bindings[id][format!("{side}_answer_sha256")]
        {
            return Err(err("sample binding mismatch"));
        }
        text(&m["scorer_id"], 128, "scorer identity")?;
        let frozen = &self.frozen[id];
        if frozen["input"].as_str().unwrap().len() > 1048576 {
            return Err(err("input must be bounded text"));
        }
        if frozen[side].as_str().unwrap().len() > 1048576 {
            return Err(err("answer must be bounded text"));
        }
        let request =
            render_judge_request(plan, &frozen["input"], &frozen[side], &frozen["expected"])?;
        let expected =
            parse_json(&request, "normalized judge request").map_err(|e| err(e.to_string()))?;
        if bare_digest(&request) != self.bindings[id][format!("{side}_request_sha256")] {
            return Err(err("rendered request digest differs from approved plan"));
        }
        let events = sample["events"]
            .as_array()
            .filter(|a| a.len() as u64 <= plan["schedule"]["max_attempts"].as_u64().unwrap())
            .ok_or_else(|| err("invalid attempt count"))?;
        let mut attempts = Vec::new();
        let mut parsed = json!({"status":"unavailable","rating":null,"value":null});
        let mut selected = None;
        for (index, event) in events.iter().enumerate() {
            if bytes(event)?.len() > 2 * 1024 * 1024 {
                return Err(err("model event exceeds the retained-event byte allowance"));
            }
            object(
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
                "model event",
            )?;
            let event_id = text(&event["uuid"], 128, "model event ID")?;
            if !event_ids.insert(event_id.to_owned()) {
                return Err(err("duplicate model event ID"));
            }
            if event["event"] != "model"
                || event["role"] != "grader"
                || event["model"] != options["grader"]
            {
                return Err(err("explicit grader event is required"));
            }
            if event["tools"] != json!([]) || event["tool_choice"] != "none" {
                return Err(err("tools are unsupported"));
            }
            if event["retries"].as_u64() != Some(0) || !event["cache"].is_null() {
                return Err(err("SDK retries and caching are unsupported"));
            }
            let config = &plan["judge"]["config"];
            let expected_config = json!({"temperature":config["temperature"].as_str().unwrap().parse::<f64>().unwrap(),"top_p":config["top_p"].as_str().unwrap().parse::<f64>().unwrap(),"max_tokens":config["max_output_tokens"],"seed":config["seed"],"reasoning_effort":config["reasoning_effort"],"max_retries":0,"timeout":options["request_timeout_seconds"],"attempt_timeout":options["request_timeout_seconds"],"max_connections":options["concurrency"],"adaptive_connections":false,"num_choices":1,"internal_tools":false,"parallel_tool_calls":false,"reasoning_summary":"none","reasoning_history":"none","cache":false,"batch":false});
            if bytes(&event["config"])? != bytes(&expected_config)? {
                return Err(err("model event has hidden or changed generation settings"));
            }
            if !crate::analysis::py_equal(&event["input"], &expected["messages"]) {
                return Err(err("model input differs from frozen request"));
            }
            let call = &event["call"];
            object(call, &["request", "response", "error"], "full API call")?;
            if !call["request"].is_object() || bytes(&call["request"])?.len() > 1048576 {
                return Err(err("full API request must be a bounded object"));
            }
            credentials(&call["request"])?;
            let output = &event["output"];
            object(
                output,
                &[
                    "model",
                    "request_id",
                    "finish_reason",
                    "usage",
                    "completion",
                ],
                "accessible output",
            )?;
            let error = &event["error"];
            let (status, error) = if error.is_null() {
                if !call["error"].is_null()
                    && !crate::analysis::py_equal(&call["error"], &json!(false))
                    || !call["response"].is_object()
                {
                    return Err(err("completed call requires full response"));
                }
                if !plan["judge"]["approved_resolved_models"]
                    .as_array()
                    .unwrap()
                    .contains(&output["model"])
                {
                    return Err(err("unapproved resolved model"));
                }
                parsed = rating(&completion(&output["completion"])?, plan);
                selected = Some(index + 1);
                ("completed", Value::Null)
            } else {
                object(error, &["status", "code", "message"], "attempt error")?;
                let status = error["status"].as_str().unwrap_or("");
                if ![
                    "transport_error",
                    "timeout_ambiguous",
                    "cancelled",
                    "refusal",
                ]
                .contains(&status)
                    || call["error"] != true
                {
                    return Err(err("unknown attempt error"));
                }
                if !call["response"].is_null() && !call["response"].is_object() {
                    return Err(err("failed API response must be an object or null"));
                }
                parsed = json!({"status":if status=="refusal"{"refusal"}else{"unavailable"},"rating":null,"value":null});
                (
                    status,
                    json!({"code":error["code"],"message":error["message"]}),
                )
            };
            let usage = &output["usage"];
            if !usage.is_null() {
                object(usage, &["input_tokens", "output_tokens"], "token usage")?;
                integer(
                    &usage["input_tokens"],
                    0,
                    options["input_tokens_per_call"].as_u64().unwrap(),
                    "input token usage",
                )?;
                integer(
                    &usage["output_tokens"],
                    0,
                    config["max_output_tokens"].as_u64().unwrap(),
                    "output token usage",
                )?;
            }
            if status == "completed" && usage.is_null() {
                return Err(err("completed call requires token usage"));
            }
            if index + 1 < events.len()
                && (status != "transport_error"
                    || !plan["schedule"]["retry_on"]
                        .as_array()
                        .unwrap()
                        .contains(&json!("transport_error")))
            {
                return Err(err("cannot retry a terminal judgment"));
            }
            let attempt = json!({"attempt":index+1,"role":"judge","resolved_model":output["model"],"status":status,"request":blob(&expected)?,"response":if status=="completed"{blob(&completion(&output["completion"])?)?}else{Value::Null},"request_id":output["request_id"],"finish_reason":output["finish_reason"],"error":error,"usage":usage,"cache":"none","source":{"source_id":"inspect-export","scorer_id":m["scorer_id"],"model_event_id":event["uuid"],"record_index":record_index,"attempt_index":index}});
            attempts.push(attempt);
        }
        Ok(
            json!({"trial_id":trial_id,"case_id":id,"side":side,"repetition":repetition,"answer_sha256":m["answer_sha256"],"plan_sha256":self.digest,"status":if parsed["status"]=="ok"{"complete"}else{"incomplete"},"attempts":attempts,"selected_attempt":selected,"parse":parsed}),
        )
    }
    /// Normalize and replay provider facts for one replacement before publication.
    pub fn validate_sample(
        &self,
        sample: &Value,
        record_index: usize,
    ) -> Result<Value, AnalysisError> {
        let trial = self.normalize_sample(sample, record_index)?;
        crate::judge_contracts::validate_imported_trial(&trial, &self.plan)?;
        for (a, e) in trial["attempts"]
            .as_array()
            .unwrap()
            .iter()
            .zip(sample["events"].as_array().unwrap())
        {
            crate::judge_inspect::validate_retained_inspect_event(a, e, &self.options)?;
        }
        Ok(trial)
    }
}
fn document(options: &Value, records: &[Value]) -> Value {
    json!({"format":"invarlock/retained-inspect-model-events-v1","inspect_version":options["inspect_version"],"collection":options,"records":records})
}
fn publish(
    sources: &mut Vec<Value>,
    options: &Value,
    source_id: &str,
    records: &[Value],
) -> Result<(), AnalysisError> {
    if sources.len() >= 1000 {
        return Err(err("retained sources exceed the source allowance"));
    }
    let payload = bytes(&document(options, records))?;
    if payload.len() > SOURCE_LIMIT {
        return Err(err("retained source exceeds byte allowance"));
    }
    sources.push(json!({"source_id":source_id,"profile":"retained-inspect-model-events-v1","encoding":"utf-8","byte_size":payload.len(),"media_type":"application/json","content":String::from_utf8(payload.clone()).unwrap(),"sha256":bare_digest(&payload)}));
    Ok(())
}
fn assemble(
    digest: &str,
    mut trials: Vec<Value>,
    samples: &[Value],
    options: &Value,
) -> Result<Value, AnalysisError> {
    let mut sources = Vec::new();
    let mut records = Vec::new();
    let mut source_id = "inspect-export".to_owned();
    let header = bytes(&document(options, &[]))?.len();
    let mut source_bytes = header;
    for (trial, sample) in trials.iter_mut().zip(samples) {
        for attempt in trial["attempts"].as_array_mut().unwrap() {
            attempt["source"]["source_id"] = json!(source_id);
            attempt["source"]["record_index"] = json!(records.len())
        }
        let mut record = json!({"trial":trial,"events":sample["events"]});
        let mut size = bytes(&record)?.len();
        if !records.is_empty() && source_bytes + 1 + size > SOURCE_LIMIT {
            publish(&mut sources, options, &source_id, &records)?;
            records.clear();
            source_bytes = header;
            source_id = format!("inspect-export-{:04}", sources.len() + 1);
            for attempt in trial["attempts"].as_array_mut().unwrap() {
                attempt["source"]["source_id"] = json!(source_id);
                attempt["source"]["record_index"] = json!(0)
            }
            record = json!({"trial":trial,"events":sample["events"]});
            size = bytes(&record)?.len()
        }
        if source_bytes + usize::from(!records.is_empty()) + size > SOURCE_LIMIT {
            return Err(err(
                "one retained trial exceeds the per-source byte allowance",
            ));
        }
        source_bytes += usize::from(!records.is_empty()) + size;
        records.push(record)
    }
    if !records.is_empty() {
        publish(&mut sources, options, &source_id, &records)?
    }
    let completed = trials.iter().filter(|t| t["status"] == "complete").count();
    let result = json!({"format":"invarlock/judge-measurements-v1","profile_id":"text-frozen-answer-v1","plan_sha256":digest,"source_profile":"retained-inspect-model-events-v1","sources":sources,"trials":trials,"completeness":{"status":if completed==trials.len(){"complete"}else{"incomplete"},"expected_trials":trials.len(),"recorded_trials":trials.len(),"completed_trials":completed}});
    if bytes(&result)?.len() > 384 * 1024 * 1024 {
        return Err(err(
            "retained measurements exceed the aggregate byte allowance",
        ));
    }
    Ok(result)
}
pub fn import_inspect_export(
    exported: &Value,
    plan: &Value,
    options: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<Value, AnalysisError> {
    let digest = validate_inspect_collection_options(options, plan)?;
    if bytes(exported)?.len() > 384 * 1024 * 1024 {
        return Err(err("export exceeds byte allowance"));
    }
    object(
        exported,
        &[
            "format",
            "profile",
            "inspect_version",
            "collection",
            "samples",
        ],
        "export",
    )?;
    if exported["format"] != "invarlock/inspect-judge-export-v1"
        || exported["profile"] != "inspect-text-frozen-answer-v1"
    {
        return Err(err("unsupported export profile"));
    }
    if exported["inspect_version"] != options["inspect_version"] {
        return Err(err("unsupported Inspect version"));
    }
    if bytes(&exported["collection"])? != bytes(options)? {
        return Err(err(
            "export collection settings differ from explicit options",
        ));
    }
    let context = InspectImportContext::prepared(plan, options, baseline, subject, digest.clone())?;
    let samples = exported["samples"]
        .as_array()
        .filter(|a| a.len() as u64 == plan["schedule"]["expected_trials"].as_u64().unwrap())
        .ok_or_else(|| err("export must retain every scheduled slot"))?;
    let mut ids = BTreeSet::new();
    let trials = samples
        .iter()
        .enumerate()
        .map(|(i, s)| context.sample(s, i, &mut ids))
        .collect::<Result<Vec<_>, _>>()?;
    let spent = trials
        .iter()
        .map(|t| t["attempts"].as_array().unwrap().len() as u64)
        .sum::<u64>();
    for (actual, limit, message) in [
        (
            spent,
            options["max_calls"].as_u64().unwrap(),
            "export exceeds the call allowance",
        ),
        (
            spent * options["input_tokens_per_call"].as_u64().unwrap(),
            options["max_input_tokens"].as_u64().unwrap(),
            "export exceeds the reserved input-token allowance",
        ),
        (
            spent
                * plan["judge"]["config"]["max_output_tokens"]
                    .as_u64()
                    .unwrap(),
            options["max_output_tokens"].as_u64().unwrap(),
            "export exceeds the reserved output-token allowance",
        ),
        (
            spent * options["cost_microusd_per_call"].as_u64().unwrap(),
            options["max_cost_microusd"].as_u64().unwrap(),
            "export exceeds the reserved cost allowance",
        ),
    ] {
        if actual > limit {
            return Err(err(message));
        }
    }
    let result = assemble(&digest, trials, samples, options)?;
    validate_measurements(&result, plan, baseline, subject)?;
    Ok(result)
}
pub fn import_inspect_sample(
    sample: &Value,
    plan: &Value,
    options: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<Value, AnalysisError> {
    InspectImportContext::new(plan, options, baseline, subject)?.validate_sample(sample, 0)
}
