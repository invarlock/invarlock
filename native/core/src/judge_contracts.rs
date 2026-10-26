//! Retained judge schedule and source replay contracts.
use crate::{validate_run, AnalysisError};
use invarlock_native_contracts::{
    canonical_json, canonical_json_no_newline, first_schema_error, parse_json, python_repr,
    sha256_digest, SchemaErrorOrder,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
const REQUEST_LIMIT: usize = 1024 * 1024;
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap()
}
pub(crate) fn bytes(v: &Value) -> Result<Vec<u8>, AnalysisError> {
    canonical_json_no_newline(v).map_err(|e| err(e.to_string()))
}
pub(crate) fn bare_digest(raw: &[u8]) -> String {
    sha256_digest(raw).trim_start_matches("sha256:").to_owned()
}
pub(crate) fn digest(v: &Value) -> Result<String, AnalysisError> {
    Ok(bare_digest(&bytes(v)?))
}
fn bounded(v: &Value, limit: usize, label: &str) -> Result<Vec<u8>, AnalysisError> {
    let b = bytes(v)?;
    if b.len() > limit {
        return Err(err(format!("{label} exceeds the {limit}-byte limit")));
    }
    Ok(b)
}
fn strict_integer(v: &Value, label: &str) -> Result<(), AnalysisError> {
    if !v.is_number() || v.to_string().contains(['.', 'e', 'E']) {
        return Err(err(format!("{label} must be an integer")));
    }
    Ok(())
}
fn schema(v: &Value, name: &str) -> Result<(), AnalysisError> {
    if let Some(violation) =
        first_schema_error(v, name, SchemaErrorOrder::Schema).map_err(|e| err(e.to_string()))?
    {
        let path = violation.path.join("/");
        let reason = violation.message.chars().take(240).collect::<String>();
        return Err(err(if name == "judge_analysis_policy" {
            format!(
                "judge analysis policy is invalid at {}: {reason}",
                if path.is_empty() { "/" } else { &path }
            )
        } else {
            format!(
                "judge {} contract is invalid{}: {reason}",
                if name == "judge_measurement_plan" {
                    "plan"
                } else {
                    "measurements"
                },
                if path.is_empty() {
                    String::new()
                } else {
                    format!(" at {path}")
                }
            )
        }));
    }
    Ok(())
}

pub fn render_judge_request(
    plan: &Value,
    input: &Value,
    answer: &Value,
    reference: &Value,
) -> Result<Vec<u8>, AnalysisError> {
    if !input.is_string() || !answer.is_string() {
        return Err(err(
            "the text-frozen-answer profile requires string inputs and answers",
        ));
    }
    let prompt = &plan["prompt"];
    let mode = prompt
        .get("reference_mode")
        .and_then(Value::as_str)
        .unwrap_or("none");
    if !["none", "per_case"].contains(&mode) {
        return Err(err("unsupported judge reference mode"));
    }
    if mode == "per_case" {
        if !reference.is_string() {
            return Err(err("per-case judging requires a string reference"));
        }
        bounded(reference, REQUEST_LIMIT, "per-case judge reference")?;
    }
    let references = prompt["references"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| json!({"id":r["id"],"text":r["text"]}))
        .collect::<Vec<_>>();
    let mut content_bytes = text(prompt, "system").len();
    let mut messages = Vec::new();
    if content_bytes > 0 {
        messages.push(json!({"role":"system","content":prompt["system"]}))
    }
    let user_content = |input: &Value,
                        answer: &Value,
                        include_reference: bool|
     -> Result<String, AnalysisError> {
        let mut content = json!({"answer":answer,"input":input,"instruction":prompt["template"],"references":references,"rubric":plan["rubric"]["text"]});
        if include_reference {
            content["reference"] = reference.clone()
        }
        String::from_utf8(bounded(
            &content,
            REQUEST_LIMIT,
            "normalized judge request",
        )?)
        .map_err(|e| err(e.to_string()))
    };
    for demo in prompt["demonstrations"].as_array().unwrap() {
        let assistant = String::from_utf8(bounded(
            &json!({"rating":demo["rating"]}),
            REQUEST_LIMIT,
            "normalized judge request",
        )?)
        .unwrap();
        content_bytes += assistant.len();
        if content_bytes > REQUEST_LIMIT {
            return Err(err(format!(
                "normalized judge request exceeds the {REQUEST_LIMIT}-byte limit"
            )));
        }
        let user = user_content(&demo["input"], &demo["answer"], false)?;
        content_bytes += user.len();
        if content_bytes > REQUEST_LIMIT {
            return Err(err(format!(
                "normalized judge request exceeds the {REQUEST_LIMIT}-byte limit"
            )));
        }
        messages.push(json!({"role":"user","content":user}));
        messages.push(json!({"role":"assistant","content":assistant}));
    }
    let user = user_content(input, answer, mode == "per_case")?;
    content_bytes += user.len();
    if content_bytes > REQUEST_LIMIT {
        return Err(err(format!(
            "normalized judge request exceeds the {REQUEST_LIMIT}-byte limit"
        )));
    }
    messages.push(json!({"role":"user","content":user}));
    bounded(
        &json!({"config":plan["judge"]["config"],"format":"invarlock/judge-request-v1","messages":messages,"model":plan["judge"]["requested_model"],"model_role":"judge","response_format":{"additional_properties":false,"rating_labels":plan["scale"]["ratings"].as_array().unwrap().iter().map(|r|r["label"].clone()).collect::<Vec<_>>(),"required":["rating"],"type":"json_object"},"tools":[]}),
        REQUEST_LIMIT,
        "normalized judge request",
    )
}
pub fn render_judge_runtime_prompt(plan: &Value, request: &[u8]) -> Result<Vec<u8>, AnalysisError> {
    match plan["prompt"]
        .get("runtime_format")
        .and_then(Value::as_str)
        .unwrap_or("canonical-json-v1")
    {
        "canonical-json-v1" => Ok(request.to_vec()),
        "chatml-v1" => {
            let value = parse_json(request, "normalized direct judge request")
                .map_err(|e| err(e.to_string()))?;
            let messages = value["messages"]
                .as_array()
                .ok_or_else(|| err("normalized direct judge request must contain messages"))?;
            if messages.is_empty() || messages.last().unwrap()["role"] != "user" {
                return Err(err(
                    "normalized direct judge request must end with a user message",
                ));
            }
            let mut text = String::new();
            for m in messages {
                if !m.as_object().is_some_and(|m| {
                    m.len() == 2 && m.contains_key("role") && m.contains_key("content")
                }) {
                    return Err(err(
                        "normalized direct judge messages have an unsupported shape",
                    ));
                }
                let role = m["role"].as_str().unwrap_or("");
                let content = m["content"].as_str().ok_or_else(|| {
                    err("normalized direct judge messages require closed text roles")
                })?;
                if !["system", "user", "assistant"].contains(&role) {
                    return Err(err(
                        "normalized direct judge messages require closed text roles",
                    ));
                }
                if ["<|im_start|>", "<|im_end|>", "<|im_sep|>"]
                    .iter()
                    .any(|s| content.contains(s))
                {
                    return Err(err(
                        "direct judge prompt content contains a reserved ChatML delimiter",
                    ));
                }
                text.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            text.push_str("<|im_start|>assistant\n");
            if text.len() > REQUEST_LIMIT {
                return Err(err(format!(
                    "direct judge prompt exceeds the {REQUEST_LIMIT}-byte limit"
                )));
            }
            Ok(text.into_bytes())
        }
        _ => Err(err("unsupported direct judge runtime prompt format")),
    }
}
pub fn validate_measurement_plan(plan: &Value) -> Result<(), AnalysisError> {
    if plan["sampling"]["case_units"]
        .as_array()
        .is_some_and(|v| v.len() > 10000)
    {
        return Err(err("judge measurement plan exceeds the case limit"));
    }
    if plan["answer_bindings"]
        .as_array()
        .is_some_and(|v| v.len() > 10000)
    {
        return Err(err(
            "judge measurement plan exceeds the answer-binding limit",
        ));
    }
    bounded(plan, 64 * 1024 * 1024, "judge measurement plan")?;
    schema(plan, "judge_measurement_plan")?;
    if bare_digest(text(&plan["rubric"], "text").as_bytes()) != plan["rubric"]["sha256"] {
        return Err(err("rubric digest does not match its UTF-8 text"));
    }
    let mut references = BTreeSet::new();
    for r in plan["prompt"]["references"].as_array().unwrap() {
        if !references.insert(text(r, "id")) {
            return Err(err("prompt reference IDs must be unique"));
        }
        if bare_digest(text(r, "text").as_bytes()) != r["sha256"] {
            return Err(err(format!(
                "prompt reference {} digest does not match",
                python_repr(&r["id"])
            )));
        }
    }
    let mut ratings = BTreeSet::new();
    let mut values = BTreeSet::new();
    for r in plan["scale"]["ratings"].as_array().unwrap() {
        if !ratings.insert(text(r, "label")) {
            return Err(err("rating labels must be unique"));
        }
        let value = crate::scoring::decimal(text(r, "value"))?;
        if !values.insert(value) {
            return Err(err("rating numeric values must be unique"));
        }
    }
    if plan["prompt"]["demonstrations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| !ratings.contains(text(d, "rating")))
    {
        return Err(err(
            "every demonstration rating must exist in the declared scale",
        ));
    }
    let judge = &plan["judge"];
    let identity = &judge["model_identity"];
    if identity["kind"] == "hosted_api" && !identity["weights_sha256"].is_null() {
        return Err(err(
            "hosted API judge identity must not claim a weights digest",
        ));
    }
    if identity["kind"] == "local_weights" && identity["weights_sha256"].is_null() {
        return Err(err("local judge identity requires a weights digest"));
    }
    if judge["provider"] == "openai_compatible" {
        if judge.get("service_identity").is_none_or(Value::is_null) {
            return Err(err(
                "OpenAI-compatible judge plans require a service identity",
            ));
        }
    } else if judge.get("service_identity").is_some_and(|v| !v.is_null()) {
        return Err(err(
            "service identity is only valid for OpenAI-compatible judge plans",
        ));
    }
    if plan["prompt"]
        .get("runtime_format")
        .is_some_and(|v| !v.is_null())
        && !matches!(
            judge["provider"].as_str(),
            Some("hf_transformers" | "llama_cpp")
        )
    {
        return Err(err(
            "runtime prompt format is only valid for direct runtime judge plans",
        ));
    }
    let mut cases = BTreeSet::new();
    for c in plan["sampling"]["case_units"].as_array().unwrap() {
        if !cases.insert(text(c, "case_id")) {
            return Err(err("sampling case IDs must be unique"));
        }
    }
    let mut bindings = BTreeSet::new();
    for c in plan["answer_bindings"].as_array().unwrap() {
        if !bindings.insert(text(c, "case_id")) {
            return Err(err("answer-binding case IDs must be unique"));
        }
    }
    if cases != bindings {
        return Err(err(
            "sampling and answer bindings must contain the same case IDs",
        ));
    }
    let schedule = &plan["schedule"];
    for k in ["repetitions", "max_attempts", "expected_trials"] {
        strict_integer(&schedule[k], &format!("schedule {k}"))?
    }
    strict_integer(
        &judge["config"]["max_output_tokens"],
        "judge max_output_tokens",
    )?;
    if !judge["config"]["seed"].is_null() {
        strict_integer(&judge["config"]["seed"], "judge seed")?
    }
    if schedule["trial_id_scheme"] != "plan-case-side-repetition-sha256-v1" {
        return Err(err("unsupported judge trial ID scheme"));
    }
    if schedule["expected_trials"].as_u64()
        != Some(cases.len() as u64 * 2 * schedule["repetitions"].as_u64().unwrap())
    {
        return Err(err(
            "expected_trials must equal cases × two sides × repetitions",
        ));
    }
    if schedule["max_attempts"].as_u64().unwrap() > 1
        && schedule["retry_on"] != json!(["transport_error"])
    {
        return Err(err(
            "multiple attempts require transport_error as the sole retry condition",
        ));
    }
    let request = render_judge_request(plan, &json!(""), &json!(""), &json!(""))?;
    render_judge_runtime_prompt(plan, &request)?;
    Ok(())
}
pub fn measurement_plan_digest(plan: &Value) -> Result<String, AnalysisError> {
    validate_measurement_plan(plan)?;
    digest(plan)
}
pub fn expected_judge_trial_id(
    plan_digest: &str,
    case_id: &str,
    side: &str,
    repetition: u64,
) -> Result<String, AnalysisError> {
    Ok(format!(
        "trial-{}",
        digest(&json!([plan_digest, case_id, side, repetition]))?
    ))
}
fn validate_policy_shape(policy: &Value) -> Result<(), AnalysisError> {
    if bytes(policy)?.len() > 65536 {
        return Err(err("judge analysis policy exceeds its byte limit"));
    }
    schema(policy, "judge_analysis_policy")?;
    for field in ["comparison_family_size", "minimum_units"] {
        if strict_integer(&policy[field], field).is_err() {
            return Err(err(format!("{field} must be a strict integer")));
        }
    }
    Ok(())
}
pub fn validate_judge_analysis_policy(policy: &Value, plan: &Value) -> Result<(), AnalysisError> {
    validate_policy_shape(policy)?;
    validate_measurement_plan(plan)?;
    check_policy_binding(policy, plan)
}
fn check_policy_binding(policy: &Value, plan: &Value) -> Result<(), AnalysisError> {
    if policy["plan_sha256"] != digest(plan)? {
        return Err(err("judge analysis policy does not bind the supplied plan"));
    }
    Ok(())
}
pub(crate) fn validate_judge_analysis_policy_prechecked(
    policy: &Value,
    plan: &Value,
) -> Result<(), AnalysisError> {
    validate_policy_shape(policy)?;
    check_policy_binding(policy, plan)
}

fn trial_integers(t: &Value) -> Result<(), AnalysisError> {
    strict_integer(&t["repetition"], "trial repetition")?;
    if !t["selected_attempt"].is_null() {
        strict_integer(&t["selected_attempt"], "selected attempt")?
    }
    let attempts = t["attempts"]
        .as_array()
        .ok_or_else(|| err("trial attempts must be an array"))?;
    for a in attempts {
        if !a.is_object() {
            return Err(err("trial attempts must be objects"));
        }
        strict_integer(&a["attempt"], "attempt number")?;
        if !a["source"].is_object() {
            return Err(err("attempt source mapping must be an object"));
        }
        strict_integer(&a["source"]["record_index"], "source record index")?;
        strict_integer(&a["source"]["attempt_index"], "source attempt index")?;
        if !a["usage"].is_null() {
            if !a["usage"].is_object() {
                return Err(err("attempt usage must be an object or null"));
            }
            for k in ["input_tokens", "output_tokens"] {
                strict_integer(&a["usage"][k], &format!("usage {k}"))?
            }
        }
    }
    Ok(())
}
fn blob(v: &Value, label: &str) -> Result<(), AnalysisError> {
    let bytes = text(v, "text").as_bytes();
    if bytes.len() > REQUEST_LIMIT {
        return Err(err(format!(
            "{label} exceeds the {REQUEST_LIMIT}-byte limit"
        )));
    }
    if bare_digest(bytes) != v["sha256"] {
        return Err(err(format!("{label} digest does not match its UTF-8 text")));
    }
    Ok(())
}
fn attempts(
    trial: &Value,
    plan: &Value,
    sources: &BTreeSet<&str>,
    request: &Value,
) -> Result<(), AnalysisError> {
    let id = python_repr(&trial["trial_id"]);
    let aa = trial["attempts"].as_array().unwrap();
    if aa
        .iter()
        .enumerate()
        .any(|(i, a)| a["attempt"].as_u64() != Some(i as u64 + 1))
    {
        return Err(err(format!(
            "trial {id} attempts must be contiguous from one"
        )));
    }
    if aa.len() as u64 > plan["schedule"]["max_attempts"].as_u64().unwrap() {
        return Err(err(format!(
            "trial {id} exceeds its approved attempt limit"
        )));
    }
    let mut selected = None;
    let mut terminal = false;
    for (i, a) in aa.iter().enumerate() {
        if terminal {
            return Err(err(format!(
                "trial {id} retained an attempt after a terminal result"
            )));
        }
        blob(&a["request"], "judge request")?;
        if &a["request"]["sha256"] != request {
            return Err(err(format!(
                "trial {id} request was not approved by the plan"
            )));
        }
        if !a["response"].is_null() {
            blob(&a["response"], "judge response")?
        }
        if !sources.contains(text(&a["source"], "source_id")) {
            return Err(err(format!("trial {id} references an unknown source")));
        }
        if a["source"]["attempt_index"].as_u64() != Some(i as u64) {
            return Err(err(format!(
                "trial {id} source attempt index is inconsistent"
            )));
        }
        if a["cache"] != "none" {
            return Err(err(
                "the fixed-answer judge profile forbids reused cached responses",
            ));
        }
        match text(a, "status") {
            "completed" => {
                selected = Some(i + 1);
                if a["response"].is_null() || !a["error"].is_null() {
                    return Err(err(
                        "completed judge attempts require a response and no error",
                    ));
                }
                if !plan["judge"]["approved_resolved_models"]
                    .as_array()
                    .unwrap()
                    .contains(&a["resolved_model"])
                {
                    return Err(err(
                        "completed judge attempt used an unapproved resolved model",
                    ));
                }
                terminal = true
            }
            "transport_error" => {
                if a["error"].is_null() {
                    return Err(err("transport errors require retained error information"));
                }
                if !a["response"].is_null() {
                    return Err(err("transport errors must not retain a completed response"));
                }
                if i + 1 < aa.len()
                    && !plan["schedule"]["retry_on"]
                        .as_array()
                        .unwrap()
                        .contains(&json!("transport_error"))
                {
                    return Err(err("transport retry was not approved by the plan"));
                }
            }
            status => {
                if ["timeout_ambiguous", "cancelled"].contains(&status) && a["error"].is_null() {
                    return Err(err(format!(
                        "{status} attempts require retained error information"
                    )));
                }
                terminal = true
            }
        }
    }
    if trial["selected_attempt"] != json!(selected) {
        return Err(err(format!(
            "trial {id} did not select the first completed response"
        )));
    }
    let parsed = if let Some(selected) = selected {
        match parse_json(
            text(&aa[selected - 1]["response"], "text").as_bytes(),
            "judge response",
        ) {
            Ok(decoded)
                if decoded
                    .as_object()
                    .is_some_and(|d| d.len() == 1 && d.contains_key("rating"))
                    && decoded["rating"].is_string() =>
            {
                let rating = plan["scale"]["ratings"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["label"] == decoded["rating"]);
                if let Some(r) = rating {
                    json!({"status":"ok","rating":r["label"],"value":r["value"]})
                } else {
                    json!({"status":"invalid","rating":null,"value":null})
                }
            }
            _ => json!({"status":"invalid","rating":null,"value":null}),
        }
    } else {
        json!({"status":if aa.last().is_some_and(|a|a["status"]=="refusal"){"refusal"}else{"unavailable"},"rating":null,"value":null})
    };
    if trial["parse"] != parsed {
        return Err(err(
            "parsed judge result does not match deterministic response replay",
        ));
    }
    if (trial["status"] == "complete") != (parsed["status"] == "ok") {
        return Err(err("trial completeness must agree with its parse outcome"));
    }
    Ok(())
}
fn frozen(plan: &Value, baseline: &Value, subject: &Value) -> Result<(), AnalysisError> {
    let mut all = BTreeMap::new();
    for (side, run) in [("baseline", baseline), ("subject", subject)] {
        validate_run(run).map_err(|e| {
            err(format!(
                "frozen answer run is invalid: {}",
                e.message.chars().take(240).collect::<String>()
            ))
        })?;
        if sha256_digest(&canonical_json(run).map_err(|e| err(e.to_string()))?)
            != plan[format!("{side}_run_sha256")]
        {
            return Err(err(format!(
                "{side} run does not match the approved plan digest"
            )));
        }
        let rows = run["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (text(r, "id"), r))
            .collect::<BTreeMap<_, _>>();
        let cases = json!({"format":"invarlock/evaluation-case-set-v1","cases":rows.values().map(|r|json!({"id":r["id"],"input":r["input"],"expected":r["expected"],"metadata":r["metadata"]})).collect::<Vec<_>>()});
        if sha256_digest(&canonical_json(&cases).map_err(|e| err(e.to_string()))?)
            != plan["case_set_sha256"]
        {
            return Err(err(format!(
                "frozen answer run is invalid: run {} does not match the planned case set",
                python_repr(&run["run_id"])
            )));
        }
        all.insert(side, rows);
    }
    let bindings = plan["answer_bindings"].as_array().unwrap();
    let ids = bindings
        .iter()
        .map(|b| text(b, "case_id"))
        .collect::<BTreeSet<_>>();
    if all
        .values()
        .any(|rows| rows.keys().copied().collect::<BTreeSet<_>>() != ids)
    {
        return Err(err(
            "frozen run membership must exactly match the judging plan",
        ));
    }
    for binding in bindings {
        let id = text(binding, "case_id");
        let shown = python_repr(&binding["case_id"]);
        let left = all["baseline"][id];
        let right = all["subject"][id];
        let lb = left["context"].get("input_projection");
        let rb = right["context"].get("input_projection");
        if lb.is_some() || rb.is_some() {
            match (lb, rb) {
                (Some(lb), Some(rb))
                    if lb["configuration_digest"] == rb["configuration_digest"]
                        && digest(&lb["source"]["input"])? == digest(&rb["source"]["input"])? => {}
                _ => return Err(err("paired input projection or original input changed")),
            }
        }
        for side in ["baseline", "subject"] {
            let r = all[side][id];
            if !r["error"].is_null() {
                return Err(err(format!(
                    "case {shown} cannot grade a failed frozen answer"
                )));
            }
            if !r["input"].is_string() || !r["output"].is_string() {
                return Err(err(
                    "the text-frozen-answer profile requires string inputs and answers",
                ));
            }
            if binding[format!("{side}_answer_sha256")] != bare_digest(text(r, "output").as_bytes())
            {
                return Err(err(format!(
                    "case {shown} does not bind the frozen {side} answer"
                )));
            }
            let request = render_judge_request(plan, &r["input"], &r["output"], &r["expected"])?;
            if binding[format!("{side}_request_sha256")] != bare_digest(&request) {
                return Err(err(format!(
                    "case {shown} request binding does not match frozen inputs"
                )));
            }
        }
    }
    Ok(())
}
pub fn validate_measurements(
    data: &Value,
    plan: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<(), AnalysisError> {
    validate_measurements_inner(data, plan, baseline, subject, true)
}
fn validate_measurements_inner(
    data: &Value,
    plan: &Value,
    baseline: &Value,
    subject: &Value,
    require_complete: bool,
) -> Result<(), AnalysisError> {
    validate_measurement_plan(plan)?;
    if data["sources"].as_array().is_some_and(|a| a.len() > 1000) {
        return Err(err("judge measurements exceed the source-count limit"));
    }
    if data["trials"].as_array().is_some_and(|a| a.len() > 200000) {
        return Err(err("judge measurements exceed the trial-count limit"));
    }
    if data["trials"].as_array().is_some_and(|a| {
        a.iter()
            .any(|t| t["attempts"].as_array().is_some_and(|a| a.len() > 3))
    }) {
        return Err(err("judge measurements exceed the per-trial attempt limit"));
    }
    bounded(data, 384 * 1024 * 1024, "judge measurements")?;
    // The publication schema requires at least two trial slots. A prefix can
    // contain its first single slot; validate the same item contract twice only
    // for this structural check, then perform all semantic checks on the real data.
    if !require_complete && data["trials"].as_array().is_some_and(|v| v.len() == 1) {
        let mut shape = data.clone();
        shape["trials"]
            .as_array_mut()
            .unwrap()
            .push(data["trials"][0].clone());
        schema(&shape, "judge_measurements")?;
    } else {
        schema(data, "judge_measurements")?;
    }
    for f in ["expected_trials", "recorded_trials", "completed_trials"] {
        strict_integer(&data["completeness"][f], &format!("completeness {f}"))?
    }
    let source_values = data["sources"].as_array().unwrap();
    let trials = data["trials"].as_array().unwrap();
    for source in source_values {
        strict_integer(&source["byte_size"], "source byte_size")?
    }
    for t in trials {
        trial_integers(t)?
    }
    if source_values
        .iter()
        .any(|s| s["profile"] != data["source_profile"])
    {
        return Err(err(
            "measurement source profile differs from the declared profile",
        ));
    }
    let plan_digest = digest(plan)?;
    if data["plan_sha256"] != plan_digest {
        return Err(err("measurements do not bind the supplied plan"));
    }
    frozen(plan, baseline, subject)?;
    let mut sources = BTreeSet::new();
    let mut replayed = BTreeMap::new();
    let mut inspect_sources = Vec::new();
    let mut compatible_sources = Vec::new();
    let mut runtime_sources = Vec::new();
    for source in source_values {
        let id = text(source, "source_id");
        let shown = python_repr(&source["source_id"]);
        if !sources.insert(id) {
            return Err(err("measurement source IDs must be unique"));
        }
        let content = text(source, "content").as_bytes();
        if source["byte_size"].as_u64() != Some(content.len() as u64) {
            return Err(err(format!(
                "source {shown} byte_size does not match content"
            )));
        }
        if source["sha256"] != bare_digest(content) {
            return Err(err(format!("source {shown} digest does not match content")));
        }
        let decoded =
            parse_json(content, &format!("source {shown}")).map_err(|e| err(e.to_string()))?;
        if bytes(&decoded)? != content {
            return Err(err(format!(
                "source {shown} content must use canonical JSON"
            )));
        }
        if !decoded.is_object() {
            return Err(err(format!(
                "source {shown} has an unsupported retained shape"
            )));
        }
        let profile = text(source, "profile");
        let source_trials = if profile == "retained-inspect-model-events-v1" {
            let trials = crate::judge_inspect::replay(&decoded)?;
            inspect_sources.push(decoded.clone());
            trials
        } else if profile == "retained-openai-compatible-judge-v1" {
            let trials = crate::judge_compatible::replay(&decoded, plan)?;
            compatible_sources.push(decoded.clone());
            trials
        } else if profile == "retained-runtime-provider-judge-v1" {
            let trials = crate::judge_runtime::replay(&decoded, plan)?;
            runtime_sources.push(decoded.clone());
            trials
        } else {
            if profile != "retained-judge-json-v1" {
                return Err(AnalysisError{reason:"unsupported_judge_source_profile",message:format!("native judge replay does not yet support retained source profile '{profile}'")});
            }
            if !decoded.as_object().is_some_and(|d| {
                d.len() == 2 && d.contains_key("format") && d.contains_key("trials")
            }) || decoded["format"] != "invarlock/retained-judge-json-v1"
                || !decoded["trials"].is_array()
            {
                return Err(err(format!(
                    "source {shown} has an unsupported retained profile"
                )));
            }
            decoded["trials"].as_array().unwrap().clone()
        };
        for (record_index, trial) in source_trials.iter().enumerate() {
            if !trial.is_object() || !trial["attempts"].is_array() {
                return Err(err(
                    "retained source trials must be objects with attempt arrays",
                ));
            }
            if trial["attempts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| !a.is_object())
            {
                return Err(err("retained source attempts must be objects"));
            }
            trial_integers(trial)?;
            let trial_id = trial["trial_id"]
                .as_str()
                .ok_or_else(|| err("retained sources must contain unique trial IDs"))?;
            if replayed.contains_key(trial_id) {
                return Err(err("retained sources must contain unique trial IDs"));
            }
            for (attempt_index, a) in trial["attempts"].as_array().unwrap().iter().enumerate() {
                let mapping = &a["source"];
                if mapping["source_id"] != id
                    || mapping["record_index"].as_u64() != Some(record_index as u64)
                    || mapping["attempt_index"].as_u64() != Some(attempt_index as u64)
                {
                    return Err(err("retained source position mapping is inconsistent"));
                }
            }
            replayed.insert(trial_id.to_owned(), trial.clone());
        }
    }
    crate::judge_runtime::family(&runtime_sources, plan)?;
    crate::judge_compatible::family(&compatible_sources, plan)?;
    crate::judge_inspect::family(
        &inspect_sources,
        plan,
        replayed
            .values()
            .map(|t| t["attempts"].as_array().unwrap().len())
            .sum(),
    )?;
    let bindings = plan["answer_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| (text(b, "case_id"), b))
        .collect::<BTreeMap<_, _>>();
    let reps = plan["schedule"]["repetitions"].as_u64().unwrap();
    let expected = bindings
        .keys()
        .flat_map(|id| {
            ["baseline", "subject"]
                .into_iter()
                .flat_map(move |side| (1..=reps).map(move |r| (*id, side, r)))
        })
        .collect::<BTreeSet<_>>();
    let mut slots = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut positions = BTreeSet::new();
    let mut events = BTreeSet::new();
    let mut response_ids = BTreeSet::new();
    let mut completed = 0;
    for trial in trials {
        let case = text(trial, "case_id");
        let side = text(trial, "side");
        let repetition = trial["repetition"].as_u64().unwrap();
        let slot = (case, side, repetition);
        if !expected.contains(&slot) || !slots.insert(slot) {
            return Err(err(
                "measurements contain an unknown or duplicate planned trial slot",
            ));
        }
        let id = expected_judge_trial_id(&plan_digest, case, side, repetition)?;
        if trial["trial_id"] != id || !ids.insert(id.clone()) {
            return Err(err("measurement trial ID does not match its planned slot"));
        }
        if trial["plan_sha256"] != plan_digest {
            return Err(err(format!("trial '{id}' does not bind the supplied plan")));
        }
        let binding = bindings[case];
        if trial["answer_sha256"] != binding[format!("{side}_answer_sha256")] {
            return Err(err(format!("trial '{id}' does not bind the frozen answer")));
        }
        attempts(
            trial,
            plan,
            &sources,
            &binding[format!("{side}_request_sha256")],
        )?;
        for a in trial["attempts"].as_array().unwrap() {
            let mapping = &a["source"];
            if !positions.insert((
                text(mapping, "source_id"),
                mapping["record_index"].as_u64().unwrap(),
                mapping["attempt_index"].as_u64().unwrap(),
            )) {
                return Err(err("source record positions must be unique across trials"));
            }
            if !events.insert((
                if inspect_sources.is_empty() {
                    text(mapping, "source_id")
                } else {
                    text(data, "source_profile")
                },
                text(mapping, "model_event_id"),
            )) {
                return Err(err(
                    "retained model events must belong to exactly one attempt",
                ));
            }
            if (!inspect_sources.is_empty() || !compatible_sources.is_empty())
                && !a["request_id"].is_null()
                && !response_ids.insert(a["request_id"].clone().to_string())
            {
                return Err(err("retained provider response IDs must be unique"));
            }
        }
        if trial["status"] == "complete" {
            completed += 1
        }
        if replayed.get(&id) != Some(trial) {
            return Err(err(format!(
                "retained source replay differs for trial '{id}'"
            )));
        }
    }
    if replayed.keys().cloned().collect::<BTreeSet<_>>() != ids {
        return Err(err(
            "retained source replay contains omitted or extra trials",
        ));
    }
    if require_complete && slots != expected {
        return Err(err("measurements omit one or more planned trial slots"));
    }
    let completeness = &data["completeness"];
    for (field, count, label) in [
        ("expected_trials", expected.len(), "expected-trial"),
        ("recorded_trials", trials.len(), "recorded-trial"),
        ("completed_trials", completed, "completed-trial"),
    ] {
        if completeness[field].as_u64() != Some(count as u64) {
            return Err(err(format!("measurement {label} count is inconsistent")));
        }
    }
    if completeness["status"]
        != if completed == expected.len() {
            "complete"
        } else {
            "incomplete"
        }
    {
        return Err(err("measurement completeness status is inconsistent"));
    }
    Ok(())
}
pub(crate) fn validate_frozen_answers(
    plan: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<(), AnalysisError> {
    frozen(plan, baseline, subject)
}
pub(crate) fn validate_imported_trial(trial: &Value, plan: &Value) -> Result<(), AnalysisError> {
    if let Some(violation) = invarlock_native_contracts::first_schema_error_fragment(
        trial,
        "judge_measurements",
        "/properties/trials/items",
        SchemaErrorOrder::Schema,
    )
    .map_err(|e| err(e.to_string()))?
    {
        let location = if violation.path.is_empty() {
            String::new()
        } else {
            format!(" at {}", violation.path.join("/"))
        };
        return Err(err(format!(
            "retained judge trial is invalid{location}: {}",
            violation.message.chars().take(240).collect::<String>()
        )));
    }
    trial_integers(trial)?;
    let binding = plan["answer_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["case_id"] == trial["case_id"])
        .ok_or_else(|| err("unknown sample binding"))?;
    attempts(
        trial,
        plan,
        &BTreeSet::from(["inspect-export"]),
        &binding[format!("{}_request_sha256", text(trial, "side"))],
    )
}

/// Validate a retained collection prefix with every source and trial check except
/// the requirement that all planned slots have already been collected.
/// Sources use the normalized measurements source objects, not decoded payloads.
pub fn validate_retained_collection_prefix(
    sources: &[Value],
    plan: &Value,
    baseline: &Value,
    subject: &Value,
) -> Result<(), AnalysisError> {
    validate_measurement_plan(plan)?;
    frozen(plan, baseline, subject)?;
    if sources.is_empty() {
        return Ok(());
    }
    if sources.len() > 1000 {
        return Err(err("judge measurements exceed the source-count limit"));
    }
    let mut trials = Vec::new();
    for source in sources {
        if let Some(violation) = invarlock_native_contracts::first_schema_error_fragment(
            source,
            "judge_measurements",
            "/properties/sources/items",
            SchemaErrorOrder::Schema,
        )
        .map_err(|e| err(e.to_string()))?
        {
            return Err(err(format!(
                "retained measurement source is invalid: {}",
                violation.message
            )));
        }
        let content = text(source, "content").as_bytes();
        let decoded =
            parse_json(content, "retained measurement source").map_err(|e| err(e.to_string()))?;
        let mut rows = match text(source, "profile") {
            "retained-openai-compatible-judge-v1" => {
                crate::judge_compatible::replay(&decoded, plan)?
            }
            "retained-runtime-provider-judge-v1" => crate::judge_runtime::replay(&decoded, plan)?,
            _ => {
                return Err(err(
                    "collection prefix requires a compatible or runtime provider source",
                ))
            }
        };
        trials.append(&mut rows);
        if trials.len() > 200000 {
            return Err(err("judge measurements exceed the trial-count limit"));
        }
    }
    let expected = plan["answer_bindings"].as_array().unwrap().len() as u64
        * 2
        * plan["schedule"]["repetitions"].as_u64().unwrap();
    let completed = trials.iter().filter(|t| t["status"] == "complete").count() as u64;
    let data = json!({"format":"invarlock/judge-measurements-v1","profile_id":"text-frozen-answer-v1","plan_sha256":digest(plan)?,"source_profile":sources[0]["profile"],"sources":sources,"trials":trials,"completeness":{"expected_trials":expected,"recorded_trials":trials.len(),"completed_trials":completed,"status":if completed==expected{"complete"}else{"incomplete"}}});
    validate_measurements_inner(&data, plan, baseline, subject, false)
}
