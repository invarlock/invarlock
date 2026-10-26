//! Replay original native judge captures without creating temporary files.
use crate::native_pack::replay;
use crate::{canonical, digest, EvidenceError, Result};
use invarlock_native_contracts::{
    canonical_json_no_newline, first_schema_error_value, packaged_schema, parse_json,
    sha256_digest, validate_schema, SchemaErrorOrder,
};
use invarlock_native_core as analysis;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

fn fail(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}
fn check(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(fail(message))
    }
}
fn bare(raw: &[u8]) -> String {
    sha256_digest(raw)[7..].to_string()
}
fn exact(value: &Value, fields: &[&str], label: &str) -> Result<()> {
    check(
        value
            .as_object()
            .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|f| m.contains_key(*f))),
        format!(
            "{label} must contain exactly: {}",
            fields
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}
fn remove_property(schema: &mut Value, name: &str) {
    if let Some(properties) = schema["properties"].as_object_mut() {
        properties.remove(name);
    }
    if let Some(required) = schema["required"].as_array_mut() {
        required.retain(|n| n != name);
    }
}
pub fn validate_recipe(value: &Value) -> Result<()> {
    check(
        canonical_json_no_newline(value)?.len() <= 4 * 1024 * 1024,
        "native judge policy exceeds its 4 MiB limit",
    )?;
    exact(
        value,
        &["format", "plan", "analysis", "collection", "runner"],
        "native judge policy",
    )?;
    check(
        value["format"] == "invarlock/native-judge-policy-v1",
        "unsupported native judge policy format",
    )?;
    let plan = &value["plan"];
    exact(
        plan,
        &[
            "format",
            "profile_id",
            "rubric",
            "prompt",
            "judge",
            "parser",
            "scale",
            "sampling",
            "schedule",
        ],
        "native judge plan template",
    )?;
    exact(&plan["rubric"], &["text"], "native judge rubric")?;
    let mut plan_schema = packaged_schema("judge_measurement_plan")?;
    for name in [
        "case_set_sha256",
        "baseline_run_sha256",
        "subject_run_sha256",
        "answer_bindings",
    ] {
        remove_property(&mut plan_schema, name);
    }
    for (parent, name) in [("rubric", "sha256"), ("schedule", "expected_trials")] {
        remove_property(&mut plan_schema["properties"][parent], name);
    }
    let mut policy_schema = packaged_schema("judge_analysis_policy")?;
    remove_property(&mut policy_schema, "plan_sha256");
    for (schema, item, label) in [
        (&plan_schema, plan, "native judge plan template"),
        (
            &policy_schema,
            &value["analysis"],
            "native judge analysis template",
        ),
    ] {
        if let Some(error) = first_schema_error_value(item, schema, SchemaErrorOrder::Schema)? {
            return Err(fail(format!(
                "{label} is invalid: {}",
                crate::bounded(&error.message)
            )));
        }
    }
    let managed = matches!(
        value["collection"]["profile"].as_str(),
        Some("runtime-provider-text-frozen-answer-v1" | "openai-compatible-text-frozen-answer-v1")
    );
    exact(
        &value["runner"],
        if managed {
            &["scorer_id"]
        } else {
            &["scorer_id", "invocation_timeout_seconds"]
        },
        "judge runner",
    )?;
    let scorer = value["runner"]["scorer_id"].as_str().unwrap_or("");
    check(
        !scorer.is_empty()
            && scorer.len() <= 128
            && scorer
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
        "judge scorer_id must be a bounded identifier",
    )?;
    if !managed {
        check(
            value["runner"]["invocation_timeout_seconds"]
                .as_u64()
                .is_some_and(|n| (1..=604800).contains(&n)),
            "judge invocation timeout must be between 1 and 604800 seconds",
        )?;
    }
    Ok(())
}
pub fn finalize_plan(recipe: &Value, baseline: &Value, subject: &Value) -> Result<(Value, Value)> {
    validate_recipe(recipe)?;
    analysis::validate_run(baseline).map_err(|e| fail(e.to_string()))?;
    analysis::validate_run(subject).map_err(|e| fail(e.to_string()))?;
    let mut plan = recipe["plan"].clone();
    plan["baseline_run_sha256"] = json!(digest(baseline)?);
    plan["subject_run_sha256"] = json!(digest(subject)?);
    let baseline_rows = baseline["records"]
        .as_array()
        .ok_or_else(|| fail("baseline records must be an array"))?;
    let cases:Vec<_>=baseline_rows.iter().map(|r|json!({"id":r["id"],"input":r["input"],"expected":r["expected"],"metadata":r["metadata"]})).collect();
    plan["case_set_sha256"] = json!(crate::judge::case_set_digest(
        &json!({"format":"invarlock/evaluation-case-set-v1","cases":cases})
    )?);
    plan["rubric"]["sha256"] = json!(bare(
        plan["rubric"]["text"].as_str().unwrap_or("").as_bytes()
    ));
    let repetitions = plan["schedule"]["repetitions"]
        .as_u64()
        .ok_or_else(|| fail("judge repetitions must use an integer value"))?;
    plan["schedule"]["expected_trials"] = json!((baseline_rows.len() as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(repetitions))
        .ok_or_else(|| fail("judge expected trial count exceeds its limit"))?);
    let baseline_map: BTreeMap<_, _> = baseline_rows
        .iter()
        .map(|r| (r["id"].as_str().unwrap_or(""), r))
        .collect();
    let subject_map: BTreeMap<_, _> = subject["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["id"].as_str().unwrap_or(""), r))
        .collect();
    check(
        baseline_map.keys().eq(subject_map.keys()),
        "native judge runs must contain identical case membership",
    )?;
    let mut bindings = vec![];
    for case in baseline_map.keys() {
        let mut binding = json!({"case_id":case});
        for (side, rows) in [("baseline", &baseline_map), ("subject", &subject_map)] {
            let row = rows[case];
            check(
                row["output"].is_string() && row["error"].is_null(),
                "native judging requires complete successful text answers",
            )?;
            let request = analysis::render_judge_request(
                &plan,
                &row["input"],
                &row["output"],
                &row["expected"],
            )
            .map_err(|e| fail(e.to_string()))?;
            binding[format!("{side}_answer_sha256")] =
                json!(bare(row["output"].as_str().unwrap().as_bytes()));
            binding[format!("{side}_request_sha256")] = json!(bare(&request));
        }
        bindings.push(binding);
    }
    plan["answer_bindings"] = Value::Array(bindings);
    analysis::validate_measurement_plan(&plan).map_err(|e| fail(e.to_string()))?;
    let mut policy = recipe["analysis"].clone();
    policy["plan_sha256"] =
        json!(analysis::measurement_plan_digest(&plan).map_err(|e| fail(e.to_string()))?);
    analysis::validate_judge_analysis_policy(&policy, &plan).map_err(|e| fail(e.to_string()))?;
    Ok((plan, policy))
}
fn decode(value: &Value, max: usize, label: &str) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .filter(|text| text.len() <= 4 * max.div_ceil(3))
        .ok_or_else(|| fail(format!("native capture {label} must be bounded base64")))?;
    let bytes = crate::decode_base64(text)
        .map_err(|_| fail(format!("native capture {label} is not valid base64")))?;
    check(
        !bytes.is_empty() && bytes.len() <= max,
        format!("native capture {label} exceeds its byte limit"),
    )?;
    Ok(bytes)
}
const SIDE_FIELDS: [(&str, &str); 6] = [
    ("run_report", "run_report"),
    ("runtime_manifest", "runtime_manifest"),
    ("runtime_config", "runtime_config"),
    ("artifact_identity", "provider_identity"),
    ("provider_receipt", "provider_receipt"),
    ("scoring_observation", "scoring_observation"),
];
pub fn validate_capture(capture: &Value) -> Result<(Value, Value)> {
    let fields = [
        "format",
        "normalized_request",
        "schedule",
        "policy_base64",
        "recipe",
        "baseline",
        "subject",
        "observations",
    ];
    check(
        capture
            .as_object()
            .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|f| m.contains_key(*f))),
        "native capture must contain exactly the defined fields",
    )?;
    check(
        capture["format"] == "invarlock/native-judge-capture-v1",
        "native capture format is unsupported",
    )?;
    let capture_bytes = canonical_json_no_newline(capture)?;
    check(
        capture_bytes.len() <= 128 * 1024 * 1024,
        "native capture exceeds its aggregate byte limit",
    )?;
    let request = &capture["normalized_request"];
    check(
        request.is_object(),
        "native capture normalized request must be an object",
    )?;
    if let Some(error) = invarlock_native_contracts::first_schema_error(
        request,
        "evaluation_request",
        SchemaErrorOrder::Instance,
    )? {
        let path = if error.path.is_empty() {
            "<root>".to_string()
        } else {
            error.path.join(".")
        };
        return Err(fail(format!(
            "normalized request schema failed at {path}: {}",
            error.message
        )));
    }
    let schedule = &capture["schedule"];
    let schedule_bytes = canonical_json_no_newline(schedule)?;
    replay::schedule(schedule, &schedule_bytes, request)?;
    let records = schedule["records"]
        .as_array()
        .ok_or_else(|| fail("native capture schedule records must be an array"))?;
    check(
        records.iter().all(|r| {
            r["input_parts"]
                .as_array()
                .is_some_and(|parts| parts.len() == 1 && parts[0]["kind"] == "text")
        }),
        "native judge capture requires exactly one text input part",
    )?;
    check(
        request["comparison"]["metric"] == "judge",
        "native judge capture requires a judge metric request",
    )?;
    check(
        request["comparison"]["policy"] == "inputs/policy.json",
        "native capture request must bind the canonical policy",
    )?;
    let policy = decode(&capture["policy_base64"], 4 * 1024 * 1024, "policy")?;
    let recipe = parse_json(&policy, "native capture original policy")?;
    check(
        recipe == capture["recipe"],
        "native capture recipe differs from its original policy",
    )?;
    prepare(&policy, schedule)?;
    let observations = capture["observations"]
        .as_array()
        .filter(|v| v.len() <= 64)
        .ok_or_else(|| fail("native capture observations must be a bounded array"))?;
    let mut observation_bindings = vec![];
    let mut ids = BTreeSet::new();
    let mut last = "";
    for observation in observations {
        check(
            observation.as_object().is_some_and(|v| {
                v.len() == 4
                    && ["id", "scope", "kind", "payload_base64"]
                        .iter()
                        .all(|k| v.contains_key(*k))
            }),
            "native capture observation fields are invalid",
        )?;
        let id = observation["id"].as_str().unwrap_or("");
        check(
            !id.is_empty()
                && id.len() <= 128
                && id.as_bytes()[0].is_ascii_alphanumeric()
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
            "observation_id is invalid",
        )?;
        check(
            ids.insert(id) && last <= id,
            "native capture observations differ from the normalized request",
        )?;
        last = id;
        let kind = observation["kind"].as_str().unwrap_or("");
        check(
            !kind.is_empty()
                && kind.as_bytes()[0].is_ascii_lowercase()
                && kind.split(['.', '_', ':', '-']).all(|part| {
                    !part.is_empty()
                        && part
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
                }),
            format!("observation {id:?} kind is invalid"),
        )?;
        check(
            matches!(
                observation["scope"].as_str(),
                Some("baseline" | "subject" | "comparison")
            ),
            format!("observation {id:?} scope is invalid"),
        )?;
        let raw = decode(&observation["payload_base64"], 1024 * 1024, "observation")?;
        let payload = parse_json(&raw, "observation")?;
        check(
            payload.is_object() && canonical(&payload)? == raw,
            format!("observation {id:?} payload must be canonical JSON"),
        )?;
        observation_bindings.push(json!({"id":id,"scope":observation["scope"],"kind":kind,"payload_digest":sha256_digest(&raw)}));
    }
    check(
        request.get("observations").unwrap_or(&json!([])) == &Value::Array(observation_bindings),
        "native capture observations differ from the normalized request",
    )?;
    let mut storage = BTreeMap::new();
    let mut runtime = Map::new();
    let mut identities = BTreeMap::new();
    let mut observations = BTreeMap::new();
    for side in ["baseline", "subject"] {
        let encoded = &capture[side];
        check(
            encoded.as_object().is_some_and(|v| {
                v.len() == 6 && SIDE_FIELDS.iter().all(|(k, _)| v.contains_key(*k))
            }),
            format!("native capture {side} must retain all six original files"),
        )?;
        for (field, role) in SIDE_FIELDS {
            storage.insert(
                format!("{side}_{role}"),
                decode(
                    &encoded[field],
                    64 * 1024 * 1024,
                    &format!("{side} {field}"),
                )?,
            );
        }
        let receipt = parse_json(
            &storage[&format!("{side}_provider_receipt")],
            "provider receipt",
        )?;
        validate_schema(&receipt, "runtime_provider_receipt")?;
        check(
            receipt["outer_image_digest"].is_string(),
            format!("native capture {side} lacks strict runtime identity"),
        )?;
        runtime.insert(side.to_string(), receipt["outer_image_digest"].clone());
        let raw = &storage[&format!("{side}_provider_identity")];
        let artifact = parse_json(raw, "model artifact identity")?;
        invarlock_native_contracts::runtime::validate_artifact(&artifact)?;
        let model = ["model_id", "artifact_name", "bundle_name"]
            .iter()
            .find_map(|name| artifact.get(*name))
            .unwrap_or(&Value::Null);
        check(
            request["comparison"][side]["artifact"]["model_id"] == *model,
            format!("native capture {side} model identity differs from request"),
        )?;
        invarlock_native_contracts::runtime::request_bindings(
            &request["comparison"][side],
            &artifact,
            &receipt,
        )
        .map_err(|e| fail(format!("native capture {side}: {e}")))?;
        identities.insert(side,json!({"digest":sha256_digest(raw),"locator":request["comparison"][side]["artifact"]["locator"]}));
        identities.insert(if side=="baseline"{"baseline_runtime"}else{"subject_runtime"},json!({"digest":runtime[side],"locator":format!("runtime:{}",runtime[side].as_str().unwrap())}));
        observations.insert(
            side,
            parse_json(
                &storage[&format!("{side}_scoring_observation")],
                "scoring observation",
            )?,
        );
    }
    identities.insert("dataset",json!({"digest":sha256_digest(&schedule_bytes),"locator":"schedule/runtime-behavioral-schedule.json"}));
    identities.insert(
        "policy",
        json!({"digest":sha256_digest(&policy),"locator":"inputs/policy.json"}),
    );
    let loaded: BTreeMap<&str, &[u8]> = storage
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let anchors = json!({"runtime_digests":runtime,"policy_digest":sha256_digest(&policy),"request_digest":digest(request)?});
    let mut scoring_request = request.clone();
    scoring_request["comparison"]["metric"] = json!("exact_match");
    let runtime_errors = replay::runtime_checks(&loaded, &anchors);
    if let Some(error) = runtime_errors.first() {
        return Err(fail(error.clone()));
    }
    replay::derive(
        &loaded,
        &scoring_request,
        schedule,
        &identities,
        &anchors,
        None,
    )?;
    let capture_digest = sha256_digest(&capture_bytes);
    let mut runs = vec![];
    for side in ["baseline", "subject"] {
        let observed = observations[side]["records"]
            .as_array()
            .ok_or_else(|| fail("scoring observation records must be an array"))?;
        let rows:Vec<_>=records.iter().zip(observed).map(|(scheduled,row)|json!({
            "id":scheduled["record_id"],"input":scheduled["input_parts"][0]["text"],"expected":scheduled["expected_output"],
            "output":row["output_text"],"scores":{},"metadata":{},"error":null,
            "context":{"model_id":request["comparison"][side]["artifact"]["model_id"],"provider":observations[side]["provider_name"],
                "runtime_digest":runtime[side],"schedule_sha256":bare(&schedule_bytes),"input_sha256":scheduled["input_sha256"],"output_sha256":row["output_sha256"]}})).collect();
        let run = json!({"format":"invarlock/evaluation-run-v1","source":{"name":"invarlock-native-judge","version":"1"},
            "run_id":format!("{side}-{}",capture_digest.trim_start_matches("sha256:")),"artifact_digest":identities[side]["digest"],
            "source_digest":capture_digest,"score_provenance":{},"records":rows});
        analysis::validate_run(&run).map_err(|e| fail(e.to_string()))?;
        runs.push(run);
    }
    let subject = runs.pop().unwrap();
    let baseline = runs.pop().unwrap();
    Ok((baseline, subject))
}

fn validate_collection(collection: &Value, plan: &Value) -> Result<Value> {
    match collection["profile"].as_str() {
        Some("runtime-provider-text-frozen-answer-v1") => {
            invarlock_native_contracts::runtime::validate_runtime_provider_collection(
                collection, plan,
            )
            .map_err(Into::into)
        }
        Some("openai-compatible-text-frozen-answer-v1") => {
            analysis::validate_openai_compatible_collection(collection, plan)
                .map_err(|e| fail(e.to_string()))
        }
        _ => analysis::validate_inspect_collection_budgets(collection, plan)
            .map_err(|e| fail(e.to_string())),
    }
}
/// Reserve a complete schedule before answer generation; no provider loading.
pub fn prepare(policy_bytes: &[u8], schedule: &Value) -> Result<Value> {
    check(
        policy_bytes.len() <= 4 * 1024 * 1024,
        "native judge policy exceeds its 4 MiB limit",
    )?;
    let recipe = parse_json(policy_bytes, "native judge policy")?;
    validate_recipe(&recipe)?;
    let rows = schedule["records"]
        .as_array()
        .ok_or_else(|| fail("schedule records must be an array"))?;
    let mut records = vec![];
    for row in rows {
        check(
            row["input_parts"]
                .as_array()
                .is_some_and(|p| p.len() == 1 && p[0]["kind"] == "text"),
            "native judging requires exactly one text part per scheduled input",
        )?;
        records.push(json!({"id":row["record_id"],"input":row["input_parts"][0]["text"],"expected":row["expected_output"],"metadata":{},"context":{},"output":"","error":null,"scores":{}}));
    }
    let preview = json!({"format":"invarlock/evaluation-run-v1","run_id":"preflight","artifact_digest":format!("sha256:{}","0".repeat(64)),"source":{"name":"invarlock-native-preflight","version":"1"},"source_digest":null,"records":records,"score_provenance":{}});
    let (plan, policy) = finalize_plan(&recipe, &preview, &preview)?;
    let budgets = validate_collection(&recipe["collection"], &plan)?;
    let output = plan["judge"]["config"]["max_output_tokens"]
        .as_u64()
        .unwrap();
    let mut capacity = budgets["max_calls"]
        .as_u64()
        .unwrap()
        .min(budgets["max_output_tokens"].as_u64().unwrap() / output);
    if budgets.get("max_input_tokens").is_some() {
        capacity = capacity.min(
            budgets["max_input_tokens"].as_u64().unwrap()
                / budgets["input_tokens_per_call"].as_u64().unwrap(),
        );
        capacity = capacity.min(
            budgets["max_cost_microusd"].as_u64().unwrap()
                / budgets["cost_microusd_per_call"].as_u64().unwrap(),
        );
    }
    let planned = plan["schedule"]["expected_trials"].as_u64().unwrap();
    check(
        capacity >= planned,
        "judge collection budgets must reserve every planned call",
    )?;
    let units = plan["sampling"]["case_units"]
        .as_array()
        .unwrap()
        .iter()
        .map(|u| u["unit_id"].as_str().unwrap())
        .collect::<BTreeSet<_>>()
        .len();
    check(
        policy["minimum_units"]
            .as_u64()
            .is_some_and(|min| units as u64 >= min),
        "judge schedule has fewer independent units than its analysis minimum",
    )?;
    let precision =
        analysis::judge_plan_precision(&plan, &policy).map_err(|e| fail(e.to_string()))?;
    Ok(
        json!({"recipe":recipe,"preflight_plan":plan,"cases":rows.len(),"independent_units":units,"minimum_units":policy["minimum_units"],"planned_trials":planned,"budgets":budgets,"maximum_admitted_calls":capacity,"judge":plan["judge"],"metric_name":policy["metric_name"],"precision":precision}),
    )
}
