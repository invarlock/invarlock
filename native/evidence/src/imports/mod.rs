//! Offline capture of retained evaluator exports. No SDK discovery or execution.
mod batch;
mod langfuse;
mod scalar;
use crate::{EvidenceError, Result};
use invarlock_native_contracts as contracts;
use invarlock_native_core as core;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
const MAX_RECORDS: usize = 50_000;
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn object<'a>(v: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| err(format!("{label} requires an object")))
}
fn array<'a>(v: &'a Value, label: &str, empty: bool) -> Result<&'a Vec<Value>> {
    v.as_array()
        .filter(|a| a.len() <= MAX_RECORDS && (empty || !a.is_empty()))
        .ok_or_else(|| {
            err(format!(
                "{label} requires a bounded {}array",
                if empty { "" } else { "nonempty " }
            ))
        })
}
fn rows<'a>(v: &'a Value, label: &str) -> Result<&'a Vec<Value>> {
    let a = array(v, label, false)?;
    if a.iter().any(|r| !r.is_object()) {
        return Err(err(format!(
            "{label} requires a non-empty array of record objects"
        )));
    }
    Ok(a)
}
fn required<'a>(v: &'a Value, k: &str) -> Result<&'a Value> {
    v.get(k).ok_or_else(|| {
        err(format!(
            "native case is missing required {}",
            contracts::python_repr(&json!(k))
        ))
    })
}
fn obj_default(v: &Value, label: &str) -> Result<Value> {
    if v.is_null() {
        Ok(json!({}))
    } else {
        object(v, label)?;
        Ok(v.clone())
    }
}
fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(x) => *x,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
        Value::Number(_) => v.as_f64() != Some(0.0),
    }
}
fn number(v: &Value, label: &str, boolean: bool) -> Result<Value> {
    let n = if boolean && v.is_boolean() {
        Some(if v == true { 1.0 } else { 0.0 })
    } else {
        v.as_f64()
    };
    n.filter(|n| n.is_finite())
        .map(|n| json!(n))
        .ok_or_else(|| err(format!("{label} requires a finite numeric score")))
}
fn numeric_equal(a: &Value, b: &Value) -> bool {
    contracts::runtime::json_equal(a, b)
}

fn identifier(v: &Value) -> Result<String> {
    if let Some(s) = v.as_str() {
        Ok(s.into())
    } else if v.is_number() && !v.to_string().contains(['.', 'e', 'E']) {
        Ok(v.to_string())
    } else {
        Err(err("native record IDs must be strings or integers"))
    }
}
fn tags(metadata: &Value) -> Result<Value> {
    let m = obj_default(metadata, "native metadata")?;
    Ok(Value::Object(
        object(&m, "metadata")?
            .iter()
            .filter(|(k, v)| v.is_string() && !k.starts_with("invarlock_"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    ))
}
fn scores(v: &Value, allow_native: bool) -> Result<Value> {
    let m = object(v, "native scores")?;
    let mut result = Map::new();
    for (k, v) in m {
        let mut value = if v.is_object() {
            v["value"].clone()
        } else {
            v.clone()
        };
        if value == "C" {
            value = json!(1.0)
        } else if value == "I" {
            value = json!(0.0)
        }
        if !value.is_number() {
            if allow_native {
                continue;
            }
            return Err(err(format!(
                "score {} is not numeric; provide an explicit SDK mapping",
                contracts::python_repr(&json!(k))
            )));
        }
        if k.is_empty() {
            return Err(err("native scores require nonempty names"));
        }
        result.insert(k.clone(), number(&value, "native scores", false)?);
    }
    Ok(Value::Object(result))
}
fn merge_scores(scores: &Value, metadata: &Value) -> Result<Value> {
    let mut merged = object(scores, "scores")?.clone();
    let empty = json!({});
    let supplied = metadata.get("invarlock_scores").unwrap_or(&empty);
    for (k, v) in object(supplied, "invarlock_scores")? {
        if k.is_empty() || !v.is_number() || v.as_f64().is_some_and(|x| !x.is_finite()) {
            return Err(err(
                "invarlock_scores requires named finite numeric observations",
            ));
        }
        if merged.get(k).is_some_and(|old| !numeric_equal(old, v)) {
            return Err(err(format!(
                "explicit observation {} conflicts with the native score",
                contracts::python_repr(&json!(k))
            )));
        }
        merged.insert(k.clone(), v.clone());
    }
    Ok(Value::Object(merged))
}
fn capture(mut record: Value, row: &Value, metadata: &Value) -> Result<Value> {
    let metadata = obj_default(metadata, "native metadata")?;
    if record.get("context").is_none() {
        record["context"] = json!({});
    }
    record["context"]["upstream_record"] = row.clone();
    record["scores"] = merge_scores(record.get("scores").unwrap_or(&json!({})), &metadata)?;
    if let Some(v) = metadata.get("invarlock_likelihood") {
        record["likelihood"] = v.clone();
    }
    Ok(record)
}
fn task_error(v: &Value) -> Result<Value> {
    if v.is_null() {
        Ok(Value::Null)
    } else if (v.is_string() || v.is_object()) && truth(v) {
        Ok(json!("upstream_error"))
    } else {
        Err(err("native task error must be nonempty text or an object"))
    }
}
fn scalar_output(mut v: &Value) -> Result<Value> {
    while let Some(a) = v.as_array() {
        if a.len() != 1 {
            break;
        }
        v = &a[0];
    }
    if v.is_string() {
        Ok(v.clone())
    } else {
        Err(err(
            "ambiguous or non-text native completion; select one response in your evaluator",
        ))
    }
}
fn inspect(v: &Value, allow_native: bool) -> Result<Vec<Value>> {
    if !v.is_object() || !matches!(v["version"].as_u64(), Some(1 | 2)) {
        return Err(err(
            "inspect-json requires a version 1 or 2 JSON EvalLog (export .eval logs as JSON)",
        ));
    }
    if !["success", "error", "cancelled"].contains(&v["status"].as_str().unwrap_or("")) {
        return Err(err("Inspect log lacks a recognized completion status"));
    }
    if v["status"] != "success" {
        return Err(err(
            "Inspect run did not complete successfully; supply a complete run",
        ));
    }
    let mut result = Vec::new();
    for row in rows(&v["samples"], "Inspect samples")? {
        let output = obj_default(&row["output"], "output")?;
        let empty = json!([]);
        let choices = array(
            output.get("choices").unwrap_or(&empty),
            "Inspect choices",
            true,
        )?;
        let error = task_error(&row["error"])?;
        let metadata = obj_default(&row["metadata"], "metadata")?;
        if choices.len() > 1
            || (choices.is_empty()
                && error.is_null()
                && metadata.get("invarlock_likelihood").is_none())
        {
            return Err(err("Inspect import requires one completion per sample"));
        }
        let mut content = choices
            .first()
            .map(|c| c["message"]["content"].clone())
            .unwrap_or(Value::Null);
        if let Some(parts) = content.as_array() {
            let mut text = String::new();
            for p in parts {
                if p["type"] != "text" {
                    return Err(err(
                        "Inspect completion contains unsupported non-text content",
                    ));
                }
                text.push_str(
                    p["text"]
                        .as_str()
                        .ok_or_else(|| err("Inspect text part requires text"))?,
                );
            }
            content = json!(text);
        }
        let mut target = row["target"].clone();
        if let Some(a) = target.as_array() {
            if a.len() != 1 {
                return Err(err(
                    "multiple Inspect targets require an explicit scoring adapter",
                ));
            }
            target = a[0].clone();
        }
        if row.get("epoch").is_some_and(|v| v.as_u64() != Some(1)) {
            return Err(err(
                "multiple Inspect epochs require an explicit paired trial mapping",
            ));
        }
        result.push(capture(json!({"id":identifier(&row["id"])? ,"input":required(row,"input")?,"expected":target,"output":content,"scores":scores(&obj_default(&row["scores"],"scores")?,allow_native)?,"metadata":tags(&metadata)?,"error":error}),row,&metadata)?);
    }
    Ok(result)
}
fn harness(v: &Value, stable: bool) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for row in rows(v, "Harness samples")? {
        for k in ["doc", "target", "arguments", "filtered_resps"] {
            required(row, k)?;
        }
        let metadata = obj_default(&row["metadata"], "metadata")?;
        let facts = &metadata["invarlock_likelihood"];
        let response = &row["filtered_resps"];
        let error = task_error(&row["error"])?;
        let output = if (!error.is_null()
            && (response.is_null() || response == &json!([]) || response == &json!([null])))
            || (!facts.is_null() && response == &json!([null]))
        {
            Value::Null
        } else if facts.is_object()
            && response
                .as_array()
                .is_some_and(|a| a.len() == 1 && a[0].as_array().is_some_and(|b| b.len() == 2))
        {
            let mut measured = response[0][0].clone();
            let mut greedy = response[0][1].clone();
            let mut arguments = row["arguments"].clone();
            if arguments
                .as_object()
                .is_some_and(|m| m.len() == 1 && m.contains_key("gen_args_0"))
                && arguments["gen_args_0"].as_object().is_some_and(|m| {
                    m.len() == 2 && m.contains_key("arg_0") && m.contains_key("arg_1")
                })
            {
                arguments = json!([[
                    arguments["gen_args_0"]["arg_0"],
                    arguments["gen_args_0"]["arg_1"]
                ]]);
                let supplied = &facts["logprob_sum"];
                if supplied.is_number()
                    && measured.as_str() == Some(contracts::python_repr(supplied).as_str())
                    && (greedy == "True" || greedy == "False")
                {
                    measured = supplied.clone();
                    greedy = json!(greedy == "True");
                }
            }
            if !measured.is_number()
                || !greedy.is_boolean()
                || !numeric_equal(&measured, &facts["logprob_sum"])
                || !arguments.as_array().is_some_and(|a| {
                    a.len() == 1
                        && a[0].as_array().is_some_and(|b| {
                            b.len() == 2 && b[0].is_string() && b[1] == row["target"]
                        })
                })
            {
                return Err(err(
                    "Harness likelihood facts differ from the native result or continuation",
                ));
            }
            Value::Null
        } else {
            scalar_output(response)?
        };
        let score = Value::Object(
            object(row, "row")?
                .iter()
                .filter(|(k, v)| k.as_str() != "doc_id" && v.is_number())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
        let id = if stable {
            metadata.get("invarlock_id").unwrap_or(&row["doc_id"])
        } else {
            &row["doc_id"]
        };
        result.push(capture(json!({"id":identifier(id)?,"input":row["doc"],"context":{"arguments":row["arguments"]},"expected":row["target"],"output":output,"scores":score,"metadata":tags(&metadata)?,"error":error}),row,&metadata)?);
    }
    Ok(result)
}
fn promptfoo_error(row: &Value, response: &Value) -> Result<Value> {
    let error = truth(&row["error"]) || truth(&response["error"]);
    let Some(reason) = row.get("failureReason") else {
        return Ok(if error {
            json!("upstream_error")
        } else {
            Value::Null
        });
    };
    match reason.as_u64() {
        Some(2) => {
            if row["success"] != false {
                return Err(err("Promptfoo runtime failure contradicts success"));
            }
            Ok(json!("upstream_error"))
        }
        Some(0) => {
            if error || row["success"] == false || row["gradingResult"]["pass"] == false {
                return Err(err("Promptfoo failureReason contradicts failure fields"));
            }
            Ok(Value::Null)
        }
        Some(1) => {
            let g = &row["gradingResult"];
            if row["success"] != false
                || !g.is_object()
                || g["pass"] != false
                || response.get("output").is_none()
                || truth(&response["error"])
                || !row["score"].is_number()
                || !g["score"].is_number()
                || !numeric_equal(&row["score"], &g["score"])
                || !g["reason"].is_string()
                || row["error"] != g["reason"]
            {
                return Err(err("Promptfoo assertion failure fields are inconsistent"));
            }
            Ok(Value::Null)
        }
        _ => Err(err("Promptfoo failureReason is unsupported or ambiguous")),
    }
}
fn promptfoo(v: &Value) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for row in rows(v, "Promptfoo results")? {
        let case = &row["testCase"];
        if !case.is_object() || case.get("vars").is_none() || row.get("prompt").is_none() {
            return Err(err("Promptfoo rows need testCase and prompt for pairing; use the full export or SDK capture"));
        }
        let metadata = obj_default(&case["metadata"], "metadata")?;
        let response = obj_default(&row["response"], "response")?;
        let id = if !metadata["invarlock_id"].is_null() {
            let s = metadata["invarlock_id"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| err("Promptfoo invarlock_id must be a nonempty string"))?;
            s.to_owned()
        } else {
            format!(
                "{}:{}",
                identifier(&row["testIdx"])?,
                identifier(&row["promptIdx"])?
            )
        };
        let prompt = if row["prompt"].is_object() {
            &row["prompt"]["raw"]
        } else {
            &row["prompt"]
        };
        if !prompt.is_string() {
            return Err(err("Promptfoo row requires the actual rendered prompt"));
        }
        let score = Value::Object(
            ["score", "latencyMs", "cost"]
                .into_iter()
                .filter_map(|k| {
                    row.get(k)
                        .filter(|v| !v.is_null())
                        .map(|v| (k.into(), v.clone()))
                })
                .collect(),
        );
        result.push(capture(json!({"id":id,"input":case["vars"],"context":{"prompt":prompt},"expected":metadata["invarlock_expected"],"output":response["output"],"scores":scores(&score,false)?,"metadata":tags(&metadata)?,"error":promptfoo_error(row,&response)?}),row,&metadata)?);
    }
    Ok(result)
}
fn depth(v: &Value, n: usize, max: usize) -> Result<()> {
    if n > max {
        return Err(err("evaluator value exceeds maximum nesting"));
    }
    match v {
        Value::Array(a) => {
            if a.len() > MAX_RECORDS {
                return Err(err("evaluator array exceeds record limit"));
            }
            for x in a {
                depth(x, n + 1, max)?;
            }
        }
        Value::Object(m) => {
            for x in m.values() {
                depth(x, n + 1, max)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn evaluator(v: &Value, options: &Value) -> Result<Vec<Value>> {
    contracts::validate_schema(v, "evaluator_export").map_err(|e| err(e.to_string()))?;
    let version = v["source_version"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| err("evaluator source version requires bounded text"))?;
    if options["source"] != json!({"name":v["evaluator"],"version":version})
        || options["run_id"] != v["run_id"]
    {
        return Err(err(
            "evaluator export source or run identity differs from the request",
        ));
    }
    let name = v["evaluator"].as_str().unwrap_or("");
    let payload = &v["payload"];
    match name {
        "inspect-ai" => inspect(payload, true),
        "lm-evaluation-harness" => harness(payload, true),
        "promptfoo" => promptfoo(payload),
        "langfuse" => langfuse::parse(payload, options),
        "deepeval"
        | "ragas"
        | "lighteval"
        | "hugging-face-evaluate"
        | "autoevals"
        | "openevals"
        | "arize-phoenix-evals"
        | "opik" => scalar::parse(name, payload),
        _ => batch::parse(name, payload),
    }
}
fn project(row: &mut Value, configuration: &Value) -> Result<()> {
    if !configuration.as_object().is_some_and(|m| {
        m.len() == 2 && m.contains_key("pointer") && m.get("kind") == Some(&json!("json-pointer"))
    }) {
        return Err(err("input projection requires kind and pointer only"));
    }
    let p = configuration["pointer"]
        .as_str()
        .filter(|s| s.chars().count() <= 4096 && s.starts_with('/'))
        .ok_or_else(|| err("input projection requires an RFC6901 JSON pointer"))?;
    for part in p.split('~').skip(1) {
        if !part.starts_with(['0', '1']) {
            return Err(err("input projection requires an RFC6901 JSON pointer"));
        }
    }
    if !["input", "context"].contains(&p.split('/').nth(1).unwrap_or("")) {
        return Err(err("input projection must select input or context"));
    }
    if row["context"].get("input_projection").is_some() {
        return Err(err("input projection context key is reserved"));
    }
    let source = json!({"input":row["input"],"context":row["context"]});
    let selected = source
        .pointer(p)
        .ok_or_else(|| err("input projection pointer does not resolve"))?;
    if !selected.is_string() {
        return Err(err("input projection must select one text string"));
    }
    row["input"] = selected.clone();
    if !row["context"].is_object() {
        row["context"] = json!({});
    }
    row["context"]["input_projection"] = json!({"configuration":configuration,"configuration_digest":contracts::canonical_json_digest(configuration)?,"source_digest":contracts::canonical_json_digest(&source)?,"source":source});
    Ok(())
}
pub fn parse(raw: &[u8], options: &Value) -> Result<Value> {
    if raw.len() > 128 * 1024 * 1024 {
        return Err(err("evaluation export exceeds its byte limit"));
    }
    let adapter = options["adapter"].as_str().unwrap_or("invarlock");
    let records = if ["jsonl", "lm-eval-samples", "promptfoo-jsonl"].contains(&adapter) {
        let mut rows = Vec::new();
        for (i, line) in raw.split(|b| *b == b'\n').enumerate() {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            if rows.len() == MAX_RECORDS {
                return Err(err(format!(
                    "{adapter} exceeds the {MAX_RECORDS}-record limit"
                )));
            }
            rows.push(contracts::parse_json(
                line,
                &format!("export line {}", i + 1),
            )?);
        }
        let v = json!(rows);
        match adapter {
            "jsonl" => self::rows(&v, adapter)?.clone(),
            "lm-eval-samples" => harness(&v, false)?,
            _ => promptfoo(&v)?,
        }
    } else {
        let mut v = contracts::parse_json(raw, "evaluation export")?;
        match adapter {
            "invarlock" => {
                core::check_run(&v).map_err(|e| err(e.to_string()))?;
                if [
                    "source",
                    "run_id",
                    "artifact_digest",
                    "service_identity",
                    "score_provenance",
                    "input_projection",
                ]
                .iter()
                .any(|k| options.get(*k).is_some_and(|x| !x.is_null()))
                {
                    return Err(err(
                        "canonical run identities cannot be overridden at import",
                    ));
                }
                return Ok(v);
            }
            "inspect-json" => inspect(&v, false)?,
            "langfuse-json" => langfuse::parse(&v, options)?,
            "evaluator-json" | "evaluator-native-json" => {
                if adapter == "evaluator-native-json" {
                    if options["source"].is_null() || options["run_id"].is_null() {
                        return Err(err(
                            "native evaluator JSON requires source name/version and run_id",
                        ));
                    }
                    if options["source"]["name"] == "langfuse" {
                        v = json!({"format":"invarlock/langfuse-export-v1","sdk_version":options["source"]["version"],"result":v});
                    }
                    v = json!({"format":"invarlock/evaluator-export-v1","evaluator":options["source"]["name"],"source_version":options["source"]["version"],"run_id":options["run_id"],"payload":v});
                }
                evaluator(&v, options)?
            }
            _ => return Err(err(format!("unsupported adapter {adapter:?}"))),
        }
    };
    if options["source"].is_null()
        || options["run_id"].is_null()
        || (options["artifact_digest"].is_null() && options["service_identity"].is_null())
    {
        return Err(err("native import requires source name/version, run_id and artifact_digest or service_identity from your pipeline"));
    }
    let mut normalized = Vec::with_capacity(records.len());
    for row in records {
        let fields = object(&row, "record")?;
        if fields.keys().any(|k| {
            ![
                "id",
                "input",
                "expected",
                "output",
                "scores",
                "metadata",
                "error",
                "context",
                "likelihood",
            ]
            .contains(&k.as_str())
        }) {
            return Err(err(
                "record has unknown fields; map them explicitly in the adapter",
            ));
        }
        let mut record = json!({"scores":{},"metadata":{},"error":null,"context":{}});
        record.as_object_mut().unwrap().extend(fields.clone());
        normalized.push(record);
    }
    let mut run = json!({"format":"invarlock/evaluation-run-v1","source":options["source"],"run_id":options["run_id"],"artifact_digest":options["artifact_digest"],"source_digest":contracts::sha256_digest(raw),"score_provenance":if truth(&options["score_provenance"]){options["score_provenance"].clone()}else{json!({})},"records":normalized});
    if !options["service_identity"].is_null() {
        run["service_identity"] = options["service_identity"].clone();
    }
    core::check_run(&run).map_err(|e| err(e.to_string()))?;
    if !options["input_projection"].is_null() {
        for row in run["records"].as_array_mut().unwrap() {
            project(row, &options["input_projection"])?;
        }
        core::check_run(&run).map_err(|e| err(e.to_string()))?;
    }
    Ok(run)
}
