use super::*;
fn one(v: &Value, name: &str) -> Result<Value> {
    let a = v.as_array().filter(|a| a.len() == 1).ok_or_else(|| {
        err(format!(
            "{name} requires exactly one per-case value; select a single response"
        ))
    })?;
    Ok(a[0].clone())
}
fn metric(name: &str, v: &Value) -> Result<()> {
    if ["lighteval", "hugging-face-evaluate"].contains(&name) {
        if name == "lighteval" && v.is_number() {
            return Ok(());
        }
        if v.as_object().is_none_or(|m| m.is_empty()) {
            return Err(err(
                "metric_result requires a nonempty per-case metric dictionary",
            ));
        }
        return Ok(());
    }
    if let Some(a) = v.as_array() {
        if !["openevals", "arize-phoenix-evals", "opik"].contains(&name) || a.is_empty() {
            return Err(err("metric_result has an unsupported or empty result list"));
        }
        for x in a {
            if x.is_array() {
                return Err(err(
                    "metric_result requires a flat list of native metric results",
                ));
            }
            metric(name, x)?;
        }
        return Ok(());
    }
    object(v, "metric_result")?;
    required(
        v,
        if ["ragas", "opik"].contains(&name) {
            "value"
        } else {
            "score"
        },
    )?;
    if ["autoevals", "opik", "openevals"].contains(&name) {
        let k = if name == "openevals" { "key" } else { "name" };
        if required(v, k)?.as_str().is_none_or(|s| s.is_empty()) {
            return Err(err("metric_result requires a nonempty metric name"));
        }
    }
    Ok(())
}
pub(super) fn parse(name: &str, v: &Value) -> Result<Vec<Value>> {
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for row in rows(v, "scalar evaluator export")? {
        depth(row, 0, 32)?;
        if let Some(m) = row.get("metric_result") {
            metric(name, m)?;
        }
        let id = required(row, "id")?
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err("native cases require unique, nonempty stable string ids"))?;
        if !seen.insert(id) {
            return Err(err(
                "native cases require unique, nonempty stable string ids",
            ));
        }
        let (input, expected, output) = match name {
            "deepeval" => {
                let c = required(row, "test_case")?;
                object(c, "test_case")?;
                (
                    required(c, "input")?.clone(),
                    c["expected_output"].clone(),
                    required(c, "actual_output")?.clone(),
                )
            }
            "ragas" => {
                let c = required(row, "sample")?;
                object(c, "sample")?;
                if truth(&c["multi_responses"]) {
                    return Err(err(
                        "Ragas multi_responses is ambiguous; select one response explicitly",
                    ));
                }
                (
                    required(c, "user_input")?.clone(),
                    c["reference"].clone(),
                    required(c, "response")?.clone(),
                )
            }
            "lighteval" => {
                let c = required(row, "doc")?;
                object(c, "doc")?;
                let response = required(row, "model_response")?;
                object(response, "model_response")?;
                let choices = required(c, "choices")?;
                let indices = required(c, "gold_index")?;
                let gold = if choices == &json!([]) && indices.as_u64() == Some(0) {
                    Value::Null
                } else {
                    let choices = array(choices, "LightEval choices", false)?;
                    if choices.iter().any(|x| !x.is_string()) {
                        return Err(err(
                            "LightEval gold_index must select valid, unique text choices",
                        ));
                    }
                    let indices = indices
                        .as_array()
                        .cloned()
                        .unwrap_or_else(|| vec![indices.clone()]);
                    let mut seen = BTreeSet::new();
                    let mut golds = Vec::new();
                    for i in indices {
                        let index = i
                            .as_u64()
                            .filter(|n| *n < choices.len() as u64)
                            .ok_or_else(|| {
                                err("LightEval gold_index must select valid, unique text choices")
                            })? as usize;
                        if !seen.insert(index) {
                            return Err(err(
                                "LightEval gold_index must select valid, unique text choices",
                            ));
                        }
                        golds.push(choices[index].clone());
                    }
                    if golds.is_empty() {
                        return Err(err(
                            "LightEval gold_index must select valid, unique text choices",
                        ));
                    }
                    if golds.len() == 1 {
                        golds.remove(0)
                    } else {
                        json!(golds)
                    }
                };
                (
                    required(c, "query")?.clone(),
                    gold,
                    one(required(response, "text")?, "LightEval ModelResponse.text")?,
                )
            }
            "hugging-face-evaluate" => (
                required(row, "input")?.clone(),
                row.get("references")
                    .map(|v| one(v, "Evaluate references"))
                    .transpose()?
                    .unwrap_or(Value::Null),
                one(required(row, "predictions")?, "Evaluate predictions")?,
            ),
            "openevals" => (
                required(row, "inputs")?.clone(),
                row["reference_outputs"].clone(),
                required(row, "outputs")?.clone(),
            ),
            "arize-phoenix-evals" | "opik" => {
                let c = required(
                    row,
                    if name == "opik" {
                        "dataset_item"
                    } else {
                        "record"
                    },
                )?;
                object(c, "native case")?;
                (
                    required(c, "input")?.clone(),
                    c[if name == "opik" {
                        "reference"
                    } else {
                        "expected"
                    }]
                    .clone(),
                    required(c, "output")?.clone(),
                )
            }
            _ => (
                required(row, "input")?.clone(),
                row["expected"].clone(),
                required(row, "output")?.clone(),
            ),
        };
        if input.is_null() {
            return Err(err("native case requires a per-case input"));
        }
        let error = &row["error"];
        if !error.is_null() && error.as_str().is_none_or(|s| s.is_empty()) {
            return Err(err("native error must be a nonempty string or null"));
        }
        let empty = json!({});
        let metadata = row.get("metadata").unwrap_or(&empty);
        object(metadata, "native entry metadata")?;
        if output.is_null() && error.is_null() && metadata.get("invarlock_likelihood").is_none() {
            return Err(err("native case requires an actual output or null output with an explicit error or likelihood observation"));
        }
        result.push(capture(json!({"id":id,"input":input,"expected":expected,"output":output,"metadata":tags(metadata)?,"error":error}),row,metadata)?);
    }
    Ok(result)
}
