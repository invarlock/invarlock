use super::*;
fn closed(v: &Value, required: &[&str], optional: &[&str], label: &str) -> Result<()> {
    let m = object(v, label)?;
    if required.iter().any(|k| !m.contains_key(*k))
        || m.keys()
            .any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(err(format!(
            "Langfuse {label} has missing or unsupported fields"
        )));
    }
    Ok(())
}
fn text(v: &Value, label: &str) -> Result<String> {
    v.as_str()
        .filter(|s| !s.is_empty() && s.chars().count() <= 4096)
        .map(str::to_owned)
        .ok_or_else(|| err(format!("Langfuse {label} requires nonempty bounded text")))
}
fn optional(v: &Value, label: &str) -> Result<()> {
    if !v.is_null() {
        text(v, label)?;
    }
    Ok(())
}
fn evaluations(v: &Value) -> Result<Value> {
    let mut scores = Map::new();
    let mut names = BTreeSet::new();
    for x in array(v, "Langfuse evaluations", true)? {
        closed(
            x,
            &["name", "value"],
            &["data_type", "comment", "metadata", "config_id"],
            "evaluation",
        )?;
        let name = text(&x["name"], "score name")?;
        if !names.insert(name.clone()) {
            return Err(err("Langfuse duplicate evaluation names are ambiguous"));
        }
        if !x["metadata"].is_null() {
            object(&x["metadata"], "Langfuse evaluation metadata")?;
        }
        for k in ["comment", "config_id"] {
            if !x[k].is_null() && !x[k].is_string() {
                return Err(err(format!("Langfuse evaluation {k} must be text or null")));
            }
        }
        let kind = &x["data_type"];
        let value = &x["value"];
        if kind == "CATEGORICAL" && value.is_string() {
            continue;
        }
        let score = if kind == "BOOLEAN"
            && (value.is_boolean()
                || value.is_number() && matches!(value.as_f64(), Some(0.0 | 1.0)))
        {
            number(value, "Langfuse evaluation", true)?
        } else if (kind.is_null() || kind == "NUMERIC") && value.is_number() {
            number(value, "Langfuse evaluation", false)?
        } else {
            return Err(err(
                "Langfuse evaluation value contradicts its declared data_type",
            ));
        };
        scores.insert(name, score);
    }
    Ok(Value::Object(scores))
}
pub(super) fn parse(v: &Value, options: &Value) -> Result<Vec<Value>> {
    closed(v, &["format", "sdk_version", "result"], &[], "export")?;
    if v["format"] != "invarlock/langfuse-export-v1" {
        return Err(err("langfuse-json requires invarlock/langfuse-export-v1"));
    }
    let version = text(&v["sdk_version"], "SDK version")?;
    if options["source"] != json!({"name":"langfuse","version":version}) {
        return Err(err(
            "Langfuse source name/version must match the export SDK version",
        ));
    }
    let r = &v["result"];
    closed(
        r,
        &[
            "name",
            "run_name",
            "experiment_id",
            "item_results",
            "run_evaluations",
        ],
        &["description", "dataset_run_id", "dataset_run_url"],
        "result",
    )?;
    for k in ["name", "run_name", "experiment_id"] {
        text(&r[k], k)?;
    }
    if options["run_id"] != r["run_name"] {
        return Err(err("Langfuse run_id must match exported run_name"));
    }
    for k in ["dataset_run_id", "dataset_run_url"] {
        optional(&r[k], k)?;
    }
    if !r["description"].is_null() && !r["description"].is_string() {
        return Err(err("Langfuse description must be text or null"));
    }
    let dataset = &r["dataset_run_id"];
    if !dataset.is_null() && dataset != &r["experiment_id"] {
        return Err(err("Langfuse dataset run and experiment IDs differ"));
    }
    evaluations(&r["run_evaluations"])?;
    let hosted = [
        "source_trace_id",
        "source_observation_id",
        "dataset_id",
        "dataset_name",
        "created_at",
        "updated_at",
    ];
    let mut result = Vec::new();
    for row in array(&r["item_results"], "Langfuse item_results", false)? {
        closed(
            row,
            &["item", "output", "evaluations"],
            &["trace_id", "dataset_run_id"],
            "item result",
        )?;
        let item = &row["item"];
        let mut optional_fields = vec![
            "expected_output",
            "metadata",
            "id",
            "status",
            "media_references",
        ];
        optional_fields.extend(hosted);
        closed(item, &["input"], &optional_fields, "item")?;
        if hosted
            .iter()
            .chain(["status", "media_references"].iter())
            .any(|k| item.get(*k).is_some())
        {
            text(&item["id"], "hosted native ID")?;
            if item.get("status").is_some()
                && item["status"] != "ACTIVE"
                && item["status"] != "ARCHIVED"
            {
                return Err(err("Langfuse dataset item status is unsupported"));
            }
            for k in hosted {
                optional(&item[k], k)?;
            }
            if let Some(media) = item.get("media_references") {
                for x in array(media, "Langfuse media references", true)? {
                    object(x, "media reference")?;
                }
            }
        }
        let metadata = obj_default(&item["metadata"], "Langfuse item metadata")?;
        let local = &metadata["invarlock_id"];
        let id = &item["id"];
        optional(local, "invarlock_id")?;
        optional(id, "item ID")?;
        if !local.is_null() && !id.is_null() && local != id {
            return Err(err("Langfuse item ID conflicts with invarlock_id"));
        }
        let record_id = text(if id.is_null() { local } else { id }, "stable item ID")?;
        optional(&row["trace_id"], "trace ID")?;
        if &row["dataset_run_id"] != dataset {
            return Err(err("Langfuse item dataset run ID differs from experiment"));
        }
        if !dataset.is_null() && id.is_null() {
            return Err(err("Langfuse dataset item requires its native ID"));
        }
        optional(&metadata["invarlock_error"], "invarlock_error")?;
        let mut record = json!({"id":record_id,"input":item["input"],"expected":item["expected_output"],"output":row["output"],"error":metadata["invarlock_error"],"scores":merge_scores(&evaluations(&row["evaluations"] )?,&metadata)?,"metadata":tags(&metadata)?,"context":{"langfuse":{"name":r["name"],"run_name":r["run_name"],"experiment_id":r["experiment_id"],"dataset_run_id":dataset,"item_result":row}}});
        if let Some(f) = metadata.get("invarlock_likelihood") {
            record["likelihood"] = f.clone();
        }
        result.push(record);
    }
    Ok(result)
}
