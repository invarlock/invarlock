use super::*;
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.to_string().contains(['.', 'e', 'E']) == y.to_string().contains(['.', 'e', 'E'])
                && numeric_equal(a, b)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}
fn id(v: &Value) -> Result<String> {
    v.as_str()
        .filter(|s| !s.is_empty() && s.chars().count() <= 4096)
        .map(str::to_owned)
        .ok_or_else(|| err("evaluator row requires an independent nonempty string ID"))
}
fn alias(v: &Value, names: &[&str], label: &str) -> Result<Value> {
    let mut selected = None;
    for k in names {
        if let Some(x) = v.get(*k) {
            if selected.as_ref().is_some_and(|s| !same(s, x)) {
                return Err(err(format!("{label} aliases conflict")));
            }
            if selected.is_none() {
                selected = Some(x.clone());
            }
        }
    }
    Ok(selected.unwrap_or(Value::Null))
}
fn error(v: &Value) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    if v.as_str().is_some_and(|s| !s.is_empty()) {
        return Ok(v.clone());
    }
    if v.is_object() {
        for k in ["message", "error_message", "msg"] {
            if v[k].as_str().is_some_and(|s| !s.is_empty()) {
                return Ok(v[k].clone());
            }
        }
    }
    Err(err("native error requires a nonempty message"))
}
#[allow(clippy::too_many_arguments)] // Each field binds one retained upstream record component.
fn record(
    native: &Value,
    ident: &Value,
    input: &Value,
    output: &Value,
    expected: &Value,
    scores: &Value,
    failure: &Value,
    summary: &Value,
) -> Result<Value> {
    let metadata = obj_default(&native["metadata"], "native metadata")?;
    let mut r = json!({"id":id(ident)?,"input":input,"expected":expected,"output":output,"scores":merge_scores(scores,&metadata)?,"error":error(failure)?,"metadata":tags(&metadata)?,"context":{"upstream_record":native}});
    if truth(summary) {
        r["context"]["upstream_summary"] = summary.clone();
    }
    for owner in [native, &metadata] {
        for k in ["likelihood", "invarlock_likelihood"] {
            if let Some(v) = owner.get(k) {
                if r.get("likelihood").is_some_and(|old| !same(old, v)) {
                    return Err(err("conflicting explicit likelihood evidence"));
                }
                r["likelihood"] = v.clone();
            }
        }
    }
    Ok(r)
}
fn table(v: &Value) -> Result<Vec<Value>> {
    depth(v, 0, 64)?;
    if v.as_object()
        .is_some_and(|m| m.len() == 2 && m.contains_key("columns") && m.contains_key("data"))
    {
        let columns = array(&v["columns"], "table columns", false)?;
        let mut seen = BTreeSet::new();
        for c in columns {
            let s = c
                .as_str()
                .ok_or_else(|| err("table columns require unique names"))?;
            if !seen.insert(s) {
                return Err(err("table columns require unique names"));
            }
        }
        let mut rows = Vec::new();
        for r in array(&v["data"], "table data", false)? {
            let r = r
                .as_array()
                .filter(|a| a.len() == columns.len())
                .ok_or_else(|| err("table row and column lengths differ"))?;
            rows.push(Value::Object(
                columns
                    .iter()
                    .zip(r)
                    .map(|(k, v)| (k.as_str().unwrap().into(), v.clone()))
                    .collect(),
            ));
        }
        Ok(rows)
    } else {
        Ok(rows(v, "table rows")?.clone())
    }
}
fn excluding(v: &Value, keys: &[&str]) -> Result<Value> {
    Ok(Value::Object(
        object(v, "report")?
            .iter()
            .filter(|(k, _)| !keys.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    ))
}
fn pydantic(v: &Value) -> Result<Vec<Value>> {
    object(v, "Pydantic Evals report")?;
    let summary = excluding(v, &["cases", "failures"])?;
    let empty = json!([]);
    let mut result = Vec::new();
    for (list, failed) in [
        (required(v, "cases")?, false),
        (v.get("failures").unwrap_or(&empty), true),
    ] {
        for row in array(list, "Pydantic Evals cases", true)? {
            object(row, "Pydantic Evals case")?;
            if row.get("inputs").is_none() || (!failed && row.get("output").is_none()) {
                return Err(err("Pydantic Evals case lacks inputs or actual output"));
            }
            let mut scores = Map::new();
            for group in ["assertions", "scores"] {
                let fallback = json!({});
                for (name, metric) in object(row.get(group).unwrap_or(&fallback), group)? {
                    if scores.contains_key(name) {
                        return Err(err("Pydantic Evals score names collide"));
                    }
                    scores.insert(
                        id(&json!(name))?,
                        number(
                            if metric.is_object() {
                                &metric["value"]
                            } else {
                                metric
                            },
                            name,
                            true,
                        )?,
                    );
                }
            }
            let failure = &row[if failed { "error_message" } else { "error" }];
            array(
                row.get("evaluator_failures").unwrap_or(&empty),
                "Pydantic Evals evaluator failures",
                true,
            )?;
            if failed && failure.is_null() {
                return Err(err("Pydantic Evals failed case lacks error_message"));
            }
            result.push(record(
                row,
                &row["name"],
                &row["inputs"],
                &row["output"],
                &row["expected_output"],
                &Value::Object(scores),
                failure,
                &summary,
            )?);
        }
    }
    Ok(result)
}
fn azure(v: &Value) -> Result<Vec<Value>> {
    object(v, "Azure AI Evaluation result")?;
    let summary = excluding(v, &["rows"])?;
    let mut result = Vec::new();
    for mut row in table(&v["rows"])? {
        if let Some(m) = row.get("inputs.metadata") {
            if row.get("metadata").is_some_and(|x| !same(x, m)) {
                return Err(err("Azure capture metadata conflicts with inputs.metadata"));
            }
            row["metadata"] = m.clone();
        }
        if row.get("inputs.response").is_none()
            && row.get("outputs.response").is_none()
            && !object(&row, "row")?
                .iter()
                .any(|(k, v)| (k == "error" || k.ends_with(".error")) && !v.is_null())
        {
            return Err(err("Azure row lacks captured response"));
        }
        let mut scores = Map::new();
        for (k, v) in object(&row, "row")? {
            if k.starts_with("outputs.")
                && k != "outputs.response"
                && (v.is_number() || v.is_boolean())
            {
                scores.insert(k.trim_start_matches("outputs.").into(), number(v, k, true)?);
            }
        }
        result.push(record(
            &row,
            &alias(
                &row,
                &["inputs.record_id", "inputs.id"],
                "Azure record identity",
            )?,
            &alias(&row, &["inputs.query", "inputs.input"], "Azure input")?,
            &alias(
                &row,
                &["outputs.response", "inputs.response"],
                "Azure response",
            )?,
            &row["inputs.ground_truth"],
            &Value::Object(scores),
            &row["error"],
            &summary,
        )?);
    }
    Ok(result)
}
fn scored_table(v: &Value, name: &str) -> Result<Vec<Value>> {
    let mut report = if v.is_object() {
        v.clone()
    } else {
        json!({"rows":v})
    };
    for k in ["rows", "dataset", "prediction_table"] {
        if let Some(v) = report.get(k) {
            report[k] = json!(table(v)?);
        }
    }
    if let Some(tables) = report.get("tables") {
        let mut out = Map::new();
        for (k, v) in object(tables, "MLflow tables")? {
            out.insert(k.clone(), json!(table(v)?));
        }
        report["tables"] = Value::Object(out);
    }
    let source = if name == "mlflow" {
        let mut sources = json!({});
        for (k, v) in [
            ("prediction_table", &report["prediction_table"]),
            ("rows", &report["rows"]),
            (
                "tables.eval_results_table",
                &report["tables"]["eval_results_table"],
            ),
        ] {
            if !v.is_null() {
                sources[k] = v.clone();
            }
        }
        let names: Vec<_> = object(&sources, "sources")?
            .keys()
            .map(String::as_str)
            .collect();
        alias(&sources, &names, "MLflow prediction table")?
    } else {
        alias(&report, &["dataset", "rows"], "Evidently dataset")?
    };
    let mut columns = if name == "mlflow" {
        json!({"id":"record_id","input":"input","expected":"target","output":"prediction"})
    } else {
        json!({"id":"record_id","input":"input","expected":"reference","output":"output"})
    };
    let fallback = json!({});
    let supplied = report.get("columns").unwrap_or(&fallback);
    for (k, v) in object(supplied, "column mapping")? {
        if columns.get(k).is_none() || v.as_str().is_none_or(|s| s.is_empty()) {
            return Err(err("unsupported table column mapping"));
        }
        columns[k] = v.clone();
    }
    let values: BTreeSet<_> = object(&columns, "columns")?
        .values()
        .map(|x| x.as_str().unwrap())
        .collect();
    if values.len() != 4 {
        return Err(err(
            "table column mapping must use a different source column for each role",
        ));
    }
    let names = report.get("score_columns").filter(|v| !v.is_null());
    if let Some(names) = names {
        let mut seen = BTreeSet::new();
        for x in array(names, "score_columns", false)? {
            let s = x
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| err("score_columns requires unique names"))?;
            if !seen.insert(s) {
                return Err(err("score_columns requires unique names"));
            }
        }
    }
    let summary = excluding(&report, &["rows", "dataset", "tables", "prediction_table"])?;
    let mut result = Vec::new();
    for row in table(&source)? {
        let mut output = columns["output"].as_str().unwrap();
        let mut expected = columns["expected"].as_str().unwrap();
        if name == "mlflow" {
            if supplied.get("output").is_none() {
                alias(&row, &["prediction", "predictions"], "MLflow prediction")?;
                if row.get("predictions").is_some() {
                    output = "predictions";
                }
            }
            if supplied.get("expected").is_none() {
                alias(&row, &["target", "targets"], "MLflow target")?;
                if row.get("targets").is_some() {
                    expected = "targets";
                }
            }
        }
        if row.get(output).is_none() && !truth(&row["error"]) {
            return Err(err(format!(
                "{name} table lacks actual model output column"
            )));
        }
        let keys: Vec<String> = if let Some(names) = names {
            names
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().into())
                .collect()
        } else {
            object(&row, "row")?
                .keys()
                .filter(|k| {
                    (name == "mlflow" && k.ends_with("/score"))
                        || (name == "evidently" && k.as_str() == "exact_match")
                })
                .cloned()
                .collect()
        };
        let mut scores = Map::new();
        for k in keys {
            let v = row
                .get(&k)
                .ok_or_else(|| err(format!("missing score column: {k}")))?;
            if !v.is_null() {
                scores.insert(k.clone(), number(v, &k, true)?);
            }
        }
        result.push(record(
            &row,
            &row[columns["id"].as_str().unwrap()],
            &row[columns["input"].as_str().unwrap()],
            &row[output],
            &row[expected],
            &Value::Object(scores),
            &row["error"],
            &summary,
        )?);
    }
    Ok(result)
}
fn garak_text(v: &Value, prompt: bool) -> Value {
    if prompt
        && v.as_object()
            .is_some_and(|m| m.keys().all(|k| ["turns", "notes"].contains(&k.as_str())))
    {
        if let Some(turns) = v["turns"].as_array().filter(|a| a.len() == 1) {
            let t = &turns[0];
            if t["role"] == "user"
                && t.as_object()
                    .is_some_and(|m| m.keys().all(|k| ["role", "content"].contains(&k.as_str())))
            {
                let selected = garak_text(&t["content"], false);
                if selected.is_string() {
                    return selected;
                }
            }
        }
        return v.clone();
    }
    if v["text"].is_string()
        && v.as_object().is_some_and(|m| {
            m.keys().all(|k| {
                [
                    "text",
                    "lang",
                    "data_path",
                    "data_type",
                    "data_checksum",
                    "notes",
                ]
                .contains(&k.as_str())
            })
        })
        && ["data_path", "data_type", "data_checksum"]
            .iter()
            .all(|k| v[*k].is_null())
    {
        v["text"].clone()
    } else {
        v.clone()
    }
}
fn garak(v: &Value) -> Result<Vec<Value>> {
    let v = if v.is_object() {
        v.clone()
    } else {
        json!({"attempts":v})
    };
    let entries = alias(&v, &["attempts", "entries"], "Garak attempt collection")?;
    let mut summary = excluding(&v, &["attempts", "entries", "source_cases"])?;
    let mut histories = Map::new();
    let mut other = Vec::new();
    for row in array(&entries, "Garak attempts", false)? {
        object(row, "Garak entry")?;
        if row.get("entry_type").is_some_and(|x| x != "attempt") {
            if !["eval", "init", "start_run", "config", "completion"]
                .contains(&row["entry_type"].as_str().unwrap_or(""))
            {
                return Err(err("unsupported Garak report entry"));
            }
            other.push(row.clone());
            continue;
        }
        let ident = id(&row["uuid"])?;
        let status = row["status"]
            .as_u64()
            .filter(|s| *s <= 2)
            .ok_or_else(|| err("Garak attempt has invalid status"))?;
        let history = histories
            .entry(ident)
            .or_insert(json!([]))
            .as_array_mut()
            .unwrap();
        if history.last().is_some_and(|last| {
            last["status"].as_u64().unwrap() >= status || !same(&last["prompt"], &row["prompt"])
        }) {
            return Err(err("Garak attempt history is ambiguous or conflicts"));
        }
        history.push(row.clone());
    }
    if !other.is_empty() {
        summary["report_entries"] = json!(other);
    }
    let mut records = Vec::new();
    for (ident, hist) in histories {
        let history = hist.as_array().unwrap();
        let row = history.last().unwrap();
        required(row, "prompt")?;
        let outputs = array(&row["outputs"], "Garak outputs", true)?;
        let fallback = json!({});
        let detectors = object(
            row.get("detector_results").unwrap_or(&fallback),
            "Garak detector results",
        )?;
        for (k, v) in detectors {
            if array(v, k, true)?.len() != outputs.len() {
                return Err(err("Garak detector scores and outputs are misaligned"));
            }
        }
        for index in 0..outputs.len().max(1) {
            if records.len() == MAX_RECORDS {
                return Err(err("Garak captured records exceed record limit"));
            }
            let output = outputs
                .get(index)
                .map(|x| garak_text(x, false))
                .unwrap_or(Value::Null);
            let mut failure = row["error"].clone();
            let likelihood = [row, &row["metadata"]].iter().any(|o| {
                ["likelihood", "invarlock_likelihood"]
                    .iter()
                    .any(|k| o.get(*k).is_some())
            });
            if failure.is_null() && (row["status"] != 2 || (output.is_null() && !likelihood)) {
                failure = json!(if row["status"] != 2 {
                    "Garak attempt incomplete"
                } else {
                    "Garak generation returned no output"
                });
            }
            let mut scores = Map::new();
            for (k, v) in detectors {
                if !outputs.is_empty() && !v[index].is_null() {
                    scores.insert(k.clone(), number(&v[index], k, true)?);
                }
            }
            let mut native = row.clone();
            native["attempt_history"] = hist.clone();
            native["generation_index"] = json!(index);
            records.push(record(
                &native,
                &json!(format!("{ident}:{index}")),
                &garak_text(&row["prompt"], true),
                &output,
                &Value::Null,
                &Value::Object(scores),
                &failure,
                &summary,
            )?);
        }
    }
    if let Some(sources) = v.get("source_cases") {
        let mut by_native = Map::new();
        for case in array(sources, "Garak source_cases", false)? {
            object(case, "Garak source case")?;
            let native = id(&case["native_id"])?;
            id(&case["id"])?;
            if case.get("input").is_none()
                || case.get("expected").is_none()
                || by_native.contains_key(&native)
            {
                return Err(err(
                    "Garak source cases require unique native_id, input and expected",
                ));
            }
            by_native.insert(native, case.clone());
        }
        if by_native.len() != records.len()
            || records
                .iter()
                .any(|r| !by_native.contains_key(r["id"].as_str().unwrap()))
        {
            return Err(err(
                "Garak source case membership differs from attempt generations",
            ));
        }
        for r in &mut records {
            let case = &by_native[r["id"].as_str().unwrap()];
            if !same(&case["input"], &r["input"])
                || case.get("output").is_some_and(|v| !same(v, &r["output"]))
            {
                return Err(err(
                    "Garak source case conflicts with actual attempt input/output",
                ));
            }
            let mut captured = record(
                case,
                &case["id"],
                &r["input"],
                &r["output"],
                &case["expected"],
                &r["scores"],
                &r["error"],
                &Value::Null,
            )?;
            let native = &r["context"]["upstream_record"];
            if captured.get("likelihood").is_some()
                && r["output"].is_null()
                && native["status"] == 2
                && native["error"].is_null()
            {
                captured["error"] = Value::Null;
            }
            if captured.get("likelihood").is_some()
                && r.get("likelihood").is_some()
                && !same(&captured["likelihood"], &r["likelihood"])
            {
                return Err(err("Garak source likelihood conflicts with attempt"));
            }
            for (k, v) in captured.as_object().unwrap() {
                if k != "context" {
                    r[k] = v.clone();
                }
            }
            r["context"]["source_case"] = case.clone();
        }
    }
    Ok(records)
}
fn run_ids(v: &Value) -> Result<BTreeSet<String>> {
    let mut ids = BTreeSet::new();
    if let Some(i) = v.get("run_id") {
        ids.insert(id(i)?);
    }
    for k in ["spec", "final_report"] {
        if let Some(o) = v.get(k) {
            object(o, "OpenAI Evals run identity")?;
            if let Some(i) = o.get("run_id") {
                ids.insert(id(i)?);
            }
        }
    }
    Ok(ids)
}
fn openai(v: &Value) -> Result<Vec<Value>> {
    let report = if v.is_object() {
        v.clone()
    } else {
        json!({"events":v})
    };
    let mut summary = excluding(&report, &["events"])?;
    let mut grouped = Map::new();
    let mut ids = run_ids(&report)?;
    let mut sample_ids = BTreeSet::new();
    let mut missing = 0;
    let mut event_ids = BTreeSet::new();
    for event in array(&report["events"], "OpenAI Evals events", false)? {
        object(event, "OpenAI Evals event")?;
        ids.extend(run_ids(event)?);
        if ((event.get("spec").is_some() || event.get("final_report").is_some())
            && event.get("type").is_none()
            && event.get("sample_id").is_none())
            || (event["type"] == "final_report" && event["sample_id"].is_null())
        {
            if summary.get("log_entries").is_none() {
                summary["log_entries"] = json!([]);
            }
            summary["log_entries"]
                .as_array_mut()
                .ok_or_else(|| err("OpenAI Evals log_entries requires an array"))?
                .push(event.clone());
            continue;
        }
        let ident = id(&event["sample_id"])?;
        object(&event["data"], "OpenAI Evals event data")?;
        if let Some(i) = event.get("run_id") {
            sample_ids.insert(id(i)?);
        } else {
            missing += 1;
        }
        if let Some(i) = event.get("event_id") {
            if !i.is_number()
                || i.to_string().contains(['.', 'e', 'E'])
                || i.to_string().starts_with('-')
                || !event_ids.insert(i.to_string())
            {
                return Err(err("OpenAI Evals duplicate or invalid event ID"));
            }
        }
        grouped
            .entry(ident)
            .or_insert(json!([]))
            .as_array_mut()
            .unwrap()
            .push(event.clone());
    }
    if ids.len() > 1 {
        return Err(err("OpenAI Evals log contains multiple runs"));
    }
    if !sample_ids.is_empty() && missing > 0 {
        return Err(err("OpenAI Evals sample events have mixed run IDs"));
    }
    let mut records = Vec::new();
    for (ident, events) in grouped {
        let a = events.as_array().unwrap();
        let select = |kind: &str| {
            a.iter()
                .filter(|e| e["type"] == kind)
                .map(|e| &e["data"])
                .collect::<Vec<_>>()
        };
        let samplings = select("sampling");
        let matches = select("match");
        let errors = select("error");
        if samplings.len() > 1 || matches.len() > 1 {
            return Err(err(
                "OpenAI Evals sample has ambiguous multiple sampling or match events",
            ));
        }
        let empty = json!({});
        let sampling = samplings.first().copied().unwrap_or(&empty);
        let matched = matches.first().copied().unwrap_or(&empty);
        for (k, message) in [
            ("prompt", "OpenAI Evals prompt aliases conflict"),
            ("sampled", "OpenAI Evals sampled outputs conflict"),
        ] {
            if sampling.get(k).is_some()
                && matched.get(k).is_some()
                && !same(&sampling[k], &matched[k])
            {
                return Err(err(message));
            }
        }
        if sampling.get("sampled").is_none()
            && matched.get("sampled").is_none()
            && errors.is_empty()
        {
            return Err(err("OpenAI Evals sample lacks actual sampled output"));
        }
        let mut scores = Map::new();
        if !matches.is_empty() {
            if !matched["correct"].is_boolean() {
                return Err(err("OpenAI Evals match requires boolean correct"));
            }
            scores.insert("match".into(), number(&matched["correct"], "match", true)?);
        }
        for event in a {
            if event["type"] == "metrics" {
                for (k, v) in object(&event["data"], "scores")? {
                    let k = id(&json!(k))?;
                    if scores.contains_key(&k) {
                        return Err(err("OpenAI Evals duplicate metric name"));
                    }
                    scores.insert(k.clone(), number(v, &k, true)?);
                }
            }
        }
        let mut native = json!({"sample_id":ident,"events":events});
        for key in ["metadata", "likelihood"] {
            for event in a {
                if let Some(v) = event["data"].get(key) {
                    if key == "metadata" && !v.is_null() {
                        object(v, "OpenAI Evals capture metadata")?;
                    }
                    if native.get(key).is_some_and(|old| !same(old, v)) {
                        return Err(err(format!("OpenAI Evals conflicting {key} evidence")));
                    }
                    native[key] = v.clone();
                }
            }
        }
        let failure = if errors.is_empty() {
            Value::Null
        } else {
            let mut messages = Vec::new();
            for e in errors {
                let v = error(e)?;
                messages.push(
                    v.as_str()
                        .ok_or_else(|| err("native error requires a nonempty message"))?
                        .to_owned(),
                );
            }
            json!(messages.join("; "))
        };
        records.push(record(
            &native,
            &json!(ident),
            sampling.get("prompt").unwrap_or(&matched["prompt"]),
            sampling.get("sampled").unwrap_or(&matched["sampled"]),
            &matched["expected"],
            &Value::Object(scores),
            &failure,
            &summary,
        )?);
    }
    Ok(records)
}
fn trulens(v: &Value) -> Result<Vec<Value>> {
    object(v, "TruLens record export")?;
    let rows = table(&v["records"])?;
    let empty = json!([]);
    let names = array(
        v.get("feedback_columns").unwrap_or(&empty),
        "TruLens feedback_columns",
        true,
    )?;
    let mut seen = BTreeSet::new();
    for x in names {
        let name = x
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err("TruLens feedback columns require unique names"))?;
        if !seen.insert(name) {
            return Err(err("TruLens feedback columns require unique names"));
        }
    }
    let mut external = Map::new();
    if let Some(fs) = v.get("feedback_results") {
        for f in array(fs, "TruLens supplied feedback", true)? {
            object(f, "TruLens supplied feedback result")?;
            let id = id(&f["record_id"])?;
            external
                .entry(id)
                .or_insert(json!([]))
                .as_array_mut()
                .unwrap()
                .push(f.clone());
        }
        if external
            .keys()
            .any(|k| !rows.iter().any(|r| r["record_id"] == k.as_str()))
        {
            return Err(err("TruLens supplied feedback has unknown record IDs"));
        }
    }
    let mut records = Vec::new();
    for mut row in rows {
        if let Some(meta) = row.get("meta") {
            if row.get("metadata").is_some_and(|x| !same(x, meta)) {
                return Err(err("TruLens meta conflicts with supplied metadata"));
            }
            row["metadata"] = meta.clone();
        }
        if let Some(add) = row["record_id"].as_str().and_then(|k| external.get(k)) {
            let mut feedback = array(
                if row["feedback_results"].is_null() {
                    &empty
                } else {
                    &row["feedback_results"]
                },
                "TruLens feedback results",
                true,
            )?
            .clone();
            feedback.extend(add.as_array().unwrap().iter().cloned());
            row["feedback_results"] = json!(feedback);
        }
        alias(&row, &["main_input", "input"], "TruLens input")?;
        alias(&row, &["main_output", "output"], "TruLens output")?;
        let (input, output) = if row.get("main_input").is_some() || row.get("main_output").is_some()
        {
            ("main_input", "main_output")
        } else {
            ("input", "output")
        };
        let failure = alias(&row, &["main_error", "error"], "TruLens error")?;
        if row.get(input).is_none() || (row.get(output).is_none() && failure.is_null()) {
            return Err(err("TruLens record lacks captured input/output"));
        }
        let mut scores = Map::new();
        for name in names {
            let k = name.as_str().unwrap();
            let value = row
                .get(k)
                .ok_or_else(|| err(format!("TruLens feedback column missing: {k}")))?;
            if !value.is_null() {
                scores.insert(k.into(), number(value, k, true)?);
            }
        }
        for f in array(
            if row["feedback_results"].is_null() {
                &empty
            } else {
                &row["feedback_results"]
            },
            "TruLens feedback results",
            true,
        )? {
            object(f, "TruLens feedback result")?;
            if f.get("record_id").unwrap_or(&row["record_id"]) != &row["record_id"] {
                return Err(err("TruLens feedback belongs to another record"));
            }
            let k = id(&f["name"])?;
            if scores.contains_key(&k) {
                return Err(err("TruLens duplicate feedback name"));
            }
            if !f["result"].is_null() {
                scores.insert(k.clone(), number(&f["result"], &k, true)?);
            }
            if !f["error"].is_null() {
                error(&f["error"])?;
            }
        }
        records.push(record(
            &row,
            &row["record_id"],
            &row[input],
            &row[output],
            &row["ground_truth"],
            &Value::Object(scores),
            &failure,
            &Value::Null,
        )?);
    }
    Ok(records)
}
pub(super) fn parse(name: &str, v: &Value) -> Result<Vec<Value>> {
    depth(v, 0, 64)?;
    let records = match name {
        "pydantic-evals" => pydantic(v)?,
        "azure-ai-evaluation" => azure(v)?,
        "mlflow" | "evidently" => scored_table(v, name)?,
        "garak" => garak(v)?,
        "openai-evals" => openai(v)?,
        "trulens" => trulens(v)?,
        _ => return Err(err(format!("unsupported evaluator {name}"))),
    };
    if records.is_empty() || records.len() > MAX_RECORDS {
        return Err(err(format!(
            "{name} captured records requires a bounded nonempty array"
        )));
    }
    let mut seen = BTreeSet::new();
    for r in &records {
        if !seen.insert(r["id"].as_str().unwrap()) {
            return Err(err(format!("{name} duplicate independent record IDs")));
        }
        depth(r, 1, 64)?;
    }
    Ok(records)
}
