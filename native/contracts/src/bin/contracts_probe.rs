//! JSON-lines test bridge. This binary is a development qualification utility.
use invarlock_native_contracts::{
    canonical_json, canonical_json_no_newline, parse_json, schema_names, sha256_digest,
    validate_schema,
};
use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let output = match line {
            Ok(line) => probe(&line),
            Err(error) => serde_json::json!({"ok":false,"message":error.to_string()}),
        };
        println!("{output}");
    }
}
fn probe(line: &str) -> serde_json::Value {
    let request: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"ok":false,"message":e.to_string()}),
    };
    if request.get("op").and_then(|v| v.as_str()) == Some("schemas") {
        return serde_json::json!({"ok":true,"schemas":schema_names().collect::<Vec<_>>()});
    }
    if request.get("op").and_then(|v| v.as_str()) == Some("schedule") {
        use invarlock_native_contracts::schedule::{
            preparation_descriptor, prepare_local, validate_preparation, Schedule,
        };
        let result = (|| {
            let task = request["task"].as_str().unwrap_or("text_causal");
            let schedule = match request["action"].as_str().unwrap_or("build") {
                "parse" => Schedule::parse(request["raw"].as_str().unwrap_or("").as_bytes())?,
                "material" => {
                    Schedule::from_material(&request["identity"], &request["records"], task)?
                }
                "prepare" => prepare_local(
                    &request["source"],
                    request["raw"].as_str().unwrap_or("").as_bytes(),
                    task,
                )?,
                _ => Schedule::build(&request["value"])?,
            };
            let mut out = serde_json::json!({"ok":true,"value":schedule.value(),"canonical":String::from_utf8_lossy(schedule.bytes()),"digest":schedule.digest()});
            if request.get("descriptor").is_some() {
                validate_preparation(&request["descriptor"], &schedule)?;
            }
            if request["make_descriptor"] == true {
                out["descriptor"] = preparation_descriptor(&request["source"], &schedule)?;
            }
            Ok::<_, invarlock_native_contracts::ContractError>(out)
        })();
        return match result {
            Ok(value) => value,
            Err(e) => serde_json::json!({"ok":false,"code":e.code,"message":e.message}),
        };
    }
    let raw = request["raw"].as_str().unwrap_or("").as_bytes();
    let result = (|| {
        let value = parse_json(raw, "probe")?;
        if let Some(schema) = request.get("schema_value") {
            if let Some(error) = invarlock_native_contracts::first_schema_error_value(
                &value,
                schema,
                invarlock_native_contracts::SchemaErrorOrder::Schema,
            )? {
                return Err(invarlock_native_contracts::ContractError::new(
                    "schema_validation",
                    error.message,
                ));
            }
        }
        if let Some(name) = request.get("schema").and_then(|v| v.as_str()) {
            validate_schema(&value, name)?;
        }
        let bytes = if request.get("newline").and_then(|v| v.as_bool()) == Some(false) {
            canonical_json_no_newline(&value)?
        } else {
            canonical_json(&value)?
        };
        Ok::<_, invarlock_native_contracts::ContractError>((
            String::from_utf8(bytes.clone()).expect("canonical UTF-8"),
            sha256_digest(&bytes),
        ))
    })();
    match result {
        Ok((canonical, digest)) => {
            serde_json::json!({"ok":true,"canonical":canonical,"digest":digest})
        }
        Err(error) => serde_json::json!({"ok":false,"code":error.code,"message":error.message}),
    }
}
