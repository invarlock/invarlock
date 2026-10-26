//! Direct-process measurement boundary for the retained offline experiment.
use invarlock_native_contracts::{canonical_json, parse_json, sha256_digest};
use invarlock_native_evidence::{transaction, EvidenceError};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{path::Path, time::Instant};
fn run() -> Result<Value, EvidenceError> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| EvidenceError::incomplete("request path required"))?;
    let raw = fs::read_file(Path::new(&path), 4 * 1024 * 1024)
        .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
    let request = parse_json(&raw, "benchmark request")?;
    let started = Instant::now();
    let mut previous = Value::Null;
    let mut outputs = vec![];
    for step in request["steps"]
        .as_array()
        .ok_or_else(|| EvidenceError::incomplete("steps required"))?
    {
        let mut config = step["config"].clone();
        if step["previous_result"] == true {
            config["result"] = previous.clone();
        }
        let result = transaction::dispatch(
            step["operation"]
                .as_str()
                .ok_or_else(|| EvidenceError::incomplete("operation required"))?,
            &config,
            None,
            None,
            None,
        )?;
        previous = result.payload.clone();
        outputs.push(json!({"status":result.status,"payload":result.payload}));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let mut files = serde_json::Map::new();
    for (label, path) in request["files"]
        .as_object()
        .ok_or_else(|| EvidenceError::incomplete("files required"))?
    {
        let raw = fs::read_file(Path::new(path.as_str().unwrap()), 384 * 1024 * 1024)
            .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
        files.insert(label.clone(), json!(sha256_digest(&raw)));
    }
    Ok(
        json!({"ok":true,"operation_wall_seconds":elapsed,"result":{"outputs":outputs,"files":files},"native_transactions":outputs.len()}),
    )
}
fn main() {
    match run() {
        Ok(value) => print!(
            "{}",
            String::from_utf8(canonical_json(&value).unwrap()).unwrap()
        ),
        Err(error) => {
            println!("{}", json!({"ok":false,"error":error.to_string()}));
            std::process::exit(1);
        }
    }
}
