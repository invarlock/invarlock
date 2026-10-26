//! Diagnostic protocol for independent transport parity fixtures.
use invarlock_native_contracts::{canonical_json_no_newline, parse_json};
use invarlock_native_evidence::{attestation, attestation_artifact};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn main() {
    let input = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in input.lock().lines() {
        let result = (|| -> Result<Value, String> {
            let line = line.map_err(|e| e.to_string())?;
            let v =
                parse_json(line.as_bytes(), "attestation request").map_err(|e| e.to_string())?;
            match v["operation"].as_str() {
                Some("report") => {
                    let result = invarlock_native_evidence::reporting::render(&v);
                    Ok(json!({"text":result.text,"payload":result.payload,"status":result.status}))
                }
                Some("native_view") => invarlock_native_evidence::reporting::load_native_report(
                    std::path::Path::new(v["path"].as_str().unwrap()),
                )
                .map_err(|e| e.to_string()),
                Some("judge_view") => invarlock_native_evidence::reporting::load_judge_report(
                    std::path::Path::new(v["path"].as_str().unwrap()),
                    v["case_ids"]
                        .as_array()
                        .map(|v| {
                            v.iter()
                                .map(|v| v.as_str().unwrap().to_owned())
                                .collect::<Vec<_>>()
                        })
                        .as_deref(),
                )
                .map_err(|e| e.to_string()),
                Some("set_view") => invarlock_native_evidence::reporting::sets::load(
                    std::path::Path::new(v["path"].as_str().unwrap()),
                    v["case_ids"]
                        .as_array()
                        .map(|v| {
                            v.iter()
                                .map(|v| v.as_str().unwrap().to_owned())
                                .collect::<Vec<_>>()
                        })
                        .as_deref(),
                )
                .map_err(|e| e.to_string()),
                Some("set_junit") => Ok(json!(String::from_utf8(
                    invarlock_native_evidence::reporting::sets::render_junit(&v["view"])
                )
                .unwrap())),
                Some("set_render") => {
                    let result = invarlock_native_evidence::reporting::sets::render(&v);
                    Ok(json!({"text":result.text,"payload":result.payload,"status":result.status}))
                }
                Some("captured_view") => {
                    invarlock_native_evidence::reporting::load_captured_report(
                        std::path::Path::new(v["path"].as_str().unwrap()),
                    )
                    .map_err(|e| e.to_string())
                }
                Some("comparison_view") => {
                    invarlock_native_evidence::reporting::build_comparison_view(&v["comparison"])
                        .map_err(|e| e.to_string())
                }
                Some("junit") => {
                    invarlock_native_evidence::reporting::render_junit(&v["comparison"])
                        .map(|v| json!(String::from_utf8(v).unwrap()))
                        .map_err(|e| e.to_string())
                }
                Some("html") => Ok(json!(invarlock_native_evidence::reporting::render_html(
                    &v["view"]
                ))),
                Some("markdown") => Ok(json!(
                    invarlock_native_evidence::reporting::render_markdown(
                        &v["view"],
                        v["include_details"].as_bool().unwrap_or(false)
                    )
                )),
                Some("write") => {
                    attestation::write_acceptance_attestation(&v).map_err(|e| e.to_string())
                }
                Some("artifact") => attestation_artifact::artifact_path_digest(
                    std::path::Path::new(v["path"].as_str().unwrap()),
                    v["digest_kind"].as_str().unwrap(),
                )
                .map(|s| json!(s))
                .map_err(|e| e.to_string()),
                _ => Ok(attestation::verify_acceptance_attestation(&v)),
            }
        })();
        let v = match result {
            Ok(v) => json!({"ok":v}),
            Err(e) => json!({"error":e}),
        };
        out.write_all(&canonical_json_no_newline(&v).unwrap())
            .unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
    }
}
