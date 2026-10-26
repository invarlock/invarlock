//! Read, render, and publish from one family snapshot.
use super::publication::{self, PublicationError, ReportOutput};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
fn exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}
pub fn render(config: &Value) -> ReportOutput {
    let evidence = Path::new(config["evidence_path"].as_str().unwrap_or(""));
    if exists(&evidence.join("evidence-set.json")) {
        return super::sets::render(config);
    }
    let judge = exists(&evidence.join("envelope.json"));
    let requested = [
        ("html", "html_path"),
        ("markdown", "markdown_path"),
        ("junit", "junit_path"),
    ]
    .into_iter()
    .filter_map(|(name, key)| {
        config[key]
            .as_str()
            .map(|p| (name.to_owned(), PathBuf::from(p)))
    })
    .collect::<Vec<_>>();
    let requested_value = requested
        .iter()
        .map(|(name, p)| (name.clone(), json!(p.to_string_lossy())))
        .collect::<Map<_, _>>();
    let mut payload = if judge {
        json!({"format_version":"invarlock/judge-evidence-report-v1","kind":"judge","ok":false,"errors":[],"requested_outputs":requested_value,"written_outputs":{},"failed_output":null})
    } else {
        json!({"format_version":"invarlock/evidence-report-v2","kind":"runtime","ok":false,"pack_manifest_digest":null,"requested_outputs":requested_value,"written_outputs":{},"failed_output":null,"errors":[]})
    };
    let result = (|| -> std::result::Result<String, PublicationError> {
        if !std::fs::symlink_metadata(evidence)
            .is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
        {
            return Err(PublicationError::new("evidence must be a real directory"));
        }
        let ids = config["case_ids"]
            .as_array()
            .map(|v| {
                v.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !judge && !ids.is_empty() {
            return Err(PublicationError::new(
                "--case-id is available only for judge evidence and evidence sets",
            ));
        }
        let captured = if judge {
            false
        } else {
            let raw=invarlock_native_filesystem::read_file(&evidence.join("manifest.json"),256*1024).map_err(|_|PublicationError::new("evidence manifest schema failed: evidence manifest could not be identified safely"))?;
            let manifest=invarlock_native_contracts::parse_json(&raw,"evidence manifest").map_err(|_|PublicationError::new("evidence manifest schema failed: evidence manifest could not be identified safely"))?;
            if manifest["format"] == "invarlock/evidence-pack-v1" {
                false
            } else if manifest["format"] == "invarlock/evidence-pack-v2"
                && manifest["kind"] == "captured"
            {
                true
            } else {
                return Err(PublicationError::new("evidence manifest schema failed: evidence manifest format or kind is unsupported"));
            }
        };
        if !judge {
            payload["kind"] = json!(if captured { "captured" } else { "runtime" })
        }
        publication::validate(&requested, Some(evidence))?;
        let loaded = if judge {
            super::judge_view::load(evidence, Some(&ids))
        } else if captured {
            super::captured_view::load(evidence)
        } else {
            super::native_load::load(evidence)
        }
        .map_err(|e| PublicationError::new(e.message))?;
        let view = &loaded["view"];
        if judge {
            payload["evidence_digest"] = json!(crate::judge::object_digest(&loaded["envelope"])
                .map_err(|e| PublicationError::new(e.message))?)
        } else {
            payload["pack_manifest_digest"] = if captured {
                json!(crate::digest(&loaded["manifest"])
                    .map_err(|e| PublicationError::new(e.message))?)
            } else {
                loaded["manifest_digest"].clone()
            }
        }
        let text = super::render_markdown(view, config["explain"].as_bool().unwrap_or(false));
        let outputs = requested
            .iter()
            .map(|(name, _)| match name.as_str() {
                "html" => super::render_html(view).into_bytes(),
                "markdown" => text.as_bytes().to_vec(),
                "junit" => {
                    if judge {
                        super::junit::judge(view, &loaded["facts"])
                    } else {
                        super::junit::recorded(view)
                    }
                }
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        payload["written_outputs"] =
            Value::Object(publication::publish(&requested, &outputs, Some(evidence))?);
        if judge {
            payload.as_object_mut().unwrap().remove("failed_output");
            if let Some(facts) = loaded["facts"].as_object() {
                payload.as_object_mut().unwrap().extend(facts.clone());
            }
        }
        payload["ok"] = json!(true);
        Ok(text)
    })();
    match result {
        Ok(text) => ReportOutput {
            text,
            payload,
            status: 0,
        },
        Err(error) => {
            payload["written_outputs"] = Value::Object(error.written_outputs);
            payload["failed_output"] = json!(error.failed_output);
            payload["errors"] = json!([error.message.chars().take(1024).collect::<String>()]);
            if judge {
                payload.as_object_mut().unwrap().remove("evidence_digest");
            }
            ReportOutput {
                text: String::new(),
                payload,
                status: 2,
            }
        }
    }
}
