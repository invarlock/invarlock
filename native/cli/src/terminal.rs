//! Native terminal presentation. Machine output and report artifacts remain separate.
use invarlock_native_evidence::{self as evidence, EvidenceError, Result};
use serde_json::{json, Value};
use std::io::{self, Write};
fn text(v: &Value) -> String {
    evidence::reporting::terminal_text(&match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        _ => v.to_string(),
    })
}
fn line(s: impl AsRef<str>) {
    println!("{}", s.as_ref());
}
pub fn help(command: &str) {
    line(match command {
        "evaluate"=>"Usage: invarlock-native evaluate REQUEST [OPTIONS]\n\nOffline modes: captured, import, judge_import.\n  --signing-key PATH       Ed25519 key (or INVARLOCK_SIGNING_KEY)\n  --unsigned               Explicit unsigned local captured/judge evidence\n  --preflight              Validate without publishing\n  --output PATH            Override the evidence destination inside the request root\n  --request-root PATH      Resolve request references inside this directory\n  --baseline-run PATH      Override the captured baseline\n  --subject-run PATH       Override the captured subject\n  --max-bootstrap-draws N  Captured-analysis draw limit\n  --fail-on-policy         Exit 7 for a non-pass decision\n  --init DIRECTORY         Write a starter example\n  --example NAME           classification, extraction, judge, native-judge\n  --keygen DIRECTORY       Generate an Ed25519 key pair\n  --freeze-cases PATH      Freeze a planned case set\n  --case-set-output PATH   Write the frozen case set\n  --json                   Emit machine-readable JSON\n  --execution-backend rust (alias --backend rust)\n\nModel, SDK and custom-plugin execution: use invarlock-native with\n  --python-worker /absolute/python [--worker-timeout-seconds N].\nThe Python environment must contain InvarLock, its native extension and the required integrations.",
        "verify"=>"Usage: invarlock-native verify EVIDENCE [OPTIONS]\n\n  --trust-profile PATH\n  --receipt PATH\n  --verifier-signing-key PATH\n  --verifier-identity ID\n  --policy PATH\n  --expected-baseline-run DIGEST\n  --expected-subject-run DIGEST\n  --expected-baseline-artifact DIGEST\n  --expected-subject-artifact DIGEST\n  --expected-baseline-runtime DIGEST\n  --expected-subject-runtime DIGEST\n  --expected-schedule DIGEST\n  --expected-signer DIGEST\n  --expected-request-digest DIGEST\n  --max-bootstrap-draws N\n  --backend rust\n  --json\n\nExplicit pins and verifier credentials also accept the corresponding\nINVARLOCK_POLICY, INVARLOCK_EXPECTED_* and INVARLOCK_VERIFIER_* environment variables.",
        _=>"Usage: invarlock-native report EVIDENCE [OPTIONS]\n\n  --html PATH\n  --markdown PATH\n  --junit PATH\n  --explain\n  --case-id ID              Repeat to select cases\n  --package DIRECTORY      Build a complete handoff\n  --review-context PATH    Required with --package\n  --retained-receipt PATH\n  --backend rust\n  --json",
    });
}
pub fn native_backend(args: &mut Vec<String>) -> Result<()> {
    let mut i = 0;
    let mut seen = false;
    while i < args.len() {
        let key = args[i].split('=').next().unwrap();
        if ["--backend", "--execution-backend"].contains(&key) {
            if seen {
                return Err(EvidenceError::incomplete("duplicate backend option"));
            }
            seen = true;
            let value = if let Some((_, v)) = args[i].split_once('=') {
                let value = v.to_owned();
                args.remove(i);
                value
            } else {
                args.remove(i);
                if i >= args.len() {
                    return Err(EvidenceError::incomplete("backend requires a value"));
                }
                args.remove(i)
            };
            if value != "rust" {
                return Err(EvidenceError::incomplete("the native command requires backend rust; select --python-worker explicitly for compatibility"));
            }
        } else {
            i += 1;
        }
    }
    Ok(())
}
fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|p| p[1].as_str())
        .or_else(|| {
            args.iter()
                .find_map(|v| v.strip_prefix(&format!("{name}=")))
        })
}
pub fn public_error(args: &[String], error: &EvidenceError) -> (Value, u8) {
    let message = &error.message;
    let mut p = json!({"ok":false,"errors":[message]});
    match args[0].as_str() {
        "evaluate" => {
            if let Some(action) = ["init", "keygen", "freeze-cases"]
                .iter()
                .find(|k| option(args, &format!("--{k}")).is_some())
            {
                return (
                    json!({"format_version":"invarlock/evaluation-setup-v1","action":action.replace('-',"_"),"ok":false,"details":null,"errors":[message.chars().take(1024).collect::<String>()]}),
                    2,
                );
            }
            let mode = args
                .get(1)
                .filter(|s| !s.starts_with('-'))
                .and_then(|path| {
                    invarlock_native_filesystem::read_file(std::path::Path::new(path), 1024 * 1024)
                        .ok()
                })
                .and_then(|raw| crate::request::parse(&raw).ok())
                .map(|v| v["execution"]["mode"].clone())
                .unwrap_or(Value::Null);
            let preflight = args.iter().any(|s| s == "--preflight");
            p["format_version"] = json!(if mode == "judge_import" || mode == "judge_collect" {
                "invarlock/judge-evaluation-result-v1"
            } else if mode == "captured" {
                if preflight {
                    "invarlock/evaluation-preflight-v3"
                } else {
                    "invarlock/evaluation-result-v2"
                }
            } else if preflight {
                "invarlock/evaluation-preflight-v2"
            } else {
                "invarlock/evaluation-result-v1"
            });
            if mode == "captured" {
                p["kind"] = json!("captured");
                p["errors"] = json!([message.chars().take(1024).collect::<String>()]);
                if preflight {
                    p["execution_mode"] = json!("captured");
                    p["requested_authentication"] =
                        json!(if args.iter().any(|v| v == "--unsigned") {
                            "unsigned_local"
                        } else {
                            "signed"
                        });
                }
            } else if mode == "judge_import" || mode == "judge_collect" {
                p["kind"] = json!("judge");
                p["errors"] = json!([message.chars().take(1024).collect::<String>()]);
            }
            (p, 2)
        }
        "report" => {
            if ["--package", "--review-context", "--retained-receipt"]
                .iter()
                .any(|k| option(args, k).is_some())
            {
                p["format_version"] = json!("invarlock/client-review-package-v1");
            } else {
                p["format_version"] = json!("invarlock/evidence-report-v2");
                p["kind"] = Value::Null;
                p["pack_manifest_digest"] = Value::Null;
                p["requested_outputs"] = json!({});
                for k in ["html", "markdown", "junit"] {
                    if let Some(v) = option(args, &format!("--{k}")) {
                        p["requested_outputs"][k] = json!(v);
                    }
                }
                p["written_outputs"] = json!({});
                p["failed_output"] = Value::Null;
            }
            (p, 2)
        }
        _ => {
            let captured = args
                .get(1)
                .filter(|p| !p.starts_with('-'))
                .and_then(|path| {
                    invarlock_native_filesystem::read_file(
                        &std::path::Path::new(path).join("manifest.json"),
                        256 * 1024,
                    )
                    .ok()
                })
                .and_then(|raw| invarlock_native_contracts::parse_json(&raw, "manifest").ok())
                .is_some_and(|m| m["kind"] == "captured");
            if captured {
                p = json!({"format_version":"invarlock/evidence-pack-verify-v2","kind":"captured","ok":false,"integrity_ok":if error.receipt_written {json!(false)} else {Value::Null},"policy_verdict":null,"decision":null,"scoring_assurance":null,"signed_receipt":if error.receipt_written {option(args,"--receipt").and_then(|p|std::path::Path::new(p).file_name()).and_then(|p|p.to_str())} else {None},"errors":[message.chars().take(1024).collect::<String>()]});
            } else {
                p["format_version"] = json!("invarlock/evidence-verification-error-v1");
                p["warnings"] = json!([]);
            }
            (p, error.exit_code)
        }
    }
}
fn metrics(rows: &Value) {
    if let Some(rows) = rows.as_array() {
        for row in rows {
            line(format!(
                "{} / {}: {}; {} usable pairs; {} missing results",
                text(&row["name"]),
                text(&row["slice"]),
                text(&row["decision"]),
                text(&row["usable_count"]),
                text(&row["missing_count"])
            ));
            if let Some(reasons) = row["reasons"].as_array().filter(|r| !r.is_empty()) {
                line(format!(
                    "Recorded reasons: {}",
                    reasons.iter().map(text).collect::<Vec<_>>().join("; ")
                ));
            }
        }
    }
}
pub fn render(args: &[String], result: &evidence::transaction::Outcome) -> Result<()> {
    let command = args[0].as_str();
    let p = &result.payload;
    if command == "report" {
        if p["ok"] == false {
            if let Some(errors) = p["errors"].as_array() {
                for e in errors {
                    line(format!("FAIL {}", text(e)));
                }
            }
            if let Some(outputs) = p["written_outputs"].as_object() {
                for (k, v) in outputs {
                    line(format!("Written {k}: {}", text(v)));
                }
            }
            if !p["failed_output"].is_null() {
                line(format!("Failed output: {}", text(&p["failed_output"])));
            }
        } else if let Some(body) = &result.terminal {
            for l in body.lines() {
                line(evidence::reporting::terminal_text(l));
            }
            if let Some(outputs) = p["written_outputs"].as_object() {
                for (k, v) in outputs {
                    line(format!("{} {}", k.to_uppercase(), text(v)));
                }
            }
        } else if p["ok"] == true {
            line(format!("Client handoff: {}", text(&p["output"])));
            line("Recipient verification has not been performed by packaging.");
        } else if let Some(errors) = p["errors"].as_array() {
            for e in errors {
                line(format!("FAIL {}", text(e)));
            }
        }
    } else if command == "evaluate" {
        if p["format_version"] == "invarlock/evaluation-setup-v1" {
            if let Some(details) = p["details"].as_object() {
                for (k, v) in details {
                    line(format!("{k}: {}", text(v)));
                }
            } else if let Some(errors) = p["errors"].as_array() {
                for e in errors {
                    line(format!("FAIL {}", text(e)));
                }
            }
        } else if p.get("ready").is_some() && p["kind"] == "judge" {
            line(if p["ready"] == true {
                "Judge preflight complete"
            } else {
                "Judge preflight needs inputs"
            });
            line(format!(
                "Cases: {}; independent units: {}; planned trials: {}; maximum attempts: {}",
                text(&p["cases"]),
                text(&p["independent_units"]),
                text(&p["planned_trials"]),
                text(&p["maximum_attempts"])
            ));
            if p["judge"].is_object() {
                line(format!("Judge: {}", text(&p["judge"]["requested_model"])));
            }
            let q = &p["precision"];
            if q.is_object() {
                if q["minimum_units_met"] == false {
                    line(format!(
                        "Sample count: {} independent units; policy requires at least {}.",
                        text(&q["independent_units"]),
                        text(&q["minimum_units"])
                    ));
                }
                for (k, label) in [
                    ("paired_effect", "Paired effect"),
                    ("subject_score", "Subject score"),
                ] {
                    let r = &q[k];
                    if r["role"] == "descriptive" {
                        continue;
                    }
                    let status = match r["status"].as_str() {
                        Some("within_limit") => "meets the width limit for all scores",
                        Some("unattainable") => "cannot meet the width limit",
                        _ => "width limit is not guaranteed",
                    };
                    line(format!(
                        "{label} precision: {status}; width <= {}; policy limit {}.",
                        text(&r["maximum_width_upper_bound"]),
                        text(&q["maximum_interval_width"])
                    ));
                    if r["units_for_guaranteed_width"].is_null() {
                        line(format!(
                            "No guarantee within the {}-unit plan limit.",
                            text(&q["unit_search_limit"])
                        ));
                    } else {
                        line(format!(
                            "Independent units for guaranteed width: {}.",
                            text(&r["units_for_guaranteed_width"])
                        ));
                    }
                }
                line("Precision assumes complete collection and independent units; it does not predict acceptance.");
            }
            if let Some(errors) = p["errors"].as_array() {
                for e in errors {
                    line(text(e));
                }
            }
            line("No model calls, signing, or publication were performed.");
        } else if p.get("checks").is_some() || p.get("requested_authentication").is_some() {
            line("Preflight complete");
            line(format!(
                "Mode: {}; paired records: {}",
                text(&p["execution_mode"]),
                text(&p["record_count"])
            ));
            if p["kind"] == "captured" {
                line(format!(
                    "Requested authentication: {}",
                    text(&p["requested_authentication"])
                ));
                line("No execution, signing, scoring, or publication was performed");
            } else {
                line(format!("Evidence destination: {}", text(&p["output"])));
                line(format!(
                    "Validated checks: {}",
                    p["checks"].as_array().map_or(0, Vec::len)
                ));
                line("No execution or publication was performed");
                line("Next: run the same evaluate command without --preflight.");
            }
        } else {
            let judge = p["kind"] == "judge";
            let captured = p["kind"] == "captured";
            line(if judge {
                "Bounded judge evidence created"
            } else if captured {
                "Captured evidence created"
            } else {
                "Evidence created"
            });
            line(format!(
                "Recorded policy result: {}",
                text(
                    p.get("policy_verdict")
                        .or_else(|| p.get("decision"))
                        .unwrap_or(&Value::String(
                            result
                                .terminal
                                .clone()
                                .unwrap_or_else(|| "unavailable".into())
                        ))
                )
            ));
            if judge {
                line(format!("Authentication: {}", text(&p["authentication"])));
            } else if captured {
                line(if p["authentication"] == "signed" {
                    "Signing: Signed evidence"
                } else {
                    "Signing: Unsigned local evidence"
                });
            }
            line(if captured || judge {
                "Independent verification: not performed"
            } else {
                "Recipient verification: not performed"
            });
            if captured {
                if let Some(extra) = &result.terminal {
                    line(extra);
                }
            }
            line(format!("Evidence: {}", text(&p["evidence"])));
            if !captured && !judge {
                line("Next: verify with independently approved trust inputs; use report to inspect the recorded checks.");
            }
        }
    } else {
        if p["ok"] == true {
            line(match p["kind"].as_str() {
                Some("captured") => "PASS Independent captured verification complete",
                Some("judge") => "PASS Bounded judge recipient verification complete",
                Some("evidence_set") => "PASS Evidence set recipient verification complete",
                _ => "PASS Independent verification complete",
            });
        } else {
            line("FAIL Recipient verification did not accept the evidence");
        }
        if p["kind"] == "judge" || p["kind"] == "evidence_set" {
            line(format!(
                "Authenticated: {}; replayed: {}; accepted: {}",
                text(&p["authenticated"]),
                text(&p["replayed"]),
                text(&p["accepted"])
            ));
            line(format!("Policy result: {}", text(&p["decision"])));
        } else {
            if p["integrity_ok"] == true || p["ok"] == true {
                line("Evidence integrity: verified");
            }
            if !p["policy_verdict"].is_null() {
                line(format!("Policy result: {}", text(&p["policy_verdict"])));
            }
        }
        if p["evidence"].is_null() {
            if let Some(input) = args.get(1).filter(|v| !v.starts_with('-')) {
                line(format!(
                    "Evidence: {}",
                    evidence::reporting::terminal_text(input)
                ));
            }
        }
        if p["receipt"].is_null() {
            if let Some(receipt) = option(args, "--receipt") {
                line(format!(
                    "Receipt: {}",
                    evidence::reporting::terminal_text(receipt)
                ));
            }
        }
        for (key, label) in [
            ("evidence", "Evidence"),
            ("comparison_id", "Comparison"),
            ("signer_fingerprint", "Evidence signer"),
            ("verifier_fingerprint", "Verifier signer"),
            ("receipt", "Receipt"),
        ] {
            if !p[key].is_null() {
                line(format!("{label}: {}", text(&p[key])));
            }
        }
        if p["kind"] == "captured" {
            line(format!("Recorded decision: {}", text(&p["decision"])));
            metrics(&p["metric_summaries"]);
        }
        if let Some(errors) = p["errors"].as_array() {
            for e in errors {
                line(text(e));
            }
        }
    }
    io::stdout()
        .flush()
        .map_err(|_| EvidenceError::incomplete("output_io_error"))
}
