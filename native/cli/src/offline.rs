#[path = "judge_import.rs"]
mod judge_import;
#[path = "runtime_import.rs"]
mod runtime_import;
#[path = "setup.rs"]
mod setup;
// Offline user commands. No Python interpreter or automatic compatibility fallback.
use invarlock_native_contracts as contracts;
use invarlock_native_evidence::{self as evidence, EvidenceError, Result};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::incomplete(s)
}
fn absolute(s: &str) -> Result<PathBuf> {
    std::path::absolute(s).map_err(|e| err(e.to_string()))
}
#[derive(Default)]
struct Args {
    values: BTreeMap<String, Vec<String>>,
    positional: Vec<String>,
}
impl Args {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Self> {
        let mut a = Self::default();
        let mut i = 0;
        while i < args.len() {
            let item = &args[i];
            if let Some(key) = item.strip_prefix("--") {
                let (key, inline) = key
                    .split_once('=')
                    .map_or((key, None), |(k, v)| (k, Some(v)));
                if !allowed.contains(&key) {
                    return Err(err(format!("unsupported option --{key}")));
                }
                let flag =
                    ["json", "explain", "unsigned", "preflight", "fail-on-policy"].contains(&key);
                let val = if flag {
                    if inline.is_some() {
                        return Err(err(format!("--{key} takes no value")));
                    }
                    "true".into()
                } else if let Some(v) = inline {
                    v.to_owned()
                } else {
                    i += 1;
                    args.get(i)
                        .filter(|s| !s.starts_with("--"))
                        .ok_or_else(|| err(format!("missing value for --{key}")))?
                        .clone()
                };
                if a.values.contains_key(key) && key != "case-id" {
                    return Err(err(format!("duplicate option --{key}")));
                }
                a.values.entry(key.into()).or_default().push(val);
            } else {
                a.positional.push(item.clone());
            }
            i += 1;
        }
        for (key, variable) in [
            ("signing-key", "INVARLOCK_SIGNING_KEY"),
            ("policy", "INVARLOCK_POLICY"),
            ("verifier-signing-key", "INVARLOCK_VERIFIER_SIGNING_KEY"),
            ("verifier-identity", "INVARLOCK_VERIFIER_IDENTITY"),
            (
                "expected-baseline-artifact",
                "INVARLOCK_EXPECTED_BASELINE_ARTIFACT",
            ),
            (
                "expected-subject-artifact",
                "INVARLOCK_EXPECTED_SUBJECT_ARTIFACT",
            ),
            ("expected-baseline-run", "INVARLOCK_EXPECTED_BASELINE_RUN"),
            ("expected-subject-run", "INVARLOCK_EXPECTED_SUBJECT_RUN"),
            (
                "expected-baseline-runtime",
                "INVARLOCK_EXPECTED_BASELINE_RUNTIME",
            ),
            (
                "expected-subject-runtime",
                "INVARLOCK_EXPECTED_SUBJECT_RUNTIME",
            ),
            ("expected-schedule", "INVARLOCK_EXPECTED_SCHEDULE"),
            ("expected-signer", "INVARLOCK_EXPECTED_SIGNER"),
            (
                "expected-request-digest",
                "INVARLOCK_EXPECTED_REQUEST_DIGEST",
            ),
        ] {
            if allowed.contains(&key) && !a.values.contains_key(key) {
                if let Ok(value) = std::env::var(variable) {
                    if !value.is_empty() {
                        a.values.insert(key.into(), vec![value]);
                    }
                }
            }
        }
        Ok(a)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.values
            .get(k)
            .and_then(|v| v.first())
            .map(String::as_str)
    }
    fn flag(&self, k: &str) -> bool {
        self.get(k) == Some("true")
    }
    fn path(&self, k: &str) -> Result<Value> {
        self.get(k)
            .map(|s| absolute(s).map(|p| json!(p)))
            .transpose()
            .map(|v| v.unwrap_or(Value::Null))
    }
    fn input(&self) -> Result<PathBuf> {
        if self.positional.len() != 1 {
            return Err(err("exactly one input path is required"));
        }
        absolute(&self.positional[0])
    }
}
fn outcome(payload: Value, status: u8) -> Result<evidence::transaction::Outcome> {
    Ok(evidence::transaction::Outcome {
        payload,
        status,
        verified_report: None,
        terminal: None,
    })
}
pub fn run(command: &str, args: &[String]) -> Result<evidence::transaction::Outcome> {
    match command {
        "report" => {
            let a = Args::parse(
                args,
                &[
                    "html",
                    "markdown",
                    "junit",
                    "explain",
                    "case-id",
                    "json",
                    "package",
                    "review-context",
                    "retained-receipt",
                ],
            )?;
            if a.get("package").is_some()
                || a.get("review-context").is_some()
                || a.get("retained-receipt").is_some()
            {
                if a.get("package").is_none() || a.get("review-context").is_none() {
                    return Err(err(
                        "--review-context and --package must be supplied together",
                    ));
                }
                if ["html", "markdown", "junit", "explain", "case-id"]
                    .iter()
                    .any(|k| a.get(k).is_some())
                {
                    return Err(err("--package cannot be combined with individual report outputs, --explain or --case-id"));
                }
                return outcome(
                    evidence::reporting::handoff::build(
                        &json!({"evidence_path":a.input()?,"output":a.path("package")?,"context_path":a.path("review-context")?,"retained_receipt":a.path("retained-receipt")?}),
                    )?,
                    0,
                );
            }
            let config = json!({"evidence_path":a.input()?,"html_path":a.path("html")?,"markdown_path":a.path("markdown")?,"junit_path":a.path("junit")?,"explain":a.flag("explain"),"case_ids":a.values.get("case-id").cloned().unwrap_or_default()});
            let result = evidence::reporting::render(&config);
            let mut output = outcome(result.payload, result.status)?;
            output.terminal = Some(result.text);
            Ok(output)
        }
        "verify" => {
            let a = Args::parse(
                args,
                &[
                    "trust-profile",
                    "receipt",
                    "verifier-signing-key",
                    "verifier-identity",
                    "max-bootstrap-draws",
                    "json",
                    "policy",
                    "expected-baseline-run",
                    "expected-subject-run",
                    "expected-baseline-artifact",
                    "expected-subject-artifact",
                    "expected-baseline-runtime",
                    "expected-subject-runtime",
                    "expected-schedule",
                    "expected-signer",
                    "expected-request-digest",
                ],
            )?;
            let input = a.input()?;
            if !std::fs::symlink_metadata(&input)
                .is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
            {
                return Err(err("evidence must be a real directory"));
            }
            let set = input.join("evidence-set.json").exists();
            let judge = input.join("envelope.json").exists();
            let trust = a.path("trust-profile")?;
            let mut config = json!({"evidence_path":input,"trust_path":trust,"recipient_policy_path":trust,"receipt_path":a.path("receipt")?,"verifier_signing_key_path":a.path("verifier-signing-key")?,"verifier_identity":a.get("verifier-identity")});
            let pins = [
                "policy",
                "expected-baseline-run",
                "expected-subject-run",
                "expected-baseline-artifact",
                "expected-subject-artifact",
                "expected-baseline-runtime",
                "expected-subject-runtime",
                "expected-schedule",
                "expected-signer",
                "expected-request-digest",
            ];
            if !trust.is_null() && pins.iter().any(|k| a.get(k).is_some()) {
                return Err(err(
                    "--trust-profile cannot be mixed with explicit trust anchors",
                ));
            }
            if !trust.is_null()
                && !judge
                && ["verifier-signing-key", "verifier-identity"]
                    .iter()
                    .any(|k| a.get(k).is_some())
            {
                return Err(err("--trust-profile cannot be mixed with verifier overrides for this evidence family"));
            }
            for key in pins {
                if let Some(v) = a.get(key) {
                    let field = if key == "policy" {
                        "policy_path".into()
                    } else {
                        key.replace('-', "_")
                    };
                    config[&field] = if key == "policy" {
                        a.path(key)?
                    } else {
                        json!(v)
                    };
                }
            }
            if let Some(s) = a.get("max-bootstrap-draws") {
                config["max_bootstrap_draws"] = json!(s
                    .parse::<u64>()
                    .map_err(|_| err("max-bootstrap-draws must be a non-negative integer"))?);
            }
            if judge {
                if trust.is_null() {
                    return Err(err("Judge verification requires --trust-profile"));
                }
                if a.get("receipt").is_some()
                    && (a.get("verifier-signing-key").is_none()
                        || a.get("verifier-identity").is_none())
                {
                    return Err(err(
                        "judge receipts require verifier signing key and identity",
                    ));
                }
                let result = evidence::judge::verify(&config)?;
                let status = if result["accepted"] == true {
                    0
                } else if result["verified"] == true {
                    7
                } else {
                    4
                };
                if a.get("receipt").is_some() {
                    config["result"] = result.clone();
                    evidence::judge::write_receipt_with_key(&config, &result, None)?;
                }
                let mut payload = result;
                let format = payload.as_object_mut().unwrap().remove("format").unwrap();
                payload["format_version"] = format;
                payload["kind"] = json!("judge");
                payload["ok"] = payload["accepted"].clone();
                if a.get("receipt").is_some() {
                    payload["receipt"] = config["receipt_path"].clone();
                }
                return outcome(payload, status);
            }
            let command = if set {
                "verify-evidence-set"
            } else if !trust.is_null() {
                "verify-trust"
            } else {
                let raw = fs::read_file(&input.join("manifest.json"), 256 * 1024)
                    .map_err(|e| err(e.to_string()))?;
                let manifest = contracts::parse_json(&raw, "evidence manifest")?;
                if manifest["kind"] == "captured" {
                    "verify-captured"
                } else {
                    "verify-native"
                }
            };
            evidence::transaction::dispatch(command, &config, None, None, None)
        }
        "evaluate" => {
            let a = Args::parse(
                args,
                &[
                    "init",
                    "example",
                    "keygen",
                    "freeze-cases",
                    "case-set-output",
                    "signing-key",
                    "unsigned",
                    "preflight",
                    "fail-on-policy",
                    "max-bootstrap-draws",
                    "output",
                    "request-root",
                    "baseline-run",
                    "subject-run",
                    "json",
                ],
            )?;
            if ["init", "keygen", "freeze-cases"]
                .iter()
                .any(|k| a.get(k).is_some())
            {
                return setup::run(&a);
            }
            let input = a.input()?;
            let raw = fs::read_file(&input, 1024 * 1024).map_err(|e| err(e.to_string()))?;
            let mut request = crate::request::parse(&raw)?;
            let root = if let Some(s) = a.get("request-root") {
                absolute(s)?
            } else {
                input.parent().unwrap().to_path_buf()
            };
            contracts::request::validate_authored(&request)?;
            if request["format_version"] == "invarlock/evaluation-request-v3" {
                return judge_import::run(&a, &request, &root);
            }
            if request["comparison"]["metric"] == "judge" {
                return Err(err("configured judge collection requires the explicit compatibility integration; retained judge evidence supports standalone verify and report"));
            }
            for (flag, side) in [("baseline-run", "baseline"), ("subject-run", "subject")] {
                if let Some(s) = a.get(flag) {
                    let path = absolute(s)?;
                    request["comparison"][side]["path"] = json!(path
                        .strip_prefix(&root)
                        .map_err(|_| err("run override must be inside request root"))?);
                }
            }
            if let Some(s) = a.get("output") {
                let path = absolute(s)?;
                request["output"]["evidence"] = json!(path
                    .strip_prefix(&root)
                    .map_err(|_| err("output must be inside request root"))?);
            }
            if request["execution"]["mode"] != "captured" {
                return runtime_import::run(&a, &request, &root);
            }
            for side in ["baseline", "subject"] {
                if request["comparison"][side].get("adapter").is_none() {
                    request["comparison"][side]["adapter"] = json!("invarlock");
                }
            }
            let budget = a
                .get("max-bootstrap-draws")
                .map(|s| {
                    s.parse::<u64>()
                        .map_err(|_| err("max-bootstrap-draws must be a non-negative integer"))
                })
                .transpose()?
                .unwrap_or(102_400_000);
            let key = a.path("signing-key")?;
            let mut config = json!({"authored_request":request,"root":root,"unsigned":a.flag("unsigned"),"max_bootstrap_draws":budget});
            if !key.is_null() {
                config["signing_key_path"] = key.clone();
            }
            let mut prepared = evidence::preparation::prepare_captured(&config)?;
            let preflight = prepared["preflight"].clone();
            if a.flag("preflight") {
                return outcome(preflight, 0);
            }
            prepared["destination"] = preflight["output"].clone();
            prepared["unsigned"] = json!(a.flag("unsigned"));
            prepared["max_bootstrap_draws"] = json!(budget);
            if !key.is_null() {
                prepared["signing_key_path"] = key;
            }
            let published = evidence::captured::publish(&prepared)?;
            let manifest = &published["manifest"];
            let verdict = &published["comparison"]["decision"];
            let mut result = outcome(
                json!({"format_version":"invarlock/evaluation-result-v2","kind":"captured","ok":true,"evidence":preflight["output"],"comparison_id":manifest["comparison_id"],"baseline_run_digest":preflight["baseline_run_digest"],"subject_run_digest":preflight["subject_run_digest"],"policy_digest":preflight["policy_digest"],"authentication":manifest["authentication"],"policy_verdict":verdict,"decision":verdict,"pack_manifest_digest":published["pack_manifest_digest"],"request_digest":preflight["request_digest"]}),
                if a.flag("fail-on-policy") && verdict != "pass" {
                    7
                } else {
                    0
                },
            )?;
            if !a.flag("json") {
                let mut lines = Vec::new();
                for metric in published["comparison"]["metrics"].as_array().unwrap() {
                    let missing = metric["missing_ids"].as_array().map_or(0, Vec::len);
                    let safe =
                        |v: &Value| evidence::reporting::terminal_text(v.as_str().unwrap_or(""));
                    lines.push(format!(
                        "{} / {}: {}; {} usable pairs; {} missing results",
                        safe(&metric["name"]),
                        safe(&metric["slice"]),
                        safe(&metric["decision"]),
                        metric["count"]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_sub(missing as u64),
                        missing
                    ));
                    if let Some(reasons) = metric["reasons"].as_array().filter(|r| !r.is_empty()) {
                        lines.push(format!(
                            "Recorded reasons: {}",
                            reasons.iter().map(safe).collect::<Vec<_>>().join("; ")
                        ));
                    }
                }
                result.terminal = Some(lines.join("\n"));
            }
            Ok(result)
        }
        _ => Err(err("unsupported standalone workflow")),
    }
}
