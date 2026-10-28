//! One command dispatch boundary shared by the Python adapter and offline CLI.
use crate::{
    captured, judge, judge_native, native_pack, required_string, trust_inputs, EvidenceError,
    Result,
};
use invarlock_native_contracts as contracts;
use invarlock_native_core as analysis;
use serde_json::{json, Value};
use std::path::Path;

pub struct Outcome {
    pub payload: Value,
    pub status: u8,
    pub verified_report: Option<Value>,
    pub terminal: Option<String>,
}
pub const COMMANDS: &[&str] = &[
    "report",
    "client-review",
    "import-evaluation-run",
    "gguf-identity",
    "canonicalize",
    "validate",
    "prepare-captured",
    "prepare-runtime-inputs",
    "normalize-runtime-request",
    "validate-request",
    "request-mode",
    "request-reference",
    "prepare-schedule",
    "compare",
    "score",
    "verify-captured",
    "publish-captured",
    "verify-captured-receipt",
    "verify-native",
    "verify-native-receipt",
    "verify-judge",
    "replay-judge",
    "publish-judge",
    "verify-judge-receipt",
    "write-judge-receipt",
    "validate-native-capture",
    "finalize-native-judge",
    "prepare-native-judge",
    "verify-stored-judge",
    "replay-judge-receipt",
    "load-trust",
    "verify-trust",
    "verify-evidence-set",
    "verify-stored-evidence-set",
    "write-evidence-set-index",
    "qualify-export",
    "write-attestation",
    "verify-attestation",
    "publish-native",
    "write-runtime-side",
    "load-runtime-side",
    "write-runtime-paired",
    "derive-runtime-transaction",
];
pub fn dispatch(
    command: &str,
    config: &Value,
    scorer: Option<&dyn native_pack::ScorerReplay>,
    policy_bytes: Option<&[u8]>,
    key_bytes: Option<&[u8]>,
) -> Result<Outcome> {
    let value = match command {
        "import-evaluation-run" => {
            let raw = crate::decode_base64(required_string(config, "raw_base64")?)
                .map_err(|_| EvidenceError::contract("invalid export base64"))?;
            crate::imports::parse(&raw, &config["options"])?
        }
        "client-review" => crate::reporting::handoff::build(config)?,
        "report" => {
            let output = crate::reporting::render(config);
            return Ok(Outcome {
                payload: json!({"text":output.text,"report":output.payload}),
                status: output.status,
                verified_report: None,
                terminal: None,
            });
        }
        "gguf-identity" => crate::gguf::dispatch(config)?,
        "canonicalize" => config.clone(),
        "prepare-runtime-inputs" => crate::preparation::prepare_runtime_inputs(config)?,
        "normalize-runtime-request" => crate::preparation::normalize_runtime_request(config)?,
        "prepare-captured" => crate::preparation::prepare_captured(config)?,
        "validate-request" => {
            if config["schema_only"] == true {
                contracts::request::validate_schema(&config["value"], config["captured"] == true)?;
            } else {
                contracts::request::validate_authored(&config["value"])?;
            }
            config["value"].clone()
        }
        "request-mode" => json!(contracts::request::mode(&config["value"])?),
        "request-reference" => json!(contracts::request::reference_parts(
            required_string(config, "reference")?,
            config["label"].as_str().unwrap_or("reference")
        )?),
        "prepare-schedule" => {
            let raw = if let Some(encoded) = config["payload_base64"].as_str() {
                crate::decode_base64(encoded).map_err(|e| EvidenceError::contract(e.to_string()))?
            } else {
                config["raw"].as_str().unwrap_or("").as_bytes().to_vec()
            };
            let task = config["task"].as_str().unwrap_or("text_causal");
            let schedule = match config["action"].as_str().unwrap_or("build") {
                "prepare" => contracts::schedule::prepare_local(&config["source"], &raw, task)?,
                "parse" => contracts::schedule::Schedule::parse(&raw)?,
                "material" => contracts::schedule::Schedule::from_material(
                    &config["identity"],
                    &config["records"],
                    task,
                )?,
                _ => contracts::schedule::Schedule::build(&config["value"])?,
            };
            let mut output = json!({"value":schedule.value(),"canonical":String::from_utf8(schedule.bytes().to_vec()).unwrap(),"digest":schedule.digest()});
            if config.get("descriptor").is_some() {
                contracts::schedule::validate_preparation(&config["descriptor"], &schedule)?;
            }
            if config["make_descriptor"] == true {
                output["descriptor"] =
                    contracts::schedule::preparation_descriptor(&config["source"], &schedule)?;
            }
            output
        }
        "validate" => {
            contracts::validate_schema(&config["value"], required_string(config, "schema")?)?;
            json!({"valid":true})
        }
        "compare" => analysis::compare_runs(
            &config["baseline"],
            &config["subject"],
            &config["policy"],
            contracts::bootstrap_budget(
                config.get("max_bootstrap_draws"),
                analysis::DEFAULT_MAX_BOOTSTRAP_DRAWS,
            )
            .map_err(|e| EvidenceError::incomplete(e.to_string()))?,
        )
        .map_err(|e| {
            EvidenceError::new(
                if e.reason == "local_work_budget_exceeded" {
                    2
                } else {
                    4
                },
                e.to_string(),
            )
        })?,
        "score" => json!(analysis::score(
            required_string(config, "kind")?,
            &config["expected"],
            &config["actual"],
            &config["configuration"]
        )
        .map_err(|e| EvidenceError::contract(e.to_string()))?),
        "publish-native" => crate::native_publication::publish_with_key(config, scorer, key_bytes)?,
        "write-runtime-side" => crate::native_publication::write_runtime_side(config)?,
        "load-runtime-side" => crate::native_publication::load_runtime_side(config)?,
        "derive-runtime-transaction" => {
            crate::native_publication::derive_transaction(config, scorer)?
        }
        "write-runtime-paired" => crate::native_publication::write_paired(config, scorer)?,
        "write-attestation" => crate::attestation::write_acceptance_attestation(config)?,
        "verify-attestation" => crate::attestation::verify_acceptance_attestation(config),
        "qualify-export" => crate::qualification::qualify(config)?,
        "verify-evidence-set" => crate::evidence_sets::verify(config)?,
        "verify-stored-evidence-set" => crate::evidence_sets::verify_stored(config)?,
        "write-evidence-set-index" => crate::evidence_sets::write_index(config)?,
        "verify-captured" => captured::verify_with_inputs(config, policy_bytes, key_bytes)?,
        "publish-captured" => captured::publish(config)?,
        "verify-captured-receipt" => captured::verify_receipt(config)?,
        "verify-native" => {
            let result = match (policy_bytes, key_bytes) {
                (None, None) => native_pack::verify_transaction(config, scorer)?,
                _ => {
                    let policy_disk;
                    let key_disk;
                    let policy = match policy_bytes {
                        Some(raw) => raw,
                        None => {
                            policy_disk = invarlock_native_filesystem::read_file(
                                Path::new(required_string(config, "policy_path")?),
                                4 * 1024 * 1024,
                            )
                            .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
                            &policy_disk
                        }
                    };
                    let key = match key_bytes {
                        Some(raw) => raw,
                        None => {
                            key_disk = invarlock_native_filesystem::read_file(
                                Path::new(required_string(config, "verifier_signing_key_path")?),
                                65536,
                            )
                            .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
                            &key_disk
                        }
                    };
                    native_pack::verify_with_inputs(config, scorer, policy, key)?
                }
            };
            return Ok(Outcome {
                payload: result.payload,
                status: result.status,
                verified_report: result.verified_report,
                terminal: None,
            });
        }
        "verify-native-receipt" => native_pack::verify_receipt(config)?,
        "verify-judge" => judge::verify(config)?,
        "replay-judge" => judge::replay(
            Path::new(required_string(config, "evidence_path")?),
            config["expected_envelope_sha256"].as_str(),
        )?,
        "publish-judge" => judge::publish_with_key(config, key_bytes)?,
        "verify-judge-receipt" => judge::verify_receipt(config)?,
        "write-judge-receipt" => {
            json!({"verifier_fingerprint":judge::write_receipt_with_key(config,&config["result"],key_bytes)?})
        }
        "verify-stored-judge" => judge::verify_stored(config, false)?,
        "replay-judge-receipt" => judge::verify_stored(config, true)?,
        "validate-native-capture" => {
            let (baseline, subject) = judge_native::validate_capture(config)?;
            json!({"baseline":baseline,"subject":subject})
        }
        "finalize-native-judge" => {
            let (plan, policy) = judge_native::finalize_plan(
                &config["recipe"],
                &config["baseline"],
                &config["subject"],
            )?;
            json!({"plan":plan,"policy":policy})
        }
        "prepare-native-judge" => judge_native::prepare(
            required_string(config, "policy_json")?.as_bytes(),
            &config["schedule"],
        )?,
        "load-trust" => {
            trust_inputs::load(Path::new(required_string(config, "trust_path")?), key_bytes)?.config
        }
        "verify-trust" => {
            let trust =
                trust_inputs::load(Path::new(required_string(config, "trust_path")?), key_bytes)?;
            let mut args = trust.config;
            args["evidence_path"] = config["evidence_path"].clone();
            args["receipt_path"] = config["receipt_path"].clone();
            if let Some(budget) = config.get("max_bootstrap_draws") {
                args["max_bootstrap_draws"] = budget.clone();
            }
            return dispatch(
                if trust.captured {
                    "verify-captured"
                } else {
                    "verify-native"
                },
                &args,
                scorer,
                Some(&trust.policy_bytes),
                Some(&trust.signing_key_bytes),
            );
        }
        _ => {
            return Err(EvidenceError::incomplete(
                "unsupported native transaction operation",
            ))
        }
    };
    let status = match command {
        "verify-attestation" => {
            if value["accepted"] == true {
                0
            } else {
                4
            }
        }
        "verify-captured" => {
            if value["ok"] == true {
                0
            } else {
                7
            }
        }
        "verify-judge"
        | "verify-stored-judge"
        | "replay-judge-receipt"
        | "verify-evidence-set"
        | "verify-stored-evidence-set" => {
            if value["accepted"] == true {
                0
            } else if value["verified"] == true {
                7
            } else {
                4
            }
        }
        "verify-captured-receipt" | "verify-native-receipt" | "verify-judge-receipt" => {
            if value["ok"] == true {
                0
            } else {
                4
            }
        }
        _ => 0,
    };
    Ok(Outcome {
        payload: value,
        status,
        verified_report: None,
        terminal: None,
    })
}
