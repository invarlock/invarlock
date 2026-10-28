//! Execution authority is kept out of the offline verifier binary.
mod compatibility;
use invarlock_native_contracts::{canonical_json, parse_json, Value};
use invarlock_native_execution::{Amounts, Checkpoint, ProcessSpec, TransactionStatus};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Duration,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn fail(message: &str) -> Box<dyn std::error::Error> {
    message.to_owned().into()
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| fail(&format!("{key} must be a string")))
}
fn integer(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| fail(&format!("{key} must be a nonnegative integer")))
}
fn object_fields(value: &Value, allowed: &[&str], required: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| fail("request must be an object"))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(fail(&format!("unexpected field: {key}")));
        }
    }
    for key in required {
        if !object.contains_key(*key) {
            return Err(fail(&format!("missing field: {key}")));
        }
    }
    Ok(())
}
fn amounts(v: &Value) -> Result<Amounts> {
    let fields = [
        "calls",
        "input_tokens",
        "output_tokens",
        "cost_microusd",
        "retained_bytes",
    ];
    object_fields(v, &fields, &fields)?;
    Ok(Amounts {
        calls: integer(v, "calls")?,
        input_tokens: integer(v, "input_tokens")?,
        output_tokens: integer(v, "output_tokens")?,
        cost_microusd: integer(v, "cost_microusd")?,
        retained_bytes: integer(v, "retained_bytes")?,
    })
}
fn amounts_json(a: Amounts) -> Value {
    json!({"calls":a.calls,"input_tokens":a.input_tokens,"output_tokens":a.output_tokens,"cost_microusd":a.cost_microusd,"retained_bytes":a.retained_bytes})
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}
fn unhex(raw: &str) -> Result<Vec<u8>> {
    if !raw.len().is_multiple_of(2) {
        return Err(fail("input_hex must have even length"));
    }
    fn digit(b: u8) -> Result<u8> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            _ => Err(fail("input_hex must use lowercase hexadecimal")),
        }
    }
    raw.as_bytes()
        .chunks_exact(2)
        .map(|p| Ok(digit(p[0])? * 16 + digit(p[1])?))
        .collect()
}
fn process_spec(v: &Value) -> Result<ProcessSpec> {
    let required = [
        "program",
        "args",
        "input_hex",
        "timeout_ms",
        "stdout_limit",
        "stderr_limit",
    ];
    object_fields(
        v,
        &[
            "program",
            "args",
            "input_hex",
            "timeout_ms",
            "stdout_limit",
            "stderr_limit",
            "env",
            "cwd",
            "pinned_executable",
        ],
        &required,
    )?;
    let args = v["args"]
        .as_array()
        .ok_or_else(|| fail("args must be an array"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(Into::into)
                .ok_or_else(|| fail("args must contain strings"))
        })
        .collect::<Result<Vec<_>>>()?;
    let env = match v.get("env") {
        None | Some(Value::Null) => None,
        Some(Value::Object(values)) => Some(
            values
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.into(),
                        v.as_str()
                            .ok_or_else(|| fail("environment values must be strings"))?
                            .into(),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        _ => return Err(fail("env must be an object or null")),
    };
    let cwd = match v.get("cwd") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(PathBuf::from(s)),
        _ => return Err(fail("cwd must be a path string or null")),
    };
    if let Some(pinned) = v.get("pinned_executable") {
        if !pinned.is_boolean() {
            return Err(fail("pinned_executable must be boolean"));
        }
    }
    Ok(ProcessSpec {
        program: text(v, "program")?.into(),
        args,
        input: unhex(text(v, "input_hex")?)?,
        timeout: Duration::from_millis(integer(v, "timeout_ms")?),
        stdout_limit: usize::try_from(integer(v, "stdout_limit")?)?,
        stderr_limit: usize::try_from(integer(v, "stderr_limit")?)?,
        env,
        cwd,
    })
}
fn run(v: &Value) -> Result<Value> {
    let spec = process_spec(v)?;
    let cancel = AtomicBool::new(false);
    let output = if v
        .get("pinned_executable")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let executable = invarlock_native_execution::PinnedExecutable::capture(
            &spec.program,
            128 * 1024 * 1024,
        )?;
        invarlock_native_execution::run_pinned_process(&spec, &cancel, &executable)?
    } else {
        invarlock_native_execution::run_process(&spec, &cancel)?
    };
    Ok(
        json!({"status":output.status,"stdout_hex":hex(&output.stdout),"stderr_hex":hex(&output.stderr)}),
    )
}
fn checkpoint(v: &Value) -> Result<Checkpoint> {
    let identity = canonical_json(
        v.get("identity")
            .ok_or_else(|| fail("missing checkpoint identity"))?,
    )?;
    Checkpoint::open(
        Path::new(text(v, "directory")?),
        &identity,
        amounts(
            v.get("maximum")
                .ok_or_else(|| fail("missing maximum reservation"))?,
        )?,
    )
    .map_err(Into::into)
}
fn checkpoint_status(c: &Checkpoint) -> Value {
    let transactions:Vec<_>=c.transactions().iter().map(|(id,t)|match &t.status{TransactionStatus::Admitted=>json!({"id":id,"state":"admitted_ambiguous","reservation":amounts_json(t.reservation),"observed":null}),TransactionStatus::Complete{observed,evidence}=>json!({"id":id,"state":"complete","reservation":amounts_json(t.reservation),"observed":observed.map(amounts_json),"evidence_hex":hex(evidence)})}).collect();
    json!({"checkpoint_format":"IVLJNL01","reserved":amounts_json(c.budget().reserved()),"observed_overage":amounts_json(c.budget().observed_overage()),"remaining":amounts_json(c.budget().remaining()),"transactions":transactions})
}
fn dispatch(v: &Value) -> Result<Value> {
    let command = text(v, "command")?;
    match command {
        "process" => {
            object_fields(v, &["command", "process"], &["command", "process"])?;
            run(&v["process"])
        }
        "inspect-export" | "inspect-admit" | "inspect-complete" => {
            let fields = if command == "inspect-export" {
                vec!["command", "directory", "header", "empty_export"]
            } else {
                vec![
                    "command",
                    "directory",
                    "header",
                    "empty_export",
                    "trial_id",
                    "attempt",
                    "event",
                ]
            };
            object_fields(v, &fields, &fields)?;
            let mut checkpoint =
                invarlock_native_execution::inspect_checkpoint::InspectCheckpoint::open(
                    Path::new(text(v, "directory")?),
                    &v["header"],
                    &v["empty_export"],
                )?;
            if command == "inspect-admit" {
                checkpoint.admit(text(v, "trial_id")?, integer(v, "attempt")?, &v["event"])?;
            }
            if command == "inspect-complete" {
                checkpoint.complete(text(v, "trial_id")?, integer(v, "attempt")?, &v["event"])?;
            }
            Ok(checkpoint.export()?)
        }
        "oci-inspect" => {
            object_fields(
                v,
                &["command", "engine", "image"],
                &["command", "engine", "image"],
            )?;
            let result = invarlock_native_execution::oci::inspect_local_image(
                Path::new(text(v, "engine")?),
                text(v, "image")?,
            )?;
            Ok(json!({"config_id":result.config_id,"repo_digests":result.repo_digests}))
        }
        "oci-resolve" => {
            object_fields(
                v,
                &["command", "image", "digest", "inspection", "allow_tag"],
                &["command", "image", "digest", "inspection", "allow_tag"],
            )?;
            let allow_tag = v["allow_tag"]
                .as_bool()
                .ok_or_else(|| fail("allow_tag must be boolean"))?;
            let inspection = invarlock_native_execution::oci::ImageInspection::parse(
                &canonical_json(&v["inspection"])?,
            )?;
            Ok(json!({"image":inspection.resolve(text(v,"image")?,text(v,"digest")?,allow_tag)?}))
        }
        "oci-pair" => {
            object_fields(v, &["command", "request"], &["command", "request"])?;
            Ok(invarlock_native_execution::oci::run_worker_pair(
                &v["request"],
                &AtomicBool::new(false),
            )?)
        }
        "oci-worker" => {
            object_fields(v, &["command", "process"], &["command", "process"])?;
            let spec = process_spec(&v["process"])?;
            let result =
                invarlock_native_execution::oci::run_side_worker(&spec, &AtomicBool::new(false))?;
            Ok(
                json!({"status":result.output.status,"stdout_hex":hex(&result.output.stdout),"stderr_hex":hex(&result.output.stderr),"container_id":result.container_id,"cleanup_errors":result.cleanup_errors}),
            )
        }
        "collection-capacity" => {
            use invarlock_native_execution::admission::{
                collection_capacity, CollectionReservations, RetainedSize,
            };
            object_fields(
                v,
                &["command", "reservations", "retained"],
                &["command", "reservations", "retained"],
            )?;
            let r = &v["reservations"];
            let z = &v["retained"];
            let rf = [
                "max_calls",
                "max_input_tokens",
                "max_output_tokens",
                "max_cost_microusd",
                "input_tokens_per_call",
                "output_tokens_per_call",
                "cost_microusd_per_call",
            ];
            let zf = [
                "header_bytes",
                "samples",
                "record_bytes",
                "retained_bytes",
                "spent_calls",
            ];
            object_fields(r, &rf, &rf)?;
            object_fields(z, &zf, &zf)?;
            let result = collection_capacity(
                CollectionReservations {
                    max_calls: integer(r, "max_calls")?,
                    max_input_tokens: integer(r, "max_input_tokens")?,
                    max_output_tokens: integer(r, "max_output_tokens")?,
                    max_cost_microusd: integer(r, "max_cost_microusd")?,
                    input_tokens_per_call: integer(r, "input_tokens_per_call")?,
                    output_tokens_per_call: integer(r, "output_tokens_per_call")?,
                    cost_microusd_per_call: integer(r, "cost_microusd_per_call")?,
                },
                RetainedSize {
                    header_bytes: integer(z, "header_bytes")?,
                    samples: integer(z, "samples")?,
                    record_bytes: integer(z, "record_bytes")?,
                    retained_bytes: integer(z, "retained_bytes")?,
                    spent_calls: integer(z, "spent_calls")?,
                },
            )?;
            Ok(
                json!({"capacity":result.capacity,"remaining_admissions":result.remaining_admissions,"limiting_resources":result.limiting_resources,"exhausted_resources":result.exhausted_resources,"source_bound":result.source_bound,"retained_bound":result.retained_bound}),
            )
        }
        "checkpoint-status" => {
            object_fields(
                v,
                &["command", "directory", "identity", "maximum"],
                &["command", "directory", "identity", "maximum"],
            )?;
            Ok(checkpoint_status(&checkpoint(v)?))
        }
        "checkpoint-admit" => {
            object_fields(
                v,
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "reservation",
                ],
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "reservation",
                ],
            )?;
            let mut c = checkpoint(v)?;
            c.admit(text(v, "id")?, amounts(&v["reservation"])?)?;
            Ok(checkpoint_status(&c))
        }
        "checkpoint-complete" => {
            object_fields(
                v,
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "observed",
                    "evidence_hex",
                ],
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "observed",
                    "evidence_hex",
                ],
            )?;
            let mut c = checkpoint(v)?;
            let observed = if v["observed"].is_null() {
                None
            } else {
                Some(amounts(&v["observed"])?)
            };
            c.complete(text(v, "id")?, observed, &unhex(text(v, "evidence_hex")?)?)?;
            Ok(checkpoint_status(&c))
        }
        "process-transaction" => {
            object_fields(
                v,
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "reservation",
                    "process",
                ],
                &[
                    "command",
                    "directory",
                    "identity",
                    "maximum",
                    "id",
                    "reservation",
                    "process",
                ],
            )?;
            // Validate before charging. The effective command is bound into the
            // immutable checkpoint identity, preventing silent resume drift.
            process_spec(&v["process"])?;
            let mut bound = v.clone();
            bound["identity"] = json!({"caller_identity":v["identity"],"process":v["process"]});
            let mut c = checkpoint(&bound)?;
            let id = text(v, "id")?;
            c.admit(id, amounts(&v["reservation"])?)?;
            let outcome = match run(&v["process"]) {
                Ok(output) => json!({"outcome":"completed","process":output}),
                Err(e) => json!({"outcome":"execution_failed","error":e.to_string()}),
            };
            let evidence = canonical_json(&outcome)?;
            c.complete(id, None, &evidence)?;
            Ok(json!({"result":outcome,"checkpoint":checkpoint_status(&c)}))
        }
        _ => Err(fail("unknown execution command")),
    }
}
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.iter().any(|v| v == "--python-worker") {
        std::process::exit(compatibility::entry(&args));
    }
    if args.is_empty()
        || args.first().is_some_and(|v| {
            [
                "evaluate",
                "verify",
                "report",
                "--help",
                "help",
                "--version",
                "--capabilities",
            ]
            .iter()
            .any(|s| v == s)
        })
    {
        let args = args
            .into_iter()
            .map(|v| v.into_string())
            .collect::<std::result::Result<Vec<_>, _>>();
        match args {
            Ok(args) => std::process::exit(invarlock_native_verifier::entry(args).into()),
            Err(_) => {
                eprintln!("command arguments must be valid UTF-8");
                std::process::exit(2);
            }
        }
    }
    let result = (|| -> Result<Value> {
        if args.len() == 1 && args[0] == "diagnostics" {
            return Ok(
                json!({"format":"invarlock-native-execution-diagnostics-v1","platform":std::env::consts::OS,"process_transport":"unix-poll-process-groups","checkpoint_format":"IVLJNL01","generic_python_checkpoint_compatible":false,"inspect_checkpoint_compatible":true,"pinned_executable_supported":cfg!(target_os="linux"),"provider_execution":false,"offline_verification_binary":"invarlock-native-verify"}),
            );
        }
        if args.len() != 2 || args[0] != "request" {
            return Err(fail(
                "usage: invarlock-native diagnostics | request REQUEST.json",
            ));
        }
        let raw = invarlock_native_filesystem::read_file(Path::new(&args[1]), 16 * 1024 * 1024)?;
        dispatch(&parse_json(&raw, "execution request")?)
    })();
    match result {
        Ok(v) => match canonical_json(&v) {
            Ok(raw) => {
                use std::io::Write;
                if std::io::stdout().write_all(&raw).is_err() {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn process_request_requires_exact_fields_and_preserves_binary() {
        let request = json!({"command":"process","process":{"program":"/bin/sh","args":["-c","printf abc"],"input_hex":"","timeout_ms":1000,"stdout_limit":3,"stderr_limit":0}});
        assert_eq!(
            dispatch(&request).unwrap(),
            json!({"status":0,"stdout_hex":"616263","stderr_hex":""})
        );
        let mut invalid = request;
        invalid["process"]["unknown"] = json!(true);
        assert!(dispatch(&invalid).is_err());
    }
    #[test]
    fn request_rejects_ambiguous_hex_and_boolean_integers() {
        assert!(unhex("AB").is_err());
        assert!(unhex("0").is_err());
        assert!(integer(&json!({"n":true}), "n").is_err());
    }
}
