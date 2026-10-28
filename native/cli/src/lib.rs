mod offline;
mod request;
mod terminal;
// Offline command front door. All verification uses the native evidence core.
use invarlock_native_contracts as contracts;
use invarlock_native_evidence::{self as evidence, EvidenceError};
use invarlock_native_filesystem as fs;
use serde_json::json;
use std::{
    io::{self, Read, Write},
    path::Path,
};

const MAX_REQUEST: usize = 384 * 1024 * 1024;
fn execute(command: &str, input: &[u8]) -> Result<(Vec<u8>, u8), EvidenceError> {
    if command == "decode-request" {
        return Ok((contracts::canonical_json(&request::parse(input)?)?, 0));
    }
    let config = contracts::parse_json(input, "command request")?;
    let result = evidence::transaction::dispatch(command, &config, None, None, None)?;
    Ok((contracts::canonical_json(&result.payload)?, result.status))
}
fn public_json(payload: &serde_json::Value) -> Result<Vec<u8>, EvidenceError> {
    let raw = contracts::canonical_json(payload)?;
    let body = std::str::from_utf8(&raw)
        .map_err(|_| EvidenceError::incomplete("output_encoding_error"))?;
    Ok(format!(
        "{}\n",
        evidence::reporting::terminal_text(body.trim_end_matches('\n'))
    )
    .into_bytes())
}
fn run(args: Vec<String>) -> Result<u8, EvidenceError> {
    if args.is_empty() || args[0] == "--help" || args[0] == "help" {
        println!("invarlock-native-verify evaluate REQUEST --signing-key KEY [--preflight] [--json]\ninvarlock-native-verify verify EVIDENCE --trust-profile TRUST --receipt RECEIPT --json\ninvarlock-native-verify report EVIDENCE [--html PATH] [--markdown PATH] [--junit PATH] --json\ninvarlock-native-verify report EVIDENCE --review-context CONTEXT --package OUTPUT --json\nLow-level transactions: COMMAND --request PATH\nUse --request - for bounded JSON stdin.\nCommands: {}",evidence::transaction::COMMANDS.join(", "));
        return Ok(0);
    }
    if args[0] == "--version" {
        println!("invarlock-native-verify {}", env!("CARGO_PKG_VERSION"));
        return Ok(0);
    }
    if args[0] == "--capabilities" {
        io::stdout()
            .write_all(&contracts::canonical_json(&json!({
            "format":"invarlock/native-capabilities-v1","experimental":true,"python_required":false,
            "commands":evidence::transaction::COMMANDS,
            "network":false,"provider_execution":false,"automatic_fallback":false}))?)
            .map_err(|_| EvidenceError::incomplete("output_io_error"))?;
        return Ok(0);
    }
    if ["evaluate", "verify", "report"].contains(&args[0].as_str())
        && args.get(1).map(String::as_str) != Some("--request")
    {
        if args.iter().any(|v| v == "--help" || v == "-h") {
            terminal::help(&args[0]);
            return Ok(0);
        }
        let mut public_args = args[1..].to_vec();
        terminal::native_backend(&mut public_args)?;
        let result = offline::run(&args[0], &public_args)?;
        if !args.iter().any(|s| s == "--json") {
            terminal::render(&args, &result)?;
            return Ok(result.status);
        }
        io::stdout()
            .write_all(&public_json(&result.payload)?)
            .map_err(|_| EvidenceError::incomplete("output_io_error"))?;
        return Ok(result.status);
    }
    if args.len() != 3 || args[1] != "--request" {
        return Err(EvidenceError::incomplete("expected COMMAND --request PATH"));
    }
    let bytes = if args[2] == "-" {
        let mut raw = Vec::new();
        io::stdin()
            .take((MAX_REQUEST + 1) as u64)
            .read_to_end(&mut raw)
            .map_err(|_| EvidenceError::incomplete("request_io_error"))?;
        if raw.len() > MAX_REQUEST {
            return Err(EvidenceError::incomplete("request exceeds byte limit"));
        }
        raw
    } else {
        fs::read_file(Path::new(&args[2]), MAX_REQUEST)
            .map_err(|e| EvidenceError::incomplete(e.to_string()))?
    };
    let (output, status) = execute(&args[0], &bytes)?;
    io::stdout()
        .write_all(&output)
        .map_err(|_| EvidenceError::incomplete("output_io_error"))?;
    Ok(status)
}
pub fn entry(args: Vec<String>) -> u8 {
    let public = args
        .first()
        .is_some_and(|s| ["evaluate", "verify", "report"].contains(&s.as_str()))
        && args.get(1).map(String::as_str) != Some("--request");
    match run(args.clone()) {
        Ok(code) => code,
        Err(error) => {
            let (payload, status) = if public {
                terminal::public_error(&args, &error)
            } else {
                (error.payload(), error.exit_code)
            };
            if public && !args.iter().any(|s| s == "--json") {
                println!(
                    "FAIL {}",
                    evidence::reporting::terminal_text(&error.to_string())
                );
            } else if let Ok(raw) = if public {
                public_json(&payload)
            } else {
                contracts::canonical_json(&payload).map_err(EvidenceError::from)
            } {
                let _ = io::stdout().write_all(&raw);
            }
            status
        }
    }
}
