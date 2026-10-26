//! Offline command front door. All verification uses the native evidence core.
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
    let config = contracts::parse_json(input, "command request")?;
    let result = evidence::transaction::dispatch(command, &config, None, None, None)?;
    Ok((contracts::canonical_json(&result.payload)?, result.status))
}
fn run() -> Result<u8, EvidenceError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "help" {
        println!("invarlock-native-verify COMMAND --request PATH\nUse --request - for bounded JSON stdin.\nCommands: {}",evidence::transaction::COMMANDS.join(", "));
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
fn main() {
    let status = match run() {
        Ok(code) => code,
        Err(error) => {
            if let Ok(raw) = contracts::canonical_json(&error.payload()) {
                let _ = io::stdout().write_all(&raw);
            }
            eprintln!("{}", error);
            error.exit_code
        }
    };
    std::process::exit(status.into());
}
