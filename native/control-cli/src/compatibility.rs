//! Explicit Python compatibility process. No automatic fallback or interpreter search.
use invarlock_native_contracts::canonical_json;
use invarlock_native_execution::{run_process, ProcessSpec};
use serde_json::json;
use std::{
    ffi::OsString,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicI32, Ordering},
    time::Duration,
};
static CANCEL: AtomicBool = AtomicBool::new(false);
static SIGNAL: AtomicI32 = AtomicI32::new(0);
extern "C" fn cancelled(signal: libc::c_int) {
    SIGNAL.store(signal, Ordering::Relaxed);
    CANCEL.store(true, Ordering::Relaxed);
}
fn run(args: &[OsString]) -> Result<i32, Box<dyn std::error::Error>> {
    let mut argv = Vec::new();
    let mut python = None;
    let mut timeout = 600u64;
    let mut timeout_seen = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i]
            .to_str()
            .ok_or("command arguments must be valid UTF-8")?;
        match arg {
            "--python-worker" => {
                if python.is_some() {
                    return Err("duplicate --python-worker".into());
                }
                i += 1;
                python = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("--python-worker requires an absolute interpreter path")?,
                ));
            }
            "--worker-timeout-seconds" => {
                if timeout_seen {
                    return Err("duplicate --worker-timeout-seconds".into());
                }
                timeout_seen = true;
                i += 1;
                timeout = args
                    .get(i)
                    .and_then(|s| s.to_str())
                    .ok_or("missing worker timeout")?
                    .parse()?;
                if !(1..=86400).contains(&timeout) {
                    return Err("worker timeout must be between 1 and 86400 seconds".into());
                }
            }
            _ => argv.push(arg.to_owned()),
        }
        i += 1;
    }
    let python = python.ok_or("an explicit Python interpreter is required")?;
    if !python.is_absolute() || !python.is_file() {
        return Err("--python-worker requires an existing absolute interpreter path".into());
    }
    if !argv
        .first()
        .is_some_and(|c| ["evaluate", "verify", "report"].contains(&c.as_str()))
    {
        return Err("Python compatibility requires evaluate, verify or report".into());
    }
    let input = canonical_json(&json!({"format":"invarlock/python-worker-v1","argv":argv}))?;
    if input.len() > 1024 * 1024 {
        return Err("compatibility request exceeds its byte limit".into());
    }
    let cwd = std::env::current_dir()?;
    CANCEL.store(false, Ordering::Relaxed);
    SIGNAL.store(0, Ordering::Relaxed);
    // The handlers only store lock-free atomics. The execution loop performs cleanup.
    let old_int =
        unsafe { libc::signal(libc::SIGINT, cancelled as *const () as libc::sighandler_t) };
    let old_term =
        unsafe { libc::signal(libc::SIGTERM, cancelled as *const () as libc::sighandler_t) };
    let result = run_process(
        &ProcessSpec {
            program: python,
            args: vec!["-m".into(), "invarlock.native_compat_worker".into()],
            input,
            timeout: Duration::from_secs(timeout),
            stdout_limit: 64 * 1024 * 1024,
            stderr_limit: 4 * 1024 * 1024,
            env: None,
            cwd: Some(cwd),
        },
        &CANCEL,
    );
    unsafe {
        libc::signal(libc::SIGINT, old_int);
        libc::signal(libc::SIGTERM, old_term);
    }
    if SIGNAL.load(Ordering::Relaxed) != 0 {
        return Ok(128 + SIGNAL.load(Ordering::Relaxed));
    }
    let result = result?;
    std::io::stdout().write_all(&result.stdout)?;
    std::io::stderr().write_all(&result.stderr)?;
    Ok(result.status)
}
pub(super) fn entry(args: &[OsString]) -> i32 {
    match run(args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Python compatibility failed: {e}");
            2
        }
    }
}
