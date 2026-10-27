//! Allocator diagnostics for the same request/output contract as transaction_benchmark.
//!
//! Run with REQUEST.json, or --schema NAME [INSTANCE.json]. Diagnostics go to
//! stderr; stdout remains the canonical result. INVARLOCK_PROFILE_PAUSE_MS adds
//! a pause after each reported phase for external heap/vmmap sampling. Requested
//! pauses are excluded from operation_wall_seconds. Counters describe successful
//! Rust System allocator requests, not RSS, allocator overhead, or mmap storage.
use invarlock_native_contracts::{
    canonical_json, packaged_schema, parse_json, sha256_digest, validate_schema,
};
use invarlock_native_evidence::{transaction, EvidenceError};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static TOTAL: AtomicU64 = AtomicU64::new(0);
static COUNT: AtomicU64 = AtomicU64::new(0);

fn allocated(size: usize) {
    let size = size as u64;
    let live = LIVE.fetch_add(size, Ordering::SeqCst) + size;
    PEAK.fetch_max(live, Ordering::SeqCst);
    TOTAL.fetch_add(size, Ordering::SeqCst);
    COUNT.fetch_add(1, Ordering::SeqCst);
}

// No allocator callback formats, logs, or allocates. Reallocation failure leaves
// the old allocation and every counter unchanged. A successful realloc counts
// as one allocation request for its full new size, while live bytes change only
// by the difference. Transient copying inside System is not observable here.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size() as u64, Ordering::SeqCst);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            let live = if new_size >= layout.size() {
                let growth = (new_size - layout.size()) as u64;
                LIVE.fetch_add(growth, Ordering::SeqCst) + growth
            } else {
                let shrink = (layout.size() - new_size) as u64;
                LIVE.fetch_sub(shrink, Ordering::SeqCst) - shrink
            };
            PEAK.fetch_max(live, Ordering::SeqCst);
            TOTAL.fetch_add(new_size as u64, Ordering::SeqCst);
            COUNT.fetch_add(1, Ordering::SeqCst);
        }
        replacement
    }
}

#[derive(Clone, Copy)]
struct Counters {
    live: u64,
    peak: u64,
    total: u64,
    count: u64,
}
impl Counters {
    fn read() -> Self {
        Self {
            live: LIVE.load(Ordering::SeqCst),
            peak: PEAK.load(Ordering::SeqCst),
            total: TOTAL.load(Ordering::SeqCst),
            count: COUNT.load(Ordering::SeqCst),
        }
    }
}
struct Phase {
    before: Counters,
    started: Instant,
}
impl Phase {
    fn start() -> Self {
        let live = LIVE.load(Ordering::SeqCst);
        PEAK.store(live, Ordering::SeqCst);
        // Capture allocations racing with the reset as well as possible. Phase
        // boundaries are sequential; counters are process-wide, so unrelated
        // concurrent work cannot be attributed separately.
        PEAK.fetch_max(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
        Self {
            before: Counters::read(),
            started: Instant::now(),
        }
    }
    fn finish(
        self,
        phase: &str,
        step: Option<usize>,
        operation: Option<&str>,
        ok: bool,
        pause: Duration,
    ) -> Duration {
        let elapsed = self.started.elapsed().as_secs_f64();
        let after = Counters::read();
        // Snapshot before constructing the diagnostic; its allocations belong
        // to neither this phase nor the next phase's allocation deltas.
        eprintln!(
            "{}",
            json!({
                "phase": phase,
                "step": step,
                "operation": operation,
                "ok": ok,
                "pid": std::process::id(),
                "elapsed_seconds": elapsed,
                "start_live_requested_bytes": self.before.live,
                "live_requested_bytes": after.live,
                "peak_live_requested_bytes": after.peak,
                "peak_above_start_requested_bytes": after.peak.saturating_sub(self.before.live),
                "total_requested_bytes": after.total,
                "phase_requested_bytes": after.total.saturating_sub(self.before.total),
                "allocation_count": after.count,
                "phase_allocation_count": after.count.saturating_sub(self.before.count),
                "pause_ms": pause.as_millis(),
            })
        );
        let started = Instant::now();
        if !pause.is_zero() {
            std::thread::sleep(pause);
        }
        started.elapsed()
    }
}
fn read(path: &str, maximum: usize, label: &str) -> Result<Value, EvidenceError> {
    let raw = fs::read_file(Path::new(path), maximum)
        .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
    Ok(parse_json(&raw, label)?)
}
fn schema_mode(
    name: &str,
    instance_path: Option<&str>,
    pause: Duration,
) -> Result<Value, EvidenceError> {
    let phase = Phase::start();
    let schema = packaged_schema(name);
    phase.finish(
        "packaged_schema_parse",
        None,
        Some(name),
        schema.is_ok(),
        pause,
    );
    drop(schema?);
    let empty = Value::Object(serde_json::Map::new());
    let phase = Phase::start();
    let empty_result = validate_schema(&empty, name);
    phase.finish(
        "schema_empty_validation",
        None,
        Some(name),
        empty_result.is_ok(),
        pause,
    );
    let mut output = json!({"ok":true,"schema":name,"empty_valid":empty_result.is_ok(),"empty_error":empty_result.err().map(|e| e.to_string())});
    if let Some(path) = instance_path {
        let phase = Phase::start();
        let instance = read(path, 128 * 1024 * 1024, "profile instance");
        phase.finish(
            "instance_read_parse",
            None,
            Some(name),
            instance.is_ok(),
            pause,
        );
        let instance = instance?;
        let phase = Phase::start();
        let result = validate_schema(&instance, name);
        phase.finish(
            "schema_instance_validation",
            None,
            Some(name),
            result.is_ok(),
            pause,
        );
        output["instance_valid"] = json!(result.is_ok());
        output["instance_error"] = json!(result.err().map(|e| e.to_string()));
    }
    Ok(output)
}
fn transaction_mode(path: &str, pause: Duration) -> Result<Value, EvidenceError> {
    let phase = Phase::start();
    let request = read(path, 4 * 1024 * 1024, "benchmark request");
    phase.finish("request_read_parse", None, None, request.is_ok(), pause);
    let request = request?;
    let started = Instant::now();
    let mut paused = Duration::ZERO;
    let mut previous = Value::Null;
    let mut outputs = vec![];
    for (index, step) in request["steps"]
        .as_array()
        .ok_or_else(|| EvidenceError::incomplete("steps required"))?
        .iter()
        .enumerate()
    {
        let phase = Phase::start();
        let operation = step["operation"].as_str();
        let result = (|| -> Result<(), EvidenceError> {
            let mut config = step["config"].clone();
            if step["previous_result"] == true {
                config["result"] = previous.clone();
            }
            let result = transaction::dispatch(
                operation.ok_or_else(|| EvidenceError::incomplete("operation required"))?,
                &config,
                None,
                None,
                None,
            )?;
            previous = result.payload.clone();
            outputs.push(json!({"status":result.status,"payload":result.payload}));
            Ok(())
        })();
        paused += phase.finish(
            "transaction_step",
            Some(index),
            operation,
            result.is_ok(),
            pause,
        );
        result?;
    }
    let elapsed = started.elapsed().saturating_sub(paused).as_secs_f64();
    let phase = Phase::start();
    let files = (|| -> Result<serde_json::Map<String, Value>, EvidenceError> {
        let mut files = serde_json::Map::new();
        for (label, path) in request["files"]
            .as_object()
            .ok_or_else(|| EvidenceError::incomplete("files required"))?
        {
            let raw = fs::read_file(Path::new(path.as_str().unwrap()), 384 * 1024 * 1024)
                .map_err(|e| EvidenceError::incomplete(e.to_string()))?;
            files.insert(label.clone(), json!(sha256_digest(&raw)));
        }
        Ok(files)
    })();
    phase.finish("output_file_hashing", None, None, files.is_ok(), pause);
    let files = files?;
    Ok(
        json!({"ok":true,"operation_wall_seconds":elapsed,"result":{"outputs":outputs,"files":files},"native_transactions":outputs.len()}),
    )
}
fn run() -> Result<Value, EvidenceError> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| EvidenceError::incomplete("request path required"))?;
    let pause = match std::env::var("INVARLOCK_PROFILE_PAUSE_MS") {
        Ok(value) => Duration::from_millis(value.parse::<u64>().map_err(|_| {
            EvidenceError::incomplete("INVARLOCK_PROFILE_PAUSE_MS must be an unsigned integer")
        })?),
        Err(std::env::VarError::NotPresent) => Duration::ZERO,
        Err(_) => {
            return Err(EvidenceError::incomplete(
                "INVARLOCK_PROFILE_PAUSE_MS must be valid Unicode",
            ))
        }
    };
    if path == "--schema" {
        let name = args
            .next()
            .ok_or_else(|| EvidenceError::incomplete("schema name required"))?;
        let instance = args.next();
        schema_mode(&name, instance.as_deref(), pause)
    } else {
        transaction_mode(&path, pause)
    }
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
