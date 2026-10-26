//! Native ownership of Inspect collection admission and completion transitions.
//! Provider SDK construction and one audited model invocation remain callbacks;
//! they cannot select slots, rewrite checkpoint identity or admit retries.
use crate::{
    admission::{collection_capacity, Capacity, CollectionReservations, RetainedSize},
    inspect_checkpoint::InspectCheckpoint,
    Error, Result,
};
use invarlock_native_contracts::{canonical_json_no_newline as canonical, parse_json, Value};
use invarlock_native_core as core;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn encode(value: &Value) -> Result<Vec<u8>> {
    canonical(value).map_err(|e| invalid(e.to_string()))
}
fn native<T>(result: std::result::Result<T, core::AnalysisError>) -> Result<T> {
    result.map_err(|e| invalid(e.to_string()))
}
#[derive(Clone, Debug)]
pub struct AdmittedCall {
    pub trial_id: String,
    pub attempt: u64,
    pub request: Value,
}
impl AdmittedCall {
    pub fn to_value(&self) -> Value {
        json!({"trial_id":self.trial_id,"attempt":self.attempt,"request":self.request})
    }
}
pub struct Pacer {
    next: Mutex<Instant>,
    spacing: Duration,
}
impl Pacer {
    pub fn wait(&self, cancel: &AtomicBool) -> Result<()> {
        let mut next = self
            .next
            .lock()
            .map_err(|_| invalid("collection pacer lock was poisoned"))?;
        while let Some(remaining) = next.checked_duration_since(Instant::now()) {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            std::thread::sleep(remaining.min(Duration::from_millis(20)));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        *next = Instant::now() + self.spacing;
        Ok(())
    }
}
pub struct CollectionCoordinator {
    plan: Value,
    options: Value,
    baseline: Value,
    subject: Value,
    checkpoint: InspectCheckpoint,
    exported: Value,
    trials: BTreeMap<String, Value>,
    trial_order: Vec<String>,
    sample_indexes: BTreeMap<String, usize>,
    importer: core::InspectImportContext,
    event_ids: BTreeSet<String>,
    response_owners: BTreeMap<String, String>,
    sizes: BTreeMap<String, (u64, u64)>,
    header_bytes: u64,
    record_bytes: u64,
    retained_bytes: u64,
    spent_calls: u64,
    requests: BTreeMap<String, Value>,
    active: BTreeSet<String>,
    started_calls: BTreeSet<String>,
    started: Instant,
    deadline: Instant,
    stop_after_batches: Option<u64>,
    completed_batches: u64,
    stop_reason: Option<String>,
    pacer: Arc<Pacer>,
    cancel: Arc<AtomicBool>,
}
impl CollectionCoordinator {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        directory: &Path,
        plan: Value,
        options: Value,
        baseline: Value,
        subject: Value,
        scorer_id: &str,
        invocation_timeout: Duration,
        stop_after_batches: Option<u64>,
    ) -> Result<Self> {
        native(core::validate_measurement_plan(&plan))?;
        if plan["schedule"]["max_attempts"] != 1 {
            return Err(invalid(
                "live Inspect collection currently requires max_attempts=1",
            ));
        }
        if options["inspect_version"] != "0.3.263" {
            return Err(invalid("live collection requires current Inspect version"));
        }
        if scorer_id.is_empty()
            || scorer_id.len() > 128
            || !scorer_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        {
            return Err(invalid("scorer_id must be a bounded identifier"));
        }
        if invocation_timeout < Duration::from_secs(1)
            || invocation_timeout > Duration::from_secs(604800)
        {
            return Err(invalid(
                "invocation_timeout_seconds must be between 1 and 604800",
            ));
        }
        if stop_after_batches.is_some_and(|v| !(1..=1_000_000).contains(&v)) {
            return Err(invalid("stop_after_batches must be between 1 and 1000000"));
        }
        let root = invarlock_native_filesystem::PinnedDirectory::open(directory, true)?;
        let metadata = root.metadata()?;
        if metadata.uid != unsafe { libc::geteuid() } || metadata.identity.mode & 0o077 != 0 {
            return Err(invalid(
                "checkpoint directory must be caller-owned and private",
            ));
        }
        let digest = native(core::measurement_plan_digest(&plan))?;
        let header = json!({"format":"invarlock/inspect-judge-export-v1","profile":"inspect-text-frozen-answer-v1","inspect_version":"0.3.263","plan_sha256":digest,"collection":options,"scorer_id":scorer_id});
        let mut bindings: Vec<_> = plan["answer_bindings"]
            .as_array()
            .ok_or_else(|| invalid("plan answer bindings are missing"))?
            .iter()
            .collect();
        bindings.sort_by_key(|row| row["case_id"].as_str().unwrap_or(""));
        let repetitions = plan["schedule"]["repetitions"]
            .as_u64()
            .ok_or_else(|| invalid("plan repetitions are invalid"))?;
        let mut samples = Vec::new();
        for binding in bindings {
            let case_id = binding["case_id"]
                .as_str()
                .ok_or_else(|| invalid("case identity is invalid"))?;
            for side in ["baseline", "subject"] {
                for repetition in 1..=repetitions {
                    let id = native(core::expected_judge_trial_id(
                        &digest, case_id, side, repetition,
                    ))?;
                    samples.push(json!({"id":id,"epoch":1,"metadata":{"case_id":case_id,"side":side,"repetition":repetition,"plan_sha256":digest,"answer_sha256":binding[format!("{side}_answer_sha256")],"scorer_id":scorer_id},"events":[]}));
                }
            }
        }
        let empty = json!({"format":"invarlock/inspect-judge-export-v1","profile":"inspect-text-frozen-answer-v1","inspect_version":"0.3.263","collection":options,"samples":samples});
        // Validate frozen runs, collection identity and all planned slots before
        // writing a checkpoint or exposing any admission to an SDK callback.
        native(core::import_inspect_export(
            &empty, &plan, &options, &baseline, &subject,
        ))?;
        let checkpoint = InspectCheckpoint::open(directory, &header, &empty)?;
        let started = Instant::now();
        let exported = checkpoint.export()?;
        let measurements = native(core::import_inspect_export(
            &exported, &plan, &options, &baseline, &subject,
        ))?;
        let importer = native(core::InspectImportContext::new(
            &plan, &options, &baseline, &subject,
        ))?;
        let mut trials = BTreeMap::new();
        let mut trial_order = Vec::new();
        let mut response_owners = BTreeMap::new();
        for trial in measurements["trials"].as_array().unwrap() {
            let id = trial["trial_id"].as_str().unwrap().to_string();
            for attempt in trial["attempts"].as_array().unwrap() {
                if let Some(response) = attempt["request_id"].as_str() {
                    response_owners.insert(response.to_string(), id.clone());
                }
            }
            trial_order.push(id.clone());
            trials.insert(id, trial.clone());
        }
        let header_bytes=encode(&json!({"format":"invarlock/retained-inspect-model-events-v1","inspect_version":options["inspect_version"],"collection":options,"records":[]}))?.len() as u64;
        let mut sample_indexes = BTreeMap::new();
        let mut event_ids = BTreeSet::new();
        let mut sizes = BTreeMap::new();
        let mut record_bytes = 0;
        let mut retained_bytes = 0;
        let mut spent_calls = 0;
        for (index, sample) in exported["samples"].as_array().unwrap().iter().enumerate() {
            let id = sample["id"].as_str().unwrap().to_string();
            sample_indexes.insert(id.clone(), index);
            let events = sample["events"].as_array().unwrap();
            spent_calls += events.len() as u64;
            for event in events {
                event_ids.insert(event["uuid"].as_str().unwrap().into());
            }
            let size = record_size(&trials[&id], &sample["events"], header_bytes)?;
            record_bytes += size.0;
            retained_bytes += size.1;
            sizes.insert(id, size);
        }
        let mut requests = BTreeMap::new();
        let baseline_rows: BTreeMap<_, _> = baseline["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r["id"].as_str().unwrap(), r))
            .collect();
        let subject_rows: BTreeMap<_, _> = subject["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r["id"].as_str().unwrap(), r))
            .collect();
        for sample in empty["samples"].as_array().unwrap() {
            let case = sample["metadata"]["case_id"].as_str().unwrap();
            let row = if sample["metadata"]["side"] == "baseline" {
                baseline_rows[case]
            } else {
                subject_rows[case]
            };
            let raw = native(core::render_judge_request(
                &plan,
                &row["input"],
                &row["output"],
                row.get("expected").unwrap_or(&Value::Null),
            ))?;
            if raw.len() > 1024 * 1024 {
                return Err(invalid("rendered request exceeds byte allowance"));
            }
            let request =
                parse_json(&raw, "normalized judge request").map_err(|e| invalid(e.to_string()))?;
            requests.insert(sample["id"].as_str().unwrap().into(), request);
        }
        let rpm = options["requests_per_minute"]
            .as_u64()
            .ok_or_else(|| invalid("requests_per_minute must be an integer"))?;
        if rpm == 0 {
            return Err(invalid("requests_per_minute must be positive"));
        }
        let pacer = Arc::new(Pacer {
            next: Mutex::new(started),
            spacing: Duration::from_secs_f64(60.0 / rpm as f64),
        });
        root.verify()?;
        Ok(Self {
            plan,
            options,
            baseline,
            subject,
            checkpoint,
            exported,
            trials,
            trial_order,
            sample_indexes,
            importer,
            event_ids,
            response_owners,
            sizes,
            header_bytes,
            record_bytes,
            retained_bytes,
            spent_calls,
            requests,
            active: BTreeSet::new(),
            started_calls: BTreeSet::new(),
            started,
            deadline: started + invocation_timeout,
            stop_after_batches,
            completed_batches: 0,
            stop_reason: None,
            pacer,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn remaining_seconds(&self) -> f64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
    }
    pub fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    pub fn prepare_start(&mut self, trial: &str) -> Result<Arc<Pacer>> {
        self.checkpoint.verify()?;
        if !self.active.contains(trial) || !self.started_calls.insert(trial.into()) {
            return Err(invalid(
                "call is not an unstarted admission from this invocation",
            ));
        }
        Ok(self.pacer.clone())
    }
    pub fn verify(&self) -> Result<()> {
        self.checkpoint.verify()
    }
    fn candidate(&self, trial: &str, event: &Value) -> Result<(Value, Value, (u64, u64))> {
        let index = *self
            .sample_indexes
            .get(trial)
            .ok_or_else(|| invalid("unknown trial identity"))?;
        let mut sample = self.exported["samples"][index].clone();
        let prior = sample["events"]
            .as_array()
            .unwrap()
            .first()
            .and_then(|event| event["uuid"].as_str());
        let event_id = event
            .get("uuid")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("model event ID is invalid"))?;
        if Some(event_id) != prior && self.event_ids.contains(event_id) {
            return Err(invalid("duplicate model event ID"));
        }
        sample["events"] = json!([event]);
        let normalized = native(self.importer.validate_sample(&sample, 0))?;
        for attempt in normalized["attempts"].as_array().unwrap() {
            if let Some(response) = attempt["request_id"].as_str() {
                if self
                    .response_owners
                    .get(response)
                    .is_some_and(|owner| owner != trial)
                {
                    return Err(invalid("duplicate provider response ID"));
                }
            }
        }
        let size = record_size(&normalized, &sample["events"], self.header_bytes)?;
        Ok((sample, normalized, size))
    }
    fn apply(&mut self, id: &str, sample: Value, trial: Value, size: (u64, u64)) {
        let index = self.sample_indexes[id];
        let old = &self.exported["samples"][index];
        if let Some(event) = old["events"].as_array().unwrap().first() {
            self.event_ids.remove(event["uuid"].as_str().unwrap());
        } else {
            self.spent_calls += 1;
        }
        for attempt in self.trials[id]["attempts"].as_array().unwrap() {
            if let Some(response) = attempt["request_id"].as_str() {
                self.response_owners.remove(response);
            }
        }
        for attempt in trial["attempts"].as_array().unwrap() {
            if let Some(response) = attempt["request_id"].as_str() {
                self.response_owners.insert(response.into(), id.into());
            }
        }
        self.event_ids
            .insert(sample["events"][0]["uuid"].as_str().unwrap().into());
        let prior = self.sizes[id];
        self.record_bytes = self.record_bytes - prior.0 + size.0;
        self.retained_bytes = self.retained_bytes - prior.1 + size.1;
        self.sizes.insert(id.into(), size);
        self.exported["samples"][index] = sample;
        self.trials.insert(id.into(), trial);
    }
    pub fn capacity(&self) -> Result<Capacity> {
        collection_capacity(
            CollectionReservations {
                max_calls: self.options["max_calls"].as_u64().unwrap(),
                max_input_tokens: self.options["max_input_tokens"].as_u64().unwrap(),
                max_output_tokens: self.options["max_output_tokens"].as_u64().unwrap(),
                max_cost_microusd: self.options["max_cost_microusd"].as_u64().unwrap(),
                input_tokens_per_call: self.options["input_tokens_per_call"].as_u64().unwrap(),
                output_tokens_per_call: self.plan["judge"]["config"]["max_output_tokens"]
                    .as_u64()
                    .unwrap(),
                cost_microusd_per_call: self.options["cost_microusd_per_call"].as_u64().unwrap(),
            },
            RetainedSize {
                header_bytes: self.header_bytes,
                samples: self.exported["samples"].as_array().unwrap().len() as u64,
                record_bytes: self.record_bytes,
                retained_bytes: self.retained_bytes,
                spent_calls: self.spent_calls,
            },
        )
    }
    pub fn begin_batch(&mut self) -> Result<Vec<AdmittedCall>> {
        self.checkpoint.verify()?;
        if !self.active.is_empty() {
            return Err(invalid(
                "previous collection batch has unresolved callbacks",
            ));
        }
        if self.stop_reason.is_some() {
            return Ok(vec![]);
        }
        let pending: Vec<String> = self
            .trial_order
            .iter()
            .filter(|id| self.trials[*id]["attempts"].as_array().unwrap().is_empty())
            .cloned()
            .collect();
        let capacity = self.capacity()?.capacity;
        let reason = if pending.is_empty() {
            Some("complete")
        } else if capacity == 0 {
            Some("capacity_exhausted")
        } else if self.remaining_seconds() == 0.0 {
            Some("deadline")
        } else if self
            .stop_after_batches
            .is_some_and(|maximum| self.completed_batches >= maximum)
        {
            Some("requested")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.stop_reason = Some(reason.into());
            return Ok(vec![]);
        }
        let count = pending
            .len()
            .min(capacity as usize)
            .min(self.options["concurrency"].as_u64().unwrap() as usize);
        let mut calls = Vec::with_capacity(count);
        for trial in pending.into_iter().take(count) {
            let request = self.requests[&trial].clone();
            let event = admission_event(&trial, &request, &self.options)?;
            let (sample, normalized, size) = self.candidate(&trial, &event)?;
            self.checkpoint.admit(&trial, 1, &event)?;
            self.apply(&trial, sample, normalized, size);
            self.active.insert(trial.clone());
            calls.push(AdmittedCall {
                trial_id: trial,
                attempt: 1,
                request,
            });
        }
        Ok(calls)
    }
    pub fn complete(&mut self, trial: &str, event: &Value) -> Result<()> {
        // A completed callback can be observed again while draining a failed
        // batch. Identical retention is idempotent; changed output is rejected.
        if !self.active.contains(trial) && self.started_calls.contains(trial) {
            if let Some(index) = self.sample_indexes.get(trial) {
                if self.exported["samples"][*index]["events"]
                    .as_array()
                    .is_some_and(|events| events.first() == Some(event))
                    && self
                        .checkpoint
                        .attempts()
                        .get(&(trial.to_string(), 1))
                        .is_some_and(|attempt| attempt.result.is_some())
                {
                    return Ok(());
                }
            }
        }

        if !self.active.contains(trial) || !self.started_calls.contains(trial) {
            return Err(invalid(
                "completion does not belong to a started call in this invocation",
            ));
        }
        let (sample, normalized, size) = self.candidate(trial, event)?;
        self.checkpoint.complete(trial, 1, event)?;
        self.apply(trial, sample, normalized, size);
        self.active.remove(trial);
        if self.active.is_empty() {
            self.completed_batches += 1;
        }
        Ok(())
    }
    pub fn stop(&mut self, reason: &str) -> Result<()> {
        if !matches!(reason, "deadline" | "cancelled" | "error" | "requested") {
            return Err(invalid("unsupported collection stop reason"));
        }
        if reason == "deadline" && self.remaining_seconds() > 0.0 {
            return Err(invalid("collection deadline has not elapsed"));
        }
        self.cancel.store(true, Ordering::Release);
        self.stop_reason = Some(reason.into());
        self.active.clear();
        self.checkpoint.verify()
    }
    pub fn sdk_configuration(&self) -> Result<Value> {
        let request = self
            .requests
            .values()
            .next()
            .ok_or_else(|| invalid("collection has no planned requests"))?;
        Ok(admission_event("configuration", request, &self.options)?["config"].clone())
    }
    pub fn measurements(&self) -> Result<Value> {
        self.checkpoint.verify()?;
        native(core::import_inspect_export(
            &self.checkpoint.export()?,
            &self.plan,
            &self.options,
            &self.baseline,
            &self.subject,
        ))
    }
    pub fn status(&self) -> Result<Value> {
        let capacity = self.capacity()?;
        Ok(
            json!({"stop_reason":self.stop_reason,"completed_batches":self.completed_batches,"active_calls":self.active.len(),"elapsed_seconds":self.elapsed_seconds(),"remaining_seconds":self.remaining_seconds(),"capacity":{"remaining_admissions":capacity.remaining_admissions,"capacity":capacity.capacity,"limiting_resources":capacity.limiting_resources,"exhausted_resources":capacity.exhausted_resources}}),
        )
    }
}
fn admission_event(trial: &str, request: &Value, options: &Value) -> Result<Value> {
    let config = &request["config"];
    let number = |name: &str| -> Result<f64> {
        let value = &config[name];
        value
            .as_f64()
            .or_else(|| value.as_str().and_then(|s| s.parse::<f64>().ok()))
            .filter(|n| n.is_finite())
            .ok_or_else(|| invalid(format!("judge {name} must be finite numeric text")))
    };
    Ok(
        json!({"event":"model","uuid":format!("admission-{trial}-1"),"role":"grader","model":options["grader"],"input":request["messages"],"tools":[],"tool_choice":"none","config":{"temperature":number("temperature")?,"top_p":number("top_p")?,"max_tokens":config["max_output_tokens"],"seed":config["seed"],"reasoning_effort":config["reasoning_effort"],"max_retries":0,"timeout":options["request_timeout_seconds"],"attempt_timeout":options["request_timeout_seconds"],"max_connections":options["concurrency"],"adaptive_connections":false,"num_choices":1,"internal_tools":false,"parallel_tool_calls":false,"reasoning_summary":"none","reasoning_history":"none","cache":false,"batch":false},"retries":0,"cache":null,"call":{"request":request,"response":null,"error":true},"output":{"model":null,"request_id":null,"finish_reason":null,"usage":null,"completion":""},"error":{"status":"timeout_ambiguous","code":"call-admitted","message":"Call was admitted but no completed checkpoint was retained."}}),
    )
}

fn record_size(trial: &Value, events: &Value, header_bytes: u64) -> Result<(u64, u64)> {
    let record = encode(&json!({"trial":trial,"events":events}))?;
    let raw_size = record.len() as u64 + 129;
    if raw_size + header_bytes > crate::admission::MAX_SOURCE_BYTES {
        return Err(invalid(
            "one retained trial exceeds the per-source byte allowance",
        ));
    }
    let quoted = encode(&Value::String(
        String::from_utf8(record).map_err(|_| invalid("canonical record is not UTF-8"))?,
    ))?
    .len() as u64;
    Ok((raw_size, quoted + encode(trial)?.len() as u64 + 258))
}
