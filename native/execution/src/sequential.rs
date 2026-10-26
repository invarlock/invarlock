//! Source-compatible sequential admission for retained runtime/HTTP collections.
//! A persisted admission without a result is ambiguous and never dispatched again.
use crate::{Error, Result};
use invarlock_native_contracts::{canonical_json_no_newline as canonical, parse_json, Value};
use invarlock_native_core as core;
use invarlock_native_filesystem::{
    write_file_no_replace, FileIdentity, PinnedDirectory, SnapshotLimits,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};
const MAX_SOURCE: usize = 16 * 1024 * 1024;
const MAX_RETAINED: usize = 384 * 1024 * 1024;
fn bad(s: impl Into<String>) -> Error {
    Error::Checkpoint(s.into())
}
fn encode(v: &Value) -> Result<Vec<u8>> {
    canonical(v).map_err(|e| bad(e.to_string()))
}
fn checked<T>(r: std::result::Result<T, core::AnalysisError>) -> Result<T> {
    r.map_err(|e| bad(e.to_string()))
}
#[derive(Clone)]
struct Slot {
    trial: Value,
    request: Vec<u8>,
    source_id: String,
    dispatch: Value,
}
pub struct SequentialCoordinator {
    directory: PathBuf,
    pinned: PinnedDirectory,
    identities: BTreeMap<String, FileIdentity>,
    identity: Value,
    plan: Value,
    baseline: Value,
    subject: Value,
    slots: Vec<Slot>,
    sources: Vec<Value>,
    active: bool,
    poisoned: bool,
    runtime: bool,
    profile: String,
}
impl SequentialCoordinator {
    pub fn open(
        directory: &Path,
        identity: Value,
        plan: Value,
        baseline: Value,
        subject: Value,
    ) -> Result<Self> {
        checked(core::validate_measurement_plan(&plan))?;
        checked(core::validate_retained_collection_prefix(
            &[],
            &plan,
            &baseline,
            &subject,
        ))?;
        let runtime = match identity["format"].as_str() {
            Some("invarlock/runtime-provider-judge-checkpoint-v1") => true,
            Some("invarlock/openai-compatible-judge-checkpoint-v1") => false,
            _ => return Err(bad("unsupported sequential checkpoint version")),
        };
        let keys = if runtime {
            vec![
                "format",
                "plan_sha256",
                "runtime_spec",
                "artifact_identity_sha256",
                "outer_image_digest",
                "device_kind",
                "source_id",
                "scorer_id",
            ]
        } else {
            vec!["format", "plan_sha256", "configuration", "scorer_id"]
        };
        if identity
            .as_object()
            .is_none_or(|o| o.len() != keys.len() || keys.iter().any(|k| !o.contains_key(*k)))
        {
            return Err(bad(
                "checkpoint identity fields differ from the closed contract",
            ));
        }
        let digest = checked(core::measurement_plan_digest(&plan))?;
        if identity["plan_sha256"] != digest {
            return Err(bad("checkpoint plan identity differs"));
        }
        let count = plan["schedule"]["expected_trials"].as_u64().unwrap_or(0);
        if !(1..=1000).contains(&count) || plan["schedule"]["max_attempts"] != 1 {
            return Err(bad(
                "sequential collection requires 1..1000 slots and one attempt",
            ));
        }
        if !runtime {
            checked(core::validate_openai_compatible_collection(
                &identity["configuration"],
                &plan,
            ))?;
        }
        if runtime {
            let spec = &identity["runtime_spec"];
            let judge = &plan["judge"];
            let config = &judge["config"];
            if !matches!(
                spec["provider_name"].as_str(),
                Some("hf_transformers" | "llama_cpp")
            ) || judge["provider"] != spec["provider_name"]
                || judge["requested_model"] != spec["model_id"]
                || judge["approved_resolved_models"] != json!([spec["model_id"]])
                || judge["model_identity"]
                    != json!({"kind":"local_weights","weights_sha256":identity["artifact_identity_sha256"]})
            {
                return Err(bad(
                    "runtime judge artifact/model differs from the approved plan",
                ));
            }
            if config["temperature"] != "0"
                || config["top_p"] != "1"
                || !config["reasoning_effort"].is_null()
                || config["seed"].is_null()
                || plan["schedule"]["retry_on"] != json!([])
                || plan["schedule"]["cache"] != "forbid"
            {
                return Err(bad(
                    "runtime judge requires deterministic sampling and no retries or cache",
                ));
            }
        }

        for key in if runtime {
            vec!["source_id", "scorer_id"]
        } else {
            vec!["scorer_id"]
        } {
            let text = identity[key]
                .as_str()
                .ok_or_else(|| bad("checkpoint identifiers must be strings"))?;
            if text.is_empty()
                || text.chars().count() > 128
                || text.chars().any(|c| c < ' ' || c == '\u{7f}')
            {
                return Err(bad(
                    "checkpoint identifiers must be bounded printable strings",
                ));
            }
        }
        let mut slots = Vec::new();
        let mut input_bytes = 0usize;
        let mut bindings: Vec<_> = plan["answer_bindings"].as_array().unwrap().iter().collect();
        bindings.sort_by_key(|b| b["case_id"].as_str().unwrap());
        for binding in bindings {
            let case = binding["case_id"].as_str().unwrap();
            for side in ["baseline", "subject"] {
                let run = if side == "baseline" {
                    &baseline
                } else {
                    &subject
                };
                let row = run["records"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["id"] == case)
                    .ok_or_else(|| bad("frozen case missing"))?;
                for repetition in 1..=plan["schedule"]["repetitions"].as_u64().unwrap() {
                    let request = checked(core::render_judge_request(
                        &plan,
                        &row["input"],
                        &row["output"],
                        &row["expected"],
                    ))?;
                    let id = checked(core::expected_judge_trial_id(
                        &digest, case, side, repetition,
                    ))?;
                    let dispatch;
                    if runtime {
                        let prompt = checked(core::render_judge_runtime_prompt(&plan, &request))?;
                        dispatch = json!({"trial_id":id,"normalized_request":std::str::from_utf8(&request).map_err(|e|bad(e.to_string()))?,"runtime_prompt":std::str::from_utf8(&prompt).map_err(|e|bad(e.to_string()))?});
                        let context = identity["runtime_spec"]["settings"]["context_length"]
                            .as_u64()
                            .ok_or_else(|| bad("runtime context_length is invalid"))?;
                        if prompt.len() as u64
                            + plan["judge"]["config"]["max_output_tokens"]
                                .as_u64()
                                .unwrap()
                            > context
                        {
                            return Err(bad(
                                "runtime judge prompt and output reservation exceed context_length",
                            ));
                        }
                    } else {
                        let normalized = parse_json(&request, "normalized judge request")
                            .map_err(|e| bad(e.to_string()))?;
                        let body = checked(core::render_compatible_judge_request(
                            &identity["configuration"],
                            &plan,
                            &normalized["messages"],
                        ))?;
                        dispatch = json!({"trial_id":id,"normalized_request":std::str::from_utf8(&request).map_err(|e|bad(e.to_string()))?,"request_body":body});
                        let size = encode(&body)?.len();
                        input_bytes = input_bytes
                            .checked_add(size)
                            .ok_or_else(|| bad("request byte count overflow"))?;
                        if size > 1024 * 1024
                            || input_bytes as u64
                                > identity["configuration"]["max_input_bytes"]
                                    .as_u64()
                                    .unwrap_or(0)
                        {
                            return Err(bad(
                                "compatible rendered requests exceed the aggregate input-byte reservation",
                            ));
                        }
                    }
                    let base = if runtime {
                        identity["source_id"].as_str().unwrap()
                    } else {
                        "openai-compatible-judge"
                    };
                    let source_id = if count == 1 {
                        base.to_string()
                    } else {
                        format!(
                            "{}-{:06}",
                            base.chars().take(115).collect::<String>(),
                            slots.len() + 1
                        )
                    };
                    slots.push(Slot{trial:json!({"trial_id":id,"case_id":case,"side":side,"repetition":repetition,"answer_sha256":binding[format!("{side}_answer_sha256")],"plan_sha256":digest}),request,source_id,dispatch});
                }
            }
        }
        if slots.len() != count as usize {
            return Err(bad("planned slot count differs"));
        }
        let pinned = PinnedDirectory::open(directory, true)?;
        let metadata = pinned.metadata()?;
        if metadata.uid != unsafe { libc::geteuid() } || metadata.identity.mode & 0o777 != 0o700 {
            return Err(bad(
                "checkpoint directory must be caller-owned and private (0700)",
            ));
        }
        if unsafe { libc::flock(pinned.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(bad("checkpoint is already locked"));
        }
        let profile = if runtime {
            "retained-runtime-provider-judge-v1"
        } else {
            "retained-openai-compatible-judge-v1"
        }
        .to_string();
        let mut state = Self {
            directory: directory.into(),
            pinned,
            identities: BTreeMap::new(),
            identity,
            plan,
            baseline,
            subject,
            slots,
            sources: Vec::new(),
            active: false,
            poisoned: false,
            runtime,
            profile,
        };
        let identity_raw = encode(&state.identity)?;
        if identity_raw.len() > 1024 * 1024 {
            return Err(bad("checkpoint identity exceeds byte limit"));
        }
        let inventory = state.inventory()?;
        if !inventory.contains_key("identity.json") {
            if !inventory.is_empty() {
                return Err(bad("checkpoint identity is missing"));
            }
            write_file_no_replace(
                &state.directory.join("identity.json"),
                &identity_raw,
                0o600,
                false,
            )?;
        }
        state.identities = state.inventory()?;
        if state
            .pinned
            .read_regular(Path::new("identity.json"), 1024 * 1024)?
            .0
            != identity_raw
        {
            return Err(bad("checkpoint belongs to a different collection"));
        }
        let mut allowed = BTreeSet::from(["identity.json".to_string()]);
        for index in 0..state.slots.len() {
            allowed.insert(format!("admission-{:06}.json", index + 1));
            allowed.insert(format!("result-{:06}.json", index + 1));
        }
        if state.identities.keys().any(|name| !allowed.contains(name)) {
            return Err(bad("checkpoint contains an unexpected entry"));
        }
        let mut gap = false;
        for index in 0..state.slots.len() {
            let admission = format!("admission-{:06}.json", index + 1);
            let result = format!("result-{:06}.json", index + 1);
            let a = state.identities.contains_key(&admission);
            let r = state.identities.contains_key(&result);
            if gap && (a || r) {
                return Err(bad("checkpoint contains a non-contiguous shard"));
            }
            if r {
                if !a {
                    return Err(bad("checkpoint result lacks prior admission"));
                }
                if state.pinned.read_regular(Path::new(&admission), 1024)?.0
                    != encode(&state.admission(index))?
                {
                    return Err(bad("checkpoint admission is invalid"));
                }
                let raw = state.pinned.read_regular(Path::new(&result), MAX_SOURCE)?.0;
                let source =
                    parse_json(&raw, "checkpoint result").map_err(|e| bad(e.to_string()))?;
                if !source.is_object() || encode(&source)? != raw {
                    return Err(bad("checkpoint result must be a canonical object"));
                }
                state.check_slot(index, &source)?;
                state.sources.push(source);
            } else if a {
                return Err(bad(
                    "inference was admitted without a retained result and cannot be retried",
                ));
            } else {
                gap = true;
            }
        }
        checked(core::validate_retained_collection_prefix(
            &state.wrappers(&state.sources)?,
            &state.plan,
            &state.baseline,
            &state.subject,
        ))?;
        state.verify()?;
        Ok(state)
    }
    fn inventory(&self) -> Result<BTreeMap<String, FileIdentity>> {
        Ok(self.pinned.flat_regular_inventory(SnapshotLimits {
            max_entries: 2001,
            max_files: 2001,
            max_file_bytes: MAX_SOURCE,
            max_total_bytes: MAX_RETAINED,
        })?)
    }
    pub fn verify(&self) -> Result<()> {
        if self.poisoned {
            return Err(bad("checkpoint transaction is uncertain"));
        }
        self.pinned.verify()?;
        if self.inventory()? != self.identities {
            return Err(bad("checkpoint inventory or file identity changed"));
        }
        Ok(())
    }
    fn admission(&self, index: usize) -> Value {
        json!({"format":if self.runtime{"invarlock/runtime-provider-judge-admission-v1"}else{"invarlock/openai-compatible-judge-admission-v1"},"batch":index+1})
    }
    fn publish(&mut self, name: &str, value: &Value) -> Result<()> {
        self.verify()?;
        let raw = encode(value)?;
        self.poisoned = true;
        write_file_no_replace(&self.directory.join(name), &raw, 0o600, false)?;
        let (observed, id) = self.pinned.read_regular(Path::new(name), MAX_SOURCE)?;
        if raw != observed {
            return Err(bad("published checkpoint bytes changed"));
        }
        self.identities.insert(name.into(), id);
        self.poisoned = false;
        self.verify()
    }
    pub fn sources(&self) -> Result<Value> {
        self.verify()?;
        Ok(Value::Array(self.sources.clone()))
    }
    pub fn admit(&mut self, index: usize, request: &[u8]) -> Result<Value> {
        self.verify()?;
        if self.active || index != self.sources.len() || index >= self.slots.len() {
            return Err(bad("admission is not the next unadmitted slot"));
        }
        if request != self.slots[index].request {
            return Err(bad(
                "admission request differs from the native planned request",
            ));
        }
        let retained = if self.runtime || !self.sources.is_empty() {
            encode(&self.measurements(false))?.len()
        } else {
            0
        };
        if retained + 3 * MAX_SOURCE + 1024 * 1024 > MAX_RETAINED {
            return Err(bad(
                "retained capacity was exhausted before the next inference admission",
            ));
        }
        self.publish(
            &format!("admission-{:06}.json", index + 1),
            &self.admission(index),
        )?;
        self.active = true;
        Ok(self.slots[index].dispatch.clone())
    }
    fn check_slot(&self, index: usize, source: &Value) -> Result<()> {
        let slot = &self.slots[index];
        let trials = source["trials"]
            .as_array()
            .filter(|a| a.len() == 1)
            .ok_or_else(|| bad("checkpoint source must contain one trial"))?;
        let trial = &trials[0];
        if slot
            .trial
            .as_object()
            .unwrap()
            .iter()
            .any(|(k, v)| &trial[k] != v)
        {
            return Err(bad("checkpoint trial differs from its planned slot"));
        }
        let mapping = json!({"source_id":slot.source_id,"scorer_id":self.identity["scorer_id"],"model_event_id":slot.trial["trial_id"],"record_index":0,"attempt_index":0});
        if trial["attempts"].as_array().is_none_or(|a| a.len() != 1)
            || trial["attempts"][0]["source"] != mapping
        {
            return Err(bad("checkpoint source mapping is invalid"));
        }
        if self.runtime {
            if source["runtime_spec"] != self.identity["runtime_spec"]
                || source["provider_receipt"]["outer_image_digest"]
                    != self.identity["outer_image_digest"]
                || source["provider_receipt"]["device"]["device_kind"]
                    != self.identity["device_kind"]
            {
                return Err(bad(
                    "checkpoint runtime resources differ from the current request",
                ));
            }
            let digest = format!(
                "{:x}",
                Sha256::digest(encode(&source["artifact_identity"])?)
            );
            if digest
                != self.identity["artifact_identity_sha256"]
                    .as_str()
                    .unwrap_or("")
            {
                return Err(bad("checkpoint artifact identity differs"));
            }
        } else if source["collection"] != self.identity["configuration"] {
            return Err(bad(
                "checkpoint configuration differs from the current request",
            ));
        }
        Ok(())
    }
    pub fn complete(&mut self, index: usize, source: Value) -> Result<()> {
        self.verify()?;
        if !self.active || index != self.sources.len() {
            return Err(bad("completion lacks its active admission"));
        }
        if encode(&source)?.len() > MAX_SOURCE {
            return Err(bad("retained source exceeds its per-source byte limit"));
        }
        self.check_slot(index, &source)?;
        let mut proposed = self.sources.clone();
        proposed.push(source.clone());
        checked(core::validate_retained_collection_prefix(
            &self.wrappers(&proposed)?,
            &self.plan,
            &self.baseline,
            &self.subject,
        ))?;
        self.publish(&format!("result-{:06}.json", index + 1), &source)?;
        self.sources = proposed;
        self.active = false;
        Ok(())
    }
    fn wrappers(&self, sources: &[Value]) -> Result<Vec<Value>> {
        sources.iter().map(|source|{let payload=encode(source)?;Ok(json!({"source_id":source["trials"][0]["attempts"][0]["source"]["source_id"],"profile":self.profile,"encoding":"utf-8","media_type":"application/json","content":std::str::from_utf8(&payload).map_err(|e|bad(e.to_string()))?,"byte_size":payload.len(),"sha256":format!("{:x}",Sha256::digest(&payload))}))}).collect()
    }
    fn measurements(&self, final_status: bool) -> Value {
        let mut retained = Vec::new();
        let mut trials = Vec::new();
        for source in &self.sources {
            let payload = encode(source).expect("validated source serializes");
            let trial = source["trials"][0].clone();
            retained.push(json!({"source_id":trial["attempts"][0]["source"]["source_id"],"profile":self.profile,"encoding":"utf-8","byte_size":payload.len(),"media_type":"application/json","content":std::str::from_utf8(&payload).expect("JSON UTF-8"),"sha256":format!("{:x}",Sha256::digest(&payload))}));
            trials.push(trial);
        }
        let completed = trials.iter().filter(|t| t["status"] == "complete").count();
        json!({"format":"invarlock/judge-measurements-v1","profile_id":self.plan["profile_id"],"plan_sha256":self.identity["plan_sha256"],"source_profile":self.profile,"sources":retained,"trials":trials,"completeness":{"status":if (!self.runtime||final_status)&&completed==self.sources.len(){"complete"}else{"incomplete"},"expected_trials":self.slots.len(),"recorded_trials":self.sources.len(),"completed_trials":completed}})
    }
    pub fn finish(&self) -> Result<Value> {
        self.verify()?;
        if self.active || self.sources.len() != self.slots.len() {
            return Err(bad("collection is incomplete"));
        }
        let result = self.measurements(true);
        if encode(&result)?.len() > MAX_RETAINED {
            return Err(bad("retained measurements exceed the aggregate byte limit"));
        }
        checked(core::validate_measurements(
            &result,
            &self.plan,
            &self.baseline,
            &self.subject,
        ))?;
        Ok(result)
    }
}
