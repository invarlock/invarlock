//! Source-compatible Inspect collection checkpoint transactions.
//!
//! The caller supplies a header and empty export derived from an already
//! validated plan. This module preserves checkpoint identity, admission/result
//! precedence and attempt order; provider-event validity belongs to the retained
//! source contract validator and must be checked before live dispatch resumes.
use crate::{Error, Result};
use invarlock_native_contracts::{canonical_json_no_newline as canonical_json, parse_json, Value};
use invarlock_native_filesystem::{write_file_no_replace, FileIdentity, PinnedDirectory};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Path, PathBuf},
};
const MAX_EVENT_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 384 * 1024 * 1024;
const MAX_ENTRIES: usize = 1_200_002;
const HEADER: &str = "collection.json";
const LOCK: &str = ".collection.lock";
fn contract<T>(result: invarlock_native_contracts::Result<T>) -> Result<T> {
    result.map_err(|e| Error::Checkpoint(e.to_string()))
}
fn bad(message: &str) -> Error {
    Error::Checkpoint(message.into())
}
#[derive(Clone, Debug)]
pub struct InspectAttempt {
    pub admission: Value,
    pub result: Option<Value>,
}
pub struct InspectCheckpoint {
    directory: PathBuf,
    pinned: PinnedDirectory,
    lock: File,
    header: Value,
    empty_export: Value,
    attempts: BTreeMap<(String, u64), InspectAttempt>,
    identities: BTreeMap<String, FileIdentity>,
    poisoned: bool,
}
impl InspectCheckpoint {
    pub fn open(directory: &Path, header: &Value, empty_export: &Value) -> Result<Self> {
        validate_templates(header, empty_export)?;
        let raw = contract(canonical_json(header))?;
        if raw.len() > 65536 {
            return Err(bad("checkpoint header exceeds byte allowance"));
        }
        let pinned = PinnedDirectory::open(directory, true)?;
        let lock_name = CString::new(LOCK).unwrap();
        let fd = unsafe {
            libc::openat(
                pinned.as_raw_fd(),
                lock_name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600 as libc::c_uint,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let lock = unsafe { File::from_raw_fd(fd) };
        let metadata = lock.metadata()?;
        if !metadata.is_file()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o077 != 0
        {
            return Err(bad("collection lock is not a safe regular file"));
        }
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(bad("another collector is already using this checkpoint"));
        }
        match pinned.read_regular(Path::new(HEADER), 65536) {
            Ok((bytes, _)) => {
                if bytes != raw {
                    return Err(bad("checkpoint identity differs from this collection"));
                }
            }
            Err(invarlock_native_filesystem::Error::Io(e))
                if e.kind() == io::ErrorKind::NotFound =>
            {
                write_file_no_replace(&directory.join(HEADER), &raw, 0o600, false)?
            }
            Err(e) => return Err(e.into()),
        }
        let mut result = Self {
            directory: directory.into(),
            pinned,
            lock,
            header: header.clone(),
            empty_export: empty_export.clone(),
            attempts: BTreeMap::new(),
            identities: BTreeMap::new(),
            poisoned: false,
        };
        result.reload()?;
        Ok(result)
    }
    pub fn header(&self) -> &Value {
        &self.header
    }
    pub fn attempts(&self) -> &BTreeMap<(String, u64), InspectAttempt> {
        &self.attempts
    }
    pub fn export(&self) -> Result<Value> {
        self.verify()?;
        let mut export = self.empty_export.clone();
        let samples = export["samples"].as_array_mut().unwrap();
        for sample in samples {
            let trial = sample["id"].as_str().unwrap();
            let events: Vec<_> = self
                .attempts
                .range((trial.to_string(), 0)..=(trial.to_string(), u64::MAX))
                .map(|(_, attempt)| {
                    attempt
                        .result
                        .as_ref()
                        .unwrap_or(&attempt.admission)
                        .clone()
                })
                .collect();
            sample["events"] = Value::Array(events);
        }
        self.verify()?;
        Ok(export)
    }
    pub fn pending_admissions(&self) -> Vec<(String, u64)> {
        self.attempts
            .iter()
            .filter_map(|(key, value)| value.result.is_none().then_some(key.clone()))
            .collect()
    }
    pub fn admit(&mut self, trial_id: &str, attempt: u64, event: &Value) -> Result<()> {
        let samples = self.empty_export["samples"].as_array().unwrap();
        if !samples.iter().any(|s| s["id"].as_str() == Some(trial_id)) {
            return Err(bad("checkpoint attempt has an unknown identity"));
        }
        let count = self
            .attempts
            .keys()
            .filter(|(id, _)| id == trial_id)
            .count() as u64;
        if attempt != count + 1 {
            return Err(bad(
                "checkpoint attempts are missing, duplicated, or reordered",
            ));
        }
        if self
            .attempts
            .range((trial_id.into(), 0)..=(trial_id.into(), u64::MAX))
            .any(|(_, a)| a.result.is_none())
        {
            return Err(bad(
                "prior admission remains ambiguous; automatic retry is forbidden",
            ));
        }
        self.write("admission-", trial_id, attempt, event)?;
        self.attempts.insert(
            (trial_id.into(), attempt),
            InspectAttempt {
                admission: event.clone(),
                result: None,
            },
        );
        Ok(())
    }
    pub fn complete(&mut self, trial_id: &str, attempt: u64, event: &Value) -> Result<()> {
        let prior = self
            .attempts
            .get(&(trial_id.into(), attempt))
            .ok_or_else(|| bad("checkpoint result has no prior call admission"))?;
        if prior.result.is_some() {
            return Err(bad("checkpoint contains duplicate results"));
        }
        self.write("result-", trial_id, attempt, event)?;
        self.attempts
            .get_mut(&(trial_id.into(), attempt))
            .unwrap()
            .result = Some(event.clone());
        Ok(())
    }
    fn write(&mut self, prefix: &str, trial: &str, attempt: u64, event: &Value) -> Result<()> {
        self.verify()?;
        if contract(canonical_json(event))?.len() > MAX_EVENT_BYTES {
            return Err(bad("retained event exceeds byte allowance"));
        }
        let mut record = Value::Object(Default::default());
        record
            .as_object_mut()
            .unwrap()
            .insert("trial_id".into(), Value::String(trial.into()));
        record
            .as_object_mut()
            .unwrap()
            .insert("attempt".into(), Value::from(attempt));
        record
            .as_object_mut()
            .unwrap()
            .insert("event".into(), event.clone());
        let raw = contract(canonical_json(&record))?;
        let total = self
            .identities
            .values()
            .map(|identity| identity.size as u128)
            .sum::<u128>()
            + raw.len() as u128;
        if total > MAX_TOTAL_BYTES as u128 || self.identities.len() >= MAX_ENTRIES {
            return Err(bad("checkpoint exceeds retained storage allowance"));
        }
        let name = format!("{prefix}{trial}-{attempt:02}.json");
        if let Err(error) = write_file_no_replace(&self.directory.join(&name), &raw, 0o600, false) {
            self.poisoned = true;
            return Err(error.into());
        }
        let (retained, identity) = self
            .pinned
            .read_regular(Path::new(&name), MAX_EVENT_BYTES + 4096)
            .map_err(|e| {
                self.poisoned = true;
                Error::from(e)
            })?;
        if retained != raw {
            self.poisoned = true;
            return Err(bad("checkpoint event changed during publication"));
        }
        self.identities.insert(name, identity);
        self.verify().inspect_err(|_| self.poisoned = true)
    }
    fn names(&self) -> Result<BTreeSet<String>> {
        self.pinned.verify()?;
        let mut names = BTreeSet::new();
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| bad("checkpoint directory contains an unknown entry"))?;
            if name == LOCK {
                continue;
            }
            if names.len() >= MAX_ENTRIES || !entry.file_type()?.is_file() {
                return Err(bad("checkpoint directory contains an unknown entry"));
            }
            if name != HEADER
                && (!(name.starts_with("admission-") || name.starts_with("result-"))
                    || !name.ends_with(".json"))
            {
                return Err(bad("checkpoint directory contains an unknown entry"));
            }
            names.insert(name);
        }
        self.pinned.verify()?;
        Ok(names)
    }
    fn reload(&mut self) -> Result<()> {
        let names = self.names()?;
        let trials: BTreeSet<_> = self.empty_export["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["id"].as_str().unwrap())
            .collect();
        let mut admissions = BTreeMap::new();
        let mut results = BTreeMap::new();
        let mut total = 0u64;
        for name in names {
            let (raw, identity) = self.pinned.read_regular(
                Path::new(&name),
                if name == HEADER {
                    65536
                } else {
                    MAX_EVENT_BYTES + 4096
                },
            )?;
            total = total
                .checked_add(raw.len() as u64)
                .ok_or_else(|| bad("checkpoint size overflow"))?;
            if total > MAX_TOTAL_BYTES {
                return Err(bad("checkpoint exceeds retained storage allowance"));
            }
            self.identities.insert(name.clone(), identity);
            if name == HEADER {
                continue;
            }
            let record = contract(parse_json(&raw, "Inspect judge checkpoint attempt"))?;
            let object = record
                .as_object()
                .ok_or_else(|| bad("checkpoint attempt has an unsupported shape"))?;
            if object.len() != 3
                || !["trial_id", "attempt", "event"]
                    .iter()
                    .all(|k| object.contains_key(*k))
            {
                return Err(bad("checkpoint attempt has an unsupported shape"));
            }
            let trial = record["trial_id"]
                .as_str()
                .filter(|id| trials.contains(id))
                .ok_or_else(|| bad("checkpoint attempt has an unknown identity"))?;
            let attempt = record["attempt"]
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| bad("checkpoint attempt has an unknown identity"))?;
            let key = (trial.to_string(), attempt);
            let target = if name.starts_with("admission-") {
                &mut admissions
            } else {
                &mut results
            };
            if target.insert(key, record["event"].clone()).is_some() {
                return Err(bad(if name.starts_with("admission-") {
                    "checkpoint contains duplicate admissions"
                } else {
                    "checkpoint contains duplicate results"
                }));
            }
        }
        if results.keys().any(|key| !admissions.contains_key(key)) {
            return Err(bad("checkpoint result has no prior call admission"));
        }
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for (key, event) in admissions {
            let count = counts.entry(key.0.clone()).or_default();
            *count += 1;
            if key.1 != *count {
                return Err(bad(
                    "checkpoint attempts are missing, duplicated, or reordered",
                ));
            }
            let result = results.remove(&key);
            self.attempts.insert(
                key,
                InspectAttempt {
                    admission: event,
                    result,
                },
            );
        }
        self.verify()
    }
    #[allow(clippy::unnecessary_cast)]
    pub fn verify(&self) -> Result<()> {
        if self.poisoned {
            return Err(bad(
                "checkpoint publication was ambiguous; close and resume before dispatch",
            ));
        }
        self.pinned.verify()?;
        let lock_name = CString::new(LOCK).unwrap();
        let mut named = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe {
            libc::fstatat(
                self.pinned.as_raw_fd(),
                lock_name.as_ptr(),
                named.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(bad("collection lock binding changed"));
        }
        let named = unsafe { named.assume_init() };
        let held = self.lock.metadata()?;
        if named.st_dev as u64 != held.dev()
            || named.st_ino as u64 != held.ino()
            || named.st_mode as u32 != held.mode()
        {
            return Err(bad("collection lock binding changed"));
        }
        if self.names()? != self.identities.keys().cloned().collect() {
            return Err(bad("checkpoint inventory changed"));
        }
        for (name, identity) in &self.identities {
            self.pinned.verify_regular(Path::new(name), identity)?;
        }
        self.pinned.verify()?;
        Ok(())
    }
}
fn validate_templates(header: &Value, empty: &Value) -> Result<()> {
    let fields = [
        "format",
        "profile",
        "inspect_version",
        "plan_sha256",
        "collection",
        "scorer_id",
    ];
    let object = header
        .as_object()
        .ok_or_else(|| bad("checkpoint header must be an object"))?;
    if object.len() != fields.len() || !fields.iter().all(|k| object.contains_key(*k)) {
        return Err(bad("checkpoint header has an unsupported shape"));
    }
    if header["format"] != "invarlock/inspect-judge-export-v1"
        || header["profile"] != "inspect-text-frozen-answer-v1"
    {
        return Err(bad("unsupported Inspect checkpoint format or profile"));
    }
    for key in ["format", "profile", "inspect_version", "collection"] {
        if empty.get(key) != header.get(key) {
            return Err(bad("empty export identity differs from checkpoint header"));
        }
    }
    let samples = empty
        .get("samples")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("empty export must contain samples"))?;
    let mut ids = BTreeSet::new();
    for sample in samples {
        let id = sample
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("sample identity is missing"))?;
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || !ids.insert(id)
        {
            return Err(bad("sample identity is unsafe or duplicated"));
        }
        if !sample
            .get("events")
            .and_then(Value::as_array)
            .is_some_and(|v| v.is_empty())
        {
            return Err(bad("checkpoint base export must have empty events"));
        }
        if sample.get("metadata").and_then(|v| v.get("plan_sha256")) != header.get("plan_sha256") {
            return Err(bad("sample plan identity differs from checkpoint header"));
        }
    }
    Ok(())
}
