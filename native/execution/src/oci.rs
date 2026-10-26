//! OCI launcher supervision. Container cleanup is distinct from reaping the
//! local engine process; failed cleanup is reported and its CID file retained.
use crate::{run_process, run_process_impl, Error, ProcessOutput, ProcessSpec, Result};
use invarlock_native_filesystem::PinnedDirectory;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Duration,
};
pub const WORKER_DIAGNOSTIC_BYTES: usize = 64 * 1024;
#[derive(Debug)]
pub struct WorkerResult {
    pub output: ProcessOutput,
    pub cleanup_errors: Vec<String>,
    pub container_id: Option<String>,
}
fn cid_path(spec: &ProcessSpec) -> Result<Option<PathBuf>> {
    let indexes: Vec<_> = spec
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a == "--cidfile").then_some(i))
        .collect();
    if indexes.is_empty() {
        return Ok(None);
    }
    if indexes.len() != 1 || indexes[0] + 1 >= spec.args.len() {
        return Err(Error::Invalid(
            "worker command has ambiguous container ID destination".into(),
        ));
    }
    Ok(Some(spec.args[indexes[0] + 1].clone().into()))
}
fn read_cid(parent: &PinnedDirectory, name: &Path) -> Result<String> {
    let (raw, _) = parent.read_regular(name, 128)?;
    let text = std::str::from_utf8(&raw)
        .map_err(|_| Error::Invalid("container ID is not ASCII".into()))?
        .trim();
    if !(12..=64).contains(&text.len())
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Invalid("container ID is invalid".into()));
    }
    Ok(text.into())
}
/// Worker logs retain a bounded prefix while both pipes continue draining.
/// Timeout=124 and cancellation=125 preserve the Python worker status contract.
/// Container cleanup status is separate evidence; an engine exit is not proof
/// that its daemon-side container stopped.
pub fn run_side_worker(spec: &ProcessSpec, cancel: &AtomicBool) -> Result<WorkerResult> {
    if !spec.input.is_empty() {
        return Err(Error::Invalid(
            "OCI workers do not accept transport stdin".into(),
        ));
    }
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(WorkerResult {
            output: ProcessOutput {
                status: 125,
                stdout: vec![],
                stderr: b"worker cancelled after paired side failure".to_vec(),
            },
            cleanup_errors: vec![],
            container_id: None,
        });
    }
    let cidfile = cid_path(spec)?;
    let cid_guard = if let Some(path) = &cidfile {
        match path.symlink_metadata() {
            Ok(_) => {
                return Err(Error::Invalid(
                    "worker container ID destination already exists".into(),
                ));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
        Some((
            PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), false)?,
            PathBuf::from(path.file_name().ok_or_else(|| {
                Error::Invalid("worker container ID path must name a file".into())
            })?),
        ))
    } else {
        None
    };
    let mut worker = spec.clone();
    worker.stdout_limit = WORKER_DIAGNOSTIC_BYTES;
    worker.stderr_limit = WORKER_DIAGNOSTIC_BYTES;
    let (output, transport_failure) = match run_process_impl(&worker, cancel, None, true) {
        Ok(output) => (output, None),
        Err(error) => (
            ProcessOutput {
                status: 126,
                stdout: vec![],
                stderr: vec![],
            },
            Some(error),
        ),
    };
    let mut cleanup_errors = Vec::new();
    let mut container_id = None;
    if matches!(output.status, 124 | 125) || transport_failure.is_some() {
        if let Some((parent, name)) = &cid_guard {
            match read_cid(parent, name) {
                Ok(id) => {
                    for action in ["stop", "kill"] {
                        let mut args = vec![action.into()];
                        if action == "stop" {
                            args.extend(["--time".into(), "5".into()]);
                        }
                        args.push(id.clone().into());
                        let control = ProcessSpec {
                            program: spec.program.clone(),
                            args,
                            input: vec![],
                            timeout: Duration::from_secs(10),
                            stdout_limit: WORKER_DIAGNOSTIC_BYTES,
                            stderr_limit: WORKER_DIAGNOSTIC_BYTES,
                            env: spec.env.clone(),
                            cwd: None,
                        };
                        match run_process(&control, &AtomicBool::new(false)) {
                            Ok(result) if result.status == 0 => (),
                            Ok(result) => cleanup_errors
                                .push(format!("container {action} exited {}", result.status)),
                            Err(e) => {
                                cleanup_errors.push(format!("container {action} failed: {e}"))
                            }
                        }
                    }
                    container_id = Some(id);
                }
                Err(e) => {
                    cleanup_errors.push(format!("container cleanup identity unavailable: {e}"))
                }
            }
        }
    }
    // CID files are retained as execution evidence. This avoids deleting the
    // sole remote handle when stop/kill cannot establish cleanup success.
    if let Some(error) = transport_failure {
        return Err(Error::Invalid(format!(
            "worker transport failed: {error}; cleanup: {}",
            if cleanup_errors.is_empty() {
                "no cleanup errors recorded; container existence may be unknown".into()
            } else {
                cleanup_errors.join("; ")
            }
        )));
    }
    Ok(WorkerResult {
        output,
        cleanup_errors,
        container_id,
    })
}
#[derive(Clone, Debug)]
pub struct WorkerLimits {
    pub cpus: String,
    pub memory_mib: u64,
    pub user: String,
}
impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            cpus: "4".into(),
            memory_mib: 65536,
            user: "65532:65532".into(),
        }
    }
}
impl WorkerLimits {
    pub fn validate(&self) -> Result<()> {
        let parts: Vec<_> = self.cpus.split('.').collect();
        let whole = parts.first().copied().unwrap_or("");
        let valid_whole = !whole.is_empty()
            && whole.len() <= 4
            && whole.bytes().all(|b| b.is_ascii_digit())
            && (whole == "0" || !whole.starts_with('0'));
        if !valid_whole
            || parts.len() > 2
            || parts.get(1).is_some_and(|s| {
                s.is_empty() || s.len() > 3 || !s.bytes().all(|b| b.is_ascii_digit())
            })
        {
            return Err(Error::Invalid(
                "runtime CPU limit must be a positive decimal with at most three places".into(),
            ));
        }
        let number: f64 = self
            .cpus
            .parse()
            .map_err(|_| Error::Invalid("runtime CPU limit is invalid".into()))?;
        if number <= 0.0 || number > 4096.0 {
            return Err(Error::Invalid(
                "runtime CPU limit must be greater than 0 and at most 4096".into(),
            ));
        }
        if !(128..=4194304).contains(&self.memory_mib) {
            return Err(Error::Invalid(
                "runtime memory limit must be between 128 and 4194304 MiB".into(),
            ));
        }
        let ids: Vec<_> = self.user.split(':').collect();
        if ids.len() != 2
            || ids.iter().any(|s| {
                s.is_empty()
                    || s.starts_with('0')
                    || !s.bytes().all(|b| b.is_ascii_digit())
                    || s.parse::<u64>().map_or(true, |n| n > 4294967294)
            })
        {
            return Err(Error::Invalid(
                "runtime user must be a non-root numeric UID:GID pair".into(),
            ));
        }
        Ok(())
    }
}
/// An explicit worker entrypoint keeps language/runtime choice in the caller's
/// declared image contract. The launch never mounts signing material implicitly.
pub struct WorkerLaunch {
    pub engine: PathBuf,
    pub engine_kind: String,
    pub image: String,
    pub device: String,
    pub limits: WorkerLimits,
    pub artifact: PathBuf,
    pub support: BTreeMap<String, PathBuf>,
    pub job: PathBuf,
    pub output: PathBuf,
    pub cidfile: PathBuf,
    pub tmpfs_gib: u64,
    pub entrypoint: String,
    pub arguments: Vec<String>,
}
fn mount(args: &mut Vec<OsString>, source: &Path, target: &str, readonly: bool) -> Result<()> {
    let source = source
        .to_str()
        .ok_or_else(|| Error::Invalid("mount source must be UTF-8".into()))?;
    if source.contains(',')
        || source.chars().any(char::is_control)
        || !Path::new(source).is_absolute()
    {
        return Err(Error::Invalid(
            "mount source must be absolute and contain no comma".into(),
        ));
    }
    args.extend([
        "--mount".into(),
        format!(
            "type=bind,source={source},target={target}{}",
            if readonly { ",readonly" } else { "" }
        )
        .into(),
    ]);
    Ok(())
}
impl WorkerLaunch {
    pub fn compose(&self, timeout: Duration) -> Result<ProcessSpec> {
        self.limits.validate()?;
        if !self.engine.is_absolute()
            || !matches!(self.engine_kind.as_str(), "docker" | "podman")
            || !(1..=64).contains(&self.tmpfs_gib)
            || self.entrypoint.is_empty()
        {
            return Err(Error::Invalid(
                "invalid OCI engine, tmpfs allowance or entrypoint".into(),
            ));
        }
        let digest = self
            .image
            .rsplit_once('@')
            .map_or(self.image.as_str(), |(_, digest)| digest);
        if !digest.starts_with("sha256:")
            || digest.len() != 71
            || !digest[7..]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Invalid(
                "runtime image must be bound to an explicit SHA-256 digest".into(),
            ));
        }
        if self.cidfile.symlink_metadata().is_ok() {
            return Err(Error::Invalid(
                "worker container ID destination already exists".into(),
            ));
        }
        PinnedDirectory::open(self.cidfile.parent().unwrap_or(Path::new(".")), false)?.verify()?;
        // Retain/check source ancestry during composition. Daemon-side mounting
        // still requires caller-controlled roots until engine launch completes.
        let mut resources = Vec::new();
        for source in std::iter::once(&self.artifact).chain(self.support.values()) {
            resources.push(RuntimeResource::pin(source, &self.limits.user, 100_000)?);
        }
        let pins = vec![
            PinnedDirectory::open(&self.job, false)?,
            PinnedDirectory::open(&self.output, false)?,
        ];
        let mut args: Vec<OsString> = ["run", "--rm", "--init", "--pull=never", "--cidfile"]
            .into_iter()
            .map(Into::into)
            .collect();
        args.push(self.cidfile.as_os_str().into());
        args.extend(
            [
                "--stop-timeout",
                "5",
                "--network",
                "none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "1024",
                "--cpus",
                &self.limits.cpus,
                "--memory",
                &format!("{}m", self.limits.memory_mib),
                "--user",
                &self.limits.user,
                "--tmpfs",
                &format!("/tmp:rw,noexec,nosuid,nodev,size={}g", self.tmpfs_gib),
            ]
            .into_iter()
            .map(Into::into),
        );
        if self.device != "cpu" {
            let index = if self.device == "cuda" {
                None
            } else if let Some(index) = self.device.strip_prefix("cuda:") {
                if index.is_empty()
                    || !index.bytes().all(|b| b.is_ascii_digit())
                    || (index.len() > 1 && index.starts_with('0'))
                {
                    return Err(Error::Invalid("runtime device is invalid".into()));
                }
                Some(index)
            } else {
                return Err(Error::Invalid("runtime device is invalid".into()));
            };
            if self.engine_kind == "docker" {
                args.extend([
                    "--gpus".into(),
                    index
                        .map_or_else(|| "all".into(), |n| format!("device={n}"))
                        .into(),
                ]);
            } else {
                args.extend([
                    "--device".into(),
                    format!("nvidia.com/gpu={}", index.unwrap_or("all")).into(),
                ]);
            }
        }
        mount(
            &mut args,
            &self.artifact,
            "/invarlock-resources/artifact",
            true,
        )?;
        mount(&mut args, &self.job, "/invarlock/job", true)?;
        mount(&mut args, &self.output, "/invarlock/output-root", false)?;
        for (i, source) in self.support.values().enumerate() {
            mount(
                &mut args,
                source,
                &format!("/invarlock-resources/support-{i}"),
                true,
            )?;
        }
        for value in [
            "INVARLOCK_CONTAINER_EXECUTION=1".to_string(),
            "INVARLOCK_EXECUTION_BACKEND=rust".to_string(),
            format!("INVARLOCK_RUNTIME_IMAGE={}", self.image),
            format!("INVARLOCK_RUNTIME_IMAGE_DIGEST={digest}"),
            "INVARLOCK_ALLOW_NETWORK=0".into(),
            "INVARLOCK_ALLOW_REMOTE_CODE=0".into(),
            "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0".into(),
            format!("LOGNAME={}", self.limits.user.split(':').next().unwrap()),
            format!("USER={}", self.limits.user.split(':').next().unwrap()),
            "HOME=/tmp".into(),
            "HF_HOME=/tmp/huggingface".into(),
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor".into(),
            "TRITON_CACHE_DIR=/tmp/triton".into(),
        ] {
            args.extend(["-e".into(), value.into()]);
        }
        args.extend([
            "--entrypoint".into(),
            self.entrypoint.clone().into(),
            self.image.clone().into(),
        ]);
        args.extend(self.arguments.iter().map(Into::into));
        for resource in &resources {
            resource.verify()?;
        }
        for pin in &pins {
            pin.verify()?;
        }
        Ok(ProcessSpec {
            program: self.engine.clone(),
            args,
            input: vec![],
            timeout,
            stdout_limit: WORKER_DIAGNOSTIC_BYTES,
            stderr_limit: WORKER_DIAGNOSTIC_BYTES,
            env: None,
            cwd: None,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits_match_source_rejections() {
        for cpus in ["0", "00.1", "1.0000", "4096.001", "-1", "nan"] {
            assert!(WorkerLimits {
                cpus: cpus.into(),
                ..WorkerLimits::default()
            }
            .validate()
            .is_err());
        }
        for cpus in ["0.001", "1", "4096"] {
            assert!(WorkerLimits {
                cpus: cpus.into(),
                ..WorkerLimits::default()
            }
            .validate()
            .is_ok());
        }
        assert!(WorkerLimits {
            user: "0:1".into(),
            ..WorkerLimits::default()
        }
        .validate()
        .is_err());
    }
    #[test]
    fn worker_prefix_is_retained_without_deadlocking() {
        let spec = ProcessSpec {
            program: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                "head -c 200000 /dev/zero; head -c 200000 /dev/zero >&2".into(),
            ],
            input: vec![],
            timeout: Duration::from_secs(3),
            stdout_limit: 1,
            stderr_limit: 1,
            env: None,
            cwd: None,
        };
        let result = run_side_worker(&spec, &AtomicBool::new(false)).unwrap();
        assert_eq!(result.output.status, 0);
        assert_eq!(result.output.stdout.len(), WORKER_DIAGNOSTIC_BYTES);
        assert_eq!(result.output.stderr.len(), WORKER_DIAGNOSTIC_BYTES);
    }
    #[test]
    fn worker_timeout_and_pre_cancel_status() {
        let spec = ProcessSpec {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "printf retained; sleep 20".into()],
            input: vec![],
            timeout: Duration::from_millis(40),
            stdout_limit: 1,
            stderr_limit: 1,
            env: None,
            cwd: None,
        };
        let result = run_side_worker(&spec, &AtomicBool::new(false)).unwrap();
        assert_eq!(result.output.status, 124);
        assert_eq!(result.output.stdout, b"retained");
        assert!(String::from_utf8_lossy(&result.output.stderr).contains("outer deadline"));
        assert_eq!(
            run_side_worker(&spec, &AtomicBool::new(true))
                .unwrap()
                .output
                .status,
            125
        );
    }
    #[test]
    fn timed_out_launcher_retains_cid_and_controls_container() {
        use std::os::unix::fs::PermissionsExt;
        let root =
            std::env::temp_dir().join(format!("invarlock-oci-cleanup-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let engine = root.join("engine");
        let log = root.join("control.log");
        let cid = root.join("container.cid");
        let script=b"#!/bin/sh\nif [ \"$1\" = run ]; then\n  printf aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa > \"$3\"\n  sleep 20\nelse\n  printf '%s\\n' \"$1\" >> \"$INVARLOCK_TEST_CONTROL_LOG\"\nfi\n";
        std::fs::write(&engine, script).unwrap();
        std::fs::set_permissions(&engine, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut env: BTreeMap<OsString, OsString> = std::env::vars_os().collect();
        env.insert("INVARLOCK_TEST_CONTROL_LOG".into(), log.as_os_str().into());
        let spec = ProcessSpec {
            program: engine,
            args: vec!["run".into(), "--cidfile".into(), cid.as_os_str().into()],
            input: vec![],
            timeout: Duration::from_millis(100),
            stdout_limit: 1,
            stderr_limit: 1,
            env: Some(env),
            cwd: None,
        };
        let result = run_side_worker(&spec, &AtomicBool::new(false)).unwrap();
        assert_eq!(result.output.status, 124);
        assert!(result.cleanup_errors.is_empty());
        assert_eq!(std::fs::read_to_string(log).unwrap(), "stop\nkill\n");
        assert!(cid.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageInspection {
    pub config_id: String,
    pub repo_digests: Vec<String>,
}
fn valid_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn repository_digest(value: &str) -> Result<(&str, &str)> {
    if value != value.trim()
        || value.matches('@').count() != 1
        || value.chars().any(|c| c < ' ' || c.is_whitespace())
    {
        return Err(Error::Invalid(
            "runtime image inspect repository digest is invalid".into(),
        ));
    }
    let (repository, digest) = value.rsplit_once('@').unwrap();
    if repository.is_empty() || repository.starts_with(['-', '/', '\\']) || !valid_digest(digest) {
        return Err(Error::Invalid(
            "runtime image inspect repository digest is invalid".into(),
        ));
    }
    Ok((repository, digest))
}
impl ImageInspection {
    pub fn parse(raw: &[u8]) -> Result<Self> {
        use invarlock_native_contracts::{parse_json, Value};
        if raw.len() > 1024 * 1024 {
            return Err(Error::Invalid(
                "runtime image inspect output exceeds the bounded size limit".into(),
            ));
        }
        let value = parse_json(raw, "runtime image inspect output")
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let object = match &value {
            Value::Array(values) if values.len() == 1 => values[0].as_object(),
            Value::Object(object) => Some(object),
            _ => None,
        }
        .ok_or_else(|| {
            Error::Invalid("runtime image inspect must return exactly one image object".into())
        })?;
        let config = object
            .get("Id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Invalid("runtime image inspect config ID is invalid".into()))?;
        let config_id = if valid_digest(config) {
            config.into()
        } else if config.len() == 64
            && config
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            format!("sha256:{config}")
        } else {
            return Err(Error::Invalid(
                "runtime image inspect config ID is invalid".into(),
            ));
        };
        let mut repo_digests = std::collections::BTreeSet::new();
        match object.get("RepoDigests") {
            None | Some(Value::Null) => (),
            Some(Value::Array(values)) => {
                for value in values {
                    let reference = value.as_str().ok_or_else(|| {
                        Error::Invalid("runtime image inspect repository digest is invalid".into())
                    })?;
                    repository_digest(reference)?;
                    repo_digests.insert(reference.into());
                }
            }
            _ => {
                return Err(Error::Invalid(
                    "runtime image inspect RepoDigests are invalid".into(),
                ));
            }
        }
        Ok(Self {
            config_id,
            repo_digests: repo_digests.into_iter().collect(),
        })
    }
    pub fn resolve(&self, image: &str, digest: &str, allow_tag: bool) -> Result<String> {
        if !valid_digest(digest) {
            return Err(Error::Invalid(
                "runtime image digest must use lowercase sha256:<64 hex>".into(),
            ));
        }
        if image.is_empty()
            || image != image.trim()
            || image.starts_with(['-', '/', '\\'])
            || image.chars().any(|c| c < ' ' || c.is_whitespace())
            || image.matches('@').count() > 1
        {
            return Err(Error::Invalid(
                "runtime image must be a portable OCI reference".into(),
            ));
        }
        if image == digest {
            if self.config_id != digest {
                return Err(Error::Invalid(
                    "runtime local config ID and supplied digest do not agree".into(),
                ));
            }
            return Ok(digest.into());
        }
        if let Some((repository, embedded)) = image.rsplit_once('@') {
            if embedded != digest || !self.repo_digests.iter().any(|v| v == image) {
                return Err(Error::Invalid(
                    "runtime repository manifest and supplied digest do not agree".into(),
                ));
            }
            return Ok(format!("{repository}@{digest}"));
        }
        if !allow_tag {
            return Err(Error::Invalid("runtime image execution reference must be an immutable config ID or repository manifest".into()));
        }
        let slash = image.rfind('/');
        let colon = image.rfind(':');
        let repository = match colon {
            Some(index) if slash.is_none_or(|slash| index > slash) => &image[..index],
            _ => image,
        };
        if repository.is_empty() {
            return Err(Error::Invalid(
                "runtime image must include a repository name".into(),
            ));
        }
        let selected = format!("{repository}@{digest}");
        if self.repo_digests.contains(&selected) {
            return Ok(selected);
        }
        if self.config_id == digest {
            return Err(Error::Invalid("tagged runtime images cannot declare a local config ID; use the config ID as both the image reference and digest".into()));
        }
        Err(Error::Invalid(
            "runtime image tag does not resolve to the supplied repository manifest".into(),
        ))
    }
}
/// Inspect local engine state only. This never fetches or pulls an image.
pub fn inspect_local_image(engine: &Path, image: &str) -> Result<ImageInspection> {
    inspect_local_image_cancellable(engine, image, &AtomicBool::new(false))
}
fn inspect_local_image_cancellable(
    engine: &Path,
    image: &str,
    cancel: &AtomicBool,
) -> Result<ImageInspection> {
    if image.is_empty() || image.starts_with('-') || image.chars().any(char::is_control) {
        return Err(Error::Invalid(
            "runtime image must be a portable OCI reference".into(),
        ));
    }
    let spec = ProcessSpec {
        program: engine.into(),
        args: vec!["image".into(), "inspect".into(), image.into()],
        input: vec![],
        timeout: Duration::from_secs(30),
        stdout_limit: 1024 * 1024,
        stderr_limit: WORKER_DIAGNOSTIC_BYTES,
        env: None,
        cwd: None,
    };
    let output = run_process(&spec, cancel)?;
    if output.status != 0 {
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        let diagnostic: String = diagnostic.trim().chars().take(512).collect();
        return Err(Error::Invalid(format!(
            "runtime image could not be inspected: {}",
            if diagnostic.is_empty() {
                "image is unavailable"
            } else {
                &diagnostic
            }
        )));
    }
    ImageInspection::parse(&output.stdout)
}

/// Descriptor-pinned metadata inventory for large file or directory mounts.
/// No model payload is loaded into memory. Container-daemon pathname binding is
/// still a separate boundary: retain this guard and verify around engine launch.
pub struct RuntimeResource {
    path: PathBuf,
    kind: ResourceKind,
    total_bytes: u64,
}
enum ResourceKind {
    File(invarlock_native_filesystem::PinnedRegularFile),
    Directory {
        root: PinnedDirectory,
        metadata: invarlock_native_filesystem::EntryMetadata,
        inventory: BTreeMap<String, invarlock_native_filesystem::EntryMetadata>,
        limits: invarlock_native_filesystem::SnapshotLimits,
    },
}
fn grants(metadata: &invarlock_native_filesystem::EntryMetadata, uid: u32, gid: u32) -> bool {
    let directory = metadata.identity.mode & libc::S_IFMT as u32 == libc::S_IFDIR as u32;
    let (read, search) = if metadata.uid == uid {
        (0o400, 0o100)
    } else if metadata.gid == gid {
        (0o040, 0o010)
    } else {
        (0o004, 0o001)
    };
    metadata.identity.mode & read != 0 && (!directory || metadata.identity.mode & search != 0)
}
impl RuntimeResource {
    pub fn pin(path: &Path, user: &str, max_entries: usize) -> Result<Self> {
        use invarlock_native_filesystem::{PinnedRegularFile, SnapshotLimits};
        WorkerLimits {
            user: user.into(),
            ..WorkerLimits::default()
        }
        .validate()?;
        let (uid, gid) = user.split_once(':').unwrap();
        let uid = uid.parse::<u32>().unwrap();
        let gid = gid.parse::<u32>().unwrap();
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let text = path
            .to_str()
            .ok_or_else(|| Error::Invalid("runtime resource path must be UTF-8".into()))?;
        if text.contains(',') || text.chars().any(char::is_control) {
            return Err(Error::Invalid(
                "runtime resource path cannot be represented as an OCI mount".into(),
            ));
        }
        let metadata = path.symlink_metadata()?;
        if metadata.is_file() {
            let file = PinnedRegularFile::open(&path)?;
            if !grants(file.metadata(), uid, gid) {
                return Err(Error::Invalid(format!(
                    "runtime resource is not readable by the runtime worker user {user}"
                )));
            }
            let total_bytes = file.metadata().identity.size;
            return Ok(Self {
                path,
                kind: ResourceKind::File(file),
                total_bytes,
            });
        }
        if !metadata.is_dir() {
            return Err(Error::Invalid(
                "runtime resource must be a regular file or directory, not a symlink".into(),
            ));
        }
        let root = PinnedDirectory::open(&path, false)?;
        let metadata = root.metadata()?;
        let limits = SnapshotLimits {
            max_entries,
            max_files: max_entries,
            max_file_bytes: usize::MAX,
            max_total_bytes: usize::MAX,
        };
        let inventory = root.inventory_metadata(limits)?;
        let mut unreadable = Vec::new();
        if !grants(&metadata, uid, gid) {
            unreadable.push(".".into());
        }
        let mut total_bytes = 0u64;
        for (relative, entry) in &inventory {
            if !grants(entry, uid, gid) {
                unreadable.push(relative.clone());
            }
            if entry.identity.mode & libc::S_IFMT as u32 == libc::S_IFREG as u32 {
                total_bytes = total_bytes
                    .checked_add(entry.identity.size)
                    .ok_or_else(|| Error::Invalid("runtime resource size overflow".into()))?;
            }
        }
        if !unreadable.is_empty() {
            unreadable.sort();
            return Err(Error::Invalid(format!(
                "runtime resource is not readable by the runtime worker user {user}: {}{}",
                unreadable
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                if unreadable.len() > 3 {
                    format!(" and {} more", unreadable.len() - 3)
                } else {
                    String::new()
                }
            )));
        }
        let result = Self {
            path,
            kind: ResourceKind::Directory {
                root,
                metadata,
                inventory,
                limits,
            },
            total_bytes,
        };
        result.verify()?;
        Ok(result)
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
    pub fn verify(&self) -> Result<()> {
        match &self.kind {
            ResourceKind::File(file) => file.verify()?,
            ResourceKind::Directory {
                root,
                metadata,
                inventory,
                limits,
            } => {
                root.verify()?;
                if root.metadata()? != *metadata || root.inventory_metadata(*limits)? != *inventory
                {
                    return Err(Error::Invalid(
                        "runtime resource changed after inspection".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
pub fn tensorrt_scratch_gib(total_bytes: u64) -> Result<u64> {
    let required = (total_bytes as u128) * 2 + 1024 * 1024 * 1024;
    let gib = required.div_ceil(1024 * 1024 * 1024).max(4);
    if gib > 64 {
        return Err(Error::Invalid(
            "TensorRT engine bundle exceeds the bounded worker scratch capacity".into(),
        ));
    }
    Ok(gib as u64)
}

pub struct PreparedWorker {
    spec: ProcessSpec,
    resources: Vec<RuntimeResource>,
    directories: Vec<PinnedDirectory>,
    inspection: ImageInspection,
    image: String,
    digest: String,
    job_files: Vec<crate::PinnedResource>,
}
impl WorkerLaunch {
    /// Resolve local image identity and retain every resource guard for the
    /// complete worker lifetime. The engine is never instructed to pull.
    pub fn prepare(&self, timeout: Duration) -> Result<PreparedWorker> {
        self.prepare_with_cancel(timeout, &AtomicBool::new(false))
    }
    fn prepare_with_cancel(
        &self,
        timeout: Duration,
        cancel: &AtomicBool,
    ) -> Result<PreparedWorker> {
        let inspection = inspect_local_image_cancellable(&self.engine, &self.image, cancel)?;
        let digest = self
            .image
            .rsplit_once('@')
            .map_or(self.image.as_str(), |(_, digest)| digest)
            .to_string();
        inspection.resolve(&self.image, &digest, false)?;
        let spec = self.compose(timeout)?;
        let resources = std::iter::once(&self.artifact)
            .chain(self.support.values())
            .map(|p| RuntimeResource::pin(p, &self.limits.user, 100_000))
            .collect::<Result<Vec<_>>>()?;
        let directories = vec![
            PinnedDirectory::open(&self.job, false)?,
            PinnedDirectory::open(&self.output, false)?,
        ];
        let job_files = ["job.json", "schedule.json"]
            .iter()
            .map(|name| crate::PinnedResource::capture(&self.job.join(name), 16 * 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedWorker {
            job_files,
            spec,
            resources,
            directories,
            inspection,
            image: self.image.clone(),
            digest,
        })
    }
}
impl PreparedWorker {
    pub fn spec(&self) -> &ProcessSpec {
        &self.spec
    }
    pub fn image_inspection(&self) -> &ImageInspection {
        &self.inspection
    }
    pub fn run(&self, cancel: &AtomicBool) -> Result<WorkerResult> {
        if cancel.load(std::sync::atomic::Ordering::Acquire) {
            return run_side_worker(&self.spec, cancel);
        }
        for file in &self.job_files {
            file.verify()?;
        }
        for resource in &self.resources {
            resource.verify()?;
        }
        for directory in &self.directories {
            directory.verify()?;
        }
        let current = inspect_local_image_cancellable(&self.spec.program, &self.image, cancel)?;
        current.resolve(&self.image, &self.digest, false)?;
        if current.config_id != self.inspection.config_id {
            return Err(Error::Invalid(
                "runtime image config changed after preparation".into(),
            ));
        }
        let result = run_side_worker(&self.spec, cancel)?;
        for file in &self.job_files {
            file.verify()?;
        }
        for resource in &self.resources {
            resource.verify()?;
        }
        for directory in &self.directories {
            directory.verify()?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    use std::{
        os::unix::fs::{symlink, PermissionsExt},
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn root() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "invarlock-runtime-resource-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::canonicalize(path).unwrap()
    }
    #[test]
    fn file_mount_does_not_read_payload_and_detects_substitution() {
        let path = root();
        let artifact = path.join("large.bin");
        let file = std::fs::File::create(&artifact).unwrap();
        file.set_len(2 * 1024 * 1024 * 1024).unwrap();
        std::fs::set_permissions(&artifact, std::fs::Permissions::from_mode(0o644)).unwrap();
        let resource = RuntimeResource::pin(&artifact, "65532:65532", 8).unwrap();
        assert_eq!(resource.total_bytes(), 2 * 1024 * 1024 * 1024);
        std::fs::rename(&artifact, path.join("old.bin")).unwrap();
        std::fs::write(&artifact, b"different").unwrap();
        assert!(resource.verify().is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn unreadable_and_symlink_resources_rejected() {
        let path = root();
        std::fs::write(path.join("secret"), b"x").unwrap();
        std::fs::set_permissions(path.join("secret"), std::fs::Permissions::from_mode(0o600))
            .unwrap();
        assert!(RuntimeResource::pin(&path, "65532:65532", 8).is_err());
        std::fs::remove_file(path.join("secret")).unwrap();
        symlink("/", path.join("escape")).unwrap();
        assert!(RuntimeResource::pin(&path, "65532:65532", 8).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn scratch_capacity_has_inclusive_boundary() {
        assert_eq!(tensorrt_scratch_gib(0).unwrap(), 4);
        assert_eq!(tensorrt_scratch_gib(31 * 1024 * 1024 * 1024).unwrap(), 63);
        assert!(tensorrt_scratch_gib(32 * 1024 * 1024 * 1024).is_err());
    }
}

/// Execute a closed pair of worker plans. Provider-specific image entrypoints
/// are explicit inputs; admission of the pair, cancellation, local image
/// identity, resources and all process I/O remain native.
pub fn run_worker_pair(
    request: &serde_json::Value,
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    use serde_json::{json, Value};
    use std::sync::atomic::Ordering;
    fn text<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
        v.get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Invalid(format!("worker field {k} must be text")))
    }
    fn number(v: &Value, k: &str) -> Result<u64> {
        v.get(k).and_then(Value::as_u64).ok_or_else(|| {
            Error::Invalid(format!("worker field {k} must be a nonnegative integer"))
        })
    }
    let object = request
        .as_object()
        .ok_or_else(|| Error::Invalid("worker request must be an object".into()))?;
    if object.len() != 2 {
        return Err(Error::Invalid(
            "worker request fields must be parallel and workers".into(),
        ));
    }
    let parallel = request
        .get("parallel")
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::Invalid("parallel must be boolean".into()))?;
    let workers = request
        .get("workers")
        .and_then(Value::as_object)
        .filter(|w| !w.is_empty() && w.len() <= 2)
        .ok_or_else(|| Error::Invalid("one or two named workers required".into()))?;
    let mut prepared = Vec::new();
    for (role, value) in workers {
        if !matches!(role.as_str(), "baseline" | "subject") {
            return Err(Error::Invalid("unknown worker role".into()));
        }
        let expected = [
            "engine",
            "engine_kind",
            "image",
            "device",
            "cpus",
            "memory_mib",
            "user",
            "artifact",
            "support",
            "job",
            "output",
            "cidfile",
            "tmpfs_gib",
            "entrypoint",
            "arguments",
            "timeout_seconds",
        ];
        let fields = value
            .as_object()
            .ok_or_else(|| Error::Invalid("worker plan must be an object".into()))?;
        if fields.len() != expected.len() || !expected.iter().all(|k| fields.contains_key(*k)) {
            return Err(Error::Invalid(
                "worker plan fields differ from the closed contract".into(),
            ));
        }
        let timeout = number(value, "timeout_seconds")?;
        if !(1..=86400).contains(&timeout) {
            return Err(Error::Invalid("worker outer timeout is invalid".into()));
        }
        let support = value["support"]
            .as_object()
            .ok_or_else(|| Error::Invalid("worker support must be an object".into()))?
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    PathBuf::from(
                        v.as_str()
                            .ok_or_else(|| Error::Invalid("support path must be text".into()))?,
                    ),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let arguments = value["arguments"]
            .as_array()
            .ok_or_else(|| Error::Invalid("worker arguments must be an array".into()))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| Error::Invalid("worker argument must be text".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let launch = WorkerLaunch {
            engine: text(value, "engine")?.into(),
            engine_kind: text(value, "engine_kind")?.into(),
            image: text(value, "image")?.into(),
            device: text(value, "device")?.into(),
            limits: WorkerLimits {
                cpus: text(value, "cpus")?.into(),
                memory_mib: number(value, "memory_mib")?,
                user: text(value, "user")?.into(),
            },
            artifact: text(value, "artifact")?.into(),
            support,
            job: text(value, "job")?.into(),
            output: text(value, "output")?.into(),
            cidfile: text(value, "cidfile")?.into(),
            tmpfs_gib: number(value, "tmpfs_gib")?,
            entrypoint: text(value, "entrypoint")?.into(),
            arguments,
        };
        prepared.push((
            role.clone(),
            launch.prepare_with_cancel(Duration::from_secs(timeout), cancel)?,
        ));
    }
    let run = |worker: &PreparedWorker| {
        let result = match worker.run(cancel) {
            Err(Error::Cancelled) if cancel.load(Ordering::Acquire) => {
                run_side_worker(worker.spec(), cancel)
            }
            result => result,
        };
        if result.as_ref().map_or(true, |r| {
            r.output.status != 0 || !r.cleanup_errors.is_empty()
        }) {
            cancel.store(true, Ordering::Release);
        }
        result
    };
    let outcomes = if parallel {
        std::thread::scope(|scope| {
            let handles: Vec<_> = prepared
                .iter()
                .map(|(role, worker)| (role.clone(), scope.spawn(|| run(worker))))
                .collect();
            handles
                .into_iter()
                .map(|(role, handle)| {
                    (
                        role,
                        handle.join().unwrap_or_else(|_| {
                            Err(Error::Invalid("worker supervisor panicked".into()))
                        }),
                    )
                })
                .collect::<Vec<_>>()
        })
    } else {
        prepared
            .iter()
            .map(|(role, worker)| (role.clone(), run(worker)))
            .collect()
    };
    let mut result = serde_json::Map::new();
    for (role, outcome) in outcomes {
        let worker = outcome?;
        result.insert(role, json!({"status":worker.output.status,"stdout":String::from_utf8_lossy(&worker.output.stdout),"stderr":String::from_utf8_lossy(&worker.output.stderr),"cleanup_errors":worker.cleanup_errors,"container_id":worker.container_id}));
    }
    Ok(Value::Object(result))
}
