//! Process authority and durable admission accounting, separate from offline evidence.
//! Unix process groups are killed on timeout, cancellation, I/O failure and return.
//! Reservations bound admission; they are not a claim about provider billing.
#![deny(unsafe_op_in_unsafe_fn)]
pub mod admission;
mod checkpoint;
pub mod collection;
pub mod inspect_checkpoint;
pub mod oci;
pub mod sequential;
pub mod workspace;
pub use checkpoint::{Amounts, Budget, Checkpoint, Transaction, TransactionStatus};
use invarlock_native_filesystem::{FileIdentity, PinnedDirectory};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io,
    os::{
        fd::{AsRawFd, RawFd},
        unix::process::{CommandExt, ExitStatusExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Filesystem(invarlock_native_filesystem::Error),
    Invalid(String),
    Timeout,
    Cancelled,
    OutputLimit(&'static str),
    Budget(String),
    Checkpoint(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Filesystem(e) => write!(f, "{e}"),
            Self::Invalid(s) | Self::Budget(s) | Self::Checkpoint(s) => write!(f, "{s}"),
            Self::Timeout => write!(f, "process timed out"),
            Self::Cancelled => write!(f, "process cancelled"),
            Self::OutputLimit(s) => write!(f, "process {s} limit exceeded"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<invarlock_native_filesystem::Error> for Error {
    fn from(e: invarlock_native_filesystem::Error) -> Self {
        Self::Filesystem(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Debug)]
pub struct ProcessSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub input: Vec<u8>,
    pub timeout: Duration,
    pub stdout_limit: usize,
    pub stderr_limit: usize,
    pub env: Option<BTreeMap<OsString, OsString>>,
    pub cwd: Option<PathBuf>,
}
#[derive(Debug)]
pub struct ProcessOutput {
    pub status: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}
struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        let pid = self.0.id() as libc::pid_t;
        unsafe { libc::kill(-pid, libc::SIGKILL) };
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn nonblock(fd: RawFd) -> Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}
/// Run one explicit command without a shell. No callbacks execute in the drain
/// loop, so an arbitrarily slow output consumer cannot block stderr or deadlines.
/// Output is returned only after bounded capture and process cleanup.
pub fn run_process(spec: &ProcessSpec, cancel: &AtomicBool) -> Result<ProcessOutput> {
    run_process_impl(spec, cancel, None, false)
}
/// Copy explicitly supplied file capabilities and rewrite exact fd-path argv
/// references to the owned copies. Closing/reusing a caller FD after this entry
/// cannot substitute the file read by the child. All copied descriptors are
/// closed on every return, and retained metadata is rechecked after execution.
pub fn run_process_with_descriptors(
    spec: &ProcessSpec,
    cancel: &AtomicBool,
    descriptors: &[RawFd],
) -> Result<ProcessOutput> {
    use std::os::fd::FromRawFd;
    use std::os::unix::fs::MetadataExt;
    if descriptors.len() > 64 || descriptors.iter().any(|fd| *fd < 3) {
        return Err(Error::Invalid(
            "invalid inherited file descriptor set".into(),
        ));
    }
    let cwd_guard = spec
        .cwd
        .as_ref()
        .map(|p| PinnedDirectory::open(p, false))
        .transpose()?;
    let mut copied = Vec::new();
    let mut replacements = BTreeMap::new();
    let mut identities = Vec::new();
    for fd in descriptors {
        if replacements.contains_key(fd) {
            return Err(Error::Invalid("duplicate inherited descriptor".into()));
        }
        let duplicate = unsafe { libc::fcntl(*fd, libc::F_DUPFD_CLOEXEC, 3) };
        if duplicate < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let file = unsafe { std::fs::File::from_raw_fd(duplicate) };
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Invalid(
                "inherited descriptor must identify a regular file".into(),
            ));
        }
        identities.push((
            metadata.dev(),
            metadata.ino(),
            metadata.mode(),
            metadata.size(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        ));
        replacements.insert(*fd, duplicate);
        copied.push(file);
    }
    let rewrite = |value: &OsString| -> OsString {
        if let Some(text) = value.to_str() {
            for (old, new) in &replacements {
                for prefix in ["/proc/self/fd/", "/dev/fd/"] {
                    if text == format!("{prefix}{old}") {
                        return format!("{prefix}{new}").into();
                    }
                }
            }
        }
        value.clone()
    };
    let mut actual = spec.clone();
    actual.program = PathBuf::from(rewrite(&spec.program.as_os_str().to_owned()));
    actual.args = spec.args.iter().map(rewrite).collect();
    let result = run_process_descriptors(&actual, cancel, None, false, &copied);
    if let Some(directory) = &cwd_guard {
        directory.verify()?;
    }
    for (file, expected) in copied.iter().zip(identities) {
        let m = file.metadata()?;
        if expected
            != (
                m.dev(),
                m.ino(),
                m.mode(),
                m.size(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        {
            return Err(Error::Invalid(
                "inherited resource changed during process execution".into(),
            ));
        }
    }
    result
}
/// Execute the retained executable inode, never a reopened pathname. This mode
/// currently supports Linux ELF binaries. Other platforms and scripts fail
/// closed; ordinary `run_process` does not claim pinned executable identity.
pub fn run_pinned_process(
    spec: &ProcessSpec,
    cancel: &AtomicBool,
    executable: &PinnedExecutable,
) -> Result<ProcessOutput> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (spec, cancel, executable);
        Err(Error::Invalid(
            "descriptor-pinned executable launch is supported only for Linux ELF binaries".into(),
        ))
    }
    #[cfg(target_os = "linux")]
    {
        executable.resource.verify()?;
        let output = run_process_impl(spec, cancel, Some(executable), false)?;
        executable.resource.verify()?;
        Ok(output)
    }
}
fn run_process_impl(
    spec: &ProcessSpec,
    cancel: &AtomicBool,
    executable: Option<&PinnedExecutable>,
    diagnostics: bool,
) -> Result<ProcessOutput> {
    run_process_descriptors(spec, cancel, executable, diagnostics, &[])
}
fn run_process_descriptors(
    spec: &ProcessSpec,
    cancel: &AtomicBool,
    executable: Option<&PinnedExecutable>,
    diagnostics: bool,
    descriptors: &[std::fs::File],
) -> Result<ProcessOutput> {
    const MAX_BYTES: usize = 1024 * 1024 * 1024;
    if !spec.program.is_absolute()
        || spec.timeout.is_zero()
        || spec.timeout > Duration::from_secs(86400)
        || spec.stdout_limit > MAX_BYTES
        || spec.stderr_limit > MAX_BYTES
        || spec.input.len() > MAX_BYTES
    {
        return Err(Error::Invalid(
            "invalid executable path, timeout or byte bounds".into(),
        ));
    }
    if cancel.load(Ordering::Acquire) {
        if diagnostics {
            return Ok(ProcessOutput {
                status: 125,
                stdout: vec![],
                stderr: b"worker cancelled after paired side failure".to_vec(),
            });
        }
        return Err(Error::Cancelled);
    }
    let deadline = Instant::now()
        .checked_add(spec.timeout)
        .ok_or_else(|| Error::Invalid("timeout overflow".into()))?;
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(env) = &spec.env {
        command.env_clear().envs(env);
    }
    if let Some(cwd) = &spec.cwd {
        command.current_dir(cwd);
    }
    if !descriptors.is_empty() {
        let inherited: Vec<_> = descriptors.iter().map(AsRawFd::as_raw_fd).collect();
        // File ownership and all allocation are established before fork. Only
        // async-signal-safe descriptor syscalls occur in the child.
        unsafe {
            command.pre_exec(move || {
                for fd in &inherited {
                    if libc::fcntl(*fd, libc::F_SETFD, 0) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
    }
    if let Some(executable) = executable {
        #[cfg(target_os = "linux")]
        {
            use std::{ffi::CString, os::unix::ffi::OsStrExt};
            let invalid =
                || Error::Invalid("executable arguments or environment contain NUL".into());
            let argv: Vec<CString> = std::iter::once(spec.program.as_os_str())
                .chain(spec.args.iter().map(|s| s.as_os_str()))
                .map(|v| CString::new(v.as_bytes()).map_err(|_| invalid()))
                .collect::<Result<_>>()?;
            let environment: BTreeMap<OsString, OsString> = spec
                .env
                .clone()
                .unwrap_or_else(|| std::env::vars_os().collect());
            let env: Vec<CString> = environment
                .iter()
                .map(|(k, v)| {
                    let mut raw = k.as_bytes().to_vec();
                    if raw.contains(&b'=') {
                        return Err(invalid());
                    }
                    raw.push(b'=');
                    raw.extend(v.as_bytes());
                    CString::new(raw).map_err(|_| invalid())
                })
                .collect::<Result<_>>()?;
            let argv_ptrs: Vec<usize> = argv
                .iter()
                .map(|v| v.as_ptr() as usize)
                .chain(std::iter::once(0))
                .collect();
            let env_ptrs: Vec<usize> = env
                .iter()
                .map(|v| v.as_ptr() as usize)
                .chain(std::iter::once(0))
                .collect();
            let fd = executable.file.as_raw_fd();
            // The closure performs only the async-signal-safe exec syscall and
            // errno read. All argument storage is allocated before fork.
            unsafe {
                command.pre_exec(move || {
                    let _keep_alive = (&argv, &env);
                    libc::fexecve(fd, argv_ptrs.as_ptr().cast(), env_ptrs.as_ptr().cast());
                    Err(io::Error::last_os_error())
                });
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = executable;
            return Err(Error::Invalid(
                "descriptor-pinned executable launch unsupported on this platform".into(),
            ));
        }
    }
    let mut process = Running(command.spawn()?);
    let mut input = process.0.stdin.take();
    let mut output = process.0.stdout.take();
    let mut error = process.0.stderr.take();
    for fd in [
        input.as_ref().map(AsRawFd::as_raw_fd),
        output.as_ref().map(AsRawFd::as_raw_fd),
        error.as_ref().map(AsRawFd::as_raw_fd),
    ]
    .into_iter()
    .flatten()
    {
        nonblock(fd)?;
    }
    if spec.input.is_empty() {
        input.take();
    }
    let mut offset: usize = 0;
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    let status = loop {
        if cancel.load(Ordering::Acquire) {
            if diagnostics {
                if !stderr.is_empty() {
                    stderr.push(b'\n');
                }
                stderr.extend(b"worker cancelled after paired side failure");
                return Ok(ProcessOutput {
                    status: 125,
                    stdout,
                    stderr,
                });
            }
            return Err(Error::Cancelled);
        }
        let remaining = match deadline.checked_duration_since(Instant::now()) {
            Some(value) => value,
            None => {
                if diagnostics {
                    if !stderr.is_empty() {
                        stderr.push(b'\n');
                    }
                    stderr.extend(
                        format!(
                            "worker exceeded its {}-second outer deadline",
                            spec.timeout.as_secs()
                        )
                        .as_bytes(),
                    );
                    return Ok(ProcessOutput {
                        status: 124,
                        stdout,
                        stderr,
                    });
                }
                return Err(Error::Timeout);
            }
        };
        if input.is_none() && output.is_none() && error.is_none() {
            if let Some(s) = process.0.try_wait()? {
                break s;
            }
        }
        let mut poll = [
            libc::pollfd {
                fd: input.as_ref().map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLOUT,
                revents: 0,
            },
            libc::pollfd {
                fd: output.as_ref().map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: error.as_ref().map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let wait = remaining.as_millis().clamp(1, 20) as i32;
        let ready = unsafe { libc::poll(poll.as_mut_ptr(), poll.len() as libc::nfds_t, wait) };
        if ready < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e.into());
        }
        if input.is_some() && poll[0].revents != 0 {
            let end = spec.input.len().min(offset.saturating_add(65536));
            let written = unsafe {
                libc::write(
                    poll[0].fd,
                    spec.input[offset..end].as_ptr().cast(),
                    end - offset,
                )
            };
            if written < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::BrokenPipe {
                    input.take();
                } else if e.kind() != io::ErrorKind::WouldBlock
                    && e.kind() != io::ErrorKind::Interrupted
                {
                    return Err(e.into());
                }
            } else {
                offset += written as usize;
                if offset == spec.input.len() {
                    input.take();
                }
            }
        }
        if output.is_some()
            && poll[1].revents != 0
            && drain(
                poll[1].fd,
                &mut stdout,
                spec.stdout_limit,
                "stdout",
                diagnostics,
            )?
        {
            output.take();
        }
        if error.is_some()
            && poll[2].revents != 0
            && drain(
                poll[2].fd,
                &mut stderr,
                spec.stderr_limit,
                "stderr",
                diagnostics,
            )?
        {
            error.take();
        }
    };

    Ok(ProcessOutput {
        status: status
            .code()
            .unwrap_or_else(|| -status.signal().unwrap_or(1)),
        stdout,
        stderr,
    })
}
/// Verify descriptor-pinned resources on both sides of execution. Returned
/// output can be accepted only if all resources still match their captured
/// identities. Use the captured bytes as process input when immutable content
/// (rather than a pathname consumed by an external program) is required.
pub fn run_process_with_resources(
    spec: &ProcessSpec,
    cancel: &AtomicBool,
    resources: &[&PinnedResource],
) -> Result<ProcessOutput> {
    for resource in resources {
        resource.verify()?;
    }
    let output = run_process(spec, cancel)?;
    for resource in resources {
        resource.verify()?;
    }
    Ok(output)
}

fn drain(
    fd: RawFd,
    target: &mut Vec<u8>,
    limit: usize,
    label: &'static str,
    diagnostics: bool,
) -> Result<bool> {
    let mut buffer = [0u8; 65536];
    let count = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
    if count < 0 {
        let e = io::Error::last_os_error();
        if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::Interrupted {
            return Ok(false);
        }
        return Err(e.into());
    }
    if count == 0 {
        return Ok(true);
    }
    let n = count as usize;
    if diagnostics {
        let retain = n.min(limit.saturating_sub(target.len()));
        target.extend_from_slice(&buffer[..retain]);
        return Ok(false);
    }
    if n > limit.saturating_sub(target.len()) {
        return Err(Error::OutputLimit(label));
    }
    target.extend_from_slice(&buffer[..n]);
    Ok(false)
}

/// Retains a descriptor chain and immutable input bytes. Call `verify` at the
/// dispatch and retention boundaries; this detects changes without claiming to
/// sandbox other processes running under the same filesystem identity.
pub struct PinnedExecutable {
    resource: PinnedResource,
    #[allow(dead_code)]
    file: std::fs::File,
}
impl PinnedExecutable {
    pub fn capture(path: &Path, max_bytes: usize) -> Result<Self> {
        use std::{
            ffi::CString,
            os::{
                fd::FromRawFd,
                unix::{ffi::OsStrExt, fs::MetadataExt},
            },
        };
        let resource = PinnedResource::capture(path, max_bytes)?;
        if !resource.bytes().starts_with(b"\x7fELF") {
            return Err(Error::Invalid(
                "pinned executable must be an ELF binary".into(),
            ));
        }
        let name = CString::new(resource.name.as_os_str().as_bytes())
            .map_err(|_| Error::Invalid("executable name contains NUL".into()))?;
        let raw = unsafe {
            libc::openat(
                resource.parent.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let file = unsafe { std::fs::File::from_raw_fd(raw) };
        let m = file.metadata()?;
        let identity = FileIdentity {
            device: m.dev(),
            inode: m.ino(),
            mode: m.mode(),
            size: m.size(),
            modified_ns: m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128,
            changed_ns: m.ctime() as i128 * 1_000_000_000 + m.ctime_nsec() as i128,
        };
        if identity != *resource.identity() {
            return Err(Error::Invalid("executable changed while pinning".into()));
        }
        resource.verify()?;
        Ok(Self { resource, file })
    }
    pub fn identity(&self) -> &FileIdentity {
        self.resource.identity()
    }
}

pub struct PinnedResource {
    parent: PinnedDirectory,
    name: PathBuf,
    identity: FileIdentity,
    bytes: Box<[u8]>,
}
impl PinnedResource {
    pub fn capture(path: &Path, max_bytes: usize) -> Result<Self> {
        let parent = PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), false)?;
        let name = PathBuf::from(
            path.file_name()
                .ok_or_else(|| Error::Invalid("resource must name a file".into()))?,
        );
        let (bytes, identity) = parent.read_regular(&name, max_bytes)?;
        Ok(Self {
            parent,
            name,
            identity,
            bytes: bytes.into_boxed_slice(),
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }
    pub fn verify(&self) -> Result<()> {
        self.parent.verify_regular(&self.name, &self.identity)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn command(script: &str) -> ProcessSpec {
        ProcessSpec {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            input: vec![],
            timeout: Duration::from_secs(3),
            stdout_limit: 2_000_000,
            stderr_limit: 2_000_000,
            env: None,
            cwd: None,
        }
    }
    #[test]
    fn simultaneous_burst_and_full_input() {
        let mut s = command("head -c 200000 /dev/zero >&2 & cat; wait");
        s.input = vec![42; 300000];
        let out = run_process(&s, &AtomicBool::new(false)).unwrap();
        assert!(out.stdout == s.input);
        assert_eq!(out.stderr.len(), 200000);
        assert_eq!(out.status, 0);
    }
    #[test]
    fn stderr_overflow_is_bounded() {
        let mut s = command("head -c 1000000 /dev/zero >&2");
        s.stderr_limit = 4096;
        assert!(matches!(
            run_process(&s, &AtomicBool::new(false)),
            Err(Error::OutputLimit("stderr"))
        ));
    }
    #[test]
    fn timeout_kills_descendants_holding_pipes() {
        let mut s = command("sleep 20 & wait");
        s.timeout = Duration::from_millis(60);
        let start = Instant::now();
        assert!(matches!(
            run_process(&s, &AtomicBool::new(false)),
            Err(Error::Timeout)
        ));
        assert!(start.elapsed() < Duration::from_secs(2));
    }
    #[test]
    fn cancellation_stops_running_process() {
        let s = command("sleep 20");
        let cancel = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(30));
                cancel.store(true, Ordering::Release);
            });
            assert!(matches!(run_process(&s, &cancel), Err(Error::Cancelled)));
        });
    }
    #[test]
    fn slow_stdin_reader_and_early_closed_stdin() {
        let mut s = command("sleep 0.05; cat");
        s.input = vec![65; 500000];
        assert_eq!(
            run_process(&s, &AtomicBool::new(false)).unwrap().stdout,
            s.input
        );
        let mut s = command("exit 7");
        s.input = vec![65; 500000];
        assert_eq!(run_process(&s, &AtomicBool::new(false)).unwrap().status, 7);
    }
    #[test]
    fn exact_limit_allowed_and_plus_one_rejected() {
        let mut s = command("printf abc");
        s.stdout_limit = 3;
        assert_eq!(
            run_process(&s, &AtomicBool::new(false)).unwrap().stdout,
            b"abc"
        );
        s.stdout_limit = 2;
        assert!(matches!(
            run_process(&s, &AtomicBool::new(false)),
            Err(Error::OutputLimit("stdout"))
        ));
    }
    #[test]
    fn pinned_resource_rejects_same_size_replacement() {
        use std::sync::atomic::AtomicU64;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "invarlock-resource-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"before").unwrap();
        let path = std::fs::canonicalize(path).unwrap();
        let resource = PinnedResource::capture(&path, 64).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"before").unwrap();
        assert!(resource.verify().is_err());
        assert!(run_process_with_resources(
            &command("exit 0"),
            &AtomicBool::new(false),
            &[&resource]
        )
        .is_err());
        assert_eq!(resource.bytes(), b"before");
        std::fs::remove_file(path).unwrap();
    }
}
