//! Source-compatible private evaluation-workspace capabilities.
use crate::{advisory_lock::AdvisoryLock, Error, Result};
use invarlock_native_contracts::{canonical_json_no_newline as canonical, parse_json};
use invarlock_native_filesystem::{write_file_no_replace, FileIdentity, PinnedDirectory};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::File,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Path, PathBuf},
};
fn bad(s: &str) -> Error {
    Error::Checkpoint(s.into())
}
pub struct Workspace {
    path: PathBuf,
    root: PinnedDirectory,
    lock: AdvisoryLock,
    retained: BTreeMap<String, FileIdentity>,
}
impl Workspace {
    pub fn open(path: &Path) -> Result<Self> {
        let root = PinnedDirectory::open(path, true)?;
        let m = root.metadata()?;
        if m.uid != unsafe { libc::geteuid() } || m.identity.mode & 0o077 != 0 {
            return Err(bad(
                "judge workspace must be caller-owned and private (0700)",
            ));
        }
        let name = CString::new(".evaluation.lock").unwrap();
        let fd = unsafe {
            libc::openat(
                root.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0o600,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let lock = unsafe { File::from_raw_fd(fd) };
        let m = lock.metadata()?;
        if !m.is_file()
            || m.uid() != unsafe { libc::geteuid() }
            || m.mode() & 0o077 != 0
            || m.nlink() != 1
        {
            return Err(bad("judge workspace lock is unsafe"));
        }
        let lock = AdvisoryLock::acquire(lock)
            .map_err(|_| bad("another evaluation is using this judge workspace"))?;
        let result = Self {
            path: path.into(),
            root,
            lock,
            retained: BTreeMap::new(),
        };
        result.verify()?;
        Ok(result)
    }
    pub fn verify(&self) -> Result<()> {
        self.lock.verify_owner()?;
        self.root.verify()?;
        let held = self.lock.metadata()?;
        let named = self.root.inspect(Path::new(".evaluation.lock"))?;
        if named.identity.device != held.dev()
            || named.identity.inode != held.ino()
            || named.identity.mode != held.mode()
            || named.links != 1
            || held.nlink() != 1
        {
            return Err(bad("judge workspace lock changed"));
        }
        for (name, identity) in &self.retained {
            self.root.verify_regular(Path::new(name), identity)?;
        }
        self.root.verify()?;
        Ok(())
    }
    fn name(name: &str) -> Result<&Path> {
        if name.is_empty()
            || name == "."
            || name == ".."
            || name.contains('/')
            || name == ".evaluation.lock"
        {
            return Err(bad("workspace file must be one safe entry"));
        }
        Ok(Path::new(name))
    }
    pub fn read(&mut self, name: &str, limit: usize) -> Result<Vec<u8>> {
        self.verify()?;
        if limit > 512 * 1024 * 1024 {
            return Err(bad("workspace read byte limit exceeded"));
        }
        let (bytes, identity) = self.root.read_regular(Self::name(name)?, limit)?;
        self.retained.insert(name.into(), identity);
        self.verify()?;
        Ok(bytes)
    }
    pub fn write(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.verify()?;
        if bytes.len() > 128 * 1024 * 1024 {
            return Err(bad("workspace write byte limit exceeded"));
        }
        write_file_no_replace(&self.path.join(Self::name(name)?), bytes, 0o600, false)?;
        if self.read(name, bytes.len())? != bytes {
            return Err(bad("workspace publication bytes changed"));
        }
        Ok(())
    }
    pub fn retain_identity(&mut self, name: &str, value: &[u8]) -> Result<()> {
        if value.len() > 512 * 1024 * 1024 {
            return Err(bad("workspace identity inputs exceed byte limit"));
        }
        let value =
            parse_json(value, "workspace identity").map_err(|e| bad_owned(e.to_string()))?;
        let raw = canonical(&value).map_err(|e| bad_owned(e.to_string()))?;
        let identity = serde_json::json!({"format":"invarlock/judge-workspace-identity-v1","sha256":format!("{:x}",Sha256::digest(&raw))});
        let payload = canonical(&identity).map_err(|e| bad_owned(e.to_string()))?;
        match self.read(name, 1024) {
            Ok(existing) if existing == payload => Ok(()),
            Ok(_) => Err(bad("judge workspace belongs to a different frozen request")),
            Err(Error::Filesystem(invarlock_native_filesystem::Error::Io(error)))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                self.write(name, &payload)
            }
            Err(error) => Err(error),
        }
    }
}
fn bad_owned(s: String) -> Error {
    Error::Checkpoint(s)
}
