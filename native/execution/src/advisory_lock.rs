//! A process-owned flock lease, released independently of inherited descriptors.
use std::{
    fs::{File, Metadata},
    io,
    os::fd::{AsRawFd, FromRawFd, RawFd},
};

pub(crate) struct AdvisoryLock {
    file: File,
    owner_pid: libc::pid_t,
}

fn flock(fd: RawFd, operation: libc::c_int) -> io::Result<()> {
    loop {
        if unsafe { libc::flock(fd, operation) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

impl AdvisoryLock {
    /// Acquire a newly opened, independently owned file description. Failed
    /// acquisition must not create a guard whose destructor could unlock it.
    pub(crate) fn acquire(file: File) -> io::Result<Self> {
        flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB)?;
        Ok(Self {
            file,
            owner_pid: unsafe { libc::getpid() },
        })
    }

    /// The caller retains a newly opened, unlocked directory descriptor. Own a
    /// close-on-exec duplicate before acquiring its shared file-description lock.
    pub(crate) fn duplicate_and_acquire(fd: RawFd) -> io::Result<Self> {
        let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
        if duplicate < 0 {
            return Err(io::Error::last_os_error());
        }
        Self::acquire(unsafe { File::from_raw_fd(duplicate) })
    }

    pub(crate) fn verify_owner(&self) -> io::Result<()> {
        if self.owner_pid != unsafe { libc::getpid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "advisory lock belongs to another process",
            ));
        }
        Ok(())
    }

    pub(crate) fn metadata(&self) -> io::Result<Metadata> {
        self.file.metadata()
    }

    #[cfg(test)]
    pub(crate) fn test_duplicate(&self) -> io::Result<File> {
        self.file.try_clone()
    }

    #[cfg(test)]
    pub(crate) fn test_set_owner_pid(&mut self, pid: libc::pid_t) {
        self.owner_pid = pid;
    }
}

impl AsRawFd for AdvisoryLock {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl Drop for AdvisoryLock {
    fn drop(&mut self) {
        if self.owner_pid == unsafe { libc::getpid() } {
            // O_CLOEXEC closes inherited descriptors only at exec. Explicitly
            // release the lease now, even while an unrelated child is pre-exec.
            let _ = flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::OpenOptions,
        os::unix::fs::OpenOptionsExt,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temporary_file() -> (PathBuf, File) {
        let path = std::env::temp_dir().join(format!(
            "invarlock-advisory-lock-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        (path, file)
    }
    fn reopen(path: &std::path::Path) -> File {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap()
    }

    #[test]
    fn owner_drop_releases_lock_with_surviving_duplicate() {
        let (path, file) = temporary_file();
        let owner = AdvisoryLock::acquire(file).unwrap();
        let inherited = owner.file.try_clone().unwrap();
        assert!(AdvisoryLock::acquire(reopen(&path)).is_err());
        // Failed acquisition must not release the active owner's lease.
        assert!(AdvisoryLock::acquire(reopen(&path)).is_err());
        drop(owner);
        let resumed = AdvisoryLock::acquire(reopen(&path)).unwrap();
        drop(inherited);
        assert!(AdvisoryLock::acquire(reopen(&path)).is_err());
        drop(resumed);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn constructor_error_releases_lock_with_surviving_duplicate() {
        let (path, file) = temporary_file();
        let mut inherited = None;
        let result = (|| -> io::Result<()> {
            let owner = AdvisoryLock::acquire(file)?;
            inherited = Some(owner.file.try_clone()?);
            Err(io::Error::other("failure after acquisition"))
        })();
        assert!(result.is_err());
        let resumed = AdvisoryLock::acquire(reopen(&path)).unwrap();
        drop(inherited);
        drop(resumed);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn foreign_owner_cannot_verify_or_unlock_parent_lease() {
        let (path, file) = temporary_file();
        let owner = AdvisoryLock::acquire(file).unwrap();
        // A fork preserves the owner's stored PID while changing getpid().
        // Model that mismatch without unsafe allocator activity after fork.
        let foreign = AdvisoryLock {
            file: owner.file.try_clone().unwrap(),
            owner_pid: 0,
        };
        assert_eq!(
            foreign.verify_owner().unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        drop(foreign);
        assert!(AdvisoryLock::acquire(reopen(&path)).is_err());
        drop(owner);
        drop(AdvisoryLock::acquire(reopen(&path)).unwrap());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn directory_lease_releases_while_original_descriptor_survives() {
        let (path, file) = temporary_file();
        drop(file);
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        let directory = File::open(&path).unwrap();
        let owner = AdvisoryLock::duplicate_and_acquire(directory.as_raw_fd()).unwrap();
        assert!(AdvisoryLock::acquire(File::open(&path).unwrap()).is_err());
        drop(owner);
        drop(AdvisoryLock::acquire(File::open(&path).unwrap()).unwrap());
        drop(directory);
        std::fs::remove_dir(path).unwrap();
    }
}
