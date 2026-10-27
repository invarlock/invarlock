//! Immutable, hash-chained journal. Admission is durable before work may start;
//! a missing completion remains charged and ambiguous after resume. Completion
//! never refunds a reservation, and unknown final usage stays unknown.
use crate::{advisory_lock::AdvisoryLock, Error, Result};
use invarlock_native_filesystem::{write_file_no_replace, PinnedDirectory};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Path, PathBuf},
};
const MAX_RECORD: usize = 16 * 1024 * 1024;
const MAX_EVENTS: usize = 100_000;
const MAX_JOURNAL: u64 = 512 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Amounts {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_microusd: u64,
    pub retained_bytes: u64,
}
impl Amounts {
    fn values(self) -> [u64; 5] {
        [
            self.calls,
            self.input_tokens,
            self.output_tokens,
            self.cost_microusd,
            self.retained_bytes,
        ]
    }
    fn from_values(v: [u64; 5]) -> Self {
        Self {
            calls: v[0],
            input_tokens: v[1],
            output_tokens: v[2],
            cost_microusd: v[3],
            retained_bytes: v[4],
        }
    }
    fn add(self, other: Self) -> Result<Self> {
        let a = self.values();
        let b = other.values();
        let mut out = [0; 5];
        for i in 0..5 {
            out[i] = a[i]
                .checked_add(b[i])
                .ok_or_else(|| Error::Budget("reservation arithmetic overflow".into()))?;
        }
        Ok(Self::from_values(out))
    }
    fn fits(self, maximum: Self) -> bool {
        self.values()
            .iter()
            .zip(maximum.values())
            .all(|(a, b)| *a <= b)
    }
}
#[derive(Clone, Debug)]
pub struct Budget {
    pub maximum: Amounts,
    reserved: Amounts,
    observed_overage: Amounts,
}
impl Budget {
    pub fn new(maximum: Amounts) -> Self {
        Self {
            maximum,
            reserved: Amounts::default(),
            observed_overage: Amounts::default(),
        }
    }
    pub fn reserved(&self) -> Amounts {
        self.reserved
    }
    pub fn remaining(&self) -> Amounts {
        let mut out = [0; 5];
        for (i, (a, b)) in self
            .maximum
            .values()
            .into_iter()
            .zip(self.reserved.values())
            .enumerate()
        {
            out[i] = a
                .saturating_sub(b)
                .saturating_sub(self.observed_overage.values()[i]);
        }
        Amounts::from_values(out)
    }
    pub fn observed_overage(&self) -> Amounts {
        self.observed_overage
    }
    fn observe(&mut self, reservation: Amounts, observed: Option<Amounts>) {
        if let Some(observed) = observed {
            let mut out = self.observed_overage.values();
            for (i, (actual, reserved)) in observed
                .values()
                .into_iter()
                .zip(reservation.values())
                .enumerate()
            {
                out[i] = out[i].saturating_add(actual.saturating_sub(reserved));
            }
            self.observed_overage = Amounts::from_values(out);
        }
    }
    pub fn reserve(&mut self, amount: Amounts) -> Result<()> {
        let next = self.reserved.add(amount)?;
        if !next.add(self.observed_overage)?.fits(self.maximum) {
            return Err(Error::Budget("admission exceeds reservation budget".into()));
        }
        self.reserved = next;
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransactionStatus {
    Admitted,
    Complete {
        observed: Option<Amounts>,
        evidence: Vec<u8>,
    },
}
#[derive(Clone, Debug)]
pub struct Transaction {
    pub reservation: Amounts,
    pub status: TransactionStatus,
}
pub struct Checkpoint {
    directory: PathBuf,
    pinned: PinnedDirectory,
    lock: AdvisoryLock,
    budget: Budget,
    transactions: BTreeMap<String, Transaction>,
    next: u64,
    previous: [u8; 32],
    journal_bytes: u64,
    poisoned: bool,
}
impl Checkpoint {
    /// The identity bytes should bind the full effective plan/configuration.
    /// A single process holds a nonblocking advisory lock for this lifetime.
    pub fn open(directory: &Path, identity: &[u8], maximum: Amounts) -> Result<Self> {
        if identity.is_empty() || identity.len() > 65536 {
            return Err(Error::Invalid(
                "checkpoint identity must contain 1..65536 bytes".into(),
            ));
        }
        let pinned = PinnedDirectory::open(directory, true)?;
        let lock_name = CString::new("collection.lock").unwrap();
        let raw = unsafe {
            libc::openat(
                pinned.as_raw_fd(),
                lock_name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600 as libc::c_uint,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let lock = unsafe { File::from_raw_fd(raw) };
        let metadata = lock.metadata()?;
        if !metadata.is_file()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o077 != 0
        {
            return Err(Error::Checkpoint(
                "collection lock is not a safe regular file".into(),
            ));
        }
        let lock = AdvisoryLock::acquire(lock).map_err(|_| {
            Error::Checkpoint("another collector is already using this checkpoint".into())
        })?;
        let mut header = b"IVLJNL01".to_vec();
        put_bytes(&mut header, identity);
        put_amounts(&mut header, maximum);
        let digest: [u8; 32] = Sha256::digest(&header).into();
        header.extend(digest);
        match pinned.read_regular(Path::new("header.bin"), 65536 + 128) {
            Ok((bytes, _)) => {
                if bytes != header {
                    return Err(Error::Checkpoint(
                        "checkpoint identity differs from this collection".into(),
                    ));
                }
            }
            Err(invarlock_native_filesystem::Error::Io(e))
                if e.kind() == io::ErrorKind::NotFound =>
            {
                write_file_no_replace(&directory.join("header.bin"), &header, 0o600, false)?;
            }
            Err(e) => return Err(e.into()),
        }
        let mut result = Self {
            directory: directory.to_path_buf(),
            pinned,
            lock,
            budget: Budget::new(maximum),
            transactions: BTreeMap::new(),
            next: 1,
            previous: digest,
            journal_bytes: header.len() as u64,
            poisoned: false,
        };
        let names = result.names()?;
        for n in names {
            let expected = format!("{:020}.event", result.next);
            if n != expected {
                return Err(Error::Checkpoint(
                    "checkpoint sequence is incomplete or contains unexpected entries".into(),
                ));
            }
            let (raw, _) = result.pinned.read_regular(Path::new(&n), MAX_RECORD)?;
            result.journal_bytes = result
                .journal_bytes
                .checked_add(raw.len() as u64)
                .ok_or_else(|| Error::Checkpoint("checkpoint size overflow".into()))?;
            if result.journal_bytes > MAX_JOURNAL {
                return Err(Error::Checkpoint("checkpoint byte limit exceeded".into()));
            }
            result.replay(&raw)?;
            result.previous = Sha256::digest(&raw).into();
            result.next += 1;
        }
        result.verify_lock()?;
        Ok(result)
    }
    pub fn budget(&self) -> &Budget {
        &self.budget
    }
    pub fn transactions(&self) -> &BTreeMap<String, Transaction> {
        &self.transactions
    }
    pub fn pending(&self) -> Vec<&str> {
        self.transactions
            .iter()
            .filter_map(|(key, t)| {
                matches!(t.status, TransactionStatus::Admitted).then_some(key.as_str())
            })
            .collect()
    }
    /// Repeated or resumed admission IDs are rejected. The caller must not retry
    /// ambiguous dispatch by inventing an ID without its declared retry policy.
    pub fn admit(&mut self, key: &str, reservation: Amounts) -> Result<()> {
        validate_key(key)?;
        if self.transactions.contains_key(key) {
            return Err(Error::Checkpoint("transaction already admitted".into()));
        }
        if reservation.calls != 1 {
            return Err(Error::Invalid(
                "each admission reserves exactly one call".into(),
            ));
        }
        let mut budget = self.budget.clone();
        budget.reserve(reservation)?;
        let mut payload = vec![1];
        put_bytes(&mut payload, key.as_bytes());
        put_amounts(&mut payload, reservation);
        self.append(&payload)?;
        self.budget = budget;
        self.transactions.insert(
            key.into(),
            Transaction {
                reservation,
                status: TransactionStatus::Admitted,
            },
        );
        Ok(())
    }
    pub fn complete(
        &mut self,
        key: &str,
        observed: Option<Amounts>,
        evidence: &[u8],
    ) -> Result<()> {
        let transaction = self
            .transactions
            .get(key)
            .ok_or_else(|| Error::Checkpoint("completion has no durable admission".into()))?;
        if !matches!(transaction.status, TransactionStatus::Admitted) {
            return Err(Error::Checkpoint("transaction already completed".into()));
        }
        if evidence.len() as u64 > transaction.reservation.retained_bytes {
            return Err(Error::Budget(
                "completion exceeds retained-byte reservation".into(),
            ));
        }
        if evidence.len() > MAX_RECORD - 2048 {
            return Err(Error::Checkpoint("completion exceeds record limit".into()));
        }
        let mut payload = vec![2];
        put_bytes(&mut payload, key.as_bytes());
        payload.push(u8::from(observed.is_some()));
        if let Some(value) = observed {
            put_amounts(&mut payload, value);
        }
        put_bytes(&mut payload, evidence);
        let reservation = transaction.reservation;
        self.append(&payload)?;
        self.budget.observe(reservation, observed);
        self.transactions.get_mut(key).unwrap().status = TransactionStatus::Complete {
            observed,
            evidence: evidence.to_vec(),
        };
        Ok(())
    }
    // libc inode/device widths vary between supported Unix targets.
    #[allow(clippy::unnecessary_cast)]
    fn verify_lock(&self) -> Result<()> {
        self.lock.verify_owner()?;
        if self.poisoned {
            return Err(Error::Checkpoint(
                "checkpoint publication was ambiguous; close and resume before dispatch".into(),
            ));
        }
        self.pinned.verify()?;
        let metadata = self.lock.metadata()?;
        let name = CString::new("collection.lock").unwrap();
        let mut named = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe {
            libc::fstatat(
                self.pinned.as_raw_fd(),
                name.as_ptr(),
                named.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(Error::Checkpoint("collection lock binding changed".into()));
        }
        let named = unsafe { named.assume_init() };
        if named.st_dev as u64 != metadata.dev()
            || named.st_ino as u64 != metadata.ino()
            || named.st_mode as u32 != metadata.mode()
        {
            return Err(Error::Checkpoint("collection lock binding changed".into()));
        }
        Ok(())
    }
    fn names(&self) -> Result<Vec<String>> {
        // Directory binding is checked around enumeration; contents are opened
        // only through pinned descriptors. No discovered directory is traversed.
        self.verify_lock()?;
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Error::Checkpoint("checkpoint name is not UTF-8".into()))?;
            if name == "header.bin" || name == "collection.lock" {
                continue;
            }
            if !entry.file_type()?.is_file()
                || !name.ends_with(".event")
                || names.len() >= MAX_EVENTS
            {
                return Err(Error::Checkpoint(
                    "checkpoint contains unexpected or incomplete entries".into(),
                ));
            }
            names.push(name);
        }
        self.verify_lock()?;
        names.sort();
        Ok(names)
    }
    fn append(&mut self, payload: &[u8]) -> Result<()> {
        self.verify_lock()?;
        if self.next > MAX_EVENTS as u64 {
            return Err(Error::Checkpoint("checkpoint event limit exceeded".into()));
        }
        let mut raw = b"IVLEVT01".to_vec();
        raw.extend(self.next.to_le_bytes());
        raw.extend(self.previous);
        raw.extend(payload);
        let checksum: [u8; 32] = Sha256::digest(&raw).into();
        raw.extend(checksum);
        let next_bytes = self
            .journal_bytes
            .checked_add(raw.len() as u64)
            .ok_or_else(|| Error::Checkpoint("checkpoint size overflow".into()))?;
        if raw.len() > MAX_RECORD || next_bytes > MAX_JOURNAL {
            return Err(Error::Checkpoint("checkpoint byte limit exceeded".into()));
        }
        let path = self.directory.join(format!("{:020}.event", self.next));
        if let Err(e) = write_file_no_replace(&path, &raw, 0o600, false) {
            self.poisoned = true;
            return Err(e.into());
        }
        if let Err(e) = self.verify_lock() {
            self.poisoned = true;
            return Err(e);
        }
        self.previous = Sha256::digest(&raw).into();
        self.next += 1;
        self.journal_bytes = next_bytes;
        Ok(())
    }
    fn replay(&mut self, raw: &[u8]) -> Result<()> {
        if raw.len() < 81 {
            return Err(Error::Checkpoint("checkpoint event is truncated".into()));
        }
        let (content, checksum) = raw.split_at(raw.len() - 32);
        if Sha256::digest(content).as_slice() != checksum {
            return Err(Error::Checkpoint("checkpoint checksum differs".into()));
        }
        let mut cursor = Cursor(content);
        if cursor.take(8)? != b"IVLEVT01"
            || cursor.u64()? != self.next
            || cursor.take(32)? != self.previous
        {
            return Err(Error::Checkpoint("checkpoint event chain differs".into()));
        }
        let kind = cursor.byte()?;
        let key = String::from_utf8(cursor.bytes(1024)?.to_vec())
            .map_err(|_| Error::Checkpoint("checkpoint key is not UTF-8".into()))?;
        validate_key(&key)?;
        match kind {
            1 => {
                let reservation = cursor.amounts()?;
                if reservation.calls != 1 || self.transactions.contains_key(&key) {
                    return Err(Error::Checkpoint(
                        "duplicate or invalid checkpoint admission".into(),
                    ));
                }
                self.budget.reserve(reservation)?;
                self.transactions.insert(
                    key,
                    Transaction {
                        reservation,
                        status: TransactionStatus::Admitted,
                    },
                );
            }
            2 => {
                let observed = match cursor.byte()? {
                    0 => None,
                    1 => Some(cursor.amounts()?),
                    _ => return Err(Error::Checkpoint("invalid observed-usage tag".into())),
                };
                let evidence = cursor.bytes(MAX_RECORD)?.to_vec();
                let transaction = self
                    .transactions
                    .get_mut(&key)
                    .ok_or_else(|| Error::Checkpoint("completion has no admission".into()))?;
                if !matches!(transaction.status, TransactionStatus::Admitted)
                    || evidence.len() as u64 > transaction.reservation.retained_bytes
                {
                    return Err(Error::Checkpoint(
                        "duplicate or unreserved completion".into(),
                    ));
                }
                self.budget.observe(transaction.reservation, observed);
                transaction.status = TransactionStatus::Complete { observed, evidence };
            }
            _ => return Err(Error::Checkpoint("unknown checkpoint event".into())),
        }
        if !cursor.0.is_empty() {
            return Err(Error::Checkpoint(
                "checkpoint event has trailing bytes".into(),
            ));
        }
        Ok(())
    }
}
fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.len() > 1024 || key.chars().any(char::is_control) {
        return Err(Error::Invalid(
            "transaction key must contain 1..1024 non-control UTF-8 bytes".into(),
        ));
    }
    Ok(())
}
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend((bytes.len() as u64).to_le_bytes());
    out.extend(bytes);
}
fn put_amounts(out: &mut Vec<u8>, amount: Amounts) {
    for n in amount.values() {
        out.extend(n.to_le_bytes());
    }
}
struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(Error::Checkpoint("checkpoint event is truncated".into()));
        }
        let (part, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(part)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn bytes(&mut self, max: usize) -> Result<&'a [u8]> {
        let length = self.u64()?;
        if length > max as u64 {
            return Err(Error::Checkpoint(
                "checkpoint field exceeds byte limit".into(),
            ));
        }
        self.take(length as usize)
    }
    fn amounts(&mut self) -> Result<Amounts> {
        let mut values = [0; 5];
        for value in &mut values {
            *value = self.u64()?;
        }
        Ok(Amounts::from_values(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn temp() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "invarlock-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::canonicalize(path).unwrap()
    }
    fn allowance() -> Amounts {
        Amounts {
            calls: 2,
            input_tokens: 200,
            output_tokens: 200,
            cost_microusd: 200,
            retained_bytes: 200,
        }
    }
    fn reservation() -> Amounts {
        Amounts {
            calls: 1,
            input_tokens: 100,
            output_tokens: 100,
            cost_microusd: 100,
            retained_bytes: 100,
        }
    }
    #[test]
    fn interrupted_admission_remains_charged_on_resume() {
        let path = temp();
        let inherited;
        {
            let mut c = Checkpoint::open(&path, b"config", allowance()).unwrap();
            inherited = c.lock.test_duplicate().unwrap();
            c.admit("one", reservation()).unwrap();
        }
        let mut c = Checkpoint::open(&path, b"config", allowance()).unwrap();
        drop(inherited);
        assert_eq!(c.pending(), vec!["one"]);
        assert_eq!(c.budget().reserved().calls, 1);
        assert!(c.admit("one", reservation()).is_err());
        c.complete("one", None, b"retained").unwrap();
        drop(c);
        let c = Checkpoint::open(&path, b"config", allowance()).unwrap();
        assert!(c.pending().is_empty());
        assert!(matches!(
            c.transactions()["one"].status,
            TransactionStatus::Complete { observed: None, .. }
        ));
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn foreign_process_checkpoint_rejects_mutation_before_publication() {
        let path = temp();
        let mut checkpoint = Checkpoint::open(&path, b"config", allowance()).unwrap();
        checkpoint.lock.test_set_owner_pid(0);
        let error = checkpoint.admit("foreign", reservation()).unwrap_err();
        assert!(matches!(error, Error::Io(e) if e.kind() == io::ErrorKind::PermissionDenied));
        assert!(!path.join("00000000000000000001.event").exists());
        assert_eq!(checkpoint.budget().reserved().calls, 0);
        checkpoint
            .lock
            .test_set_owner_pid(unsafe { libc::getpid() });
        checkpoint.admit("owner", reservation()).unwrap();
        assert!(path.join("00000000000000000001.event").is_file());
        drop(checkpoint);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn budget_is_atomic_and_never_refunds() {
        let mut b = Budget::new(allowance());
        b.reserve(reservation()).unwrap();
        b.reserve(reservation()).unwrap();
        let before = b.reserved();
        assert!(b.reserve(reservation()).is_err());
        assert_eq!(b.reserved(), before);
    }
    #[test]
    fn mismatched_config_and_double_collector_rejected() {
        let path = temp();
        let c = Checkpoint::open(&path, b"config", allowance()).unwrap();
        assert!(Checkpoint::open(&path, b"config", allowance()).is_err());
        drop(c);
        assert!(Checkpoint::open(&path, b"changed", allowance()).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn truncated_and_missing_events_rejected() {
        let path = temp();
        {
            let mut c = Checkpoint::open(&path, b"config", allowance()).unwrap();
            c.admit("one", reservation()).unwrap();
            c.complete("one", None, b"done").unwrap();
        }
        std::fs::remove_file(path.join("00000000000000000001.event")).unwrap();
        assert!(Checkpoint::open(&path, b"config", allowance()).is_err());
        std::fs::write(path.join("00000000000000000001.event"), b"truncated").unwrap();
        assert!(Checkpoint::open(&path, b"config", allowance()).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn completed_usage_may_exceed_reservation_without_rewriting_history() {
        let path = temp();
        let mut c = Checkpoint::open(&path, b"config", allowance()).unwrap();
        c.admit("one", reservation()).unwrap();
        let observed = Amounts {
            cost_microusd: 999,
            ..Amounts::default()
        };
        c.complete("one", Some(observed), b"response").unwrap();
        assert_eq!(c.budget().reserved().cost_microusd, 100);
        assert_eq!(c.budget().observed_overage().cost_microusd, 899);
        assert!(c.admit("two", reservation()).is_err());
        drop(c);
        let c = Checkpoint::open(&path, b"config", allowance()).unwrap();
        assert!(matches!(
            c.transactions()["one"].status,
            TransactionStatus::Complete {
                observed: Some(Amounts {
                    cost_microusd: 999,
                    ..
                }),
                ..
            }
        ));
        std::fs::remove_dir_all(path).unwrap();
    }
}
