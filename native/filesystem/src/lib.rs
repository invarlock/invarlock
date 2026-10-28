//! Offline filesystem authority. No process launch, network, or Python dependency.
//! Descriptor binding checks detect mutation; they are not isolation from another
//! process running as the same user. Publication errors after rename are ambiguous
//! and never remove the destination by pathname.
#![deny(unsafe_op_in_unsafe_fn)]
#[cfg(not(unix))]
compile_error!("descriptor-pinned filesystem support currently requires Unix");
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{CStr, CString, OsStr},
    fs::File,
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::{Component, Path},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    UnsafePath(String),
    Changed(String),
    Limit(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::UnsafePath(s) | Self::Changed(s) | Self::Limit(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub device: u64,
    pub inode: u64,
    pub mode: u32,
    pub size: u64,
    pub modified_ns: i128,
    pub changed_ns: i128,
}
impl FileIdentity {
    fn metadata(m: &std::fs::Metadata) -> Self {
        Self {
            device: m.dev(),
            inode: m.ino(),
            mode: m.mode(),
            size: m.size(),
            modified_ns: m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128,
            changed_ns: m.ctime() as i128 * 1_000_000_000 + m.ctime_nsec() as i128,
        }
    }
    // libc integer widths vary between supported Unix targets.
    #[allow(clippy::unnecessary_cast)]
    fn stat(s: &libc::stat) -> Self {
        Self {
            device: s.st_dev as u64,
            inode: s.st_ino as u64,
            mode: s.st_mode as u32,
            size: s.st_size as u64,
            modified_ns: s.st_mtime as i128 * 1_000_000_000 + s.st_mtime_nsec as i128,
            changed_ns: s.st_ctime as i128 * 1_000_000_000 + s.st_ctime_nsec as i128,
        }
    }
    fn binding_eq(&self, o: &Self) -> bool {
        self.device == o.device && self.inode == o.inode && self.mode == o.mode
    }
    fn contents_eq(&self, o: &Self) -> bool {
        self.binding_eq(o) && self.size == o.size && self.modified_ns == o.modified_ns
    }
    fn directory(&self) -> bool {
        self.mode & libc::S_IFMT as u32 == libc::S_IFDIR as u32
    }
    fn regular(&self) -> bool {
        self.mode & libc::S_IFMT as u32 == libc::S_IFREG as u32
    }
}
fn name(s: &OsStr) -> Result<CString> {
    let b = s.as_bytes();
    if b.is_empty() || b == b"." || b == b".." || b.contains(&b'/') {
        return Err(Error::UnsafePath(
            "expected one safe directory entry".into(),
        ));
    }
    CString::new(b).map_err(|_| Error::UnsafePath("path contains NUL".into()))
}
fn stat_at(fd: RawFd, n: &CStr) -> Result<FileIdentity> {
    let mut s = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstatat(fd, n.as_ptr(), s.as_mut_ptr(), libc::AT_SYMLINK_NOFOLLOW) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(FileIdentity::stat(&unsafe { s.assume_init() }))
}
fn open_at(fd: RawFd, n: &CStr, flags: i32, mode: u32) -> Result<File> {
    let raw = unsafe {
        libc::openat(
            fd,
            n.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            mode as libc::c_uint,
        )
    };
    if raw < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_fd(raw) })
}
// Match the Python filesystem contract: synchronize file data and metadata with
// fsync at every existing publication boundary. On Apple, File::sync_all uses
// the stronger F_FULLFSYNC operation instead; sync_data does the same.
fn fsync_once(file: &File) -> io::Result<()> {
    #[cfg(test)]
    tests::before_sync()?;
    if unsafe { libc::fsync(file.as_raw_fd()) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
fn fsync_file(file: &File) -> io::Result<()> {
    loop {
        match fsync_once(file) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => return result,
        }
    }
}
fn id(f: &File) -> Result<FileIdentity> {
    Ok(FileIdentity::metadata(&f.metadata()?))
}
fn directory_at(fd: RawFd, n: &CStr) -> Result<File> {
    let before = stat_at(fd, n)?;
    if !before.directory() {
        return Err(Error::UnsafePath(
            "path must use non-symlink directories".into(),
        ));
    }
    let file = open_at(fd, n, libc::O_RDONLY | libc::O_DIRECTORY, 0)?;
    if !before.binding_eq(&id(&file)?) {
        return Err(Error::Changed("directory changed while opening".into()));
    }
    Ok(file)
}
fn mkdir_at(fd: RawFd, n: &CStr) -> Result<()> {
    if unsafe { libc::mkdirat(fd, n.as_ptr(), 0o700) } < 0 {
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::AlreadyExists {
            return Err(e.into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryMetadata {
    pub identity: FileIdentity,
    pub uid: u32,
    pub gid: u32,
    pub links: u64,
}
fn entry_metadata(file: &File) -> Result<EntryMetadata> {
    let metadata = file.metadata()?;
    Ok(EntryMetadata {
        identity: FileIdentity::metadata(&metadata),
        uid: metadata.uid(),
        gid: metadata.gid(),
        links: metadata.nlink(),
    })
}
/// An open regular-file capability without reading or allocating its payload.
pub struct PinnedRegularFile {
    parent: PinnedDirectory,
    name: CString,
    file: File,
    metadata: EntryMetadata,
    content_digest: OnceLock<String>,
}
impl PinnedRegularFile {
    pub fn open(path: &Path) -> Result<Self> {
        let parent = PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), false)?;
        let name = name(
            path.file_name()
                .ok_or_else(|| Error::UnsafePath("missing file name".into()))?,
        )?;
        let expected = stat_at(parent.as_raw_fd(), &name)?;
        if !expected.regular() {
            return Err(Error::UnsafePath(
                "input must be a non-symlink regular file".into(),
            ));
        }
        let file = open_at(
            parent.as_raw_fd(),
            &name,
            libc::O_RDONLY | libc::O_NONBLOCK,
            0,
        )?;
        let metadata = entry_metadata(&file)?;
        if expected != metadata.identity {
            return Err(Error::Changed("input changed while opening".into()));
        }
        let result = Self {
            parent,
            name,
            file,
            metadata,
            content_digest: OnceLock::new(),
        };
        result.verify()?;
        Ok(result)
    }
    /// Stream a pinned file without an input-sized allocation. The first
    /// completed read binds its digest; later reads must match those bytes.
    pub fn copy_to(&self, output: &mut impl Write, max_bytes: u64) -> Result<u64> {
        use std::os::unix::fs::FileExt;
        self.verify_binding()?;
        let size = self.metadata.identity.size;
        if size > max_bytes {
            return Err(Error::Limit("copy byte limit exceeded".into()));
        }
        let mut offset = 0;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        while offset < size {
            let wanted = (size - offset).min(buffer.len() as u64) as usize;
            let count = match self.file.read_at(&mut buffer[..wanted], offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                return Err(Error::Changed("copy source shrank".into()));
            }
            hash.update(&buffer[..count]);
            output.write_all(&buffer[..count])?;
            offset += count as u64;
        }
        self.verify_binding()?;
        self.bind_digest(format!("{:x}", hash.finalize()))?;
        Ok(offset)
    }
    fn bind_digest(&self, digest: String) -> Result<()> {
        let expected = self.content_digest.get_or_init(|| digest.clone());
        if expected != &digest {
            return Err(Error::Changed(
                "pinned regular file contents changed".into(),
            ));
        }
        Ok(())
    }
    pub fn sha256_digest(&self, max_bytes: u64) -> Result<String> {
        self.verify_binding()?;
        let digest = hash_file(&self.file, &self.metadata.identity, max_bytes)?;
        self.verify_binding()?;
        self.bind_digest(digest.clone())?;
        Ok(digest)
    }
    /// Append raw bytes to a caller-owned digest. Discard that digest on error.
    pub fn hash_into(&self, hash: &mut Sha256, max_bytes: u64) -> Result<()> {
        struct HashWriter<'a>(&'a mut Sha256);
        impl Write for HashWriter<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        self.copy_to(&mut HashWriter(hash), max_bytes)?;
        Ok(())
    }
    pub fn metadata(&self) -> &EntryMetadata {
        &self.metadata
    }
    /// Check inode, path and metadata only. A caller using this weaker check
    /// must independently bind every consumed byte to its expected digest.
    pub fn verify_binding(&self) -> Result<()> {
        if id(&self.file)? != self.metadata.identity
            || stat_at(self.parent.as_raw_fd(), &self.name)? != self.metadata.identity
        {
            return Err(Error::Changed("pinned regular file changed".into()));
        }
        self.parent.verify()
    }
    /// Recheck the bytes of a completed read as well as the retained binding.
    pub fn verify(&self) -> Result<()> {
        self.verify_binding()?;
        if let Some(expected) = self.content_digest.get() {
            let actual = hash_file(
                &self.file,
                &self.metadata.identity,
                self.metadata.identity.size,
            )?;
            if &actual != expected {
                return Err(Error::Changed(
                    "pinned regular file contents changed".into(),
                ));
            }
            self.verify_binding()?;
        }
        Ok(())
    }
}
impl AsRawFd for PinnedRegularFile {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

fn hash_file(file: &File, expected: &FileIdentity, max_bytes: u64) -> Result<String> {
    let mut hash = Sha256::new();
    hash_file_into(file, expected, max_bytes, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}
fn hash_file_into(
    file: &File,
    expected: &FileIdentity,
    max_bytes: u64,
    hash: &mut Sha256,
) -> Result<()> {
    use std::os::unix::fs::FileExt;
    if expected.size > max_bytes {
        return Err(Error::Limit("input exceeds hash byte limit".into()));
    }
    if id(file)? != *expected {
        return Err(Error::Changed("input changed before hashing".into()));
    }
    let mut offset = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    while offset < expected.size {
        let wanted = (expected.size - offset).min(buffer.len() as u64) as usize;
        let count = match file.read_at(&mut buffer[..wanted], offset) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        if count == 0 {
            return Err(Error::Changed("input changed while hashing".into()));
        }
        hash.update(&buffer[..count]);
        offset += count as u64;
    }
    if id(file)? != *expected {
        return Err(Error::Changed("input changed while hashing".into()));
    }
    Ok(())
}

/// Retains every ancestor until verification, avoiding path-based reopen races.
pub struct PinnedDirectory {
    directories: Vec<File>,
    names: Vec<CString>,
}
impl PinnedDirectory {
    pub fn open(path: &Path, create: bool) -> Result<Self> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let mut result = Self {
            directories: vec![File::open("/")?],
            names: vec![],
        };
        for c in absolute.components() {
            match c {
                Component::RootDir | Component::CurDir => (),
                Component::Normal(s) => {
                    let n = name(s)?;
                    if create {
                        mkdir_at(result.as_raw_fd(), &n)?;
                    }
                    let file = directory_at(result.as_raw_fd(), &n)?;
                    result.names.push(n);
                    result.directories.push(file);
                }
                _ => return Err(Error::UnsafePath("parent traversal is forbidden".into())),
            }
        }
        result.verify()?;
        Ok(result)
    }
    pub fn verify(&self) -> Result<()> {
        for (i, n) in self.names.iter().enumerate() {
            let expected = id(&self.directories[i + 1])?;
            if !expected.binding_eq(
                &stat_at(self.directories[i].as_raw_fd(), n)
                    .map_err(|_| Error::Changed("directory source was replaced".into()))?,
            ) {
                return Err(Error::Changed("directory source was replaced".into()));
            }
        }
        Ok(())
    }
    pub fn read_regular(&self, relative: &Path, limit: usize) -> Result<(Vec<u8>, FileIdentity)> {
        let components = safe_relative(relative)?;
        let mut descendants = Vec::new();
        let mut parent = self.as_raw_fd();
        for n in &components[..components.len() - 1] {
            let f = directory_at(parent, n)?;
            parent = f.as_raw_fd();
            descendants.push(f);
        }
        let leaf = components.last().unwrap();
        let result = read_at(parent, leaf, limit)?;
        parent = self.as_raw_fd();
        for (n, f) in components.iter().zip(descendants.iter()) {
            if !id(f)?.binding_eq(&stat_at(parent, n)?) {
                return Err(Error::Changed("directory changed while reading".into()));
            }
            parent = f.as_raw_fd();
        }
        self.verify()?;
        Ok(result)
    }
    pub fn metadata(&self) -> Result<EntryMetadata> {
        entry_metadata(self.directories.last().unwrap())
    }
    pub fn sha256_file(&self, relative: &Path, max_bytes: u64) -> Result<String> {
        let mut hash = Sha256::new();
        self.hash_file_into(relative, &mut hash, max_bytes)?;
        Ok(format!("{:x}", hash.finalize()))
    }
    pub fn hash_file_into(&self, relative: &Path, hash: &mut Sha256, max_bytes: u64) -> Result<()> {
        let components = safe_relative(relative)?;
        let mut descendants = Vec::new();
        let mut parent = self.as_raw_fd();
        for n in &components[..components.len() - 1] {
            let child = directory_at(parent, n)?;
            parent = child.as_raw_fd();
            descendants.push(child);
        }
        let leaf = components.last().unwrap();
        let expected = stat_at(parent, leaf)?;
        if !expected.regular() {
            return Err(Error::UnsafePath(
                "hash input must be a regular file".into(),
            ));
        }
        let file = open_at(parent, leaf, libc::O_RDONLY | libc::O_NONBLOCK, 0)?;
        hash_file_into(&file, &expected, max_bytes, hash)?;
        if stat_at(parent, leaf)? != expected {
            return Err(Error::Changed("input changed while hashing".into()));
        }
        parent = self.as_raw_fd();
        for (n, file) in components.iter().zip(descendants.iter()) {
            if !id(file)?.binding_eq(&stat_at(parent, n)?) {
                return Err(Error::Changed("directory changed while hashing".into()));
            }
            parent = file.as_raw_fd();
        }
        self.verify()?;
        Ok(())
    }
    pub fn inspect(&self, relative: &Path) -> Result<EntryMetadata> {
        let components = safe_relative(relative)?;
        let mut descendants = Vec::new();
        let mut parent = self.as_raw_fd();
        for n in &components[..components.len() - 1] {
            let f = directory_at(parent, n)?;
            parent = f.as_raw_fd();
            descendants.push(f);
        }
        let leaf = components.last().unwrap();
        let before = stat_at(parent, leaf)?;
        if !before.regular() && !before.directory() {
            return Err(Error::UnsafePath(
                "entry must be a regular file or directory".into(),
            ));
        }
        let file = open_at(
            parent,
            leaf,
            libc::O_RDONLY
                | libc::O_NONBLOCK
                | if before.directory() {
                    libc::O_DIRECTORY
                } else {
                    0
                },
            0,
        )?;
        let metadata = entry_metadata(&file)?;
        if metadata.identity != before || stat_at(parent, leaf)? != before {
            return Err(Error::Changed("entry changed while inspecting".into()));
        }
        parent = self.as_raw_fd();
        for (n, f) in components.iter().zip(descendants.iter()) {
            if !id(f)?.binding_eq(&stat_at(parent, n)?) {
                return Err(Error::Changed("directory changed while inspecting".into()));
            }
            parent = f.as_raw_fd();
        }
        self.verify()?;
        Ok(metadata)
    }
    /// List only regular files directly beneath this retained directory. Never
    /// descend an unexpected directory, including during failed checkpoint replay.
    pub fn flat_regular_inventory(
        &self,
        limits: SnapshotLimits,
    ) -> Result<BTreeMap<String, FileIdentity>> {
        fn read(
            root: &PinnedDirectory,
            limits: SnapshotLimits,
        ) -> Result<BTreeMap<String, FileIdentity>> {
            use std::os::fd::IntoRawFd;
            let dot = CString::new(".").unwrap();
            let raw = open_at(
                root.as_raw_fd(),
                &dot,
                libc::O_RDONLY | libc::O_DIRECTORY,
                0,
            )?
            .into_raw_fd();
            let directory = unsafe { libc::fdopendir(raw) };
            if directory.is_null() {
                unsafe { libc::close(raw) };
                return Err(io::Error::last_os_error().into());
            }
            struct Listing(*mut libc::DIR);
            impl Drop for Listing {
                fn drop(&mut self) {
                    unsafe { libc::closedir(self.0) };
                }
            }
            let listing = Listing(directory);
            let mut result = BTreeMap::new();
            let mut bytes = 0u64;
            loop {
                set_errno_zero();
                let entry = unsafe { libc::readdir(listing.0) };
                if entry.is_null() {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() != Some(0) {
                        return Err(error.into());
                    }
                    break;
                }
                let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
                if matches!(name.to_bytes(), b"." | b"..") {
                    continue;
                }
                let text = name
                    .to_str()
                    .map_err(|_| Error::UnsafePath("checkpoint paths must be UTF-8".into()))?;
                let identity = stat_at(root.as_raw_fd(), name)?;
                if !identity.regular() {
                    return Err(Error::UnsafePath(format!(
                        "unexpected non-regular checkpoint entry: {text}"
                    )));
                }
                bytes = bytes
                    .checked_add(identity.size)
                    .ok_or_else(|| Error::Limit("inventory size overflow".into()))?;
                if result.len() >= limits.max_entries
                    || result.len() >= limits.max_files
                    || identity.size > limits.max_file_bytes as u64
                    || bytes > limits.max_total_bytes as u64
                {
                    return Err(Error::Limit("flat inventory limit exceeded".into()));
                }
                result.insert(text.to_string(), identity);
            }
            root.verify()?;
            Ok(result)
        }
        self.verify()?;
        let first = read(self, limits)?;
        if first != read(self, limits)? {
            return Err(Error::Changed(
                "flat inventory changed during inspection".into(),
            ));
        }
        Ok(first)
    }
    pub fn inventory_metadata(
        &self,
        limits: SnapshotLimits,
    ) -> Result<BTreeMap<String, EntryMetadata>> {
        self.inventory_metadata_filtered(limits, &BTreeSet::new())
    }
    pub fn inventory_metadata_filtered(
        &self,
        limits: SnapshotLimits,
        excluded_dirs: &BTreeSet<String>,
    ) -> Result<BTreeMap<String, EntryMetadata>> {
        let before = inventory_filtered(self, limits, excluded_dirs)?;
        let mut entries = BTreeMap::new();
        for (relative, expected) in &before {
            let metadata = self.inspect(Path::new(relative))?;
            if &metadata.identity != expected {
                return Err(Error::Changed("inventory changed during inspection".into()));
            }
            entries.insert(relative.clone(), metadata);
        }
        if inventory_filtered(self, limits, excluded_dirs)? != before {
            return Err(Error::Changed("inventory changed during inspection".into()));
        }
        Ok(entries)
    }
    pub fn verify_regular(&self, relative: &Path, expected: &FileIdentity) -> Result<()> {
        let components = safe_relative(relative)?;
        let mut descendants = Vec::new();
        let mut parent = self.as_raw_fd();
        for n in &components[..components.len() - 1] {
            let f = directory_at(parent, n)?;
            parent = f.as_raw_fd();
            descendants.push(f);
        }
        let leaf = components.last().unwrap();
        if &stat_at(parent, leaf)? != expected {
            return Err(Error::Changed("file identity changed".into()));
        }
        parent = self.as_raw_fd();
        for (n, f) in components.iter().zip(descendants.iter()) {
            if !id(f)?.binding_eq(&stat_at(parent, n)?) {
                return Err(Error::Changed("directory changed while inspecting".into()));
            }
            parent = f.as_raw_fd();
        }
        self.verify()
    }
    pub fn capture_selected(
        &self,
        limits: SnapshotLimits,
        files: &BTreeMap<String, usize>,
        dirs: &BTreeSet<String>,
    ) -> Result<SelectedSnapshot<'_>> {
        for path in files.keys().chain(dirs.iter()) {
            safe_relative(Path::new(path))?;
        }
        let before = selected_inventory(self, limits, files, dirs)?;
        let mut entries = BTreeMap::new();
        for (relative, limit) in files {
            let (bytes, identity) = self.read_regular(Path::new(relative), *limit)?;
            if before.get(relative) != Some(&identity) {
                return Err(Error::Changed(
                    "snapshot input changed before capture".into(),
                ));
            }
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            entries.insert(
                relative.clone(),
                SnapshotEntry {
                    bytes: bytes.into_boxed_slice(),
                    identity,
                    sha256,
                },
            );
        }
        if selected_inventory(self, limits, files, dirs)? != before {
            return Err(Error::Changed(
                "snapshot inventory changed during capture".into(),
            ));
        }
        self.verify()?;
        Ok(SelectedSnapshot {
            root: self,
            inventory: before,
            digests: entries
                .iter()
                .map(|(name, entry)| (name.clone(), entry.sha256.clone()))
                .collect(),
            entries,
            limits,
            files: files.clone(),
            dirs: dirs.clone(),
        })
    }
    pub fn sync(&self) -> Result<()> {
        fsync_file(self.directories.last().unwrap())?;
        Ok(())
    }
}
impl AsRawFd for PinnedDirectory {
    fn as_raw_fd(&self) -> RawFd {
        self.directories.last().unwrap().as_raw_fd()
    }
}
fn safe_relative(p: &Path) -> Result<Vec<CString>> {
    let mut out = Vec::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(name(s)?),
            _ => {
                return Err(Error::UnsafePath(
                    "expected a relative path without traversal".into(),
                ));
            }
        }
    }
    if out.is_empty() {
        return Err(Error::UnsafePath("empty file path".into()));
    }
    Ok(out)
}
fn read_at(parent: RawFd, n: &CStr, limit: usize) -> Result<(Vec<u8>, FileIdentity)> {
    let before = stat_at(parent, n)?;
    if !before.regular() {
        return Err(Error::UnsafePath(
            "input must be a non-symlink regular file".into(),
        ));
    }
    if before.size > limit as u64 {
        return Err(Error::Limit("input exceeds byte limit".into()));
    }
    let mut f = open_at(parent, n, libc::O_RDONLY | libc::O_NONBLOCK, 0)?;
    if id(&f)? != before {
        return Err(Error::Changed("input changed while opening".into()));
    }
    let mut bytes = Vec::with_capacity(before.size as usize);
    (&mut f)
        .take(before.size.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != before.size || id(&f)? != before || stat_at(parent, n)? != before {
        return Err(Error::Changed("input changed while reading".into()));
    }
    Ok((bytes, before))
}
pub fn read_file(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let n = path
        .file_name()
        .ok_or_else(|| Error::UnsafePath("missing file name".into()))?;
    let d = PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), false)?;
    Ok(d.read_regular(Path::new(n), max_bytes)?.0)
}

#[derive(Clone, Copy, Debug)]
pub struct SnapshotLimits {
    pub max_entries: usize,
    pub max_files: usize,
    pub max_file_bytes: usize,
    pub max_total_bytes: usize,
}
impl Default for SnapshotLimits {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_files: 128,
            max_file_bytes: 64 * 1024 * 1024,
            max_total_bytes: 512 * 1024 * 1024,
        }
    }
}
#[derive(Debug)]
pub struct SnapshotEntry {
    bytes: Box<[u8]>,
    sha256: String,
    identity: FileIdentity,
}
impl SnapshotEntry {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }
}
pub struct SelectedSnapshot<'a> {
    root: &'a PinnedDirectory,
    inventory: BTreeMap<String, FileIdentity>,
    entries: BTreeMap<String, SnapshotEntry>,
    digests: BTreeMap<String, String>,
    limits: SnapshotLimits,
    files: BTreeMap<String, usize>,
    dirs: BTreeSet<String>,
}
impl<'a> SelectedSnapshot<'a> {
    pub fn entries(&self) -> &BTreeMap<String, SnapshotEntry> {
        &self.entries
    }
    pub fn verify_source(&self) -> Result<()> {
        verify_selected_source(
            self.root,
            &self.inventory,
            self.limits,
            &self.files,
            &self.dirs,
            &self.digests,
        )
    }
    /// Separate captured entries from the unchanged source verification state,
    /// allowing a consumer to release each entry as it finishes decoding it.
    pub fn into_parts(
        self,
    ) -> (
        BTreeMap<String, SnapshotEntry>,
        SelectedSnapshotSourceGuard<'a>,
    ) {
        let Self {
            root,
            inventory,
            entries,
            digests,
            limits,
            files,
            dirs,
        } = self;
        let guard = SelectedSnapshotSourceGuard {
            root,
            inventory,
            digests,
            limits,
            files,
            dirs,
        };
        (entries, guard)
    }
    /// Release captured payloads while retaining every source identity and
    /// inventory check needed for a final verification after replay.
    pub fn into_source_guard(self) -> SelectedSnapshotSourceGuard<'a> {
        let (entries, guard) = self.into_parts();
        drop(entries);
        guard
    }
}
/// Compact source identities and content digests retained after payload release.
/// Verification rereads bounded source bytes; timestamps alone are insufficient.
pub struct SelectedSnapshotSourceGuard<'a> {
    root: &'a PinnedDirectory,
    inventory: BTreeMap<String, FileIdentity>,
    digests: BTreeMap<String, String>,
    limits: SnapshotLimits,
    files: BTreeMap<String, usize>,
    dirs: BTreeSet<String>,
}
impl SelectedSnapshotSourceGuard<'_> {
    pub fn verify_source(&self) -> Result<()> {
        verify_selected_source(
            self.root,
            &self.inventory,
            self.limits,
            &self.files,
            &self.dirs,
            &self.digests,
        )
    }
}
fn verify_selected_source(
    root: &PinnedDirectory,
    inventory: &BTreeMap<String, FileIdentity>,
    limits: SnapshotLimits,
    files: &BTreeMap<String, usize>,
    dirs: &BTreeSet<String>,
    digests: &BTreeMap<String, String>,
) -> Result<()> {
    root.verify()?;
    if selected_inventory(root, limits, files, dirs)? != *inventory {
        return Err(Error::Changed(
            "snapshot sources changed after capture".into(),
        ));
    }
    for (name, expected) in digests {
        if root.sha256_file(Path::new(name), files[name] as u64)? != *expected {
            return Err(Error::Changed(
                "snapshot source contents changed after capture".into(),
            ));
        }
        root.verify_regular(Path::new(name), &inventory[name])?;
    }
    root.verify()
}
pub struct PackSnapshot {
    root: PinnedDirectory,
    inventory: BTreeMap<String, FileIdentity>,
    entries: BTreeMap<String, SnapshotEntry>,
    limits: SnapshotLimits,
}
impl PackSnapshot {
    pub fn capture(path: &Path, limits: SnapshotLimits) -> Result<Self> {
        let root = PinnedDirectory::open(path, false)?;
        let inventory = inventory(&root, limits)?;
        let mut entries = BTreeMap::new();
        for (relative, expected) in &inventory {
            if expected.regular() {
                let (bytes, identity) =
                    root.read_regular(Path::new(relative), limits.max_file_bytes)?;
                if identity != *expected {
                    return Err(Error::Changed(
                        "snapshot input changed before capture".into(),
                    ));
                }
                let sha256 = format!("{:x}", Sha256::digest(&bytes));
                entries.insert(
                    relative.clone(),
                    SnapshotEntry {
                        bytes: bytes.into_boxed_slice(),
                        sha256,
                        identity,
                    },
                );
            }
        }
        let result = Self {
            root,
            inventory,
            entries,
            limits,
        };
        result.verify_source()?;
        Ok(result)
    }
    pub fn entries(&self) -> &BTreeMap<String, SnapshotEntry> {
        &self.entries
    }
    pub fn inventory(&self) -> &BTreeMap<String, FileIdentity> {
        &self.inventory
    }
    pub fn verify_source(&self) -> Result<()> {
        self.root.verify()?;
        if inventory(&self.root, self.limits)? != self.inventory {
            return Err(Error::Changed("pack snapshot changed after capture".into()));
        }
        for (name, entry) in &self.entries {
            if self
                .root
                .sha256_file(Path::new(name), self.limits.max_file_bytes as u64)?
                != entry.sha256
            {
                return Err(Error::Changed(
                    "pack snapshot contents changed after capture".into(),
                ));
            }
            self.root.verify_regular(Path::new(name), &entry.identity)?;
        }
        self.root.verify()
    }
}
fn inventory(
    root: &PinnedDirectory,
    limits: SnapshotLimits,
) -> Result<BTreeMap<String, FileIdentity>> {
    inventory_filtered(root, limits, &BTreeSet::new())
}
fn walk(
    fd: RawFd,
    prefix: &str,
    limits: SnapshotLimits,
    out: &mut BTreeMap<String, FileIdentity>,
    files: &mut usize,
    bytes: &mut u64,
) -> Result<()> {
    walk_filtered(fd, prefix, limits, out, files, bytes, &BTreeSet::new())
}
fn inventory_filtered(
    root: &PinnedDirectory,
    limits: SnapshotLimits,
    excluded_dirs: &BTreeSet<String>,
) -> Result<BTreeMap<String, FileIdentity>> {
    let mut out = BTreeMap::new();
    let mut files = 0;
    let mut bytes = 0u64;
    walk_filtered(
        root.as_raw_fd(),
        "",
        limits,
        &mut out,
        &mut files,
        &mut bytes,
        excluded_dirs,
    )?;
    root.verify()?;
    Ok(out)
}
fn walk_filtered(
    fd: RawFd,
    prefix: &str,
    limits: SnapshotLimits,
    out: &mut BTreeMap<String, FileIdentity>,
    files: &mut usize,
    bytes: &mut u64,
    excluded_dirs: &BTreeSet<String>,
) -> Result<()> {
    // A separate open file description prevents readdir from sharing an offset.
    let dot = CString::new(".").unwrap();
    let listing = open_at(fd, &dot, libc::O_RDONLY | libc::O_DIRECTORY, 0)?;
    use std::os::fd::IntoRawFd;
    let raw = listing.into_raw_fd();
    let dir = unsafe { libc::fdopendir(raw) };
    if dir.is_null() {
        unsafe { libc::close(raw) };
        return Err(io::Error::last_os_error().into());
    }
    struct Dir(*mut libc::DIR);
    impl Drop for Dir {
        fn drop(&mut self) {
            unsafe { libc::closedir(self.0) };
        }
    }
    let guard = Dir(dir);
    loop {
        set_errno_zero();
        let entry = unsafe { libc::readdir(guard.0) };
        if entry.is_null() {
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(0) {
                return Err(e.into());
            }
            break;
        }
        let n = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if n.to_bytes() == b"." || n.to_bytes() == b".." {
            continue;
        }
        if out.len() >= limits.max_entries {
            return Err(Error::Limit("snapshot entry limit exceeded".into()));
        }
        let part = n
            .to_str()
            .map_err(|_| Error::UnsafePath("evidence paths must be UTF-8".into()))?;
        let relative = if prefix.is_empty() {
            part.to_string()
        } else {
            format!("{prefix}/{part}")
        };
        let before = stat_at(fd, n)?;
        if before.regular() {
            *files += 1;
            *bytes = bytes
                .checked_add(before.size)
                .ok_or_else(|| Error::Limit("snapshot size overflow".into()))?;
            if *files > limits.max_files
                || before.size > limits.max_file_bytes as u64
                || *bytes > limits.max_total_bytes as u64
            {
                return Err(Error::Limit("snapshot file or byte limit exceeded".into()));
            }
            out.insert(relative, before);
        } else if before.directory() {
            if excluded_dirs.contains(part) {
                continue;
            }
            let child = directory_at(fd, n)?;
            out.insert(relative.clone(), before.clone());
            walk_filtered(
                child.as_raw_fd(),
                &relative,
                limits,
                out,
                files,
                bytes,
                excluded_dirs,
            )?;
            if !before.binding_eq(&stat_at(fd, n)?) {
                return Err(Error::Changed("snapshot directory changed".into()));
            }
        } else {
            return Err(Error::UnsafePath(
                "evidence packs contain only regular files and directories".into(),
            ));
        }
    }
    Ok(())
}
fn selected_inventory(
    root: &PinnedDirectory,
    limits: SnapshotLimits,
    files: &BTreeMap<String, usize>,
    dirs: &BTreeSet<String>,
) -> Result<BTreeMap<String, FileIdentity>> {
    fn visit(
        fd: RawFd,
        prefix: &str,
        out: &mut BTreeMap<String, FileIdentity>,
        files: &BTreeMap<String, usize>,
        dirs: &BTreeSet<String>,
        limits: SnapshotLimits,
        total: &mut u64,
    ) -> Result<()> {
        use std::os::fd::IntoRawFd;
        let dot = CString::new(".").unwrap();
        let raw = open_at(fd, &dot, libc::O_RDONLY | libc::O_DIRECTORY, 0)?.into_raw_fd();
        let dir = unsafe { libc::fdopendir(raw) };
        if dir.is_null() {
            unsafe { libc::close(raw) };
            return Err(io::Error::last_os_error().into());
        }
        struct Dir(*mut libc::DIR);
        impl Drop for Dir {
            fn drop(&mut self) {
                unsafe { libc::closedir(self.0) };
            }
        }
        let guard = Dir(dir);
        loop {
            set_errno_zero();
            let e = unsafe { libc::readdir(guard.0) };
            if e.is_null() {
                let err = io::Error::last_os_error();
                if err.raw_os_error() != Some(0) {
                    return Err(err.into());
                }
                break;
            }
            let n = unsafe { CStr::from_ptr((*e).d_name.as_ptr()) };
            if n.to_bytes() == b"." || n.to_bytes() == b".." {
                continue;
            }
            let part = n
                .to_str()
                .map_err(|_| Error::UnsafePath("evidence paths must be UTF-8".into()))?;
            let rel = if prefix.is_empty() {
                part.into()
            } else {
                format!("{prefix}/{part}")
            };
            if !files.contains_key(&rel) && !dirs.contains(&rel) {
                return Err(Error::UnsafePath(format!(
                    "unexpected evidence entry: {rel}"
                )));
            }
            if out.len() >= limits.max_entries {
                return Err(Error::Limit("snapshot entry limit exceeded".into()));
            }
            let before = stat_at(fd, n)?;
            if let Some(limit) = files.get(&rel) {
                if !before.regular() {
                    return Err(Error::UnsafePath(format!(
                        "{rel} must be a non-symlink regular file"
                    )));
                }
                *total = total
                    .checked_add(before.size)
                    .ok_or_else(|| Error::Limit("snapshot size overflow".into()))?;
                if before.size > (*limit).min(limits.max_file_bytes) as u64
                    || *total > limits.max_total_bytes as u64
                {
                    return Err(Error::Limit("snapshot byte limit exceeded".into()));
                }
                out.insert(rel, before);
            } else {
                if !before.directory() {
                    return Err(Error::UnsafePath(format!(
                        "{rel} must be a non-symlink directory"
                    )));
                }
                let child = directory_at(fd, n)?;
                out.insert(rel.clone(), before.clone());
                visit(child.as_raw_fd(), &rel, out, files, dirs, limits, total)?;
                if !before.binding_eq(&stat_at(fd, n)?) {
                    return Err(Error::Changed("snapshot directory changed".into()));
                }
            }
        }
        Ok(())
    }
    if files.len() > limits.max_files {
        return Err(Error::Limit("snapshot file limit exceeded".into()));
    }
    let mut out = BTreeMap::new();
    visit(root.as_raw_fd(), "", &mut out, files, dirs, limits, &mut 0)?;
    if out.len() != files.len() + dirs.len()
        || files
            .keys()
            .chain(dirs.iter())
            .any(|n| !out.contains_key(n))
    {
        return Err(Error::UnsafePath(
            "evidence inventory does not match required entries".into(),
        ));
    }
    root.verify()?;
    Ok(out)
}
fn set_errno_zero() {
    unsafe {
        #[cfg(target_os = "linux")]
        {
            *libc::__errno_location() = 0;
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        {
            *libc::__error() = 0;
        }
    }
}
fn rename_exclusive(src: RawFd, s: &CStr, dst: RawFd, d: &CStr) -> Result<()> {
    #[cfg(target_os = "linux")]
    let result =
        unsafe { libc::renameat2(src, s.as_ptr(), dst, d.as_ptr(), libc::RENAME_NOREPLACE) };
    #[cfg(target_os = "macos")]
    let result = unsafe { libc::renameatx_np(src, s.as_ptr(), dst, d.as_ptr(), libc::RENAME_EXCL) };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return Err(Error::UnsafePath(
        "atomic no-replace publication unsupported on this platform".into(),
    ));
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    if result < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);
struct Stage<'a> {
    parent: &'a PinnedDirectory,
    file: File,
    name: CString,
    payload: Option<File>,
}
impl Drop for Stage<'_> {
    fn drop(&mut self) {
        let payload = CString::new("payload").unwrap();
        if let Some(f) = &self.payload {
            if let (Ok(a), Ok(b)) = (id(f), stat_at(self.file.as_raw_fd(), &payload)) {
                if a.binding_eq(&b) {
                    unsafe { libc::unlinkat(self.file.as_raw_fd(), payload.as_ptr(), 0) };
                }
            }
        }
        if let (Ok(a), Ok(b)) = (id(&self.file), stat_at(self.parent.as_raw_fd(), &self.name)) {
            if a.binding_eq(&b) {
                unsafe {
                    libc::unlinkat(
                        self.parent.as_raw_fd(),
                        self.name.as_ptr(),
                        libc::AT_REMOVEDIR,
                    )
                };
            }
        }
    }
}
pub fn write_file_no_replace(
    path: &Path,
    payload: &[u8],
    mode: u32,
    create_parents: bool,
) -> Result<()> {
    if mode & !0o777 != 0 {
        return Err(Error::UnsafePath("unsupported permission bits".into()));
    }
    let leaf = name(
        path.file_name()
            .ok_or_else(|| Error::UnsafePath("missing destination name".into()))?,
    )?;
    let parent = PinnedDirectory::open(path.parent().unwrap_or(Path::new(".")), create_parents)?;
    let mut staged = None;
    for _ in 0..100 {
        let n = CString::new(format!(
            ".invarlock-write-{}-{}",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap();
        if unsafe { libc::mkdirat(parent.as_raw_fd(), n.as_ptr(), 0o700) } == 0 {
            staged = Some(n);
            break;
        }
        if io::Error::last_os_error().kind() != io::ErrorKind::AlreadyExists {
            return Err(io::Error::last_os_error().into());
        }
    }
    let stage_name =
        staged.ok_or_else(|| Error::UnsafePath("unable to allocate private stage".into()))?;
    let stage_file = directory_at(parent.as_raw_fd(), &stage_name)?;
    let mut stage = Stage {
        parent: &parent,
        file: stage_file,
        name: stage_name,
        payload: None,
    };
    let metadata = stage.file.metadata()?;
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o777 != 0o700 {
        return Err(Error::Changed(
            "private staging directory identity changed".into(),
        ));
    }
    let payload_name = CString::new("payload").unwrap();
    stage.payload = Some(open_at(
        stage.file.as_raw_fd(),
        &payload_name,
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        0o600,
    )?);
    let file = stage.payload.as_mut().unwrap();
    file.write_all(payload)?;
    if unsafe { libc::fchmod(file.as_raw_fd(), mode as libc::mode_t) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    fsync_file(file)?;
    let expected = id(file)?;
    if !expected.contents_eq(&stat_at(stage.file.as_raw_fd(), &payload_name)?) {
        return Err(Error::Changed("staged file changed".into()));
    }
    fsync_file(&stage.file)?;
    parent.sync()?;
    parent.verify()?;
    rename_exclusive(
        stage.file.as_raw_fd(),
        &payload_name,
        parent.as_raw_fd(),
        &leaf,
    )?;
    parent.sync()?;
    if !expected.contents_eq(&stat_at(parent.as_raw_fd(), &leaf)?)
        || !expected.contents_eq(&id(stage.payload.as_ref().unwrap())?)
    {
        return Err(Error::Changed(
            "published file changed; destination retained".into(),
        ));
    }
    parent.verify()
}
pub fn publish_directory_no_replace(staging: &Path, destination: &Path) -> Result<()> {
    if staging == destination {
        return Err(Error::UnsafePath(
            "staging and destination must differ".into(),
        ));
    }
    let s = name(
        staging
            .file_name()
            .ok_or_else(|| Error::UnsafePath("missing staging name".into()))?,
    )?;
    let d = name(
        destination
            .file_name()
            .ok_or_else(|| Error::UnsafePath("missing destination name".into()))?,
    )?;
    let sp = PinnedDirectory::open(staging.parent().unwrap_or(Path::new(".")), false)?;
    let dp = PinnedDirectory::open(destination.parent().unwrap_or(Path::new(".")), false)?;
    let source = directory_at(sp.as_raw_fd(), &s)?;
    let expected = id(&source)?;
    sp.verify()?;
    dp.verify()?;
    if !expected.binding_eq(&stat_at(sp.as_raw_fd(), &s)?) {
        return Err(Error::Changed("staging directory changed".into()));
    }
    rename_exclusive(sp.as_raw_fd(), &s, dp.as_raw_fd(), &d)?;
    sp.sync()?;
    dp.sync()?;
    if !expected.binding_eq(&stat_at(dp.as_raw_fd(), &d)?) {
        return Err(Error::Changed(
            "published directory changed; destination retained".into(),
        ));
    }
    sp.verify()?;
    dp.verify()
}

/// A complete private tree whose entries remain descriptor-owned through
/// publication. Drop removes only retained, identity-matching staged entries;
/// a destination that reached rename is never rolled back by pathname.
pub struct StagedDirectory {
    limits: SnapshotLimits,
    path: std::path::PathBuf,
    parent: PinnedDirectory,
    stage: File,
    stage_name: CString,
    destination: CString,
    directories: BTreeMap<String, File>,
    files: BTreeMap<String, (File, FileIdentity)>,
    published: bool,
    total_bytes: u64,
}
impl StagedDirectory {
    pub fn new(destination: &Path) -> Result<Self> {
        Self::with_limits(
            destination,
            SnapshotLimits {
                max_entries: 256,
                max_files: 256,
                max_file_bytes: 128 * 1024 * 1024,
                max_total_bytes: 512 * 1024 * 1024,
            },
        )
    }
    pub fn with_limits(destination: &Path, limits: SnapshotLimits) -> Result<Self> {
        let leaf = name(
            destination
                .file_name()
                .ok_or_else(|| Error::UnsafePath("destination must name a directory".into()))?,
        )?;
        let parent = PinnedDirectory::open(destination.parent().unwrap_or(Path::new(".")), false)?;
        let mut selected = None;
        for _ in 0..100 {
            let n = CString::new(format!(
                ".invarlock-stage-{}-{}",
                std::process::id(),
                NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
            ))
            .unwrap();
            if unsafe { libc::mkdirat(parent.as_raw_fd(), n.as_ptr(), 0o700) } == 0 {
                selected = Some(n);
                break;
            }
            if io::Error::last_os_error().kind() != io::ErrorKind::AlreadyExists {
                return Err(io::Error::last_os_error().into());
            }
        }
        let stage_name = selected.ok_or_else(|| {
            Error::UnsafePath("unable to allocate private directory stage".into())
        })?;
        let stage = match directory_at(parent.as_raw_fd(), &stage_name) {
            Ok(file) => file,
            Err(error) => {
                unsafe {
                    libc::unlinkat(parent.as_raw_fd(), stage_name.as_ptr(), libc::AT_REMOVEDIR)
                };
                return Err(error);
            }
        };
        let metadata = stage.metadata()?;
        if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o777 != 0o700 {
            return Err(Error::Changed(
                "private staging directory identity changed".into(),
            ));
        }
        let path = destination
            .parent()
            .unwrap_or(Path::new("."))
            .join(OsStr::from_bytes(stage_name.as_bytes()));
        Ok(Self {
            limits,
            path,
            parent,
            stage,
            stage_name,
            destination: leaf,
            directories: BTreeMap::new(),
            files: BTreeMap::new(),
            published: false,
            total_bytes: 0,
        })
    }
    /// Path is usable only while this owned stage remains bound to its parent.
    pub fn path(&self) -> Result<&Path> {
        self.verify()?;
        Ok(&self.path)
    }
    fn directory_fd(&self, relative: &str) -> RawFd {
        if relative.is_empty() {
            self.stage.as_raw_fd()
        } else {
            self.directories[relative].as_raw_fd()
        }
    }
    pub fn create_directory(&mut self, relative: &Path) -> Result<()> {
        let components = safe_relative(relative)?;
        let mut prefix = String::new();
        for component in components {
            let part = component
                .to_str()
                .map_err(|_| Error::UnsafePath("staged paths must be UTF-8".into()))?;
            let next = if prefix.is_empty() {
                part.into()
            } else {
                format!("{prefix}/{part}")
            };
            if !self.directories.contains_key(&next) {
                if self.directories.len() + self.files.len() >= self.limits.max_entries {
                    return Err(Error::Limit("staged entry limit exceeded".into()));
                }
                let parent = self.directory_fd(&prefix);
                if unsafe { libc::mkdirat(parent, component.as_ptr(), 0o700) } != 0 {
                    return Err(io::Error::last_os_error().into());
                }
                let directory = directory_at(parent, &component)?;
                self.directories.insert(next.clone(), directory);
            }
            prefix = next;
        }
        Ok(())
    }
    pub fn write(&mut self, relative: &Path, payload: &[u8], mode: u32) -> Result<()> {
        if payload.len() > self.limits.max_file_bytes
            || self.total_bytes.saturating_add(payload.len() as u64)
                > self.limits.max_total_bytes as u64
        {
            return Err(Error::Limit("staged file or byte limit exceeded".into()));
        }
        self.write_with(relative, mode, |file| {
            file.write_all(payload)?;
            Ok(())
        })
    }
    /// Write an owned staged file with bounded producers (for streaming copies and archives).
    pub fn write_with(
        &mut self,
        relative: &Path,
        mode: u32,
        producer: impl FnOnce(&mut File) -> Result<()>,
    ) -> Result<()> {
        safe_relative(relative)?;
        if mode & !0o777 != 0 {
            return Err(Error::UnsafePath("unsupported permission bits".into()));
        }
        let relative_text = relative
            .to_str()
            .ok_or_else(|| Error::UnsafePath("staged paths must be UTF-8".into()))?;
        if self.files.contains_key(relative_text) {
            return Err(
                io::Error::new(io::ErrorKind::AlreadyExists, "staged file already exists").into(),
            );
        }
        if self.files.len() + self.directories.len() >= self.limits.max_entries
            || self.files.len() >= self.limits.max_files
        {
            return Err(Error::Limit("staged file or byte limit exceeded".into()));
        }
        let parent_path = relative.parent().unwrap_or(Path::new(""));
        if !parent_path.as_os_str().is_empty() {
            self.create_directory(parent_path)?;
        }
        let parent_text = parent_path
            .to_str()
            .ok_or_else(|| Error::UnsafePath("staged paths must be UTF-8".into()))?;
        let leaf = name(relative.file_name().unwrap())?;
        let parent = self.directory_fd(parent_text);
        let mut file = open_at(
            parent,
            &leaf,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
            0o600,
        )?;
        // Retain ownership before write/chmod/fsync can fail, so Drop knows the
        // exact created inode even when interrupted during initialization.
        let initial = id(&file)?;
        self.files
            .insert(relative_text.into(), (file.try_clone()?, initial));
        producer(&mut file)?;
        let size = file.metadata()?.len();
        if size > self.limits.max_file_bytes as u64
            || self.total_bytes.saturating_add(size) > self.limits.max_total_bytes as u64
        {
            return Err(Error::Limit("staged file or byte limit exceeded".into()));
        }
        if unsafe { libc::fchmod(file.as_raw_fd(), mode as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        fsync_file(&file)?;
        let expected = id(&file)?;
        self.files.insert(relative_text.into(), (file, expected));
        self.total_bytes += size;
        Ok(())
    }
    fn verify(&self) -> Result<()> {
        let mut observed = BTreeMap::new();
        walk(
            self.stage.as_raw_fd(),
            "",
            self.limits,
            &mut observed,
            &mut 0,
            &mut 0,
        )?;
        let expected: BTreeSet<_> = self.files.keys().chain(self.directories.keys()).collect();
        if observed.keys().collect::<BTreeSet<_>>() != expected {
            return Err(Error::Changed("staging inventory changed".into()));
        }
        self.parent.verify()?;
        if !id(&self.stage)?.binding_eq(&stat_at(self.parent.as_raw_fd(), &self.stage_name)?) {
            return Err(Error::Changed("staging directory was replaced".into()));
        }
        for (relative, directory) in &self.directories {
            let path = Path::new(relative);
            let parent = path.parent().unwrap().to_str().unwrap();
            let leaf = name(path.file_name().unwrap())?;
            if !id(directory)?.binding_eq(&stat_at(self.directory_fd(parent), &leaf)?) {
                return Err(Error::Changed("staged child directory changed".into()));
            }
        }
        for (relative, (file, expected)) in &self.files {
            let path = Path::new(relative);
            let parent = path.parent().unwrap().to_str().unwrap();
            let leaf = name(path.file_name().unwrap())?;
            if &id(file)? != expected || &stat_at(self.directory_fd(parent), &leaf)? != expected {
                return Err(Error::Changed("staged file changed".into()));
            }
        }
        Ok(())
    }
    /// Seal the complete owned tree before an exclusive publication. If sealing
    /// or rename fails, Drop restores only retained staging-directory modes so
    /// it can remove owned entries; an existing destination is never changed.
    pub fn commit_read_only(mut self) -> Result<()> {
        self.verify()?;
        for (file, expected) in self.files.values_mut() {
            if &id(file)? != expected {
                return Err(Error::Changed("staged file changed before sealing".into()));
            }
            if unsafe { libc::fchmod(file.as_raw_fd(), 0o444) } != 0 {
                return Err(io::Error::last_os_error().into());
            }
            let sealed = id(file)?;
            let mut wanted = expected.clone();
            wanted.mode = (wanted.mode & !0o777) | 0o444;
            if !wanted.contents_eq(&sealed) {
                return Err(Error::Changed("staged file changed during sealing".into()));
            }
            *expected = sealed;
            fsync_file(file)?;
        }
        for directory in self
            .directories
            .values()
            .chain(std::iter::once(&self.stage))
        {
            if unsafe { libc::fchmod(directory.as_raw_fd(), 0o555) } != 0 {
                return Err(io::Error::last_os_error().into());
            }
        }
        self.commit()
    }
    pub fn commit(mut self) -> Result<()> {
        self.verify()?;
        for directory in self.directories.values() {
            fsync_file(directory)?;
        }
        fsync_file(&self.stage)?;
        self.parent.sync()?;
        let expected = id(&self.stage)?;
        rename_exclusive(
            self.parent.as_raw_fd(),
            &self.stage_name,
            self.parent.as_raw_fd(),
            &self.destination,
        )?;
        self.published = true;
        self.parent.sync()?;
        if !expected.binding_eq(&stat_at(self.parent.as_raw_fd(), &self.destination)?) {
            return Err(Error::Changed(
                "published directory identity differs; destination retained".into(),
            ));
        }
        self.parent.verify()
    }
}
impl Drop for StagedDirectory {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        // Sealing can fail part-way through. Retained descriptors identify our
        // original directories even when a pathname has been substituted.
        for directory in self
            .directories
            .values()
            .chain(std::iter::once(&self.stage))
        {
            unsafe { libc::fchmod(directory.as_raw_fd(), 0o700) };
        }
        for (relative, (file, _)) in &self.files {
            let path = Path::new(relative);
            let parent = path.parent().unwrap().to_str().unwrap();
            let Ok(leaf) = name(path.file_name().unwrap()) else {
                continue;
            };
            if let (Ok(expected), Ok(named)) = (id(file), stat_at(self.directory_fd(parent), &leaf))
            {
                if expected.binding_eq(&named) {
                    unsafe { libc::unlinkat(self.directory_fd(parent), leaf.as_ptr(), 0) };
                }
            }
        }
        let mut dirs: Vec<_> = self.directories.keys().collect();
        dirs.sort_by_key(|path| std::cmp::Reverse(path.matches('/').count()));
        for relative in dirs {
            let path = Path::new(relative);
            let parent = path.parent().unwrap().to_str().unwrap();
            let Ok(leaf) = name(path.file_name().unwrap()) else {
                continue;
            };
            if let (Ok(expected), Ok(named)) = (
                id(&self.directories[relative]),
                stat_at(self.directory_fd(parent), &leaf),
            ) {
                if expected.binding_eq(&named) {
                    unsafe {
                        libc::unlinkat(self.directory_fd(parent), leaf.as_ptr(), libc::AT_REMOVEDIR)
                    };
                }
            }
        }
        if let (Ok(expected), Ok(named)) = (
            id(&self.stage),
            stat_at(self.parent.as_raw_fd(), &self.stage_name),
        ) {
            if expected.binding_eq(&named) {
                unsafe {
                    libc::unlinkat(
                        self.parent.as_raw_fd(),
                        self.stage_name.as_ptr(),
                        libc::AT_REMOVEDIR,
                    )
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{os::unix::fs::symlink, path::PathBuf};
    struct SyncFault {
        fail_on: usize,
        errno: i32,
        calls: usize,
    }
    thread_local! {
        static SYNC_FAULT: std::cell::RefCell<Option<SyncFault>> = const {
            std::cell::RefCell::new(None)
        };
    }
    pub(super) fn before_sync() -> io::Result<()> {
        SYNC_FAULT.with(|state| {
            let mut state = state.borrow_mut();
            if let Some(fault) = state.as_mut() {
                fault.calls += 1;
                if fault.calls == fault.fail_on {
                    return Err(io::Error::from_raw_os_error(fault.errno));
                }
            }
            Ok(())
        })
    }
    fn with_sync_fault<T>(fail_on: usize, errno: i32, action: impl FnOnce() -> T) -> (T, usize) {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                SYNC_FAULT.with(|state| *state.borrow_mut() = None);
            }
        }
        SYNC_FAULT.with(|state| {
            let mut state = state.borrow_mut();
            assert!(state.is_none());
            *state = Some(SyncFault {
                fail_on,
                errno,
                calls: 0,
            });
        });
        let _reset = Reset;
        let result = action();
        let calls = SYNC_FAULT.with(|state| state.borrow().as_ref().unwrap().calls);
        (result, calls)
    }
    fn temp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "invarlock-fs-test-{}-{}",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        std::fs::canonicalize(p).unwrap()
    }
    #[test]
    fn sealed_publication_modes_and_failed_rename_cleanup() {
        use std::os::unix::fs::PermissionsExt;
        let p = temp();
        let destination = p.join("sealed");
        let mut stage = StagedDirectory::new(&destination).unwrap();
        stage.create_directory(Path::new("inputs")).unwrap();
        stage
            .write(Path::new("inputs/value"), b"retained", 0o600)
            .unwrap();
        stage.commit_read_only().unwrap();
        for relative in ["", "inputs"] {
            assert_eq!(
                std::fs::metadata(destination.join(relative))
                    .unwrap()
                    .mode()
                    & 0o777,
                0o555
            );
        }
        assert_eq!(
            std::fs::metadata(destination.join("inputs/value"))
                .unwrap()
                .mode()
                & 0o777,
            0o444
        );
        let mut collision = StagedDirectory::new(&destination).unwrap();
        collision.create_directory(Path::new("inputs")).unwrap();
        collision
            .write(Path::new("inputs/other"), b"other", 0o600)
            .unwrap();
        assert!(collision.commit_read_only().is_err());
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 1);
        assert_eq!(
            std::fs::read(destination.join("inputs/value")).unwrap(),
            b"retained"
        );
        for relative in ["", "inputs"] {
            std::fs::set_permissions(
                destination.join(relative),
                std::fs::Permissions::from_mode(0o700),
            )
            .unwrap();
        }
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn capture_is_immutable_and_detects_changes() {
        let p = temp();
        std::fs::write(p.join("a"), b"first").unwrap();
        let s = PackSnapshot::capture(&p, SnapshotLimits::default()).unwrap();
        std::fs::write(p.join("a"), b"second").unwrap();
        assert_eq!(s.entries()["a"].bytes(), b"first");
        assert!(s.verify_source().is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn rejects_symlinks_and_fifo() {
        let p = temp();
        std::fs::write(p.join("a"), b"x").unwrap();
        symlink("a", p.join("b")).unwrap();
        assert!(read_file(&p.join("b"), 1).is_err());
        assert!(PackSnapshot::capture(&p, SnapshotLimits::default()).is_err());
        std::fs::remove_file(p.join("b")).unwrap();
        let n = CString::new(p.join("fifo").as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(n.as_ptr(), 0o600) }, 0);
        assert!(read_file(&p.join("fifo"), 1).is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn fsync_retries_interruption_and_preserves_permanent_errno() {
        let root = temp();
        let path = root.join("file");
        let file = File::create(&path).unwrap();
        let (result, calls) = with_sync_fault(1, libc::EINTR, || fsync_file(&file));
        result.unwrap();
        assert_eq!(calls, 2);
        let (result, calls) = with_sync_fault(1, libc::EIO, || fsync_file(&file));
        assert_eq!(result.unwrap_err().raw_os_error(), Some(libc::EIO));
        assert_eq!(calls, 1);
        drop(file);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn publication_sync_failures_preserve_atomicity_and_owned_cleanup() {
        // File, private stage, parent before rename, parent after rename.
        for phase in 1..=4 {
            let root = temp();
            let destination = root.join("result");
            let payload = b"complete signed output";
            let (result, calls) = with_sync_fault(phase, libc::EIO, || {
                write_file_no_replace(&destination, payload, 0o444, false)
            });
            assert!(matches!(result, Err(Error::Io(ref error))
                if error.raw_os_error() == Some(libc::EIO)));
            assert_eq!(calls, phase);
            if phase == 4 {
                assert_eq!(std::fs::read(&destination).unwrap(), payload);
                assert_eq!(
                    std::fs::metadata(&destination).unwrap().mode() & 0o777,
                    0o444
                );
                assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
            } else {
                assert!(!destination.exists());
                assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn prepublication_sync_failure_preserves_existing_destination() {
        for phase in 1..=3 {
            let root = temp();
            let destination = root.join("result");
            std::fs::write(&destination, b"other writer").unwrap();
            let (result, calls) = with_sync_fault(phase, libc::EIO, || {
                write_file_no_replace(&destination, b"replacement", 0o600, false)
            });
            assert!(matches!(result, Err(Error::Io(ref error))
                if error.raw_os_error() == Some(libc::EIO)));
            assert_eq!(calls, phase);
            assert_eq!(std::fs::read(&destination).unwrap(), b"other writer");
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
            std::fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn no_clobber_and_stage_cleanup() {
        let p = temp();
        let f = p.join("out");
        write_file_no_replace(&f, b"first", 0o600, false).unwrap();
        assert!(write_file_no_replace(&f, b"next", 0o600, false).is_err());
        assert_eq!(read_file(&f, 100).unwrap(), b"first");
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 1);
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn ancestor_substitution_detected() {
        let p = temp();
        std::fs::create_dir(p.join("a")).unwrap();
        let pinned = PinnedDirectory::open(&p.join("a"), false).unwrap();
        std::fs::rename(p.join("a"), p.join("old")).unwrap();
        std::fs::create_dir(p.join("a")).unwrap();
        assert!(pinned.verify().is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn inventory_limits_and_new_empty_directory() {
        let p = temp();
        std::fs::write(p.join("a"), b"12").unwrap();
        let limits = SnapshotLimits {
            max_file_bytes: 1,
            ..SnapshotLimits::default()
        };
        assert!(PackSnapshot::capture(&p, limits).is_err());
        let s = PackSnapshot::capture(&p, SnapshotLimits::default()).unwrap();
        std::fs::create_dir(p.join("new")).unwrap();
        assert!(s.verify_source().is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn directory_publication_never_replaces() {
        let p = temp();
        std::fs::create_dir(p.join("source")).unwrap();
        std::fs::create_dir(p.join("target")).unwrap();
        assert!(publish_directory_no_replace(&p.join("source"), &p.join("target")).is_err());
        std::fs::remove_dir(p.join("target")).unwrap();
        publish_directory_no_replace(&p.join("source"), &p.join("target")).unwrap();
        assert!(p.join("target").is_dir());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn selected_snapshot_rejects_unknown_directory_without_descent() {
        let p = temp();
        std::fs::write(p.join("manifest.json"), b"{}").unwrap();
        std::fs::create_dir(p.join("unexpected")).unwrap();
        symlink("/", p.join("unexpected/escape")).unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let files = BTreeMap::from([("manifest.json".into(), 64)]);
        let result = root.capture_selected(SnapshotLimits::default(), &files, &BTreeSet::new());
        assert!(
            matches!(result,Err(Error::UnsafePath(ref s)) if s=="unexpected evidence entry: unexpected")
        );
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn selected_snapshot_checks_all_sources_and_retains_immutable_bytes() {
        let p = temp();
        std::fs::create_dir(p.join("inputs")).unwrap();
        std::fs::write(p.join("manifest.json"), b"{}").unwrap();
        std::fs::write(p.join("inputs/a"), b"abc").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let files = BTreeMap::from([("manifest.json".into(), 64), ("inputs/a".into(), 8)]);
        let dirs = BTreeSet::from(["inputs".into()]);
        let snapshot = root
            .capture_selected(SnapshotLimits::default(), &files, &dirs)
            .unwrap();
        snapshot.verify_source().unwrap();
        std::fs::write(p.join("inputs/a"), b"abd").unwrap();
        assert!(snapshot.verify_source().is_err());
        assert_eq!(snapshot.entries()["inputs/a"].bytes(), b"abc");
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn selected_source_guard_moves_metadata_and_verifies_without_payloads() {
        let p = temp();
        std::fs::create_dir(p.join("inputs")).unwrap();
        std::fs::write(p.join("manifest.json"), b"{}").unwrap();
        std::fs::write(p.join("inputs/a"), b"abc").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let files = BTreeMap::from([("manifest.json".into(), 64), ("inputs/a".into(), 8)]);
        let dirs = BTreeSet::from(["inputs".into()]);
        let snapshot = root
            .capture_selected(SnapshotLimits::default(), &files, &dirs)
            .unwrap();
        assert_eq!(snapshot.entries()["inputs/a"].bytes(), b"abc");
        snapshot.verify_source().unwrap();
        let inventory_key = snapshot.inventory.first_key_value().unwrap().0.as_ptr();
        let files_key = snapshot.files.first_key_value().unwrap().0.as_ptr();
        let directory_key = snapshot.dirs.first().unwrap().as_ptr();
        let guard = snapshot.into_source_guard();
        assert!(std::ptr::eq(guard.root, &root));
        assert_eq!(
            guard.inventory.first_key_value().unwrap().0.as_ptr(),
            inventory_key
        );
        assert_eq!(guard.files.first_key_value().unwrap().0.as_ptr(), files_key);
        assert_eq!(guard.dirs.first().unwrap().as_ptr(), directory_key);
        assert_eq!(
            guard.limits.max_file_bytes,
            SnapshotLimits::default().max_file_bytes
        );
        guard.verify_source().unwrap();
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn selected_parts_move_entries_and_keep_guard_after_individual_release() {
        let p = temp();
        std::fs::write(p.join("one"), b"one").unwrap();
        std::fs::write(p.join("two"), b"two").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let files = BTreeMap::from([("one".into(), 8), ("two".into(), 8)]);
        let snapshot = root
            .capture_selected(SnapshotLimits::default(), &files, &BTreeSet::new())
            .unwrap();
        let pointer = snapshot.entries()["one"].bytes().as_ptr();
        let (mut entries, guard) = snapshot.into_parts();
        let one = entries.remove("one").unwrap();
        assert_eq!(one.bytes().as_ptr(), pointer);
        assert_eq!(one.bytes(), b"one");
        drop(one);
        guard.verify_source().unwrap();
        drop(entries);
        guard.verify_source().unwrap();
        std::fs::write(p.join("one"), b"new").unwrap();
        assert!(guard.verify_source().is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn selected_source_guard_rejects_changed_files_and_inventories() {
        use std::os::unix::fs::PermissionsExt;
        for mutation in [
            "replacement",
            "content",
            "metadata",
            "addition",
            "missing",
            "symlink",
        ] {
            let p = temp();
            std::fs::create_dir(p.join("inputs")).unwrap();
            std::fs::write(p.join("inputs/a"), b"abc").unwrap();
            let root = PinnedDirectory::open(&p, false).unwrap();
            let snapshot = root
                .capture_selected(
                    SnapshotLimits::default(),
                    &BTreeMap::from([("inputs/a".into(), 8)]),
                    &BTreeSet::from(["inputs".into()]),
                )
                .unwrap();
            let guard = snapshot.into_source_guard();
            match mutation {
                "replacement" => {
                    std::fs::write(p.join("replacement"), b"abc").unwrap();
                    std::fs::rename(p.join("replacement"), p.join("inputs/a")).unwrap();
                }
                "content" => std::fs::write(p.join("inputs/a"), b"xyz").unwrap(),
                "metadata" => {
                    let current = std::fs::metadata(p.join("inputs/a")).unwrap().mode();
                    std::fs::set_permissions(
                        p.join("inputs/a"),
                        std::fs::Permissions::from_mode(current ^ 0o100),
                    )
                    .unwrap();
                }
                "addition" => std::fs::write(p.join("inputs/new"), b"x").unwrap(),
                "missing" => std::fs::remove_file(p.join("inputs/a")).unwrap(),
                "symlink" => {
                    std::fs::remove_file(p.join("inputs/a")).unwrap();
                    symlink("/", p.join("inputs/a")).unwrap();
                }
                _ => unreachable!(),
            }
            assert!(guard.verify_source().is_err(), "accepted {mutation}");
            std::fs::remove_dir_all(p).unwrap();
        }
    }
    #[test]
    fn selected_source_guard_rejects_replaced_root_directory() {
        let p = temp();
        std::fs::write(p.join("a"), b"abc").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let guard = root
            .capture_selected(
                SnapshotLimits::default(),
                &BTreeMap::from([("a".into(), 8)]),
                &BTreeSet::new(),
            )
            .unwrap()
            .into_source_guard();
        let moved = p.with_extension("moved");
        std::fs::rename(&p, &moved).unwrap();
        std::fs::create_dir(&p).unwrap();
        let error = guard.verify_source().unwrap_err();
        assert_eq!(error.to_string(), "directory source was replaced");
        std::fs::remove_dir_all(p).unwrap();
        std::fs::remove_dir_all(moved).unwrap();
    }
    #[test]
    fn selected_snapshot_enforces_per_file_and_missing_entry_bounds() {
        let p = temp();
        std::fs::write(p.join("a"), b"123").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        assert!(root
            .capture_selected(
                SnapshotLimits::default(),
                &BTreeMap::from([("a".into(), 2)]),
                &BTreeSet::new()
            )
            .is_err());
        assert!(root
            .capture_selected(
                SnapshotLimits::default(),
                &BTreeMap::from([("a".into(), 8), ("absent".into(), 8)]),
                &BTreeSet::new()
            )
            .is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn detector_identity_is_rechecked_after_failed_capture() {
        let p = temp();
        std::fs::write(p.join("manifest.json"), b"{}").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let (_, identity) = root.read_regular(Path::new("manifest.json"), 64).unwrap();
        assert!(root
            .capture_selected(
                SnapshotLimits::default(),
                &BTreeMap::from([("manifest.json".into(), 64), ("missing".into(), 8)]),
                &BTreeSet::new()
            )
            .is_err());
        root.verify_regular(Path::new("manifest.json"), &identity)
            .unwrap();
        std::fs::rename(p.join("manifest.json"), p.join("old")).unwrap();
        std::fs::write(p.join("manifest.json"), b"{}").unwrap();
        assert!(root
            .verify_regular(Path::new("manifest.json"), &identity)
            .is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn staged_tree_publishes_whole_and_cleans_interrupted_work() {
        let p = temp();
        let target = p.join("published");
        {
            let mut stage = StagedDirectory::new(&target).unwrap();
            stage.create_directory(Path::new("reports")).unwrap();
            stage
                .write(Path::new("inputs/a"), b"retained", 0o444)
                .unwrap();
        }
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 0);
        let mut stage = StagedDirectory::new(&target).unwrap();
        stage.create_directory(Path::new("reports")).unwrap();
        stage
            .write(Path::new("inputs/a"), b"retained", 0o444)
            .unwrap();
        stage.commit().unwrap();
        assert_eq!(
            read_file(&target.join("inputs/a"), 64).unwrap(),
            b"retained"
        );
        assert!(target.join("reports").is_dir());
        let mut rejected = StagedDirectory::new(&target).unwrap();
        rejected.write(Path::new("other"), b"new", 0o444).unwrap();
        assert!(rejected.commit().is_err());
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 1);
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn staged_tree_rejects_substitution_without_removing_replacement() {
        let p = temp();
        let mut stage = StagedDirectory::new(&p.join("published")).unwrap();
        stage.write(Path::new("a"), b"retained", 0o444).unwrap();
        let root = p.join(stage.stage_name.to_str().unwrap());
        std::fs::rename(&root, p.join("moved")).unwrap();
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("foreign"), b"keep").unwrap();
        assert!(stage.commit().is_err());
        assert_eq!(std::fs::read(root.join("foreign")).unwrap(), b"keep");
        assert!(!p.join("published").exists());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn streaming_digest_is_repeatable_and_bounded() {
        let p = temp();
        std::fs::write(p.join("a"), b"abc").unwrap();
        let file = PinnedRegularFile::open(&p.join("a")).unwrap();
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(file.sha256_digest(3).unwrap(), expected);
        assert_eq!(file.sha256_digest(3).unwrap(), expected);
        assert!(file.sha256_digest(2).is_err());
        let root = PinnedDirectory::open(&p, false).unwrap();
        assert_eq!(root.sha256_file(Path::new("a"), 3).unwrap(), expected);
        std::fs::write(p.join("a"), b"xyz").unwrap();
        assert!(file.sha256_digest(3).is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn excluded_directories_are_not_traversed_but_links_are_rejected() {
        let p = temp();
        std::fs::create_dir(p.join(".cache")).unwrap();
        symlink("/", p.join(".cache/ignored-link")).unwrap();
        std::fs::write(p.join("a"), b"abc").unwrap();
        let root = PinnedDirectory::open(&p, false).unwrap();
        let excludes = BTreeSet::from([".cache".into()]);
        let inventory = root
            .inventory_metadata_filtered(SnapshotLimits::default(), &excludes)
            .unwrap();
        assert_eq!(inventory.len(), 1);
        let mut hash = Sha256::new();
        hash.update(b"domain");
        root.hash_file_into(Path::new("a"), &mut hash, 3).unwrap();
        assert_eq!(
            format!("{:x}", hash.finalize()),
            format!("{:x}", Sha256::digest(b"domainabc"))
        );
        std::fs::remove_dir_all(p.join(".cache")).unwrap();
        symlink("/", p.join(".cache")).unwrap();
        assert!(root
            .inventory_metadata_filtered(SnapshotLimits::default(), &excludes)
            .is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
}
