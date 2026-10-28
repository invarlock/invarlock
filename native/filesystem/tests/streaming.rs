use invarlock_native_filesystem::{PinnedRegularFile, SnapshotLimits, StagedDirectory};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "invarlock-stream-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn copies_without_changing_the_file_offset_and_enforces_size() {
    let root = Temp::new();
    let path = root.0.join("source");
    let bytes = vec![42; 200_000];
    fs::write(&path, &bytes).unwrap();
    let pinned = PinnedRegularFile::open(&path).unwrap();
    assert!(pinned.copy_to(&mut io::sink(), 199_999).is_err());
    for _ in 0..2 {
        let mut actual = Vec::new();
        assert_eq!(pinned.copy_to(&mut actual, 200_000).unwrap(), 200_000);
        assert_eq!(actual, bytes);
    }
}
#[test]
fn source_mutation_is_detected_after_streaming() {
    struct Mutator(PathBuf, bool);
    impl Write for Mutator {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if !self.1 {
                self.1 = true;
                fs::OpenOptions::new()
                    .append(true)
                    .open(&self.0)?
                    .write_all(b"changed")?;
            }
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let root = Temp::new();
    let path = root.0.join("source");
    fs::write(&path, vec![7; 100_000]).unwrap();
    let pinned = PinnedRegularFile::open(&path).unwrap();
    assert!(pinned.copy_to(&mut Mutator(path, false), 100_000).is_err());
}
#[test]
fn failed_producer_or_limit_never_publishes_or_leaks_the_stage() {
    for fail in [false, true] {
        let root = Temp::new();
        let output = root.0.join("output");
        {
            let mut stage = StagedDirectory::with_limits(
                &output,
                SnapshotLimits {
                    max_entries: 3,
                    max_files: 2,
                    max_file_bytes: 8,
                    max_total_bytes: 16,
                },
            )
            .unwrap();
            let result = stage.write_with(std::path::Path::new("file"), 0o600, |f| {
                f.write_all(b"more than eight bytes")?;
                if fail {
                    return Err(io::Error::other("interrupted producer").into());
                }
                Ok(())
            });
            assert!(result.is_err());
        }
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }
}
