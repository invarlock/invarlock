//! Bounded GGUF v2/v3 identity inspection over a pinned regular-file capability.
use crate::{EvidenceError, Result};
use invarlock_native_contracts::{canonical_json_no_newline, python_repr};
use invarlock_native_filesystem::{PinnedDirectory, PinnedRegularFile};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufReader, Read},
    os::{
        fd::{AsRawFd, BorrowedFd},
        unix::ffi::{OsStrExt, OsStringExt},
    },
    path::{Component, Path, PathBuf},
};

const MAX_FILE: u64 = 2 * 1024u64.pow(4);
const MAX_HEADER: u64 = 512 * 1024 * 1024;
const MAX_ARRAY: u64 = 2_000_000;
const MAX_TOTAL_ARRAY: u64 = 4_000_000;
fn error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(message)
}

struct Header<R> {
    input: R,
    size: u64,
    position: u64,
    array_items: u64,
}
impl<R: Read> Header<R> {
    fn read(&mut self, size: u64, label: &str) -> Result<Vec<u8>> {
        let end = self
            .position
            .checked_add(size)
            .ok_or_else(|| error("GGUF header size exceeds the configured bound"))?;
        if end > MAX_HEADER {
            return Err(error("GGUF header size exceeds the configured bound"));
        }
        if end > self.size {
            return Err(error(format!("truncated GGUF {label}")));
        }
        let mut value = vec![0; size as usize];
        self.input
            .read_exact(&mut value)
            .map_err(|_| error(format!("truncated GGUF {label}")))?;
        self.position = end;
        Ok(value)
    }
    fn u32(&mut self, label: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(4, label)?.try_into().unwrap()))
    }
    fn u64(&mut self, label: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read(8, label)?.try_into().unwrap()))
    }
    fn string(&mut self, maximum: u64, label: &str) -> Result<Vec<u8>> {
        let size = self.u64(&format!("{label} length"))?;
        if size > maximum {
            return Err(error(format!(
                "GGUF {label} length exceeds the configured bound"
            )));
        }
        self.read(size, label)
    }
    fn value(&mut self, kind: u32, depth: usize) -> Result<([u8; 32], Option<u32>)> {
        if depth > 4 {
            return Err(error("GGUF metadata array nesting exceeds the bound"));
        }
        let mut hash = Sha256::new();
        hash.update(b"invarlock/gguf-value-v1\0");
        hash.update(kind.to_le_bytes());
        let size = match kind {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        let mut scalar = None;
        if let Some(size) = size {
            let encoded = self.read(size, "metadata scalar")?;
            if kind == 7 && encoded[0] > 1 {
                return Err(error("GGUF boolean metadata must be zero or one"));
            }
            if (kind == 6
                && !f32::from_le_bytes(encoded.as_slice().try_into().unwrap()).is_finite())
                || (kind == 12
                    && !f64::from_le_bytes(encoded.as_slice().try_into().unwrap()).is_finite())
            {
                return Err(error("GGUF floating metadata must be finite"));
            }
            if kind == 4 {
                scalar = Some(u32::from_le_bytes(encoded.as_slice().try_into().unwrap()));
            }
            hash.update(encoded);
        } else if kind == 8 {
            let encoded = self.string(16 * 1024 * 1024, "metadata string")?;
            std::str::from_utf8(&encoded)
                .map_err(|_| error("GGUF metadata string must be UTF-8"))?;
            hash.update((encoded.len() as u64).to_le_bytes());
            hash.update(encoded);
        } else if kind == 9 {
            if depth >= 4 {
                return Err(error("GGUF metadata array nesting exceeds the bound"));
            }
            let element = self.u32("metadata array element type")?;
            if element > 12 {
                return Err(error("GGUF metadata array has unsupported value type"));
            }
            let count = self.u64("metadata array length")?;
            if count > MAX_ARRAY {
                return Err(error("GGUF metadata array length exceeds the bound"));
            }
            self.array_items += count;
            if self.array_items > MAX_TOTAL_ARRAY {
                return Err(error("GGUF total metadata array items exceed the bound"));
            }
            hash.update(element.to_le_bytes());
            hash.update(count.to_le_bytes());
            for _ in 0..count {
                hash.update(self.value(element, depth + 1)?.0);
            }
        } else {
            return Err(error(format!(
                "GGUF metadata uses unsupported value type {kind}"
            )));
        }
        Ok((hash.finalize().into(), scalar))
    }
}
fn layout(kind: u32) -> Option<(u64, u64)> {
    Some(match kind {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        _ => return None,
    })
}
fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
fn records_digest(records: Vec<Value>) -> Result<String> {
    Ok(hex(Sha256::digest(canonical_json_no_newline(
        &Value::Array(records),
    )?)))
}
fn parse<R: Read>(input: R, size: u64) -> Result<(String, String, String)> {
    let mut reader = Header {
        input,
        size,
        position: 0,
        array_items: 0,
    };
    if reader.read(4, "magic")? != b"GGUF" {
        return Err(error("GGUF magic is invalid"));
    }
    let version = reader.u32("version")?;
    if ![2, 3].contains(&version) {
        return Err(error(format!(
            "unsupported GGUF version or endian encoding: {version}"
        )));
    }
    let tensors = reader.u64("tensor count")?;
    let metadata = reader.u64("metadata count")?;
    if tensors > 262_144 {
        return Err(error("GGUF tensor count exceeds the configured bound"));
    }
    if metadata > 16_384 {
        return Err(error("GGUF metadata count exceeds the configured bound"));
    }
    let mut records = Vec::new();
    let mut tokenizers = Vec::new();
    let mut keys = BTreeSet::new();
    let mut alignment = 32u64;
    for _ in 0..metadata {
        let bytes = reader.string(65_535, "metadata key")?;
        if !bytes.is_ascii() {
            return Err(error("GGUF metadata key must be ASCII"));
        }
        let key = String::from_utf8(bytes).unwrap();
        if key.split(['.', '_']).any(|segment| {
            segment.is_empty()
                || !segment
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        }) {
            return Err(error("GGUF metadata key is not canonical"));
        }
        if !keys.insert(key.clone()) {
            return Err(error(format!(
                "duplicate metadata key {}",
                python_repr(&json!(key))
            )));
        }
        let kind = reader.u32("metadata value type")?;
        let (digest, scalar) = reader.value(kind, 0)?;
        let record = json!({"key":key,"value_type":kind,"value_sha256":hex(digest)});
        if key.starts_with("tokenizer.") {
            tokenizers.push(record.clone());
        }
        records.push(record);
        if key == "general.alignment" {
            alignment =
                scalar.ok_or_else(|| error("GGUF general.alignment must be uint32"))? as u64;
        }
    }
    if !(8..=1024 * 1024).contains(&alignment) || !alignment.is_power_of_two() {
        return Err(error("GGUF general.alignment is invalid"));
    }
    let mut names = BTreeSet::new();
    let mut tensor_records = Vec::new();
    let mut extents = Vec::new();
    for _ in 0..tensors {
        let bytes = reader.string(64, "tensor name")?;
        let name = String::from_utf8(bytes).map_err(|_| error("GGUF tensor name must be UTF-8"))?;
        if name.is_empty() || name.chars().any(|c| c < ' ') {
            return Err(error("GGUF tensor name must be non-empty and printable"));
        }
        if !names.insert(name.clone()) {
            return Err(error(format!(
                "duplicate tensor name {}",
                python_repr(&json!(name))
            )));
        }
        let count = reader.u32("tensor dimension count")?;
        if !(1..=4).contains(&count) {
            return Err(error("GGUF tensor dimension count is invalid"));
        }
        let mut dimensions = Vec::new();
        let mut elements = 1u64;
        for _ in 0..count {
            let dimension = reader.u64("tensor dimension")?;
            if dimension == 0 || elements > ((1u64 << 63) - 1) / dimension {
                return Err(error("GGUF tensor dimensions exceed the bound"));
            }
            elements *= dimension;
            dimensions.push(dimension);
        }
        let kind = reader.u32("tensor type")?;
        let offset = reader.u64("tensor offset")?;
        let (block, encoded) =
            layout(kind).ok_or_else(|| error(format!("GGUF tensor type {kind} is unsupported")))?;
        if !dimensions[0].is_multiple_of(block) {
            return Err(error(
                "GGUF quantized tensor row is not a multiple of its block size",
            ));
        }
        let blocks = elements / block;
        if blocks > MAX_FILE / encoded {
            return Err(error("GGUF tensor byte extent exceeds the file bound"));
        }
        let length = blocks * encoded;
        extents.push((offset, length));
        tensor_records.push(json!({"byte_length":length,"dimensions":dimensions,"name":name,"offset":offset,"tensor_type":kind}));
    }
    let padding = (alignment - reader.position % alignment) % alignment;
    if reader
        .read(padding, "header padding")?
        .iter()
        .any(|b| *b != 0)
    {
        return Err(error("GGUF header padding must be zero-filled"));
    }
    let data_size = size - reader.position;
    if tensors > 0 && data_size == 0 {
        return Err(error("GGUF tensor data is missing or truncated"));
    }
    for &(offset, length) in &extents {
        if !offset.is_multiple_of(alignment) {
            return Err(error("GGUF tensor offset is not aligned"));
        }
        if offset >= data_size {
            return Err(error("GGUF tensor offset exceeds tensor data"));
        }
        if length > data_size - offset {
            return Err(error("GGUF tensor data is missing or truncated"));
        }
    }
    extents.sort();
    for pair in extents.windows(2) {
        if pair[1].0 < pair[0].0 + pair[0].1 {
            return Err(error("GGUF tensor data ranges overlap"));
        }
    }
    records.sort_by(|a, b| a["key"].as_str().cmp(&b["key"].as_str()));
    tokenizers.sort_by(|a, b| a["key"].as_str().cmp(&b["key"].as_str()));
    tensor_records.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Ok((
        records_digest(records)?,
        records_digest(tensor_records)?,
        records_digest(tokenizers)?,
    ))
}
// The metadata timestamps on some Linux filesystems do not advance for every
// write. Bind parsed metadata to the complete byte stream from this inspection
// pass, and compare it with the initial content digest before returning identity.
struct DigestReader<R> {
    input: R,
    hash: Sha256,
    count: u64,
    limit: u64,
}
impl<R: Read> Read for DigestReader<R> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.input.read(output)?;
        self.count += count as u64;
        if self.count > self.limit {
            return Err(std::io::Error::other("GGUF artifact grew while being read"));
        }
        self.hash.update(&output[..count]);
        Ok(count)
    }
}
fn inspect_content<R: Read>(
    input: R,
    size: u64,
    expected: &str,
) -> Result<(String, String, String)> {
    let mut reader = BufReader::with_capacity(
        64 * 1024,
        DigestReader {
            input,
            hash: Sha256::new(),
            count: 0,
            limit: size,
        },
    );
    let parsed = parse(&mut reader, size)?;
    std::io::copy(&mut reader, &mut std::io::sink())
        .map_err(|_| error("GGUF artifact changed while being read"))?;
    let checked = reader.into_inner();
    if checked.count != size || format!("{:x}", checked.hash.finalize()) != expected {
        return Err(error("GGUF artifact changed while being read"));
    }
    Ok(parsed)
}
fn absolute(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().as_bytes().contains(&0) {
        return Err(error("GGUF artifact path is invalid"));
    }
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|_| error("GGUF artifact path is invalid"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for part in full.components() {
        match part {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    if normalized.file_name().is_none() {
        return Err(error("GGUF artifact path must identify a regular file"));
    }
    Ok(normalized)
}
fn read_with(path: &Path, after_hash: impl FnOnce()) -> Result<Value> {
    let path = absolute(path)?;
    // This preflight provides source-compatible diagnostics; all actual reads use
    // the independently opened capability and its retained ancestor bindings.
    let parent = PinnedDirectory::open(path.parent().unwrap(), false)
        .map_err(|_| error("GGUF artifact path contains a symlink or inaccessible directory"))?;
    let metadata =
        std::fs::symlink_metadata(&path).map_err(|_| error("GGUF artifact is unavailable"))?;
    if metadata.file_type().is_symlink() {
        return Err(error("GGUF artifact must not be a symlink"));
    }
    if !metadata.is_file() {
        return Err(error("GGUF artifact must be a stable regular file"));
    }
    let artifact = PinnedRegularFile::open(&path)
        .map_err(|_| error("GGUF artifact cannot be opened safely"))?;
    parent
        .verify()
        .map_err(|_| error("GGUF artifact changed while being opened"))?;
    let size = artifact.metadata().identity.size;
    if !(24..=MAX_FILE).contains(&size) {
        return Err(error("GGUF file size is outside the configured bounds"));
    }
    let digest = artifact
        .sha256_digest(MAX_FILE)
        .map_err(|_| error("GGUF artifact changed or truncated while hashing"))?;
    after_hash();
    // SAFETY: artifact owns this live descriptor throughout the borrowed lifetime;
    // cloning creates an independently owned descriptor before the borrow ends.
    let borrowed = unsafe { BorrowedFd::borrow_raw(artifact.as_raw_fd()) };
    let file = File::from(
        borrowed
            .try_clone_to_owned()
            .map_err(|_| error("GGUF artifact cannot be inspected safely"))?,
    );
    let (metadata_digest, tensor_digest, tokenizer_digest) = inspect_content(file, size, &digest)?;
    // inspect_content has already checked the complete consumed stream digest.
    artifact
        .verify_binding()
        .map_err(|_| error("GGUF artifact changed while being read"))?;
    Ok(
        json!({"artifact_name":format!("gguf-sha256-{digest}.gguf"),"sha256":digest,"byte_length":size,"gguf_metadata_sha256":metadata_digest,"tensor_inventory_sha256":tensor_digest,"tokenizer_metadata_sha256":tokenizer_digest}),
    )
}
pub fn read_identity(path: &Path) -> Result<Value> {
    read_with(path, || {})
}
pub fn dispatch(config: &Value) -> Result<Value> {
    let path = if let Some(encoded) = config["path_base64"].as_str() {
        PathBuf::from(std::ffi::OsString::from_vec(
            crate::decode_base64(encoded).map_err(|_| error("GGUF artifact path is invalid"))?,
        ))
    } else {
        PathBuf::from(crate::required_string(config, "path")?)
    };
    read_identity(&path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn fixture() -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(32, 0);
        bytes
    }
    fn path() -> PathBuf {
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "gguf-native-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn inspection_checks_content_without_relying_on_timestamps() {
        let mut original = fixture();
        original.push(0);
        let expected = format!("{:x}", Sha256::digest(&original));
        assert!(inspect_content(Cursor::new(&original), original.len() as u64, &expected).is_ok());
        // The change is outside the parsed header; a header-only check misses it.
        original[32] = 1;
        assert_eq!(
            inspect_content(Cursor::new(&original), original.len() as u64, &expected)
                .unwrap_err()
                .message,
            "GGUF artifact changed while being read"
        );
    }
    #[test]
    fn reject_same_size_mutation_after_hash() {
        let path = path();
        let mut initial = fixture();
        initial.push(0);
        std::fs::write(&path, &initial).unwrap();
        let outcome = read_with(&path, || {
            initial[32] = 1;
            std::fs::write(&path, initial).unwrap();
        });
        assert_eq!(
            outcome.unwrap_err().message,
            "GGUF artifact changed while being read"
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn reject_replacement_after_hash() {
        let path = path();
        std::fs::write(&path, fixture()).unwrap();
        let replacement = path.with_extension("replacement");
        std::fs::write(&replacement, fixture()).unwrap();
        let outcome = read_with(&path, || std::fs::rename(&replacement, &path).unwrap());
        assert_eq!(
            outcome.unwrap_err().message,
            "GGUF artifact changed while being read"
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn header_and_total_array_limits_precede_allocation() {
        let mut reader = Header {
            input: Cursor::new(Vec::<u8>::new()),
            size: MAX_FILE,
            position: MAX_HEADER,
            array_items: 0,
        };
        assert_eq!(
            reader.read(1, "value").unwrap_err().message,
            "GGUF header size exceeds the configured bound"
        );
        let mut raw = 0u32.to_le_bytes().to_vec();
        raw.extend(1u64.to_le_bytes());
        let mut reader = Header {
            input: Cursor::new(raw),
            size: 12,
            position: 0,
            array_items: MAX_TOTAL_ARRAY,
        };
        assert_eq!(
            reader.value(9, 0).unwrap_err().message,
            "GGUF total metadata array items exceed the bound"
        );
    }
}
