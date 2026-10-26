//! Exact recipient-side model-content identities using pinned streaming reads.
use crate::{EvidenceError, Result};
use invarlock_native_contracts::{canonical_json_no_newline, parse_json, python_repr};
use invarlock_native_filesystem::{PinnedDirectory, PinnedRegularFile, SnapshotLimits};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    path::{Component, Path},
};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn regular(mode: u32) -> bool {
    mode & 0o170000 == 0o100000
}
fn limits() -> SnapshotLimits {
    SnapshotLimits {
        max_entries: usize::MAX,
        max_files: usize::MAX,
        max_file_bytes: usize::MAX,
        max_total_bytes: usize::MAX,
    }
}
fn tree(path: &Path) -> Result<String> {
    if path.components().any(|c| c == Component::ParentDir) {
        return Err(err(
            "local checkpoint path must be absolute without parent traversal",
        ));
    }
    let root=PinnedDirectory::open(path,false).map_err(|_|err("local checkpoint could not be securely opened as a regular directory without symbolic links"))?;
    let excluded = [
        ".cache",
        ".git",
        "__pycache__",
        "log",
        "logs",
        "reports",
        "runs",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<BTreeSet<_>>();
    let before = root
        .inventory_metadata_filtered(limits(), &excluded)
        .map_err(|e| err(e.to_string()))?;
    let excluded_files = [
        ".ds_store",
        ".model_id",
        "adapter_runtime_summary.json",
        "checkpoint_refs.json",
        "evidence.meta.json",
        "external_edit_summary.json",
        "fixture_summary.json",
        "model_summary.json",
        "run_summary.txt",
        "training_binding.json",
        "training_receipt.json",
    ];
    let mut files = before
        .iter()
        .filter(|(name, m)| {
            let name = Path::new(name.as_str())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_lowercase();
            regular(m.identity.mode)
                && !excluded_files.contains(&name.as_str())
                && !name.starts_with(".tmp-")
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|(name, _)| name.split('/').collect::<Vec<_>>());
    if files.is_empty() {
        return Err(err(format!(
            "local checkpoint contains no checkpoint files: {}",
            path.display()
        )));
    }
    let mut hash = Sha256::new();
    hash.update(b"invarlock-model-checkpoint-tree-v1\0");
    for (name, m) in files {
        let relative = Path::new(name);
        root.verify_regular(relative, &m.identity)
            .map_err(|_| err(format!("checkpoint file changed before hashing: {name}")))?;
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name.as_bytes());
        hash.update(m.identity.size.to_be_bytes());
        root.hash_file_into(relative, &mut hash, u64::MAX)
            .map_err(|_| err(format!("checkpoint file changed while hashing: {name}")))?;
    }
    if before
        != root
            .inventory_metadata_filtered(limits(), &excluded)
            .map_err(|e| err(e.to_string()))?
    {
        return Err(err(format!(
            "checkpoint tree changed while hashing: {}",
            path.display()
        )));
    }
    root.verify().map_err(|e| err(e.to_string()))?;
    Ok(format!("sha256:{:x}", hash.finalize()))
}
fn bounded_text(v: &Value, field: &str) -> Result<()> {
    let s = v
        .as_str()
        .filter(|s| !s.is_empty() && s.trim() == *s && s.chars().count() <= 256)
        .ok_or_else(|| {
            err(
                format!("TensorRT-LLM config field {field:?} must be bounded non-empty text")
                    .replace('"', "'"),
            )
        })?;
    if s.chars().any(|c| c < ' ') {
        return Err(err(format!(
            "TensorRT-LLM config field {field:?} must be printable"
        )
        .replace('"', "'")));
    }
    Ok(())
}
fn trt(path: &Path) -> Result<String> {
    if path.components().any(|c| c == Component::ParentDir) {
        return Err(err(
            "engine bundle path must not contain traversal components",
        ));
    }
    let root = PinnedDirectory::open(path, false)
        .map_err(|_| err("engine bundle path contains a symlink or inaccessible directory"))?;
    let lim = SnapshotLimits {
        max_entries: 288,
        max_files: 256,
        max_file_bytes: 256 * 1024usize.pow(3),
        max_total_bytes: 2 * 1024usize.pow(4),
    };
    let before = root
        .inventory_metadata(lim)
        .map_err(|e| err(e.to_string()))?;
    let mut folded = BTreeSet::new();
    let mut dirs = 1;
    let mut files = Vec::new();
    for (name, m) in &before {
        if name.len() > 4096 {
            return Err(err("engine bundle entry name exceeds the configured bound"));
        }
        let folded_name = invarlock_native_core::unicode_casefold(name);
        if !folded.insert(folded_name) {
            return Err(err(format!(
                "engine bundle contains a casefold collision at {}",
                python_repr(&json!(name))
            )));
        }
        if regular(m.identity.mode) {
            if m.links != 1 {
                return Err(err(format!(
                    "engine bundle entry {} must not be hard-linked",
                    python_repr(&json!(name))
                )));
            }
            files.push((name, m));
        } else {
            dirs += 1;
            if dirs > 32 {
                return Err(err(
                    "engine bundle directory count exceeds the configured bound",
                ));
            }
            if name.split('/').count() > 8 {
                return Err(err("engine bundle tree depth exceeds the configured bound"));
            }
        }
    }
    if files.is_empty() {
        return Err(err("TensorRT-LLM engine bundle is empty"));
    }
    if !files.iter().any(|(n, _)| *n == "config.json") {
        return Err(err("TensorRT-LLM engine bundle is missing 'config.json'"));
    }
    let raw = root
        .read_regular(Path::new("config.json"), 16 * 1024 * 1024)
        .map_err(|_| err("TensorRT-LLM config exceeds the configured bound"))?
        .0;
    let config = parse_json(&raw, "TensorRT-LLM config")
        .map_err(|_| err("TensorRT-LLM config is not strict JSON"))?;
    if !config.is_object() {
        return Err(err("TensorRT-LLM config must be a JSON object"));
    }
    let mut stack = vec![(&config, 0)];
    let mut items = 0;
    while let Some((v, depth)) = stack.pop() {
        if depth > 64 {
            return Err(err(
                "TensorRT-LLM config depth exceeds the configured bound",
            ));
        }
        if let Some(o) = v.as_object() {
            items += 2 * o.len();
            stack.extend(o.values().map(|v| (v, depth + 1)))
        } else if let Some(a) = v.as_array() {
            items += a.len();
            stack.extend(a.iter().map(|v| (v, depth + 1)))
        }
        if items > 1_000_000 {
            return Err(err(
                "TensorRT-LLM config item count exceeds the configured bound",
            ));
        }
    }
    if config.as_object().unwrap().len() != 3
        || ["version", "pretrained_config", "build_config"]
            .iter()
            .any(|k| config.get(k).is_none())
    {
        return Err(err(
            "TensorRT-LLM config must contain exactly version, pretrained_config, and build_config",
        ));
    }
    bounded_text(&config["version"], "version")?;
    for k in ["pretrained_config", "build_config"] {
        if config[k].as_object().is_none_or(|o| o.is_empty()) {
            return Err(err(format!("TensorRT-LLM {k} must be a non-empty object")));
        }
    }
    for k in ["architecture", "dtype"] {
        bounded_text(&config["pretrained_config"][k], k)?
    }
    let mapping = &config["pretrained_config"]["mapping"];
    if !mapping.is_object() {
        return Err(err(
            "TensorRT-LLM pretrained_config.mapping must be an object",
        ));
    }
    let size = mapping["world_size"]
        .as_u64()
        .filter(|n| (1..=256).contains(n))
        .ok_or_else(|| err("TensorRT-LLM mapping.world_size is outside the supported bound"))?;
    let mut expected = BTreeSet::from(["config.json".to_owned()]);
    expected.extend((0..size).map(|i| format!("rank{i}.engine")));
    let actual = files
        .iter()
        .map(|(n, _)| (*n).clone())
        .collect::<BTreeSet<_>>();
    if let Some(missing) = expected.difference(&actual).next() {
        return Err(err(format!(
            "TensorRT-LLM engine bundle is missing {}",
            python_repr(&json!(missing))
        )));
    }
    if let Some(extra) = actual.difference(&expected).next() {
        return Err(err(format!(
            "TensorRT-LLM engine bundle contains unsupported entry {}",
            python_repr(&json!(extra))
        )));
    }
    let mut inventory = Vec::new();
    for (name, m) in files {
        if name.ends_with(".engine") && m.identity.size == 0 {
            return Err(err(format!(
                "TensorRT-LLM engine file {} must not be empty",
                python_repr(&json!(name))
            )));
        }
        root.verify_regular(Path::new(name), &m.identity)
            .map_err(|e| err(e.to_string()))?;
        let digest = root
            .sha256_file(Path::new(name), 256 * 1024u64.pow(3))
            .map_err(|e| err(e.to_string()))?;
        inventory.push(json!({"byte_length":m.identity.size,"logical_name":name,"sha256":digest}));
    }
    if before
        != root
            .inventory_metadata(lim)
            .map_err(|e| err(e.to_string()))?
    {
        return Err(err(
            "TensorRT-LLM engine bundle changed while being authenticated",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"invarlock/tensorrt-llm-engine-tree-v1\0");
    hash.update(canonical_json_no_newline(&json!(inventory))?);
    Ok(format!("sha256:{:x}", hash.finalize()))
}
pub fn artifact_path_digest(path: &Path, kind: &str) -> Result<String> {
    match kind {
        "file_sha256" => {
            let metadata = std::fs::symlink_metadata(path)
                .map_err(|_| err("subject artifact could not be read safely"))?;
            if metadata.file_type().is_symlink() {
                return Err(err("subject artifact must not be a symlink"));
            }
            if !metadata.is_file() {
                return Err(err("file-bound subject artifact must be a regular file"));
            }
            let f = PinnedRegularFile::open(path)
                .map_err(|_| err("subject artifact could not be read safely"))?;
            let hash = f
                .sha256_digest(u64::MAX)
                .map_err(|_| err("subject artifact changed while it was being hashed"))?;
            Ok(format!("sha256:{hash}"))
        }
        "hf_snapshot_tree_sha256" => tree(path),
        "tensorrt_llm_engine_tree_sha256" => trt(path),
        _ => Err(err("subject artifact digest kind is unsupported")),
    }
}
