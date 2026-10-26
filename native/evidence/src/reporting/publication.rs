//! No-clobber report publication with explicit partial publication outcomes.
use invarlock_native_filesystem as fs;
use serde_json::{Map, Value};
use std::{
    collections::VecDeque,
    fmt,
    path::{Component, Path, PathBuf},
};
#[derive(Debug, Clone)]
pub struct PublicationError {
    pub message: String,
    pub failed_output: Option<String>,
    pub written_outputs: Map<String, Value>,
}
impl PublicationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            failed_output: None,
            written_outputs: Map::new(),
        }
    }
}
impl fmt::Display for PublicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for PublicationError {}
pub type Result<T> = std::result::Result<T, PublicationError>;
pub type RequestedOutputs = Vec<(String, PathBuf)>;
fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.into())
    } else {
        Ok(std::env::current_dir()
            .map_err(|e| PublicationError::new(e.to_string()))?
            .join(path))
    }
}
fn resolved(path: &Path) -> Result<PathBuf> {
    let mut pending: VecDeque<_> = absolute(path)?
        .components()
        .map(|c| c.as_os_str().to_owned())
        .collect();
    let mut result = PathBuf::new();
    let mut links = 0;
    while let Some(part) = pending.pop_front() {
        match Path::new(&part).components().next() {
            Some(Component::RootDir) => result.push(Path::new("/")),
            Some(Component::CurDir) => {}
            Some(Component::ParentDir) => {
                result.pop();
            }
            _ => {
                result.push(&part);
                if std::fs::symlink_metadata(&result).is_ok_and(|m| m.file_type().is_symlink()) {
                    links += 1;
                    if links > 40 {
                        return Err(PublicationError::new(
                            "report destination contains a symlink loop",
                        ));
                    }
                    let target = std::fs::read_link(&result)
                        .map_err(|e| PublicationError::new(e.to_string()))?;
                    result.pop();
                    let components = target
                        .components()
                        .map(|c| c.as_os_str().to_owned())
                        .collect::<Vec<_>>();
                    for component in components.into_iter().rev() {
                        pending.push_front(component);
                    }
                }
            }
        }
    }
    Ok(result)
}
/// Validate the whole output set before rendering or creating any output.
pub fn validate(requested: &[(String, PathBuf)], evidence: Option<&Path>) -> Result<Vec<PathBuf>> {
    let evidence = evidence.map(absolute).transpose()?;
    let evidence_resolved = evidence.as_deref().map(resolved).transpose()?;
    let mut destinations = Vec::new();
    let mut canonical = Vec::<PathBuf>::new();
    for (_, path) in requested {
        let destination = absolute(path)?;
        if destination.file_name().is_none() {
            return Err(PublicationError::new(
                "report destination must name a regular file",
            ));
        }
        let normalized = resolved(&destination)?;
        if evidence
            .as_ref()
            .is_some_and(|p| destination.starts_with(p))
            || evidence_resolved
                .as_ref()
                .is_some_and(|p| normalized.starts_with(p))
        {
            return Err(PublicationError::new(
                "report destination must remain outside the immutable evidence pack",
            ));
        }
        if canonical
            .iter()
            .any(|other| normalized.starts_with(other) || other.starts_with(&normalized))
        {
            return Err(PublicationError::new("report destinations collide"));
        }
        if std::fs::symlink_metadata(&destination).is_ok() {
            return Err(PublicationError::new(format!(
                "report destination already exists: {}",
                destination.display()
            )));
        }
        for parent in destination.ancestors().skip(1) {
            if std::fs::symlink_metadata(parent)
                .is_ok_and(|m| m.file_type().is_symlink() || !m.is_dir())
            {
                return Err(PublicationError::new(
                    "report destination parent must be a real directory",
                ));
            }
        }
        canonical.push(normalized);
        destinations.push(destination);
    }
    Ok(destinations)
}
/// All renderer bytes must be prepared before the first publication.
pub fn publish(
    requested: &[(String, PathBuf)],
    rendered: &[Vec<u8>],
    evidence: Option<&Path>,
) -> Result<Map<String, Value>> {
    publish_with(requested, rendered, evidence, |path, bytes| {
        fs::write_file_no_replace(path, bytes, 0o600, true).map_err(|e| e.to_string())
    })
}
fn publish_with<F>(
    requested: &[(String, PathBuf)],
    rendered: &[Vec<u8>],
    evidence: Option<&Path>,
    mut writer: F,
) -> Result<Map<String, Value>>
where
    F: FnMut(&Path, &[u8]) -> std::result::Result<(), String>,
{
    let destinations = validate(requested, evidence)?;
    if rendered.len() != requested.len() {
        return Err(PublicationError::new(
            "report output and renderer inventories differ",
        ));
    }
    let mut written = Map::new();
    for (((name, original), destination), bytes) in requested.iter().zip(destinations).zip(rendered)
    {
        if let Err(message) = writer(&destination, bytes) {
            return Err(PublicationError {
                message,
                failed_output: Some(name.clone()),
                written_outputs: written,
            });
        }
        written.insert(
            name.clone(),
            Value::String(original.to_string_lossy().into_owned()),
        );
    }
    Ok(written)
}
/// A complete presentation transaction; the payload is the public report contract.
pub struct ReportOutput {
    pub text: String,
    pub payload: Value,
    pub status: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_partial_publication_and_never_overwrites() {
        let root = std::env::temp_dir().join(format!(
            "invarlock-report-publication-{}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let request = vec![
            ("html".into(), root.join("a.html")),
            ("markdown".into(), root.join("b.md")),
        ];
        let mut calls = 0;
        let error = publish_with(
            &request,
            &[b"first".to_vec(), b"second".to_vec()],
            None,
            |p, b| {
                calls += 1;
                if calls == 2 {
                    Err("injected publication failure".into())
                } else {
                    fs::write_file_no_replace(p, b, 0o600, true).map_err(|e| e.to_string())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error.failed_output.as_deref(), Some("markdown"));
        assert_eq!(error.written_outputs.len(), 1);
        assert_eq!(std::fs::read(&request[0].1).unwrap(), b"first");
        assert!(!request[1].1.exists());
        assert!(publish(&request, &[b"replace".to_vec(), b"second".to_vec()], None).is_err());
        assert_eq!(std::fs::read(&request[0].1).unwrap(), b"first");
        std::fs::remove_dir_all(root).unwrap();
    }
}
