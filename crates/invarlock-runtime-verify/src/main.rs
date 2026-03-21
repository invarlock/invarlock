use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const CONTRACT_VERSION: &str = "runtime-manifest-v1";

struct Args {
    report: PathBuf,
    manifest: PathBuf,
    json: bool,
}

struct Verdict {
    ok: bool,
    errors: Vec<String>,
}

fn main() -> ExitCode {
    let args = match parse_args(env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let verdict = verify(&args.report, &args.manifest);
    let exit_code = if verdict.ok { 0 } else { 1 };

    if args.json {
        println!(
            "{{\"ok\":{},\"errors\":[{}],\"report\":\"{}\",\"manifest\":\"{}\"}}",
            if verdict.ok { "true" } else { "false" },
            verdict
                .errors
                .iter()
                .map(|item| format!("\"{}\"", json_escape(item)))
                .collect::<Vec<_>>()
                .join(","),
            json_escape(&args.report.display().to_string()),
            json_escape(&args.manifest.display().to_string())
        );
    } else if verdict.ok {
        println!(
            "runtime verify ok report={} manifest={}",
            args.report.display(),
            args.manifest.display()
        );
    } else {
        for error in verdict.errors {
            eprintln!("{error}");
        }
    }

    ExitCode::from(exit_code)
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    let mut report: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut json = false;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--report requires a path".to_string())?;
                report = Some(PathBuf::from(value));
            }
            "--manifest" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--manifest requires a path".to_string())?;
                manifest = Some(PathBuf::from(value));
            }
            "--json" => {
                json = true;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: invarlock-runtime-verify --report <path> --manifest <path> [--json]"
                        .to_string(),
                );
            }
            other => {
                return Err(format!("unknown argument: {other}"));
            }
        }
        index += 1;
    }

    Ok(Args {
        report: report.ok_or_else(|| "--report is required".to_string())?,
        manifest: manifest.ok_or_else(|| "--manifest is required".to_string())?,
        json,
    })
}

fn verify(report_path: &Path, manifest_path: &Path) -> Verdict {
    let mut errors: Vec<String> = Vec::new();

    let report_bytes = match fs::read(report_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            errors.push(format!("unable to read report: {err}"));
            return Verdict { ok: false, errors };
        }
    };
    let manifest_text = match fs::read_to_string(manifest_path) {
        Ok(text) => text,
        Err(err) => {
            errors.push(format!("unable to read manifest: {err}"));
            return Verdict { ok: false, errors };
        }
    };

    let manifest_contract = extract_json_string(&manifest_text, "verifier_contract_version");
    if manifest_contract.as_deref() != Some(CONTRACT_VERSION) {
        errors.push(format!(
            "unexpected verifier contract version: {}",
            manifest_contract.unwrap_or_else(|| "<missing>".to_string())
        ));
    }

    let execution_mode = extract_json_string(&manifest_text, "execution_mode");
    if execution_mode.as_deref() != Some("container") {
        errors.push(format!(
            "execution_mode must be \"container\", got {}",
            execution_mode.unwrap_or_else(|| "<missing>".to_string())
        ));
    }

    let container_execution = extract_json_bool(&manifest_text, "container_execution");
    if container_execution != Some(true) {
        errors.push("runtime.container_execution must be true".to_string());
    }

    let image_digest = extract_json_string(&manifest_text, "image_digest");
    if image_digest.as_deref().unwrap_or("").is_empty() {
        errors.push("runtime.image_digest must be present".to_string());
    }

    let expected_report_sha = extract_json_object_string(&manifest_text, "report", "sha256");
    match (expected_report_sha, compute_sha256(report_path)) {
        (Some(expected), Ok(actual)) => {
            if expected != actual {
                errors.push(format!(
                    "report digest mismatch: manifest={} actual={}",
                    expected, actual
                ));
            }
        }
        (None, _) => errors.push("manifest is missing report.sha256".to_string()),
        (_, Err(err)) => errors.push(err),
    }

    if report_bytes.is_empty() {
        errors.push("report file is empty".to_string());
    }

    Verdict {
        ok: errors.is_empty(),
        errors,
    }
}

fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let key_pattern = format!("\"{key}\"");
    let start = text.find(&key_pattern)?;
    let mut index = start + key_pattern.len();
    index = skip_whitespace(text, index);
    if text.as_bytes().get(index).copied()? != b':' {
        return None;
    }
    index = skip_whitespace(text, index + 1);
    if text.as_bytes().get(index).copied()? != b'"' {
        return None;
    }
    parse_json_string(text, index + 1)
}

fn extract_json_object_string(text: &str, object_key: &str, field_key: &str) -> Option<String> {
    let object_text = extract_json_object(text, object_key)?;
    extract_json_string(object_text, field_key)
}

fn extract_json_object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let key_pattern = format!("\"{key}\"");
    let start = text.find(&key_pattern)?;
    let mut index = start + key_pattern.len();
    index = skip_whitespace(text, index);
    if text.as_bytes().get(index).copied()? != b':' {
        return None;
    }
    index = skip_whitespace(text, index + 1);
    if text.as_bytes().get(index).copied()? != b'{' {
        return None;
    }

    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut cursor = index;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return text.get(index..=cursor);
                }
            }
            b'"' => {
                cursor += 1;
                while cursor < bytes.len() {
                    match bytes[cursor] {
                        b'\\' => cursor += 1,
                        b'"' => break,
                        _ => {}
                    }
                    cursor += 1;
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

fn extract_json_bool(text: &str, key: &str) -> Option<bool> {
    let key_pattern = format!("\"{key}\"");
    let start = text.find(&key_pattern)?;
    let mut index = start + key_pattern.len();
    index = skip_whitespace(text, index);
    if text.as_bytes().get(index).copied()? != b':' {
        return None;
    }
    index = skip_whitespace(text, index + 1);
    let tail = &text[index..];
    if tail.starts_with("true") {
        Some(true)
    } else if tail.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn skip_whitespace(text: &str, mut index: usize) -> usize {
    while let Some(byte) = text.as_bytes().get(index) {
        if !matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
            break;
        }
        index += 1;
    }
    index
}

fn parse_json_string(text: &str, start: usize) -> Option<String> {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut out = String::new();
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => {
                index += 1;
                let escaped = *bytes.get(index)?;
                match escaped {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    _ => return None,
                }
            }
            b'"' => return Some(out),
            value => out.push(value as char),
        }
        index += 1;
    }
    None
}

fn compute_sha256(path: &Path) -> Result<String, String> {
    let path_str = path.display().to_string();
    let commands: [&[&str]; 2] = [
        &["sha256sum", &path_str],
        &["shasum", "-a", "256", &path_str],
    ];
    for command in commands {
        let output = Command::new(command[0]).args(&command[1..]).output();
        let output = match output {
            Ok(output) => output,
            Err(_) => continue,
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(token) = stdout.split_whitespace().next() {
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }
    }
    Err("unable to compute sha256 digest using sha256sum/shasum".to_string())
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = env::temp_dir().join(format!("invarlock-{name}-{stamp}"));
        fs::create_dir_all(&path).expect("mkdir");
        path
    }

    #[test]
    fn verify_passes_for_attested_container_manifest() {
        let dir = unique_dir("runtime-verify-ok");
        let report_path = dir.join("evaluation.report.json");
        fs::write(&report_path, "{\"ok\":true}\n").expect("write report");
        let digest = compute_sha256(&report_path).expect("digest");
        let manifest_path = dir.join("runtime.manifest.json");
        fs::write(
            &manifest_path,
            format!(
                "{{\n  \"execution_mode\": \"container\",\n  \"report\": {{\"sha256\": \"{digest}\"}},\n  \"runtime\": {{\"container_execution\": true, \"image_digest\": \"sha256:abc\"}},\n  \"verifier_contract_version\": \"{CONTRACT_VERSION}\"\n}}\n"
            ),
        )
        .expect("write manifest");

        let verdict = verify(&report_path, &manifest_path);
        assert!(verdict.ok, "{:?}", verdict.errors);
    }

    #[test]
    fn verify_fails_without_image_digest() {
        let dir = unique_dir("runtime-verify-fail");
        let report_path = dir.join("evaluation.report.json");
        fs::write(&report_path, "{\"ok\":true}\n").expect("write report");
        let digest = compute_sha256(&report_path).expect("digest");
        let manifest_path = dir.join("runtime.manifest.json");
        fs::write(
            &manifest_path,
            format!(
                "{{\n  \"execution_mode\": \"container\",\n  \"report\": {{\"sha256\": \"{digest}\"}},\n  \"runtime\": {{\"container_execution\": true, \"image_digest\": \"\"}},\n  \"verifier_contract_version\": \"{CONTRACT_VERSION}\"\n}}\n"
            ),
        )
        .expect("write manifest");

        let verdict = verify(&report_path, &manifest_path);
        assert!(!verdict.ok);
        assert!(verdict
            .errors
            .iter()
            .any(|item| item.contains("runtime.image_digest must be present")));
    }

    #[test]
    fn verify_reads_report_sha256_even_when_config_sha256_is_null() {
        let dir = unique_dir("runtime-verify-nested-report-sha");
        let report_path = dir.join("evaluation.report.json");
        fs::write(&report_path, "{\"ok\":true}\n").expect("write report");
        let digest = compute_sha256(&report_path).expect("digest");
        let manifest_path = dir.join("runtime.manifest.json");
        fs::write(
            &manifest_path,
            format!(
                "{{\n  \"config\": {{\"path\": null, \"sha256\": null, \"source\": \"missing\"}},\n  \"execution_mode\": \"container\",\n  \"report\": {{\"filename\": \"evaluation.report.json\", \"path\": \"{}\", \"sha256\": \"{digest}\"}},\n  \"runtime\": {{\"container_execution\": true, \"image_digest\": \"sha256:abc\"}},\n  \"verifier_contract_version\": \"{CONTRACT_VERSION}\"\n}}\n",
                report_path.display()
            ),
        )
        .expect("write manifest");

        let verdict = verify(&report_path, &manifest_path);
        assert!(verdict.ok, "{:?}", verdict.errors);
    }
}
