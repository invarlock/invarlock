//! Descriptor-bound native evidence presentation snapshot.
use super::{native_checks, native_view, presentation::*};
use crate::{native_pack as pack, EvidenceError, Result};
use invarlock_native_contracts::{
    canonical_json, first_schema_error, parse_json, sha256_digest, SchemaErrorOrder,
};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn object(snapshot: &fs::PackSnapshot, path: &str, label: &str) -> Result<Value> {
    let v = parse_json(pack::raw(snapshot, path, 64 * 1024 * 1024)?, label)?;
    if !v.is_object() {
        return Err(err(format!("{label} must be a JSON object")));
    }
    Ok(v)
}
fn text(v: &Value) -> String {
    let Some(s) = v.as_str().filter(|s| !s.trim().is_empty()) else {
        return "Unavailable in evidence".into();
    };
    let mut text = s.chars().take(256).collect::<String>();
    if s.chars().count() > 256 {
        text += "… (preview)"
    }
    text
}
pub fn load(path: &Path) -> Result<Value> {
    let snapshot = fs::PackSnapshot::capture(
        path,
        fs::SnapshotLimits {
            max_entries: 256,
            max_files: 128,
            max_file_bytes: 64 * 1024 * 1024,
            max_total_bytes: 512 * 1024 * 1024,
        },
    )
    .map_err(|e| err(e.to_string()))?;
    let manifest_raw = pack::raw(&snapshot, "manifest.json", 256 * 1024)?;
    let manifest = object(&snapshot, "manifest.json", "evidence manifest")?;
    if let Some(v) = first_schema_error(&manifest, "evidence_pack", SchemaErrorOrder::Instance)? {
        return Err(err(format!(
            "evidence manifest schema failed at {}: {}",
            if v.path.is_empty() {
                "<root>".into()
            } else {
                v.path.join(".")
            },
            v.message
        )));
    }
    if canonical_json(&manifest)? != manifest_raw {
        return Err(err("evidence manifest is not canonical JSON"));
    }
    let signer = pack::authenticate(
        &snapshot,
        &manifest,
        manifest_raw,
        &json!({"signer_fingerprint":manifest["signing_key_fingerprint"]}),
    )?;
    pack::checksums(&snapshot, &manifest)?;
    let mut expected = BTreeSet::new();
    let mut identities = BTreeMap::new();
    for block in ["inputs", "evidence"] {
        for (role, reference) in manifest[block].as_object().unwrap() {
            let rel = t(reference, "path");
            expected.insert(rel.to_owned());
            let raw = pack::raw(&snapshot, rel, 64 * 1024 * 1024)?;
            if reference["digest"] != sha256_digest(raw) {
                return Err(err(format!("manifest digest does not bind {rel}")));
            }
            if block == "inputs" {
                let v = object(&snapshot, rel, &format!("input identity {role}"))?;
                if reference["material_digest"] != v["digest"] {
                    return Err(err(format!("manifest material digest does not bind {rel}")));
                }
                identities.insert(role.as_str(), v);
            }
        }
    }
    let paired = &manifest["paired_records"];
    let paired_path = t(paired, "path");
    if !paired_path.is_empty() {
        expected.insert(paired_path.to_owned());
        let raw = pack::raw(&snapshot, paired_path, 64 * 1024 * 1024)?;
        if paired["digest"] != sha256_digest(raw) {
            return Err(err("manifest digest does not bind paired records"));
        }
        let value = object(&snapshot, paired_path, "paired records")?;
        if value["records"].as_array().map(|a| a.len() as u64) != paired["count"].as_u64() {
            return Err(err("manifest count does not bind paired records"));
        }
    }
    let request = object(&snapshot, "request.json", "normalized request")?;
    if canonical_json(&request)? != pack::raw(&snapshot, "request.json", 64 * 1024 * 1024)? {
        return Err(err("normalized request is not canonical JSON"));
    }
    pack::replay::observations(&snapshot, &manifest, &request, &identities)?;
    let mut observations = Vec::new();
    if let Some(refs) = manifest["observations"].as_object() {
        let refs = refs.iter().collect::<BTreeMap<_, _>>();
        for (_, r) in refs {
            let path = t(r, "path");
            expected.insert(path.into());
            observations.push(object(&snapshot, path, "observation")?);
        }
    }
    let covered = std::str::from_utf8(pack::raw(&snapshot, "checksums.sha256", 1024 * 1024)?)
        .map_err(|e| err(e.to_string()))?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            l[66..]
                .replace('\\', "/")
                .trim_start_matches("./")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    if expected != covered {
        let missing = expected.difference(&covered).cloned().collect::<Vec<_>>();
        let extra = covered.difference(&expected).cloned().collect::<Vec<_>>();
        let mut errors = Vec::new();
        if !missing.is_empty() {
            errors.push(format!(
                "checksums.sha256 does not cover manifest payloads: {}",
                missing.join(", ")
            ))
        }
        if !extra.is_empty() {
            errors.push(format!(
                "checksums.sha256 covers files outside the evidence manifest: {}",
                extra.join(", ")
            ))
        }
        return Err(err(errors.join("; ")));
    }
    expected.extend(
        [
            "manifest.json",
            "manifest.signature.json",
            "checksums.sha256",
        ]
        .map(str::to_owned),
    );
    let actual = snapshot.entries().keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected {
        let extra = actual.difference(&expected).cloned().collect::<Vec<_>>();
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        return Err(err(if !extra.is_empty() {
            format!(
                "evidence pack contains files outside the evidence manifest: {}",
                extra.join(", ")
            )
        } else {
            format!(
                "evidence pack is missing closed inventory files: {}",
                missing.join(", ")
            )
        }));
    }
    let report = object(
        &snapshot,
        "reports/evaluation.report.json",
        "canonical evaluation report",
    )?;
    if canonical_json(&report)?
        != pack::raw(
            &snapshot,
            "reports/evaluation.report.json",
            64 * 1024 * 1024,
        )?
    {
        return Err(err("canonical evaluation report is not canonical JSON"));
    }
    native_checks::validate(&report)?;
    let mut subjects = Vec::new();
    for (role, label) in [
        ("baseline", "Baseline artifact"),
        ("subject", "Candidate artifact"),
    ] {
        let i = &identities[role];
        subjects.push(json!([
            label,
            i.get("locator")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| t(i, "digest").into())
        ]));
    }
    let comparison = &request["comparison"];
    let mode = t(&request["execution"], "mode");
    let mut context = vec![
        json!([
            "Workflow",
            match mode {
                "run" => "Runtime execution",
                "import" => "Imported runtime evidence",
                _ => "Unavailable in evidence",
            }
        ]),
        json!(["Task", text(&comparison["task"])]),
        json!([
            "Paired records",
            comma(report["record_count"].as_u64().unwrap())
        ]),
    ];
    for (side, label) in [("baseline", "Baseline"), ("subject", "Candidate")] {
        context.extend([
            json!([
                format!("{label} model ID"),
                text(&comparison[side]["artifact"]["model_id"])
            ]),
            json!([
                format!("{label} provider"),
                text(&comparison[side]["runtime"]["provider"])
            ]),
        ]);
    }
    let dataset = &identities["dataset"];
    let prep = &comparison["dataset"];
    context.push(json!(["Schedule digest", text(&dataset["digest"])]));
    context.extend([
        json!([
            "Dataset",
            text(
                prep.get("name")
                    .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
                    .unwrap_or(&dataset["locator"])
            )
        ]),
        json!(["Dataset split", text(&prep["split"])]),
    ]);
    if mode == "run" && prep.as_object().is_some_and(|m| !m.is_empty()) {
        context.extend([
            json!(["Dataset source SHA-256", text(&prep["source_sha256"])]),
            json!(["Dataset source format", text(&prep["source_format"])]),
        ]);
        for (key, label) in [
            ("selected_record_count", "Selected records"),
            ("limit", "Selection limit"),
        ] {
            context.push(json!([
                label,
                prep[key]
                    .as_u64()
                    .map(comma)
                    .unwrap_or("Not specified".into())
            ]));
        }
    }
    let changes = json!([
        if identities["baseline"]["digest"] == identities["subject"]["digest"] {
            "Baseline and candidate have the same authenticated artifact digest."
        } else {
            "Baseline and candidate have different authenticated artifact digests. These digests alone do not establish how the candidate was produced."
        }
    ]);
    let view = native_view::build_native_view(
        &report,
        &signer,
        &json!(observations),
        &json!(subjects),
        &json!(context),
        &changes,
    )?;
    snapshot.verify_source().map_err(|e| err(e.to_string()))?;
    Ok(
        json!({"view":view,"manifest_digest":sha256_digest(manifest_raw),"manifest":manifest,"report":report}),
    )
}
