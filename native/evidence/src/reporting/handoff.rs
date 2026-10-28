//! Client commentary and immutable evidence packaging. This never authenticates a recipient.
use super::presentation::*;
use crate::{EvidenceError, Result};
use invarlock_native_contracts as contracts;
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
fn error(e: impl std::fmt::Display) -> EvidenceError {
    EvidenceError::incomplete(e.to_string())
}
const LIMIT: usize = 2 * 1024 * 1024 * 1024;
fn hash(path: &Path) -> Result<String> {
    fs::PinnedRegularFile::open(path)
        .and_then(|f| f.sha256_digest(LIMIT as u64 * 2))
        .map_err(error)
}
fn pretty(v: &Value) -> Vec<u8> {
    let mut s = serde_json::to_vec_pretty(v).expect("JSON value");
    s.push(b'\n');
    s
}
fn context(raw: &[u8]) -> Result<Value> {
    let v = contracts::parse_json(raw, "review context")?;
    let required = [
        "format",
        "title",
        "review_date",
        "byline",
        "context",
        "next_action",
    ];
    let optional = ["label", "recommendation", "decision", "basis"];
    let Some(map) = v.as_object() else {
        return Err(error("review context has missing or unsupported fields"));
    };
    if required.iter().any(|k| !map.contains_key(*k))
        || map
            .keys()
            .any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(error("review context has missing or unsupported fields"));
    }
    if v["format"] != "invarlock/client-review-context-v1" {
        return Err(error("unsupported review context format"));
    }
    let valid = |v: &Value| {
        v.as_str().is_some_and(|s| {
            !s.trim().is_empty()
                && s.chars().count() <= 4000
                && !s.chars().any(|c| c < ' ' && c != '\n' && c != '\t')
        })
    };
    for k in required
        .into_iter()
        .chain(optional)
        .filter(|k| !matches!(*k, "context" | "basis"))
    {
        if let Some(value) = v.get(k) {
            if !valid(value) {
                return Err(error(format!(
                    "review context {k} must be nonempty text of at most 4000 characters"
                )));
            }
        }
    }
    for k in ["context", "basis"] {
        if let Some(rows) = v.get(k) {
            let rows = rows.as_array().filter(|r| r.len() <= 32).ok_or_else(|| {
                error(format!(
                    "review context {k} must contain at most 32 labeled rows"
                ))
            })?;
            for row in rows {
                if !row
                    .as_array()
                    .is_some_and(|r| r.len() == 2 && r.iter().all(valid))
                {
                    return Err(error(format!(
                        "review context {k} rows require two nonempty strings"
                    )));
                }
            }
        }
    }
    if a(&v, "context").is_empty() {
        return Err(error(
            "review context requires intended-use or decision context",
        ));
    }
    let date = t(&v, "review_date");
    if date.len() != 10
        || date.as_bytes()[4] != b'-'
        || date.as_bytes()[7] != b'-'
        || crate::attestation_time::parse(&format!("{date}T00:00:00Z"), "review_date").is_err()
    {
        return Err(error("review_date must be YYYY-MM-DD"));
    }
    Ok(v)
}
fn loaded(path: &Path) -> Result<(Value, &'static str, String)> {
    if path.join("evidence-set.json").exists() {
        let x = super::sets::load(path, None)?;
        return Ok((
            x["view"].clone(),
            "evidence_set",
            t(&x, "index_sha256").into(),
        ));
    }
    if path.join("envelope.json").exists() {
        let x = super::load_judge_report(path, None)?;
        return Ok((
            x["view"].clone(),
            "judge",
            crate::judge::object_digest(&x["envelope"])?,
        ));
    }
    let raw = fs::read_file(&path.join("manifest.json"), 256 * 1024).map_err(error)?;
    let manifest = contracts::parse_json(&raw, "evidence manifest")?;
    if manifest["kind"] == "captured" {
        let x = super::captured_view::load_projection(path)?;
        return Ok((
            x["view"].clone(),
            "captured",
            crate::digest(&x["manifest"])?,
        ));
    }
    let x = super::load_native_report(path)?;
    Ok((
        x["view"].clone(),
        "runtime",
        t(&x, "manifest_digest").into(),
    ))
}
fn excerpt(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        return s.into();
    }
    let short = s.chars().take(limit).collect::<String>();
    format!(
        "{}… (full text in review)",
        short.rsplit_once(' ').map_or(short.as_str(), |(s, _)| s)
    )
}
fn document(title: &str, body: &str) -> String {
    format!("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\"><title>{}</title><style>{}</style></head><body><main>{body}</main></body></html>\n",e(title),include_str!("assets/handoff/style.css"))
}
fn brief(v: &Value, digest: &str, receipt: bool, html: bool) -> String {
    let r = &v["review"];
    let context = &a(r, "context")[..a(r, "context").len().min(2)];
    let metrics = &a(v, "metrics")[..a(v, "metrics").len().min(6)];
    let note = if receipt {
        "Not authenticated by packaging. The attached historical receipt references this evidence; its signature and recipient authority have not been checked here."
    } else {
        "No historical receipt supplied."
    };
    let more=format!("Showing {} of {} metric/scope results; all results and checks are in the technical appendix.",metrics.len(),a(v,"metrics").len());
    let limits = a(v, "limitations")
        .iter()
        .take(2)
        .map(s)
        .collect::<Vec<_>>()
        .join(" ");
    let limits = if limits.is_empty() {
        "No limitations were supplied. Confirm workload coverage before deciding."
    } else {
        &limits
    };
    let intro="Authored commentary is separate from signed evidence. A policy result is not organizational approval.";
    if !html {
        let clean = super::markdown::clean;
        let mut lines = vec![
            format!("# {}", clean(t(v, "title"))),
            "".into(),
            "Decision brief".into(),
            "".into(),
            clean(t(r, "label")),
            "".into(),
            clean(t(r, "byline")),
            "".into(),
            intro.into(),
            "".into(),
            format!("**Owner decision:** {}", clean(t(r, "decision"))),
            "".into(),
            format!(
                "**Recorded policy result:** {}",
                clean(decision(t(v, "decision")))
            ),
            "".into(),
            clean(t(v, "summary")),
            "".into(),
            format!(
                "**Recommendation:** {}",
                clean(&excerpt(t(r, "recommendation"), 360))
            ),
            "".into(),
            "## Intended use and review context".into(),
            "".into(),
        ];
        lines.extend(context.iter().map(|p| {
            format!(
                "- **{}:** {}",
                clean(s(&p[0])),
                clean(&excerpt(s(&p[1]), 360))
            )
        }));
        lines.extend([
            "".into(),
            "## Results at a glance".into(),
            "".into(),
            more,
            "".into(),
        ]);
        lines.extend(metrics.iter().map(|m| {
            format!(
                "- {} / {}: baseline {}; candidate {}; change {}; {}.",
                clean(&display_label(t(m, "name"))),
                clean(&display_label(t(m, "scope"))),
                clean(t(m, "baseline")),
                clean(t(m, "candidate")),
                clean(t(m, "change")),
                clean(decision(t(m, "decision")))
            )
        }));
        lines.extend(["".into(),"## Decision limits".into(),"".into(),clean(&excerpt(limits,600)),"".into(),"Full recorded limitations and authored rationale are in the review and appendix.".into(),"".into(),"## Recipient check".into(),"".into(),"Not performed for this recipient. Follow the verification guide with separately approved inputs.".into(),"".into(),note.into(),"".into(),"## Next action".into(),"".into(),clean(&excerpt(t(r,"next_action"),360)),"".into(),format!("Evidence statement: {digest}"),"".into(),a(v,"handoff_links").iter().map(|p|format!("[{}]({})",s(&p[0]),s(&p[1]))).collect::<Vec<_>>().join(" · "),"".into()]);
        return lines.join("\n");
    }
    let nav = a(v, "handoff_links")
        .iter()
        .map(|p| format!("<a href=\"{}\">{}</a>", e(s(&p[1])), e(s(&p[0]))))
        .collect::<String>();
    let rows=metrics.iter().map(|m|format!("<tr><th scope=\"row\">{}<br><span class=\"scope\">{}</span></th><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",e(&display_label(t(m,"name"))),e(&display_label(t(m,"scope"))),e(t(m,"baseline")),e(t(m,"candidate")),e(t(m,"change")),e(decision(t(m,"decision"))))).collect::<String>();
    let contexts = context
        .iter()
        .map(|p| {
            format!(
                "<p class=\"context\"><strong>{}.</strong> {}</p>",
                e(s(&p[0])),
                e(&excerpt(s(&p[1]), 360))
            )
        })
        .collect::<String>();
    let body=format!("<div class=\"brand\">InvarLock · Decision brief</div><h1>{}</h1><p class=\"scope\">{} · {}</p><p class=\"scope\">{intro}</p><nav aria-label=\"Review package\">{nav}</nav><section class=\"decision\"><p><strong>Owner decision.</strong> {}</p><p><strong>Recorded policy result.</strong> {}</p><p>{}</p><p><strong>Recommendation.</strong> {}</p></section><section><h2>Intended use and review context</h2>{contexts}<p class=\"scope\">Full context and attributed rationale: <a href=\"review.html\">full review</a>.</p></section><section><h2>Results at a glance</h2><p class=\"scope\">{more}</p><div class=\"scroll\"><table><thead><tr><th scope=\"col\">Metric / scope</th><th scope=\"col\">Baseline</th><th scope=\"col\">Candidate</th><th scope=\"col\">Change</th><th scope=\"col\">Recorded result</th></tr></thead><tbody>{rows}</tbody></table></div></section><div class=\"grid\"><section><h2>Decision limits</h2><p>{}</p><p class=\"scope\">Full recorded limitations and authored rationale remain in the review and appendix.</p></section><section><h2>Recipient check</h2><p><strong>Not performed for this recipient.</strong> Follow the verification guide with separately approved inputs.</p><p class=\"scope\">{note}</p></section></div><section><h2>Next action</h2><p>{}</p></section><footer>Evidence statement identity (not approval): <span class=\"identity\">{}</span><br>Keep this brief with the full review, technical appendix, immutable evidence and separate recipient result.</footer>",e(t(v,"title")),e(t(r,"label")),e(t(r,"byline")),e(&excerpt(t(r,"decision"),600)),e(decision(t(v,"decision"))),e(t(v,"summary")),e(&excerpt(t(r,"recommendation"),360)),e(&excerpt(limits,600)),e(&excerpt(t(r,"next_action"),360)),e(digest));
    document(t(v, "title"), &body)
}
fn guide(kind: &str, html: bool, verdict: &str) -> String {
    let template = match (kind, html) {
        ("captured", true) => include_str!("assets/handoff/captured.html"),
        ("captured", false) => include_str!("assets/handoff/captured.md"),
        ("runtime", true) => include_str!("assets/handoff/runtime.html"),
        ("runtime", false) => include_str!("assets/handoff/runtime.md"),
        ("judge", true) => include_str!("assets/handoff/judge.html"),
        ("judge", false) => include_str!("assets/handoff/judge.md"),
        (_, true) => include_str!("assets/handoff/evidence_set.html"),
        (_, false) => include_str!("assets/handoff/evidence_set.md"),
    };
    template.replace(
        "recorded policy result: Decision unavailable.",
        &format!("recorded policy result: {}.", decision(verdict)),
    )
}
pub fn build(config: &Value) -> Result<Value> {
    let evidence =
        std::path::absolute(crate::required_string(config, "evidence_path")?).map_err(error)?;
    let output = std::path::absolute(crate::required_string(config, "output")?).map_err(error)?;
    super::publication::validate(&[("package".into(), output.clone())], Some(&evidence))
        .map_err(error)?;
    let context_path = Path::new(crate::required_string(config, "context_path")?);
    let raw = fs::read_file(context_path, 64 * 1024).map_err(error)?;
    let context = context(&raw)?;
    let limits = fs::SnapshotLimits {
        max_entries: 4096,
        max_files: 4096,
        max_file_bytes: LIMIT,
        max_total_bytes: LIMIT,
    };
    let source = fs::PinnedDirectory::open(&evidence, false).map_err(error)?;
    let inventory = source.inventory_metadata(limits).map_err(error)?;
    let parent = output
        .parent()
        .ok_or_else(|| error("missing package parent"))?;
    let _parent = fs::PinnedDirectory::open(parent, true).map_err(error)?;
    let mut stage = fs::StagedDirectory::with_limits(
        &output,
        fs::SnapshotLimits {
            max_entries: 8192,
            max_files: 8192,
            max_file_bytes: LIMIT * 2,
            max_total_bytes: LIMIT * 3,
        },
    )
    .map_err(error)?;
    stage
        .create_directory(Path::new("evidence"))
        .map_err(error)?;
    let root = stage.path().map_err(error)?.to_path_buf();
    let mut originals = BTreeMap::new();
    for (name, entry) in &inventory {
        if name.split('/').count() > 17 {
            return Err(error("evidence directory nesting exceeds handoff limit"));
        }
        if name.contains('\\') || name.chars().any(|c| c < ' ') {
            return Err(error("evidence filename is not portable in a handoff"));
        }
        let relative = PathBuf::from("evidence").join(name);
        if entry.identity.mode & 0o170000 == 0o040000 {
            stage.create_directory(&relative).map_err(error)?;
            continue;
        }
        let file = fs::PinnedRegularFile::open(&evidence.join(name)).map_err(error)?;
        if file.metadata().identity != entry.identity {
            return Err(error("evidence changed before copying"));
        }
        stage
            .write_with(&relative, 0o644, |out| {
                file.copy_to(out, LIMIT as u64)?;
                Ok(())
            })
            .map_err(error)?;
        originals.insert(format!("evidence/{name}"), hash(&root.join(&relative))?);
    }
    let root = stage.path().map_err(error)?.to_path_buf();
    let (mut view, kind, digest) = loaded(&root.join("evidence"))?;
    let receipt = config["retained_receipt"].as_str();
    if let Some(path) = receipt {
        let raw = fs::read_file(Path::new(path), 8 * 1024 * 1024).map_err(error)?;
        let v = contracts::parse_json(&raw, "historical receipt")?;
        if !v.is_object() {
            return Err(error("historical receipt must be a JSON object"));
        }
        if v.get("statement").is_some_and(|v| !v.is_object()) {
            return Err(error("historical receipt statement must be an object"));
        }
        let claimed = match kind {
            "evidence_set" => v["index_sha256"].as_str(),
            "judge" => v["statement"]["result"]["envelope_sha256"].as_str(),
            _ => v["statement"]["pack_manifest_digest"].as_str(),
        };
        if claimed.map(|s| s.trim_start_matches("sha256:"))
            != Some(digest.trim_start_matches("sha256:"))
        {
            return Err(error(
                "historical receipt does not bind this evidence statement",
            ));
        }
        stage
            .write(Path::new("verification.receipt.json"), &raw, 0o644)
            .map_err(error)?;
        originals.insert(
            "verification.receipt.json".into(),
            hash(&root.join("verification.receipt.json"))?,
        );
    }
    stage
        .write(Path::new("review-context.json"), &raw, 0o644)
        .map_err(error)?;
    stage
        .write(
            Path::new("original-file-hashes.json"),
            &pretty(&json!(originals)),
            0o644,
        )
        .map_err(error)?;
    let mut links = vec![
        json!(["Decision brief", "brief.html"]),
        json!(["Full review", "review.html"]),
        json!(["Technical appendix", "technical-report.html"]),
        json!(["Verify this package", "verification.html"]),
    ];
    if receipt.is_some() {
        links.push(json!([
            "Historical receipt (unverified attachment)",
            "verification.receipt.json"
        ]));
    }
    view["handoff_links"] = json!(links);
    for (name, bytes) in [
        ("technical-report.html", super::render_html(&view)),
        ("technical-report.md", super::render_markdown(&view, false)),
    ] {
        stage
            .write(Path::new(name), bytes.as_bytes(), 0o644)
            .map_err(error)?;
    }
    view["title"] = context["title"].clone();
    view["review"] = json!({"label":context.get("label").cloned().unwrap_or(json!("Authored client review")),"byline":format!("{} · {}",t(&context,"byline"),t(&context,"review_date")),"context":context["context"],"recommendation":context.get("recommendation").cloned().unwrap_or(json!("No owner recommendation supplied.")),"decision":context.get("decision").cloned().unwrap_or(json!("Pending owner decision")),"basis":context.get("basis").cloned().unwrap_or(json!([])),"next_action":context["next_action"]});
    if let Some(metrics) = view["metrics"].as_array_mut() {
        for m in metrics {
            m["evidence_source"] = json!(format!("evidence/{}", t(m, "evidence_source")));
        }
    }
    for (name, bytes) in [
        ("review.html", super::render_html(&view)),
        ("review.md", super::render_markdown(&view, false)),
        ("brief.html", brief(&view, &digest, receipt.is_some(), true)),
        ("brief.md", brief(&view, &digest, receipt.is_some(), false)),
        ("verification.html", guide(kind, true, t(&view, "decision"))),
        ("verification.md", guide(kind, false, t(&view, "decision"))),
        (
            "README.txt",
            include_str!("assets/handoff/README.txt").into(),
        ),
    ] {
        stage
            .write(Path::new(name), bytes.as_bytes(), 0o644)
            .map_err(error)?;
    }
    let target = fs::PinnedDirectory::open(&root, false).map_err(error)?;
    let files = target
        .inventory_metadata(fs::SnapshotLimits {
            max_entries: 8192,
            max_files: 8192,
            max_file_bytes: LIMIT,
            max_total_bytes: LIMIT + 16 * 1024 * 1024,
        })
        .map_err(error)?
        .into_iter()
        .filter(|(_, e)| e.identity.mode & 0o170000 == 0o100000)
        .map(|(name, _)| hash(&root.join(&name)).map(|h| (name, h)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = json!({"format":"invarlock/client-review-package-v1","generator_version":"0.17.3","generator_source_sha256":contracts::sha256_digest(include_bytes!("handoff.rs")).trim_start_matches("sha256:"),"evidence_kind":kind,"evidence_statement_sha256":digest,"recipient_verification":"not_performed","historical_receipt":if receipt.is_some(){"unverified_attachment"}else{"not_supplied"},"inventory_scope":"All delivered files except this inventory and handoff.zip; unsigned identity record, not acceptance.","files":files});
    stage
        .write(
            Path::new("package-manifest.json"),
            &pretty(&manifest),
            0o644,
        )
        .map_err(error)?;
    let mut names = files.keys().cloned().collect::<Vec<_>>();
    names.push("package-manifest.json".into());
    names.sort();
    stage
        .write_with(Path::new("handoff.zip"), 0o644, |out| {
            let mut zip = zip::ZipWriter::new(out);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o644);
            for name in names {
                zip.start_file(&name, options)
                    .map_err(std::io::Error::other)?;
                fs::PinnedRegularFile::open(&root.join(&name))?.copy_to(&mut zip, LIMIT as u64)?;
            }
            zip.finish().map_err(std::io::Error::other)?;
            Ok(())
        })
        .map_err(error)?;
    for (name, expected) in originals {
        if hash(&root.join(name))? != expected {
            return Err(error("copied originals changed during packaging"));
        }
    }
    let after = source.inventory_metadata(limits).map_err(error)?;
    if inventory.len() != after.len()
        || inventory
            .iter()
            .any(|(k, v)| after.get(k).is_none_or(|x| x.identity != v.identity))
    {
        return Err(error("evidence changed during handoff preparation"));
    }
    source.verify().map_err(error)?;
    stage.commit().map_err(error)?;
    Ok(
        json!({"format_version":"invarlock/client-review-package-v1","ok":true,"kind":kind,"output":output,"brief":output.join("brief.html"),"archive":output.join("handoff.zip"),"evidence_statement_sha256":digest,"recipient_verification":"not_performed"}),
    )
}
