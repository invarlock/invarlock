//! Bounded projections of evaluator-recorded model and prompt context.
use super::presentation::*;
use crate::{EvidenceError, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn error(s: &str) -> EvidenceError {
    EvidenceError::contract(s)
}
pub(super) fn short(s: &str, limit: usize) -> String {
    if s.chars().count() > limit {
        format!("{}… (truncated)", s.chars().take(limit).collect::<String>())
    } else {
        s.into()
    }
}
pub(super) fn preview(v: &Value) -> Value {
    fn walk(v: &Value, depth: usize, remaining: &mut usize) -> String {
        if *remaining == 0 {
            return "… node budget reached".into();
        }
        *remaining -= 1;
        if let Some(s) = v.as_str() {
            let count = s.chars().count();
            let mut out = s.chars().take(256).collect::<String>();
            if count > 256 {
                out += &format!("… ({count} characters; preview)")
            }
            return pretty(&json!(out), 0, false);
        }
        if v.is_object() || v.is_array() {
            if depth >= 6 {
                return "… depth limit reached".into();
            }
            let mut parts = Vec::new();
            let children: Vec<_> = if let Some(o) = v.as_object() {
                o.iter().map(|(k, v)| (Some(k), v)).collect()
            } else {
                v.as_array().unwrap().iter().map(|v| (None, v)).collect()
            };
            for (k, v) in children {
                if *remaining == 0 {
                    parts.push("… node budget reached".into());
                    break;
                }
                let key = k
                    .map(|k| format!("{}: ", walk(&json!(k), depth + 1, remaining)))
                    .unwrap_or_default();
                parts.push(format!("{key}{}", walk(v, depth + 1, remaining)))
            }
            let (open, close) = if v.is_object() {
                ("{", "}")
            } else {
                ("[", "]")
            };
            if parts.is_empty() {
                return format!("{open}{close}");
            }
            let indent = "  ".repeat(depth + 1);
            return format!(
                "{open}\n{indent}{}\n{}{close}",
                parts.join(&format!(",\n{indent}")),
                "  ".repeat(depth)
            );
        }
        pretty(v, 0, false)
    }
    json!({"preview_only":true,"limits":{"depth":6,"nodes":128,"string_characters":256},"text":walk(v,0,&mut 128),"note":"Display preview, not a replacement configuration. Full configuration remains in evidence under the original policy binding."})
}
pub(super) fn policy_preview(p: &Value) -> Value {
    let mut p = p.clone();
    for m in p["metrics"].as_array_mut().unwrap() {
        m["configuration"] = preview(&m["configuration"])
    }
    p
}
fn common(records: &[Value], key: &str, container: &str) -> (Option<String>, String) {
    let values = records
        .iter()
        .filter_map(|r| r[container][key].as_str())
        .filter(|v| !v.trim().is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        (None, "Unavailable in recorded context".into())
    } else if values.len() != records.len() || values.iter().any(|v| *v != values[0]) {
        (None, "Mixed or incomplete across records".into())
    } else {
        (Some(values[0].into()), short(values[0], 256))
    }
}
fn summary(records: &[Value], key: &str) -> String {
    let values = records
        .iter()
        .filter_map(|r| r["metadata"][key].as_str())
        .filter(|v| !v.trim().is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return "Unavailable in recorded context".into();
    }
    if values.len() != records.len() {
        return "Mixed or incomplete across records".into();
    }
    let values = values.into_iter().collect::<BTreeSet<_>>();
    if values.len() == 1 {
        return short(values.first().unwrap(), 256);
    }
    let mut out = values
        .iter()
        .take(8)
        .map(|v| short(v, 96))
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 8 {
        out += &format!(", … ({} values)", comma(values.len() as u64))
    }
    format!("Multiple recorded values: {out}")
}
fn model(run: &Value, records: &[Value]) -> Result<Option<Value>> {
    let mut identities = Vec::new();
    for r in records {
        let candidates = [
            &r["context"]["upstream_record"]["metadata"]["invarlock_model_execution"],
            &r["context"]["langfuse"]["item_result"]["item"]["metadata"]
                ["invarlock_model_execution"],
        ]
        .into_iter()
        .filter(|v| !v.is_null())
        .collect::<Vec<_>>();
        if candidates.is_empty() {
            identities.push(None);
            continue;
        }
        let mut models = Vec::new();
        for candidate in candidates {
            let m = &candidate["model"];
            if !["id", "revision", "artifact_digest"]
                .iter()
                .all(|k| m[k].as_str().is_some_and(|s| !s.trim().is_empty()))
            {
                models.clear();
                break;
            }
            models.push(json!({"id":m["id"],"revision":m["revision"],"artifact_digest":m["artifact_digest"]}))
        }
        if models.is_empty() {
            identities.push(None);
            continue;
        }
        if models.iter().any(|m| m != &models[0]) {
            return Err(error(
                "recorded model execution identities contradict each other",
            ));
        }
        identities.push(Some(models.remove(0)))
    }
    let present = identities.iter().flatten().collect::<Vec<_>>();
    if present.is_empty() {
        return Ok(None);
    }
    if present.iter().any(|v| *v != present[0]) {
        return Err(error("recorded model identities differ across records"));
    }
    let m = present[0];
    if !run["artifact_digest"].is_string() {
        return Ok(None);
    }
    if m["artifact_digest"] != run["artifact_digest"] {
        return Err(error(
            "recorded model identity contradicts the attributed artifact",
        ));
    }
    for (key, legacy) in [("id", "model_id"), ("revision", "model_revision")] {
        for r in records {
            if let Some(v) = r["context"].get(legacy) {
                if v.as_str().is_none_or(|s| s.trim().is_empty()) {
                    return Err(error("recorded model identity has invalid shallow context"));
                }
                if *v != m[key] {
                    return Err(error("recorded model identity contradicts shallow context"));
                }
            }
        }
    }
    Ok(if present.len() == records.len() {
        Some(m.clone())
    } else {
        None
    })
}
type Messages<'a> = BTreeMap<&'a str, Vec<(&'a str, &'a str)>>;
fn messages(records: &[Value], http: bool) -> Option<Messages<'_>> {
    let mut out = Messages::new();
    for r in records {
        let list = if http {
            &r["context"]["http_observation"]["request"]["messages"]
        } else {
            &r["context"]["effective_messages"]
        };
        let list = list.as_array()?;
        if list.is_empty() {
            return None;
        }
        let mut parsed = Vec::new();
        for m in list {
            if m.as_object()?.len() != 2
                || t(m, "role").trim().is_empty()
                || !m["content"].is_string()
            {
                return None;
            }
            parsed.push((t(m, "role"), t(m, "content")))
        }
        let id = t(r, "id");
        if id.is_empty() || out.insert(id, parsed).is_some() {
            return None;
        }
    }
    Some(out)
}
fn paired(a: &Option<Messages<'_>>, b: &Option<Messages<'_>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => !a.is_empty() && !b.is_empty() && a.keys().eq(b.keys()),
        _ => false,
    }
}
fn roles(m: &Option<Messages<'_>>) -> String {
    let Some(m) = m.as_ref().filter(|m| !m.is_empty()) else {
        return "Unavailable in recorded context".into();
    };
    let first = m
        .first_key_value()
        .unwrap()
        .1
        .iter()
        .map(|v| &v.0)
        .collect::<Vec<_>>();
    if m.values()
        .any(|v| v.iter().map(|v| &v.0).collect::<Vec<_>>() != first)
    {
        return "Mixed across records".into();
    }
    let mut out = first
        .iter()
        .take(8)
        .map(|s| short(s, 32))
        .collect::<Vec<_>>()
        .join(" → ");
    if first.len() > 8 {
        out += " → … (truncated)"
    }
    out
}
fn prompt(a: &Option<Messages<'_>>, b: &Option<Messages<'_>>, http: bool) -> (String, Vec<Value>) {
    if !paired(a, b) {
        return("Prompt comparison unavailable in this report projection: no complete, uniquely paired supported message view is available.".into(),vec![]);
    }
    let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
    let desc = if http {
        "HTTP request messages"
    } else {
        "effective messages"
    };
    let changed = a.iter().filter(|(k, v)| b[*k] != **v).count();
    let count = comma(a.len() as u64);
    if changed == 0 {
        return (
            format!("Recorded {desc} are unchanged across all {count} paired cases."),
            vec![],
        );
    }
    let mut instruction = None;
    let mut uniform = true;
    for (k, before) in a {
        let after = &b[k];
        if after.len() != before.len() + 1 || after[0].0 != "system" || after[1..] != before[..] {
            uniform = false;
            break;
        }
        if instruction.is_some_and(|s| s != &after[0].1) {
            uniform = false;
            break;
        }
        instruction = Some(&after[0].1)
    }
    if uniform {
        let text = instruction.unwrap();
        let details = json!({"source":if http{"Recorded context.http_observation.request.messages; not hidden backend instructions or an effective internal prompt"}else{"Evaluator-recorded context.effective_messages; not execution attestation"},"paired_cases_checked":a.len(),"system_instruction":text.chars().take(4096).collect::<String>(),"characters":text.chars().count(),"truncated":text.chars().count()>4096,"sha256":invarlock_native_contracts::sha256_digest(text.as_bytes()).trim_start_matches("sha256:"),"note":"Full messages remain in the captured run records. User messages are not expanded here."});
        return(format!("The subject adds the same system instruction across all {count} paired cases; all other {desc} are unchanged."),vec![json!(["Recorded added system instruction",details])]);
    }
    (format!("Recorded {desc} differ in {} of {count} paired cases; the change is not one uniform added system instruction.",comma(changed as u64)),vec![])
}
pub(super) fn identities(inputs: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for (side, label) in [("baseline", "Baseline"), ("subject", "Subject")] {
        let r = &inputs[side];
        let service = r.get("service_identity");
        out.extend([
            json!([format!("{label} run"), r["run_id"]]),
            json!([format!("{label} run digest"), crate::digest(r)?]),
            json!([
                format!(
                    "{label} {}",
                    if service.is_some() {
                        "service identity"
                    } else {
                        "attributed artifact"
                    }
                ),
                if let Some(v) = service {
                    json!(crate::digest(v)?)
                } else {
                    r["artifact_digest"].clone()
                }
            ]),
            json!([
                format!("{label} evaluator"),
                format!("{} {}", t(&r["source"], "name"), t(&r["source"], "version"))
            ]),
        ]);
        if let Some(v) = service {
            out.extend([
                json!([
                    format!("{label} service configuration"),
                    v["configuration_digest"]
                ]),
                json!([
                    format!("{label} harness source"),
                    v["harness"]["source_digest"]
                ]),
            ])
        }
    }
    Ok(out)
}
type ContextProjection = (Vec<Value>, Vec<Value>, Vec<Value>, Vec<Value>);
pub(super) fn captured(inputs: &Value) -> Result<ContextProjection> {
    let mut subjects = Vec::new();
    let mut context = vec![json!(["Evaluation mode", "Captured evaluator outputs"])];
    let mut details = Vec::new();
    let mut keys = Vec::new();
    let mut models = Vec::new();
    let mut messages = [
        messages(a(&inputs["baseline"], "records"), false),
        messages(a(&inputs["subject"], "records"), false),
    ];
    let mut http = false;
    if !paired(&messages[0], &messages[1]) {
        let requests = [
            self::messages(a(&inputs["baseline"], "records"), true),
            self::messages(a(&inputs["subject"], "records"), true),
        ];
        if paired(&requests[0], &requests[1]) {
            messages = requests;
            http = true
        }
    }
    for (index, (side, label)) in [("baseline", "Baseline"), ("subject", "Subject")]
        .into_iter()
        .enumerate()
    {
        let r = &inputs[side];
        let records = a(r, "records");
        let fields = ["model_key", "model_id", "model_revision", "role"]
            .into_iter()
            .map(|k| (k, common(records, k, "context")))
            .collect::<BTreeMap<_, _>>();
        let service = r.get("service_identity").filter(|v| !v.is_null());
        let model = if service.is_none() {
            model(r, records)?
        } else {
            None
        };
        keys.push(fields["model_key"].0.clone());
        let mut identity = if fields["model_id"].0.is_some() {
            format!("Recorded model ID: {}", fields["model_id"].1)
        } else if let Some(m) = &model {
            format!("Recorded model ID: {}", short(t(m, "id"), 256))
        } else if fields["model_key"].0.is_some() {
            format!("Recorded model key: {}", fields["model_key"].1)
        } else {
            format!("Recorded run: {}", short(t(r, "run_id"), 128))
        };
        if let Some(sv) = service {
            let observed = t(sv, "observed_model");
            identity = format!(
                "{}: {} · Deployment: {}",
                if observed.is_empty() {
                    "Requested model"
                } else {
                    "Observed model"
                },
                short(
                    if observed.is_empty() {
                        t(sv, "requested_model")
                    } else {
                        observed
                    },
                    256
                ),
                short(t(sv, "deployment"), 256)
            );
            for k in [
                "provider",
                "service",
                "deployment",
                "requested_model",
                "observed_model",
                "exposed_revision",
            ] {
                let value = t(sv, k);
                context.push(json!([
                    format!("{label} {}", k.replace('_', " ")),
                    short(
                        if value.is_empty() {
                            "Not exposed"
                        } else {
                            value
                        },
                        256
                    )
                ]))
            }
            context.extend([
                json!([
                    format!("{label} observation window"),
                    format!(
                        "{} → {}",
                        t(&sv["observation_window"], "started_at"),
                        t(&sv["observation_window"], "ended_at")
                    )
                ]),
                json!([
                    format!("{label} service provenance"),
                    "Captured service declaration. Model weights are not identified."
                ]),
                json!([
                    format!("{label} service harness"),
                    short(
                        &format!(
                            "{} {}",
                            t(&sv["harness"], "name"),
                            t(&sv["harness"], "version")
                        ),
                        256
                    )
                ]),
            ]);
            let mut p = preview(&sv["configuration"]);
            p["note"]=json!("Full declared configuration remains in the captured run. Its digest identifies the declaration.");
            details.push(json!([
                format!("{label} declared service configuration"),
                p
            ]))
        }
        subjects.push(json!([label, identity]));
        let evaluator = if r["source"]["name"].is_string() && r["source"]["version"].is_string() {
            short(
                &format!("{} {}", t(&r["source"], "name"), t(&r["source"], "version")),
                256,
            )
        } else {
            "Unavailable in recorded run".into()
        };
        let entries: [(String, String); 7] = [
            ("evaluator".into(), evaluator),
            (
                "model revision".into(),
                model
                    .as_ref()
                    .map(|m| short(t(m, "revision"), 256))
                    .unwrap_or(fields["model_revision"].1.clone()),
            ),
            ("capture role".into(), fields["role"].1.clone()),
            ("workflow".into(), common(records, "workflow", "metadata").1),
            ("dataset".into(), summary(records, "dataset")),
            ("records".into(), comma(records.len() as u64)),
            (
                if http {
                    "HTTP request message roles"
                } else {
                    "effective message roles"
                }
                .into(),
                roles(&messages[index]),
            ),
        ];
        for (k, v) in entries {
            if let Some(sv) = service {
                if v == "Unavailable in recorded context"
                    || (k == "evaluator"
                        && r["source"]["name"] == sv["harness"]["name"]
                        && r["source"]["version"] == sv["harness"]["version"])
                {
                    continue;
                }
            }
            context.push(json!([format!("{label} {k}"), v]))
        }
        models.push(model)
    }
    let services = [
        inputs["baseline"]
            .get("service_identity")
            .filter(|v| !v.is_null()),
        inputs["subject"]
            .get("service_identity")
            .filter(|v| !v.is_null()),
    ];
    let change = if services.iter().any(Option::is_some) {
        context.push(json!([
            "Measurement scope",
            "Verification checks retained observations; it does not remeasure the service."
        ]));
        if let [Some(left), Some(right)] = services {
            let changed = [
                "provider",
                "service",
                "deployment",
                "requested_model",
                "observed_model",
                "exposed_revision",
                "configuration",
                "harness",
            ]
            .into_iter()
            .filter(|k| left[*k] != right[*k])
            .map(|k| k.replace('_', " "))
            .collect::<Vec<_>>();
            format!("{} This comparison does not identify the cause of an observed performance change or establish identical hidden weights.",if changed.is_empty(){"Declared service and configuration fields are unchanged across the recorded observation windows.".into()}else{format!("Declared service fields differ: {}.",changed.join(", "))})
        } else {
            "The runs use different identity profiles: a model artifact and a hosted service. The service declaration does not identify model weights.".into()
        }
    } else if let (Some(left), Some(right)) = (&models[0], &models[1]) {
        if left == right {
            format!("Both runs record model ID {} at the same revision and attributed artifact. Verification checks the retained identity bindings; the report does not reexecute the model.",short(t(left,"id"),256))
        } else {
            "The runs record different model IDs, revisions or attributed artifacts. Verification checks these retained identity bindings; the report does not reexecute either model.".into()
        }
    } else if keys[0].is_some() && keys[0] == keys[1] {
        format!("Both runs record model key {}. This does not establish identical model weights or revisions.",short(keys[0].as_ref().unwrap(),256))
    } else if keys.iter().all(Option::is_some) {
        "The runs record different model keys; these labels do not establish checkpoint identities."
            .into()
    } else {
        "Model-key comparison is unavailable because recorded values are missing or mixed.".into()
    };
    let (prompt, mut prompt_details) = prompt(&messages[0], &messages[1], http);
    prompt_details.extend(details);
    if http {
        context.push(json!(["Prompt observation scope","Recorded HTTP request messages; these do not establish hidden backend instructions or the service's effective internal prompt."]))
    }
    Ok((
        subjects,
        context,
        vec![json!(change), json!(prompt)],
        prompt_details,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_views_borrow_record_strings_and_keep_duplicate_rejection() {
        let records = vec![json!({"id":"one","context":{"effective_messages":[
            {"role":"user","content":"é😀 prompt"}
        ]}})];
        let view = messages(&records, false).unwrap();
        let (&id, pairs) = view.first_key_value().unwrap();
        assert!(std::ptr::eq(
            id.as_ptr(),
            records[0]["id"].as_str().unwrap().as_ptr()
        ));
        assert!(std::ptr::eq(
            pairs[0].1.as_ptr(),
            records[0]["context"]["effective_messages"][0]["content"]
                .as_str()
                .unwrap()
                .as_ptr()
        ));
        assert!(messages(&[records[0].clone(), records[0].clone()], false).is_none());
        assert!(messages(&[json!({"id":"one","context":{"effective_messages":[{"role":"user","content":"x","extra":true}]}})], false).is_none());
        assert!(messages(
            &[json!({"id":"one","context":{"effective_messages":[{"role":" ","content":"x"}]}})],
            false
        )
        .is_none());
        assert!(messages(
            &[json!({"id":"one","context":{"effective_messages":[]}})],
            false
        )
        .is_none());
    }

    #[test]
    fn message_projection_keeps_http_fallback_and_uniform_instruction_detection() {
        let baseline = json!({"run_id":"base","records":[{"id":"one","context":{
            "effective_messages":[{"role":"user","content":false}],
            "http_observation":{"request":{"messages":[{"role":"user","content":"q"}]}}
        }}]});
        let subject = json!({"run_id":"subject","records":[{"id":"one","context":{
            "http_observation":{"request":{"messages":[{"role":"system","content":"instruction"},{"role":"user","content":"q"}]}}
        }}]});
        let inputs = json!({"baseline":baseline,"subject":subject});
        let (_, context, conclusions, details) = captured(&inputs).unwrap();
        assert!(context
            .iter()
            .any(|row| row[0] == "Prompt observation scope"));
        assert_eq!(conclusions[1], "The subject adds the same system instruction across all 1 paired cases; all other HTTP request messages are unchanged.");
        assert_eq!(details[0][1]["system_instruction"], "instruction");
        assert_eq!(details[0][1]["paired_cases_checked"], 1);
    }
}
