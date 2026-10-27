//! Projection of recorded captured comparisons; no scoring is performed here.
use super::presentation::*;
use crate::{EvidenceError, Result};
use serde_json::{json, Value};
fn n(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn integer(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}
fn binary(m: &Value) -> bool {
    matches!(
        t(m, "kind"),
        "exact_match" | "normalized_match" | "numeric_tolerance" | "json_exact"
    )
}
fn match_counts(m: &Value) -> Option<(u64, u64)> {
    let count = integer(&m["count"]);
    if m["aggregation"] != "mean"
        || !a(m, "missing_ids").is_empty()
        || count == 0
        || count > 1 << 53
    {
        return None;
    }
    let mut out = Vec::new();
    for key in ["baseline_mean", "subject_mean"] {
        let mean = m[key]
            .as_f64()
            .filter(|m| m.is_finite() && (0.0..=1.0).contains(m))?;
        let matches = (mean * count as f64).round_ties_even() as u64;
        if matches as f64 / count as f64 != mean {
            return None;
        }
        out.push(matches)
    }
    Some((out[0], out[1]))
}
fn opening(m: &Value) -> String {
    let interval = &m["interval"];
    let fallback = if t(m, "explanation").is_empty() {
        "No explanation was supplied for this recorded result."
    } else {
        t(m, "explanation")
    };
    if m["decision"] == "pass" {
        return if interval.is_object()&&!interval["threshold"].is_null(){"The uncertainty interval is within the required range, and the other configured checks passed."}else{fallback}.into();
    }
    let failures = a(m, "checks")
        .iter()
        .filter(|c| c["passed"] == false)
        .collect::<Vec<_>>();
    let bound = failures
        .iter()
        .find(|c| matches!(t(c, "name"), "Allowed change" | "Maximum NLL ratio"));
    let explanation = |name: &str| match name {
        "Complete paired results" => {
            "Some included cases are missing a baseline or subject result."
        }
        "Included pair count" => "There are fewer paired records than the policy requires.",
        "Interval width" => "The uncertainty interval is wider than the policy permits.",
        "Subject minimum" => "The subject's observed score is below the required minimum.",
        "Subject maximum" => "The subject's observed score is above the permitted maximum.",
        _ => "",
    };
    let mut out = Vec::<String>::new();
    if let Some(bound) = bound.filter(|_| {
        m["decision"] == "regression"
            && interval.is_object()
            && !interval["threshold"].is_null()
            && matches!(t(interval, "threshold_direction"), "minimum" | "maximum")
    }) {
        let label = if t(bound, "name") == "Maximum NLL ratio" {
            "NLL ratio"
        } else {
            "change"
        };
        let within = if interval["threshold_direction"] == "minimum" {
            n(&interval["estimate"]) >= n(&interval["threshold"])
        } else {
            n(&interval["estimate"]) <= n(&interval["threshold"])
        };
        out.push(if within{format!("The observed {label} was within the policy limit, but its uncertainty interval extended beyond it.")}else if label=="NLL ratio"{"The observed NLL ratio exceeded the policy limit. Lower NLL is better.".into()}else{"The observed change was outside the range allowed by the policy.".into()});
    }
    out.extend(
        failures
            .iter()
            .map(|c| explanation(t(c, "name")))
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
    );
    if out.is_empty() {
        fallback.into()
    } else {
        out.join(" ")
    }
}
fn metric_views(comparison: &Value, policies: &Value) -> Result<Vec<Value>> {
    let mut views = Vec::new();
    for (index, m) in a(comparison, "metrics").iter().enumerate() {
        let policy = policies
            .as_array()
            .and_then(|p| p.iter().find(|p| p["name"] == m["name"]));
        let p = policy.unwrap_or(&Value::Null);
        let likelihood = m["kind"] == "normalized_nll_per_utf8_byte";
        let is_binary = binary(m);
        let change = &m[if likelihood { "ratio" } else { "delta" }];
        let mut display = vec![&m["baseline_mean"], &m["subject_mean"], &m["delta"]];
        if m["interval"].is_object() {
            display.extend([&m["interval"]["lower"], &m["interval"]["upper"]])
        }
        if policy.is_some() {
            display.extend([
                &p["maximum_regression"],
                &p["maximum_interval_width"],
                &p["subject_minimum"],
                &p["subject_maximum"],
            ])
        }
        if display.iter().any(|v| !v.is_null() && v.as_f64().is_none()) {
            return Err(EvidenceError::contract("int too large to convert to float"));
        }
        let percentage = is_binary
            && display
                .iter()
                .filter(|v| !v.is_null())
                .all(|v| (n(v) * 100.0).is_finite());
        let scale = if percentage { 100.0 } else { 1.0 };
        let unit = if likelihood {
            "ratio"
        } else if percentage {
            "pp"
        } else {
            t(m, "unit")
        };
        let display_unit = if likelihood {
            "nats / byte"
        } else {
            t(m, "unit")
        };
        let suffix = if percentage {
            "%".to_owned()
        } else {
            format!(" {display_unit}")
        };
        let count = integer(&m["count"]);
        let missing = a(m, "missing_ids").len() as u64;
        let complete = count - missing;
        let mut checks = vec![check(
            "Complete paired results",
            format!("{} of {}", comma(complete), comma(count)),
            "All included pairs".into(),
            json!(missing == 0),
            "Missing results are retained; they cannot be dropped to obtain a pass.",
        )];
        if policy.is_some() {
            checks.push(check(
                "Included pair count",
                comma(count),
                format!(">= {}", comma(integer(&p["minimum_count"]))),
                json!(count as f64 >= n(&p["minimum_count"])),
                "",
            ))
        } else {
            checks.push(check(
                "Policy thresholds",
                "Not supplied to renderer".into(),
                "Bound policy to explain requirements".into(),
                Value::Null,
                "",
            ))
        }
        let interval = &m["interval"];
        let mut visual = Value::Null;
        if interval.is_object() {
            let higher = m["direction"] == "higher";
            let threshold = policy.map(|p| {
                if likelihood {
                    n(&p["ratio_max"])
                } else if higher {
                    -n(&p["maximum_regression"])
                } else {
                    n(&p["maximum_regression"])
                }
            });
            let confidence = number(n(&interval["mass"]) * 100.0, false);
            let label = if likelihood {
                format!("{confidence}% paired-schedule ratio resampling interval")
            } else if is_binary {
                format!("Paired {confidence}% confidence interval")
            } else {
                format!("{confidence}% paired-schedule resampling interval")
            };
            let method = t(interval, "method");
            let basis = if [
                "newcombe_hybrid_score_paired_v1",
                "newcombe_hybrid_score_paired_v2",
            ]
            .contains(&method)
            {
                json!([format!("Baseline and subject are paired on the same {} cases in this scope. Newcombe hybrid score uses their paired match outcomes to form a nominal {confidence}% confidence interval for the accuracy change.",comma(complete)),if percentage{"Change is subject accuracy minus baseline accuracy in percentage points, not relative percent change. The confidence level describes the interval method, not the share of correct answers."}else{"Change is subject score minus baseline score in the displayed score units, not relative percent change. The confidence level describes the interval method, not the share of correct answers."},"Each metric and scope has its own interval; these intervals do not provide a simultaneous confidence guarantee across all results or establish a representative production sample."])
            } else if [
                "paired_mean_shake256_percentile_v1",
                "paired_percentile_bootstrap_sha256_v1",
            ]
            .contains(&method)
            {
                json!([format!("The declared percentile method resamples the same {} paired records in this scope with replacement, keeping each baseline and subject result together. Its retained resampling count is {}.",comma(complete),comma(integer(&interval["replicates"]))),if likelihood{format!("The central {confidence}% of that distribution describes the subject-to-baseline mean NLL ratio. Each side's NLL uses nats per expected UTF-8 byte.")}else{format!("The central {confidence}% of that distribution describes subject-minus-baseline mean score change in {display_unit} units.")},"This is a fixed-schedule resampling interval, not a population confidence interval. Separate metric and scope intervals do not give a simultaneous confidence guarantee or establish a representative production sample."])
            } else {
                json!([format!("Retained interval method: {method}."),"The report supplies these endpoints; a calculation explanation is unavailable for this method."])
            };
            visual = json!({"lower":n(&interval["lower"])*scale,"upper":n(&interval["upper"])*scale,"estimate":n(change)*scale,"threshold":threshold.map(|v|v*scale),"label":label,"unit":unit,"threshold_direction":if policy.is_some(){json!(if higher{"minimum"}else{"maximum"})}else{Value::Null},"neutral":if likelihood{1.0}else{0.0},"method":match method{"newcombe_hybrid_score_paired_v1"|"newcombe_hybrid_score_paired_v2"=>"Newcombe hybrid score","paired_mean_shake256_percentile_v1"|"paired_percentile_bootstrap_sha256_v1"=>"Paired percentile resampling",_=>""},"basis":basis});
            if let Some(threshold) = threshold {
                let bound = n(&interval[if higher { "lower" } else { "upper" }]);
                checks.push(check(
                    if likelihood {
                        "Maximum NLL ratio"
                    } else {
                        "Allowed change"
                    },
                    format!("{} {unit}", number(bound * scale, false)),
                    format!(
                        "{} {} {unit}",
                        if higher { ">=" } else { "<=" },
                        number(threshold * scale, false)
                    ),
                    json!(if higher {
                        bound >= threshold
                    } else {
                        bound <= threshold
                    }),
                    "Uses the interval bound, not just the observed change.",
                ));
                let width = n(&interval["upper"]) - n(&interval["lower"]);
                checks.push(check(
                    "Interval width",
                    format!("{} {unit}", number(width * scale, false)),
                    format!(
                        "<= {} {unit}",
                        number(n(&p["maximum_interval_width"]) * scale, false)
                    ),
                    json!(width <= n(&p["maximum_interval_width"])),
                    "",
                ));
                for (key, label, minimum) in [
                    ("subject_minimum", "Subject minimum", true),
                    ("subject_maximum", "Subject maximum", false),
                ] {
                    if p.get(key).is_some() {
                        let value = n(&m["subject_mean"]);
                        let threshold = n(&p[key]);
                        checks.push(check(
                            label,
                            format!("{}{suffix}", number(value * scale, false)),
                            format!(
                                "{} {}{suffix}",
                                if minimum { ">=" } else { "<=" },
                                number(threshold * scale, false)
                            ),
                            json!(if minimum {
                                value >= threshold
                            } else {
                                value <= threshold
                            }),
                            "",
                        ))
                    }
                }
            }
        } else if policy.is_some() {
            checks.push(check(
                "Score and uncertainty",
                "Not evaluated".into(),
                "Complete results and sufficient count".into(),
                Value::Null,
                "",
            ))
        }
        if m["decision"] == "pass" && checks.iter().any(|c| c["passed"] == false) {
            return Err(EvidenceError::contract(
                "captured recorded pass contradicts a displayed policy requirement",
            ));
        }
        let explanation = if policy.is_none() {
            "The policy thresholds are unavailable in this report; the retained decision has not been independently replayed."
        } else {
            "The retained comparison does not include an explanation for this result."
        };
        let mut notes=vec![json!(if m["direction"]=="higher"{"Higher values are better."}else{"Lower values are better."}),json!(if likelihood{"Recorded scoring basis: captured reference-continuation likelihood facts; NLL means, ratio and interval were recomputed when the comparison was created. Model execution and the supplied likelihood facts are not independently established."}else if m["scoring_assurance"]=="recomputed"{"Recorded scoring basis: expected and output values, scored when the comparison was created."}else{"Recorded scoring basis: external measurements or judgments, aggregated when the comparison was created."}),json!(format!("{} usable pairs; {} missing results; {} included pairs. Counts in overlapping slices must not be added together.",comma(complete),comma(missing),comma(count)))];
        let counts = if is_binary { match_counts(m) } else { None };
        if let Some((b, s)) = counts {
            notes.push(json!(format!(
                "Matches: baseline {} of {}; subject {} of {}.",
                comma(b),
                comma(count),
                comma(s),
                comma(count)
            )))
        }
        if is_binary && policy.is_some() && p.get("subject_minimum").is_none() {
            notes.push(json!(
                "No absolute minimum score is required by this policy."
            ))
        }
        if policy.is_none() {
            notes.push(json!("Requirements are unavailable in this comparison-only view. The original decision is displayed without independent replay."))
        }
        let mut metric = json!({"name":m["name"],"scope":m["slice"],"decision":m["decision"],"baseline":if m["baseline_mean"].is_null(){"Unavailable".into()}else{format!("{}{suffix}",number(n(&m["baseline_mean"])*scale,false))},"candidate":if m["subject_mean"].is_null(){"Unavailable".into()}else{format!("{}{suffix}",number(n(&m["subject_mean"])*scale,false))},"change":if change.is_null(){"Unavailable".into()}else{format!("{} {unit}",number(n(change)*scale,!likelihood))},"count":comma(complete),"count_label":"Usable pairs","count_detail":format!("{} missing · {} included{}",comma(missing),comma(count),if policy.is_some(){format!(" · ≥ {} required",comma(integer(&p["minimum_count"])))}else{String::new()}),"baseline_detail":counts.map(|(b,_)|format!("{} of {} matched",comma(b),comma(count))).unwrap_or_default(),"candidate_detail":counts.map(|(_,s)|format!("{} of {} matched",comma(s),comma(count))).unwrap_or_default(),"explanation":explanation,"checks":checks,"interval":visual,"notes":notes,"evidence_source":format!("reports/evaluation.report.json#/metrics/{index}")});
        metric["explanation"] = json!(opening(&metric));
        views.push(metric)
    }
    views.sort_by_key(|m| match t(m, "decision") {
        "regression" => 0,
        "insufficient_evidence" => 1,
        _ => 2,
    });
    Ok(views)
}
fn summary(metrics: &[Value], comparison: &Value) -> String {
    if metrics.len() != 1 {
        let failed = metrics
            .iter()
            .filter(|m| m["decision"] == "regression")
            .count();
        let insufficient = metrics
            .iter()
            .filter(|m| m["decision"] == "insufficient_evidence")
            .count();
        let passed = metrics.len() - failed - insufficient;
        let overview=format!("{} metric / scope results: {} passed, {} did not meet policy, and {} {} more evidence. Each result applies to its recorded scope; overlapping slice counts must not be added together.",comma(metrics.len()as u64),comma(passed as u64),comma(failed as u64),comma(insufficient as u64),if insufficient==1{"needs"}else{"need"});
        if let Some(focus) = metrics
            .iter()
            .find(|m| m["decision"] == "regression")
            .or_else(|| {
                metrics
                    .iter()
                    .find(|m| m["decision"] == "insufficient_evidence")
            })
        {
            return format!(
                "{} ({}): {} {overview}",
                display_label(t(focus, "name")),
                display_label(t(focus, "scope")),
                opening(focus)
            );
        }
        return overview;
    }
    let m = &metrics[0];
    let recorded = a(comparison, "metrics")
        .iter()
        .find(|item| item["name"] == m["name"] && item["slice"] == m["scope"])
        .unwrap();
    let likelihood = recorded["kind"] == "normalized_nll_per_utf8_byte";
    let mut parts = vec![opening(m)];
    let scope = if m["scope"] == "overall" {
        String::new()
    } else {
        format!(" within the {} slice", display_label(t(m, "scope")))
    };
    let pairs = if m["count"] == "1" { "pair" } else { "pairs" };
    if m["baseline"] != "Unavailable" && m["candidate"] != "Unavailable" {
        let mut sentence=format!("Across {} usable {pairs}{scope}, the subject's {} was {}, compared with {} for the baseline",t(m,"count"),display_label(t(m,"name")),t(m,"candidate"),t(m,"baseline"));
        if m["change"] != "Unavailable" {
            sentence += &if likelihood {
                format!(
                    ", giving a subject-to-baseline ratio of {}",
                    number(n(&recorded["ratio"]), false)
                )
            } else {
                format!(", a change of {}", t(m, "change"))
            }
        }
        parts.push(sentence + ".")
    } else {
        let unavailable = if m["baseline"] == "Unavailable" && m["candidate"] == "Unavailable" {
            "Baseline and subject scores"
        } else if m["baseline"] == "Unavailable" {
            "The baseline score"
        } else {
            "The subject score"
        };
        parts.push(format!(
            "{unavailable} for {} {} unavailable across {} usable {pairs}{scope}.",
            display_label(t(m, "name")),
            if unavailable == "Baseline and subject scores" {
                "are"
            } else {
                "is"
            },
            t(m, "count")
        ))
    }
    let missing = a(recorded, "missing_ids").len();
    if missing > 0 {
        let count = integer(&recorded["count"]);
        parts.push(format!(
            "Of {} included {}, {} {}",
            comma(count),
            if count == 1 { "pair" } else { "pairs" },
            comma(missing as u64),
            if missing == 1 {
                "has a missing result."
            } else {
                "have missing results."
            }
        ))
    }
    let i = &m["interval"];
    if i.is_object() {
        parts.push(format!(
            "The {}% interval for the {} runs from {} to {}{}",
            number(n(&recorded["interval"]["mass"]) * 100.0, false),
            if likelihood { "ratio" } else { "change" },
            number(n(&i["lower"]), false),
            number(n(&i["upper"]), false),
            if likelihood {
                ".".into()
            } else {
                format!(" {}.", t(i, "unit"))
            }
        ))
    } else if m["change"] == "Unavailable" {
        parts.push(format!(
            "No {} estimate or uncertainty interval is available.",
            if likelihood { "ratio" } else { "change" }
        ))
    } else {
        parts.push("No uncertainty interval is available.".into())
    }
    for check in a(m, "checks") {
        let name = t(check, "name");
        let (relation, value) = t(check, "required").split_once(' ').unwrap_or(("", ""));
        if name == "Included pair count" && check["passed"] == false {
            parts.push(format!(
                "The policy requires at least {value} included pairs."
            ))
        } else if ["Allowed change", "Maximum NLL ratio"].contains(&name) {
            parts.push(format!(
                "The policy requires the interval to stay {} {}.",
                if relation == ">=" {
                    "at or above"
                } else {
                    "at or below"
                },
                if name == "Maximum NLL ratio" {
                    value.trim_end_matches(" ratio")
                } else {
                    value
                }
            ))
        } else if ["Subject minimum", "Subject maximum"].contains(&name) {
            parts.push(format!(
                "The subject score must be {} {value}.",
                if relation == ">=" {
                    "at least"
                } else {
                    "at most"
                }
            ))
        }
    }
    parts.extend(
        a(m, "notes")
            .iter()
            .filter(|n| **n == "No absolute minimum score is required by this policy.")
            .map(|s| self::s(s).into()),
    );
    parts.join(" ")
}
pub fn build_comparison_view(comparison: &Value) -> Result<Value> {
    invarlock_native_contracts::validate_schema(comparison, "comparison")?;
    assemble(comparison, None, None, None)
}
pub(crate) fn assemble(
    comparison: &Value,
    inputs: Option<&Value>,
    manifest: Option<&Value>,
    signer: Option<&str>,
) -> Result<Value> {
    let metrics = metric_views(
        comparison,
        inputs
            .map(|v| &v["policy"]["metrics"])
            .unwrap_or(&Value::Null),
    )?;
    let mut identity = Vec::new();
    if let (Some(inputs), Some(m)) = (inputs, manifest) {
        identity.extend([
            json!(["Manifest", crate::digest(m)?]),
            json!(["Comparison", m["comparison_id"]]),
            json!(["Evidence signer", signer.unwrap_or("None")]),
        ]);
        identity.extend(super::context::identities(inputs)?);
    }
    if let Some(bindings) = comparison["bindings"].as_object() {
        identity.extend(
            bindings
                .iter()
                .map(|(k, v)| json!([format!("{} binding", capitalize(k)), v])),
        )
    }
    let mut technical = comparison.clone();
    for m in technical["metrics"].as_array_mut().unwrap() {
        let missing = a(m, "missing_ids").to_vec();
        m.as_object_mut().unwrap().remove("missing_ids");
        m["missing_count"] = json!(missing.len());
        m["missing_ids_preview"] = json!(missing.iter().take(20).collect::<Vec<_>>());
        m["missing_ids_preview_is_complete"] = json!(missing.len() <= 20)
    }
    let signing=match manifest.map(|m|t(m,"authentication")){Some("unsigned_local")=>"Unsigned local evidence; no signer authentication.",Some(_)=>"Signed manifest verified. This rendering has not authenticated it against a recipient-owned key.",None=>"Signing state unavailable: this view was created from comparison data only."};
    let mut assurance = Vec::new();
    if manifest.is_some() {
        assurance.push(json!(["Pack format", "invarlock/evidence-pack-v2"]))
    }
    assurance.extend([
        json!([
            "Input checks",
            if inputs.is_some() {
                "Evidence structure and comparison bindings checked."
            } else {
                "Comparison structure checked; evidence inputs were not supplied."
            }
        ]),
        json!(["Signing", signing]),
        json!([
            "Replay and scoring",
            "Scoring and replay were not performed by report."
        ]),
        json!([
            "Independent recipient verification",
            "Not performed by report. Use invarlock verify with independently supplied inputs."
        ]),
    ]);
    let mut view = empty_view();
    view["title"] = json!("InvarLock captured comparison");
    view["family"] = json!("Captured evaluation evidence");
    view["decision"] = comparison["decision"].clone();
    view["summary"] = json!(summary(&metrics, comparison));
    view["metrics"] = json!(metrics);
    view["assurance"] = json!(assurance);
    view["identity"] = json!(identity);
    view["next_steps"]=json!(["Review the recorded checks against the approved requirements.","For a signed handoff, run invarlock verify with independent signer, policy, request and complete-run anchors and an external signed receipt.","Use the comparison JSON and JUnit outputs in CI; preserve missing results and the original policy."]);
    view["limitations"] = comparison["limitations"].clone();
    if let Some(inputs) = inputs {
        let (subjects, context, changes, mut details) = super::context::captured(inputs)?;
        details.push(json!([
            "Bound policy (configuration preview)",
            super::context::policy_preview(&inputs["policy"])
        ]));
        view["subjects"] = json!(subjects);
        view["context"] = json!(context);
        view["changes"] = json!(changes);
        view["details"] = json!(details);
        view["limitations"].as_array_mut().unwrap().push(json!("Model and prompt context is evaluator-recorded provenance, not independent execution or model-identity attestation."));
    }
    view["technical"] = technical;
    Ok(view)
}

/// Authenticate one immutable captured snapshot, then project its stored results.
pub fn load(path: &std::path::Path) -> Result<Value> {
    let loaded = crate::evidence_sets::load_captured(path)?;
    let payloads = Value::Object(loaded.payloads.into_values());
    let comparison = &payloads["report"];
    let policy = &payloads["policy"];
    let policies = a(policy, "metrics")
        .iter()
        .map(|m| (t(m, "name"), m))
        .collect::<std::collections::BTreeMap<_, _>>();
    let scopes = std::iter::once("overall")
        .chain(a(policy, "slices").iter().map(|s| t(s, "name")))
        .collect::<Vec<_>>();
    let expected = policies
        .keys()
        .flat_map(|name| scopes.iter().map(move |scope| (*name, *scope)))
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    for item in a(comparison, "metrics") {
        if !item.is_object() {
            return Err(EvidenceError::contract(
                "captured report metrics are invalid",
            ));
        }
        if !item["name"].is_string()
            || !["pass", "regression", "insufficient_evidence"].contains(&t(item, "decision"))
        {
            return Err(EvidenceError::contract(
                "captured report metric decision is invalid",
            ));
        }
        let key = (t(item, "name"), t(item, "slice"));
        if !expected.contains(&key) {
            return Err(EvidenceError::contract(
                "captured report metric scope is not in policy",
            ));
        }
        if !seen.insert(key) {
            return Err(EvidenceError::contract(
                "captured report metric scope is duplicated",
            ));
        }
        let p = policies[key.0];
        if ["kind", "direction", "unit", "aggregation"]
            .iter()
            .any(|k| item[k] != p[k])
            || item["scoring_assurance"]
                != if p["kind"] == "recorded" {
                    "recorded"
                } else {
                    "recomputed"
                }
        {
            return Err(EvidenceError::contract(
                "captured report metric contradicts policy",
            ));
        }
        if !item["missing_ids"].is_array() {
            return Err(EvidenceError::contract(
                "captured report missing result inventory is invalid",
            ));
        }
    }
    if seen.is_empty() {
        return Err(EvidenceError::contract(
            "captured report contains no metrics",
        ));
    }
    if seen != expected {
        return Err(EvidenceError::contract(
            "captured report omits configured metric scopes",
        ));
    }
    let view = assemble(
        comparison,
        Some(&payloads),
        Some(&loaded.manifest),
        Some(&loaded.signer),
    )?;
    // Move the authenticated payloads: json! serializes borrowed expressions,
    // which would duplicate both complete runs before dropping the originals.
    let mut snapshot = serde_json::Map::new();
    snapshot.insert("view".into(), view);
    snapshot.insert("manifest".into(), loaded.manifest);
    snapshot.insert("payloads".into(), payloads);
    snapshot.insert("signer".into(), Value::String(loaded.signer));
    Ok(Value::Object(snapshot))
}
