//! Reports from one replayed judge snapshot; recipient acceptance is separate.
use super::context;
use super::presentation::*;
use crate::{EvidenceError, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn n(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}
fn count(v: &Value) -> String {
    let raw = v.to_string();
    let chars = raw.chars().collect::<Vec<_>>();
    chars
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "{}{}",
                if i > 0 && (chars.len() - i).is_multiple_of(3) {
                    ","
                } else {
                    ""
                },
                c
            )
        })
        .collect()
}
fn label(v: &Value, singular: &str) -> String {
    format!("{v} {singular}{}", if v == 1 { "" } else { "s" })
}
fn excerpt(v: &Value) -> String {
    v.as_str().unwrap_or("").chars().take(2000).collect()
}
fn observed(v: &Value) -> String {
    if v.is_null() {
        "Unavailable; incomplete planned schedule".into()
    } else {
        format!(
            "mean {}; lower {}; upper {}",
            t(v, "mean"),
            t(v, "lower"),
            t(v, "upper")
        )
    }
}
fn reason(s: &str) -> String {
    match s {
        "incomplete_planned_schedule" => "Some planned ratings did not complete.".into(),
        "minimum_units_not_met" => {
            "There are fewer independent units than the policy requires.".into()
        }
        "maximum_interval_width_exceeded" => {
            "The interval is wider than the policy permits.".into()
        }
        "interval_crosses_decision_threshold" => {
            "The interval includes outcomes that meet the bound and outcomes that do not.".into()
        }
        _ => s.replace('_', " "),
    }
}
fn explanation(analysis: &Value, p: &Value) -> String {
    let higher = p["direction"] == "higher";
    let allowance = t(p, "allowed_degradation");
    let bound = t(p, "subject_bound");
    match t(analysis,"decision"){
"pass"=>{let mut out=format!("The interval for the score change stays within the allowed loss of {allowance} score points.");if !p["subject_bound"].is_null(){out+=&format!(" The subject's score interval also stays {} the required {} of {bound}.",if higher{"at or above"}else{"at or below"},if higher{"minimum"}else{"maximum"})}out},
"regression"=>a(analysis,"gates").iter().filter(|g|g["decision"]=="regression").map(|g|if g["name"]=="paired_effect"{format!("The interval for the score change puts the loss beyond the allowance of {allowance} score points.")}else{format!("The subject's score interval lies entirely {} the required {} of {bound}.",if higher{"below"}else{"above"},if higher{"minimum"}else{"maximum"})}).collect::<Vec<_>>().join(" "),
_=>{let c=&analysis["counts"];if c["incomplete_trials"]!=0{return format!("Only {} of {} planned ratings completed, so the paired comparison could not be calculated.",count(&c["completed_trials"]),count(&c["expected_trials"]))}let mut out=Vec::new();if a(analysis,"reasons").contains(&json!("minimum_units_not_met")){out.push(format!("The complete schedule contains {} independent units; the policy requires at least {}.",count(&c["complete_units"]),count(&p["minimum_units"])))}for g in a(analysis,"gates").iter().filter(|g|g["decision"]=="insufficient_evidence"){if a(g,"reasons").contains(&json!("maximum_interval_width_exceeded")){out.push(format!("The {} is wider than the permitted {} score points.",if g["name"]=="paired_effect"{"interval for the score change"}else{"subject's score interval"},t(p,"maximum_interval_width")))}
if a(g,"reasons").contains(&json!("interval_crosses_decision_threshold")){out.push(if g["name"]=="paired_effect"{"The interval for the score change includes losses both within and beyond the policy allowance.".into()}else{format!("The subject's score interval includes values that meet the required {} of {bound} and values that do not.",if higher{"minimum"}else{"maximum"})})}}out.join(" ")}}
}
fn checks(analysis: &Value, p: &Value, numbers: &Value) -> Vec<Value> {
    let c = &analysis["counts"];
    let role = t(p, "decision_role");
    let complete = c["incomplete_trials"] == 0;
    let enough =
        invarlock_native_contracts::runtime::json_equal(&c["complete_units"], &p["minimum_units"])
            || c["complete_units"]
                .as_u64()
                .zip(p["minimum_units"].as_u64())
                .is_some_and(|(a, b)| a >= b);
    let mut out = vec![
        check(
            "Schedule completeness",
            format!(
                "{}/{} completed trials",
                c["completed_trials"], c["expected_trials"]
            ),
            format!(
                "{}/{} completed trials ({role})",
                c["expected_trials"], c["expected_trials"]
            ),
            if complete { json!(true) } else { Value::Null },
            "",
        ),
        check(
            "Independent units",
            format!(
                "{} complete / {} scheduled",
                c["complete_units"], c["scheduled_units"]
            ),
            format!(
                "at least {} complete independent units ({role})",
                p["minimum_units"]
            ),
            if complete && enough {
                json!(true)
            } else {
                Value::Null
            },
            "",
        ),
    ];
    for name in ["paired_effect", "subject_bound"] {
        if name == "subject_bound" && p["subject_bound"].is_null() {
            continue;
        }
        let (label, key) = if name == "paired_effect" {
            ("Paired score change", "effect_interval")
        } else {
            ("Subject score bound", "subject_interval")
        };
        let interval = &analysis[key];
        let gate = a(analysis, "gates").iter().find(|g| g["name"] == name);
        let decision = gate
            .map(|g| t(g, "decision"))
            .unwrap_or("insufficient_evidence");
        let width = &numbers["widths"][key];
        out.push(check(
            &format!("{label} interval width"),
            if width.is_null() {
                "Unavailable".into()
            } else {
                t(width, "width").into()
            },
            format!(
                "interval width <= {} ({role})",
                t(p, "maximum_interval_width")
            ),
            width["passed"].clone(),
            "",
        ));
        let mut threshold = t(
            p,
            if name == "paired_effect" {
                "allowed_degradation"
            } else {
                "subject_bound"
            },
        )
        .to_owned();
        if name == "paired_effect" && p["direction"] == "higher" {
            threshold = format!("-{threshold}")
        }
        let requirement = if p["direction"] == "higher" {
            format!("lower >= {threshold}; adverse if upper < {threshold}")
        } else {
            format!("upper <= {threshold}; adverse if lower > {threshold}")
        };
        out.push(check(
            label,
            format!(
                "{}; {}",
                observed(interval),
                match decision {
                    "pass" => "satisfied",
                    "regression" => "adverse",
                    _ => "inconclusive",
                }
            ),
            format!("{requirement} ({role})"),
            match decision {
                "pass" => json!(true),
                "regression" => json!(false),
                _ => Value::Null,
            },
            &a(gate.unwrap_or(analysis), "reasons")
                .iter()
                .map(|v| reason(s(v)))
                .collect::<Vec<_>>()
                .join(" "),
        ))
    }
    out
}
fn record_context(records: &[Value]) -> Result<Value> {
    let mut settings = BTreeMap::new();
    let mut settings_count = 0;
    let mut projections = BTreeMap::new();
    let mut projection_count = 0;
    for r in records {
        let c = &r["context"];
        if let Some(v) = c.get("settings") {
            settings_count += 1;
            settings.entry(crate::digest(v)?).or_insert(v);
        }
        if c["input_projection"].is_object() {
            projection_count += 1;
            let p = &c["input_projection"];
            projections.insert(t(p,"configuration_digest"),json!({"configuration_digest":p["configuration_digest"],"kind":p["configuration"]["kind"],"pointer":context::short(t(&p["configuration"],"pointer"),256)}));
        }
    }
    let mut result = json!({});
    if settings_count > 0 {
        let common = settings.len() == 1 && settings_count == records.len();
        result["settings"] = json!({"status":if common{"common"}else{"mixed_or_incomplete"},"present_records":settings_count,"included_records":records.len()});
        if common {
            let (id, v) = settings.first_key_value().unwrap();
            result["settings"]["digest"] = json!(id);
            result["settings"]["preview"] =
                json!(context::short(t(&context::preview(v), "text"), 2000));
        }
    }
    if projection_count > 0 {
        result["input_projection"] = json!({"present_records":projection_count,"included_records":records.len(),"configuration_count":projections.len(),"configurations":projections.values().take(8).collect::<Vec<_>>(),"note":"At most eight mapping identities are shown; original structured inputs remain in retained records."})
    }
    Ok(result)
}
pub fn load(path: &std::path::Path, case_ids: Option<&[String]>) -> Result<Value> {
    let publication = crate::judge::replay_snapshot(path, None)?;
    let (view, facts) = build(
        &publication,
        &publication["artifacts"],
        case_ids.unwrap_or(&[]),
    )?;
    Ok(
        json!({"view":view,"facts":facts,"envelope":publication["envelope"],"artifacts":publication["artifacts"]}),
    )
}
pub fn build(
    publication: &Value,
    artifacts: &Value,
    case_ids: &[String],
) -> Result<(Value, Value)> {
    let plan = &artifacts["plan"];
    let p = &artifacts["analysis_policy"];
    let analysis = &publication["analysis_result"];
    let effect = &analysis["effect_interval"];
    let subject = &analysis["subject_interval"];
    let c = &analysis["counts"];
    let required = p["decision_role"] == "required";
    let per_case = plan["prompt"]["reference_mode"] == "per_case";
    let numbers =
        invarlock_native_core::judge_report_numbers(plan, &artifacts["measurements"], p, analysis)
            .map_err(|e| err(e.to_string()))?;
    let baseline_mean = &numbers["baseline_mean"];
    let confidence = t(&numbers, "confidence");
    let mut explanation = explanation(analysis, p);
    if !required {
        explanation =
            format!("This metric is advisory and does not gate required decisions. {explanation}")
    }
    let checks = checks(analysis, p, &numbers);
    let interval = if effect.is_null() {
        Value::Null
    } else {
        json!({"lower":n(&effect["lower"]),"upper":n(&effect["upper"]),"estimate":n(&effect["mean"]),"threshold":if n(&p["allowed_degradation"])==0.0{0.0}else{n(&p["allowed_degradation"])*if p["direction"]=="higher"{-1.0}else{1.0}},"label":"Paired independent-unit effect interval","unit":"score","threshold_direction":if p["direction"]=="higher"{"minimum"}else{"maximum"},"neutral":0.0,"method":"Two-sided Hoeffding bound","basis":[format!("The complete schedule contains {} cases grouped into {} independent units, with {} ratings per side for each case. Repeated ratings are averaged within each case, then cases within each unit; units receive equal weight.",count(&c["scheduled_cases"]),count(&c["complete_units"]),count(&c["repetitions"])),"Two-sided Hoeffding bounds use the declared score range and the number of independent units. Repetitions and cases within a unit do not add independent units. The paired effect is subject minus baseline in normalized rubric score points, not accuracy percentage points.",format!("The declared family error budget is alpha {}, shared across {} interval claims. Each interval receives alpha / family size ({} / {}); family confidence is at least {confidence}% under the declared independence and bounded-score assumptions.",t(p,"alpha"),p["comparison_family_size"],t(p,"alpha"),p["comparison_family_size"]),"This concerns expected judge scores on the fixed benchmark. It does not establish representative production performance or the correctness of the judge's ratings."]})
    };
    let mut notes=vec![json!("Scores use the rubric mapped to 0 through 1; they are not percentages of correct answers. Changes and policy limits use the same score units."),json!(format!("{}; {}/{} completed {}.",label(&c["scheduled_units"],"independent unit"),c["completed_trials"],c["expected_trials"],if c["expected_trials"]==1{"trial"}else{"trials"})),json!(if required{"Decision role: required."}else{"Decision role: advisory; this metric does not gate required decisions."}),json!(format!("Allowed degradation: {} ({} is better).",t(p,"allowed_degradation"),t(p,"direction"))),json!(format!("Minimum units: {}; maximum interval width: {}.",p["minimum_units"],t(p,"maximum_interval_width"))),json!(format!("Two-sided Hoeffding intervals; family confidence at least {confidence}% (alpha {}; comparison family size {}); Bonferroni error allocation alpha / comparison family size per interval.",t(p,"alpha"),p["comparison_family_size"])),json!("Effect is subject minus baseline; tabulated interval endpoints and widths retain analysis precision."),json!("Inconclusive checks do not establish a bound: completeness, minimum units and precision must be met before an interval can establish satisfaction or an adverse result.")];
    if p["subject_bound"].is_null() {
        notes.push(json!(format!(
            "Subject interval (descriptive; no subject bound configured): {}.",
            observed(subject)
        )))
    }
    notes.extend([json!("Baseline and subject means describe the same complete schedule with equal independent-unit weights."),json!("Repetitions do not increase the number of independent units.")]);
    let metric = json!({"evidence_source":"analysis_result.json#","name":p["metric_name"],"scope":if required{"Fixed benchmark; equal independent-unit weights"}else{"Advisory metric; fixed benchmark; equal independent-unit weights"},"decision":analysis["decision"],"baseline":if baseline_mean.is_null(){"Unavailable".into()}else{format!("{} score",number(n(baseline_mean),false))},"candidate":if subject.is_null(){"Unavailable".into()}else{format!("{} score",number(n(&subject["mean"]),false))},"change":if effect.is_null(){"Unavailable".into()}else{format!("{} score",number(n(&effect["mean"]),true))},"count":count(&c["complete_units"]),"count_label":"Complete independent units","count_detail":format!("{} scheduled units; {} cases; {}/{} completed trials",count(&c["scheduled_units"]),count(&c["scheduled_cases"]),count(&c["completed_trials"]),count(&c["expected_trials"])),"baseline_detail":"Mean normalized rubric score","candidate_detail":"Mean normalized rubric score","explanation":explanation,"checks":checks,"interval":interval,"notes":notes});
    let baseline_rows = a(&artifacts["baseline_run"], "records")
        .iter()
        .map(|r| (t(r, "id"), r))
        .collect::<BTreeMap<_, _>>();
    let subject_rows = a(&artifacts["subject_run"], "records")
        .iter()
        .map(|r| (t(r, "id"), r))
        .collect::<BTreeMap<_, _>>();
    let available = a(&plan["sampling"], "case_units")
        .iter()
        .map(|r| t(r, "case_id").to_owned())
        .collect::<BTreeSet<_>>();
    if case_ids.len() > 50 {
        return Err(err("at most 50 --case-id values may be requested"));
    }
    let requested = case_ids.iter().cloned().collect::<BTreeSet<_>>();
    if requested.len() != case_ids.len() {
        return Err(err("--case-id values must be unique"));
    }
    let missing = requested.difference(&available).collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(err(format!(
            "requested case IDs are not in the judge plan: {}{}",
            missing
                .iter()
                .take(5)
                .map(|v| invarlock_native_contracts::python_repr(&json!(v)))
                .collect::<Vec<_>>()
                .join(", "),
            if missing.len() > 5 {
                format!(" and {} more", missing.len() - 5)
            } else {
                String::new()
            }
        )));
    }
    let selected = if case_ids.is_empty() {
        available.iter().take(50).cloned().collect::<Vec<_>>()
    } else {
        case_ids.to_vec()
    };
    let mut details = Vec::new();
    for id in &selected {
        let rows=a(&artifacts["measurements"],"trials").iter().filter(|t|t["case_id"]==*id).map(|r|json!({"side":r["side"],"repetition":r["repetition"],"status":r["status"],"parse":r["parse"],"attempts":a(r,"attempts").iter().map(|a|json!({"status":a["status"],"resolved_model":a["resolved_model"],"response_excerpt":if a["response"].is_null(){Value::Null}else{json!(excerpt(&a["response"]["text"]))},"error":a["error"]})).collect::<Vec<_>>()})).collect::<Vec<_>>();
        let mut detail = json!({"input_excerpt":excerpt(&baseline_rows[id.as_str()]["input"]),"baseline_answer_excerpt":excerpt(&baseline_rows[id.as_str()]["output"]),"subject_answer_excerpt":excerpt(&subject_rows[id.as_str()]["output"]),"trials":rows});
        if per_case {
            detail["reference_excerpt"] = json!(excerpt(&baseline_rows[id.as_str()]["expected"]))
        }
        details.push(json!([id, detail]))
    }
    let envelope = &publication["envelope"];
    let signed = !envelope["signature"].is_null();
    let signer = &envelope["signer"];
    let local = artifacts["measurements"]["source_profile"] == "retained-runtime-provider-judge-v1";
    let compatible =
        artifacts["measurements"]["source_profile"] == "retained-openai-compatible-judge-v1";
    let resolved = a(&artifacts["measurements"], "trials")
        .iter()
        .flat_map(|t| a(t, "attempts"))
        .filter_map(|a| a["resolved_model"].as_str())
        .collect::<BTreeSet<_>>();
    let mut prompt = json!({"system_excerpt":excerpt(&plan["prompt"]["system"]),"template_excerpt":excerpt(&plan["prompt"]["template"]),"demonstrations":a(&plan["prompt"],"demonstrations").len(),"references":a(&plan["prompt"],"references").iter().map(|r|json!({"id":r["id"],"sha256":r["sha256"]})).collect::<Vec<_>>()});
    if per_case {
        prompt["reference_mode"] = json!("per_case")
    }
    if local {
        prompt["runtime_format"] = plan["prompt"]
            .get("runtime_format")
            .cloned()
            .unwrap_or(json!("canonical-json-v1"))
    }
    let mut facts = json!({"assurance":{"authentication":if signed{"signature_present_not_recipient_authorized"}else{"unsigned"},"replay":"completed_without_recipient_authorization","policy_decision":analysis["decision"],"decision_role":p["decision_role"],"recipient_acceptance":"not_performed","signer_identity":signer["identity"],"signer_public_key_sha256":signer["public_key_sha256"]},"baseline":artifacts["baseline_run"]["run_id"],"subject":artifacts["subject_run"]["run_id"],"comparison":{},"judge":plan["judge"],"prompt":prompt,"resolved_models":resolved,"rubric":{"sha256":plan["rubric"]["sha256"],"excerpt":excerpt(&plan["rubric"]["text"])},"scale":plan["scale"],"analysis":analysis,"descriptive_means":{"baseline":baseline_mean,"subject":subject["mean"],"change":effect["mean"],"weighting":"equal_independent_units"},"detail_limits":{"shown_cases":selected.len(),"total_cases":available.len(),"text_excerpt_characters":2000,"selection":if case_ids.is_empty(){"first_by_case_id"}else{"requested"},"case_ids":selected}});
    for side in ["baseline", "subject"] {
        let r = &artifacts[format!("{side}_run")];
        facts["comparison"][side] = json!({"run_id":r["run_id"],"artifact_digest":r["artifact_digest"],"source":r["source"]})
    }
    if per_case {
        facts["per_case_references"]=json!(selected.iter().map(|id|json!({"case_id":id,"reference_excerpt":excerpt(&baseline_rows[id.as_str()]["expected"])})).collect::<Vec<_>>())
    }
    let mut context_values = Vec::new();
    let mut identity = Vec::new();
    let mut assurance = Vec::new();
    let subjects;
    let native = artifacts.get("native_capture");
    if let Some(native) = native {
        let request = &native["normalized_request"]["comparison"];
        let same_artifact = artifacts["baseline_run"]["artifact_digest"]
            == artifacts["subject_run"]["artifact_digest"];
        let same_settings = request["baseline"]["runtime"] == request["subject"]["runtime"];
        let intent = format!(
            "{}; {}",
            if same_artifact {
                "Same artifacts"
            } else {
                "Different artifacts"
            },
            if same_settings {
                "same runtime settings"
            } else {
                "different runtime settings"
            }
        );
        let observations = native["normalized_request"]
            .get("observations")
            .cloned()
            .unwrap_or(json!([]));
        facts["native_capture"] = json!({"sha256":envelope["bindings"]["native_capture_sha256"],"execution_mode":native["normalized_request"]["execution"]["mode"],"same_artifact":same_artifact,"same_runtime_settings":same_settings,"comparison_intent":intent,"observations":observations});
        context_values.push(json!(["Comparison inputs", intent]));
        for (side, label) in [("baseline", "Baseline"), ("subject", "Subject")] {
            let id = &request[side]["artifact"]["model_id"];
            facts[side] = id.clone();
            let runtime = &request[side]["runtime"];
            let run = &artifacts[format!("{side}_run")];
            let digest = &run["records"][0]["context"]["runtime_digest"];
            facts["comparison"][side]["model_id"] = id.clone();
            facts["comparison"][side]["runtime"] = runtime.clone();
            facts["comparison"][side]["runtime_digest"] = digest.clone();
            context_values.extend([
                json!([format!("{label} provider"), runtime["provider"]]),
                json!([
                    format!("{label} runtime settings"),
                    String::from_utf8(invarlock_native_contracts::canonical_json_no_newline(
                        &runtime["settings"]
                    )?)
                    .unwrap()
                ]),
            ]);
            identity.extend([
                json!([format!("{label} runtime"), digest]),
                json!([format!("{label} run"), run["run_id"]]),
            ]);
        }
        let obs = observations
            .as_array()
            .unwrap()
            .iter()
            .map(|v| format!("{} ({}, {})", t(v, "id"), t(v, "scope"), t(v, "kind")))
            .collect::<Vec<_>>()
            .join("; ");
        context_values.push(json!([
            "Retained observations",
            if obs.is_empty() { "None" } else { &obs }
        ]));
        assurance.push(json!(["Native runtime provenance","Replayed offline from retained provider reports, runtime manifests, configurations, receipts, artifact identities, and scoring observations."]));
        identity.push(json!(["Native capture", facts["native_capture"]["sha256"]]));
        subjects = vec![
            json!(["Baseline", facts["baseline"]]),
            json!(["Subject", facts["subject"]]),
        ];
    } else {
        let inputs =
            json!({"baseline":artifacts["baseline_run"],"subject":artifacts["subject_run"]});
        let (s, ctx, changes, context_details) = context::captured(&inputs)?;
        subjects = s;
        context_values.extend(ctx.clone());
        context_values.extend(changes.iter().map(|v| json!(["Captured comparison", v])));
        identity.extend(context::identities(&inputs)?);
        details.extend(context_details);
        let mut captured = json!({"context":ctx.iter().map(|v|(sstr(&v[0]).to_owned(),v[1].clone())).collect::<Map<_,_>>(),"changes":changes});
        for (side, label) in [("baseline", "Baseline"), ("subject", "Subject")] {
            let r = &inputs[side];
            facts["comparison"][side]["run_digest"] = json!(crate::digest(r)?);
            facts["comparison"][side]["source_digest"] = r["source_digest"].clone();
            if let Some(s) = r.get("service_identity") {
                let mut s = s.clone();
                s.as_object_mut().unwrap().remove("configuration");
                facts["comparison"][side]["service_identity"] = s
            }
            identity.push(json!([
                format!("{label} source digest"),
                r.get("source_digest")
                    .filter(|v| !v.is_null())
                    .cloned()
                    .unwrap_or(json!("Unavailable in captured run"))
            ]));
            let recorded = record_context(a(r, "records"))?;
            captured[side] = recorded.clone();
            for (k, v) in recorded.as_object().unwrap() {
                let heading = format!("{label} recorded {}", k.replace('_', " "));
                details.push(json!([heading, v]));
                if k == "settings" {
                    context_values.push(json!([
                        heading,
                        v.get("preview")
                            .cloned()
                            .unwrap_or(json!("Mixed or incomplete across records"))
                    ]));
                    if let Some(d) = v.get("digest") {
                        identity.push(json!([heading, d]))
                    }
                } else {
                    context_values.push(json!([
                        heading,
                        format!(
                            "{} of {} records; {} mapping configurations",
                            v["present_records"], v["included_records"], v["configuration_count"]
                        )
                    ]));
                    for p in a(v, "configurations") {
                        context_values
                            .push(json!([format!("{label} projection pointer"), p["pointer"]]));
                        identity.push(json!([
                            format!("{label} projection configuration"),
                            p["configuration_digest"]
                        ]));
                    }
                }
            }
        }
        facts["captured_context"] = captured;
        assurance.push(json!(["Captured answer provenance","Evaluator, artifact, configuration and model labels are source assertions retained with the captured runs; runtime execution has not been independently established."]));
    }
    if compatible {
        let source = invarlock_native_contracts::parse_json(
            t(&artifacts["measurements"]["sources"][0], "content").as_bytes(),
            "retained judge service",
        )?;
        let collection = &source["collection"];
        let service = match t(collection, "service") {
            "vllm" => "vLLM",
            "ollama" => "Ollama",
            "lm_studio" => "LM Studio",
            _ => "OpenAI-compatible service",
        };
        let response = collection
            .get("response_format")
            .cloned()
            .unwrap_or(json!("json_object"));
        context_values.extend([
            json!(["Judge service", service]),
            json!(["Judge endpoint", collection["base_url"]]),
            json!([
                "Judge response format",
                if response == "json_schema" {
                    "JSON schema"
                } else {
                    "JSON object"
                }
            ]),
        ]);
        identity.push(json!([
            "Judge endpoint digest",
            source["service_identity"]["endpoint_sha256"]
        ]));
        facts["judge_service"] = json!({"service":collection["service"],"base_url":collection["base_url"],"endpoint_sha256":source["service_identity"]["endpoint_sha256"],"model_identity_basis":"service_assertion","response_format":response});
    }
    if local {
        assurance.push(json!(["Judge execution evidence","Local artifact, runtime receipt, prompts and responses replayed offline. This does not rerun the judge or establish rating correctness."]));
        context_values.push(json!([
            "Judge prompt format",
            if plan["prompt"]["runtime_format"] == "chatml-v1" {
                "ChatML"
            } else {
                "Canonical JSON"
            }
        ]));
        identity.push(json!([
            "Judge artifact identity",
            plan["judge"]["model_identity"]["weights_sha256"]
        ]));
    }
    if compatible {
        assurance.push(json!(["Judge service evidence","The configured endpoint, model labels, requests and responses are retained. Model labels are service assertions; the judge's model files and execution environment are not authenticated. Offline verification does not call the service again."]));
    }
    if per_case {
        context_values.push(json!(["Per-case references","Enabled; each judge request includes its case reference separately from the model input."]));
    }
    let provider = match t(&plan["judge"], "provider") {
        "hf_transformers" => "Hugging Face Transformers",
        "llama_cpp" => "llama.cpp",
        "openai_compatible" => "OpenAI-compatible service",
        "openai" => "OpenAI",
        "anthropic" => "Anthropic",
        "google" => "Google",
        "openrouter" => "OpenRouter",
        s => s,
    };
    context_values.extend([
        json!(["Judge provider", provider]),
        json!(["Requested judge", plan["judge"]["requested_model"]]),
        json!([
            "Resolved judges",
            if resolved.is_empty() {
                "Unavailable".into()
            } else {
                resolved.iter().copied().collect::<Vec<_>>().join(", ")
            }
        ]),
        json!([
            "Reasoning effort",
            plan["judge"]["config"]
                .get("reasoning_effort")
                .filter(|v| !v.is_null() && v != &&json!(""))
                .cloned()
                .unwrap_or(json!("Not configured"))
        ]),
        json!(["Rubric", excerpt(&plan["rubric"]["text"])]),
        json!(["Judge prompt", excerpt(&plan["prompt"]["template"])]),
        json!([
            "Prompt demonstrations",
            a(&plan["prompt"], "demonstrations").len().to_string()
        ]),
        json!([
            "Prompt references",
            a(&plan["prompt"], "references").len().to_string()
        ]),
        json!([
            "Scale",
            a(&plan["scale"], "ratings")
                .iter()
                .map(|v| format!("{} = {}", t(v, "label"), t(v, "value")))
                .collect::<Vec<_>>()
                .join(", ")
        ]),
        json!(["Coverage", label(&c["scheduled_cases"], "case")]),
    ]);
    assurance.extend([
        json!([
            "Authentication",
            if signed {
                "Signature present; recipient signer authorization has not been performed."
            } else {
                "Unsigned evidence; no signer authentication."
            }
        ]),
        json!([
            "Measurement and analysis replay",
            "Completed offline against the retained plan and source records."
        ]),
        json!(["Policy result", decision(t(analysis, "decision"))]),
        json!(["Recipient acceptance", "Not performed by report."]),
    ]);
    for (side, label) in [("baseline", "Baseline"), ("subject", "Subject")] {
        let d = &artifacts[format!("{side}_run")]["artifact_digest"];
        if !d.is_null() {
            identity.push(json!([format!("{label} artifact"), d]))
        }
    }
    identity.extend([
        json!(["Plan", envelope["bindings"]["plan_sha256"]]),
        json!(["Intended subject", envelope["intended_subject"]]),
        json!([
            "Signer",
            if signer.is_null() {
                json!("Unsigned")
            } else {
                signer["identity"].clone()
            }
        ]),
        json!([
            "Signer key",
            if signer.is_null() {
                json!("Unavailable")
            } else {
                signer["public_key_sha256"].clone()
            }
        ]),
    ]);
    let mut next = Vec::new();
    if !signed {
        next.push(json!("For recipient verification, republish the same retained inputs to a new evidence destination with evaluate --signing-key and the declared signer identity. Do not modify this evidence bundle."));
    }
    if analysis["decision"] == "regression" {
        next.push(json!("Inspect the unmet bounds and retained answers with report --case-id before changing the model or prompt."))
    } else if analysis["decision"] == "insufficient_evidence" {
        next.push(json!("Inspect schedule completeness, independent-unit count, interval-width checks and unresolved decision bounds. Preserve this result; declare any new collection before measuring again."))
    }
    next.push(json!("Use verify with an independently maintained judge recipient policy before relying on this result."));
    let mut view = empty_view();
    view["title"] = json!(if required {
        "InvarLock bounded judge report"
    } else {
        "InvarLock advisory judge report"
    });
    view["family"] = json!(if required {
        "Bounded judge measurement evidence"
    } else {
        "Advisory judge measurement evidence"
    });
    view["decision"] = analysis["decision"].clone();
    view["summary"] = json!(format!(
        "{explanation}{}",
        if !effect.is_null() && !subject.is_null() && !baseline_mean.is_null() {
            format!(" Across {} complete independent units, the subject's mean rubric score was {}, compared with {} for the baseline, a change of {} score points.",count(&c["complete_units"]),number(n(&subject["mean"]),false),number(n(baseline_mean),false),number(n(&effect["mean"]),true))
        } else {
            " Mean scores and a paired effect are unavailable for the incomplete schedule.".into()
        }
    ));
    view["metrics"] = json!([metric]);
    view["assurance"] = json!(assurance);
    view["subjects"] = json!(subjects);
    view["context"] = json!(context_values);
    view["identity"] = json!(identity);
    view["next_steps"] = json!(next);
    view["limitations"]=json!([analysis["estimand"],if case_ids.is_empty(){format!("Details show the first {} cases by case ID and at most 2000 characters per text excerpt; use report --case-id to select any retained case.",available.len().min(50))}else{format!("Details show the {} requested cases and at most 2000 characters per text excerpt; complete data remains in evidence.",selected.len())},if native.is_some(){"Native runtime provenance binds the captured model answers. Judge measurements do not establish immunity to prompt injection."}else{"A retained judgment does not establish model execution or immunity to prompt injection."},"Verification replays retained measurements offline; it does not remeasure the evaluated service or independently establish that every judge rating is correct."]);
    view["details"] = json!(details);
    view["technical"] = json!({"method":analysis["method"],"assumptions":analysis["assumptions"],"policy":p,"effect_interval":effect,"subject_interval":subject});
    let key = if native.is_some() {
        "native_capture"
    } else {
        "captured_context"
    };
    view["technical"][key] = facts[key].clone();
    Ok((view, facts))
}
fn sstr(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
