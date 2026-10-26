use super::*;
use invarlock_native_core::fsum;
fn check(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(integrity(message))
    }
}
fn list<'a>(v: &'a Value, label: &str) -> Result<&'a Vec<Value>> {
    v.as_array()
        .ok_or_else(|| integrity(format!("{label} must be an array")))
}
fn digest_no_newline(v: &Value) -> Result<String> {
    Ok(hex(&canonical_json_no_newline(v)?))
}
pub(crate) fn typed_sidecar(raw: &[u8], label: &str, schema: &str) -> Result<Value> {
    use invarlock_native_contracts::{first_schema_error, runtime, SchemaErrorOrder};
    let value = object(raw, label)?;
    if let Some(error) = first_schema_error(&value, schema, SchemaErrorOrder::Instance)? {
        let path = if error.path.is_empty() {
            "<root>".into()
        } else {
            error.path.join(".")
        };
        return Err(integrity(format!(
            "{label} schema validation failed at {path}: {}",
            error.message
        )));
    }
    let checked = match schema {
        "model_artifact_identity" => runtime::validate_artifact(&value),
        "runtime_scoring_observation" => runtime::validate_observation(&value),
        "runtime_provider_receipt" => runtime::validate_receipt(&value),
        _ => unreachable!(),
    };
    checked.map_err(|error| {
        integrity(format!(
            "{label} could not be reconstructed: {}",
            error.message
        ))
    })?;
    Ok(value)
}
pub(crate) fn schedule(value: &Value, bytes: &[u8], request: &Value) -> Result<()> {
    let typed = invarlock_native_contracts::schedule::Schedule::build(value)
        .map_err(|e| integrity(e.message))?;
    check(
        typed.bytes() == bytes,
        "runtime behavioral schedule is not canonical JSON",
    )?;
    check(
        request["comparison"].is_object() && request["execution"].is_object(),
        "normalized request dataset binding is invalid",
    )?;
    check(
        request["comparison"]["task"] == value["task"],
        "request task does not match the canonical schedule",
    )?;
    if request["execution"]["mode"] == "import" {
        check(
            request["comparison"]["dataset"] == "schedule/runtime-behavioral-schedule.json",
            "import dataset must name the canonical schedule",
        )
    } else {
        check(
            request["execution"]["mode"] == "run" && request["comparison"]["dataset"].is_object(),
            "run dataset must contain its path-free preparation descriptor",
        )?;
        invarlock_native_contracts::schedule::validate_preparation(
            &request["comparison"]["dataset"],
            &typed,
        )
        .map_err(|e| integrity(e.message))
    }
}
pub(crate) fn observations(
    snapshot: &fs::PackSnapshot,
    manifest: &Value,
    request: &Value,
    identities: &BTreeMap<&str, Value>,
) -> Result<Vec<Value>> {
    let requested = request
        .get("observations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut descriptors = BTreeMap::new();
    for d in &requested {
        let id = d["id"].as_str().unwrap_or("");
        check(
            !id.is_empty() && descriptors.insert(id, d).is_none(),
            "normalized request observation entry is invalid",
        )?;
    }
    let refs = manifest.get("observations").and_then(Value::as_object);
    let keys: BTreeSet<_> = refs
        .map(|r| r.keys().map(String::as_str).collect())
        .unwrap_or_default();
    check(
        keys == descriptors.keys().copied().collect(),
        "manifest observations do not match normalized request",
    )?;
    let mut verified = vec![];
    if let Some(refs) = refs {
        let mut entries: Vec<_> = refs.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (id, r) in entries {
            let path = format!("observations/{id}.json");
            let bytes = raw(snapshot, &path, MAX_BYTES)?;
            let v = object(bytes, &format!("observation {id}"))?;
            validate_schema(&v, "evidence_observation")?;
            check(
                canonical(&v)? == bytes,
                format!("observation {id:?} must use canonical JSON"),
            )?;
            check(
                r == &json!({"path":path,"digest":sha256_digest(bytes),"kind":v["kind"],"scope":v["scope"]}),
                format!("observation {id:?} manifest reference is invalid"),
            )?;
            let p = canonical(&v["payload"])?;
            check(
                p.len() <= 1024 * 1024,
                format!("observation {id:?} payload exceeds the 1048576-byte limit"),
            )?;
            check(
                descriptors[id.as_str()]
                    == &json!({"id":id,"kind":v["kind"],"scope":v["scope"],"payload_digest":sha256_digest(&p)}),
                format!("observation {id:?} does not match normalized request"),
            )?;
            check(v["observation_id"]==*id&&v["bindings"]==json!({"comparison_id":manifest["comparison_id"],"schedule_digest":identities["dataset"]["digest"],"policy_digest":identities["policy"]["digest"],"artifact_digests":{"baseline":identities["baseline"]["digest"],"subject":identities["subject"]["digest"]}}),format!("observation {id:?} does not bind the comparison, schedule, policy, and artifacts"))?;
            verified.push(json!({"observation_id":id,"kind":v["kind"],"scope":v["scope"],"digest":sha256_digest(bytes)}));
        }
    }
    Ok(verified)
}

struct Side {
    artifact: Value,
    observation: Value,
}
pub(crate) fn derive(
    loaded: &BTreeMap<&str, &[u8]>,
    request: &Value,
    schedule: &Value,
    identities: &BTreeMap<&str, Value>,
    anchors: &Value,
    bridge: Option<&dyn ScorerReplay>,
) -> Result<Value> {
    let binding = request["comparison"].get("scorer_extension");
    let metric = binding
        .and_then(|b| b["scorer_id"].as_str())
        .or_else(|| request["comparison"]["metric"].as_str())
        .ok_or_else(|| integrity("request comparison.metric is unsupported"))?;
    check(
        binding.is_some() || ["exact_match", "normalized_nll_per_utf8_byte"].contains(&metric),
        "request comparison.metric is unsupported",
    )?;
    if let Some(binding) = binding {
        validate_schema(binding, "scorer_extension_binding")?;
        check(
            binding["configuration_sha256"] == digest_no_newline(&binding["configuration"])?,
            "configuration_sha256 does not match configuration",
        )?;
    }
    let schedule_hash = digest_no_newline(schedule)?;
    let scheduled = list(&schedule["records"], "schedule records")?;
    let mut sides = BTreeMap::new();
    for side in ["baseline", "subject"] {
        let get = |role: &str| -> Result<&[u8]> {
            loaded
                .get(format!("{side}_{role}").as_str())
                .copied()
                .ok_or_else(|| integrity(format!("missing {side} {role}")))
        };
        let artifact_raw = get("provider_identity")?;
        let artifact = typed_sidecar(
            artifact_raw,
            "model artifact identity",
            "model_artifact_identity",
        )?;
        let receipt_raw = get("provider_receipt")?;
        let receipt = typed_sidecar(
            receipt_raw,
            "runtime provider receipt",
            "runtime_provider_receipt",
        )?;
        let observation_raw = get("scoring_observation")?;
        let observation = typed_sidecar(
            observation_raw,
            "runtime scoring observation",
            "runtime_scoring_observation",
        )
        .map_err(|error| {
            integrity(format!(
                "{side} runtime provider evidence is invalid: {}",
                error.message
            ))
        })?;
        let artifact_hash = digest_no_newline(&artifact)?;
        check(json_equal(&receipt["artifact_identity"],&artifact),format!("{side} runtime provider evidence is invalid: receipt artifact_identity does not match the bound artifact"))?;
        check(receipt["scoring_observation_sha256"]==hex(observation_raw),format!("{side} runtime provider evidence is invalid: receipt scoring_observation_sha256 does not match observation bytes"))?;
        check(receipt["plugin"]["name"]==receipt["capabilities"]["provider_name"]&&receipt["plugin"]["name"]==observation["provider_name"],format!("{side} runtime provider evidence is invalid: receipt and observation provider names do not agree"))?;
        check(observation["artifact_identity_sha256"]==artifact_hash,format!("{side} runtime provider evidence is invalid: observation artifact_identity_sha256 does not match bound artifact"))?;
        check(receipt["capabilities"]["artifact_formats"].as_array().is_some_and(|a|a.contains(&artifact["artifact_format"])),format!("{side} runtime provider evidence is invalid: bound artifact format is not declared by provider capabilities"))?;
        check(receipt["outer_image_digest"]==anchors["runtime_digests"][side],format!("{side} runtime provider evidence is invalid: receipt outer_image_digest does not match expected runtime image"))?;
        check(
            receipt["capabilities"]["metrics"]
                .as_array()
                .is_some_and(|a| {
                    a.contains(&json!(if binding.is_some() {
                        "exact_match"
                    } else {
                        metric
                    }))
                }),
            format!("{side} runtime provider evidence does not declare metric {metric:?}"),
        )?;
        check(
            receipt["capabilities"]["tasks"]
                .as_array()
                .is_some_and(|a| a.contains(&schedule["task"])),
            format!("{side} runtime provider evidence does not declare task"),
        )?;
        check(
            sha256_digest(artifact_raw) == identities[side]["digest"],
            format!("{side} input identity does not match provider artifact identity"),
        )?;
        check(
            observation["schedule_sha256"] == schedule_hash,
            format!("{side} observation does not bind the canonical schedule"),
        )?;
        let records = list(&observation["records"], "observation records")?;
        check(
            records.len() == scheduled.len(),
            format!("{side} observation record order does not match the schedule"),
        )?;
        for (r, s) in records.iter().zip(scheduled) {
            let id = r["record_id"].as_str().unwrap_or("");
            check(
                r["record_id"] == s["record_id"],
                format!("{side} observation record order does not match the schedule"),
            )?;
            check(
                r["status"] == "ok",
                format!("{side} observation record {id:?} is not successful"),
            )?;
            if let Some(output) = r["output_text"].as_str() {
                check(
                    r["output_sha256"] == hex(output.as_bytes()),
                    format!(
                        "{side} observation record {id:?} output digest does not match output text"
                    ),
                )?;
            }
            check(
                r["output_text"].is_null() == r["output_sha256"].is_null(),
                "output_text and output_sha256 must be present together",
            )?;
            if !r["logprob_sum"].is_null() {
                check(
                    r["logprob_sum"].as_f64().is_some_and(f64::is_finite)
                        && r["token_count"].as_u64().is_some_and(|n| n > 0)
                        && r["utf8_byte_count"].as_u64().is_some_and(|n| n > 0),
                    "logprob_sum requires positive token_count and utf8_byte_count",
                )?;
            }
            check(
                r["input_sha256"] == s["input_sha256"],
                format!("{side} record {id:?} input digest does not match the schedule"),
            )?;
        }
        check(
            observation["aggregate_source_sha256"] == digest_no_newline(&observation["records"])?,
            format!("{side} observation aggregate_source_sha256 is invalid"),
        )?;
        let report = object(get("run_report")?, &format!("{side} runtime side report"))?;
        check(
            report
                == json!({"format":"invarlock/runtime-side-report-v1","provider":observation["provider_name"],"artifact_identity_sha256":artifact_hash,"scoring_observation_sha256":hex(observation_raw),"schedule_sha256":schedule_hash,"record_count":records.len()}),
            format!("{side} runtime side report does not bind its provider observation"),
        )?;
        request_bindings(&request["comparison"][side], &artifact, &receipt, side)?;
        let provider = request["comparison"][side]["runtime"]["provider"]
            .as_str()
            .unwrap_or("");
        if provider == "llama_cpp" {
            check(
                anchors.get("request_digest").is_some(),
                "independent request anchor is required for llama_cpp evidence",
            )?;
        }
        let model = match artifact["artifact_format"].as_str() {
            Some("hf_snapshot") => &artifact["model_id"],
            Some("gguf") => &artifact["artifact_name"],
            _ => &artifact["bundle_name"],
        };
        check(
            request["comparison"][side]["artifact"]["model_id"] == *model,
            format!("{side} request model_id does not match provider artifact identity"),
        )?;
        check(
            identities[side]["locator"] == request["comparison"][side]["artifact"]["locator"],
            format!("{side} input locator does not match the normalized request binding"),
        )?;
        check(
            identities[format!("{side}_runtime").as_str()]["locator"]
                == format!(
                    "runtime:{}",
                    identities[format!("{side}_runtime").as_str()]["digest"]
                        .as_str()
                        .unwrap_or("")
                ),
            format!("{side}_runtime input locator does not match the normalized request binding"),
        )?;
        let config_bytes = get("runtime_config")?;
        let config = object(config_bytes, &format!("{side} runtime config"))?;
        check(config==json!({"format":"invarlock/runtime-side-config-v1","role":side,"provider":provider,"artifact_identity_sha256":artifact_hash,"schedule_sha256":schedule_hash,"policy_digest":anchors["policy_digest"]}),format!("{side} runtime config does not bind role, provider, artifact, schedule, and policy"))?;
        check(
            canonical(&config)? == config_bytes,
            format!("{side} runtime config must use canonical JSON"),
        )?;

        sides.insert(
            side,
            Side {
                artifact,
                observation,
            },
        );
    }
    for (role, locator) in [
        ("dataset", "schedule/runtime-behavioral-schedule.json"),
        ("policy", "inputs/policy.json"),
    ] {
        check(
            identities[role]["locator"] == locator,
            format!("{role} input locator does not match the normalized request binding"),
        )?;
    }
    let mut extension_results = Map::new();
    if let Some(binding) = binding {
        let bridge = bridge
            .ok_or_else(|| integrity("an explicitly authorized scorer registry is required"))?;
        let kinds: BTreeSet<_> = scheduled
            .iter()
            .flat_map(|r| r["input_parts"].as_array().into_iter().flatten())
            .filter_map(|p| p["kind"].as_str())
            .collect();
        for side in ["baseline", "subject"] {
            let observed = list(&sides[side].observation["records"], "observation records")?;
            let records:Vec<_>=scheduled.iter().zip(observed).map(|(s,o)|json!({"record_id":s["record_id"],"input_sha256":s["input_sha256"],"facts":{"expected_output":s["expected_output"],"output_text":o["output_text"],"output_sha256":o["output_sha256"]}})).collect();
            check(
                records
                    .iter()
                    .all(|r| r["facts"]["output_text"].is_string()),
                format!("{side} record lacks text scorer facts"),
            )?;
            let replay_request = json!({"binding":binding,"task":schedule["task"],"input_kinds":kinds,"output_kind":"text","schedule_sha256":schedule_hash,"records":records});
            let result = bridge.replay(&replay_request)?;
            validate_schema(&result, "scorer_extension_result")?;
            for field in [
                "scorer_id",
                "scorer_version",
                "descriptor_sha256",
                "configuration_sha256",
            ] {
                check(
                    result[field] == binding[field],
                    format!("scorer result {field} does not match the replay request"),
                )?;
            }
            check(
                result["schedule_sha256"] == schedule_hash
                    && result["source_records_sha256"]
                        == digest_no_newline(&replay_request["records"])?,
                "scorer result does not bind the replay request",
            )?;
            let values = list(&result["record_results"], "scorer record results")?;
            check(
                values.len() == scheduled.len()
                    && values.iter().zip(scheduled).all(|(r, s)| {
                        r["record_id"] == s["record_id"] && r["input_sha256"] == s["input_sha256"]
                    }),
                "scorer result record bindings do not match replay input",
            )?;
            let mean = fsum(
                values
                    .iter()
                    .map(|v| v["value"].as_f64().unwrap_or(f64::NAN)),
            )
            .map_err(|e| integrity(e.to_string()))?
                / values.len() as f64;
            check(
                result["aggregate"].as_f64() == Some(mean),
                "aggregate must equal the core arithmetic mean of record results",
            )?;
            check(
                result["aggregate_source_sha256"] == digest_no_newline(&result["record_results"])?,
                "aggregate_source_sha256 does not match record_results",
            )?;
            extension_results.insert(side.into(), result);
        }
    }
    let mut paired = vec![];
    for (i, s) in scheduled.iter().enumerate() {
        let mut pair = json!({"record_id":s["record_id"],"input_sha256":s["input_sha256"]});
        for side in ["baseline", "subject"] {
            let r = &sides[side].observation["records"][i];
            let score = if binding.is_some() {
                extension_results[side]["record_results"][i]["value"]
                    .as_f64()
                    .unwrap_or(f64::NAN)
            } else if metric == "exact_match" {
                let output = r["output_text"]
                    .as_str()
                    .ok_or_else(|| integrity(format!("{side} record lacks output text")))?;
                if Some(output) == s["expected_output"].as_str() {
                    1.0
                } else {
                    0.0
                }
            } else {
                let count = r["utf8_byte_count"]
                    .as_u64()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| {
                        integrity(format!("{side} record lacks normalized NLL facts"))
                    })?;
                check(
                    count as usize == s["expected_output"].as_str().unwrap_or("").len(),
                    format!("{side} record utf8_byte_count does not match expected_output"),
                )?;
                let value = -r["logprob_sum"].as_f64().ok_or_else(|| {
                    integrity(format!("{side} record lacks normalized NLL facts"))
                })? / count as f64;
                check(
                    value.is_finite() && value >= 0.0,
                    format!("{side} record has invalid normalized NLL"),
                )?;
                value
            };
            pair[side] = json!({"observation_record_digest":format!("sha256:{}",digest_no_newline(r)?),"score":score});
        }
        paired.push(pair);
    }
    let mut result = json!({"format":"invarlock/paired-records-v1","metric":metric,"schedule_sha256":schedule_hash,"records":paired});
    if metric == "normalized_nll_per_utf8_byte" {
        result["derived_measurements"] = json!({"perplexity_ratio":perplexity(&sides)?});
    }
    if let Some(binding) = binding {
        result["scorer_extension"] = binding.clone();
        result["scorer_replay"] = Value::Object(extension_results);
    }
    Ok(result)
}
fn perplexity(sides: &BTreeMap<&str, Side>) -> Result<Value> {
    let mut unavailable = json!({"status":"unavailable","basis":"authenticated_target_likelihood","method":"target_token_weighted_perplexity_ratio_v1"});
    let mut reason = None;
    let b = &sides["baseline"];
    let s = &sides["subject"];
    let mut tokens = 0u64;
    let mut bp = vec![];
    let mut sp = vec![];
    if b.artifact["tokenizer_metadata_sha256"] != s.artifact["tokenizer_metadata_sha256"] {
        reason = Some("tokenizer_contracts_differ");
    }
    if reason.is_none() {
        for (b, s) in list(&b.observation["records"], "records")?
            .iter()
            .zip(list(&s.observation["records"], "records")?)
        {
            let bn = b["token_count"].as_u64().unwrap_or(0);
            let sn = s["token_count"].as_u64().unwrap_or(0);
            if bn == 0 || sn == 0 {
                reason = Some("target_token_counts_unavailable");
                break;
            }
            if bn != sn {
                reason = Some("target_token_counts_differ");
                break;
            }
            tokens = tokens
                .checked_add(bn)
                .ok_or_else(|| integrity("target token count overflow"))?;
            bp.push(b["logprob_sum"].as_f64().unwrap_or(f64::NAN));
            sp.push(s["logprob_sum"].as_f64().unwrap_or(f64::NAN));
        }
    }
    if let Some(reason) = reason {
        unavailable["reason"] = json!(reason);
        return Ok(unavailable);
    }
    let bn = -fsum(bp).map_err(|e| integrity(e.to_string()))? / tokens as f64;
    let sn = -fsum(sp).map_err(|e| integrity(e.to_string()))? / tokens as f64;
    let (bv, sv, ratio) = (bn.exp(), sn.exp(), (sn - bn).exp());
    if ![bv, sv, ratio].iter().all(|v| v.is_finite() && *v > 0.0) {
        unavailable["reason"] = json!("derived_value_non_finite");
        return Ok(unavailable);
    }
    Ok(
        json!({"status":"available","basis":"authenticated_target_likelihood","method":"target_token_weighted_perplexity_ratio_v1","tokenizer_metadata_sha256":b.artifact["tokenizer_metadata_sha256"],"target_token_count":tokens,"baseline_perplexity":bv,"subject_perplexity":sv,"ratio":ratio}),
    )
}
pub(crate) fn request_bindings(
    request: &Value,
    artifact: &Value,
    receipt: &Value,
    side: &str,
) -> Result<()> {
    invarlock_native_contracts::runtime::request_bindings(request, artifact, receipt)
        .map_err(|e| integrity(format!("{side} {e}")))
}
pub(crate) fn runtime_manifest<'a>(
    side: &str,
    manifest: &Value,
    receipt: &Value,
    runtime: &Value,
    get: &impl Fn(&str) -> Result<&'a [u8]>,
) -> Result<()> {
    validate_schema(manifest, "runtime_manifest")?;
    let prefix = format!("{side} runtime manifest");
    check(
        manifest["report"]["sha256"] == hex(get("run_report")?),
        format!("{prefix}: report digest mismatch"),
    )?;
    check(
        manifest["report"]["path"] == "report.json"
            && manifest["report"]["filename"] == "report.json",
        format!("{prefix}: report filename does not match the verified report filename"),
    )?;
    check(
        manifest["config"]["source"] == "file",
        format!("{prefix}: strict runtime manifest requires a verifiable file config"),
    )?;
    check(
        manifest["config"]["path"] == "run.yaml"
            && manifest["config"]["sha256"] == hex(get("runtime_config")?),
        format!("{prefix}: runtime config digest or path does not match"),
    )?;
    let outer = &manifest["outer_container"];
    check(
        outer["image_digest"] == *runtime,
        format!("{prefix}: runtime image digest mismatch"),
    )?;
    let reference = outer["image_ref"].as_str().unwrap_or("");
    let digest = runtime.as_str().unwrap_or("");
    check(
        reference == digest
            || reference
                .split_once('@')
                .is_some_and(|(r, d)| !r.is_empty() && !d.contains('@') && d == digest),
        format!("{prefix}: outer_container.image_ref must bind image_digest"),
    )?;
    for name in [
        "allow_network",
        "allow_remote_code",
        "allow_third_party_plugins",
    ] {
        check(
            outer[name] == false,
            format!("{prefix}: strict runtime forbids {name}=true"),
        )?;
    }
    for (role, file, key) in [
        (
            "receipt",
            "runtime-provider.receipt.json",
            "provider_receipt",
        ),
        (
            "scoring_observation",
            "runtime-scoring.observation.json",
            "scoring_observation",
        ),
        (
            "artifact_identity",
            "model-artifact.identity.json",
            "provider_identity",
        ),
    ] {
        let r = &manifest["runtime_provider"][role];
        check(
            r["filename"] == file && r["sha256"] == hex(get(key)?),
            format!("{prefix}: {role} digest or filename does not match bound sidecar"),
        )?;
    }
    check(
        receipt["outer_image_digest"] == outer["image_digest"],
        format!("{prefix}: receipt outer image does not match manifest"),
    )?;
    Ok(())
}

pub(crate) fn runtime_checks(loaded: &BTreeMap<&str, &[u8]>, anchors: &Value) -> Vec<String> {
    let mut errors = vec![];
    for side in ["baseline", "subject"] {
        let get = |role: &str| -> Result<&[u8]> {
            loaded
                .get(format!("{side}_{role}").as_str())
                .copied()
                .ok_or_else(|| integrity(format!("missing {side} {role}")))
        };
        let result = (|| {
            runtime_manifest(
                side,
                &object(get("runtime_manifest")?, "runtime manifest")?,
                &object(get("provider_receipt")?, "provider receipt")?,
                &anchors["runtime_digests"][side],
                &get,
            )
        })();
        if let Err(e) = result {
            errors.push(e.message);
        }
    }
    errors
}
