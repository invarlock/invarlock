use crate::{
    fsum, paired_exact_match_statistics, paired_resampling_interval, score, validate_configuration,
};
use invarlock_native_contracts::{
    canonical_json, canonical_json_digest, canonical_json_matches, canonical_json_no_newline,
    canonical_json_to_writer, validate_schema, SchemaCheckedValue,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use serde_json::{json, Value};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
pub const DEFAULT_MAX_BOOTSTRAP_DRAWS: u64 = 102_400_000;
const MAX_BYTES: usize = 128 * 1024 * 1024;
const NLL: &str = "normalized_nll_per_utf8_byte";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError {
    pub reason: &'static str,
    pub message: String,
}
impl AnalysisError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            reason: "invalid_evaluation_records",
            message: message.into(),
        }
    }
}
impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for AnalysisError {}
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn canonical(v: &Value) -> Result<Vec<u8>, AnalysisError> {
    canonical_json(v).map_err(|e| err(e.to_string()))
}
fn digest(v: &Value) -> Result<String, AnalysisError> {
    canonical_json_digest(v).map_err(|e| err(e.to_string()))
}
// Preserve the canonical case-identity projection without owning another copy
// of every record. These fixed keys are in canonical Unicode lexical order;
// nested field values still use the shared canonical encoder.
fn case_seed_to_writer(
    pairs: &[(&Value, &Value)],
    writer: &mut impl std::io::Write,
) -> Result<(), AnalysisError> {
    fn put(writer: &mut impl std::io::Write, bytes: &[u8]) -> Result<(), AnalysisError> {
        writer
            .write_all(bytes)
            .map_err(|e| err(format!("value is not canonical JSON: {e}")))
    }
    put(writer, b"[")?;
    for (index, (left, _)) in pairs.iter().enumerate() {
        if index != 0 {
            put(writer, b",")?;
        }
        put(writer, b"{")?;
        for (index, (key, encoded)) in [
            ("expected", b"\"expected\":".as_slice()),
            ("id", b"\"id\":".as_slice()),
            ("input", b"\"input\":".as_slice()),
            ("metadata", b"\"metadata\":".as_slice()),
        ]
        .iter()
        .enumerate()
        {
            if index != 0 {
                put(writer, b",")?;
            }
            put(writer, encoded)?;
            canonical_json_to_writer(&left[*key], writer, false).map_err(|e| err(e.to_string()))?;
        }
        put(writer, b"}")?;
    }
    // digest() previously used canonical_json(), which includes this newline.
    put(writer, b"]\n")
}
fn case_seed(pairs: &[(&Value, &Value)]) -> Result<String, AnalysisError> {
    struct HashWriter(sha2::Sha256);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            sha2::Digest::update(&mut self.0, bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(sha2::Sha256::default());
    case_seed_to_writer(pairs, &mut writer)?;
    Ok(format!("sha256:{:x}", sha2::Digest::finalize(writer.0)))
}
fn schema(v: &Value, name: &str) -> Result<(), AnalysisError> {
    validate_schema(v, name).map_err(|e| err(e.to_string()))
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap()
}
fn numeric(v: &Value) -> Result<f64, AnalysisError> {
    v.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
        err(
            if v.is_number() && !v.to_string().contains(['.', 'e', 'E']) {
                "int too large to convert to float"
            } else {
                "record score must be finite"
            },
        )
    })
}
fn binary(kind: &str) -> bool {
    matches!(
        kind,
        "exact_match" | "normalized_match" | "numeric_tolerance" | "json_exact"
    )
}
pub(crate) fn py_number(v: &Value) -> Option<BigRational> {
    if let Some(b) = v.as_bool() {
        return Some(BigRational::from_integer(BigInt::from(u8::from(b))));
    }
    let n = v.as_number()?;
    let s = n.to_string();
    if !s.contains(['.', 'e', 'E']) {
        s.parse::<BigInt>().ok().map(BigRational::from_integer)
    } else {
        BigRational::from_float(n.as_f64()?)
    }
}
fn float_cmp(value: f64, threshold: &Value) -> std::cmp::Ordering {
    if value == f64::INFINITY {
        return std::cmp::Ordering::Greater;
    }
    if value == f64::NEG_INFINITY {
        return std::cmp::Ordering::Less;
    }
    BigRational::from_float(value)
        .expect("finite computed comparison")
        .cmp(&py_number(threshold).expect("validated numeric threshold"))
}
pub(crate) fn py_equal(a: &Value, b: &Value) -> bool {
    if let (Some(a), Some(b)) = (py_number(a), py_number(b)) {
        return a == b;
    }
    match (a, b) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| py_equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| py_equal(a, b)))
        }
        _ => a == b,
    }
}
fn projection_tokens(v: &Value) -> Result<String, AnalysisError> {
    let obj = v
        .as_object()
        .ok_or_else(|| err("input projection requires kind and pointer only"))?;
    if obj.len() != 2 || v["kind"] != "json-pointer" || !obj.contains_key("pointer") {
        return Err(err("input projection requires kind and pointer only"));
    }
    let p = v["pointer"]
        .as_str()
        .ok_or_else(|| err("input projection requires an RFC6901 JSON pointer"))?;
    if p.chars().count() > 4096
        || !p.starts_with('/')
        || p.split('/')
            .any(|p| p.replace("~0", "").replace("~1", "").contains('~'))
    {
        return Err(err("input projection requires an RFC6901 JSON pointer"));
    }
    let first = p
        .split('/')
        .nth(1)
        .unwrap()
        .replace("~1", "/")
        .replace("~0", "~");
    if !["input", "context"].contains(&first.as_str()) {
        return Err(err("input projection must select input or context"));
    }
    Ok(p.into())
}
pub fn verify_input_projection(row: &Value) -> Result<(), AnalysisError> {
    let Some(context) = row["context"].as_object() else {
        return Ok(());
    };
    let Some(binding) = context.get("input_projection") else {
        return Ok(());
    };
    if !binding.as_object().is_some_and(|b| {
        b.len() == 4
            && [
                "configuration",
                "configuration_digest",
                "source",
                "source_digest",
            ]
            .iter()
            .all(|k| b.contains_key(*k))
    }) {
        return Err(err("input projection binding has invalid fields"));
    }
    let source = &binding["source"];
    if !source
        .as_object()
        .is_some_and(|s| s.len() == 2 && s.contains_key("input") && s.contains_key("context"))
    {
        return Err(err("input projection source has invalid fields"));
    }
    let configuration = &binding["configuration"];
    let pointer = projection_tokens(configuration)?;
    let selected = crate::scoring::pointer(source, &pointer)
        .ok_or_else(|| err("input projection pointer does not resolve"))?;
    if !selected.is_string() {
        return Err(err("input projection must select one text string"));
    }
    if digest(configuration)? != binding["configuration_digest"]
        || digest(source)? != binding["source_digest"]
        || selected != &row["input"]
    {
        return Err(err("input projection does not replay its retained binding"));
    }
    let original = if source["context"].is_object() {
        source["context"].clone()
    } else {
        json!({})
    };
    let mut remaining = context.clone();
    remaining.remove("input_projection");
    if original.get("input_projection").is_some()
        || digest(&Value::Object(remaining))? != digest(&original)?
    {
        return Err(err("input projection original context does not match"));
    }
    Ok(())
}
fn timestamp(v: &Value) -> Result<[u32; 7], AnalysisError> {
    let fail = || err("service_identity requires valid UTC timestamps");
    let s = v.as_str().ok_or_else(fail)?;
    let chunks = s
        .trim_end_matches('Z')
        .split(['-', 'T', ':', '.'])
        .collect::<Vec<_>>();
    if !(6..=7).contains(&chunks.len()) {
        return Err(fail());
    }
    let mut parts = [0; 7];
    for i in 0..6 {
        parts[i] = chunks[i].parse().map_err(|_| fail())?
    }
    if chunks.len() == 7 {
        parts[6] = format!("{:0<6}", chunks[6]).parse().map_err(|_| fail())?
    }
    let [year, month, day, hour, minute, second, _] = parts;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    if year == 0 || year > 9999 || day == 0 || day > days || hour > 23 || minute > 59 || second > 59
    {
        return Err(fail());
    }
    Ok(parts)
}
fn service_digest(run: &Value) -> Result<Option<String>, AnalysisError> {
    let Some(identity) = run.get("service_identity") else {
        return Ok(None);
    };
    if canonical(identity)?.len() > 1024 * 1024 {
        return Err(err("service_identity exceeds the 1 MiB byte limit"));
    }
    if identity["configuration_digest"] != digest(&identity["configuration"])? {
        return Err(err("service_identity configuration digest differs"));
    }
    let window = &identity["observation_window"];
    let start = timestamp(&window["started_at"])?;
    let end = timestamp(&window["ended_at"])?;
    if start > end {
        return Err(err("service_identity observation window is reversed"));
    }
    if !run["artifact_digest"].is_null() {
        return Err(err("service identity cannot claim an artifact digest"));
    }
    Ok(Some(digest(identity)?))
}
fn likelihood_record(row: &Value, run: &Value, service: Option<&str>) -> Result<(), AnalysisError> {
    let Some(facts) = row.get("likelihood") else {
        return Ok(());
    };
    let label = format!(
        "record {}: reference-continuation likelihood",
        text(row, "id")
    );
    let reference = row["expected"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(format!("{label} requires nonempty reference text")))?;
    for key in ["token_count", "utf8_byte_count"] {
        let s = facts[key].to_string();
        if !facts[key].is_number()
            || s.contains(['.', 'e', 'E'])
            || s.parse::<BigInt>()
                .map(|x| x <= BigInt::from(0))
                .unwrap_or(true)
        {
            return Err(err(format!(
                "{label} {key} must be a positive exact integer"
            )));
        }
    }
    if facts["utf8_byte_count"].as_u64() != Some(reference.len() as u64) {
        return Err(err(format!(
            "{label} utf8_byte_count differs from reference"
        )));
    }
    let captured = row["context"]
        .get("input_projection")
        .map(|p| &p["source"]["input"])
        .unwrap_or(&row["input"]);
    if run.get("service_identity").is_some() {
        if service.is_none() || facts["service_identity_digest"].as_str() != service {
            return Err(err(format!(
                "{label} service_identity_digest binding differs"
            )));
        }
        if facts["configuration_digest"] != run["service_identity"]["configuration_digest"] {
            return Err(err(format!(
                "{label} service configuration binding differs"
            )));
        }
    } else if facts.get("service_identity_digest").is_some() {
        return Err(err(format!(
            "{label} service identity cannot replace an artifact"
        )));
    }
    for (key, expected) in [
        ("input_digest", json!(digest(captured)?)),
        ("reference_digest", json!(digest(&row["expected"])?)),
        ("artifact_digest", run["artifact_digest"].clone()),
        ("source", run["source"].clone()),
    ] {
        if !py_equal(&facts[key], &expected) {
            return Err(err(format!("{label} {key} binding differs")));
        }
    }
    Ok(())
}
pub fn check_run(run: &Value) -> Result<(), AnalysisError> {
    schema(run, "run")?;
    check_run_semantics(run)
}
fn check_run_semantics(run: &Value) -> Result<(), AnalysisError> {
    let service = service_digest(run)?;
    let rows = run["records"].as_array().unwrap();
    for row in rows {
        verify_input_projection(row)?;
        likelihood_record(row, run, service.as_deref())?
    }
    let mut ids = BTreeSet::new();
    if rows.iter().any(|r| !ids.insert(text(r, "id"))) {
        return Err(err(
            "duplicate record IDs; repeated trials need distinct paired IDs",
        ));
    }
    Ok(())
}
pub fn check_policy(policy: &Value) -> Result<(), AnalysisError> {
    schema(policy, "policy")?;
    check_policy_semantics(policy)
}
fn check_policy_semantics(policy: &Value) -> Result<(), AnalysisError> {
    let mut names = BTreeSet::new();
    let mut slices = BTreeSet::new();
    if policy["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| !names.insert(text(m, "name")))
        || policy["slices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| !slices.insert(text(s, "name")) || s["name"] == "overall")
    {
        return Err(err(
            "metric and slice names must be unique; overall is reserved",
        ));
    }
    for metric in policy["metrics"].as_array().unwrap() {
        let kind = text(metric, "kind");
        if kind == "recorded" {
            if metric.get("accepted_provenance").is_none()
                || metric
                    .get("score_key")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                return Err(err(
                    "recorded metric requires explicit accepted_provenance and score_key",
                ));
            }
            if metric["configuration"]
                .as_object()
                .is_some_and(|v| !v.is_empty())
            {
                return Err(err("recorded metric cannot redefine the recorded scorer"));
            }
            let provenance = &metric["accepted_provenance"];
            if provenance["unit"] != metric["unit"] {
                return Err(err("recorded metric unit differs from approved provenance"));
            }
            if matches!(provenance["kind"].as_str(), Some("judge" | "human"))
                && provenance["rubric_digest"].is_null()
            {
                return Err(err(
                    "recorded rating provenance requires an approved rubric digest",
                ));
            }
        } else if kind == NLL {
            if metric["direction"] != "lower" || metric["unit"] != "nats_per_utf8_byte" {
                return Err(err(
                    "normalized NLL uses lower-is-better nats_per_utf8_byte",
                ));
            }
            if metric.get("maximum_regression").is_some() {
                return Err(err(
                    "normalized NLL requires ratio_max, not maximum_regression",
                ));
            }
        } else {
            if metric.get("score_key").is_some() || metric.get("accepted_provenance").is_some() {
                return Err(err("recomputed metrics cannot accept recorded provenance"));
            }
            if metric["direction"] != "higher" || metric["unit"] != "score" {
                return Err(err(
                    "deterministic scorers use higher-is-better scores in [0,1]",
                ));
            }
            validate_configuration(kind, &metric["configuration"])?
        }
        if metric
            .get("subject_minimum")
            .zip(metric.get("subject_maximum"))
            .is_some_and(|(lower, upper)| py_number(lower).unwrap() > py_number(upper).unwrap())
        {
            return Err(err("subject minimum exceeds maximum"));
        }
    }
    Ok(())
}
pub use check_policy as validate_policy;
pub use check_run as validate_run;
pub struct ValidatedRun<'a> {
    value: &'a Value,
    checked: Option<&'a SchemaCheckedValue>,
}
impl<'a> ValidatedRun<'a> {
    pub fn new(value: &'a Value) -> Result<Self, AnalysisError> {
        check_run(value)?;
        Ok(Self {
            value,
            checked: None,
        })
    }
    pub fn from_schema_checked(checked: &'a SchemaCheckedValue) -> Result<Self, AnalysisError> {
        if checked.schema() != "run" {
            return Self::new(checked.value());
        }
        check_run_semantics(checked.value())?;
        Ok(Self {
            value: checked.value(),
            checked: Some(checked),
        })
    }
    pub fn value(&self) -> &'a Value {
        self.value
    }
}
pub struct ValidatedPolicy<'a> {
    value: &'a Value,
    checked: Option<&'a SchemaCheckedValue>,
}
impl<'a> ValidatedPolicy<'a> {
    pub fn new(value: &'a Value) -> Result<Self, AnalysisError> {
        check_policy(value)?;
        Ok(Self {
            value,
            checked: None,
        })
    }
    pub fn from_schema_checked(checked: &'a SchemaCheckedValue) -> Result<Self, AnalysisError> {
        if checked.schema() != "policy" {
            return Self::new(checked.value());
        }
        check_policy_semantics(checked.value())?;
        Ok(Self {
            value: checked.value(),
            checked: Some(checked),
        })
    }
    pub fn value(&self) -> &'a Value {
        self.value
    }
}
pub(crate) fn missing(left: &Value, right: &Value, metric: &Value) -> bool {
    let kind = text(metric, "kind");
    [left, right].iter().any(|row| {
        !row["error"].is_null()
            || if kind == "recorded" {
                row["scores"].get(text(metric, "score_key")).is_none()
            } else if kind == NLL {
                row.get("likelihood").is_none()
            } else if matches!(kind, "exact_match" | "normalized_match" | "token_f1") {
                !row["output"].is_string()
            } else {
                kind == "numeric_tolerance" && row["output"].is_null()
            }
    })
}
fn interval(b: &[f64], s: &[f64], seed: &str, is_binary: bool) -> Result<Value, AnalysisError> {
    if is_binary {
        let stats =
            paired_exact_match_statistics(&json!(b), &json!(s), "newcombe_hybrid_score_paired_v2")?;
        let ci = &stats["effect_size_confidence_interval"];
        return Ok(
            json!({"lower":ci["lower_pp"].as_f64().unwrap()/100.0,"upper":ci["upper_pp"].as_f64().unwrap()/100.0,"method":ci["method"],"mass":0.95,"replicates":0}),
        );
    }
    let differences = b.iter().zip(s).map(|(b, s)| s - b).collect::<Vec<_>>();
    let count = differences.len();
    let first = differences[0];
    let (lower, upper) = if differences.iter().all(|v| *v == first) {
        (first, first)
    } else {
        let ceiling = ((1u128 << 64) / count as u128) * count as u128;
        let mut draws = Vec::with_capacity(2048);
        for replicate in 0..2048 {
            let mut values = Vec::with_capacity(count);
            let mut block = 0;
            while values.len() < count {
                let mut shake = Shake256::default();
                shake.update(format!("{seed}:{replicate}:{block}").as_bytes());
                let mut reader = shake.finalize_xof();
                for _ in 0..count {
                    let mut bytes = [0u8; 8];
                    reader.read(&mut bytes);
                    let v = u64::from_le_bytes(bytes);
                    if (v as u128) < ceiling {
                        values.push(differences[(v % count as u64) as usize]);
                        if values.len() == count {
                            break;
                        }
                    }
                }
                block += 1
            }
            draws.push(fsum(values).map_err(|e| err(e.to_string()))? / count as f64)
        }
        draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile = |q: f64| {
            let pos = (draws.len() - 1) as f64 * q;
            let low = pos as usize;
            draws[low] + (draws[(low + 1).min(draws.len() - 1)] - draws[low]) * (pos - low as f64)
        };
        (percentile(0.025), percentile(0.975))
    };
    Ok(
        json!({"lower":lower,"upper":upper,"method":"paired_mean_shake256_percentile_v1","mass":0.95,"replicates":2048}),
    )
}
fn metric_result(
    baseline: &Value,
    subject: &Value,
    metric: &Value,
    slice: &str,
    pairs: &[(&Value, &Value)],
) -> Result<Value, AnalysisError> {
    let kind = text(metric, "kind");
    let recorded = kind == "recorded";
    let likelihood = kind == NLL;
    if likelihood {
        for &(left, right) in pairs {
            for (side, row) in [("baseline", left), ("subject", right)] {
                if let Some(facts) = row.get("likelihood") {
                    if facts["configuration_digest"]
                        != metric["configuration"]["configuration_digest"]
                    {
                        return Err(err(format!(
                            "record {}: likelihood configuration differs from policy",
                            text(row, "id")
                        )));
                    }
                    if facts["tokenizer_digest"]
                        != metric["configuration"][format!("{side}_tokenizer_digest")]
                    {
                        return Err(err(format!(
                            "record {}: likelihood tokenizer differs from policy",
                            text(row, "id")
                        )));
                    }
                }
            }
        }
    }
    if recorded {
        for run in [baseline, subject] {
            if !py_equal(
                &run["score_provenance"][text(metric, "score_key")],
                &metric["accepted_provenance"],
            ) {
                return Err(err(format!(
                    "metric {}: scorer provenance differs from approved policy",
                    text(metric, "name")
                )));
            }
        }
    }
    let mut values = [Vec::new(), Vec::new()];
    let mut missing_ids = Vec::new();
    for &(left, right) in pairs {
        if missing(left, right, metric) {
            missing_ids.push(left["id"].clone());
            continue;
        }
        for (side, row) in [left, right].iter().enumerate() {
            let result = if recorded {
                numeric(&row["scores"][text(metric, "score_key")])
            } else if likelihood {
                numeric(&row["likelihood"]["logprob_sum"]).and_then(|sum| {
                    numeric(&row["likelihood"]["utf8_byte_count"]).map(|count| -sum / count)
                })
            } else {
                score(
                    kind,
                    &row["expected"],
                    &row["output"],
                    &metric["configuration"],
                )
            };
            let value = result.map_err(|e| {
                err(format!(
                    "metric {}, record {}: {e}",
                    text(metric, "name"),
                    text(row, "id")
                ))
            })?;
            if !value.is_finite() {
                return Err(err("record score must be finite"));
            }
            values[side].push(value)
        }
    }
    let mut result = json!({"name":metric["name"],"slice":slice,"kind":metric["kind"],"unit":metric["unit"],"direction":metric["direction"],"aggregation":"mean","scoring_assurance":if recorded{"recorded"}else{"recomputed"},"count":pairs.len(),"missing_ids":missing_ids,"baseline_mean":null,"subject_mean":null,"delta":null,"interval":null,"decision":"insufficient_evidence","reasons":[]});
    if likelihood {
        result["ratio"] = Value::Null;
        result["interval_unit"] = json!("ratio")
    }
    if !missing_ids.is_empty() || (pairs.len() as f64) < metric["minimum_count"].as_f64().unwrap() {
        result["reasons"] = json!([if !missing_ids.is_empty() && likelihood {
            "missing reference-continuation likelihood facts or errored results; supply typed likelihood for every pair"
        } else if !missing_ids.is_empty() {
            "missing results"
        } else {
            "minimum record count not met"
        }]);
        return Ok(result);
    }
    let count = values[0].len() as f64;
    let b = fsum(values[0].iter().copied())
        .map_err(|_| err("metric arithmetic exceeded finite numeric range"))?
        / count;
    let c = fsum(values[1].iter().copied())
        .map_err(|_| err("metric arithmetic exceeded finite numeric range"))?
        / count;
    let seed = case_seed(pairs)?;
    let ci = if likelihood {
        if b <= 0.0 {
            return Err(err(
                "captured normalized NLL: normalized NLL baseline mean must be greater than zero",
            ));
        }
        let ratio = c / b;
        let (lower, upper) = paired_resampling_interval(
            NLL,
            &values[0],
            &values[1],
            seed.trim_start_matches("sha256:"),
        )
        .map_err(|e| err(format!("captured normalized NLL: {e}")))?;
        if ![ratio, lower, upper].iter().all(|v| v.is_finite()) {
            return Err(err(
                "normalized NLL ratio arithmetic exceeded finite numeric range",
            ));
        }
        result["ratio"] = json!(ratio);
        json!({"lower":lower,"upper":upper,"method":"paired_percentile_bootstrap_sha256_v1","mass":0.95,"replicates":2048})
    } else {
        interval(&values[0], &values[1], &seed, binary(kind))?
    };
    let lower = ci["lower"].as_f64().unwrap_or(f64::NAN);
    let upper = ci["upper"].as_f64().unwrap_or(f64::NAN);
    if ![b, c, c - b, lower, upper].iter().all(|v| v.is_finite()) {
        return Err(err("metric arithmetic exceeded finite numeric range"));
    }
    result["baseline_mean"] = json!(b);
    result["subject_mean"] = json!(c);
    result["delta"] = json!(c - b);
    result["interval"] = ci;
    let mut reasons = Vec::new();
    if metric["direction"] == "higher" && float_cmp(-lower, &metric["maximum_regression"]).is_gt() {
        reasons.push("lower interval bound exceeds allowed regression")
    }
    if metric["direction"] == "lower"
        && float_cmp(
            upper,
            &metric[if likelihood {
                "ratio_max"
            } else {
                "maximum_regression"
            }],
        )
        .is_gt()
    {
        reasons.push(if likelihood {
            "upper ratio interval bound exceeds ratio_max"
        } else {
            "upper interval bound exceeds allowed regression"
        })
    }
    if metric
        .get("subject_minimum")
        .is_some_and(|minimum| float_cmp(c, minimum).is_lt())
        || metric
            .get("subject_maximum")
            .is_some_and(|maximum| float_cmp(c, maximum).is_gt())
    {
        reasons.push("subject mean fails absolute bounds")
    }
    if !reasons.is_empty() {
        result["decision"] = json!("regression");
        result["reasons"] = json!(reasons)
    } else if float_cmp(upper - lower, &metric["maximum_interval_width"]).is_gt() {
        result["reasons"] = json!(["interval is too wide"])
    } else {
        result["decision"] = json!("pass")
    }
    Ok(result)
}
pub fn compare_runs(
    baseline: &Value,
    subject: &Value,
    policy: &Value,
    max_bootstrap_draws: Option<u64>,
) -> Result<Value, AnalysisError> {
    let baseline = ValidatedRun::new(baseline)?;
    let subject = ValidatedRun::new(subject)?;
    let policy = ValidatedPolicy::new(policy)?;
    compare_validated_runs(&baseline, &subject, &policy, max_bootstrap_draws)
}
pub fn compare_validated_runs(
    baseline: &ValidatedRun<'_>,
    subject: &ValidatedRun<'_>,
    policy: &ValidatedPolicy<'_>,
    max_bootstrap_draws: Option<u64>,
) -> Result<Value, AnalysisError> {
    let checked = [baseline.checked, subject.checked, policy.checked];
    let baseline = baseline.value;
    let subject = subject.value;
    let policy = policy.value;
    if let Some(expected) = policy.get("expected_case_set_digest") {
        for run in [baseline, subject] {
            let mut rows = run["records"]
                .as_array()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            rows.sort_by_key(|r| text(r, "id"));
            let cases = json!({"format":"invarlock/evaluation-case-set-v1","cases":rows.into_iter().map(|r|json!({"id":r["id"],"input":r["input"],"expected":r["expected"],"metadata":r["metadata"]})).collect::<Vec<_>>()});
            schema(&cases, "case_set")?;
            if digest(&cases)? != expected.as_str().unwrap() {
                return Err(err(format!(
                    "run {} does not match the planned case set",
                    invarlock_native_contracts::python_repr(&run["run_id"])
                )));
            }
        }
    }
    fn rows(run: &Value) -> BTreeMap<&str, &Value> {
        run["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| (text(row, "id"), row))
            .collect()
    }
    let left = rows(baseline);
    let right = rows(subject);
    if left.keys().ne(right.keys()) {
        return Err(err(
            "baseline/subject record IDs differ; export the complete paired schedule",
        ));
    }
    let mut pairs = Vec::new();
    for (id, left) in left {
        let right = right[id];
        let lb = left["context"].get("input_projection");
        let rb = right["context"].get("input_projection");
        if lb.is_some() || rb.is_some() {
            if let (Some(lb), Some(rb)) = (lb, rb) {
                if lb["configuration_digest"] != rb["configuration_digest"]
                    || digest(&lb["source"]["input"])? != digest(&rb["source"]["input"])?
                {
                    return Err(err("paired input projection or original input changed"));
                }
            } else {
                return Err(err("paired input projection or original input changed"));
            }
        }
        for key in ["input", "expected", "metadata"] {
            if !canonical_json_matches(&right[key], &canonical(&left[key])?)
                .map_err(|e| err(e.to_string()))?
            {
                return Err(err(format!("record {id}: {key} changed between runs")));
            }
        }
        pairs.push((left, right))
    }
    let mut scopes = vec![("overall", pairs.clone())];
    for subset in policy["slices"].as_array().unwrap() {
        let selected = pairs
            .iter()
            .copied()
            .filter(|(b, _)| {
                subset["where"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .all(|(k, v)| py_equal(b["metadata"].get(k).unwrap_or(&Value::Null), v))
            })
            .collect::<Vec<_>>();
        scopes.push((text(subset, "name"), selected))
    }
    let metrics = policy["metrics"].as_array().unwrap();
    let scope_count = scopes.iter().map(|(_, s)| s.len() as u64).sum::<u64>();
    let required =
        scope_count * metrics.iter().filter(|m| !binary(text(m, "kind"))).count() as u64 * 2048;
    if max_bootstrap_draws.is_some_and(|max| required > max) {
        return Err(AnalysisError{reason:"local_work_budget_exceeded",message:format!("comparison plans {required} bootstrap draws; local budget is {}. Increase the local budget only on a suitable host; the complete policy has not been evaluated.",max_bootstrap_draws.unwrap())});
    }
    let mut missing_bytes = 0;
    let mut lengths = BTreeMap::<&str, usize>::new();
    for (_, selected) in &scopes {
        for metric in metrics {
            missing_bytes += 2;
            let mut first = true;
            for &(left, right) in selected {
                if missing(left, right, metric) {
                    let id = text(left, "id");
                    let len = if let Some(n) = lengths.get(id) {
                        *n
                    } else {
                        let n = canonical_json_no_newline(&left["id"])
                            .map_err(|e| err(e.to_string()))?
                            .len();
                        lengths.insert(id, n);
                        n
                    };
                    missing_bytes += len + usize::from(!first);
                    first = false
                }
                if missing_bytes > MAX_BYTES {
                    return Err(err(format!("comparison missing-ID arrays already require at least {missing_bytes} bytes; the byte limit is {MAX_BYTES}. The complete policy has not been evaluated.")));
                }
            }
        }
    }
    let mut results = Vec::new();
    for (slice, selected) in &scopes {
        for metric in metrics {
            results.push(metric_result(baseline, subject, metric, slice, selected)?);
        }
    }
    let decision = if results.iter().any(|r| r["decision"] == "regression") {
        "regression"
    } else if results
        .iter()
        .any(|r| r["decision"] == "insufficient_evidence")
    {
        "insufficient_evidence"
    } else {
        "pass"
    };
    let mut limitations=vec!["Recorded scores retain source judgments; only their aggregation is recomputed.","Intervals describe the paired schedule under the stated method; representativeness requires an external sampling design.","Metric and slice intervals are marginal, not a simultaneous family-wide confidence guarantee.","A policy pass does not establish truthful model execution, general quality, safety or compliance."];
    if [baseline, subject]
        .iter()
        .any(|run| run.get("service_identity").is_some())
    {
        limitations.push("Hosted identity digests authenticate declared service descriptors and observation windows; they do not identify underlying weights or establish hidden configuration or later service behavior.")
    }
    if metrics.iter().any(|m| m["kind"] == NLL) {
        limitations.push("Normalized NLL replays captured reference-continuation likelihood facts; their model execution, tokenizer and configuration claims are source assertions, not independently established runtime facts.")
    }
    let binding_digest = |value: &Value, checked: Option<&SchemaCheckedValue>| match checked {
        Some(checked) => checked
            .canonical_digest()
            .map(str::to_owned)
            .map_err(|e| err(e.to_string())),
        None => digest(value),
    };
    let result = json!({"format":"invarlock/multi-metric-comparison-v1","decision":decision,"bindings":{"baseline":binding_digest(baseline, checked[0])?,"subject":binding_digest(subject, checked[1])?,"policy":binding_digest(policy, checked[2])?},"metrics":results,"limitations":limitations});
    schema(&result, "comparison")?;
    Ok(result)
}

#[cfg(test)]
mod seed_tests {
    use super::*;
    use invarlock_native_contracts::sha256_digest;

    fn owned_projection(pairs: &[(&Value, &Value)]) -> Value {
        Value::Array(
            pairs
                .iter()
                .map(|(left, _)| {
                    json!({
                        "id": left["id"],
                        "input": left["input"],
                        "expected": left["expected"],
                        "metadata": left["metadata"]
                    })
                })
                .collect(),
        )
    }

    fn assert_projection_parity(records: &[Value]) {
        let pairs: Vec<_> = records.iter().map(|row| (row, &Value::Null)).collect();
        let expected = canonical(&owned_projection(&pairs)).unwrap();
        let mut actual = Vec::new();
        case_seed_to_writer(&pairs, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(case_seed(&pairs).unwrap(), sha256_digest(&expected));
    }

    #[test]
    fn streamed_case_seed_preserves_projection_bytes_and_numeric_spellings() {
        let records: Vec<Value> = serde_json::from_str(
            r#"[
                {
                    "metadata": {"z": -0.0, "a": [1e-7, 1e16, 1.0, 18446744073709551616]},
                    "output": "not part of the case identity",
                    "input": {"z": "é😀\n\t\"\\", "a": {"β": false, "A": null}},
                    "id": "later",
                    "expected": [null, true, {"b": 2, "a": 1}]
                },
                {"id": "earlier", "input": "", "expected": -0, "metadata": {}},
                {}
            ]"#,
        )
        .unwrap();
        assert_projection_parity(&records);
        assert_projection_parity(&[]);
        let reversed: Vec<_> = records.into_iter().rev().collect();
        assert_projection_parity(&reversed);
    }

    #[test]
    fn streamed_case_seed_matches_owned_many_record_identity() {
        let text = "bounded fixture 😀\n".repeat(128);
        let records: Vec<_> = (0..1024)
            .map(|index| {
                json!({
                    "id": format!("record-{index}"),
                    "input": {"text": text, "index": index},
                    "expected": "reference",
                    "metadata": {"group": index % 7}
                })
            })
            .collect();
        assert_projection_parity(&records);
    }

    #[test]
    fn streamed_case_seed_preserves_canonicalization_errors() {
        let huge: Value = serde_json::from_str("1e999").unwrap();
        let row = json!({"id": "invalid", "input": huge, "expected": null, "metadata": {}});
        let pairs = [(&row, &Value::Null)];
        assert_eq!(
            case_seed(&pairs).unwrap_err(),
            digest(&owned_projection(&pairs)).unwrap_err()
        );
    }
}
