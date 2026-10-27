use invarlock_native_contracts::{canonical_json, SchemaCheckedValue, Value};
use invarlock_native_core::{compare_runs, compare_validated_runs, ValidatedPolicy, ValidatedRun};
use serde_json::json;

fn artifact(name: &str) -> Value {
    match name {
        "baseline.json" | "subject-pass.json" => {
            let subject = name == "subject-pass.json";
            let provenance = json!({"kind":"measurement","rubric_digest":null,
                "source":"reviewed-meter","unit":"points","version":"1"});
            json!({
                "format":"invarlock/evaluation-run-v1",
                "artifact_digest":format!("sha256:{}", if subject { "2" } else { "1" }.repeat(64)),
                "run_id":if subject {"subject"} else {"baseline"},
                "source":{"name":"reviewed-export","version":"1"},"source_digest":null,
                "score_provenance":{"quality":provenance},
                "records":[
                    {"id":"a","input":"first","expected":"yes","output":"yes","metadata":{},
                        "context":{},"error":null,"scores":{"quality":if subject {3} else {2}}},
                    {"id":"b","input":"second","expected":"yes","output":"yes","metadata":{},
                        "context":{},"error":null,"scores":{"quality":if subject {5} else {4}}}
                ]
            })
        }
        "policy.json" => json!({
            "format":"invarlock/comparison-policy-v1","slices":[],
            "metrics":[{"name":"quality","kind":"recorded","score_key":"quality",
                "configuration":{},"direction":"higher","unit":"points","aggregation":"mean",
                "minimum_count":2,"maximum_regression":0,"maximum_interval_width":0.5,
                "accepted_provenance":{"kind":"measurement","rubric_digest":null,
                    "source":"reviewed-meter","unit":"points","version":"1"}}]
        }),
        "comparison-pass.json" => json!({
            "format":"invarlock/multi-metric-comparison-v1","decision":"pass",
            "bindings":{
                "baseline":"sha256:e6726ae30d25ba44c44308219a917614ebfeccb53b7ced4eb814fa529be4179a",
                "subject":"sha256:f109b400703450ce2fa947935e36221756385d73389b75ce3642e565a7ec35e9",
                "policy":"sha256:f07ed71980fb1fd9cfee05bd55cdab1b7c73fadf7be911af26ce9c8f387a4a43"},
            "limitations":[
                "Recorded scores retain source judgments; only their aggregation is recomputed.",
                "Intervals describe the paired schedule under the stated method; representativeness requires an external sampling design.",
                "Metric and slice intervals are marginal, not a simultaneous family-wide confidence guarantee.",
                "A policy pass does not establish truthful model execution, general quality, safety or compliance."],
            "metrics":[{"name":"quality","kind":"recorded","slice":"overall",
                "direction":"higher","unit":"points","aggregation":"mean","count":2,
                "baseline_mean":3.0,"subject_mean":4.0,"delta":1.0,"decision":"pass",
                "missing_ids":[],"reasons":[],"scoring_assurance":"recorded",
                "interval":{"lower":1.0,"upper":1.0,"mass":0.95,
                    "method":"paired_mean_shake256_percentile_v1","replicates":2048}}]
        }),
        _ => panic!("unknown test artifact: {name}"),
    }
}

#[test]
fn schema_evidence_does_not_bypass_run_or_policy_semantics() {
    let mut run = artifact("baseline.json");
    let duplicate = run["records"][0].clone();
    run["records"].as_array_mut().unwrap().push(duplicate);
    let checked = SchemaCheckedValue::new(run.clone(), "run").unwrap();
    let ordinary = ValidatedRun::new(&run).err().unwrap();
    let reused = ValidatedRun::from_schema_checked(&checked).err().unwrap();
    assert_eq!(ordinary, reused);
    assert!(reused.message.contains("duplicate record IDs"));

    let mut policy = artifact("policy.json");
    let duplicate = policy["metrics"][0].clone();
    policy["metrics"].as_array_mut().unwrap().push(duplicate);
    let checked = SchemaCheckedValue::new(policy.clone(), "policy").unwrap();
    assert_eq!(
        ValidatedPolicy::new(&policy).err().unwrap(),
        ValidatedPolicy::from_schema_checked(&checked)
            .err()
            .unwrap()
    );
    assert_eq!(
        ValidatedRun::new(&policy).err().unwrap(),
        ValidatedRun::from_schema_checked(&checked).err().unwrap()
    );
}

#[test]
fn immutable_values_and_cached_bindings_preserve_comparison_bytes() {
    let baseline = SchemaCheckedValue::new(artifact("baseline.json"), "run").unwrap();
    let subject = SchemaCheckedValue::new(artifact("subject-pass.json"), "run").unwrap();
    let policy = SchemaCheckedValue::new(artifact("policy.json"), "policy").unwrap();
    let digest = baseline.canonical_digest().unwrap().to_owned();
    let mut caller_copy = baseline.value().clone();
    caller_copy["records"] = Value::Null;
    assert!(SchemaCheckedValue::new(caller_copy, "run").is_err());
    assert_eq!(baseline.canonical_digest().unwrap(), digest);
    let ordinary = compare_runs(baseline.value(), subject.value(), policy.value(), None).unwrap();
    let reused = compare_validated_runs(
        &ValidatedRun::from_schema_checked(&baseline).unwrap(),
        &ValidatedRun::from_schema_checked(&subject).unwrap(),
        &ValidatedPolicy::from_schema_checked(&policy).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        canonical_json(&ordinary).unwrap(),
        canonical_json(&reused).unwrap()
    );
    assert_eq!(
        canonical_json(&reused).unwrap(),
        canonical_json(&artifact("comparison-pass.json")).unwrap()
    );
    let mut consumed = baseline.into_value();
    consumed["records"] = Value::Null;
    assert!(SchemaCheckedValue::new(consumed, "run").is_err());
}

#[test]
fn reused_schema_still_checks_projection_and_likelihood_before_duplicate_ids() {
    let original = artifact("baseline.json");
    let mut projected = original.clone();
    projected["records"][0]["context"]["input_projection"] = json!({});
    let mut likelihood = original;
    let row = &likelihood["records"][0];
    let facts = json!({
        "basis":"reference_continuation", "logprob_sum":-1.0,
        "token_count":1,"utf8_byte_count":4,
        "input_digest":invarlock_native_contracts::canonical_json_digest(&row["input"]).unwrap(),
        "reference_digest":invarlock_native_contracts::canonical_json_digest(&row["expected"]).unwrap(),
        "artifact_digest":likelihood["artifact_digest"],
        "configuration_digest":invarlock_native_contracts::canonical_json_digest(&json!({})).unwrap(),
        "tokenizer_digest":invarlock_native_contracts::canonical_json_digest(&json!("tokenizer")).unwrap(),
        "source":likelihood["source"]
    });
    likelihood["records"][0]["likelihood"] = facts;
    for (mut run, expected) in [
        (projected, "input projection binding has invalid fields"),
        (likelihood, "utf8_byte_count differs from reference"),
    ] {
        let duplicate = run["records"][0].clone();
        run["records"].as_array_mut().unwrap().push(duplicate);
        let checked = SchemaCheckedValue::new(run.clone(), "run").unwrap();
        let error = ValidatedRun::from_schema_checked(&checked).err().unwrap();
        assert_eq!(error, ValidatedRun::new(&run).err().unwrap());
        assert!(error.message.contains(expected), "{}", error.message);
    }
}

#[test]
fn planned_case_digest_preserves_matching_outputs_and_side_error_order() {
    let baseline = artifact("baseline.json");
    let mut subject = artifact("subject-pass.json");
    let mut policy = artifact("policy.json");
    let mut rows: Vec<_> = baseline["records"].as_array().unwrap().iter().collect();
    rows.sort_by_key(|row| row["id"].as_str().unwrap());
    let cases = json!({"format":"invarlock/evaluation-case-set-v1","cases":rows.iter().map(|row|
        json!({"id":row["id"],"input":row["input"],"expected":row["expected"],"metadata":row["metadata"]})
    ).collect::<Vec<_>>()});
    policy["expected_case_set_digest"] =
        json!(invarlock_native_contracts::canonical_json_digest(&cases).unwrap());
    let ordinary = compare_runs(&baseline, &subject, &policy, None).unwrap();
    assert_eq!(ordinary["decision"], "pass");
    let checked_baseline = SchemaCheckedValue::new(baseline.clone(), "run").unwrap();
    let checked_subject = SchemaCheckedValue::new(subject.clone(), "run").unwrap();
    let checked_policy = SchemaCheckedValue::new(policy.clone(), "policy").unwrap();
    let reused = compare_validated_runs(
        &ValidatedRun::from_schema_checked(&checked_baseline).unwrap(),
        &ValidatedRun::from_schema_checked(&checked_subject).unwrap(),
        &ValidatedPolicy::from_schema_checked(&checked_policy).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        canonical_json(&ordinary).unwrap(),
        canonical_json(&reused).unwrap()
    );
    subject["records"][0]["input"] = json!("changed");
    assert_eq!(
        compare_runs(&baseline, &subject, &policy, None)
            .unwrap_err()
            .message,
        "run 'subject' does not match the planned case set"
    );
    let mut baseline = baseline;
    baseline["records"][0]["input"] = json!("also changed");
    assert_eq!(
        compare_runs(&baseline, &subject, &policy, None)
            .unwrap_err()
            .message,
        "run 'baseline' does not match the planned case set"
    );
}
