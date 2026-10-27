use invarlock_native_contracts::{canonical_json, SchemaCheckedValue, Value};
use invarlock_native_core::{compare_runs, compare_validated_runs, ValidatedPolicy, ValidatedRun};
use serde_json::json;

fn artifact(name: &str) -> Value {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/evaluation_contracts/captured-v2.json"
    ))
    .unwrap();
    serde_json::from_str(corpus["artifacts"][name]["bytes"].as_str().unwrap()).unwrap()
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
