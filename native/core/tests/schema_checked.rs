use invarlock_native_contracts::{
    canonical_json, parse_run, ParsedRun, RunView, SchemaCheckedValue, Value,
};
use invarlock_native_core::{
    compare_runs, compare_validated_run_views, compare_validated_runs, AnalysisError,
    ValidatedPolicy, ValidatedRun, ValidatedRunView,
};
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

fn parsed_view(parsed: &ParsedRun) -> RunView<'_> {
    match parsed {
        ParsedRun::Compact(run) => run.view(),
        ParsedRun::Json(value) => RunView::Json(value),
    }
}
fn compact_compare(
    baseline: &Value,
    subject: &Value,
    policy: &Value,
    budget: Option<u64>,
) -> Result<Value, AnalysisError> {
    let baseline = parse_run(&canonical_json(baseline).unwrap(), "baseline").unwrap();
    let subject = parse_run(&canonical_json(subject).unwrap(), "subject").unwrap();
    assert!(matches!(baseline, ParsedRun::Compact(_)));
    assert!(matches!(subject, ParsedRun::Compact(_)));
    let baseline = ValidatedRunView::from_run_view(parsed_view(&baseline))?;
    let subject = ValidatedRunView::from_run_view(parsed_view(&subject))?;
    let policy = ValidatedPolicy::new(policy)?;
    compare_validated_run_views(&baseline, &subject, &policy, budget)
}

#[test]
fn compact_comparison_keeps_lexical_pairing_slice_order_budget_and_binding_bytes() {
    let mut baseline = artifact("baseline.json");
    let mut subject = artifact("subject-pass.json");
    for run in [&mut baseline, &mut subject] {
        run["records"][0]["id"] = json!("é");
        run["records"][1]["id"] = json!("A");
        for row in run["records"].as_array_mut().unwrap() {
            row["metadata"] = json!({"group":"all"});
        }
    }
    // Input row order differs while paired evaluation stays lexical by ID.
    subject["records"].as_array_mut().unwrap().reverse();
    let mut policy = artifact("policy.json");
    policy["slices"] = json!([
        {"name":"second","where":{"group":"all"}},
        {"name":"first","where":{"group":"all"}}
    ]);
    let expected = compare_runs(&baseline, &subject, &policy, None).unwrap();
    let actual = compact_compare(&baseline, &subject, &policy, None).unwrap();
    assert_eq!(
        canonical_json(&actual).unwrap(),
        canonical_json(&expected).unwrap()
    );
    assert_eq!(
        actual["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["slice"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["overall", "second", "first"]
    );
    assert_eq!(
        compact_compare(&baseline, &subject, &policy, Some(0)).unwrap_err(),
        compare_runs(&baseline, &subject, &policy, Some(0)).unwrap_err()
    );
    let mut rows: Vec<_> = baseline["records"].as_array().unwrap().iter().collect();
    rows.sort_by_key(|row| row["id"].as_str().unwrap());
    let cases = json!({"format":"invarlock/evaluation-case-set-v1","cases":rows.iter().map(|row|
        json!({"id":row["id"],"input":row["input"],"expected":row["expected"],"metadata":row["metadata"]})
    ).collect::<Vec<_>>()});
    policy["expected_case_set_digest"] =
        json!(invarlock_native_contracts::canonical_json_digest(&cases).unwrap());
    assert_eq!(
        canonical_json(&compact_compare(&baseline, &subject, &policy, None).unwrap()).unwrap(),
        canonical_json(&compare_runs(&baseline, &subject, &policy, None).unwrap()).unwrap()
    );
    subject["records"][0]["input"] = json!("changed");
    assert_eq!(
        compact_compare(&baseline, &subject, &policy, None).unwrap_err(),
        compare_runs(&baseline, &subject, &policy, None).unwrap_err()
    );
}

#[test]
fn compact_semantic_checks_keep_original_row_order_before_duplicate_detection() {
    for key in ["input_projection", "likelihood"] {
        let mut run = artifact("baseline.json");
        if key == "input_projection" {
            run["records"][0]["context"]["input_projection"] = json!({});
        } else {
            let row = &run["records"][0];
            run["records"][0]["likelihood"] = json!({
                "basis":"reference_continuation","logprob_sum":-1.0,"token_count":1,"utf8_byte_count":4,
                "input_digest":invarlock_native_contracts::canonical_json_digest(&row["input"]).unwrap(),
                "reference_digest":invarlock_native_contracts::canonical_json_digest(&row["expected"]).unwrap(),
                "artifact_digest":run["artifact_digest"],"source":run["source"],
                "configuration_digest":format!("sha256:{}","3".repeat(64)),
                "tokenizer_digest":format!("sha256:{}","4".repeat(64))
            });
        }
        let duplicate = run["records"][0].clone();
        run["records"].as_array_mut().unwrap().push(duplicate);
        let parsed = parse_run(&canonical_json(&run).unwrap(), "run").unwrap();
        assert!(matches!(parsed, ParsedRun::Compact(_)));
        let actual = ValidatedRunView::from_run_view(parsed_view(&parsed))
            .err()
            .unwrap();
        let expected = ValidatedRun::new(&run).err().unwrap();
        assert_eq!(actual, expected);
        assert!(!actual.message.contains("duplicate record IDs"));
    }
    let mut run = artifact("baseline.json");
    let duplicate = run["records"][0].clone();
    run["records"].as_array_mut().unwrap().push(duplicate);
    let parsed = parse_run(&canonical_json(&run).unwrap(), "run").unwrap();
    assert_eq!(
        ValidatedRunView::from_run_view(parsed_view(&parsed))
            .err()
            .unwrap(),
        ValidatedRun::new(&run).err().unwrap()
    );
}

#[test]
fn compact_missing_results_preserve_sorted_ids_and_complete_scope_preflight() {
    let mut baseline = artifact("baseline.json");
    let mut subject = artifact("subject-pass.json");
    for run in [&mut baseline, &mut subject] {
        run["records"][0]["id"] = json!("z");
        run["records"][1]["id"] = json!("a");
        for row in run["records"].as_array_mut().unwrap() {
            row["metadata"] = json!({"group":"all"});
        }
    }
    baseline["records"][0]["error"] = json!("missing");
    subject["records"][1]["scores"] = json!({});
    let mut policy = artifact("policy.json");
    policy["slices"] = json!([{"name":"overlap","where":{"group":"all"}}]);
    let expected = compare_runs(&baseline, &subject, &policy, None).unwrap();
    let actual = compact_compare(&baseline, &subject, &policy, None).unwrap();
    assert_eq!(
        canonical_json(&actual).unwrap(),
        canonical_json(&expected).unwrap()
    );
    assert_eq!(actual["metrics"][0]["missing_ids"], json!(["a", "z"]));
    assert_eq!(actual["metrics"][1]["missing_ids"], json!(["a", "z"]));
}

#[test]
fn overlapping_scope_preflight_precedes_first_metric_error() {
    let mut baseline = artifact("baseline.json");
    let mut subject = artifact("subject-pass.json");
    for run in [&mut baseline, &mut subject] {
        for row in run["records"].as_array_mut().unwrap() {
            row["metadata"] = json!({"group":"all"});
        }
    }
    baseline["records"][0]["error"] = json!("missing");
    let mut policy = artifact("policy.json");
    policy["slices"] = json!((0..16)
        .map(|index| json!({"name":format!("slice-{index}"),"where":{"group":"all"}}))
        .collect::<Vec<_>>());
    let mut first = policy["metrics"][0].clone();
    first["name"] = json!("first");
    first["accepted_provenance"]["version"] = json!("2");
    let mut second = first.clone();
    second["name"] = json!("second");
    policy["metrics"] = json!([first, second]);
    const REQUIRED: u64 = 2 * 17 * 2 * 2048;
    let budget_error =
        compact_compare(&baseline, &subject, &policy, Some(REQUIRED - 1)).unwrap_err();
    assert_eq!(budget_error.reason, "local_work_budget_exceeded");
    assert!(budget_error
        .message
        .contains(&format!("plans {REQUIRED} bootstrap draws")));
    let first_error = compact_compare(&baseline, &subject, &policy, Some(REQUIRED)).unwrap_err();
    assert_eq!(
        first_error.message,
        "metric first: scorer provenance differs from approved policy"
    );
    assert_eq!(
        first_error,
        compare_runs(&baseline, &subject, &policy, Some(REQUIRED)).unwrap_err()
    );
}

#[test]
fn compact_likelihood_preserves_nll_bytes_and_large_exact_count_semantics() {
    let configuration = format!("sha256:{}", "3".repeat(64));
    let tokenizer = format!("sha256:{}", "4".repeat(64));
    let mut baseline = artifact("baseline.json");
    let mut subject = artifact("subject-pass.json");
    let huge: Value = serde_json::from_str("340282366920938463463374607431768211457").unwrap();
    for (side, run) in [(0, &mut baseline), (1, &mut subject)] {
        for index in 0..2 {
            let row = &run["records"][index];
            let facts = json!({"basis":"reference_continuation", "logprob_sum":-((index+1)as f64)*(if side==0{3.0}else{2.97}),
                "token_count":huge,"utf8_byte_count":3,
                "input_digest":invarlock_native_contracts::canonical_json_digest(&row["input"]).unwrap(),
                "reference_digest":invarlock_native_contracts::canonical_json_digest(&row["expected"]).unwrap(),
                "artifact_digest":run["artifact_digest"],"source":run["source"],
                "configuration_digest":configuration,"tokenizer_digest":tokenizer});
            run["records"][index]["likelihood"] = facts;
        }
    }
    let policy = json!({"format":"invarlock/comparison-policy-v1","slices":[],
        "metrics":[{"name":"nll","kind":"normalized_nll_per_utf8_byte",
            "configuration":{"configuration_digest":configuration,"baseline_tokenizer_digest":tokenizer,"subject_tokenizer_digest":tokenizer},
            "direction":"lower","unit":"nats_per_utf8_byte","aggregation":"mean","minimum_count":2,"ratio_max":1.1,"maximum_interval_width":0.5}]});
    let expected = compare_runs(&baseline, &subject, &policy, None).unwrap();
    let actual = compact_compare(&baseline, &subject, &policy, None).unwrap();
    assert_eq!(
        canonical_json(&actual).unwrap(),
        canonical_json(&expected).unwrap()
    );
    baseline["records"][0]["likelihood"]["utf8_byte_count"] = huge;
    let parsed = parse_run(&canonical_json(&baseline).unwrap(), "run").unwrap();
    let error = ValidatedRunView::from_run_view(parsed_view(&parsed))
        .err()
        .unwrap();
    assert!(error
        .message
        .ends_with("utf8_byte_count differs from reference"));
    assert_eq!(error, ValidatedRun::new(&baseline).err().unwrap());
}
