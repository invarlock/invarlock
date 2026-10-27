use invarlock_native_contracts::{
    canonical_json, canonical_json_digest, parse_json, parse_run, validate_schema, ParsedRun,
    RecordRef, RunView, Value,
};
use serde_json::json;
fn row() -> Value {
    json!({"id":"one","input":{"z":[null,true,18446744073709551616_u128],"a":"é😀"},"expected":"yes",
        "output":null,"scores":{},"metadata":{"group":"all"},"error":null,"context":{}})
}
fn run(rows: Vec<Value>) -> Value {
    json!({"format":"invarlock/evaluation-run-v1","run_id":"run","artifact_digest":format!("sha256:{}","a".repeat(64)),
        "source_digest":null,"source":{"name":"fixture","version":"1"},"score_provenance":{},"records":rows})
}
#[test]
fn compact_run_preserves_all_fields_numbers_and_canonical_identity() {
    let mut row = row();
    row["context"] = serde_json::from_str(
        r#"{"numbers":[-0.0,1e-7,1e16,123456789012345678901234567890],"nested":{"records":[1,2]}}"#,
    )
    .unwrap();
    let value = run(vec![row]);
    let raw = canonical_json(&value).unwrap();
    let ParsedRun::Compact(compact) = parse_run(&raw, "run").unwrap() else {
        panic!("canonical valid run must use compact ownership")
    };
    assert_eq!(compact.schema(), "run");
    assert_eq!(
        compact.canonical_digest().unwrap(),
        canonical_json_digest(&value).unwrap()
    );
    let view = compact.view();
    assert_eq!(view.record_count(), 1);
    assert_eq!(view.records().len(), 1);
    assert_eq!(view.record(0).id(), "one");
    assert_eq!(view.record(0).get("likelihood"), None);
    assert_eq!(view.record(0)["absent"], Value::Null);
    assert!(matches!(view.record(0), RecordRef::Compact(_)));
    assert!(view.canonical_matches(&raw).unwrap());
    let mut bytes = Vec::new();
    view.canonical_to_writer(&mut bytes, true).unwrap();
    assert_eq!(bytes, raw);
    let ordinary = parse_json(&raw, "run").unwrap();
    let rebuilt = view.to_value();
    assert_eq!(
        serde_json::to_vec(&rebuilt).unwrap(),
        serde_json::to_vec(&ordinary).unwrap()
    );
    assert_eq!(
        rebuilt.as_object().unwrap().keys().collect::<Vec<_>>(),
        ordinary.as_object().unwrap().keys().collect::<Vec<_>>()
    );
    assert_eq!(
        rebuilt["records"][0]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ordinary["records"][0]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>()
    );
    let consumed = compact.into_value();
    assert_eq!(
        serde_json::to_vec(&consumed).unwrap(),
        serde_json::to_vec(&ordinary).unwrap()
    );
    assert_eq!(canonical_json(&consumed).unwrap(), raw);
}
#[test]
fn generic_fallback_keeps_schema_noncanonical_and_syntax_diagnostics() {
    let valid = run(vec![row()]);
    let canonical = canonical_json(&valid).unwrap();
    let pretty = serde_json::to_vec_pretty(&valid).unwrap();
    for raw in [
        pretty.as_slice(),
        &canonical[..canonical.len() - 1],
        b"{}",
        b"[]",
        b"null",
    ] {
        let ParsedRun::Json(value) = parse_run(raw, "input").unwrap() else {
            panic!("unsupported/noncanonical input must retain generic validation")
        };
        assert_eq!(value, parse_json(raw, "input").unwrap());
    }
    for key in [
        "id", "input", "expected", "output", "scores", "metadata", "error", "context",
    ] {
        let mut invalid = valid.clone();
        invalid["records"][0].as_object_mut().unwrap().remove(key);
        let raw = canonical_json(&invalid).unwrap();
        let ParsedRun::Json(value) = parse_run(&raw, "input").unwrap() else {
            panic!("missing required field {key}")
        };
        assert_eq!(
            validate_schema(&value, "run").unwrap_err(),
            validate_schema(&invalid, "run").unwrap_err()
        );
    }
    for raw in [
        br#"{"records":[{"id":"a","id":"b","child":NaN}]}"#.as_slice(),
        br#"{"records":[{"id":"a","id":"b","child":{"z":0,"z":1}}]}"#,
        br#"{"records":[{}],"records":[],"child":NaN}"#,
        br#"{"records":[{}],"child":"\ud800"}"#,
        br#"{"records":[{}],}"#,
        b"\xff",
    ] {
        assert_eq!(
            parse_run(raw, "input").unwrap_err(),
            parse_json(raw, "input").unwrap_err()
        );
    }
}
#[test]
fn direct_row_parser_keeps_the_original_container_depth_boundary() {
    for nested in [894, 895, 896, 897, 898, 899, 900, 901] {
        let mut value = run(vec![row()]);
        let marker = "depth_marker";
        value["records"][0]["input"] = json!(marker);
        let raw = String::from_utf8(canonical_json(&value).unwrap())
            .unwrap()
            .replace(
                &format!("\"{marker}\""),
                &format!("{}0{}", "[".repeat(nested), "]".repeat(nested)),
            );
        match parse_json(raw.as_bytes(), "depth") {
            Ok(value) => {
                let parsed = parse_run(raw.as_bytes(), "depth").unwrap();
                let restored = match parsed {
                    ParsedRun::Compact(run) => run.into_value(),
                    ParsedRun::Json(value) => value,
                };
                assert_eq!(
                    canonical_json(&restored).unwrap(),
                    canonical_json(&value).unwrap()
                );
            }
            Err(error) => assert_eq!(parse_run(raw.as_bytes(), "depth").unwrap_err(), error),
        }
    }
}
#[test]
fn compact_and_generic_views_have_the_same_row_and_header_access() {
    let mut likelihood = row();
    likelihood["likelihood"] = json!({"basis":"reference_continuation","logprob_sum":-1.0,"token_count":1,
        "utf8_byte_count":3,"input_digest":format!("sha256:{}","1".repeat(64)),"reference_digest":format!("sha256:{}","2".repeat(64)),
        "artifact_digest":format!("sha256:{}","a".repeat(64)),"configuration_digest":format!("sha256:{}","3".repeat(64)),
        "tokenizer_digest":format!("sha256:{}","4".repeat(64)),"source":{"name":"fixture","version":"1"}});
    let value = run(vec![likelihood, row()]);
    let raw = canonical_json(&value).unwrap();
    let ParsedRun::Compact(compact) = parse_run(&raw, "run").unwrap() else {
        panic!("valid likelihood row")
    };
    let legacy = RunView::Json(&value);
    let compact = compact.view();
    for key in [
        "format",
        "run_id",
        "artifact_digest",
        "source_digest",
        "source",
        "score_provenance",
        "absent",
    ] {
        assert_eq!(legacy.get(key), compact.get(key));
        assert_eq!(legacy[key], compact[key]);
    }
    for (left, right) in legacy.records().zip(compact.records()) {
        for key in [
            "id",
            "input",
            "expected",
            "output",
            "scores",
            "metadata",
            "error",
            "context",
            "likelihood",
            "absent",
        ] {
            assert_eq!(left.get(key), right.get(key));
            assert_eq!(left[key], right[key]);
        }
        let mut bytes = Vec::new();
        right.canonical_to_writer(&mut bytes, true).unwrap();
        assert_eq!(bytes, canonical_json(&left.to_value()).unwrap());
    }
}
#[test]
fn wide_run_compacts_rows_without_retaining_a_generic_record_array() {
    let rows = (0..1024)
        .map(|index| {
            let mut row = row();
            row["id"] = json!(format!("row-{index}"));
            row
        })
        .collect();
    let value = run(rows);
    let raw = canonical_json(&value).unwrap();
    let ParsedRun::Compact(compact) = parse_run(&raw, "run").unwrap() else {
        panic!("wide canonical run")
    };
    assert_eq!(compact.view().record_count(), 1024);
    assert!(compact
        .view()
        .records()
        .all(|row| matches!(row, RecordRef::Compact(_))));
    assert_eq!(
        compact.canonical_digest().unwrap(),
        canonical_json_digest(&value).unwrap()
    );
}

#[test]
fn complete_header_conditions_are_certified_or_fall_back_without_new_errors() {
    let mut value = run(vec![row()]);
    let digest = invarlock_native_contracts::sha256_digest(b"configuration");
    value["artifact_digest"] = Value::Null;
    value["service_identity"] = json!({"kind":"hosted_service","provider":"provider","service":"service",
        "deployment":"deployment","requested_model":"model","observed_model":null,"exposed_revision":null,
        "configuration":{},"configuration_digest":digest,"harness":{"name":"harness","version":"1","source_digest":digest},
        "observation_window":{"started_at":"2026-01-01T00:00:00Z","ended_at":"2026-01-01T01:00:00Z"}});
    validate_schema(&value, "run").unwrap();
    let raw = canonical_json(&value).unwrap();
    let ParsedRun::Compact(compact) = parse_run(&raw, "run").unwrap() else {
        panic!("complete hosted-service header")
    };
    assert_eq!(
        serde_json::to_vec(&compact.into_value()).unwrap(),
        serde_json::to_vec(&parse_json(&raw, "run").unwrap()).unwrap()
    );
    for (key, invalid) in [
        ("artifact_digest", json!(digest)),
        ("format", json!("wrong")),
        ("unknown", json!(true)),
    ] {
        let mut invalid_run = value.clone();
        invalid_run[key] = invalid;
        let raw = canonical_json(&invalid_run).unwrap();
        let ParsedRun::Json(parsed) = parse_run(&raw, "run").unwrap() else {
            panic!("invalid header {key} must retain generic diagnostics")
        };
        assert_eq!(
            validate_schema(&parsed, "run").unwrap_err(),
            validate_schema(&invalid_run, "run").unwrap_err()
        );
    }
}

#[test]
fn invalid_likelihood_shapes_keep_full_generic_schema_diagnostics() {
    for facts in [json!({}), json!({"unexpected":true}), Value::Null] {
        let mut row = row();
        row["likelihood"] = facts;
        let value = run(vec![row]);
        let raw = canonical_json(&value).unwrap();
        let ParsedRun::Json(fallback) = parse_run(&raw, "run").unwrap() else {
            panic!("invalid likelihood must retain original diagnostics")
        };
        assert_eq!(
            validate_schema(&fallback, "run").unwrap_err(),
            validate_schema(&value, "run").unwrap_err()
        );
    }
}

#[test]
fn large_nested_header_keeps_compact_identity_and_generic_failure_semantics() {
    let configuration = json!({"nested":[
        {"payload":"x".repeat(256 * 1024), "unicode":"é😀"},
        {"values":[null, true, {"deeper":[1,2,3]}]}
    ]});
    let digest = canonical_json_digest(&configuration).unwrap();
    let mut value = run(vec![row()]);
    value["artifact_digest"] = Value::Null;
    value["service_identity"] = json!({"kind":"hosted_service", "provider":"provider", "service":"service",
        "deployment":"deployment", "requested_model":"model", "observed_model":null, "exposed_revision":null,
        "configuration":configuration, "configuration_digest":digest,
        "harness":{"name":"harness", "version":"1", "source_digest":digest},
        "observation_window":{"started_at":"2026-01-01T00:00:00Z", "ended_at":"2026-01-01T01:00:00Z"}});
    validate_schema(&value, "run").unwrap();
    let raw = canonical_json(&value).unwrap();
    let ordinary = parse_json(&raw, "large header").unwrap();
    let ParsedRun::Compact(compact) = parse_run(&raw, "large header").unwrap() else {
        panic!("large valid nested header must remain compact")
    };
    assert_eq!(
        compact.canonical_digest().unwrap(),
        canonical_json_digest(&ordinary).unwrap()
    );
    assert_eq!(
        compact.view()["service_identity"],
        ordinary["service_identity"]
    );
    let mut encoded = Vec::new();
    compact
        .view()
        .canonical_to_writer(&mut encoded, true)
        .unwrap();
    assert_eq!(encoded, raw);
    assert_eq!(
        serde_json::to_vec(&compact.into_value()).unwrap(),
        serde_json::to_vec(&ordinary).unwrap()
    );

    value["unsupported_header"] = configuration;
    let raw = canonical_json(&value).unwrap();
    let ordinary = parse_json(&raw, "large header").unwrap();
    let ParsedRun::Json(fallback) = parse_run(&raw, "large header").unwrap() else {
        panic!("unsupported large header must retain generic validation")
    };
    assert_eq!(
        serde_json::to_vec(&fallback).unwrap(),
        serde_json::to_vec(&ordinary).unwrap()
    );
    assert_eq!(
        validate_schema(&fallback, "run").unwrap_err(),
        validate_schema(&ordinary, "run").unwrap_err()
    );
}
