use invarlock_native_contracts::*;
use serde_json::json;
#[test]
fn strict_duplicates_constants_and_utf8() {
    for (raw, code) in [
        (br#"{"a":1,"\u0061":2}"#.as_slice(), "duplicate_key"),
        (b"NaN", "nonstandard_constant"),
        (b"1e400", "nonfinite_number"),
        (b"\xff", "invalid_utf8"),
        (b"[1,]", "invalid_json"),
        (br#""\ud800""#, "unpaired_surrogate"),
    ] {
        assert_eq!(parse_json(raw, "input").unwrap_err().code, code);
    }
    assert_eq!(
        parse_json(br#"{"a":1,"a":2,"z":NaN}"#, "input")
            .unwrap_err()
            .code,
        "nonstandard_constant"
    );
}
#[test]
fn scalar_types_and_canonical_profile() {
    let raw=br#"{"z":-0,"float":-0.0,"big":123456789012345678901234567890,"small":1e-7,"integer_float":1e3,"unicode":"\ud83d\ude00"}"#;
    let value = parse_json(raw, "input").unwrap();
    let expected="{\"big\":123456789012345678901234567890,\"float\":-0.0,\"integer_float\":1000.0,\"small\":1e-07,\"unicode\":\"😀\",\"z\":0}\n";
    assert_eq!(canonical_json(&value).unwrap(), expected.as_bytes());
    assert_eq!(
        canonical_json_no_newline(&value).unwrap(),
        expected.trim_end().as_bytes()
    );
}
#[test]
fn snapshot_binds_one_immutable_parse() {
    let raw=br#"{"format":"invarlock/evaluation-case-set-v1","cases":[{"id":"one","input":"q","expected":"a","metadata":{}}]}"#;
    let parsed = ParsedDocument::parse(raw.as_slice(), "case set").unwrap();
    let validated = parsed.validate("case_set").unwrap();
    assert_eq!(validated.digest(), sha256_digest(raw));
    assert_eq!(validated.value(), parsed.value());
    assert!(parsed.require_canonical().is_err());
    let canonical = canonical_json(parsed.value()).unwrap();
    ParsedDocument::parse(canonical, "case set")
        .unwrap()
        .require_canonical()
        .unwrap();
}
#[test]
fn all_packaged_schemas_compile_and_reject_empty_input() {
    let names: Vec<_> = schema_names().collect();
    assert!(names.len() >= 40);
    for name in names {
        let error = validate_schema(&json!({}), name).unwrap_err();
        assert_eq!(error.code, "schema_validation", "{name}: {error}");
    }
}
#[test]
fn closed_contract_and_record_preflight() {
    assert_eq!(
        validate_schema(&json!({}), "run").unwrap_err().message,
        "invalid run : 'format' is a required property"
    );
    let value = json!({"records":vec![Value::Null; MAX_RECORDS+1]});
    assert_eq!(
        validate_schema(&value, "run").unwrap_err().code,
        "record_limit"
    );
    assert_eq!(
        validate_schema(&json!({}), "missing").unwrap_err().code,
        "unknown_schema"
    );
}
#[test]
fn native_parser_bounds_are_explicit() {
    let raw = format!("{}0{}", "[".repeat(901), "]".repeat(901));
    assert_eq!(
        parse_json(raw.as_bytes(), "input").unwrap_err().code,
        "json_depth"
    );
    assert!(parse_json("9".repeat(4300).as_bytes(), "input").is_ok());
    assert!(parse_json("9".repeat(4301).as_bytes(), "input").is_err());
}

#[test]
fn runtime_collection_and_typed_scalar_contracts() {
    use invarlock_native_contracts::{
        first_schema_error_fragment, parse_json, runtime, SchemaErrorOrder,
    };
    use serde_json::json;
    let plan =
        json!({"schedule":{"expected_trials":2},"judge":{"config":{"max_output_tokens":128}}});
    let valid = json!({"profile":"runtime-provider-text-frozen-answer-v1","max_calls":2,"max_output_tokens":256});
    assert_eq!(
        runtime::validate_runtime_provider_collection(&valid, &plan).unwrap(),
        json!({"max_calls":2,"max_output_tokens":256})
    );
    for value in [json!(true), json!(2.0), json!(0), json!(200001)] {
        let mut config = valid.clone();
        config["max_calls"] = value;
        assert_eq!(
            runtime::validate_runtime_provider_collection(&config, &plan)
                .unwrap_err()
                .message,
            "runtime judge max_calls must be a bounded positive integer"
        );
    }
    let mut insufficient = valid;
    insufficient["max_output_tokens"] = json!(255);
    assert_eq!(
        runtime::validate_runtime_provider_collection(&insufficient, &plan)
            .unwrap_err()
            .message,
        "runtime judge collection must reserve every planned inference"
    );
    let mut spec = json!({"provider_name":"llama_cpp","model_id":"model","settings":{}});
    let raw = format!("1{}", "0".repeat(400));
    spec["settings"]["integer"] = parse_json(raw.as_bytes(), "integer").unwrap();
    runtime::validate_spec(&spec).unwrap();
    spec["model_id"] = json!(" model");
    assert_eq!(
        runtime::validate_spec(&spec).unwrap_err().message,
        "model_id must be a non-empty trimmed string"
    );
    for _ in 0..2 {
        assert!(first_schema_error_fragment(
            &json!({}),
            "judge_measurements",
            "/properties/trials/items",
            SchemaErrorOrder::Schema
        )
        .unwrap()
        .is_some());
    }
}

#[test]
fn arbitrary_integer_schema_keywords_do_not_narrow_or_panic() {
    use invarlock_native_contracts::{first_schema_error_value, parse_json, SchemaErrorOrder};
    use serde_json::json;
    let huge = parse_json(format!("1{}", "0".repeat(400)).as_bytes(), "integer").unwrap();
    for schema in [
        json!({"type":"integer"}),
        json!({"minimum":0}),
        json!({"exclusiveMinimum":0}),
    ] {
        assert!(
            first_schema_error_value(&huge, &schema, SchemaErrorOrder::Schema)
                .unwrap()
                .is_none()
        );
    }
    for schema in [
        json!({"maximum":1}),
        json!({"exclusiveMaximum":1}),
        json!({"const":1}),
        json!({"enum":[1,true]}),
    ] {
        assert!(
            first_schema_error_value(&huge, &schema, SchemaErrorOrder::Schema)
                .unwrap()
                .is_some()
        );
    }
    assert!(first_schema_error_value(
        &json!([huge.clone(), huge]),
        &json!({"uniqueItems":true}),
        SchemaErrorOrder::Schema
    )
    .unwrap()
    .is_some());
}
