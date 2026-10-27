use invarlock_native_contracts::{
    canonical_json, canonical_json_digest, canonical_json_matches, canonical_json_no_newline,
    canonical_json_to_writer, sha256_digest, Value,
};
use serde_json::json;

#[test]
fn streaming_canonical_golden_bytes_and_digest() {
    let value: Value = serde_json::from_str(
        r#"{"😀":"\u0000\n\t\"\\","\ue000":1e16,"é":0.0001,"a":[-0,-0.0,1e-5,1e15,123456789012345678901234567890],"":true}"#,
    )
    .unwrap();
    let expected = "{\"\":true,\"a\":[0,-0.0,1e-05,1000000000000000.0,123456789012345678901234567890],\"é\":0.0001,\"\u{e000}\":1e+16,\"😀\":\"\\u0000\\n\\t\\\"\\\\\"}\n";
    assert_eq!(canonical_json(&value).unwrap(), expected.as_bytes());
    assert!(canonical_json_matches(&value, expected.as_bytes()).unwrap());
    assert_eq!(
        canonical_json_digest(&value).unwrap(),
        sha256_digest(expected.as_bytes())
    );
    assert_eq!(
        canonical_json_no_newline(&value).unwrap(),
        expected.as_bytes()[..expected.len() - 1]
    );
}

#[test]
fn streaming_equality_checks_every_byte_and_length() {
    let value = json!({"a": [true, null, "é"]});
    let canonical = b"{\"a\":[true,null,\"\xc3\xa9\"]}\n";
    assert!(canonical_json_matches(&value, canonical).unwrap());
    for index in 0..canonical.len() {
        let mut altered = canonical.to_vec();
        altered[index] ^= 1;
        assert!(
            !canonical_json_matches(&value, &altered).unwrap(),
            "byte {index}"
        );
    }
    for length in 0..canonical.len() {
        assert!(!canonical_json_matches(&value, &canonical[..length]).unwrap());
    }
    let mut trailing = canonical.to_vec();
    trailing.push(b' ');
    assert!(!canonical_json_matches(&value, &trailing).unwrap());
}

#[test]
fn early_byte_mismatch_still_reports_late_number_error() {
    let invalid: Value = serde_json::from_str("1e400").unwrap();
    let value = json!({"a": "first", "z": [0, invalid]});
    let expected = canonical_json(&value).unwrap_err();
    assert_eq!(expected.code, "nonfinite_number");
    for raw in [b"".as_slice(), b"different", b"{\"a\":\"first\",\"z\":[0,"] {
        assert_eq!(canonical_json_matches(&value, raw).unwrap_err(), expected);
    }
    assert_eq!(canonical_json_digest(&value).unwrap_err(), expected);
}

#[test]
fn wide_array_streams_exactly_across_digest_buffer_boundaries() {
    let value = Value::Array((0..100_000).map(|i| json!(i)).collect());
    let mut expected = String::from("[");
    for i in 0..100_000 {
        if i != 0 {
            expected.push(',');
        }
        expected.push_str(&i.to_string());
    }
    expected.push_str("]\n");
    assert!(canonical_json_matches(&value, expected.as_bytes()).unwrap());
    assert_eq!(
        canonical_json_digest(&value).unwrap(),
        sha256_digest(expected.as_bytes())
    );
    let mut written = Vec::new();
    canonical_json_to_writer(&value, &mut written, true).unwrap();
    assert_eq!(written, expected.as_bytes());
}

#[test]
fn deep_objects_preserve_order_and_iterative_traversal() {
    const DEPTH: usize = 700;
    let mut value = json!("leaf");
    for _ in 0..DEPTH {
        value = json!({"z": false, "a": value});
    }
    let expected = format!(
        "{}\"leaf\"{}\n",
        "{\"a\":".repeat(DEPTH),
        ",\"z\":false}".repeat(DEPTH)
    );
    assert!(canonical_json_matches(&value, expected.as_bytes()).unwrap());
    assert_eq!(
        canonical_json_digest(&value).unwrap(),
        sha256_digest(expected.as_bytes())
    );
    assert_eq!(canonical_json(&value).unwrap(), expected.as_bytes());
}
