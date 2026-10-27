use invarlock_native_contracts::{parse_json, Value};
use serde_json::json;

#[test]
fn compact_objects_preserve_field_order_at_the_buffer_boundary() {
    for count in [0, 1, 2, 3, 4, 7, 8, 14, 16, 31, 32, 33, 65] {
        let fields = (0..count)
            .map(|i| format!(r#""field_{i:02}":{{"text":"é😀","array":[1,2,3]}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let raw = format!("{{{fields}}}");
        let value = parse_json(raw.as_bytes(), "input").unwrap();
        let ordinary: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value, ordinary);
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            ordinary.as_object().unwrap().keys().collect::<Vec<_>>()
        );
        for child in value.as_object().unwrap().values() {
            let array = child["array"].as_array().unwrap();
            assert_eq!(array.capacity(), array.len());
        }
    }
}

#[test]
fn buffered_and_wide_duplicates_keep_nested_and_syntax_error_precedence() {
    for count in [0, 31, 32, 33, 65] {
        let prefix = (0..count)
            .map(|i| format!(r#""field_{i}":0,"#))
            .collect::<String>();
        for (suffix, code, message) in [
            (
                r#""z":0,"z":1,"a":0,"a":1}"#,
                "duplicate_key",
                "JSON object has duplicate key 'z'",
            ),
            (
                r#""a":0,"\u0061":1}"#,
                "duplicate_key",
                "JSON object has duplicate key 'a'",
            ),
            (
                r#""a":0,"a":1,"child":{"z":0,"z":1}}"#,
                "duplicate_key",
                "JSON object has duplicate key 'z'",
            ),
            (
                r#""a":0,"a":1,"child":NaN}"#,
                "nonstandard_constant",
                "JSON contains non-standard constant 'NaN'",
            ),
            (
                r#""a":0,"a":1,}"#,
                "invalid_json",
                "input is not valid JSON",
            ),
        ] {
            let raw = format!("{{{prefix}{suffix}");
            let error = parse_json(raw.as_bytes(), "input").unwrap_err();
            assert_eq!(error.code, code, "{raw}");
            assert_eq!(error.message, message, "{raw}");
        }
    }
}

#[test]
fn string_fast_path_preserves_utf8_escape_and_control_validation() {
    for raw in [
        r#""""#,
        r#""plain é 😀""#,
        r#""escaped \\ \" \n \t \u0061 \ud83d\ude00""#,
        r#""literal \\ud800""#,
    ] {
        let expected: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(parse_json(raw.as_bytes(), "input").unwrap(), expected);
    }
    assert_eq!(
        parse_json(br#""a\u0000b""#, "input").unwrap(),
        json!("a\0b")
    );
    assert_eq!(
        parse_json(b"\"a\0b\"", "input").unwrap_err().code,
        "invalid_json"
    );
    assert_eq!(
        parse_json(br#""\ud800""#, "input").unwrap_err().code,
        "unpaired_surrogate"
    );
    assert_eq!(
        parse_json(br#""\x00""#, "input").unwrap_err().code,
        "invalid_json"
    );
}
