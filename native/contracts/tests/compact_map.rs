use serde_json::{json, Map, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
#[test]
fn mutation_cloning_and_consumption_preserve_public_behavior() {
    let mut map = Map::new();
    assert_eq!(map.iter_mut().count(), 0);
    assert_eq!(map.values_mut().count(), 0);
    assert_eq!(map.get_mut("missing"), None);
    assert_eq!(map.remove("missing"), None);
    assert_eq!(map.shift_remove_entry("missing"), None);
    assert_eq!(map.swap_remove_entry("missing"), None);
    map.retain(|_, _| panic!("empty retain"));
    map.sort_keys();
    map.clear();
    map.extend(std::iter::empty());
    map.append(&mut Map::new());
    assert!(map.is_empty());
    for key in ["b", "a", "c"] {
        map.insert(key.into(), json!(key));
    }
    let mut cloned = map.clone();
    cloned["a"] = json!(1);
    assert_eq!(map["a"], json!("a"));
    assert_eq!(
        map.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["b", "a", "c"]
    );
    map.shift_insert(0, "c".into(), json!(3));
    assert_eq!(
        map.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["c", "b", "a"]
    );
    assert_eq!(map.swap_remove("c"), Some(json!(3)));
    assert_eq!(
        map.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    let mut other = Map::new();
    other.append(&mut map);
    assert!(map.is_empty());
    let mut equal = Map::new();
    equal.insert("b".into(), json!("b"));
    equal.insert("a".into(), json!("a"));
    assert_eq!(other, equal);
    let hash = |m: &Map<String, Value>| {
        let mut h = DefaultHasher::new();
        m.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash(&other), hash(&equal));
    assert_eq!(
        other.clone().into_values().collect::<Vec<_>>(),
        vec![json!("a"), json!("b")]
    );
    assert_eq!(
        other.into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    equal.clone_from(&Map::new());
    assert!(equal.is_empty());
    assert_eq!(serde_json::to_string(&equal).unwrap(), "{}");
}

#[test]
fn entry_and_collection_round_trip_keep_insertion_order() {
    let mut map: Map<String, Value> = [("b".into(), json!(2)), ("a".into(), json!(1))]
        .into_iter()
        .collect();
    map.entry("c").or_insert(json!(3));
    *map.entry("a").or_insert(Value::Null) = json!(4);
    assert_eq!(
        serde_json::to_string(&map).unwrap(),
        r#"{"b":2,"a":4,"c":3}"#
    );
    let parsed: Map<String, Value> =
        serde_json::from_str(&serde_json::to_string(&map).unwrap()).unwrap();
    assert_eq!(map, parsed);
    assert_eq!(
        parsed.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["b", "a", "c"]
    );
    assert_eq!(
        parsed.get_key_value("a"),
        Some((&"a".to_owned(), &json!(4)))
    );
    map.clone_from(&parsed);
    assert_eq!(
        map.into_iter().collect::<Vec<_>>(),
        parsed.into_iter().collect::<Vec<_>>()
    );
}

#[test]
fn empty_maps_keep_equality_hashing_and_owned_iteration() {
    let empty = Map::new();
    let parsed: Map<String, Value> = serde_json::from_str("{}").unwrap();
    let collected: Map<String, Value> = std::iter::empty().collect();
    let mut cleared = Map::with_capacity(4);
    cleared.insert("x".into(), json!(1));
    cleared.clear();
    let hash = |m: &Map<String, Value>| {
        let mut h = DefaultHasher::new();
        m.hash(&mut h);
        h.finish()
    };
    for map in [parsed, collected, cleared] {
        assert_eq!(map, empty);
        assert_eq!(hash(&map), hash(&empty));
        assert_eq!(map.clone().into_values().count(), 0);
        assert_eq!(map.into_iter().count(), 0);
    }
}
