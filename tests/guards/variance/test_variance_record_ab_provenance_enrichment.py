from invarlock.guards.variance import VarianceGuard


def test_record_ab_provenance_enriches_with_meta_hashes_and_model():
    g = VarianceGuard()
    g._pairing_digest = "dig123"
    g._dataset_meta = {"dataset_hash": "ds123", "tokenizer_hash": "tok456"}
    g._report_meta = {"model_id": "m1", "seed": 7}

    g._record_ab_provenance(
        "condition_a",
        tag="t",
        window_ids=["a", "b"],
        fingerprint="fp",
        mode="edited_no_ve",
        status="evaluated",
    )
    prov = g._stats.get("ab_provenance", {}).get("condition_a", {})
    assert prov.get("dataset_hash") == "ds123"
    assert prov.get("tokenizer_hash") == "tok456"
    assert prov.get("model_id") == "m1"
    assert prov.get("window_count") == 2
