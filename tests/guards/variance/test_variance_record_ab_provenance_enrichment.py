from types import SimpleNamespace

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


def test_set_run_context_derives_arm_identity_before_public_report_assembly():
    guard = VarianceGuard()
    report = SimpleNamespace(
        meta={"config": {"model": {"id": "model-from-config"}}},
        context={
            "seeds": {"python": 17},
            "dataset_meta": {
                "dataset_hash": "dataset-hash",
                "tokenizer_hash": "tokenizer-hash",
            },
            "pairing_baseline": {
                "preview": {"window_ids": [0]},
                "final": {"window_ids": [1]},
            },
        },
        edit={},
    )

    guard.set_run_context(report)
    guard._record_ab_provenance(
        "condition_a",
        tag="post_edit",
        window_ids=["preview::0"],
        fingerprint="fingerprint",
        mode="edited_no_ve",
        status="evaluated",
    )

    provenance = guard._stats["ab_provenance"]["condition_a"]
    assert provenance["model_id"] == "model-from-config"
    assert provenance["run_seed"] == 17
    assert provenance["dataset_hash"] == "dataset-hash"
    assert provenance["tokenizer_hash"] == "tokenizer-hash"


def test_set_run_context_derives_arm_identity_from_run_context():
    guard = VarianceGuard()
    report = SimpleNamespace(
        meta={"config": {}},
        context={
            "model_id": "model-from-run-context",
            "seeds": {"python": 17},
            "dataset_meta": {
                "dataset_hash": "dataset-hash",
                "tokenizer_hash": "tokenizer-hash",
            },
            "pairing_baseline": {
                "preview": {"window_ids": [0]},
                "final": {"window_ids": [1]},
            },
        },
        edit={},
    )

    guard.set_run_context(report)
    guard._record_ab_provenance(
        "condition_a",
        tag="post_edit",
        window_ids=["preview::0"],
        fingerprint="fingerprint",
        mode="edited_no_ve",
        status="evaluated",
    )

    provenance = guard._stats["ab_provenance"]["condition_a"]
    assert provenance["model_id"] == "model-from-run-context"
