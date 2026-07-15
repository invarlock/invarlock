from __future__ import annotations

import hashlib
import json

import pytest

from scripts.evidence_packs.python import dataset_provider_policy

IMMUTABLE_REVISION = "a" * 40


def _snapshot_payload(provider: dict[str, object]) -> dict[str, object]:
    rendered = json.dumps(
        provider,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": dataset_provider_policy.DATASET_PROVIDER_SNAPSHOT_SCHEMA,
        "provider": provider,
        "provider_sha256": "sha256:" + hashlib.sha256(rendered).hexdigest(),
    }


def test_manifest_parameters_use_effective_generated_provider(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "hf_text")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_NAME", "demo/corpus")
    monkeypatch.setenv("INVARLOCK_HF_CONFIG_NAME", "default")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", IMMUTABLE_REVISION)

    assert dataset_provider_policy.dataset_provider_manifest_parameters() == {
        "kind": "hf_text",
        "dataset_name": "demo/corpus",
        "config_name": "default",
        "revision": IMMUTABLE_REVISION,
    }


def test_raw_mapping_is_authoritative_for_manifest(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "wikitext2")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", IMMUTABLE_REVISION)
    monkeypatch.setenv(
        "INVARLOCK_DATASET_PROVIDER_JSON",
        json.dumps({"kind": "hf_text", "dataset_name": "demo/corpus"}),
    )

    assert dataset_provider_policy.dataset_provider_manifest_parameters() == {
        "kind": "hf_text",
        "dataset_name": "demo/corpus",
    }


def test_local_provider_manifest_parameters(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "local_jsonl")
    monkeypatch.setenv("INVARLOCK_LOCAL_JSONL_FILE", "dataset.jsonl")

    assert dataset_provider_policy.dataset_provider_manifest_parameters() == {
        "kind": "local_jsonl",
        "file": "dataset.jsonl",
    }


@pytest.mark.parametrize(
    "reference",
    (
        "/srv/private/input.jsonl",
        r"C:\private\input.jsonl",
        "~/private/input.jsonl",
        "file:///srv/private/input.jsonl",
    ),
)
def test_public_dataset_provider_rejects_host_local_references(
    monkeypatch, reference: str
) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "local_jsonl")
    monkeypatch.setenv(
        "INVARLOCK_DATASET_PROVIDER_JSON",
        json.dumps({"kind": "local_jsonl", "file": reference}),
    )

    with pytest.raises(ValueError, match="must not contain host-local paths"):
        dataset_provider_policy.dataset_provider_manifest_parameters()
    with pytest.raises(ValueError, match="must not contain host-local paths"):
        dataset_provider_policy.build_dataset_provider_snapshot()
    with pytest.raises(ValueError, match="must not contain host-local paths"):
        dataset_provider_policy.validate_dataset_provider_snapshot(
            _snapshot_payload({"kind": "local_jsonl", "file": reference})
        )


@pytest.mark.parametrize("reference", ("../private/input.jsonl", "data/../input.jsonl"))
def test_public_dataset_provider_rejects_traversal_references(
    monkeypatch, reference: str
) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "local_jsonl")
    monkeypatch.setenv(
        "INVARLOCK_DATASET_PROVIDER_JSON",
        json.dumps({"kind": "local_jsonl", "data_files": reference}),
    )

    with pytest.raises(ValueError, match="portable relative path"):
        dataset_provider_policy.build_dataset_provider_snapshot()


def test_public_dataset_provider_keeps_portable_local_reference(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_DATASET", "local_jsonl")
    monkeypatch.setenv(
        "INVARLOCK_DATASET_PROVIDER_JSON",
        json.dumps({"kind": "local_jsonl", "data_files": "fixtures/part-*.jsonl"}),
    )

    snapshot = dataset_provider_policy.build_dataset_provider_snapshot()

    assert snapshot["provider"] == {
        "kind": "local_jsonl",
        "data_files": "fixtures/part-*.jsonl",
    }


def test_snapshot_is_atomic_and_resume_rejects_provider_drift(
    monkeypatch, tmp_path
) -> None:
    snapshot_path = tmp_path / "state" / "dataset_provider.json"
    monkeypatch.setenv("INVARLOCK_DATASET", "wikitext2")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", IMMUTABLE_REVISION)

    dataset_provider_policy.write_or_validate_dataset_provider_snapshot(
        snapshot_path,
        resume=False,
    )
    original_bytes = snapshot_path.read_bytes()
    snapshot = dataset_provider_policy.load_dataset_provider_snapshot(snapshot_path)
    assert snapshot["provider"] == {
        "config_name": "wikitext-2-raw-v1",
        "dataset_name": "Salesforce/wikitext",
        "kind": "wikitext2",
        "revision": IMMUTABLE_REVISION,
    }
    assert not list(snapshot_path.parent.glob(".dataset_provider.json.*.tmp"))

    with pytest.raises(ValueError, match="already exists"):
        dataset_provider_policy.write_or_validate_dataset_provider_snapshot(
            snapshot_path,
            resume=False,
        )

    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="differs from the persisted run input"):
        dataset_provider_policy.write_or_validate_dataset_provider_snapshot(
            snapshot_path,
            resume=True,
        )
    assert snapshot_path.read_bytes() == original_bytes


def test_snapshot_manifest_parameters_ignore_later_environment(
    monkeypatch, tmp_path
) -> None:
    snapshot_path = tmp_path / "dataset_provider.json"
    monkeypatch.setenv("INVARLOCK_DATASET", "wikitext2")
    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", IMMUTABLE_REVISION)
    dataset_provider_policy.write_or_validate_dataset_provider_snapshot(
        snapshot_path,
        resume=False,
    )

    monkeypatch.setenv("INVARLOCK_HF_DATASET_REVISION", "b" * 40)
    assert dataset_provider_policy.dataset_provider_parameters_from_snapshot(
        snapshot_path,
    ) == {
        "config_name": "wikitext-2-raw-v1",
        "dataset_name": "Salesforce/wikitext",
        "kind": "wikitext2",
        "revision": IMMUTABLE_REVISION,
    }


def test_snapshot_digest_tamper_and_missing_snapshot_fail_closed(
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "dataset_provider.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": dataset_provider_policy.DATASET_PROVIDER_SNAPSHOT_SCHEMA,
                "provider": {
                    "config_name": "wikitext-2-raw-v1",
                    "dataset_name": "Salesforce/wikitext",
                    "kind": "wikitext2",
                    "revision": IMMUTABLE_REVISION,
                },
                "provider_sha256": "sha256:" + ("0" * 64),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match provider"):
        dataset_provider_policy.load_dataset_provider_snapshot(snapshot_path)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="required dataset provider snapshot"):
        dataset_provider_policy.dataset_provider_parameters_from_snapshot(missing)
