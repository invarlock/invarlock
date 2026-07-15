from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.dataset_identity import (
    canonical_dataset_revision,
    dataset_identity_from_provider,
    dataset_identity_from_report,
    is_hosted_dataset_provider,
)


def test_hosted_dataset_identity_captures_exact_provider_coordinates() -> None:
    revision = "a" * 40
    provider = SimpleNamespace(
        name="hf_text",
        dataset_name="Salesforce/wikitext",
        config_name="wikitext-2-raw-v1",
        revision=revision,
    )

    assert dataset_identity_from_provider(provider) == {
        "provider": "hf_text",
        "dataset_name": "Salesforce/wikitext",
        "config_name": "wikitext-2-raw-v1",
        "revision": revision,
    }


def test_dataset_identity_preserves_declared_malformed_revision_for_verifier() -> None:
    provider = SimpleNamespace(
        name="wikitext2",
        dataset_name="Salesforce/wikitext",
        config_name="wikitext-2-raw-v1",
        revision="main",
    )

    assert dataset_identity_from_provider(provider)["revision"] == "main"
    assert canonical_dataset_revision("main") is None

    provider.revision = f" {'a' * 40} "
    assert dataset_identity_from_provider(provider)["revision"] == provider.revision


def test_non_hosted_provider_does_not_invent_remote_coordinates() -> None:
    provider = SimpleNamespace(
        name="local_jsonl",
        dataset_name=None,
        config_name=None,
        revision=None,
    )

    assert dataset_identity_from_provider(provider) == {"provider": "local_jsonl"}
    assert is_hosted_dataset_provider("local_jsonl") is False
    assert is_hosted_dataset_provider(" HF_SEQ2SEQ ") is True


def test_dataset_revision_requires_exact_lowercase_hex() -> None:
    revision = "b" * 64

    assert canonical_dataset_revision(revision) == revision
    assert canonical_dataset_revision(revision.upper()) is None
    assert canonical_dataset_revision(f" {revision} ") is None
    assert canonical_dataset_revision("c" * 39) is None
    assert is_hosted_dataset_provider(None) is False


def test_dataset_identity_ignores_provider_attribute_access_failures() -> None:
    class ExplodingProvider:
        @property
        def name(self) -> str:
            raise RuntimeError("unavailable")

    assert dataset_identity_from_provider(ExplodingProvider()) == {}


def test_report_dataset_identity_fails_closed_for_missing_dataset_object() -> None:
    expected = {
        "provider": None,
        "dataset_name": None,
        "config_name": None,
        "split": None,
        "revision": None,
    }

    assert dataset_identity_from_report(None) == expected
    assert dataset_identity_from_report({"dataset": []}) == expected


def test_report_dataset_identity_accepts_exact_trimmed_coordinates() -> None:
    expected = {
        "provider": "hf_text",
        "dataset_name": "Salesforce/wikitext",
        "config_name": "wikitext-2-raw-v1",
        "split": "test",
        "revision": "a" * 40,
    }

    assert dataset_identity_from_report({"dataset": expected}) == expected
