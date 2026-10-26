"""Actual worker publication via Rust, retaining original provider facts."""

import json
from dataclasses import replace

import pytest

from invarlock.runtime_providers.native_execution import using_execution_backend
from tests.runtime import test_runtime_evidence_transaction as source


def forbidden(*args, **kwargs):
    raise AssertionError("Rust worker called Python evidence publication")


def test_public_worker_uses_native_publication(tmp_path, monkeypatch):
    import invarlock.native_backend as bridge

    root = tmp_path.resolve()
    (root / "python").mkdir()
    (root / "rust").mkdir()
    expected, expected_provider = source._run_fixture_side(monkeypatch, root / "python")
    for name in (
        "write_runtime_provider_evidence",
        "write_runtime_manifest",
        "verify_runtime_manifest_snapshot",
        "load_runtime_provider_evidence",
    ):
        monkeypatch.setattr(source.runtime_transaction, name, forbidden)
    calls = []
    dispatch = bridge.transact

    def traced(operation, config, **kwargs):
        calls.append(operation)
        return dispatch(operation, config, **kwargs)

    monkeypatch.setattr(bridge, "transact", traced)
    with using_execution_backend("rust"):
        actual, provider = source._run_fixture_side(monkeypatch, root / "rust")
    assert calls == ["write-runtime-side"]
    assert provider.session.closed and expected_provider.session.closed
    assert replace(actual.evidence, paths=expected.evidence.paths) == expected.evidence
    for name in (p.name for p in expected.directory.iterdir()):
        observed = (actual.directory / name).read_bytes()
        reference = (expected.directory / name).read_bytes()
        if name == "runtime.manifest.json":
            observed, reference = json.loads(observed), json.loads(reference)
            # Each provider run observes its own publication clock.
            observed.pop("generated_at_utc")
            reference.pop("generated_at_utc")
        assert observed == reference, name
    verified = source.verify_runtime_manifest(
        actual.report_path,
        actual.manifest_path,
        expected_image_digest=source._IMAGE_DIGEST,
        require_strict_runtime=True,
    )
    assert verified.ok, verified.errors


def test_public_worker_uses_validated_schedule_snapshot(tmp_path, monkeypatch):
    with using_execution_backend("rust"):
        source.test_runtime_side_scores_the_validated_schedule_snapshot_not_changed_path(
            monkeypatch, tmp_path.resolve()
        )


@pytest.mark.parametrize("field", ["outer_image_digest", "scoring_observation_sha256"])
def test_public_worker_rejects_original_receipt_mismatch(tmp_path, monkeypatch, field):
    original = source._Session.runtime_receipt

    def changed(self):
        value = "9" * 64
        if field == "outer_image_digest":
            value = "sha256:" + value
        return replace(original(self), **{field: value})

    monkeypatch.setattr(source._Session, "runtime_receipt", changed)
    for backend in ("python", "rust"):
        directory = tmp_path.resolve() / backend
        directory.mkdir()
        with using_execution_backend(backend), pytest.raises(ValueError):
            source._run_fixture_side(monkeypatch, directory)
        assert not (directory / "baseline").exists()


def test_public_worker_native_publication_never_replaces_destination(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    with using_execution_backend("rust"):
        actual, _ = source._run_fixture_side(monkeypatch, root)
        before = {p.name: p.read_bytes() for p in actual.directory.iterdir()}
        with pytest.raises(ValueError, match="already exists"):
            source._run_fixture_side(monkeypatch, root)
    assert {p.name: p.read_bytes() for p in actual.directory.iterdir()} == before
