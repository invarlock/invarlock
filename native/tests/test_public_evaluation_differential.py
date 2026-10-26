"""Installed extension through public evaluation APIs, with Python core disabled."""

from dataclasses import replace
from pathlib import Path

import pytest

from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.evaluation_transaction import (
    evaluate_request_file,
    preflight_evaluation_request,
)
from tests.cli.test_import_journey import (
    _key,
    _materialize_request,
    _text_scorer_registry_and_binding,
)


def files(path):
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file()
    }


def forbidden(*_args, **_kwargs):
    raise AssertionError(
        "selected Rust path called Python portable analysis/publication"
    )


def test_captured_public_api(tmp_path, monkeypatch):
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "experiments/rust/python")
    )
    from invarlock_rust_experiment.fixtures import make_fixture

    import invarlock.captured_evaluation as captured

    root = tmp_path.resolve() / "fixture"
    make_fixture(root, 12, "pass")
    request = load_evaluation_request(root / "request.json", output=root / "reference")
    expected_preflight = preflight_evaluation_request(
        request, signing_key_path=root / "signer.pem"
    )
    expected = evaluate_request_file(request, signing_key_path=root / "signer.pem")
    request = replace(request, evidence=root / "native")
    for name in (
        "compare_runs",
        "_prepare_captured_request",
        "publish_captured_evidence",
    ):
        monkeypatch.setattr(captured, name, forbidden)
    actual_preflight = preflight_evaluation_request(
        request, signing_key_path=root / "signer.pem", backend="rust"
    )
    assert (
        replace(actual_preflight, output=expected_preflight.output)
        == expected_preflight
    )
    actual = evaluate_request_file(
        request, signing_key_path=root / "signer.pem", backend="rust"
    )
    assert replace(actual, evidence_path=expected.evidence_path) == expected
    assert files(actual.evidence_path) == files(expected.evidence_path)


@pytest.mark.parametrize("extension", [False, True])
def test_runtime_import_public_api(tmp_path, monkeypatch, extension):
    import invarlock.evaluation_transaction as transaction

    root = tmp_path.resolve()
    registry, binding = (
        _text_scorer_registry_and_binding() if extension else (None, None)
    )
    material = _materialize_request(
        root, scorer_binding=binding, scorer_registry=registry
    )
    key, _ = _key(root / "signer.pem")
    request = load_evaluation_request(material["request"])
    expected_preflight = preflight_evaluation_request(
        request, signing_key_path=key, scorer_registry=registry
    )
    expected = evaluate_request_file(
        request, signing_key_path=key, scorer_registry=registry
    )
    request = replace(request, output=replace(request.output, evidence=root / "native"))
    for name in (
        "build_comparison_report",
        "derive_paired_records",
        "publish_comparison_evidence",
        "_preflight_policy",
    ):
        monkeypatch.setattr(transaction, name, forbidden)
    actual_preflight = preflight_evaluation_request(
        request, signing_key_path=key, scorer_registry=registry, backend="rust"
    )
    assert (
        replace(actual_preflight, output=expected_preflight.output)
        == expected_preflight
    )
    actual = evaluate_request_file(
        request, signing_key_path=key, scorer_registry=registry, backend="rust"
    )
    assert replace(actual, evidence_path=expected.evidence_path) == expected
    assert files(actual.evidence_path) == files(expected.evidence_path)


def test_runtime_judge_import_uses_native_pair_derivation(tmp_path, monkeypatch):
    import shutil

    import invarlock.evaluation_transaction as transaction
    from invarlock.judge_measurements import native_workflow
    from tests.core.test_native_judge_transaction import (
        _completed_measurements,
        _import,
    )

    root = tmp_path.resolve()
    reference_root = root / "reference"
    reference_root.mkdir()
    path, key, _ = _import(reference_root)
    native_root = root / "native"
    shutil.copytree(reference_root, native_root)
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        lambda *args, **kwargs: {"credential_available": True, "network_calls": 0},
    )
    monkeypatch.setattr(native_workflow, "collect_frozen", _completed_measurements)
    expected = evaluate_request_file(path, signing_key_path=key)
    monkeypatch.setattr(transaction, "derive_paired_records", forbidden)
    actual = evaluate_request_file(
        native_root / path.relative_to(reference_root),
        signing_key_path=native_root / key.relative_to(reference_root),
        backend="rust",
    )
    expected_payload = dict(expected.payload)
    expected_payload.update(
        evidence=actual.payload["evidence"], workspace=actual.payload["workspace"]
    )
    assert actual.payload == expected_payload
    assert files(Path(actual.payload["evidence"])) == files(
        Path(expected.payload["evidence"])
    )
