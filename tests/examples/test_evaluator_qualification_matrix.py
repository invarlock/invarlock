from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from examples.integrations.evaluator_transaction.build_attestation import (
    load_level3_build_attestation,
    verify_level3_build_attestation,
)
from invarlock.core.runtime_provider import load_runtime_behavioral_schedule
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.runtime_import_authoring import (
    RuntimeImportAuthoringError,
    load_external_scoring_records_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "evaluator-qualification"
AUTHORITATIVE = EXAMPLE / "authoritative"
SIGNED_TRANSACTIONS = EXAMPLE / "signed-transactions"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _matrix_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluator_qualification_matrix_example", EXAMPLE / "matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_matrix_requalifies_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "matrix.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("verified ") == 19


def test_matrix_is_a_secondary_catalog_with_a_compact_release_focus() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    profiles = matrix["profiles"]
    assert isinstance(profiles, list)
    assert len(profiles) == 19
    expected = {
        "arize-phoenix-evals",
        "autoevals",
        "azure-ai-evaluation",
        "deepeval",
        "evidently",
        "garak",
        "hugging-face-evaluate",
        "inspect-ai",
        "langfuse",
        "lighteval",
        "lm-evaluation-harness",
        "mlflow",
        "openai-evals",
        "opik",
        "openevals",
        "promptfoo",
        "pydantic-evals",
        "ragas",
        "trulens",
    }
    assert {profile["profile_id"] for profile in profiles} == expected
    categories = matrix["categories"]
    assert set(categories) == {
        "application-evaluation-sdk",
        "benchmark-harness",
        "evaluation-observability-platform",
        "general-metric-library",
        "security-red-team",
    }
    assert all(profile["category"] in categories for profile in profiles)
    assert matrix["selection"]["reviewed_on"] == "2026-07-27"
    assert matrix["selection"]["minimum_activity_window_months"] == 12
    assert matrix["release_focus"] == {
        "flagship_profiles": ["lm-evaluation-harness", "inspect-ai"]
    }

    for profile in profiles:
        assert profile["support_status"] == "maintained_adapter"
        artifact = EXAMPLE / "artifacts" / profile["profile_id"]
        raw = _load(artifact / "upstream-output.json")
        result = _load(artifact / "qualification-result.json")
        assert raw["upstream"] == profile["upstream"]
        assert isinstance(raw["entrypoint"], str) and raw["entrypoint"]
        assert raw["entrypoint"] != "precomputed"
        assert result["profile_id"] == profile["profile_id"]
        if profile["authority"]["mode"] == "deterministic_per_record":
            assert [record["score"] for record in raw["records"]] == [1.0, 0.0]
            assert result["authority"] == "verdict_authority"
            assert result["record_count"] == 2
        else:
            assert "summary" in raw
            assert result["authority"] == "observation_only"
            assert result["record_count"] == 0


def test_python_execution_evidence_names_the_pinned_upstream_package() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    for profile in matrix["profiles"]:
        if profile["upstream"]["ecosystem"] != "pypi":
            continue
        raw = _load(
            EXAMPLE / "artifacts" / profile["profile_id"] / "upstream-output.json"
        )
        inventory = {item["name"]: item["version"] for item in raw["environment"]}
        package_name = profile["upstream"]["name"].lower().replace("_", "-")
        assert inventory[package_name] == profile["upstream"]["version"]


def test_promptfoo_execution_binds_registry_integrity() -> None:
    raw = _load(EXAMPLE / "artifacts" / "promptfoo" / "upstream-output.json")
    package = raw["environment"][0]
    declaration = dict(
        line.split("=", 1)
        for line in (EXAMPLE / "locks" / "promptfoo.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert declaration["package"] == f"promptfoo@{package['version']}"
    assert declaration["integrity"] == package["integrity"]
    assert declaration["shasum"] == package["shasum"]


def test_public_retained_artifacts_do_not_contain_local_paths_or_secrets() -> None:
    forbidden = (
        b"/users/",
        b"/private/tmp/",
        b"root@",
        b"authorization:",
        b"api_key",
        b"bearer ",
    )
    for root in (
        EXAMPLE / "artifacts",
        AUTHORITATIVE / "artifacts",
        SIGNED_TRANSACTIONS,
    ):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            payload = path.read_bytes().lower()
            for marker in forbidden:
                assert marker not in payload, path
            assert b"begin private key" not in payload, path


def test_matrix_preserves_independent_support_authority_and_maturity_axes() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    demonstrations = _load(EXAMPLE / "demonstrations.json")["profiles"]
    replayable = []
    retained = []
    profiles = {profile["profile_id"]: profile for profile in matrix["profiles"]}
    for profile_id, levels in demonstrations.items():
        assert set(levels) == {"retained_signed_transaction"}
        profile = profiles[profile_id]
        if profile["authority"]["mode"] == "deterministic_per_record":
            replayable.append(profile_id)
        if levels["retained_signed_transaction"]:
            retained.append(profile_id)

    assert len(replayable) == 17
    assert retained == ["inspect-ai", "lm-evaluation-harness"]


@pytest.mark.parametrize("profile_id", ["lm-evaluation-harness", "inspect-ai"])
def test_flagship_signed_transaction_is_retained_and_replays_offline(
    profile_id: str,
) -> None:
    root = SIGNED_TRANSACTIONS / profile_id
    transaction = _load(root / "transaction.json")
    verification = transaction["verification"]

    evidence = verify_comparison_evidence(
        root / "evidence",
        policy_path=root / "policy.json",
        expected_artifact_digests=verification["artifact_digests"],
        expected_schedule_digest=verification["schedule_digest"],
        expected_runtime_digests=verification["runtime_digests"],
        expected_signer_fingerprint=verification["evidence_signer_fingerprint"],
    )
    receipt = verify_signed_verification_receipt(
        root / "verification.receipt.json",
        root / "evidence",
        policy_path=root / "policy.json",
        expected_artifact_digests=verification["artifact_digests"],
        expected_schedule_digest=verification["schedule_digest"],
        expected_runtime_digests=verification["runtime_digests"],
        expected_pack_signer_fingerprint=verification["evidence_signer_fingerprint"],
        expected_verifier_identity=verification["verifier_identity"],
        expected_verifier_fingerprint=verification["verifier_fingerprint"],
        expected_trust_profile_digest=verification["trust_profile_digest"],
    )

    loaded_key = serialization.load_pem_public_key(
        (root / "builder.public.pem").read_bytes()
    )
    assert isinstance(loaded_key, ed25519.Ed25519PublicKey)
    attestation = load_level3_build_attestation(root / "build-attestation.json")
    build = verify_level3_build_attestation(
        attestation,
        builder_public_key=loaded_key,
        evaluator=profile_id,
        evaluator_version=transaction["evaluator_version"],
        runtime_image_id=transaction["runtime_image_id"],
        base_image_id=transaction["base_image_id"],
        source_commit=transaction["source_commit"],
        source_bundle_sha256=transaction["source_bundle_sha256"],
        lock_sha256=transaction["lock_sha256"],
        entrypoint=transaction["entrypoint"],
    )

    report = _load(root / "evidence/reports/evaluation.report.json")
    assert evidence.payload["ok"] is True
    assert receipt.ok is True
    assert build["runtime_image_id"] == transaction["runtime_image_id"]
    assert report["record_count"] == 102
    assert report["verdict"] == "pass"


def test_retained_transaction_metadata_is_strict_and_bounded(tmp_path: Path) -> None:
    module = _matrix_module()
    source = SIGNED_TRANSACTIONS / "inspect-ai" / "transaction.json"
    transaction = _load(source)
    path = tmp_path / "transaction.json"

    path.write_text(
        source.read_text(encoding="utf-8").replace(
            '"format": "invarlock/retained-evaluator-transaction-v1",',
            '"format": "first", "format": "second",',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.load_retained_transaction(path, profile_id="inspect-ai")

    transaction["unexpected"] = True
    path.write_text(json.dumps(transaction), encoding="utf-8")
    with pytest.raises(ValueError, match="fields are invalid"):
        module.load_retained_transaction(path, profile_id="inspect-ai")

    path.write_bytes(b"{" + b" " * (1024 * 1024))
    with pytest.raises(ValueError, match="exceeds"):
        module.load_retained_transaction(path, profile_id="inspect-ai")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(format="wrong"), "metadata is invalid"),
        (
            lambda value: value.update(runtime_image_id="wrong"),
            "must be a sha256 digest",
        ),
        (lambda value: value.update(source_commit="wrong"), "source commit"),
        (lambda value: value.update(executed_on="yesterday"), "execution date"),
        (lambda value: value.update(evaluator_version=""), "evaluator version"),
        (lambda value: value.update(entrypoint=[]), "entrypoint"),
        (
            lambda value: value["verification"].update(unexpected=True),
            "verification fields",
        ),
        (
            lambda value: value["verification"].update(artifact_digests={}),
            "exactly baseline and subject",
        ),
        (
            lambda value: value["verification"].update(schedule_digest="wrong"),
            "must be a sha256 digest",
        ),
        (
            lambda value: value["verification"].update(verifier_identity=" "),
            "verifier identity",
        ),
    ],
)
def test_retained_transaction_metadata_rejects_malformed_fields(
    mutation: object,
    message: str,
    tmp_path: Path,
) -> None:
    module = _matrix_module()
    transaction = json.loads(
        (SIGNED_TRANSACTIONS / "inspect-ai" / "transaction.json").read_bytes()
    )
    assert callable(mutation)
    mutation(transaction)
    path = tmp_path / "transaction.json"
    path.write_text(json.dumps(transaction), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.load_retained_transaction(path, profile_id="inspect-ai")


def test_line_coverage_exemptions_are_exactly_the_isolated_execution_surface() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    expected = {
        f"examples/evaluator-qualification/{profile['runner']}"
        for profile in matrix["profiles"]
    }
    expected.update(
        f"examples/evaluator-qualification/{asset}"
        for profile in matrix["profiles"]
        for asset in profile["runner_assets"]
        if asset.endswith(".py")
    )
    expected.add("examples/evaluator-qualification/authoritative/generate_cases.py")
    expected.update(
        {
            "examples/integrations/evaluator_level3.py",
            "examples/integrations/evaluator_level3_launch.py",
            "examples/integrations/evaluator_transaction/build_attestation.py",
            "examples/integrations/evaluator_transaction/worker.py",
            "examples/integrations/gguf_llama_cpp.py",
            "examples/integrations/hf_vision_text.py",
            "examples/integrations/inspect-ai/launch.py",
            "examples/integrations/launch.py",
            "examples/integrations/lm-evaluation-harness/example.py",
            "examples/integrations/lm-evaluation-harness/launch.py",
            "examples/integrations/openai-evals/launch.py",
            "examples/integrations/trust_material.py",
        }
    )
    exemptions = {
        line
        for line in (ROOT / "examples/coverage-exemptions.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }

    assert exemptions == expected


def test_authoritative_corpus_is_real_pinned_model_execution() -> None:
    cases = _load(AUTHORITATIVE / "cases.json")
    source_evaluation = cases["source_evaluation"]
    model = source_evaluation["model"]
    records = cases["records"]

    assert cases["format"] == "invarlock/evaluator-authoritative-cases-v1"
    assert source_evaluation["kind"] == "model_execution"
    assert model["model_id"] == "Qwen/Qwen3-0.6B"
    assert re.fullmatch("[0-9a-f]{40}", model["immutable_revision"])
    assert re.fullmatch("sha256:[0-9a-f]{64}", model["snapshot_tree_sha256"])
    assert source_evaluation["generation"] == {
        "backend": "transformers",
        "do_sample": False,
        "dtype": "float32",
        "max_new_tokens": 1,
        "seed": 0,
    }
    assert len(records) == 102
    assert all(record["output"] for record in records)
    scores = [record["output"] == record["reference"] for record in records]
    assert any(scores)
    assert not all(scores)


def test_retained_independently_replayable_imports_replay_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "matrix.py"), "verify-replayable"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("verified independently replayable import ") == 17

    matrix = _load(EXAMPLE / "matrix.json")
    replayable = [
        profile["profile_id"]
        for profile in matrix["profiles"]
        if profile["authority"]["mode"] == "deterministic_per_record"
    ]
    for profile_id in replayable:
        artifact = AUTHORITATIVE / "artifacts" / profile_id
        result = _load(artifact / "qualification-result.json")
        replay = _load(artifact / "import-replay.json")
        records = (artifact / "runtime-import-records.jsonl").read_bytes().splitlines()
        raw = _load(artifact / "upstream-output.json")

        assert result["outcome"] == "qualified_for_import"
        assert result["authority"] == "verdict_authority"
        assert result["record_count"] == 102
        assert result["scores"].count(1.0) == 52
        assert result["scores"].count(0.0) == 50
        assert len(records) == 102
        assert replay["record_count"] == 102
        assert replay["profile_id"] == profile_id
        assert replay["source_kind"] == "model_execution"
        assert raw["source_evaluation"]["model"]["model_id"] == "Qwen/Qwen3-0.6B"
        assert len(raw["records"]) == 102
        assert raw["entrypoint"] != "precomputed"


def test_authoritative_import_rejects_post_qualification_record_tampering(
    tmp_path: Path,
) -> None:
    source = AUTHORITATIVE / "artifacts" / "inspect-ai" / "runtime-import-records.jsonl"
    records = source.read_bytes().splitlines()
    first = json.loads(records[0])
    first["output_text"] = "tampered"
    records[0] = json.dumps(
        first,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    tampered = tmp_path / "records.jsonl"
    tampered.write_bytes(b"\n".join(records) + b"\n")
    schedule = load_runtime_behavioral_schedule(AUTHORITATIVE / "runtime-schedule.json")

    with pytest.raises(RuntimeImportAuthoringError, match="output digest is invalid"):
        load_external_scoring_records_jsonl(tampered, schedule=schedule)
