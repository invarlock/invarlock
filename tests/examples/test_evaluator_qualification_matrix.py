from __future__ import annotations

import argparse
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
    load_evaluator_build_attestation,
    verify_evaluator_build_attestation,
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
        assert set(levels) == {"retained_signed_transactions"}
        profile = profiles[profile_id]
        if profile["authority"]["mode"] == "deterministic_per_record":
            replayable.append(profile_id)
        retained_transactions = levels["retained_signed_transactions"]
        assert isinstance(retained_transactions, list)
        for transaction in retained_transactions:
            assert transaction["record_count"] == 400
            retained.append((profile_id, transaction["role"]))

    assert len(replayable) == 17
    assert retained == [
        ("inspect-ai", "flagship"),
        ("inspect-ai", "deployment_approval"),
        ("lm-evaluation-harness", "flagship"),
        ("lm-evaluation-harness", "portability"),
    ]


def test_flagship_proof_map_links_every_retained_evidence_stage() -> None:
    proof_map = (SIGNED_TRANSACTIONS / "README.md").read_text(encoding="utf-8")
    common = (
        "profile.json",
        "upstream-output.json",
        "export.json",
        "qualification-result.json",
        "import-replay.json",
        "evidence/manifest.json",
        "verification.receipt.json",
        "build-attestation.json",
        "transaction.json",
    )
    packages = {
        "deployment-approval-inspect-ai": "inspect-ai",
        "gemma4-lm-evaluation-harness": "lm-evaluation-harness",
        "qwen35-inspect-ai": "inspect-ai",
        "qwen35-lm-evaluation-harness": "lm-evaluation-harness",
    }
    for package_id, profile_id in packages.items():
        targets = (
            *(f"../artifacts/{profile_id}/{name}" for name in common[:4]),
            f"../authoritative/artifacts/{profile_id}/{common[4]}",
            *(f"{package_id}/{name}" for name in common[5:]),
        )
        for target in targets:
            assert f"({target})" in proof_map
            assert (SIGNED_TRANSACTIONS / target).resolve().is_file()


@pytest.mark.parametrize(
    ("package_id", "profile_id", "expected_verdict"),
    [
        ("deployment-approval-inspect-ai", "inspect-ai", "pass"),
        ("gemma4-lm-evaluation-harness", "lm-evaluation-harness", "fail"),
        ("qwen35-inspect-ai", "inspect-ai", "fail"),
        ("qwen35-lm-evaluation-harness", "lm-evaluation-harness", "fail"),
    ],
)
def test_flagship_signed_transaction_is_retained_and_replays_offline(
    package_id: str,
    profile_id: str,
    expected_verdict: str,
) -> None:
    root = SIGNED_TRANSACTIONS / package_id
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
    attestation = load_evaluator_build_attestation(root / "build-attestation.json")
    build = verify_evaluator_build_attestation(
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
    assert evidence.payload["integrity_ok"] is True
    assert evidence.payload["policy_verdict"] == expected_verdict
    assert evidence.payload["ok"] is (expected_verdict == "pass")
    assert receipt.ok is True
    assert build["runtime_image_id"] == transaction["runtime_image_id"]
    assert report["record_count"] == 400
    assert report["verdict"] == expected_verdict


def test_flagship_comparison_reports_native_agreement_without_a_new_verdict() -> None:
    module = _matrix_module()
    retained = _load(SIGNED_TRANSACTIONS / "flagship-comparison.json")

    assert retained == module.flagship_comparison_document(
        ["qwen35-lm-evaluation-harness", "qwen35-inspect-ai"]
    )
    assert retained["record_count"] == 400
    assert retained["sides"]["baseline"]["score_agreement"] == 1.0
    assert retained["sides"]["subject"]["score_agreement"] == 1.0
    assert "verdict" not in retained


def test_flagship_comparison_identifies_score_and_record_digest_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _matrix_module()
    _comparison_fixture(tmp_path)
    _mutate_json(
        tmp_path / "right/evidence/records/paired-records.json",
        lambda value: (
            value["records"][0]["baseline"].update(score=0.0),
            value["records"][0]["subject"].update(
                observation_record_digest="sha256:" + "e" * 64
            ),
        ),
    )
    monkeypatch.setattr(module, "SIGNED_TRANSACTIONS", tmp_path)

    comparison = module.flagship_comparison_document(["left", "right"])

    assert comparison["sides"]["baseline"]["score_agreement"] == 0.0
    assert comparison["sides"]["baseline"]["score_mismatch_ids"] == ["record-1"]
    assert comparison["sides"]["subject"]["record_digest_agreement"] == 0.0
    assert comparison["sides"]["subject"]["record_digest_mismatch_ids"] == ["record-1"]


@pytest.mark.parametrize(
    "value",
    [
        {},
        [{}],
        [
            {
                "dataset_name": "fixed",
                "package_id": "../unsafe",
                "record_count": 400,
                "role": "flagship",
            }
        ],
        [
            {
                "dataset_name": "fixed",
                "package_id": "fixed",
                "record_count": True,
                "role": "flagship",
            }
        ],
    ],
)
def test_retained_transactions_reject_ambiguous_or_invalid_claims(
    value: object,
) -> None:
    module = _matrix_module()

    with pytest.raises(ValueError, match="demonstration status is invalid"):
        module.retained_transactions(value, profile_id="flagship")


def test_matrix_rejects_extra_demonstration_status_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _matrix_module()
    levels = module.demonstration_levels()
    levels["inspect-ai"]["undeclared_maturity"] = True
    monkeypatch.setattr(module, "demonstration_levels", lambda: levels)

    with pytest.raises(ValueError, match="demonstration status is invalid"):
        module.verify()


def _comparison_fixture(root: Path) -> None:
    record = {
        "baseline": {
            "observation_record_digest": "sha256:" + "a" * 64,
            "score": 1.0,
        },
        "input_sha256": "b" * 64,
        "record_id": "record-1",
        "subject": {
            "observation_record_digest": "sha256:" + "c" * 64,
            "score": 0.0,
        },
    }
    paired = {
        "format": "invarlock/paired-records-v1",
        "metric": "exact_match",
        "records": [record],
        "schedule_sha256": "d" * 64,
    }
    schedule = {
        "dataset_identity": {"dataset_name": "fixed"},
        "records": [{"record_id": "record-1"}],
    }
    report = {
        "baseline": {"mean_score": 1.0},
        "paired_binary": {"effect_size_pp": -100.0},
        "record_count": 1,
        "sample_qualification": {"interval_width": {"observed": 100.0}},
        "subject": {"mean_score": 0.0},
        "verdict": "pass",
    }
    for profile_id in ("left", "right"):
        evidence = root / profile_id / "evidence"
        paired_path = evidence / "records/paired-records.json"
        schedule_path = evidence / "schedule/runtime-behavioral-schedule.json"
        report_path = evidence / "reports/evaluation.report.json"
        paired_path.parent.mkdir(parents=True)
        schedule_path.parent.mkdir(parents=True)
        report_path.parent.mkdir(parents=True)
        paired_path.write_text(json.dumps(paired), encoding="utf-8")
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda root: (
                root / "left/evidence/records/paired-records.json"
            ).write_text("{}", encoding="utf-8"),
            "paired records are invalid",
        ),
        (
            lambda root: _mutate_json(
                root / "right/evidence/records/paired-records.json",
                lambda value: value.update(schedule_sha256="e" * 64),
            ),
            "schedules do not match",
        ),
        (
            lambda root: _mutate_json(
                root / "right/evidence/schedule/runtime-behavioral-schedule.json",
                lambda value: value.update(extra=True),
            ),
            "schedules do not match",
        ),
        (
            lambda root: [
                _mutate_json(
                    root
                    / profile_id
                    / "evidence/schedule/runtime-behavioral-schedule.json",
                    lambda value: value.update(dataset_identity=[]),
                )
                for profile_id in ("left", "right")
            ],
            "dataset identity is invalid",
        ),
        (
            lambda root: _mutate_json(
                root / "right/evidence/records/paired-records.json",
                lambda value: value["records"][0].update(record_id="other"),
            ),
            "record identities do not match",
        ),
        (
            lambda root: _mutate_json(
                root / "right/evidence/records/paired-records.json",
                lambda value: value["records"][0].update(baseline=[]),
            ),
            "record sides are invalid",
        ),
        (
            lambda root: _mutate_json(
                root / "right/evidence/reports/evaluation.report.json",
                lambda value: value.update(verdict="unknown"),
            ),
            "comparison report is invalid",
        ),
    ],
)
def test_flagship_comparison_rejects_divergent_or_malformed_inputs(
    mutation: object,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _matrix_module()
    _comparison_fixture(tmp_path)
    assert callable(mutation)
    mutation(tmp_path)
    monkeypatch.setattr(module, "SIGNED_TRANSACTIONS", tmp_path)

    with pytest.raises(ValueError, match=message):
        module.flagship_comparison_document(["left", "right"])


def _mutate_json(path: Path, mutation: object) -> None:
    value = json.loads(path.read_bytes())
    assert callable(mutation)
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_flagship_comparison_requires_two_distinct_profiles() -> None:
    module = _matrix_module()

    with pytest.raises(ValueError, match="exactly two"):
        module.flagship_comparison_document(["same", "same"])


def test_retained_transaction_claim_is_verified_against_signed_dataset() -> None:
    module = _matrix_module()

    with pytest.raises(ValueError, match="declared dataset"):
        module.verify_signed_transaction(
            "inspect-ai",
            {
                "dataset_name": "lambada-openai-qwen3-one-token-400-v1",
                "package_id": "deployment-approval-inspect-ai",
                "record_count": 401,
                "role": "deployment_approval",
            },
        )


def test_matrix_rejects_a_stale_flagship_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _matrix_module()
    stale = tmp_path / "flagship-comparison.json"
    stale.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "FLAGSHIP_COMPARISON", stale)

    with pytest.raises(ValueError, match="comparison is stale"):
        module.verify()


def test_write_flagship_comparison_command_verifies_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _matrix_module()
    output = tmp_path / "flagship-comparison.json"
    claims = {
        profile_id: {
            "retained_signed_transactions": [
                {
                    "dataset_name": "fixed",
                    "package_id": f"{profile_id}-package",
                    "record_count": 1,
                    "role": "flagship",
                }
            ]
        }
        for profile_id in ("left", "right")
    }
    verified: list[str] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(command="write-flagship-comparison"),
    )
    monkeypatch.setattr(module, "release_focus", lambda: ["left", "right"])
    monkeypatch.setattr(module, "demonstration_levels", lambda: claims)
    monkeypatch.setattr(
        module,
        "verify_signed_transaction",
        lambda profile_id, _claim: verified.append(profile_id),
    )
    monkeypatch.setattr(
        module,
        "flagship_comparison_document",
        lambda package_ids: {"format": "comparison", "transactions": package_ids},
    )
    monkeypatch.setattr(module, "FLAGSHIP_COMPARISON", output)

    module.main()

    assert verified == ["left", "right"]
    assert json.loads(output.read_bytes()) == {
        "format": "comparison",
        "transactions": ["left-package", "right-package"],
    }


def test_write_flagship_comparison_requires_every_retained_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _matrix_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(command="write-flagship-comparison"),
    )
    monkeypatch.setattr(module, "release_focus", lambda: ["left", "right"])
    monkeypatch.setattr(
        module,
        "demonstration_levels",
        lambda: {
            "left": {"retained_signed_transactions": []},
            "right": {
                "retained_signed_transactions": [
                    {
                        "dataset_name": "fixed",
                        "package_id": "right-package",
                        "record_count": 1,
                        "role": "flagship",
                    }
                ]
            },
        },
    )

    with pytest.raises(ValueError, match="exactly one retained flagship"):
        module.main()


def test_retained_transaction_metadata_is_strict_and_bounded(tmp_path: Path) -> None:
    module = _matrix_module()
    source = SIGNED_TRANSACTIONS / "deployment-approval-inspect-ai" / "transaction.json"
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
        (
            lambda value: value["verification"].update(policy_verdict="unknown"),
            "policy verdict",
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
        (
            SIGNED_TRANSACTIONS / "deployment-approval-inspect-ai" / "transaction.json"
        ).read_bytes()
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
            "examples/integrations/evaluator_transaction/cli.py",
            "examples/integrations/evaluator_transaction/adapters.py",
            "examples/integrations/evaluator_transaction/config.py",
            "examples/integrations/evaluator_transaction/launcher.py",
            "examples/integrations/evaluator_transaction/transaction.py",
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
