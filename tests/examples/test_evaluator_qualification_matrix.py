from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import load_runtime_behavioral_schedule
from invarlock.runtime_import_authoring import (
    RuntimeImportAuthoringError,
    load_external_scoring_records_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "evaluator-qualification"
AUTHORITATIVE = EXAMPLE / "authoritative"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


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


def test_matrix_is_expansible_real_upstream_execution_catalog() -> None:
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

    for profile in profiles:
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
    for root in (EXAMPLE / "artifacts", AUTHORITATIVE / "artifacts"):
        for path in root.rglob("*.json"):
            payload = path.read_bytes().lower()
            for marker in forbidden:
                assert marker not in payload, path


def test_matrix_preserves_three_distinct_demonstration_levels() -> None:
    demonstrations = _load(EXAMPLE / "demonstrations.json")["profiles"]
    authoritative = []
    end_to_end = []
    for profile_id, levels in demonstrations.items():
        assert levels["qualification_profile"] is True
        if levels["authoritative_import"]:
            authoritative.append(profile_id)
        if levels["end_to_end_transaction"]:
            end_to_end.append(profile_id)

    assert len(authoritative) == 17
    assert end_to_end == ["lm-evaluation-harness"]


def test_authoritative_corpus_is_real_pinned_model_execution() -> None:
    cases = _load(AUTHORITATIVE / "cases.json")
    producer = cases["producer"]
    model = producer["model"]
    records = cases["records"]

    assert cases["format"] == "invarlock/evaluator-authoritative-cases-v1"
    assert producer["kind"] == "model_execution"
    assert model["model_id"] == "Qwen/Qwen3-0.6B"
    assert re.fullmatch("[0-9a-f]{40}", model["immutable_revision"])
    assert re.fullmatch("sha256:[0-9a-f]{64}", model["snapshot_tree_sha256"])
    assert producer["generation"] == {
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


def test_seventeen_authoritative_imports_replay_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "matrix.py"), "verify-authoritative"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("verified authoritative import ") == 17

    demonstrations = _load(EXAMPLE / "demonstrations.json")["profiles"]
    authoritative = [
        profile
        for profile, levels in demonstrations.items()
        if levels["authoritative_import"]
    ]
    for profile_id in authoritative:
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
