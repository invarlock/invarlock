from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from invarlock.public_contracts import (
    load_evidence_observation_schema,
    load_evidence_pack_schema,
)


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _manifest() -> dict[str, object]:
    input_names = (
        "baseline",
        "subject",
        "dataset",
        "baseline_runtime",
        "subject_runtime",
        "policy",
    )
    evidence_paths = {
        "request": "request.json",
        "schedule": "schedule/runtime-behavioral-schedule.json",
        "evaluation_report": "reports/evaluation.report.json",
        "baseline_run_report": "runs/baseline/report.json",
        "subject_run_report": "runs/subject/report.json",
        "baseline_runtime_manifest": "providers/baseline/runtime.manifest.json",
        "subject_runtime_manifest": "providers/subject/runtime.manifest.json",
        "baseline_runtime_config": "providers/baseline/run.yaml",
        "subject_runtime_config": "providers/subject/run.yaml",
        "baseline_provider_identity": (
            "providers/baseline/model-artifact.identity.json"
        ),
        "subject_provider_identity": ("providers/subject/model-artifact.identity.json"),
        "baseline_provider_receipt": (
            "providers/baseline/runtime-provider.receipt.json"
        ),
        "subject_provider_receipt": ("providers/subject/runtime-provider.receipt.json"),
        "baseline_scoring_observation": (
            "providers/baseline/runtime-scoring.observation.json"
        ),
        "subject_scoring_observation": (
            "providers/subject/runtime-scoring.observation.json"
        ),
    }
    return {
        "format": "evidence-pack-v1",
        "comparison_id": "acceptance-001",
        "inputs": {
            name: {
                "path": f"inputs/{name}.json",
                "digest": _digest("a"),
                "material_digest": _digest("b"),
            }
            for name in input_names
        },
        "evidence": {
            role: {"path": path, "digest": _digest("c")}
            for role, path in evidence_paths.items()
        },
        "paired_records": {
            "path": "records/paired-records.json",
            "digest": _digest("d"),
            "count": 2,
        },
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": "e" * 64,
        "signing_key_fingerprint": _digest("f"),
    }


def test_source_and_packaged_manifest_contracts_are_identical() -> None:
    root = Path(__file__).resolve().parents[3]
    source = json.loads((root / "contracts/evidence_pack.schema.json").read_text())
    packaged = json.loads(
        (root / "src/invarlock/_data/contracts/evidence_pack.schema.json").read_text()
    )

    assert source == packaged == load_evidence_pack_schema()
    jsonschema.Draft202012Validator.check_schema(source)
    jsonschema.validate(_manifest(), source)


def test_source_and_packaged_observation_contracts_are_identical() -> None:
    root = Path(__file__).resolve().parents[3]
    source = json.loads(
        (root / "contracts/evidence_observation.schema.json").read_text()
    )
    packaged = json.loads(
        (
            root / "src/invarlock/_data/contracts/evidence_observation.schema.json"
        ).read_text()
    )
    payload = {
        "format": "invarlock/evidence-observation-v1",
        "observation_id": "subject-spectral",
        "kind": "spectral",
        "scope": "subject",
        "authority": "observation",
        "bindings": {
            "comparison_id": "acceptance-001",
            "schedule_digest": _digest("a"),
            "policy_digest": _digest("b"),
            "artifact_digests": {
                "baseline": _digest("c"),
                "subject": _digest("d"),
            },
        },
        "payload": {"stable_rank": 2.0},
    }

    assert source == packaged == load_evidence_observation_schema()
    jsonschema.Draft202012Validator.check_schema(source)
    jsonschema.validate(payload, source)

    changed = copy.deepcopy(payload)
    changed["authority"] = "acceptance"
    with pytest.raises(jsonschema.ValidationError, match="observation"):
        jsonschema.validate(changed, source)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "Additional properties"),
        (
            lambda value: value["inputs"]["baseline"].update(
                path="inputs/subject.json"
            ),
            "inputs/baseline.json",
        ),
        (
            lambda value: value["evidence"]["request"].update(
                path="reports/evaluation.report.json"
            ),
            "request.json",
        ),
        (
            lambda value: value["paired_records"].update(count=0),
            "less than the minimum",
        ),
    ],
)
def test_manifest_contract_rejects_open_or_misbound_shapes(
    mutate: object,
    match: str,
) -> None:
    manifest = copy.deepcopy(_manifest())
    assert callable(mutate)
    mutate(manifest)

    with pytest.raises(jsonschema.ValidationError, match=match):
        jsonschema.validate(manifest, load_evidence_pack_schema())
