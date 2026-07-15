from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invarlock.evidence_pack_edit_common import EDIT_METADATA_SCHEMA
from scripts.evidence_packs.python.editing.training_contract import (
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_profile_snapshot import (
    produce_training_profile_snapshot,
)
from tests.evidence_packs._support_training_evidence_proof import _proof_for
from tests.evidence_packs._support_training_receipt import valid_training_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = REPO_ROOT / "scripts/evidence_packs/training_profiles.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _build_training_pack(
    tmp_path: Path,
    *,
    profile_id: str,
    edit_spec: str,
    scope: str,
) -> tuple[Path, Path]:
    profile = load_training_profile(profile_id)
    receipt = valid_training_receipt(profile)
    proof, baseline_identity, artifact_identity = _proof_for(receipt)
    pack_dir = tmp_path / "pack"
    scenario_id = "training_subject"
    snapshot_relative = f"metadata/training_profiles/{profile_id}.json"
    snapshot_path = pack_dir / snapshot_relative
    snapshot = produce_training_profile_snapshot(
        profile_id=profile_id,
        scope=scope,
        output_path=snapshot_path,
        profiles_path=PROFILES_PATH,
        repo_root=REPO_ROOT,
    )
    report_dir = pack_dir / "reports" / "tiny_gpt2" / scenario_id / "run_1"
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "scenarios": [
                {
                    "id": scenario_id,
                    "strictness": "informational",
                    "artifact_class": "validation_subject_checkpoint",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": edit_spec,
                        "version": "trained",
                    },
                    "training_profile": {
                        "profile_id": profile_id,
                        "profile_sha256": snapshot["profile_sha256"],
                        "snapshot_path": snapshot_relative,
                        "snapshot_sha256": snapshot["snapshot_sha256"],
                    },
                }
            ]
        },
    )
    provider = {"kind": "test-fixture"}
    _write_json(
        pack_dir / "metadata/dataset_provider.json",
        {
            "schema": "invarlock.dataset-provider-input.v1",
            "provider": provider,
            "provider_sha256": _canonical_sha256(provider),
        },
    )
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
            "dataset": {"provider": "test-fixture"},
        },
    )
    _write_json(
        report_dir / "edit_metadata.json",
        {
            "schema": EDIT_METADATA_SCHEMA,
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": profile.edit_type,
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "coverage": {
                "edited_tensors": receipt["changes"]["changed_tensors"],
                "edited_params": receipt["changes"]["changed_params"],
                "total_params": receipt["changes"]["total_params"],
                "coverage_ratio": (
                    receipt["changes"]["changed_params"]
                    / receipt["changes"]["total_params"]
                ),
            },
        },
    )
    _write_json(report_dir / "training_receipt.json", receipt)
    _write_json(report_dir / "training_evidence_proof.json", proof)
    return pack_dir, report_dir
