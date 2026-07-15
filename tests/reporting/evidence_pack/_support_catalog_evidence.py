from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

import tests.cli.verify._support_runtime_provenance as provenance_support
from invarlock.catalog_inputs import build_evaluation_input_binding
from invarlock.cli.run_config import prepare_config_for_run
from invarlock.core.evaluate_plan import (
    build_subject_noop_run_config,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
)
from invarlock.evidence_catalog import input_digest
from invarlock.evidence_pack_json import sha256_prefixed
from invarlock.runtime_security import RuntimeManifestExecution
from invarlock.runtime_security_helpers import write_runtime_manifest
from tests._support_evidence_pack_signing import (
    generate_signing_keypair,
    sign_manifest,
)

IMAGE_DIGEST = "sha256:" + ("a" * 64)
MODEL_REVISION = "b" * 40
DATASET_REVISION = "d" * 40
SOURCE_COMMIT = "c" * 40
SOURCE_BUNDLE_DIGEST = "sha256:" + ("e" * 64)
SIGNER_FINGERPRINT = "sha256:" + ("f" * 64)

_REQUIRED_ARTIFACTS = [
    {"role": "report", "path": "evaluation.report.json"},
    {"role": "runtime_manifest", "path": "runtime.manifest.json"},
    {"role": "final_verdict", "path": "final_verdict.json"},
    {"role": "source_provenance", "path": "source_repo.json"},
    {"role": "resolved_inputs", "path": "resolved-inputs.json"},
    {"role": "runtime_config", "path": "resolved-config.yaml"},
    {"role": "preset", "path": "preset.yaml"},
    {"role": "independent_baseline", "path": "baseline.report.json"},
    {"role": "policy_pack", "path": "policy-pack.json"},
]


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def catalog_entry(
    *,
    lane_id: str,
    model_id: str,
    preset_digest: str,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "kind": "preset_provider",
        "source": {
            "provider": "wikitext2",
            "dataset_id": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "split": "validation",
        },
    }
    inputs["digest"] = input_digest(inputs)
    return {
        "lane_id": lane_id,
        "slug": lane_id,
        "execution": {
            "profile": "release",
            "profile_sha256": (
                "sha256:368a928b080908122a156c20b869660855fdca70267fe247b14302a9ce8ac31d"
            ),
            "tier": "balanced",
            "assurance_mode": "strict",
            "execution_mode": "container",
            "edit_name": "noop",
            "preview_n": 400,
            "final_n": 400,
        },
        "model": {"id": model_id, "adapter": "hf_causal"},
        "preset": {
            "path": "configs/presets/catalog-test.yaml",
            "sha256": preset_digest,
        },
        "inputs": inputs,
        "required_artifacts": list(_REQUIRED_ARTIFACTS),
    }


@dataclass(frozen=True)
class CatalogEvidenceFixture:
    root: Path
    catalog: Path
    entry: dict[str, object]
    preset: Path
    resolved_inputs: Path
    runtime_config: Path
    report: Path
    runtime_manifest: Path
    baseline: Path
    policy_pack: Path
    final_verdict: Path
    source_provenance: Path
    signing_key: Path
    evaluation_binding: dict[str, object]


def write_catalog_evidence_fixture(
    root: Path, *, include_relabel_target: bool = False
) -> CatalogEvidenceFixture:
    preset_payload = {
        "model": {"id": "catalog-placeholder", "adapter": "hf_causal"},
        "dataset": {
            "provider": "wikitext2",
            "split": "validation",
            "seq_len": 1,
            "stride": 1,
        },
        "guards": {
            "order": [
                "invariants",
                "spectral",
                "rmt",
                "variance",
                "invariants",
            ]
        },
        "output": {"dir": "runs", "save_model": False, "save_report": True},
    }
    preset = root / "catalog-test.yaml"
    preset.write_text(yaml.safe_dump(preset_payload, sort_keys=False), encoding="utf-8")
    preset_digest = sha256_prefixed(preset.read_bytes())
    entry = catalog_entry(
        lane_id="text-a",
        model_id="strict-test-model",
        preset_digest=preset_digest,
    )
    entries = [entry]
    if include_relabel_target:
        entries.append(
            catalog_entry(
                lane_id="text-b",
                model_id="other-model",
                preset_digest=preset_digest,
            )
        )
    catalog = write_json(
        root / "catalog.json",
        {
            "format_version": "invarlock/evidence-catalog-v1",
            "entry_count": len(entries),
            "entries": entries,
        },
    )
    resolved_payload = {
        "format_version": "invarlock/resolved-inputs-v1",
        "lane_id": "text-a",
        "model": {
            "id": "strict-test-model",
            "adapter": "hf_causal",
            "revision": MODEL_REVISION,
        },
        "dataset": {
            "provider": "wikitext2",
            "id": "Salesforce/wikitext",
            "revision": DATASET_REVISION,
            "config_name": "wikitext-2-raw-v1",
            "split": "validation",
        },
        "preset": entry["preset"],
    }
    resolved_inputs = write_json(root / "resolved-inputs.json", resolved_payload)
    evaluation_binding = build_evaluation_input_binding(
        catalog_path=catalog,
        lane_id="text-a",
        resolved_inputs_path=resolved_inputs,
        preset_path=preset,
    )

    prepared_payload = deepcopy(preset_payload)
    prepared_payload["model"] = {
        **prepared_payload["model"],
        "id": "strict-test-model",
        "adapter": "hf_causal",
        "model_identity": {
            "kind": "remote_revision",
            "revision": MODEL_REVISION,
        },
    }
    prepared_payload["dataset"] = {
        **prepared_payload["dataset"],
        "provider": {
            "kind": "wikitext2",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": DATASET_REVISION,
        },
        "split": "validation",
    }
    sanitized = sanitize_preset_data_for_evaluate(
        prepared_payload,
        adapter_name="hf_causal",
    )
    sanitized["context"] = {"evaluation_inputs": evaluation_binding}
    runtime_request_payload = build_subject_noop_run_config(
        sanitized,
        model_id="strict-test-model",
        adapter_name="hf_causal",
        model_identity={
            "kind": "remote_revision",
            "revision": MODEL_REVISION,
        },
        output_dir="run/edited",
        profile="release",
        tier="balanced",
        guards_order=resolve_guards_order(sanitized, require_canonical=True),
        assurance_mode="strict",
        execution_mode="container",
    )
    runtime_request = root / "runtime-request.yaml"
    runtime_request.write_text(
        yaml.safe_dump(runtime_request_payload, sort_keys=False), encoding="utf-8"
    )
    runtime_config = root / "resolved-config.yaml"
    prepare_config_for_run(
        config_path=str(runtime_request),
        profile="release",
        edit=None,
        tier="balanced",
        probes=None,
        resolved_config_out=str(runtime_config),
    )

    report_payload = provenance_support._strict_provenance_gate_cert()
    report_payload["meta"]["commit"] = SOURCE_COMMIT
    report_payload["meta"]["model_identity"]["revision"] = MODEL_REVISION
    report_payload["subject_ref"]["model_identity"]["revision"] = MODEL_REVISION
    report_payload["dataset"].update(
        {
            "provider": "wikitext2",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": DATASET_REVISION,
            "split": "validation",
        }
    )
    report_payload["context"]["evaluation_inputs"] = evaluation_binding
    baseline_payload = provenance_support._matching_strict_ppl_baseline(report_payload)
    baseline_payload["meta"]["commit"] = SOURCE_COMMIT
    baseline_payload["data"].update(
        {
            "provider": "wikitext2",
            "dataset": "wikitext2",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": DATASET_REVISION,
            "split": "validation",
        }
    )
    provenance_support._bind_strict_baseline(report_payload, baseline_payload)
    report_dir = root / "report"
    report = write_json(report_dir / "evaluation.report.json", report_payload)
    baseline = write_json(root / "baseline.report.json", baseline_payload)
    runtime_manifest = write_runtime_manifest(
        report,
        config_path=runtime_config,
        extra={
            "evaluation_inputs": evaluation_binding,
            "source_bundle": {
                "read_only": True,
                "sha256": SOURCE_BUNDLE_DIGEST,
            },
        },
        execution=RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="ghcr.io/invarlock/invarlock-runtime:test",
            image_digest=IMAGE_DIGEST,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )
    policy_pack = provenance_support._write_matching_strict_policy_pack(
        root / "policy-pack.json", report_payload
    )
    final_verdict = write_json(
        root / "final_verdict.json",
        {
            "verdict": "PASS",
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    )
    source_provenance = write_json(
        root / "source_repo.json",
        {
            "format_version": "invarlock/source-provenance-v1",
            "commit": SOURCE_COMMIT,
            "source_bundle_sha256": SOURCE_BUNDLE_DIGEST,
            "dirty": False,
        },
    )
    signing_key = root / "signing.pem"
    generate_signing_keypair(
        signing_key, public_key_path=signing_key.with_suffix(".pub.pem")
    )
    return CatalogEvidenceFixture(
        root=root,
        catalog=catalog,
        entry=entry,
        preset=preset,
        resolved_inputs=resolved_inputs,
        runtime_config=runtime_config,
        report=report,
        runtime_manifest=runtime_manifest,
        baseline=baseline,
        policy_pack=policy_pack,
        final_verdict=final_verdict,
        source_provenance=source_provenance,
        signing_key=signing_key,
        evaluation_binding=evaluation_binding,
    )


def assemble_signed_catalog_pack(
    fixture: CatalogEvidenceFixture,
    out_dir: Path,
    *,
    lane_id: str = "text-a",
) -> tuple[Path, str]:
    """Assemble a signed pack exclusively for verifier tests.

    This fixture serializer is deliberately test-only: it does not make an
    acceptance decision and is not shipped in the public wheel.
    """

    catalog = json.loads(fixture.catalog.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in catalog["entries"]
        if isinstance(item, dict) and item.get("lane_id") == lane_id
    )
    out_dir.mkdir(parents=True)
    sources = {
        "reports/report-001/evaluation.report.json": fixture.report,
        "reports/report-001/runtime.manifest.json": fixture.runtime_manifest,
        "results/final_verdict.json": fixture.final_verdict,
        "baselines/baseline-001/evaluation.report.json": fixture.baseline,
        "policy/policy-pack.json": fixture.policy_pack,
        "metadata/source_repo.json": fixture.source_provenance,
        "metadata/catalog.json": fixture.catalog,
        "metadata/resolved-inputs.json": fixture.resolved_inputs,
        "metadata/runtime-config.yaml": fixture.runtime_config,
        "metadata/preset.yaml": fixture.preset,
    }
    for relative, source in sources.items():
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    write_json(
        out_dir / "metadata/scenarios.json",
        {
            "schema": "evidence_pack_scenarios_v1",
            "schema_version": 1,
            "scenarios": [
                {
                    "id": "report-001",
                    "strictness": "must_pass",
                    "intent": "submitted_report",
                    "primary_guard": "primary_metric",
                    "artifact_class": "evidence_only_pack",
                    "generation": {"kind": "evidence_only"},
                }
            ],
        },
    )
    covered = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file()
    )
    checksum_lines = [
        f"{hashlib.sha256((out_dir / relative).read_bytes()).hexdigest()}  {relative}"
        for relative in covered
    ]
    checksums_path = out_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    def digest(relative: str) -> str:
        return sha256_prefixed((out_dir / relative).read_bytes())

    policy_payload = json.loads(
        (out_dir / "policy/policy-pack.json").read_text(encoding="utf-8")
    )
    binding = {
        "path": "metadata/catalog.json",
        "digest": sha256_prefixed(fixture.catalog.read_bytes()),
        "entry_id": lane_id,
        "entry_digest": sha256_prefixed(
            json.dumps(
                entry, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
    }
    manifest = {
        "format": "evidence-pack-v1",
        "evidence_level": "high",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": hashlib.sha256(
            checksums_path.read_bytes()
        ).hexdigest(),
        "subject": {
            "name": "final_verdict",
            "path": "results/final_verdict.json",
            "digest": digest("results/final_verdict.json"),
        },
        "invocation": {
            "config_source": {
                "path": "metadata/source_repo.json",
                "digest": digest("metadata/source_repo.json"),
            }
        },
        "materials": [
            {"name": name, "path": relative, "digest": digest(relative)}
            for name, relative in (
                ("catalog", "metadata/catalog.json"),
                ("resolved-inputs", "metadata/resolved-inputs.json"),
                ("runtime-config", "metadata/runtime-config.yaml"),
                ("preset", "metadata/preset.yaml"),
            )
        ],
        "verification": {
            "clean_reports": 1,
            "error_injection_reports": 0,
            "failed_reports": 0,
            "report_assurance": "strict",
            "subject_mode": "catalog_bound_noop",
        },
        "verification_baselines": [
            {
                "name": "baseline-001",
                "path": "baselines/baseline-001/evaluation.report.json",
                "digest": digest("baselines/baseline-001/evaluation.report.json"),
                "report_paths": ["reports/report-001/evaluation.report.json"],
            }
        ],
        "verification_policy_pack": {
            "path": "policy/policy-pack.json",
            "digest": digest("policy/policy-pack.json"),
            "policy_digest": policy_payload["policy_digest"],
        },
        "catalog": binding,
    }
    write_json(out_dir / "manifest.json", manifest)
    fingerprint = sign_manifest(
        out_dir / "manifest.json", signing_key_path=fixture.signing_key
    )
    return out_dir, fingerprint
