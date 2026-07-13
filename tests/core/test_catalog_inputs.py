from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from invarlock.catalog_inputs import materialize_catalog_input, prepare_catalog_preset
from invarlock.evidence_catalog import EvidenceCatalogError, input_digest

ROOT = Path(__file__).resolve().parents[2]


class _Image:
    def save(self, handle: BytesIO, *, format: str) -> None:
        assert format == "PNG"
        handle.write(b"image-bytes")


def _write_catalog(path: Path, *, sample_count: int = 1) -> Path:
    inputs = {
        "kind": "vision_text",
        "source": {
            "provider": "vision_text",
            "dataset_id": "org/dataset",
            "config_name": None,
            "split": "validation",
        },
        "materialization": {
            "dataset": "org/dataset",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "split": "validation",
            "max_samples": sample_count,
            "min_usable_samples": sample_count,
            "seed": 42,
            "shuffle": False,
            "image_field": "image",
            "prompt_field": "question",
            "answer_field": "answer",
            "id_field": "id",
            "prompt_template": "Question: {question}",
            "image_format": "png",
        },
    }
    inputs["digest"] = input_digest(inputs)
    payload = {
        "format_version": "invarlock/evidence-catalog-v1",
        "entry_count": 1,
        "entries": [
            {
                "lane_id": "vision-a",
                "slug": "vision_a",
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
                "model": {"id": "org/vision", "adapter": "hf_multimodal"},
                "preset": {
                    "path": "configs/vision.yaml",
                    "sha256": "sha256:" + ("a" * 64),
                },
                "inputs": inputs,
                "required_artifacts": [
                    {"role": "report", "path": "evaluation.report.json"},
                    {
                        "role": "runtime_manifest",
                        "path": "runtime.manifest.json",
                    },
                    {"role": "final_verdict", "path": "final_verdict.json"},
                    {"role": "source_provenance", "path": "source_repo.json"},
                    {"role": "resolved_inputs", "path": "resolved-inputs.json"},
                    {"role": "runtime_config", "path": "resolved-config.yaml"},
                    {"role": "preset", "path": "preset.yaml"},
                    {
                        "role": "independent_baseline",
                        "path": "baseline.report.json",
                    },
                    {"role": "policy_pack", "path": "policy-pack.json"},
                    {
                        "role": "input_materialization",
                        "path": "dataset/dataset_evidence.json",
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_resolved_inputs(path: Path) -> Path:
    payload = {
        "format_version": "invarlock/resolved-inputs-v1",
        "lane_id": "vision-a",
        "model": {
            "id": "org/vision",
            "adapter": "hf_multimodal",
            "revision": "a" * 40,
        },
        "dataset": {
            "provider": "vision_text",
            "id": "org/dataset",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "config_name": None,
            "split": "validation",
        },
        "preset": {
            "path": "configs/vision.yaml",
            "sha256": "sha256:" + ("a" * 64),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_materialize_catalog_input_uses_a_pinned_vision_entry(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    output_dir = tmp_path / "dataset"
    seen: dict[str, object] = {}

    def load_rows(**kwargs):
        seen.update(kwargs)
        return [
            {
                "id": "sample-1",
                "image": _Image(),
                "question": "What is shown?",
                "answer": "a test image",
            }
        ]

    result = materialize_catalog_input(
        catalog_path=catalog_path,
        lane_id="vision-a",
        output_dir=output_dir,
        load_rows=load_rows,
    )

    assert result["ok"] is True
    assert seen == {
        "dataset": "org/dataset",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "split": "validation",
        "config_name": None,
    }
    assert (output_dir / "manifest.jsonl").is_file()
    assert (output_dir / "materialization_summary.json").is_file()
    assert result["manifest"]["path"] == "manifest.jsonl"


def test_prepare_catalog_preset_requires_bound_materialization(tmp_path: Path) -> None:
    preset = tmp_path / "source-preset.yaml"
    preset.write_text(
        "dataset:\n  provider:\n    kind: vision_text\n    path: placeholder.jsonl\n",
        encoding="utf-8",
    )
    catalog_path = _write_catalog(tmp_path / "catalog.json", sample_count=800)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    import hashlib

    catalog["entries"][0]["preset"]["sha256"] = (
        "sha256:" + hashlib.sha256(preset.read_bytes()).hexdigest()
    )
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    resolved_inputs = _write_resolved_inputs(tmp_path / "resolved-inputs.json")
    resolved = json.loads(resolved_inputs.read_text(encoding="utf-8"))
    resolved["preset"]["sha256"] = catalog["entries"][0]["preset"]["sha256"]
    resolved_inputs.write_text(json.dumps(resolved), encoding="utf-8")
    materialization_dir = tmp_path / "dataset"
    materialize_catalog_input(
        catalog_path=catalog_path,
        lane_id="vision-a",
        output_dir=materialization_dir,
        load_rows=lambda **_kwargs: [
            {
                "id": f"sample-{index}",
                "image": _Image(),
                "question": "What is shown?",
                "answer": "a test image",
            }
            for index in range(800)
        ],
    )

    output = tmp_path / "prepared.yaml"
    prepared = prepare_catalog_preset(
        catalog_path=catalog_path,
        lane_id="vision-a",
        resolved_inputs_path=resolved_inputs,
        preset_path=preset,
        output_path=output,
        materialization_dir=materialization_dir,
    )
    assert prepared["ok"] is True
    prepared_payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert prepared_payload["model"] == {
        "id": "org/vision",
        "adapter": "hf_multimodal",
        "model_identity": {"kind": "remote_revision", "revision": "a" * 40},
    }
    assert prepared_payload["dataset"]["provider"]["kind"] == "vision_text"

    manifest = materialization_dir / "manifest.jsonl"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(EvidenceCatalogError, match="not bound"):
        prepare_catalog_preset(
            catalog_path=catalog_path,
            lane_id="vision-a",
            resolved_inputs_path=resolved_inputs,
            preset_path=preset,
            output_path=tmp_path / "tampered.yaml",
            materialization_dir=materialization_dir,
        )


def test_prepare_catalog_preset_overlays_exact_text_model_and_dataset(
    tmp_path: Path,
) -> None:
    catalog_path = ROOT / "contracts" / "evidence_catalog_v1.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = next(
        candidate
        for candidate in catalog["entries"]
        if candidate["lane_id"] == "bert-mlm-hf"
    )
    source = entry["inputs"]["source"]
    resolved_inputs = tmp_path / "resolved-inputs.json"
    resolved_inputs.write_text(
        json.dumps(
            {
                "format_version": "invarlock/resolved-inputs-v1",
                "lane_id": entry["lane_id"],
                "model": {
                    **entry["model"],
                    "revision": "b" * 40,
                },
                "dataset": {
                    "provider": source["provider"],
                    "id": source["dataset_id"],
                    "revision": "c" * 40,
                    "config_name": source["config_name"],
                    "split": source["split"],
                },
                "preset": entry["preset"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared.yaml"

    prepare_catalog_preset(
        catalog_path=catalog_path,
        lane_id=entry["lane_id"],
        resolved_inputs_path=resolved_inputs,
        preset_path=ROOT / entry["preset"]["path"],
        output_path=output,
    )

    prepared = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert prepared["model"]["id"] == "bert-base-uncased"
    assert prepared["model"]["adapter"] == "hf_mlm"
    assert prepared["model"]["model_identity"] == {
        "kind": "remote_revision",
        "revision": "b" * 40,
    }
    assert prepared["dataset"]["provider"] == {
        "kind": "wikitext2",
        "dataset_name": "Salesforce/wikitext",
        "config_name": "wikitext-2-raw-v1",
        "revision": "c" * 40,
    }
