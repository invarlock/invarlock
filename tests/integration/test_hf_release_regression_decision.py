from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import invarlock.runtime_behavior.transaction as runtime_transaction
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evaluation_runtime import CallerRuntimeResources
from invarlock.evaluation_transaction import evaluate_request_file
from invarlock.evidence_pack_contract import EVIDENCE_PATHS, canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_support import EvidencePackStatus
from invarlock.evidence_verification import (
    EvidenceVerificationError,
    verify_evidence,
)
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
)

_IMAGE_DIGEST = "sha256:" + "6" * 64
_ACCEPTED_RATIO_MAX = 1.0
_REJECTED_RATIO_MAX = 1.01
_MINIMUM_PROVEN_REGRESSION_RATIO = 10.0
_MAXIMUM_PROVEN_IMPROVEMENT_RATIO = 0.1


def _signing_key(path: Path) -> tuple[Path, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path, public_key_fingerprint(key.public_key())


def _tokenizer(checkpoint: Path, transformers: Any, tokenizers: Any) -> Any:
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "alpha": 4,
        "beta": 5,
        "target": 6,
        "other": 7,
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(checkpoint)
    return tokenizer


def _distinct_checkpoints(tmp_path: Path) -> tuple[dict[str, Path], str]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")

    torch.manual_seed(20_260_716)
    seed_model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=8,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    seed_model.eval()
    token_ids = torch.tensor([[4, 5]], dtype=torch.long)
    with torch.inference_mode():
        hidden = seed_model.transformer(
            input_ids=token_ids,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state[0, -1]
        direction = hidden / hidden.norm()

    models = {
        "favored": copy.deepcopy(seed_model),
        "suppressed": copy.deepcopy(seed_model),
    }
    with torch.no_grad():
        # The target token never occurs in the prompt. Its tied output embedding
        # therefore changes target likelihood without changing the prompt hidden
        # state. One checkpoint favors the fixed target; the other suppresses it.
        models["favored"].transformer.wte.weight[6].copy_(4.0 * direction)
        models["suppressed"].transformer.wte.weight[6].copy_(-4.0 * direction)

    checkpoints: dict[str, Path] = {}
    tokenizer_digest: str | None = None
    for role, model in models.items():
        model.eval()
        checkpoint = tmp_path / "models" / role
        checkpoint.mkdir(parents=True)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer = _tokenizer(checkpoint, transformers, tokenizers)
        observed_tokenizer_digest = hf_tokenizer_contract_sha256(tokenizer)
        if tokenizer_digest is None:
            tokenizer_digest = observed_tokenizer_digest
        else:
            assert observed_tokenizer_digest == tokenizer_digest
        checkpoints[role] = checkpoint

    assert checkpoint_tree_sha256(checkpoints["favored"]) != checkpoint_tree_sha256(
        checkpoints["suppressed"]
    )
    assert tokenizer_digest is not None
    return checkpoints, tokenizer_digest


def _settings(checkpoint: Path, tokenizer_digest: str) -> dict[str, JSONScalar]:
    return {
        "batch_size": 1,
        "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
        "context_length": 8,
        "max_output_tokens": 1,
        "offline": True,
        "seed": 20_260_716,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": tokenizer_digest,
    }


def _request(
    root: Path,
    *,
    checkpoints: dict[str, Path],
    tokenizer_digest: str,
    policy_path: Path,
    evidence_name: str,
    dataset_sha256: str,
    comparison_order: dict[str, str],
    metric: str,
) -> tuple[Path, dict[str, str]]:
    settings = {
        role: _settings(checkpoint, tokenizer_digest)
        for role, checkpoint in checkpoints.items()
    }

    def side(role: str) -> dict[str, object]:
        checkpoint_name = comparison_order[role]
        model_id = f"invarlock-behavior-proof/{checkpoint_name}"
        return {
            "artifact": {
                "path": checkpoints[checkpoint_name].relative_to(root).as_posix(),
                "model_id": model_id,
                "locator": f"hf://{model_id}@{'a' * 40}",
            },
            "runtime": {
                "provider": "hf_transformers",
                "settings": settings[checkpoint_name],
            },
        }

    payload = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": dataset_sha256,
                "format": "jsonl",
                "name": "hf-release-regression-proof",
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": policy_path.relative_to(root).as_posix(),
            "task": "text_causal",
            "metric": metric,
        },
        "execution": {"mode": "run"},
        "output": {"evidence": evidence_name},
    }
    request_path = root / f"request-{evidence_name}.yaml"
    request_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    provider = HFTransformersProvider()
    artifact_anchors = {
        role: "sha256:"
        + artifact_identity_sha256(
            provider.identify_artifact(
                ModelRuntimeSpec(
                    provider_name="hf_transformers",
                    model_id=("invarlock-behavior-proof/" + comparison_order[role]),
                    settings=settings[comparison_order[role]],
                )
            )
        )
        for role in ("baseline", "subject")
    }
    return request_path, artifact_anchors


def _policy(path: Path, *, metric: str, ratio_max: float) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {"resolved_policy": {"metrics": {metric: {"ratio_max": ratio_max}}}}
        )
    )
    return path


def _report(evidence: Path) -> dict[str, Any]:
    value = json.loads((evidence / EVIDENCE_PATHS["evaluation_report"]).read_bytes())
    assert isinstance(value, dict)
    return value


def test_real_hf_scores_drive_accepted_and_rejected_normalized_nll_decisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This test exercises real Torch/Transformers scoring in process. The only
    # substituted facts are the container-boundary observations that cannot be
    # obtained inside a normal pytest worker.
    image_ref = f"registry.invalid/invarlock@{_IMAGE_DIGEST}"
    for module in (hf_transformers, runtime_transaction):
        monkeypatch.setattr(module, "strict_container_boundary_present", lambda: True)
        monkeypatch.setattr(
            module, "resolve_runtime_image_digest", lambda: _IMAGE_DIGEST
        )
        monkeypatch.setattr(module, "resolve_runtime_image", lambda: image_ref)

    root = tmp_path / "comparison"
    (root / "inputs").mkdir(parents=True)
    checkpoints, tokenizer_digest = _distinct_checkpoints(root)
    dataset_bytes = (
        b'{"id":"regression-1","prompt":"alpha beta","expected":" target"}\n'
    )
    dataset_path = root / "inputs" / "records.jsonl"
    dataset_path.write_bytes(dataset_bytes)
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    expected_schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset_path,
            sha256=dataset_sha256,
            name="hf-release-regression-proof",
            split="validation",
            input_field="prompt",
            expected_output_field="expected",
            id_field="id",
        ),
        dataset_bytes,
    )
    evidence_key, evidence_fingerprint = _signing_key(tmp_path / "evidence-key.pem")
    verifier_key, _ = _signing_key(tmp_path / "verifier-key.pem")
    runtime_resources = CallerRuntimeResources(container_image_digest=_IMAGE_DIGEST)

    metric_kinds = {"normalized_nll_per_utf8_byte": "normalized_nll_ratio"}
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in metric_kinds:
        metric_slug = metric.replace("_", "-")
        cases = {
            "accepted": {
                "policy": _policy(
                    root / "inputs" / f"{metric_slug}-accepted-policy.json",
                    metric=metric,
                    ratio_max=_ACCEPTED_RATIO_MAX,
                ),
                "order": {"baseline": "suppressed", "subject": "favored"},
            },
            "rejected": {
                "policy": _policy(
                    root / "inputs" / f"{metric_slug}-rejected-policy.json",
                    metric=metric,
                    ratio_max=_REJECTED_RATIO_MAX,
                ),
                "order": {"baseline": "favored", "subject": "suppressed"},
            },
        }
        metric_reports: dict[str, dict[str, Any]] = {}
        for verdict, case in cases.items():
            policy_path = case["policy"]
            comparison_order = case["order"]
            assert isinstance(policy_path, Path)
            assert isinstance(comparison_order, dict)
            evidence_name = f"{metric_slug}-{verdict}-evidence"
            request_path, artifact_anchors = _request(
                root,
                checkpoints=checkpoints,
                tokenizer_digest=tokenizer_digest,
                policy_path=policy_path,
                evidence_name=evidence_name,
                dataset_sha256=dataset_sha256,
                comparison_order=comparison_order,
                metric=metric,
            )
            evaluated = evaluate_request_file(
                request_path,
                signing_key_path=evidence_key,
                resource_resolver=runtime_resources,
            )
            metric_reports[verdict] = _report(evaluated.evidence_path)
            receipt_path = tmp_path / f"{metric_slug}-{verdict}-receipt.json"
            verify_arguments: dict[str, Any] = {
                "policy_path": policy_path,
                "expected_baseline_artifact": artifact_anchors["baseline"],
                "expected_subject_artifact": artifact_anchors["subject"],
                "expected_schedule": f"sha256:{expected_schedule.schedule_sha256}",
                "expected_baseline_runtime": _IMAGE_DIGEST,
                "expected_subject_runtime": _IMAGE_DIGEST,
                "expected_signer": evidence_fingerprint,
                "receipt_path": receipt_path,
                "verifier_signing_key_path": verifier_key,
                "verifier_identity": "integration/hf-release-regression",
            }
            if verdict == "accepted":
                verified = verify_evidence(evaluated.evidence_path, **verify_arguments)
                assert verified.payload["ok"] is True
                assert verified.payload["policy_verdict"] == "pass"
            else:
                with pytest.raises(EvidenceVerificationError) as caught:
                    verify_evidence(evaluated.evidence_path, **verify_arguments)
                assert caught.value.exit_code == int(EvidencePackStatus.REPORTS)
                assert caught.value.payload["integrity_ok"] is True
                assert caught.value.payload["policy_verdict"] == "fail"
            assert receipt_path.is_file()
        reports[metric] = metric_reports

    for metric, comparison_kind in metric_kinds.items():
        accepted = reports[metric]["accepted"]
        rejected = reports[metric]["rejected"]
        assert accepted["metric"] == metric
        assert accepted["comparison"]["kind"] == comparison_kind
        assert accepted["verdict"] == "pass"
        assert rejected["verdict"] == "fail"
        assert accepted["comparison"]["value"] * rejected["comparison"][
            "value"
        ] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
        assert 0.0 < accepted["comparison"]["value"] < _MAXIMUM_PROVEN_IMPROVEMENT_RATIO
        assert _MINIMUM_PROVEN_REGRESSION_RATIO < rejected["comparison"]["value"]
        assert rejected["comparison"]["value"] > _REJECTED_RATIO_MAX
        assert accepted["baseline"]["mean_score"] > accepted["subject"]["mean_score"]
        assert rejected["baseline"]["mean_score"] < rejected["subject"]["mean_score"]
        assert accepted["derived_measurements"]["perplexity_ratio"]["status"] == (
            "available"
        )
        assert rejected["derived_measurements"]["perplexity_ratio"]["status"] == (
            "available"
        )
