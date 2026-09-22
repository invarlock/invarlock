"""Real offline local inference tests execution, not judge quality.

Run inside the authenticated runtime container with the explicit opt-in below.
The tiny random checkpoint intentionally cannot produce a valid rubric rating.
No provider, model loading, inference, receipt, or replay implementation is mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    artifact_identity_sha256,
)
from invarlock.judge_measurements.contracts import (
    measurement_plan_digest,
    render_judge_request,
)
from invarlock.judge_measurements.evidence import (
    publish_judge_evidence,
    replay_judge_evidence,
)
from invarlock.judge_measurements.reporting import render_judge_evidence
from invarlock.judge_measurements.runtime_provider import collect_runtime_provider
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
)
from tests.judge_measurements.test_analysis import FIXTURES, _bundle, _runs

pytestmark = pytest.mark.skipif(
    os.environ.get("INVARLOCK_RUN_LOCAL_JUDGE_SMOKE") != "1",
    reason="requires an authenticated offline runtime container and explicit opt-in",
)


def test_real_local_judge_preserves_invalid_ratings_and_replays(tmp_path: Path):
    import tokenizers
    import torch
    import transformers

    torch.manual_seed(17)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=2048,
            n_embd=8,
            n_layer=1,
            n_head=1,
            eos_token_id=None,
            bos_token_id=1,
            pad_token_id=0,
        )
    )
    model.eval()
    artifact = tmp_path / "model"
    model.save_pretrained(artifact, safe_serialization=True)
    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<unk>": 2,
                **{f"word{i}": i + 3 for i in range(29)},
            },
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(artifact)
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="local-judge-execution-test",
        settings={
            "checkpoint_tree_sha256": checkpoint_tree_sha256(artifact),
            "tokenizer_metadata_sha256": hf_tokenizer_contract_sha256(tokenizer),
            "offline": True,
            "seed": 17,
            "context_length": 2048,
            "batch_size": 1,
            "max_output_tokens": 2,
            "timeout_seconds": 30,
        },
    )
    provider = HFTransformersProvider()
    identity = provider.authenticate_artifact(spec, artifact)
    plan, _ = _bundle(groups=("one",))
    plan["judge"].update(
        provider=provider.name,
        requested_model=spec.model_id,
        approved_resolved_models=[spec.model_id],
        model_identity={
            "kind": "local_weights",
            "weights_sha256": artifact_identity_sha256(identity),
        },
    )
    plan["judge"]["config"].update(
        temperature="0",
        top_p="1",
        reasoning_effort=None,
        seed=17,
        max_output_tokens=2,
    )
    plan["schedule"].update(max_attempts=1, retry_on=[], cache="forbid")
    baseline, subject = _runs(plan)
    for binding in plan["answer_bindings"]:
        for side, run in (("baseline", baseline), ("subject", subject)):
            record = next(r for r in run["records"] if r["id"] == binding["case_id"])
            request = render_judge_request(
                plan,
                input_text=record["input"],
                answer_text=record["output"],
                reference_text=record["expected"],
            )
            binding[f"{side}_request_sha256"] = hashlib.sha256(request).hexdigest()
    measurements = collect_runtime_provider(
        plan,
        provider=provider,
        spec=spec,
        resources=RuntimeArtifactResources(
            root=tmp_path,
            primary_artifact="model",
            support_resources={},
            device_kind="cpu",
            container_image_digest=os.environ["INVARLOCK_RUNTIME_IMAGE_DIGEST"],
        ),
        baseline_run=baseline,
        subject_run=subject,
    )
    assert len(measurements["trials"]) == 2
    assert all(t["attempts"] for t in measurements["trials"])
    assert measurements["completeness"]["completed_trials"] == 0
    assert measurements["source_profile"] == "retained-runtime-provider-judge-v1"
    policy = json.loads((FIXTURES / "analysis_policy.json").read_bytes())
    policy["plan_sha256"] = measurement_plan_digest(plan)
    published = publish_judge_evidence(
        tmp_path / "evidence",
        plan=plan,
        measurements=measurements,
        baseline_run=baseline,
        subject_run=subject,
        analysis_policy=policy,
    )
    assert published.analysis_result.to_dict()["decision"] == "insufficient_evidence"
    assert (
        replay_judge_evidence(published.path).analysis_result
        == published.analysis_result
    )
    report = render_judge_evidence(published.path, html_path=tmp_path / "report.html")
    assert not report.errors
    assert "Judge artifact identity" in report.text
