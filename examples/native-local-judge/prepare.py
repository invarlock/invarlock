"""Prepare a bounded local-judge starter from inspected, local model material.

This authenticates files and freezes the plan. It does not generate model output.
The bundled answer pair is illustrative fixture data, not measured model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from invarlock.core.runtime_provider import artifact_identity_sha256
from invarlock.engine import ModelRuntimeSpec, prepare_evaluator_judge
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider
from invarlock.runtime_providers.llama_cpp import LlamaCppProvider

HERE = Path(__file__).resolve().parent


def prepare(workspace: Path, model_path: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    model = json.loads(model_path.read_bytes())
    provider_name = model["runtime"]["provider"]
    provider: HFTransformersProvider | LlamaCppProvider
    if provider_name == "hf_transformers":
        provider = HFTransformersProvider()
    elif provider_name == "llama_cpp":
        provider = LlamaCppProvider()
    else:
        raise ValueError("choose hf_transformers or llama_cpp")
    reference = model["artifact"]["path"]
    relative = Path(reference)
    if (
        not relative.parts
        or relative.as_posix() != reference
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("judge artifact must be beneath the workspace")
    artifact_path = workspace / relative
    for depth in range(1, len(relative.parts) + 1):
        part = workspace.joinpath(*relative.parts[:depth])
        if part.is_symlink():
            raise ValueError("judge artifact path must not contain symlinks")
    spec = ModelRuntimeSpec(
        provider_name=provider_name,
        model_id=model["artifact"]["model_id"],
        settings=model["runtime"]["settings"],
    )
    artifact = provider.authenticate_artifact(spec, artifact_path)
    artifact_digest = artifact_identity_sha256(artifact)
    recipe = json.loads((HERE / "recipe-template.json").read_bytes())
    recipe["plan"]["judge"].update(
        provider=provider_name,
        requested_model=spec.model_id,
        approved_resolved_models=[spec.model_id],
        model_identity={"kind": "local_weights", "weights_sha256": artifact_digest},
    )
    recipe["plan"]["judge"]["config"].update(
        seed=spec.settings["seed"],
        max_output_tokens=spec.settings["max_output_tokens"],
    )
    baseline = json.loads((HERE / "baseline_run.json").read_bytes())
    subject = json.loads((HERE / "subject_run.json").read_bytes())
    plan, policy = prepare_evaluator_judge(recipe, baseline, subject)
    recipe["collection"]["max_calls"] = plan["schedule"]["expected_trials"]
    recipe["collection"]["max_output_tokens"] = plan["schedule"][
        "expected_trials"
    ] * cast(int, spec.settings["max_output_tokens"])
    request = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {
            "mode": "judge_collect",
            "collection": {
                "integration": "runtime-provider-judge",
                "configuration": "collection.json",
                "model": model,
                "workspace": "judge-work",
                "scorer_id": recipe["runner"]["scorer_id"],
            },
        },
        "comparison": {
            "baseline_run": "baseline_run.json",
            "subject_run": "subject_run.json",
            "plan": "plan.json",
            "measurements": None,
            "policy": "analysis-policy.json",
        },
        "output": {"evidence": "evidence", "signer_identity": "local-judge-signer"},
    }
    files = {
        "baseline_run.json": baseline,
        "subject_run.json": subject,
        "plan.json": plan,
        "analysis-policy.json": policy,
        "collection.json": recipe["collection"],
        "request.json": request,
    }
    if any(
        (workspace / name).exists() or (workspace / name).is_symlink() for name in files
    ):
        raise ValueError("generated destinations must be new")
    encoded = {
        name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        for name, value in files.items()
    }
    for name, text in encoded.items():
        with (workspace / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    return {
        "provider": provider_name,
        "judge_artifact_identity_sha256": artifact_digest,
        "expected_trials": plan["schedule"]["expected_trials"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.model), sort_keys=True))


if __name__ == "__main__":
    main()
