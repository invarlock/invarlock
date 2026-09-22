"""Freeze an endpoint judge request without contacting the server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from invarlock.engine import prepare_evaluator_judge
from invarlock.judge_measurements.openai_compatible import (
    openai_compatible_service_identity,
    validate_openai_compatible_collection,
)

HERE = Path(__file__).resolve().parent


def prepare(
    workspace: Path,
    baseline_path: Path,
    subject_path: Path,
    *,
    service: str,
    base_url: str,
    model: str,
    authentication: str = "none",
    reference_mode: str = "per_case",
    max_input_bytes: int | None = None,
) -> dict:
    workspace = workspace.resolve(strict=True)
    recipe = json.loads((HERE / "recipe-template.json").read_bytes())
    recipe["plan"]["judge"].update(
        requested_model=model, approved_resolved_models=[model]
    )
    recipe["plan"]["prompt"]["reference_mode"] = reference_mode
    collection = recipe["collection"]
    collection.update(
        service=service,
        base_url=base_url,
        model=model,
        authentication=authentication,
    )
    if service == "lm_studio":
        collection["response_format"] = "json_schema"
    recipe["plan"]["judge"]["service_identity"] = openai_compatible_service_identity(
        collection
    )
    baseline = json.loads(baseline_path.read_bytes())
    subject = json.loads(subject_path.read_bytes())
    plan, policy = prepare_evaluator_judge(recipe, baseline, subject)
    calls = plan["schedule"]["expected_trials"]
    collection.update(
        max_calls=calls,
        max_input_bytes=calls * 65536 if max_input_bytes is None else max_input_bytes,
        max_output_tokens=calls * plan["judge"]["config"]["max_output_tokens"],
    )
    validate_openai_compatible_collection(collection, plan)
    request = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {
            "mode": "judge_collect",
            "collection": {
                "integration": "openai-compatible-judge",
                "configuration": "collection.json",
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
        "output": {"evidence": "evidence", "signer_identity": "endpoint-judge-signer"},
    }
    files = {
        "baseline_run.json": baseline,
        "subject_run.json": subject,
        "plan.json": plan,
        "analysis-policy.json": policy,
        "collection.json": collection,
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
    return {"service": service, "requested_model": model, "expected_trials": calls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument(
        "--service",
        choices=["vllm", "ollama", "lm_studio", "openai_compatible"],
        required=True,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--authentication", choices=["none", "bearer_env"], default="none"
    )
    parser.add_argument(
        "--reference-mode", choices=["per_case", "none"], default="per_case"
    )
    parser.add_argument("--max-input-bytes", type=int)
    args = parser.parse_args()
    result = prepare(
        args.workspace,
        args.baseline,
        args.subject,
        service=args.service,
        base_url=args.base_url,
        model=args.model,
        authentication=args.authentication,
        reference_mode=args.reference_mode,
        max_input_bytes=args.max_input_bytes,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
