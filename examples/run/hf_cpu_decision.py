#!/usr/bin/env python3
"""Prepare and optionally execute a real Hugging Face CPU comparison.

The example creates two tiny, distinct GPT-2 checkpoints.  The baseline
suppresses the fixed expected token while the subject favors it.  A successful
decision therefore depends on real Transformers scoring rather than identical
fixtures or a prewritten report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, normalize_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
)

_SEED = 20_260_716
_MODEL_IDS = {
    "baseline": "invarlock-example/tiny-suppressed-target",
    "subject": "invarlock-example/tiny-favored-target",
}


@dataclass(frozen=True)
class ExamplePaths:
    """The paths shared by preparation and the three public commands."""

    root: Path
    evaluation: Path
    request: Path
    evidence: Path
    verifier: Path
    trusted_inputs: Path
    independent_policy: Path
    evidence_key: Path
    verifier_key: Path
    receipt: Path
    html_report: Path


def _paths(root: Path) -> ExamplePaths:
    evaluation = root / "evaluation"
    verifier = root / "verifier"
    return ExamplePaths(
        root=root,
        evaluation=evaluation,
        request=evaluation / "request.yaml",
        evidence=evaluation / "evidence",
        verifier=verifier,
        trusted_inputs=verifier / "trusted-inputs.json",
        independent_policy=verifier / "policy" / "acceptance.json",
        evidence_key=root / "keys" / "evidence-signer.pem",
        verifier_key=verifier / "keys" / "verifier.pem",
        receipt=verifier / "verification.receipt.json",
        html_report=root / "comparison-report.html",
    )


def _write_private_key(path: Path) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return public_key_fingerprint(key.public_key())


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


def _create_distinct_checkpoints(paths: ExamplePaths) -> tuple[dict[str, Path], str]:
    try:
        import safetensors  # noqa: F401
        import tokenizers  # type: ignore[import-untyped]
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "the example requires the InvarLock Hugging Face dependencies"
        ) from exc

    torch.manual_seed(_SEED)
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
    prompt_ids = torch.tensor([[4, 5]], dtype=torch.long)
    with torch.inference_mode():
        hidden = seed_model.transformer(
            input_ids=prompt_ids,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state[0, -1]
        direction = hidden / hidden.norm()

    models = {
        "baseline": copy.deepcopy(seed_model),
        "subject": copy.deepcopy(seed_model),
    }
    with torch.no_grad():
        # Token 6 does not occur in the prompt.  Changing only its tied output
        # embedding changes the expected-token likelihood without changing the
        # prompt hidden state used to construct this comparison.
        models["baseline"].transformer.wte.weight[6].copy_(-4.0 * direction)
        models["subject"].transformer.wte.weight[6].copy_(4.0 * direction)

    checkpoints: dict[str, Path] = {}
    tokenizer_digest: str | None = None
    for role in ("baseline", "subject"):
        checkpoint = paths.evaluation / "models" / role
        checkpoint.mkdir(parents=True)
        models[role].eval()
        models[role].save_pretrained(checkpoint, safe_serialization=True)
        tokenizer = _tokenizer(checkpoint, transformers, tokenizers)
        observed_digest = hf_tokenizer_contract_sha256(tokenizer)
        if tokenizer_digest is None:
            tokenizer_digest = observed_digest
        elif observed_digest != tokenizer_digest:
            raise RuntimeError("the generated tokenizer contracts do not match")
        checkpoints[role] = checkpoint

    if checkpoint_tree_sha256(checkpoints["baseline"]) == checkpoint_tree_sha256(
        checkpoints["subject"]
    ):
        raise RuntimeError("the generated checkpoints are not distinct")
    assert tokenizer_digest is not None
    return checkpoints, tokenizer_digest


def _settings(checkpoint: Path, tokenizer_digest: str) -> dict[str, JSONScalar]:
    return {
        "batch_size": 1,
        "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
        "context_length": 8,
        "max_output_tokens": 1,
        "offline": True,
        "seed": _SEED,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": tokenizer_digest,
    }


def _prepare_workspace(
    root: Path, *, runtime_image_digest: str
) -> tuple[ExamplePaths, dict[str, str]]:
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"workspace already exists: {root}; choose a new disposable path"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    paths = _paths(root)
    try:
        (paths.evaluation / "inputs").mkdir(parents=True)
        paths.independent_policy.parent.mkdir(parents=True)
        paths.evidence_key.parent.mkdir(parents=True)
        paths.verifier_key.parent.mkdir(parents=True)

        checkpoints, tokenizer_digest = _create_distinct_checkpoints(paths)
        settings = {
            role: _settings(checkpoints[role], tokenizer_digest)
            for role in ("baseline", "subject")
        }

        dataset_bytes = (
            b'{"id":"expected-token","prompt":"alpha beta","expected":" target"}\n'
        )
        dataset = paths.evaluation / "inputs" / "records.jsonl"
        dataset.write_bytes(dataset_bytes)
        dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()

        schedule = prepare_local_evaluation_schedule_bytes(
            LocalDatasetRequest(
                path=dataset,
                sha256=dataset_sha256,
                name="tiny-hf-release-decision",
                split="validation",
                input_field="prompt",
                expected_output_field="expected",
                id_field="id",
            ),
            dataset_bytes,
        )

        policy_bytes = canonical_json_bytes(
            {
                "resolved_policy": {
                    "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.0}}
                }
            }
        )
        request_policy = paths.evaluation / "inputs" / "acceptance.json"
        request_policy.write_bytes(policy_bytes)
        paths.independent_policy.write_bytes(policy_bytes)

        def side(role: str) -> dict[str, object]:
            return {
                "artifact": {
                    "path": f"models/{role}",
                    "model_id": _MODEL_IDS[role],
                    "locator": f"hf://{_MODEL_IDS[role]}@{'a' * 40}",
                },
                "runtime": {
                    "provider": "hf_transformers",
                    "settings": settings[role],
                },
            }

        request = {
            "format_version": "invarlock/evaluation-request-v1",
            "comparison": {
                "baseline": side("baseline"),
                "subject": side("subject"),
                "dataset": {
                    "path": "inputs/records.jsonl",
                    "sha256": dataset_sha256,
                    "format": "jsonl",
                    "name": "tiny-hf-release-decision",
                    "split": "validation",
                    "input_field": "prompt",
                    "expected_output_field": "expected",
                    "id_field": "id",
                },
                "policy": "inputs/acceptance.json",
                "task": "text_causal",
                "metric": "normalized_nll_per_utf8_byte",
            },
            "execution": {"mode": "run"},
            "output": {"evidence": "evidence"},
        }
        paths.request.write_text(
            yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
        )

        provider = HFTransformersProvider()
        artifact_anchors = {
            role: "sha256:"
            + artifact_identity_sha256(
                provider.identify_artifact(
                    ModelRuntimeSpec(
                        provider_name="hf_transformers",
                        model_id=_MODEL_IDS[role],
                        settings=settings[role],
                    )
                )
            )
            for role in ("baseline", "subject")
        }
        evidence_signer = _write_private_key(paths.evidence_key)
        verifier = _write_private_key(paths.verifier_key)
        paths.evidence_key.with_suffix(".fingerprint").write_text(
            evidence_signer + "\n", encoding="ascii"
        )
        paths.verifier_key.with_suffix(".fingerprint").write_text(
            verifier + "\n", encoding="ascii"
        )
        anchors = {
            "baseline_artifact_digest": artifact_anchors["baseline"],
            "subject_artifact_digest": artifact_anchors["subject"],
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": normalize_digest(
                runtime_image_digest, label="runtime image digest"
            ),
            "subject_runtime_digest": normalize_digest(
                runtime_image_digest, label="runtime image digest"
            ),
            "evidence_signer_fingerprint": evidence_signer,
        }
        trust_profile = {
            "format": "invarlock/trust-inputs-v1",
            "policy": {"path": "policy/acceptance.json"},
            "anchors": anchors,
            "verifier": {
                "identity": "local-hf-cpu-example",
                "signing_key_path": "keys/verifier.pem",
            },
            "allow_installed_scorers": False,
        }
        paths.trusted_inputs.write_bytes(canonical_json_bytes(trust_profile))
        return paths, anchors
    except Exception:
        shutil.rmtree(root)
        raise


def _run(command: list[str]) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {rendered}"
        )


def _execute(
    paths: ExamplePaths,
    *,
    container_engine: str,
    runtime_image: str,
    runtime_image_digest: str,
) -> None:
    base = [sys.executable, "-m", "invarlock"]
    _run(
        [
            *base,
            "evaluate",
            str(paths.request),
            "--signing-key",
            str(paths.evidence_key),
            "--container-engine",
            container_engine,
            "--runtime-image",
            runtime_image,
            "--runtime-image-digest",
            runtime_image_digest,
            "--runtime-device",
            "cpu",
            "--json",
        ]
    )
    _run(
        [
            *base,
            "verify",
            str(paths.evidence),
            "--trust-profile",
            str(paths.trusted_inputs),
            "--receipt",
            str(paths.receipt),
            "--json",
        ]
    )
    _run(
        [
            *base,
            "report",
            str(paths.evidence),
            "--html",
            str(paths.html_report),
        ]
    )
    report = json.loads(
        (paths.evidence / "reports" / "evaluation.report.json").read_text(
            encoding="utf-8"
        )
    )
    ratio = report.get("comparison", {}).get("value")
    if report.get("verdict") != "pass" or not isinstance(ratio, (int, float)):
        raise RuntimeError("the evaluated comparison did not produce a passing ratio")
    print(f"PASS subject normalized-NLL ratio: {ratio:.6f}")
    print(f"Evidence: {paths.evidence}")
    print(f"Receipt: {paths.receipt}")
    print(f"Report: {paths.html_report}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="New disposable workspace; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare inputs and print the canonical commands without running OCI.",
    )
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument(
        "--runtime-image",
        help="Locally available CPU runtime image reference used by evaluate.",
    )
    parser.add_argument(
        "--runtime-image-digest",
        required=True,
        help=(
            "Pinned sha256 digest of the local CPU runtime image; required to "
            "close the generated verifier trust profile."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.prepare_only and not arguments.runtime_image:
        raise SystemExit("full execution requires --runtime-image")
    try:
        paths, _anchors = _prepare_workspace(
            arguments.workspace,
            runtime_image_digest=arguments.runtime_image_digest,
        )
        print(f"Prepared: {paths.root}")
        print(f"Request: {paths.request}")
        print(f"Independent trust inputs: {paths.trusted_inputs}")
        print(f"Keys outside request tree: {paths.evidence_key.parent}")
        if arguments.prepare_only:
            print(
                "Use the generated request and trust profile with evaluate, verify, "
                "and report; the checked-in README contains the complete commands."
            )
            return 0
        _execute(
            paths,
            container_engine=arguments.container_engine,
            runtime_image=arguments.runtime_image,
            runtime_image_digest=arguments.runtime_image_digest,
        )
    except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
