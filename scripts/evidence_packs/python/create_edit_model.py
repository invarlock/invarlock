from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from .editing.streaming_pruning import materialize_magnitude_pruned_artifact
    from .editing.streaming_transform import materialize_transformation_artifact
    from .editing.training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        TrainingProfileError,
        load_training_profile,
    )
    from .editing.training_runtime import (
        TrainingRuntimeError,
        run_training_profile,
        verify_training_artifact,
    )
    from .editing.transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.streaming_pruning import materialize_magnitude_pruned_artifact
    from editing.streaming_transform import materialize_transformation_artifact
    from editing.training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        TrainingProfileError,
        load_training_profile,
    )
    from editing.training_runtime import (
        TrainingRuntimeError,
        run_training_profile,
        verify_training_artifact,
    )
    from editing.transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )


def _configure_determinism() -> None:
    """Disable gradients; streaming transformations use canonical CPU math."""

    torch.set_grad_enabled(False)


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _canonical_transformation_inputs(
    args: argparse.Namespace,
    *,
    edit_type: str,
) -> tuple[dict[str, object], str]:
    """Turn CLI fields into exactly the replay contract's typed input."""

    raw_parameters: dict[str, object]
    if edit_type == "quant_rtn":
        raw_parameters = {
            "bits": int(args.bits),
            "group_size": int(args.group_size),
        }
    elif edit_type == SYNTHETIC_LOWRANK_DELTA:
        raw_parameters = {"rank": int(args.rank), "scale": float(args.scale)}
    elif edit_type == SYNTHETIC_DENSE_UPDATE:
        raw_parameters = {
            "step_size": float(args.step_size),
            "iterations": int(args.iterations),
        }
    else:  # defensive: the CLI exposes only the three supported families
        raise TransformationContractError(
            f"{edit_type!r} has no verifier-grade transformation contract"
        )

    specification = canonical_transformation_spec(edit_type, raw_parameters)
    parameters = specification.get("parameters")
    if not isinstance(parameters, Mapping):  # contract invariant
        raise TransformationContractError("canonical transformation parameters missing")
    return dict(parameters), validate_transformation_scope(str(args.scope))


def _create_streaming_transformation(
    args: argparse.Namespace,
    *,
    edit_type: str,
) -> int:
    """Materialize a replayable subject without loading a mutable model."""

    _configure_determinism()
    parameters, scope = _canonical_transformation_inputs(args, edit_type=edit_type)
    max_output_shard_mib = int(getattr(args, "max_output_shard_mib", 1024))
    result = materialize_transformation_artifact(
        baseline_path=Path(args.baseline_path),
        output_path=Path(args.output_path),
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
        max_output_shard_bytes=max_output_shard_mib * 1024 * 1024,
        restart=bool(getattr(args, "restart", False)),
    )
    print(
        "Materialized replayable transformation "
        f"{edit_type} over {result['selected_tensors']} tensors "
        f"({result['selected_params']:,} parameters; "
        f"{result['actual_changes']['value_changed_params']:,} changed)."
    )
    print(f"Saved replayable edited model to {args.output_path}")
    return 0


def _create_quant_rtn(args: argparse.Namespace) -> int:
    return _create_streaming_transformation(args, edit_type="quant_rtn")


def _create_magnitude_prune(args: argparse.Namespace) -> int:
    sparsity = float(args.sparsity)
    scope = str(args.scope)
    result = materialize_magnitude_pruned_artifact(
        baseline_path=Path(args.baseline_path),
        output_path=Path(args.output_path),
        sparsity=sparsity,
        scope=scope,
        max_output_shard_bytes=int(args.max_output_shard_mib) * 1024 * 1024,
        restart=bool(args.restart),
    )
    print(
        "Pruned "
        f"{result['selected_tensors']} tensors "
        f"({result['selected_params']:,} parameters; "
        f"{result['effective_changed_params']:,} changed) on {result['device']}"
    )
    print(f"Saved replayable pruned model to {args.output_path}")
    return 0


def _create_synthetic_lowrank_delta(args: argparse.Namespace) -> int:
    return _create_streaming_transformation(
        args,
        edit_type=SYNTHETIC_LOWRANK_DELTA,
    )


def _create_synthetic_dense_update(args: argparse.Namespace) -> int:
    return _create_streaming_transformation(
        args,
        edit_type=SYNTHETIC_DENSE_UPDATE,
    )


def _create_training_profile(args: argparse.Namespace) -> int:
    """Run one immutable tiny-model training profile and publish its receipt."""

    repo_root = Path(args.repo_root).resolve()
    profile = load_training_profile(
        str(args.profile_id),
        profiles_path=Path(args.profiles_path).resolve(),
        repo_root=repo_root,
    )
    result = run_training_profile(
        profile,
        Path(args.output_path),
        repo_root=repo_root,
        local_files_only=not bool(args.allow_network),
    )
    print(
        json.dumps(
            {
                "edit_type": profile.edit_type,
                "profile_id": profile.profile_id,
                "profile_sha256": profile.profile_sha256,
                "receipt_path": str(result.receipt_path),
                "receipt_sha256": result.receipt["receipt_sha256"],
                "subject_dir": str(result.subject_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_training_profile(args: argparse.Namespace) -> int:
    """Independently recompute a published tiny-training artifact contract."""

    repo_root = Path(args.repo_root).resolve()
    profile = load_training_profile(
        str(args.profile_id),
        profiles_path=Path(args.profiles_path).resolve(),
        repo_root=repo_root,
    )
    receipt = verify_training_artifact(
        profile,
        Path(args.subject_path),
        repo_root=repo_root,
        local_files_only=not bool(args.allow_network),
    )
    print(
        json.dumps(
            {
                "profile_id": profile.profile_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "status": "verified",
                "subject_dir": str(Path(args.subject_path).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("baseline_path")
    parser.add_argument("output_path")


def _add_streaming_materialization_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-output-shard-mib",
        type=int,
        default=1024,
        help="Maximum materialized safetensors shard size (default: 1024 MiB).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard a stale resumable transformation staging directory before starting.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a single edited evidence-pack subject checkpoint."
    )
    subparsers = parser.add_subparsers(dest="edit_type", required=True)

    quant = subparsers.add_parser("quant-rtn")
    _add_common_paths(quant)
    quant.add_argument("bits")
    quant.add_argument("group_size")
    quant.add_argument("scope")
    _add_streaming_materialization_options(quant)
    quant.set_defaults(func=_create_quant_rtn)

    prune = subparsers.add_parser("magnitude-prune")
    _add_common_paths(prune)
    prune.add_argument("sparsity")
    prune.add_argument("scope")
    _add_streaming_materialization_options(prune)
    prune.set_defaults(func=_create_magnitude_prune)

    synthetic_lowrank = subparsers.add_parser("synthetic-lowrank-delta")
    _add_common_paths(synthetic_lowrank)
    synthetic_lowrank.add_argument("rank")
    synthetic_lowrank.add_argument("scale")
    synthetic_lowrank.add_argument("scope")
    _add_streaming_materialization_options(synthetic_lowrank)
    synthetic_lowrank.set_defaults(func=_create_synthetic_lowrank_delta)

    synthetic_dense = subparsers.add_parser("synthetic-dense-update")
    _add_common_paths(synthetic_dense)
    synthetic_dense.add_argument("step_size")
    synthetic_dense.add_argument("iterations")
    synthetic_dense.add_argument("scope")
    _add_streaming_materialization_options(synthetic_dense)
    synthetic_dense.set_defaults(func=_create_synthetic_dense_update)

    training = subparsers.add_parser(
        "train-profile",
        help=(
            "Run an immutable tiny-model training profile, verify its "
            "artifacts, and publish the subject atomically."
        ),
    )
    training.add_argument("profile_id")
    training.add_argument("output_path")
    training.add_argument(
        "--profiles-path",
        default=str(DEFAULT_TRAINING_PROFILES_PATH),
        help="Immutable training-profile JSON document.",
    )
    training.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root used to resolve vendored training data.",
    )
    training.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow retrieval of the pinned model and tokenizer revision.",
    )
    training.set_defaults(func=_create_training_profile)

    training_verify = subparsers.add_parser(
        "verify-training-profile",
        help="Recompute artifact evidence for a tiny training-profile subject.",
    )
    training_verify.add_argument("profile_id")
    training_verify.add_argument("subject_path")
    training_verify.add_argument(
        "--profiles-path",
        default=str(DEFAULT_TRAINING_PROFILES_PATH),
        help="Immutable training-profile JSON document.",
    )
    training_verify.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root used to resolve vendored training data.",
    )
    training_verify.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow retrieval of the pinned model and tokenizer revision.",
    )
    training_verify.set_defaults(func=_verify_training_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            return int(args.func(args))
        except (
            TrainingProfileError,
            TrainingRuntimeError,
            TransformationContractError,
            ValueError,
            OSError,
            RuntimeError,
        ) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    finally:
        _clear_memory()


if __name__ == "__main__":
    raise SystemExit(main())
