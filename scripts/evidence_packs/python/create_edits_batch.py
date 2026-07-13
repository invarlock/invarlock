from __future__ import annotations

import argparse
import gc
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from .editing.implementations import (
        generated_transformation_edit_dir_name,
        parse_edit_specs_json,
        real_training_edit_migration_message,
        resolve_batch_entry,
    )
    from .editing.streaming_pruning import materialize_magnitude_pruned_artifact
    from .editing.streaming_transform import materialize_transformation_artifact
    from .editing.transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.implementations import (
        generated_transformation_edit_dir_name,
        parse_edit_specs_json,
        real_training_edit_migration_message,
        resolve_batch_entry,
    )
    from editing.streaming_pruning import materialize_magnitude_pruned_artifact
    from editing.streaming_transform import materialize_transformation_artifact
    from editing.transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        validate_transformation_scope,
    )

_STREAMING_TRANSFORMATION_TYPES = frozenset(
    {"quant_rtn", SYNTHETIC_LOWRANK_DELTA, SYNTHETIC_DENSE_UPDATE}
)
_UNSUPPORTED_GENERATED_EDIT_TYPES = frozenset({"fp8_quant", "lowrank_svd"})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create many evidence-pack edits from one baseline checkpoint."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument(
        "--edit-specs-json",
        required=True,
        help="JSON array of objects with keys: spec, version.",
    )
    return parser.parse_args(argv)


def _parse_edit_specs_json(raw_payload: str) -> list[object]:
    return parse_edit_specs_json(raw_payload)


def _preflight_reject_real_training_specs(edit_specs: list[object]) -> None:
    """Reject real training labels before no-grad setup or model loading."""

    for spec_entry in edit_specs:
        if not isinstance(spec_entry, dict):
            continue
        raw_spec = str(spec_entry.get("spec") or spec_entry.get("type") or "")
        requested_type = raw_spec.split(":", maxsplit=1)[0]
        message = real_training_edit_migration_message(requested_type)
        if message is not None:
            raise ValueError(message)


def _raw_edit_type(spec_entry: object) -> str:
    if not isinstance(spec_entry, dict):
        return ""
    raw_spec = str(spec_entry.get("spec") or spec_entry.get("type") or "")
    return raw_spec.split(":", maxsplit=1)[0]


def _preflight_reject_unverifiable_generated_specs(edit_specs: list[object]) -> None:
    """Reject unsupported generated families before any model/runtime setup."""

    for spec_entry in edit_specs:
        requested_type = _raw_edit_type(spec_entry)
        if requested_type in _UNSUPPORTED_GENERATED_EDIT_TYPES:
            raise ValueError(
                f"{requested_type} requires a dedicated storage and replay contract"
            )


def _is_magnitude_prune(parsed_spec: dict[str, object]) -> bool:
    return str(parsed_spec.get("type") or "") == "magnitude_prune"


def _is_streaming_transformation(parsed_spec: dict[str, object]) -> bool:
    return str(parsed_spec.get("type") or "") in _STREAMING_TRANSFORMATION_TYPES


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "throughput":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)


def _reject_removed_batch_strategy_selector() -> None:
    """Fail closed instead of silently preserving the removed mutable mode."""

    if "PACK_BATCH_EDIT_STRATEGY" in os.environ:
        raise ValueError(
            "PACK_BATCH_EDIT_STRATEGY is no longer supported; verifier-grade "
            "generated edits are always materialized directly from safetensors"
        )


def _get_edit_dir_name(parsed_spec: dict[str, object], version: str) -> str:
    edit_type = str(parsed_spec["type"])
    migration_message = real_training_edit_migration_message(edit_type)
    if migration_message is not None:
        raise ValueError(migration_message)

    def canonical_generated_dir_name(computed: str) -> str:
        supplied = parsed_spec.get("edit_dir_name")
        if supplied not in (None, "", computed):
            raise ValueError(
                "verifier-grade generated transformations require their canonical "
                "directory identity"
            )
        return computed

    if edit_type == "quant_rtn":
        return canonical_generated_dir_name(
            generated_transformation_edit_dir_name(
                edit_type=edit_type,
                parameters={
                    "bits": parsed_spec.get("bits"),
                    "group_size": parsed_spec.get("group_size"),
                },
                scope=parsed_spec.get("scope"),
                version=version,
            )
        )
    if edit_type == "magnitude_prune":
        pct = int(float(parsed_spec["ratio"]) * 100)
        return f"prune_{pct}pct_{version}"
    if edit_type in _UNSUPPORTED_GENERATED_EDIT_TYPES:
        raise ValueError(
            f"{edit_type} requires a dedicated storage and replay contract"
        )
    if edit_type == "synthetic_lowrank_delta":
        return canonical_generated_dir_name(
            generated_transformation_edit_dir_name(
                edit_type=edit_type,
                parameters={
                    "rank": parsed_spec.get("rank"),
                    "scale": parsed_spec.get("scale"),
                },
                scope=parsed_spec.get("scope"),
                version=version,
            )
        )
    if edit_type == "synthetic_dense_update":
        return canonical_generated_dir_name(
            generated_transformation_edit_dir_name(
                edit_type=edit_type,
                parameters={
                    "step_size": parsed_spec.get("step_size"),
                    "iterations": parsed_spec.get("iterations"),
                },
                scope=parsed_spec.get("scope"),
                version=version,
            )
        )
    if parsed_spec.get("edit_dir_name"):
        return str(parsed_spec["edit_dir_name"])
    return f"{edit_type}_{version}"


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_streaming_magnitude_prune_artifact(
    *,
    baseline_path: Path,
    parsed_spec: dict[str, object],
    edit_path: Path,
) -> None:
    """Materialize pruning without loading a mutable Transformers model."""
    if not _is_magnitude_prune(parsed_spec):
        raise ValueError("streaming pruning helper received a non-pruning edit")
    ratio = float(parsed_spec["ratio"])
    scope = str(parsed_spec["scope"])
    result = materialize_magnitude_pruned_artifact(
        baseline_path=baseline_path,
        output_path=edit_path,
        sparsity=ratio,
        scope=scope,
    )
    print(
        "    Streaming prune: "
        f"{result['selected_tensors']} tensors, "
        f"{result['effective_changed_params']:,} changed"
    )


def _canonical_transformation_inputs(
    parsed_spec: Mapping[str, object],
) -> tuple[str, dict[str, object], str]:
    """Validate a resolved batch payload against the replay contract."""

    edit_type = str(parsed_spec.get("type") or "")
    scope = validate_transformation_scope(parsed_spec.get("scope"))
    if edit_type == "quant_rtn":
        raw_parameters: dict[str, object] = {
            "bits": parsed_spec.get("bits"),
            "group_size": parsed_spec.get("group_size"),
        }
    elif edit_type == SYNTHETIC_LOWRANK_DELTA:
        raw_parameters = {
            "rank": parsed_spec.get("rank"),
            "scale": parsed_spec.get("scale"),
        }
    elif edit_type == SYNTHETIC_DENSE_UPDATE:
        raw_parameters = {
            "step_size": parsed_spec.get("step_size"),
            "iterations": parsed_spec.get("iterations"),
        }
    else:
        raise TransformationContractError(
            f"{edit_type!r} is not a verifier-grade generated transformation"
        )
    specification = canonical_transformation_spec(edit_type, raw_parameters)
    parameters = specification.get("parameters")
    if not isinstance(parameters, Mapping):  # contract invariant
        raise TransformationContractError("canonical transformation parameters missing")
    return edit_type, dict(parameters), scope


def _create_streaming_transformation_artifact(
    *,
    baseline_path: Path,
    parsed_spec: dict[str, object],
    edit_path: Path,
) -> None:
    """Materialize a supported transform without loading a Transformers model."""

    edit_type, parameters, scope = _canonical_transformation_inputs(parsed_spec)
    result = materialize_transformation_artifact(
        baseline_path=baseline_path,
        output_path=edit_path,
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
    )
    print(
        "    Streaming transformation: "
        f"{edit_type}, {result['selected_tensors']} tensors, "
        f"{result['actual_changes']['value_changed_params']:,} changed"
    )


def _materialize_pending_edit_artifact(
    *,
    baseline_path: Path,
    parsed_spec: dict[str, object],
    edit_path: Path,
) -> None:
    if _is_magnitude_prune(parsed_spec):
        _create_streaming_magnitude_prune_artifact(
            baseline_path=baseline_path,
            parsed_spec=parsed_spec,
            edit_path=edit_path,
        )
        return
    if _is_streaming_transformation(parsed_spec):
        _create_streaming_transformation_artifact(
            baseline_path=baseline_path,
            parsed_spec=parsed_spec,
            edit_path=edit_path,
        )
        return
    raise ValueError(
        f"Unsupported batch edit type: {parsed_spec.get('type')!r}; "
        "use a dedicated verifier-grade generation path"
    )


def _process_spec_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
    baseline_path: Path,
) -> tuple[int, int]:
    try:
        pending, created, failed = _resolve_pending_spec_entry(
            spec_entry=spec_entry,
            model_output_dir=model_output_dir,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return 0, 1
    if pending is None:
        return created, failed

    parsed, edit_path = pending
    try:
        _materialize_pending_edit_artifact(
            baseline_path=baseline_path,
            parsed_spec=parsed,
            edit_path=edit_path,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return 0, 1

    print(f"    Saved: {edit_path}")
    return 1, 0


def _resolve_pending_spec_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
) -> tuple[tuple[dict[str, object], Path] | None, int, int]:
    if not isinstance(spec_entry, dict):
        return None, 0, 0

    spec_str = str(spec_entry.get("spec", ""))
    version = str(spec_entry.get("version", "clean"))
    parsed_resolved = resolve_batch_entry(
        spec_entry=spec_entry,
        model_output_dir=model_output_dir,
    )
    if parsed_resolved is None:
        return None, 0, 0
    parsed = parsed_resolved.to_batch_payload()

    # The lifecycle rewrites only receipt-verified clean selections into
    # explicit literals before entering this generic streaming helper.  The exact
    # canonical parameters and scope must bind the final path: a selection
    # bundle cannot point two different candidates at one shared directory.
    if "v2_selection_edit_dir_name" in spec_entry:
        raise ValueError(
            "retired v2 clean-selection artifact directory field is not accepted"
        )
    selected_dir = spec_entry.get("selection_edit_dir_name")
    if selected_dir is not None:
        expected_dir = _get_edit_dir_name(parsed, version)
        if (
            version != "clean"
            or not isinstance(selected_dir, str)
            or selected_dir != expected_dir
        ):
            raise ValueError("invalid clean-selection artifact directory")
        parsed["edit_dir_name"] = selected_dir

    if parsed_resolved.skip:
        print(f"  Skip (tuned edit preset skipped): {spec_str}")
        return None, 0, 0
    if not parsed_resolved.selected:
        reason = getattr(parsed_resolved, "reason", "")
        if reason:
            raise ValueError(reason)
        raise ValueError(
            f"Tuned edit preset missing for {spec_str}: {parsed_resolved.status}"
        )

    edit_dir_name = _get_edit_dir_name(parsed, version)
    edit_path = model_output_dir / "models" / edit_dir_name
    # A final artifact must never be trusted or counted solely because it has
    # metadata/receipt-shaped files.  The materializers can resume only their
    # own digest-bound staging directories; an occupied final path is either a
    # prior publication or untrusted residue and requires explicit operator
    # action rather than a silent reuse.
    if edit_path.exists() or edit_path.is_symlink():
        raise ValueError(
            "refusing final artifact reuse at "
            f"{edit_path}; resumable generation uses the materializer staging "
            "directory only"
        )

    print(f"  Creating: {edit_dir_name}...")
    return (parsed, edit_path), 0, 0


def _process_edit_specs(
    *,
    edit_specs: list[object],
    model_output_dir: Path,
    baseline_path: Path,
) -> tuple[int, int]:
    created_count = 0
    failed_count = 0
    for spec_entry in edit_specs:
        created, failed = _process_spec_entry(
            spec_entry=spec_entry,
            model_output_dir=model_output_dir,
            baseline_path=baseline_path,
        )
        created_count += created
        failed_count += failed
    return created_count, failed_count


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_path = Path(args.baseline)
    model_output_dir = Path(args.model_output_dir)

    try:
        edit_specs = _parse_edit_specs_json(args.edit_specs_json)
        _preflight_reject_real_training_specs(edit_specs)
        _preflight_reject_unverifiable_generated_specs(edit_specs)
        _reject_removed_batch_strategy_selector()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _configure_determinism()
    print(f"Creating {len(edit_specs)} edits...")
    print("Materializing every verifier-grade edit directly from safetensors...")
    try:
        created_count, failed_count = _process_edit_specs(
            edit_specs=edit_specs,
            baseline_path=baseline_path,
            model_output_dir=model_output_dir,
        )
    finally:
        _clear_memory()

    print(f"Batch complete: {created_count} created, {failed_count} failed")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
