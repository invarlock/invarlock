#!/usr/bin/env bash
# Shared immutable-profile helpers for the training-profile integration examples.

integration_default_training_profile() {
  local edit_type="$1"
  local device="$2"
  case "${edit_type}:${device}" in
    lora_merge:cuda*) printf 'tiny_gpt2_lora_cuda_v1\n' ;;
    lora_merge:*) printf 'tiny_gpt2_lora_v1\n' ;;
    fine_tune:cuda*) printf 'tiny_gpt2_full_ft_cuda_v1\n' ;;
    fine_tune:*) printf 'tiny_gpt2_full_ft_v1\n' ;;
    *)
      echo "Unsupported training edit/device combination: ${edit_type}/${device}" >&2
      return 2
      ;;
  esac
}

integration_load_training_profile() {
  local python_bin="$1"
  local profiles_path="$2"
  local profile_id="$3"
  local expected_edit_type="$4"

  local fields
  if ! fields="$("$python_bin" - "$profiles_path" "$profile_id" "$expected_edit_type" <<'PY'
from pathlib import Path
import sys

profiles_path = Path(sys.argv[1])
profile_id = sys.argv[2]
expected_edit_type = sys.argv[3]
repo_root = profiles_path.resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "evidence_packs" / "python"))

from editing.training_contract import load_training_profile

profile = load_training_profile(
    profile_id,
    expected_edit_type=expected_edit_type,
    profiles_path=profiles_path,
    repo_root=repo_root,
)
if profile.edit_type != expected_edit_type:
    raise SystemExit(
        f"Training profile {profile_id} has edit_type={profile.edit_type!r}; "
        f"expected {expected_edit_type!r}"
    )
values = (
    profile.model_id,
    profile.model_revision,
    profile.device,
    profile.profile_sha256,
)
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit(f"Training profile {profile_id} is missing required identity fields")
if any("\t" in value or "\n" in value for value in values):
    raise SystemExit(f"Training profile {profile_id} contains an unsafe identity field")
print("\t".join(values))
PY
)"; then
    return 2
  fi
  IFS=$'\t' read -r TRAINING_MODEL_ID TRAINING_MODEL_REVISION TRAINING_DEVICE TRAINING_PROFILE_SHA256 <<<"$fields"
  export TRAINING_MODEL_ID TRAINING_MODEL_REVISION TRAINING_DEVICE TRAINING_PROFILE_SHA256
}

integration_preflight_training_device() {
  local python_bin="$1"
  local training_device="$2"
  local example_name="$3"
  if [[ "$training_device" != cuda* ]]; then
    return 0
  fi
  if ! "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    cat >&2 <<MSG
$example_name selected a CUDA training profile, but the host Python environment
does not expose torch.cuda. Use the CPU training profile or a CUDA-enabled host
Python environment. Container evaluation starts after subject materialization.
MSG
    return 2
  fi
}

integration_prepare_training_output() {
  local python_bin="$1"
  local repo_root="$2"
  local subject_dir="$3"
  local force="$4"
  "$python_bin" - "$repo_root" "$subject_dir" "$force" <<'PY'
import os
from pathlib import Path
import shutil
import tempfile
import sys

MARKER_SCHEMA = "invarlock.integration_training_output.v2"

repo_root = Path(sys.argv[1]).expanduser().resolve()
sys.path.insert(0, str(repo_root / "src"))

from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot

requested_path = Path(sys.argv[2]).expanduser()
if requested_path.is_symlink():
    raise SystemExit(f"Refusing symlink subject output path: {requested_path}")
path = requested_path.resolve(strict=False)
force = sys.argv[3] == "1"
protected = {Path("/"), Path.home().resolve(), repo_root, *repo_root.parents}
if path in protected:
    raise SystemExit(f"Refusing protected subject output path: {path}")

marker = path.parent / f".{path.name}.invarlock-training-output.json"
if not path.exists() and not path.is_symlink():
    if marker.exists() or marker.is_symlink():
        raise SystemExit(f"Refusing unexpected ownership-marker path: {marker}")
    raise SystemExit(0)
if not force:
    raise SystemExit(
        f"Subject output already exists: {path} (pass --force to replace it)."
    )
if path.is_symlink() or not path.is_dir():
    raise SystemExit(f"Refusing to replace non-directory subject output: {path}")
if marker.is_symlink() or not marker.is_file():
    raise SystemExit(
        f"Refusing to replace unowned subject output (missing marker): {path}"
    )
try:
    _, ownership = read_json_object_snapshot(marker, label="training ownership marker")
except StrictJsonError as exc:
    raise SystemExit(f"Refusing invalid ownership marker {marker}: {exc}") from exc
stat = path.stat(follow_symlinks=False)
expected = {
    "schema": MARKER_SCHEMA,
    "subject_path": str(path),
    "st_dev": stat.st_dev,
    "st_ino": stat.st_ino,
    "st_mtime_ns": stat.st_mtime_ns,
    "st_ctime_ns": stat.st_ctime_ns,
}
if ownership != expected:
    raise SystemExit(f"Refusing ownership-marker mismatch for subject output: {path}")
descriptor, quarantine_name = tempfile.mkstemp(
    prefix=f".{path.name}.delete-", dir=path.parent
)
os.close(descriptor)
quarantine = Path(quarantine_name)
quarantine.unlink()
path.rename(quarantine)
quarantine_stat = quarantine.stat(follow_symlinks=False)
if (quarantine_stat.st_dev, quarantine_stat.st_ino) != (stat.st_dev, stat.st_ino):
    if not path.exists() and not path.is_symlink():
        quarantine.rename(path)
    raise SystemExit(
        f"Refusing subject output changed during protected replacement: {path}"
    )
try:
    shutil.rmtree(quarantine)
except Exception:
    if quarantine.exists() and not path.exists() and not path.is_symlink():
        quarantine.rename(path)
    raise
marker.unlink()
PY
}

integration_mark_training_output() {
  local python_bin="$1"
  local repo_root="$2"
  local subject_dir="$3"
  "$python_bin" - "$repo_root" "$subject_dir" <<'PY'
import json
import os
from pathlib import Path
import tempfile
import sys

MARKER_SCHEMA = "invarlock.integration_training_output.v2"

repo_root = Path(sys.argv[1]).expanduser().resolve()
requested_path = Path(sys.argv[2]).expanduser()
if requested_path.is_symlink():
    raise SystemExit(f"Refusing symlink subject output path: {requested_path}")
path = requested_path.resolve(strict=False)
protected = {Path("/"), Path.home().resolve(), repo_root, *repo_root.parents}
if path in protected:
    raise SystemExit(f"Refusing protected subject output path: {path}")
if path.is_symlink() or not path.is_dir():
    raise SystemExit(f"Cannot mark non-directory training output: {path}")
stat = path.stat(follow_symlinks=False)
payload = {
    "schema": MARKER_SCHEMA,
    "subject_path": str(path),
    "st_dev": stat.st_dev,
    "st_ino": stat.st_ino,
    "st_mtime_ns": stat.st_mtime_ns,
    "st_ctime_ns": stat.st_ctime_ns,
}
marker = path.parent / f".{path.name}.invarlock-training-output.json"
if marker.exists() or marker.is_symlink():
    raise SystemExit(f"Refusing existing ownership-marker path: {marker}")
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.ownership-", dir=path.parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, marker)
    except FileExistsError as exc:
        raise SystemExit(f"Refusing existing ownership-marker path: {marker}") from exc
finally:
    temporary.unlink(missing_ok=True)
PY
}

integration_run_training_verifier_to_stderr() {
  local verify_output
  local verify_status
  verify_output="$(mktemp "${TMPDIR:-/tmp}/invarlock-training-verify.XXXXXX")" || return 1
  if integration_run_source_archive_clean "$@" > "$verify_output"; then
    verify_status=0
  else
    verify_status=$?
  fi
  if [[ -s "$verify_output" ]]; then
    cat "$verify_output" >&2
  fi
  rm -f "$verify_output"
  return "$verify_status"
}

integration_run_training_profile() {
  local python_bin="$1"
  local repo_root="$2"
  local profiles_path="$3"
  local profile_id="$4"
  local subject_dir="$5"
  local allow_network="$6"
  local command=(
    "$python_bin"
    "$repo_root/scripts/evidence_packs/python/create_edit_model.py"
    train-profile
    "$profile_id"
    "$subject_dir"
    --profiles-path "$profiles_path"
    --repo-root "$repo_root"
  )
  if [[ "$allow_network" == "1" ]]; then
    command+=(--allow-network)
  fi
  integration_run_source_archive_clean "${command[@]}" || return $?

  local verify_command=(
    "$python_bin"
    "$repo_root/scripts/evidence_packs/python/create_edit_model.py"
    verify-training-profile
    "$profile_id"
    "$subject_dir"
    --profiles-path "$profiles_path"
    --repo-root "$repo_root"
  )
  if [[ "$allow_network" == "1" ]]; then
    verify_command+=(--allow-network)
  fi
  integration_run_training_verifier_to_stderr "${verify_command[@]}" || return $?
  integration_mark_training_output "$python_bin" "$repo_root" "$subject_dir" || return $?
}

integration_verify_training_binding() {
  local python_bin="$1"
  local repo_root="$2"
  local profiles_path="$3"
  local profile_id="$4"
  local subject_dir="$5"
  local copied_receipt="$6"
  local allow_network="$7"
  local verify_command=(
    "$python_bin"
    "$repo_root/scripts/evidence_packs/python/create_edit_model.py"
    verify-training-profile
    "$profile_id"
    "$subject_dir"
    --profiles-path "$profiles_path"
    --repo-root "$repo_root"
  )
  if [[ "$allow_network" == "1" ]]; then
    verify_command+=(--allow-network)
  fi
  integration_run_training_verifier_to_stderr "${verify_command[@]}" || return $?

  "$python_bin" - "$repo_root" "$subject_dir" "$copied_receipt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
requested_subject = Path(sys.argv[2]).expanduser()
if requested_subject.is_symlink():
    raise SystemExit(f"Refusing symlink subject output path: {requested_subject}")
subject_dir = requested_subject.resolve()
copied_receipt_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo_root / "scripts" / "evidence_packs" / "python"))
sys.path.insert(0, str(repo_root / "src"))

from editing.training_receipt import canonical_receipt_digest
from editing.training_runtime import directory_sha256
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_json_object_snapshot,
    read_regular_file_bytes,
)

marker = subject_dir.parent / f".{subject_dir.name}.invarlock-training-output.json"
try:
    _, ownership = read_json_object_snapshot(marker, label="training ownership marker")
except StrictJsonError as exc:
    raise SystemExit(f"Unable to validate training-output ownership marker: {exc}") from exc
stat = subject_dir.stat(follow_symlinks=False)
expected_ownership = {
    "schema": "invarlock.integration_training_output.v2",
    "subject_path": str(subject_dir),
    "st_dev": stat.st_dev,
    "st_ino": stat.st_ino,
    "st_mtime_ns": stat.st_mtime_ns,
    "st_ctime_ns": stat.st_ctime_ns,
}
if ownership != expected_ownership:
    raise SystemExit("Training subject identity changed after materialization")
subject_receipt_path = subject_dir / "training_receipt.json"
try:
    subject_bytes = read_regular_file_bytes(
        subject_receipt_path, label="subject training receipt"
    )
    copied_bytes = read_regular_file_bytes(
        copied_receipt_path, label="copied training receipt"
    )
except StrictJsonError as exc:
    raise SystemExit(f"Unable to read training receipt for post-evaluation binding: {exc}") from exc
if copied_bytes != subject_bytes:
    raise SystemExit(
        "Copied training receipt is not byte-identical to the subject receipt"
    )
try:
    receipt = parse_json_bytes(subject_bytes, label="subject training receipt")
except StrictJsonError as exc:
    raise SystemExit(f"Subject training receipt is not valid JSON: {exc}") from exc
if not isinstance(receipt, dict):
    raise SystemExit("Subject training receipt is not a JSON object")
observed_receipt_digest = canonical_receipt_digest(receipt)
if observed_receipt_digest != receipt.get("receipt_sha256"):
    raise SystemExit("Subject training receipt canonical digest mismatch")
expected_tree = receipt.get("hashes", {}).get("subject_tree_sha256")
observed_tree = directory_sha256(
    subject_dir, exclude=frozenset({"training_receipt.json"})
)
if observed_tree != expected_tree:
    raise SystemExit(
        "Subject artifact tree no longer matches the copied training receipt"
    )
print(
    json.dumps(
        {
            "schema": "invarlock.integration_training_binding.v1",
            "receipt_sha256": observed_receipt_digest,
            "training_receipt_file_sha256": hashlib.sha256(copied_bytes).hexdigest(),
            "subject_tree_sha256": observed_tree,
            "verified": True,
        },
        sort_keys=True,
    )
)
PY
}

integration_mark_training_binding_failed() {
  local report_out="$1"
  local transient_path="${2:-}"
  local binding_path="$report_out/training_binding.json"
  local summary_path="$report_out/run_summary.txt"
  local summary_tmp

  rm -f "$binding_path" 2>/dev/null || true
  if [[ -n "$transient_path" ]]; then
    rm -f "$transient_path" 2>/dev/null || true
  fi
  summary_tmp="$(mktemp "$report_out/.run-summary.XXXXXX")" || return 1
  if ! {
    printf 'status: failed\n'
    if [[ -f "$summary_path" ]]; then
      awk '!/^status:|^training_binding_status:|^training_binding:/' "$summary_path"
    fi
    printf 'training_binding_status: failed\n'
  } > "$summary_tmp"; then
    rm -f "$summary_tmp"
    return 1
  fi
  if ! mv "$summary_tmp" "$summary_path"; then
    rm -f "$summary_tmp"
    return 1
  fi
}

integration_finalize_training_binding() {
  local python_bin="$1"
  local repo_root="$2"
  local profiles_path="$3"
  local profile_id="$4"
  local subject_dir="$5"
  local copied_receipt="$6"
  local allow_network="$7"
  local report_out="$8"
  local binding_path="$report_out/training_binding.json"
  local summary_path="$report_out/run_summary.txt"
  local binding_tmp
  local summary_tmp

  if ! rm -f "$binding_path"; then
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! binding_tmp="$(mktemp "$report_out/.training-binding.XXXXXX")"; then
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! integration_verify_training_binding \
    "$python_bin" "$repo_root" "$profiles_path" "$profile_id" \
    "$subject_dir" "$copied_receipt" "$allow_network" > "$binding_tmp"; then
    integration_mark_training_binding_failed "$report_out" "$binding_tmp" || true
    return 1
  fi
  if ! "$python_bin" - "$repo_root" "$binding_tmp" \
    "$report_out/evaluation.report.json" "$report_out/verify.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
binding_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
verify_path = Path(sys.argv[4])
sys.path.insert(0, str(repo_root / "src"))

from invarlock.evidence_pack_json import (
    read_json_object_snapshot,
    read_regular_file_bytes,
)

_, binding = read_json_object_snapshot(binding_path, label="training binding")
report_bytes = read_regular_file_bytes(report_path, label="evaluation report")
verify_bytes = read_regular_file_bytes(verify_path, label="verification artifact")
binding["evaluation_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
binding["verify_artifact_sha256"] = hashlib.sha256(verify_bytes).hexdigest()
binding_path.write_text(
    json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
  then
    integration_mark_training_binding_failed "$report_out" "$binding_tmp" || true
    return 1
  fi
  if ! mv "$binding_tmp" "$binding_path"; then
    integration_mark_training_binding_failed "$report_out" "$binding_tmp" || true
    return 1
  fi

  if ! summary_tmp="$(mktemp "$report_out/.run-summary.XXXXXX")"; then
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! {
    if [[ -f "$summary_path" ]]; then
      awk '!/^training_binding_status:|^training_binding:/' "$summary_path"
    else
      printf 'status: success\n'
    fi
    printf 'training_binding_status: verified\n'
    printf 'training_binding: %s\n' "$binding_path"
  } > "$summary_tmp"; then
    integration_mark_training_binding_failed "$report_out" "$summary_tmp" || true
    return 1
  fi
  if ! mv "$summary_tmp" "$summary_path"; then
    integration_mark_training_binding_failed "$report_out" "$summary_tmp" || true
    return 1
  fi
}

integration_require_finalized_training_binding() {
  # Call order alone is not a security boundary: a proof staging helper can be
  # invoked independently. Require the receipt, report, and verifier result to
  # be sealed by integration_finalize_training_binding before proof generation.
  local python_bin="$1"
  local repo_root="$2"
  local report_out="$3"

  "$python_bin" - "$repo_root" "$report_out" <<'PY'
import hashlib
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
report_out = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))

from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot

try:
    receipt_bytes, receipt = read_json_object_snapshot(
        report_out / "training_receipt.json", label="training receipt"
    )
    report_bytes, _ = read_json_object_snapshot(
        report_out / "evaluation.report.json", label="evaluation report"
    )
    verify_bytes, _ = read_json_object_snapshot(
        report_out / "verify.json", label="verification result"
    )
    _, binding = read_json_object_snapshot(
        report_out / "training_binding.json", label="post-evaluation training binding"
    )
except (OSError, StrictJsonError) as exc:
    raise SystemExit(f"finalized post-evaluation training binding is unavailable: {exc}") from exc

hashes = receipt.get("hashes")
if not isinstance(hashes, dict):
    raise SystemExit("finalized post-evaluation training binding has no receipt tree")
expected = {
    "schema": "invarlock.integration_training_binding.v1",
    "receipt_sha256": receipt.get("receipt_sha256"),
    "training_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    "subject_tree_sha256": hashes.get("subject_tree_sha256"),
    "evaluation_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    "verify_artifact_sha256": hashlib.sha256(verify_bytes).hexdigest(),
    "verified": True,
}
if set(binding) != set(expected):
    raise SystemExit("finalized post-evaluation training binding has an invalid shape")
for field, value in expected.items():
    if binding.get(field) != value:
        raise SystemExit(
            "finalized post-evaluation training binding does not bind " + field
        )
PY
}

integration_stage_training_evidence() {
  # Publish proof material only after the post-evaluation subject binding has
  # succeeded. Scope is an explicit reviewed policy input; never infer it from
  # adapter module names or a model architecture.
  local python_bin="$1"
  local repo_root="$2"
  local profiles_path="$3"
  local profile_id="$4"
  local subject_dir="$5"
  local report_out="$6"
  local allow_network="$7"
  local scope="$8"
  local receipt_path="$report_out/training_receipt.json"
  local proof_path="$report_out/training_evidence_proof.json"
  local snapshot_path="$report_out/training_profile_snapshot.json"
  local report_path="$report_out/evaluation.report.json"

  if [[ ! -f "$receipt_path" || ! -f "$report_path" ]]; then
    echo "Missing receipt or evaluation report for training evidence staging." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! integration_require_finalized_training_binding \
    "$python_bin" "$repo_root" "$report_out"; then
    echo "Training evidence staging requires a finalized post-evaluation binding." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  local -a proof_command=(
    "$python_bin"
    "$repo_root/scripts/evidence_packs/python/editing/training_evidence_proof.py"
    --profile-id "$profile_id"
    --subject "$subject_dir"
    --out "$proof_path"
    --profiles-path "$profiles_path"
    --repo-root "$repo_root"
  )
  if [[ "$allow_network" == "1" ]]; then
    proof_command+=(--allow-network)
  fi
  if ! integration_run_source_archive_clean "${proof_command[@]}" >/dev/null; then
    echo "Failed to produce artifact-replay training evidence proof." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! integration_run_source_archive_clean \
    "$python_bin" \
    "$repo_root/scripts/evidence_packs/python/editing/training_profile_snapshot.py" \
    --profile-id "$profile_id" \
    --scope "$scope" \
    --out "$snapshot_path" \
    --profiles-path "$profiles_path" \
    --repo-root "$repo_root" >/dev/null; then
    echo "Failed to stage immutable training profile snapshot." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  if ! "$python_bin" - "$repo_root" "$proof_path" "$receipt_path" "$report_path" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
proof_path = Path(sys.argv[2])
receipt_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])
sys.path.insert(0, str(repo_root / "src"))

from invarlock.evidence_pack_json import read_json_object_snapshot
from invarlock.training_evidence import require_valid_training_evidence_proof

_, proof = read_json_object_snapshot(proof_path, label="training evidence proof")
_, receipt = read_json_object_snapshot(receipt_path, label="training receipt")
_, report = read_json_object_snapshot(report_path, label="evaluation report")
meta = report.get("meta")
baseline_ref = report.get("baseline_ref")
artifact_identity = meta.get("model_identity") if isinstance(meta, dict) else None
baseline_identity = (
    baseline_ref.get("model_identity") if isinstance(baseline_ref, dict) else None
)
if not isinstance(artifact_identity, dict) or not isinstance(baseline_identity, dict):
    raise SystemExit("evaluation report does not expose both model identities")
edit_type = receipt.get("edit_type")
if not isinstance(edit_type, str):
    raise SystemExit("training receipt edit_type is unavailable")
require_valid_training_evidence_proof(
    proof,
    receipt,
    expected_edit_type=edit_type,
    expected_baseline_identity=baseline_identity,
    expected_artifact_identity=artifact_identity,
)
PY
  then
    echo "Training evidence proof does not bind the staged evaluation report." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
  # Re-read the binding after proof construction so a report or verifier result
  # cannot change between the pre-stage gate and publication of sidecars.
  if ! integration_require_finalized_training_binding \
    "$python_bin" "$repo_root" "$report_out"; then
    echo "Training evidence staging observed post-evaluation binding drift." >&2
    integration_mark_training_binding_failed "$report_out" "" || true
    return 1
  fi
}
