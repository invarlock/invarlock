#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'Native runtime example: %s\n' "$*" >&2
  exit 2
}

required_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "set $name"
}

required_file() {
  local label="$1"
  local path="$2"
  [[ -f "$path" && ! -L "$path" ]] || die "$label must be a regular non-symlink file: $path"
}

required_directory() {
  local label="$1"
  local path="$2"
  [[ -d "$path" && ! -L "$path" ]] || die "$label must be a non-symlink directory: $path"
}

required_output_file() {
  local label="$1"
  local path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" && -r "$path" ]] ||
    die "$label was not published as a nonempty readable regular file: $path"
}

required_side_bundle() {
  local role="$1"
  local path="$2"
  required_directory "$role side" "$path"
  [[ -r "$path" && -x "$path" ]] || die "$role side is not readable: $path"
  local name
  for name in \
    evaluation.report.json \
    model-artifact.identity.json \
    runtime-behavior.config.json \
    runtime-provider.receipt.json \
    runtime-scoring.observation.json \
    runtime.manifest.json; do
    required_output_file "$role side file" "$path/$name"
  done
}

safe_mount_source() {
  local label="$1"
  local path="$2"
  [[ "$path" != *','* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] ||
    die "$label contains a character that cannot be represented in a Docker mount"
}

absolute_existing_path() {
  local path="$1"
  local directory
  local basename
  directory="$(dirname -- "$path")"
  basename="$(basename -- "$path")"
  printf '%s/%s\n' "$(cd -- "$directory" && pwd -P)" "$basename"
}

validate_image_digest() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^sha256:[a-f0-9]{64}$ ]] || die "$label must be an immutable sha256 image digest"
  [[ "$value" != "sha256:$(printf '0%.0s' {1..64})" ]] || die "$label still contains the template digest"
}

validate_provider() {
  local label="$1"
  local value="$2"
  [[ "$value" == "llama_cpp" || "$value" == "tensorrt_llm" ]] ||
    die "$label must be llama_cpp or tensorrt_llm"
}

validate_gpu_selector() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^device=([0-9]+|GPU-[A-Fa-f0-9-]{20,80})$ ]] ||
    die "$label must be device=<nonnegative-index> or device=<GPU-UUID>"
}

model_id_from_settings() {
  local provider="$1"
  local settings="$2"
  "$PYTHON_BIN" - "$provider" "$settings" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

provider, raw_path = sys.argv[1:]
value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("settings must be a JSON object")
key = "artifact_sha256" if provider == "llama_cpp" else "engine_bundle_tree_sha256"
digest = value.get(key)
if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
    raise SystemExit(f"{key} must be a lowercase sha256 digest")
if digest == "0" * 64:
    raise SystemExit(f"{key} still contains the template digest")
prefix = "gguf-sha256-" if provider == "llama_cpp" else "tensorrt-llm-sha256-"
suffix = ".gguf" if provider == "llama_cpp" else ""
print(prefix + digest + suffix)
PY
}

container_cli() {
  local phase="$1"
  local role="$2"
  local provider="$3"
  local image_digest="$4"
  local artifact="$5"
  local settings="$6"
  local tokenizer_contract="$7"
  local gpu_selector="$8"
  local model_id="$9"
  local role_prefix

  if [[ "$role" == "baseline" ]]; then
    role_prefix="BASELINE"
  elif [[ "$role" == "subject" ]]; then
    role_prefix="SUBJECT"
  else
    die "unsupported provider role: $role"
  fi

  local artifact_target
  local backend_executable
  local backend_source=""
  local tmpfs_size="1g"
  local -a provider_mounts=()
  local -a provider_options=()
  local -a launcher=()
  local -a gpu_options=()
  local -a cache_options=()

  if [[ "$provider" == "llama_cpp" ]]; then
    artifact_target="/models/model.gguf"
    backend_executable="/opt/llama.cpp/llama-completion"
    backend_source="/opt/llama.cpp/source/llama.cpp-b10015.tar.gz"
    provider_options=(--backend-source "$backend_source")
    launcher=(--entrypoint /usr/local/bin/invarlock "$image_digest")
  else
    artifact_target="/engines/model"
    backend_executable="/opt/invarlock/bin/tensorrt-llm-runner"
    [[ -n "$tokenizer_contract" ]] || die "$role TensorRT-LLM side requires INVARLOCK_${role_prefix}_TOKENIZER_CONTRACT"
    required_file "$role tokenizer contract" "$tokenizer_contract"
    tokenizer_contract="$(absolute_existing_path "$tokenizer_contract")"
    safe_mount_source "$role tokenizer contract" "$tokenizer_contract"
    [[ -n "$gpu_selector" ]] || die "$role TensorRT-LLM side requires INVARLOCK_${role_prefix}_GPU"
    validate_gpu_selector "INVARLOCK_${role_prefix}_GPU" "$gpu_selector"
    [[ "$TENSORRT_TMPFS_SIZE" =~ ^[1-9][0-9]*[mMgG]$ ]] || die "INVARLOCK_TENSORRT_TMPFS_SIZE must be a positive Docker size such as 16g"
    tmpfs_size="$TENSORRT_TMPFS_SIZE"
    provider_mounts=(--mount "type=bind,src=$tokenizer_contract,dst=/inputs/tokenizer-contract.json,readonly")
    provider_options=(--tokenizer-contract /inputs/tokenizer-contract.json)
    gpu_options=(--gpus "$gpu_selector")
    cache_options=(
      --env HOME=/tmp/invarlock-home
      --env XDG_CACHE_HOME=/tmp/invarlock-cache
      --env HF_HOME=/tmp/invarlock-hf
      --env FLASHINFER_WORKSPACE_DIR=/tmp/invarlock-flashinfer
    )
    launcher=(
      --entrypoint /bin/bash "$image_digest"
      -c 'exec "$@"' -- /opt/invarlock/cli-venv/bin/invarlock
    )
  fi

  local -a command=(
    "$CONTAINER_ENGINE" run --rm
    "${gpu_options[@]}"
    --user "$HOST_UID:$HOST_GID"
    --network none
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges
    --pids-limit 256
    --tmpfs "/tmp:rw,noexec,nosuid,nodev,size=$tmpfs_size,mode=1777"
    --env FORCE_DETERMINISTIC=1
    --env INVARLOCK_ALLOW_HOST_EXECUTION=0
    --env INVARLOCK_ALLOW_NETWORK=0
    --env INVARLOCK_ALLOW_REMOTE_CODE=0
    --env INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0
    --env INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE=0
    --env INVARLOCK_CONTAINER_EXECUTION=1
    --env "INVARLOCK_RUNTIME_IMAGE=$image_digest"
    --env "INVARLOCK_RUNTIME_IMAGE_DIGEST=$image_digest"
    "${cache_options[@]}"
    --mount "type=bind,src=$artifact,dst=$artifact_target,readonly"
    --mount "type=bind,src=$settings,dst=/inputs/settings.json,readonly"
    "${provider_mounts[@]}"
  )

  if [[ "$phase" == "prepare" ]]; then
    command+=(
      --mount "type=bind,src=$BINDINGS_DIR,dst=/outputs"
      "${launcher[@]}"
      advanced runtime-behavior prepare-binding
      --provider "$provider"
      --model-id "$model_id"
      --settings /inputs/settings.json
      --artifact "$artifact_target"
      --backend-executable "$backend_executable"
      "${provider_options[@]}"
      --container-image-digest "$image_digest"
      --out "/outputs/$role-binding.json"
      --json
    )
  elif [[ "$phase" == "run" ]]; then
    command+=(
      --mount "type=bind,src=$SCHEDULE,dst=/inputs/schedule.json,readonly"
      --mount "type=bind,src=$POLICY_PACK,dst=/inputs/policy-pack.json,readonly"
      --mount "type=bind,src=$SIDES_DIR,dst=/outputs"
      "${launcher[@]}"
      advanced runtime-behavior run-side
      --role "$role"
      --provider "$provider"
      --model-id "$model_id"
      --settings /inputs/settings.json
      --artifact "$artifact_target"
      --backend-executable "$backend_executable"
      "${provider_options[@]}"
      --container-image-digest "$image_digest"
      --schedule /inputs/schedule.json
      --policy-pack /inputs/policy-pack.json
      --out "/outputs/$role"
      --json
    )
  else
    die "unsupported container phase: $phase"
  fi

  "${command[@]}"
}

for name in \
  INVARLOCK_RECORDS \
  INVARLOCK_DATASET_IDENTITY \
  INVARLOCK_NATIVE_WORK_DIR \
  INVARLOCK_BASELINE_PROVIDER \
  INVARLOCK_BASELINE_IMAGE_DIGEST \
  INVARLOCK_BASELINE_ARTIFACT \
  INVARLOCK_BASELINE_SETTINGS \
  INVARLOCK_SUBJECT_PROVIDER \
  INVARLOCK_SUBJECT_IMAGE_DIGEST \
  INVARLOCK_SUBJECT_ARTIFACT \
  INVARLOCK_SUBJECT_SETTINGS; do
  required_value "$name"
done

CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
INVARLOCK_CLI="${INVARLOCK_CLI:-invarlock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TENSORRT_TMPFS_SIZE="${INVARLOCK_TENSORRT_TMPFS_SIZE:-16g}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

[[ "$HOST_UID" =~ ^[0-9]+$ && "$HOST_GID" =~ ^[0-9]+$ ]] ||
  die "invoking uid and gid must be nonnegative integers"

command -v "$CONTAINER_ENGINE" >/dev/null || die "$CONTAINER_ENGINE is unavailable"
command -v "$INVARLOCK_CLI" >/dev/null || die "$INVARLOCK_CLI is unavailable"
command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN is unavailable"

validate_provider INVARLOCK_BASELINE_PROVIDER "$INVARLOCK_BASELINE_PROVIDER"
validate_provider INVARLOCK_SUBJECT_PROVIDER "$INVARLOCK_SUBJECT_PROVIDER"
validate_image_digest INVARLOCK_BASELINE_IMAGE_DIGEST "$INVARLOCK_BASELINE_IMAGE_DIGEST"
validate_image_digest INVARLOCK_SUBJECT_IMAGE_DIGEST "$INVARLOCK_SUBJECT_IMAGE_DIGEST"

required_file records "$INVARLOCK_RECORDS"
required_file "dataset identity" "$INVARLOCK_DATASET_IDENTITY"
required_file "baseline settings" "$INVARLOCK_BASELINE_SETTINGS"
required_file "subject settings" "$INVARLOCK_SUBJECT_SETTINGS"
if [[ "$INVARLOCK_BASELINE_PROVIDER" == "llama_cpp" ]]; then
  required_file "baseline artifact" "$INVARLOCK_BASELINE_ARTIFACT"
else
  required_directory "baseline artifact" "$INVARLOCK_BASELINE_ARTIFACT"
fi
if [[ "$INVARLOCK_SUBJECT_PROVIDER" == "llama_cpp" ]]; then
  required_file "subject artifact" "$INVARLOCK_SUBJECT_ARTIFACT"
else
  required_directory "subject artifact" "$INVARLOCK_SUBJECT_ARTIFACT"
fi

RECORDS="$(absolute_existing_path "$INVARLOCK_RECORDS")"
DATASET_IDENTITY="$(absolute_existing_path "$INVARLOCK_DATASET_IDENTITY")"
BASELINE_ARTIFACT="$(absolute_existing_path "$INVARLOCK_BASELINE_ARTIFACT")"
BASELINE_SETTINGS="$(absolute_existing_path "$INVARLOCK_BASELINE_SETTINGS")"
SUBJECT_ARTIFACT="$(absolute_existing_path "$INVARLOCK_SUBJECT_ARTIFACT")"
SUBJECT_SETTINGS="$(absolute_existing_path "$INVARLOCK_SUBJECT_SETTINGS")"

for pair in \
  "records:$RECORDS" \
  "dataset identity:$DATASET_IDENTITY" \
  "baseline artifact:$BASELINE_ARTIFACT" \
  "baseline settings:$BASELINE_SETTINGS" \
  "subject artifact:$SUBJECT_ARTIFACT" \
  "subject settings:$SUBJECT_SETTINGS"; do
  safe_mount_source "${pair%%:*}" "${pair#*:}"
done

[[ ! -e "$INVARLOCK_NATIVE_WORK_DIR" ]] || die "work directory must not already exist: $INVARLOCK_NATIVE_WORK_DIR"
mkdir -p -- "$INVARLOCK_NATIVE_WORK_DIR"
WORK_DIR="$(cd -- "$INVARLOCK_NATIVE_WORK_DIR" && pwd -P)"
CONTROL_DIR="$WORK_DIR/control"
BINDINGS_DIR="$WORK_DIR/bindings"
SIDES_DIR="$WORK_DIR/sides"
mkdir -- "$CONTROL_DIR" "$BINDINGS_DIR" "$SIDES_DIR"
SCHEDULE="$CONTROL_DIR/behavioral-schedule.json"
POLICY_PACK="$CONTROL_DIR/acceptance-policy-pack.json"
PAIR_RECEIPT="$WORK_DIR/paired-receipt.json"

BASELINE_MODEL_ID="$(model_id_from_settings "$INVARLOCK_BASELINE_PROVIDER" "$BASELINE_SETTINGS")"
SUBJECT_MODEL_ID="$(model_id_from_settings "$INVARLOCK_SUBJECT_PROVIDER" "$SUBJECT_SETTINGS")"

"$INVARLOCK_CLI" advanced runtime-behavior build-schedule \
  --records "$RECORDS" \
  --dataset-identity "$DATASET_IDENTITY" \
  --out "$SCHEDULE" \
  --json
required_output_file "behavioral schedule" "$SCHEDULE"

container_cli prepare baseline \
  "$INVARLOCK_BASELINE_PROVIDER" \
  "$INVARLOCK_BASELINE_IMAGE_DIGEST" \
  "$BASELINE_ARTIFACT" \
  "$BASELINE_SETTINGS" \
  "${INVARLOCK_BASELINE_TOKENIZER_CONTRACT:-}" \
  "${INVARLOCK_BASELINE_GPU:-}" \
  "$BASELINE_MODEL_ID"
required_output_file "baseline binding" "$BINDINGS_DIR/baseline-binding.json"

container_cli prepare subject \
  "$INVARLOCK_SUBJECT_PROVIDER" \
  "$INVARLOCK_SUBJECT_IMAGE_DIGEST" \
  "$SUBJECT_ARTIFACT" \
  "$SUBJECT_SETTINGS" \
  "${INVARLOCK_SUBJECT_TOKENIZER_CONTRACT:-}" \
  "${INVARLOCK_SUBJECT_GPU:-}" \
  "$SUBJECT_MODEL_ID"
required_output_file "subject binding" "$BINDINGS_DIR/subject-binding.json"

"$INVARLOCK_CLI" advanced runtime-behavior build-policy \
  --schedule "$SCHEDULE" \
  --baseline-binding "$BINDINGS_DIR/baseline-binding.json" \
  --subject-binding "$BINDINGS_DIR/subject-binding.json" \
  --tier balanced \
  --minimum-subject-score 0.95 \
  --maximum-regression 0.01 \
  --evidence-surface behavior \
  --evidence-surface tokenizer \
  --out "$POLICY_PACK" \
  --json
required_output_file "acceptance policy pack" "$POLICY_PACK"

container_cli run baseline \
  "$INVARLOCK_BASELINE_PROVIDER" \
  "$INVARLOCK_BASELINE_IMAGE_DIGEST" \
  "$BASELINE_ARTIFACT" \
  "$BASELINE_SETTINGS" \
  "${INVARLOCK_BASELINE_TOKENIZER_CONTRACT:-}" \
  "${INVARLOCK_BASELINE_GPU:-}" \
  "$BASELINE_MODEL_ID"
required_side_bundle baseline "$SIDES_DIR/baseline"

container_cli run subject \
  "$INVARLOCK_SUBJECT_PROVIDER" \
  "$INVARLOCK_SUBJECT_IMAGE_DIGEST" \
  "$SUBJECT_ARTIFACT" \
  "$SUBJECT_SETTINGS" \
  "${INVARLOCK_SUBJECT_TOKENIZER_CONTRACT:-}" \
  "${INVARLOCK_SUBJECT_GPU:-}" \
  "$SUBJECT_MODEL_ID"
required_side_bundle subject "$SIDES_DIR/subject"

"$INVARLOCK_CLI" advanced runtime-behavior verify-pair \
  --baseline "$SIDES_DIR/baseline" \
  --subject "$SIDES_DIR/subject" \
  --schedule "$SCHEDULE" \
  --policy-pack "$POLICY_PACK" \
  --receipt "$PAIR_RECEIPT" \
  --json
required_output_file "paired receipt" "$PAIR_RECEIPT"

printf 'Native runtime pair verified: %s\n' "$PAIR_RECEIPT"
