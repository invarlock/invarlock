#!/usr/bin/env bash

smoke_ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

smoke_select_python() {
  local repo_root="$1"
  local configured="${2:-}"
  if [[ -n "$configured" ]]; then
    printf '%s\n' "$configured"
    return 0
  fi
  bash "$repo_root/scripts/select_workspace_python.sh"
}

smoke_setup_pythonpath() {
  local repo_root="$1"
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
}

smoke_host_gpu_visible() {
  [[ -e /dev/nvidiactl ]] || command -v nvidia-smi >/dev/null 2>&1
}

smoke_seed_local_runtime_image() {
  local device="${1:-auto}"
  if [[ -n "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
    return 0
  fi
  if [[ "$device" != "cpu" ]] \
    && smoke_host_gpu_visible \
    && command -v docker >/dev/null 2>&1 \
    && docker image inspect invarlock-runtime:cuda-local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  if command -v docker >/dev/null 2>&1 \
    && docker image inspect invarlock-runtime:local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
  fi
}

smoke_ensure_current_runtime_image() {
  local mode="${1:-container}"
  local device="${2:-auto}"
  if [[ "$mode" != "container" ]]; then
    return 0
  fi
  if [[ -n "${INVARLOCK_RUNTIME_IMAGE:-}" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:local" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:cuda-local" ]]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$device" != "cpu" ]] && smoke_host_gpu_visible; then
    echo "[smoke] refreshing local CUDA container runtime image"
    make runtime-image-cuda
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  echo "[smoke] refreshing local container runtime image"
  make runtime-image
  export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
}

smoke_copy_cached_tree_if_present() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ -d "$source_dir" && ! -e "$target_dir" ]]; then
    mkdir -p "$(dirname -- "$target_dir")"
    cp -a "$source_dir" "$target_dir"
  fi
}

smoke_ensure_writable_hf_cache() {
  local smoke_cache_root="$1"
  local candidate_home="${HF_HOME:-$smoke_cache_root}"
  local candidate_datasets="${HF_DATASETS_CACHE:-${candidate_home}/datasets}"
  local candidate_hub="${HF_HUB_CACHE:-${candidate_home}/hub}"

  if mkdir -p "$candidate_home" "$candidate_datasets" "$candidate_hub" >/dev/null 2>&1 \
    && touch "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1; then
    rm -f "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1 || true
    export HF_HOME="$candidate_home"
    export HF_HUB_CACHE="$candidate_hub"
    export HF_DATASETS_CACHE="$candidate_datasets"
    unset TRANSFORMERS_CACHE
    return 0
  fi

  export HF_HOME="$smoke_cache_root"
  export HF_HUB_CACHE="$smoke_cache_root/hub"
  export HF_DATASETS_CACHE="$smoke_cache_root/datasets"
  unset TRANSFORMERS_CACHE
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
  echo "[smoke] falling back to writable HF cache under $HF_HOME"
}

smoke_expected_exit_match() {
  local actual="$1"
  local expected_csv="${2:-0}"
  local expected=""
  IFS=',' read -r -a expected <<<"$expected_csv"
  for expected in "${expected[@]}"; do
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
  done
  return 1
}

smoke_plan_markers() {
  local kind="$1"
  local source_path="$2"
  local marker="# smoke-plan-${kind}:"
  awk -v marker="$marker" '
    index($0, marker) {
      value = substr($0, index($0, marker) + length(marker))
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if (value != "" && !(value in seen)) {
        seen[value] = 1
        print value
      }
    }
  ' "$source_path"
}

smoke_resolve_container_engine() {
  if command -v docker >/dev/null 2>&1; then
    echo "docker"
    return 0
  fi
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return 0
  fi
  return 1
}
