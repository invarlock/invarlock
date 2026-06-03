#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: smoke_example_runtime_image.sh FAMILY [--engine docker|podman]

Families:
  cuda-bnb
  cuda-compressed-tensors
  cuda-gptqmodel
  cuda-hqq
  cuda-quanto
  cuda-torchao
USAGE
}

family="${1:-}"
if [[ -n "$family" ]]; then
  shift
fi

engine="${CONTAINER_ENGINE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      engine="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$family" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$engine" ]]; then
  if command -v docker >/dev/null 2>&1; then
    engine="docker"
  elif command -v podman >/dev/null 2>&1; then
    engine="podman"
  else
    echo "An OCI container engine (Docker or Podman) is required." >&2
    exit 2
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
image="invarlock-example-runtime:${family}"
require_gpu="${INVARLOCK_EXAMPLE_RUNTIME_REQUIRE_GPU:-auto}"
gpu_args=()

case "$family" in
  cuda-bnb)
    smoke_args=(--adapters hf_bnb)
    ;;
  cuda-compressed-tensors)
    smoke_args=(--adapters hf_ct)
    ;;
  cuda-gptqmodel)
    smoke_args=(--adapters hf_awq,hf_gptq --require-cuda-toolchain)
    ;;
  cuda-hqq)
    smoke_args=(--adapters hf_hqq)
    ;;
  cuda-quanto)
    smoke_args=(--adapters hf_quanto)
    ;;
  cuda-torchao)
    smoke_args=(--adapters hf_torchao)
    ;;
  *)
    echo "Unknown family: $family" >&2
    usage >&2
    exit 2
    ;;
esac

case "$require_gpu" in
  auto)
    if [[ "$engine" == "docker" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      gpu_args=(--gpus all)
      smoke_args+=(--require-gpu)
    fi
    ;;
  1|true|yes)
    if [[ "$engine" != "docker" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "GPU-required smoke needs Docker on a host with nvidia-smi." >&2
      exit 2
    fi
    gpu_args=(--gpus all)
    smoke_args+=(--require-gpu)
    ;;
  0|false|no)
    ;;
  *)
    echo "Invalid INVARLOCK_EXAMPLE_RUNTIME_REQUIRE_GPU=$require_gpu" >&2
    exit 2
    ;;
esac

"$engine" run --rm \
  "${gpu_args[@]}" \
  -v "$SCRIPT_DIR/quant_runtime_image_smoke.py:/tmp/quant_runtime_image_smoke.py:ro" \
  --entrypoint python \
  "$image" \
  /tmp/quant_runtime_image_smoke.py "${smoke_args[@]}"
