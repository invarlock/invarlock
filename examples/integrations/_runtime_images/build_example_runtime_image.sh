#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: build_example_runtime_image.sh FAMILY [--engine docker|podman]

Families:
  cuda-bnb
  cuda-compressed-tensors
  cuda-gptqmodel
  cuda-hqq
  cuda-quanto
  cuda-torchao

Builds example-only CUDA runtime images for integration evidence runs.
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
requirements="$SCRIPT_DIR/requirements/${family}-py312-cu128.txt"
image="invarlock-example-runtime:${family}"
context_dir="${TMPDIR:-/tmp}/invarlock-example-runtime-${family}-context"

case "$family" in
  cuda-bnb|cuda-compressed-tensors|cuda-hqq|cuda-quanto|cuda-torchao)
    base_args=()
    ;;
  cuda-gptqmodel)
    base_args=(
      --build-arg RUNTIME_BASE_IMAGE="${RUNTIME_IMAGE_CUDA_QUANT_BASE:-nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66}"
      --build-arg RUNTIME_CUDA_HOME=/usr/local/cuda
      --build-arg RUNTIME_KEEP_BUILD_TOOLCHAIN=1
      --build-arg RUNTIME_PATH_PREFIX=/usr/local/cuda/bin:
    )
    ;;
  *)
    echo "Unknown family: $family" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -s "$requirements" ]]; then
  echo "Missing locked requirements: $requirements" >&2
  exit 2
fi

rm -rf "$context_dir"
mkdir -p "$context_dir"
cp "$SCRIPT_DIR/Dockerfile" "$context_dir/Dockerfile"
cp "$requirements" "$context_dir/requirements.txt"
cp -R "$REPO_ROOT/src" "$context_dir/src"

"$engine" build \
  "${base_args[@]}" \
  --build-arg PYTORCH_EXTRA_INDEX_URL="${PYTORCH_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  -f "$context_dir/Dockerfile" \
  -t "$image" \
  "$context_dir"

"$engine" image inspect "$image" --format '{{.RepoTags}} {{.Id}} {{.Size}}'
