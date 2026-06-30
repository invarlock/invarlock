#!/usr/bin/env bash

# Shared preflight helpers for integration example runners. Source this file
# after SCRIPT_DIR, REPO_ROOT, and PYTHON_BIN have been resolved.

integration_log_header() {
  local title="$1"
  printf '\n==> %s\n' "$title" >&2
}

integration_log_step() {
  local message="$1"
  printf '  -> %s\n' "$message" >&2
}

integration_log_kv() {
  local key="$1"
  local value="$2"
  printf '     %s: %s\n' "$key" "${value:-<unset>}" >&2
}

integration_filter_source_archive_stderr() {
  local line

  while IFS= read -r line; do
    case "$line" in
      "fatal: not a git repository (or any of the parent directories): .git")
        ;;
      *)
        printf '%s\n' "$line" >&2
        ;;
    esac
  done
}

integration_run_source_archive_clean() {
  if [[ -n "${REPO_ROOT:-}" && -d "$REPO_ROOT/.git" ]]; then
    "$@"
  else
    local stderr_file
    stderr_file="$(mktemp "${TMPDIR:-/tmp}/invarlock-source-archive-stderr.XXXXXX")" || return 1
    local had_errexit=0
    case "$-" in
      *e*)
        had_errexit=1
        set +e
        ;;
    esac
    "$@" 2>"$stderr_file"
    local rc=$?
    if [[ "$had_errexit" == "1" ]]; then
      set -e
    fi
    integration_filter_source_archive_stderr <"$stderr_file"
    rm -f "$stderr_file"
    return "$rc"
  fi
}

integration_effective_execution_mode() {
  local lane="$1"
  local execution_mode="$2"

  case "$lane" in
    host)
      printf 'host\n'
      ;;
    cuda)
      printf 'container\n'
      ;;
    *)
      printf '%s\n' "$execution_mode"
      ;;
  esac
}

integration_effective_assurance() {
  local lane="$1"
  local assurance="$2"

  case "$lane" in
    host)
      printf 'off\n'
      ;;
    cuda)
      printf 'strict\n'
      ;;
    *)
      printf '%s\n' "$assurance"
      ;;
  esac
}

integration_effective_device() {
  local lane="$1"
  local device="$2"

  case "$lane" in
    cuda)
      printf 'cuda\n'
      ;;
    *)
      printf '%s\n' "$device"
      ;;
  esac
}

integration_default_host_device() {
  local execution_mode="$1"
  local device="$2"

  if [[ "$execution_mode" == "host" && -z "$device" ]]; then
    printf 'cpu\n'
    return 0
  fi

  printf '%s\n' "$device"
}

integration_lane_artifact_label() {
  local execution_mode="$1"
  local assurance="$2"
  local device="$3"

  if [[ "$execution_mode" == "container" && "$assurance" == "strict" && "$device" == cuda* ]]; then
    printf 'cuda-container-strict\n'
    return 0
  fi

  if [[ "$execution_mode" == "host" && "$assurance" == "off" ]]; then
    case "$device" in
      cuda*)
        printf 'cuda-host-off\n'
        ;;
      mps*)
        printf 'mps-host-off\n'
        ;;
      *)
        printf 'cpu-host-off\n'
        ;;
    esac
    return 0
  fi

  printf '%s-%s-%s\n' "${device:-auto}" "$execution_mode" "$assurance"
}

integration_lane_report_out() {
  local report_out="$1"
  local report_out_was_default="$2"
  local lane_artifact_label="$3"

  if [[ "$report_out_was_default" == "1" ]]; then
    printf '%s/%s\n' "$report_out" "$lane_artifact_label"
    return 0
  fi

  printf '%s\n' "$report_out"
}

integration_preflight_host_cuda_device() {
  local python_bin="$1"
  local execution_mode="$2"
  local device="$3"
  local example_name="$4"

  if [[ "$execution_mode" != "host" || "$device" != cuda* ]]; then
    return 0
  fi

  if ! "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    cat >&2 <<MSG
$example_name host CUDA lane requires torch.cuda to be available.

Use --device cpu for a cpu-host-off run when the backend supports CPU, or use
--lane cuda with the documented runtime image for container-backed strict
evidence on a CUDA host.
MSG
    return 2
  fi
}

integration_preflight_gptqmodel_host_runtime() {
  local python_bin="$1"
  local execution_mode="$2"

  if [[ "$execution_mode" != "host" ]]; then
    return 0
  fi

  if ! command -v gcc >/dev/null 2>&1; then
    cat >&2 <<'MSG'
GPTQModel host lanes require a C compiler because Triton may compile CUDA
runtime helpers during model loading.

Install gcc/build-essential for the host Python environment, or use --lane cuda
with the documented container image for strict evidence.
MSG
    return 2
  fi

  if ! "$python_bin" - <<'PY' >/dev/null 2>&1
from pathlib import Path
import sysconfig

candidate_dirs = {
    sysconfig.get_paths().get("include"),
    sysconfig.get_paths().get("platinclude"),
    sysconfig.get_config_var("INCLUDEPY"),
    sysconfig.get_config_var("CONFINCLUDEPY"),
}
raise SystemExit(
    0
    if any(
        include_dir and (Path(str(include_dir)) / "Python.h").exists()
        for include_dir in candidate_dirs
    )
    else 1
)
PY
  then
    cat >&2 <<'MSG'
GPTQModel host lanes require Python development headers because Triton may
compile CUDA runtime helpers during model loading.

Install the matching Python development package for this interpreter, for
example python3-dev/python3.12-dev on Debian or Ubuntu, or use --lane cuda with
the documented container image for strict evidence.
MSG
    return 2
  fi
}
