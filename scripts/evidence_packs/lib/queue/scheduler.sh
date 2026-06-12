#!/usr/bin/env bash
# scheduler.sh - Compatibility facade for memory-aware scheduling modules
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: queue_manager.sh, task_serialization.sh, nvidia-smi
# Usage: sourced by gpu_worker.sh to select tasks per GPU memory headroom

SCHEDULER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scheduler_core.sh
[[ -z "${SCHEDULER_CORE_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/scheduler_core.sh"
# shellcheck source=scheduler_gpu_runtime.sh
[[ -z "${SCHEDULER_GPU_RUNTIME_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/scheduler_gpu_runtime.sh" && export SCHEDULER_GPU_RUNTIME_LOADED=1
# shellcheck source=scheduler_reservations.sh
[[ -z "${SCHEDULER_RESERVATIONS_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/scheduler_reservations.sh" && export SCHEDULER_RESERVATIONS_LOADED=1
# shellcheck source=scheduler_selection.sh
[[ -z "${SCHEDULER_SELECTION_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/scheduler_selection.sh" && export SCHEDULER_SELECTION_LOADED=1

# Keep the facade safe to source from strict callers when the final guarded
# source is skipped because the module was already loaded.
true
