#!/usr/bin/env bash
# scheduler.sh - Memory-aware scheduling module aggregator
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: queue_manager.sh, task_serialization.sh, nvidia-smi
# Usage: sourced by gpu_worker.sh to select tasks per GPU memory headroom

SCHEDULER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scheduler_core.sh
if ! declare -F acquire_scheduler_lock >/dev/null 2>&1; then
    source "${SCHEDULER_SCRIPT_DIR}/scheduler_core.sh"
fi
SCHEDULER_CORE_LOADED=1
export -n SCHEDULER_CORE_LOADED 2>/dev/null || true
# shellcheck source=scheduler_gpu_runtime.sh
if ! declare -F get_gpu_available_memory >/dev/null 2>&1; then
    source "${SCHEDULER_SCRIPT_DIR}/scheduler_gpu_runtime.sh"
fi
SCHEDULER_GPU_RUNTIME_LOADED=1
export -n SCHEDULER_GPU_RUNTIME_LOADED 2>/dev/null || true
# shellcheck source=scheduler_reservations.sh
if ! declare -F reserve_gpus >/dev/null 2>&1; then
    source "${SCHEDULER_SCRIPT_DIR}/scheduler_reservations.sh"
fi
SCHEDULER_RESERVATIONS_LOADED=1
export -n SCHEDULER_RESERVATIONS_LOADED 2>/dev/null || true
# shellcheck source=scheduler_selection.sh
if ! declare -F find_and_claim_task >/dev/null 2>&1; then
    source "${SCHEDULER_SCRIPT_DIR}/scheduler_selection.sh"
fi
SCHEDULER_SELECTION_LOADED=1
export -n SCHEDULER_SELECTION_LOADED 2>/dev/null || true

# Keep the aggregator safe to source from strict callers when the final guarded
# source is skipped because the module was already loaded.
true
