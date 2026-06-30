#!/usr/bin/env bash
# queue_manager.sh - Queue management module aggregator
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: task_serialization.sh, jq
# Usage: sourced by gpu_worker.sh and scheduler to manage task lifecycle

QUEUE_MANAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
if ! declare -F init_queue >/dev/null 2>&1; then
    source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_core.sh"
fi
QUEUE_CORE_LOADED=1
export -n QUEUE_CORE_LOADED 2>/dev/null || true
# shellcheck source=queue_lifecycle.sh
if ! declare -F claim_task >/dev/null 2>&1; then
    source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_lifecycle.sh"
fi
QUEUE_LIFECYCLE_LOADED=1
export -n QUEUE_LIFECYCLE_LOADED 2>/dev/null || true
# shellcheck source=queue_dependencies.sh
if ! declare -F check_dependencies_met >/dev/null 2>&1; then
    source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_dependencies.sh"
fi
QUEUE_DEPENDENCIES_LOADED=1
export -n QUEUE_DEPENDENCIES_LOADED 2>/dev/null || true
# shellcheck source=queue_memory_plan.sh
if ! declare -F update_model_task_memory >/dev/null 2>&1; then
    source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_memory_plan.sh"
fi
QUEUE_MEMORY_PLAN_LOADED=1
export -n QUEUE_MEMORY_PLAN_LOADED 2>/dev/null || true
# shellcheck source=queue_generation.sh
if ! declare -F generate_all_tasks >/dev/null 2>&1; then
    source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_generation.sh"
fi
QUEUE_GENERATION_LOADED=1
export -n QUEUE_GENERATION_LOADED 2>/dev/null || true

# Keep the aggregator safe to source from strict callers when the final guarded
# source is skipped because the module was already loaded.
true
