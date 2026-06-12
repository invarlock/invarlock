#!/usr/bin/env bash
# queue_manager.sh - Compatibility facade for queue management modules
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: task_serialization.sh, jq
# Usage: sourced by gpu_worker.sh and scheduler to manage task lifecycle

QUEUE_MANAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
[[ -z "${QUEUE_CORE_LOADED:-}" ]] && source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_core.sh"
# shellcheck source=queue_lifecycle.sh
[[ -z "${QUEUE_LIFECYCLE_LOADED:-}" ]] && source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_lifecycle.sh" && export QUEUE_LIFECYCLE_LOADED=1
# shellcheck source=queue_dependencies.sh
[[ -z "${QUEUE_DEPENDENCIES_LOADED:-}" ]] && source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_dependencies.sh" && export QUEUE_DEPENDENCIES_LOADED=1
# shellcheck source=queue_memory_plan.sh
[[ -z "${QUEUE_MEMORY_PLAN_LOADED:-}" ]] && source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_memory_plan.sh" && export QUEUE_MEMORY_PLAN_LOADED=1
# shellcheck source=queue_generation.sh
[[ -z "${QUEUE_GENERATION_LOADED:-}" ]] && source "${QUEUE_MANAGER_SCRIPT_DIR}/queue_generation.sh" && export QUEUE_GENERATION_LOADED=1

# Keep the facade safe to source from strict callers when the final guarded
# source is skipped because the module was already loaded.
true
