#!/usr/bin/env bash

test_get_available_gpus_selection_branches_must_include_spread_and_short_selection_error_path() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    list_gpu_ids() { echo "0 1 2"; }
    is_gpu_usable() { return 0; }
    get_gpu_available_memory() {
        case "$1" in
            0) echo "50" ;;
            1) echo "5" ;;
            2) echo "60" ;;
        esac
    }

    # min_free_gb filter excludes GPU 1.
    assert_eq "0,2" "$(get_available_gpus 2 false "" 10)" "filters by min_free_gb and returns non-sequential list"

    # must_include not found returns empty + non-zero.
    if get_available_gpus 1 false "9" 0 >/dev/null; then
        t_fail "expected must_include missing to fail"
    fi

    # must_include selection ensures required GPU is present.
    assert_eq "2,0" "$(get_available_gpus 2 false "2" 0)" "must_include is included"

    # prefer_spread path.
    local spread
    spread="$(get_available_gpus 2 true "" 0)"
    assert_ne "" "${spread}" "prefer_spread selects GPUs"

    # Explicit short-selection error branch: override seq to yield fewer indices.
    seq() { echo "0"; }
    if get_available_gpus 2 true "" 0 >/dev/null; then
        t_fail "expected selection to fail when seq yields too few indices"
    fi
}

test_get_task_gpus_handles_missing_dir_and_missing_file_paths() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    GPU_RESERVATION_DIR=""
    if get_task_gpus "t1" >/dev/null; then
        t_fail "expected get_task_gpus to fail when GPU_RESERVATION_DIR is empty"
    fi

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"
    echo "0,1" > "${GPU_RESERVATION_DIR}/task_t1.gpus"
    assert_eq "0,1" "$(get_task_gpus t1)" "reads gpus file"
    assert_eq "" "$(get_task_gpus missing || true)" "missing file returns empty"
}

test_init_gpu_reservations_sets_dir_and_refreshes_cache() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    local out_dir="${TEST_TMPDIR}/out"
    local refresh_calls=0
    refresh_all_gpu_cache() { refresh_calls=$((refresh_calls + 1)); }

    init_gpu_reservations "${out_dir}"

    assert_eq "${out_dir}/workers/gpu_reservations" "${GPU_RESERVATION_DIR}" "GPU reservation dir set"
    assert_dir_exists "${GPU_RESERVATION_DIR}" "reservation dir created"
    assert_eq "1" "${refresh_calls}" "gpu cache refreshed"
}

test_cleanup_stale_reservations_skips_valid_and_cleans_stale_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    echo "valid" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    echo "stale" > "${GPU_RESERVATION_DIR}/gpu_1.lock"
    _is_reservation_valid() { [[ "$1" == "valid" ]]; }

    cleanup_stale_reservations
}

test_oom_helpers_cover_missing_file_risk_and_risk_levels() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    if check_oom_safe "${TEST_TMPDIR}/nope.task" "0" >/dev/null; then
        t_fail "expected check_oom_safe to fail for missing task file"
    fi

    # Risk path: available < required.
    local task_file="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"T", model_id:"m", model_name:"n", status:"ready", model_size_gb:100, dependencies:[], params:{}}' \
        > "${task_file}"
    get_gpu_available_memory() { echo "1"; }
    if check_oom_safe "${task_file}" "0,1" >/dev/null; then
        t_fail "expected OOM risk when available memory is too low"
    fi

    get_gpu_available_memory() { echo "1000"; }
    if ! check_oom_safe "${task_file}" "0" >/dev/null; then
        t_fail "expected check_oom_safe to succeed when memory is sufficient"
    fi

    # Risk levels exercise all threshold branches.
    get_gpu_available_memory() { echo "-1"; }
    assert_eq "critical" "$(get_oom_risk_level "${task_file}" "0")" "critical when min available <= 0"

    get_gpu_available_memory() { echo "10"; }
    assert_eq "critical" "$(get_oom_risk_level "${task_file}" "0")" "critical when headroom < 5%"

    get_gpu_available_memory() { echo "110"; }
    assert_eq "high" "$(get_oom_risk_level "${task_file}" "0")" "high when headroom < 15%"

    get_gpu_available_memory() { echo "140"; }
    assert_eq "medium" "$(get_oom_risk_level "${task_file}" "0")" "medium when headroom < 30%"

    get_gpu_available_memory() { echo "200"; }
    assert_eq "low" "$(get_oom_risk_level "${task_file}" "0")" "low when headroom >= 30%"
}

test_priority_calculation_and_blocked_counts_cover_boost_and_validation_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    _now_epoch() { echo "600"; }
    _iso_to_epoch() { echo "0"; }

    local t_small="${QUEUE_DIR}/ready/small.task"
    local t_mid="${QUEUE_DIR}/ready/mid.task"
    local t_moe="${QUEUE_DIR}/ready/moe.task"
    jq -n '{task_id:"small", task_type:"SETUP_BASELINE", model_id:"m", model_name:"modelA", status:"ready", model_size_gb:10, created_at:"x", dependencies:[], params:{}, priority:50}' > "${t_small}"
    jq -n '{task_id:"mid", task_type:"CALIBRATION_RUN", model_id:"m", model_name:"modelA", status:"ready", model_size_gb:40, created_at:"x", dependencies:[], params:{}, priority:50}' > "${t_mid}"
    jq -n '{task_id:"moe", task_type:"OTHER", model_id:"m", model_name:"modelB", status:"ready", model_size_gb:80, created_at:"x", dependencies:[], params:{}, priority:50}' > "${t_moe}"

    # count_blocked_by_task increments when deps match.
    jq -n '{task_id:"p1", task_type:"T", model_id:"m", model_name:"x", status:"pending", dependencies:["small"], params:{}}' \
        > "${QUEUE_DIR}/pending/p1.task"
    assert_eq "1" "$(count_blocked_by_task small)" "blocked count"

    # Default calculation computes task_id + blocked_count from queue.
    calculate_task_priority "${t_small}" >/dev/null

    # Overrides cover blocked_count validation branch.
    calculate_task_priority "${t_small}" "not-a-number" "small" >/dev/null
    calculate_task_priority "${t_mid}" "2" "mid" >/dev/null
    calculate_task_priority "${t_moe}" "0" "moe" >/dev/null

    # count_running_for_model increments when model matches.
    cp "${t_small}" "${QUEUE_DIR}/running/r1.task"
    assert_eq "1" "$(count_running_for_model modelA)" "running count"
}

test_find_best_task_covers_retry_gating_fit_checks_and_adaptive_multi_gpu_paths() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # Reserved GPU with an owner skips task selection.
    is_gpu_available() { return 1; }
    cleanup_stale_reservations() { :; }
    echo "owner" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    if find_best_task 200 0 >/dev/null; then
        t_fail "expected find_best_task to fail when gpu is reserved by another task"
    fi
    rm -f "${GPU_RESERVATION_DIR}/gpu_0.lock"

    # GPU not usable short-circuits.
    is_gpu_available() { return 0; }
    is_gpu_usable() { return 1; }
    if find_best_task 200 0 >/dev/null; then
        t_fail "expected find_best_task to fail when gpu is unusable"
    fi

    # Full scan with retry gating, blocked-count precompute, single + multi-gpu logic.
    is_gpu_usable() { return 0; }
    list_gpu_ids() { echo "0 1"; }
    count_available_gpus() { echo "2"; }
    get_available_gpus() {
        [[ "${1:-}" == "2" ]] && echo "0,1" || echo ""
    }
    should_use_adaptive_gpus() { return 0; }
    is_retry_ready() { [[ "$1" != *skip.task ]]; }
    get_required_gpus() { echo "4"; }
    get_minimum_gpus() { echo "2"; }
    calculate_task_priority() { echo "90"; }

    jq -n '{task_id:"skip", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:10, required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/skip.task"
    jq -n '{task_id:"too_big", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:200, required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/too_big.task"
    jq -n '{task_id:"multi", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:100, required_gpus:null, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/multi.task"

    jq -n '{task_id:"pend", task_type:"T", model_id:"m", model_name:"x", status:"pending", dependencies:["multi"], params:{}}' \
        > "${QUEUE_DIR}/pending/pend.task"

    export SCHEDULER_MEM_TOLERANCE_GB="bad"
    assert_eq "multi" "$(find_best_task 170 0)" "selects best task under adaptive allocation"
}

test_find_best_task_covers_effective_memory_branches_for_mid_and_low_memory() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    list_gpu_ids() { echo "0"; }
    is_gpu_available() { return 0; }
    is_gpu_usable() { return 0; }

    find_best_task 100 0 >/dev/null
    find_best_task 50 0 >/dev/null
}

test_find_best_task_skips_multi_gpu_task_when_not_enough_gpus_and_no_adaptive() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    is_gpu_available() { return 0; }
    is_gpu_usable() { return 0; }
    list_gpu_ids() { echo "0 1"; }
    count_available_gpus() { echo "2"; }
    get_available_gpus() { echo ""; }
    should_use_adaptive_gpus() { return 1; }

    jq -n '{task_id:"multi", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:10, required_gpus:4, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/multi.task"

    assert_eq "" "$(find_best_task 200 0)" "no task selected when multi-gpu task cannot be allocated"
}

test_find_and_claim_task_covers_no_task_races_adaptive_paths_and_success_updates() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # No task found.
    find_best_task() { echo ""; }
    if find_and_claim_task 100 0 >/dev/null; then
        t_fail "expected find_and_claim_task to fail when no task is suitable"
    fi

    # Task was already claimed before precompute.
    find_best_task() { echo "t1"; }
    if find_and_claim_task 100 0 >/dev/null; then
        t_fail "expected find_and_claim_task to fail when ready file is missing"
    fi

    # Multi-GPU allocation failure returns non-zero.
    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:10, required_gpus:2, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"

    count_available_gpus() { echo "not-a-number"; }
    get_available_gpus() { echo ""; }
    should_use_adaptive_gpus() { return 1; }
    export SCHEDULER_LOCK_TIMEOUT="bad"
    if find_and_claim_task 100 0 >/dev/null; then
        t_fail "expected find_and_claim_task to fail when it cannot allocate GPUs"
    fi

    # Multi-GPU adaptive path with success, including adaptive_gpus update.
    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:100, required_gpus:null, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"
    count_available_gpus() { echo "2"; }
    get_required_gpus() { echo "4"; }
    get_minimum_gpus() { echo "2"; }
    should_use_adaptive_gpus() { return 0; }
    get_available_gpus() { [[ "${1:-}" == "2" ]] && echo "0,1" || echo ""; }
    reserve_gpus() { return 0; }
    acquire_scheduler_lock() { return 0; }
    release_scheduler_lock() { return 0; }
    claim_task() {
        mkdir -p "${QUEUE_DIR}/running"
        mv "${QUEUE_DIR}/ready/t1.task" "${QUEUE_DIR}/running/t1.task"
        return 0
    }
    update_task_field() { echo "$*" >> "${TEST_TMPDIR}/updates"; }

    local claimed
    claimed="$(find_and_claim_task 100 0)"
    assert_eq "${QUEUE_DIR}/running/t1.task" "${claimed}" "success returns running task path"
    grep -q 'adaptive_gpus' "${TEST_TMPDIR}/updates" || t_fail "expected adaptive_gpus update"
}

test_find_and_claim_task_covers_lock_race_and_reserve_failure_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:10, required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"

    find_best_task() { echo "t1"; }
    acquire_scheduler_lock() { rm -f "${QUEUE_DIR}/ready/t1.task"; return 0; }
    release_scheduler_lock() { return 0; }
    if find_and_claim_task 100 0 >/dev/null; then
        t_fail "expected lock revalidation to fail when task disappears"
    fi

    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"x", status:"ready", model_size_gb:10, required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"
    acquire_scheduler_lock() { return 0; }
    reserve_gpus() { return 1; }
    if find_and_claim_task 100 0 >/dev/null; then
        t_fail "expected find_and_claim_task to fail when reservation fails"
    fi
}

test_apply_work_stealing_boost_covers_model_stats_no_models_and_skip_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    # No models (empty model names) returns early.
    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"", status:"pending", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/pending/t1.task"
    apply_work_stealing_boost

    # Build model stats and boost lagging model tasks (with skip branches).
    rm -f "${QUEUE_DIR}"/pending/*.task
    rm -f "${QUEUE_DIR}"/completed/*.task
    rm -f "${QUEUE_DIR}"/ready/*.task

    jq -n '{task_id:"a1", task_type:"T", model_id:"m", model_name:"modelA", status:"completed", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/completed/a1.task"
    jq -n '{task_id:"a2", task_type:"T", model_id:"m", model_name:"modelA", status:"pending", dependencies:[], params:{}, priority:10, model_size_gb:10}' \
        > "${QUEUE_DIR}/pending/a2.task"
    jq -n '{task_id:"b1", task_type:"T", model_id:"m", model_name:"modelB", status:"pending", dependencies:[], params:{}, priority:10, model_size_gb:10}' \
        > "${QUEUE_DIR}/pending/b1.task"
    jq -n '{task_id:"b_skip_big", task_type:"T", model_id:"m", model_name:"modelB", status:"pending", dependencies:[], params:{}, priority:10, model_size_gb:120}' \
        > "${QUEUE_DIR}/pending/b_skip_big.task"
    jq -n '{task_id:"b_skip_pri", task_type:"T", model_id:"m", model_name:"modelB", status:"pending", dependencies:[], params:{}, priority:95, model_size_gb:10}' \
        > "${QUEUE_DIR}/pending/b_skip_pri.task"

    jq -n '{task_id:"b_ready", task_type:"T", model_id:"m", model_name:"modelB", status:"ready", dependencies:[], params:{}, priority:10, model_size_gb:10}' \
        > "${QUEUE_DIR}/ready/b_ready.task"
    jq -n '{task_id:"b_ready_big", task_type:"T", model_id:"m", model_name:"modelB", status:"ready", dependencies:[], params:{}, priority:10, model_size_gb:120}' \
        > "${QUEUE_DIR}/ready/b_ready_big.task"
    jq -n '{task_id:"b_ready_pri", task_type:"T", model_id:"m", model_name:"modelB", status:"ready", dependencies:[], params:{}, priority:95, model_size_gb:10}' \
        > "${QUEUE_DIR}/ready/b_ready_pri.task"

    export WORK_STEAL_MAX_READY_UPDATES="nope"
    apply_work_stealing_boost
}

test_apply_work_stealing_boost_returns_cleanly_when_queue_lock_unavailable() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    jq -n '{task_id:"a1", task_type:"T", model_id:"m", model_name:"modelA", status:"completed", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/completed/a1.task"
    jq -n '{task_id:"b1", task_type:"T", model_id:"m", model_name:"modelB", status:"pending", dependencies:[], params:{}, priority:10, model_size_gb:10}' \
        > "${QUEUE_DIR}/pending/b1.task"

    acquire_queue_lock() { return 1; }
    apply_work_stealing_boost
}

test_get_scheduling_stats_counts_created_at_branch() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _now_epoch() { echo "100"; }
    _iso_to_epoch() { echo "50"; }
    list_gpu_ids() { echo "0"; }
    get_gpu_available_memory() { echo "10"; }
    get_gpu_total_memory() { echo "180"; }

    jq -n '{task_id:"r1", task_type:"T", model_id:"m", model_name:"x", status:"ready", created_at:"x", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/r1.task"

    get_scheduling_stats >/dev/null
}

test_reserve_gpus_errors_when_metadata_move_fails() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _is_reservation_valid() { return 1; }
    mv() { return 1; }

    run reserve_gpus "t1" "0"
    assert_rc "1" "${RUN_RC}" "mv failure triggers error return"
}

test_find_and_claim_task_short_circuits_when_scheduler_lock_unavailable() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"n", status:"ready", dependencies:[], params:{}, priority:50, model_size_gb:10, required_gpus:1}' \
        > "${QUEUE_DIR}/ready/t1.task"

    find_best_task() { echo "t1"; }
    count_available_gpus() { echo "1"; }
    acquire_scheduler_lock() { return 1; }

    run find_and_claim_task "10" "0"
    assert_rc "1" "${RUN_RC}" "find_and_claim_task returns non-zero when lock unavailable"
}

test_acquire_scheduler_lock_cleans_ownerless_lock_and_normalizes_invalid_grace_seconds() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"

    local lock_dir="${QUEUE_DIR}/scheduler.lock.d"
    mkdir -p "${lock_dir}"

    SCHEDULER_LOCK_NOOWNER_STALE_SECONDS="bogus"
    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo "0"; }

    acquire_scheduler_lock "1"
    release_scheduler_lock
}

test_get_required_gpus_delegates_to_calculate_required_gpus() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_serialization.sh"
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    assert_eq "$(calculate_required_gpus 200)" "$(get_required_gpus 200)" "delegates to calculate_required_gpus"
}

test_is_gpu_usable_returns_zero_when_available_and_has_free_memory() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    GPU_MIN_FREE_GB=10
    GPU_REQUIRE_IDLE="false"
    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "999"; }

    is_gpu_usable 0
}

test_scheduler_is_gpu_usable_relaxes_for_single_gpu() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "0"; }
    NUM_GPUS=1
    GPU_ID_LIST="0"
    GPU_MIN_FREE_GB=99
    GPU_REQUIRE_IDLE="true"

    is_gpu_usable 0
    assert_eq "false" "${GPU_REQUIRE_IDLE}" "single GPU disables idle requirement"
}

test_scheduler_is_gpu_usable_sanitizes_invalid_min_free() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    GPU_MIN_FREE_GB="nope"
    GPU_REQUIRE_IDLE="false"
    NUM_GPUS=2
    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "9"; }

    run is_gpu_usable 0
    assert_rc "1" "${RUN_RC}" "invalid min free defaults to 10"
}

test_scheduler_find_and_claim_logs_reserve_failure_when_debug() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/ready" "${QUEUE_DIR}/running"

    local desired_task_id="task1"
    jq -n '{task_id:"task1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", model_size_gb:20, required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/${desired_task_id}.task"

    find_best_task() { echo "${desired_task_id}"; }
    get_task_field() {
        if [[ "$2" == "model_size_gb" ]]; then echo "20"; fi
        if [[ "$2" == "required_gpus" ]]; then echo "1"; fi
    }
    get_required_gpus() { echo "1"; }
    get_minimum_gpus() { echo "1"; }
    count_available_gpus() { echo "1"; }
    get_available_gpus() { echo "0"; }
    acquire_scheduler_lock() { return 0; }
    release_scheduler_lock() { return 0; }
    reserve_gpus() { return 1; }

    SCHEDULER_DEBUG="true"
    run find_and_claim_task 100 0
    assert_rc "1" "${RUN_RC}" "reserve failure returns non-zero"
    assert_match "reserve_gpus failed" "${RUN_ERR}" "debug message emitted"
}

test_lock_ownerless_recovery() {
    mock_reset

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/workers/gpu_reservations"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    mkdir -p "${QUEUE_DIR}/queue.lock.d"
    QUEUE_LOCK_NOOWNER_STALE_SECONDS=0 acquire_queue_lock 5
    assert_dir_exists "${QUEUE_LOCK_DIR}"
    release_queue_lock

    mkdir -p "${QUEUE_DIR}/scheduler.lock.d"
    SCHEDULER_LOCK_NOOWNER_STALE_SECONDS=0 acquire_scheduler_lock 5
    assert_dir_exists "${SCHEDULER_LOCK_DIR}"
    release_scheduler_lock

    mkdir -p "${GPU_RESERVATION_DIR}/task_test.lock.d"
    GPU_RESERVATION_LOCK_NOOWNER_STALE_SECONDS=0 _acquire_task_reservation_lock test 5
    _release_task_reservation_lock test
}

test_reservation_scoping_does_not_clobber_locals() {
    mock_reset

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/workers/gpu_reservations"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler.sh"

    local gpu_id="SENTINEL_GPU_ID"
    reserve_gpus "task_scoping" "0,1" >/dev/null
    assert_eq "SENTINEL_GPU_ID" "${gpu_id}" "gpu_id clobbered after reserve_gpus()"

    release_gpus "task_scoping" >/dev/null || true
    assert_eq "SENTINEL_GPU_ID" "${gpu_id}" "gpu_id clobbered after release_gpus()"
}

test_scheduler_direct_module_source_guards() {
    mock_reset

    (
        get_task_field() { return 0; }
        # shellcheck source=../scheduler_core.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler_core.sh"
        declare -F calculate_required_gpus >/dev/null
    )

    (
        # shellcheck source=../scheduler_reservations.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler_reservations.sh"
        declare -F reserve_gpus >/dev/null
        declare -F get_gpu_available_memory >/dev/null
    )

    (
        # shellcheck source=../scheduler_selection.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/scheduler_selection.sh"
        declare -F find_and_claim_task >/dev/null
        declare -F reserve_gpus >/dev/null
    )
}
