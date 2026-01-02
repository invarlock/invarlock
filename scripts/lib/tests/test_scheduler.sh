#!/usr/bin/env bash

test_get_gpu_available_memory_uses_mock_nvidia_smi() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    mock_nvidia_smi_set_mem_free_mib 0 20480
    mock_nvidia_smi_set_mem_total_mib 0 184320
    mock_nvidia_smi_set_pids 0 ""  # idle

    assert_eq "20" "$(get_gpu_available_memory 0)" "free memory in GB"
    is_gpu_idle 0
}

test_is_reservation_valid_ttl_and_liveness() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _now_epoch() { echo "100"; }

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/${task_id}.task"

    local meta_file="${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    printf "timestamp=0\nowner_pid=123\ngpu_list=0\n" > "${meta_file}"
    GPU_RESERVATION_TTL=60
    _pid_is_alive() { return 0; }
    ! _is_reservation_valid "${task_id}"

    printf "timestamp=90\nowner_pid=123\ngpu_list=0\n" > "${meta_file}"
    _pid_is_alive() { return 1; }
    ! _is_reservation_valid "${task_id}"

    _pid_is_alive() { return 0; }
    _is_reservation_valid "${task_id}"

    mv "${QUEUE_DIR}/ready/${task_id}.task" "${QUEUE_DIR}/running/${task_id}.task"
    _pid_is_alive() { return 1; }
    _is_reservation_valid "${task_id}"
}

test_is_gpu_available_cleans_stale_reservation() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _now_epoch() { echo "100"; }
    GPU_RESERVATION_TTL=60

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/${task_id}.task"

    echo "${task_id}" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    printf "timestamp=0\nowner_pid=123\ngpu_list=0\n" > "${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    echo "0" > "${GPU_RESERVATION_DIR}/task_${task_id}.gpus"

    _pid_is_alive() { return 0; }
    is_gpu_available 0
    assert_rc "0" "$?" "stale reservation cleaned"
    [[ ! -f "${GPU_RESERVATION_DIR}/gpu_0.lock" ]] || t_fail "expected stale gpu lock removed"

    echo "${task_id}" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    printf "timestamp=90\nowner_pid=123\ngpu_list=0\n" > "${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    echo "0" > "${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
    _pid_is_alive() { return 0; }
    ! is_gpu_available 0
}

test_find_and_claim_task_releases_reservation_when_claim_fails() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", model_size_gb:10, required_gpus:1, retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/${task_id}.task"

    find_best_task() { echo "t1"; }
    acquire_scheduler_lock() { return 0; }
    release_scheduler_lock() { return 0; }
    reserve_gpus() { echo "reserve $*" >> "${TEST_TMPDIR}/calls"; return 0; }
    release_gpus() { echo "release $*" >> "${TEST_TMPDIR}/calls"; return 0; }
    claim_task() { return 1; }

    run find_and_claim_task "999" "0"
    assert_rc "1" "${RUN_RC}" "claim failure returns 1"
    assert_match 'release t1' "$(cat "${TEST_TMPDIR}/calls")" "reservation released"
}

test_list_gpu_ids_and_cache_helpers_cover_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    # GPU_ID_LIST takes precedence and is parsed into a newline list.
    export GPU_ID_LIST="2,4,,7"
    assert_eq $'2\n4\n7' "$(list_gpu_ids)" "GPU_ID_LIST parsing"

    # NUM_GPUS fallback: invalid input defaults to 8; values < 1 clamp to 1.
    unset GPU_ID_LIST
    export NUM_GPUS="not-a-number"
    assert_eq "8" "$(list_gpu_ids | wc -l | tr -d ' ')" "invalid NUM_GPUS defaults to 8"
    export NUM_GPUS="0"
    assert_eq "0" "$(list_gpu_ids)" "NUM_GPUS < 1 clamps to 1"

    # Cache helpers use GPU_RESERVATION_DIR when set.
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"
    assert_match '\.gpu_cache_0' "$(_gpu_cache_file 0)" "gpu cache file path"

    # Cache read expires when age > TTL.
    GPU_CACHE_TTL=5
    printf "free_mem=123\nis_idle=true\n" > "${GPU_RESERVATION_DIR}/.gpu_cache_0"
    _file_mtime_epoch() { echo "0"; }
    _now_epoch() { echo "100"; }
    if _read_gpu_cache 0 "free_mem" >/dev/null; then
        t_fail "expected cache expired"
    fi
}

test_gpu_cache_ttl_sanitizes_invalid_value() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"
    printf "free_mem=77\n" > "${GPU_RESERVATION_DIR}/.gpu_cache_0"

    GPU_CACHE_TTL="nope"
    _file_mtime_epoch() { echo "0"; }
    _now_epoch() { echo "1"; }

    assert_eq "77" "$(_read_gpu_cache 0 "free_mem")" "invalid GPU_CACHE_TTL defaults to 5s"
}

test_task_reservation_lock_sanitizes_invalid_timeout() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"
    GPU_RESERVATION_LOCK_TIMEOUT="bad"

    run _acquire_task_reservation_lock "tlock"
    assert_rc "0" "${RUN_RC}" "invalid lock timeout falls back to default"
    _release_task_reservation_lock "tlock"
}

test_is_reservation_valid_sanitizes_invalid_ttl() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/${task_id}.task"
    printf "timestamp=99\nowner_pid=123\ngpu_list=0\n" > "${GPU_RESERVATION_DIR}/task_${task_id}.meta"

    GPU_RESERVATION_TTL="bad"
    _now_epoch() { echo "100"; }
    _pid_is_alive() { return 0; }

    _is_reservation_valid "${task_id}"
}

test_refresh_all_gpu_cache_and_scheduler_lock_file_empty_fallback_cover_lines() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    local calls=""
    list_gpu_ids() { echo "0 1"; }
    _refresh_gpu_cache() { calls+="$1,"; }

    refresh_all_gpu_cache
    assert_eq "0,1," "${calls}" "refresh_all_gpu_cache iterates all GPUs"

    unset QUEUE_DIR
    unset GPU_RESERVATION_DIR
    assert_eq "" "$(scheduler_lock_file)" "scheduler_lock_file returns empty when dirs unset"
}

test_scheduler_lock_and_task_reservation_lock_sleep_on_contention() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    local out="${TEST_TMPDIR}/out"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local lock_dir="${QUEUE_DIR}/scheduler.lock.d"
    mkdir -p "${lock_dir}"
    echo "123" > "${lock_dir}/owner"

    _pid_is_alive() { return 0; }
    local now_state="${TEST_TMPDIR}/now_epoch.calls"
    : >"${now_state}"
    _now_epoch() {
        # NOTE: acquire_* lock helpers capture _now_epoch via command substitution (subshell),
        # so persist the call counter via a file for deterministic timeouts.
        local n=0
        n="$(cat "${now_state}" 2>/dev/null || echo "0")"
        n=$((n + 1))
        printf '%s' "${n}" >"${now_state}"
        case "${n}" in
            1|2) echo "0" ;;
            *) echo "1" ;;
        esac
    }

    local slept=0
    _sleep() { slept=$((slept + 1)); }
    run acquire_scheduler_lock 1
    assert_rc "1" "${RUN_RC}" "acquire_scheduler_lock times out under contention"
    assert_eq "1" "${slept}" "scheduler lock sleeps before retry"

    export GPU_RESERVATION_DIR="${out}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"
    local task_lock_dir="${GPU_RESERVATION_DIR}/task_t.lock.d"
    mkdir -p "${task_lock_dir}"
    echo "123" > "${task_lock_dir}/owner"

    : >"${now_state}"
    slept=0
    GPU_RESERVATION_LOCK_TIMEOUT=1
    run _acquire_task_reservation_lock "t" 1
    assert_rc "1" "${RUN_RC}" "task reservation lock times out under contention"
    assert_eq "1" "${slept}" "task reservation lock sleeps before retry"
}

test_scheduler_gpu_memory_and_process_helpers_cover_lines() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    purge_multi_gpu_memory "0,1"

    mock_nvidia_smi_set_pids 0 $'123\n456\n'
    local kill_calls=""
    _cmd_kill() { kill_calls+="$1 $2;"; return 0; }
    _sleep() { :; }
    kill_gpu_processes "0"
    assert_match '-TERM 123' "${kill_calls}" "TERM sent"
    assert_match '-KILL 456' "${kill_calls}" "KILL sent"

    mock_nvidia_smi_set_mem_total_mib 0 184320
    fixture_write "nvidia-smi/utilization.0" "15"
    assert_eq "180" "$(get_gpu_total_memory 0)" "total memory in GB"
    assert_eq "15" "$(get_gpu_utilization 0)" "utilization query"

    list_gpu_ids() { echo "0 1 2"; }
    is_gpu_usable() { [[ "$1" != "2" ]]; }
    assert_eq "2" "$(count_available_gpus)" "counts usable GPUs"

    local released=""
    release_gpus() { released+="${1},"; }
    release_task_gpus "t1"
    assert_match 't1,' "${released}" "release_task_gpus delegates"
}

test_reserve_gpus_cleans_stale_existing_reservation() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    local out="${TEST_TMPDIR}/out"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}
    export GPU_RESERVATION_DIR="${out}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    echo "oldtask" > "${GPU_RESERVATION_DIR}/gpu_0.lock"

    local cleaned=""
    _cleanup_task_reservation() { cleaned+="${1},"; }
    _is_reservation_valid() { return 1; }
    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _now_epoch() { echo "100"; }

    reserve_gpus "newtask" "0"
    assert_match 'oldtask,' "${cleaned}" "cleans stale existing task reservation"
}

test_release_gpus_cleans_locks_even_when_task_gpu_list_file_is_missing() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    echo "t1" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    printf "timestamp=100\nowner_pid=123\ngpu_list=0\n" > "${GPU_RESERVATION_DIR}/task_t1.meta"
    rm -f "${GPU_RESERVATION_DIR}/task_t1.gpus"

    release_gpus "t1"
    [[ ! -f "${GPU_RESERVATION_DIR}/gpu_0.lock" ]] || t_fail "expected gpu lock removed for released task"
    [[ ! -f "${GPU_RESERVATION_DIR}/task_t1.meta" ]] || t_fail "expected metadata removed for released task"
}

test_print_scheduling_report_outputs_sections() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    local out="${TEST_TMPDIR}/out"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    export GPU_ID_LIST="0"
    mock_nvidia_smi_set_mem_free_mib 0 20480
    mock_nvidia_smi_set_mem_total_mib 0 184320
    fixture_write "nvidia-smi/utilization.0" "15"

    jq -n '{task_id:"r1", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"running", model_size_gb:14, required_gpus:1, retries:0, max_retries:3, created_at:"x", started_at:"x", completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/running/r1.task"
    jq -n '{task_id:"q1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", model_size_gb:14, required_gpus:1, retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/q1.task"

    local report
    report="$(print_scheduling_report)"
    assert_match 'SCHEDULING REPORT' "${report}" "header"
    assert_match 'GPU 0: 20/180 GB free, 15% utilization' "${report}" "gpu line"
    assert_match 'RUNNING TASKS' "${report}" "running section"
    assert_match 'TOP READY TASKS' "${report}" "ready section"
}

test_gpu_cache_file_returns_empty_when_reservation_dir_is_empty() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    GPU_RESERVATION_DIR=""
    assert_eq "" "$(_gpu_cache_file 0)" "no cache file when reservation dir is empty"
}

test_refresh_gpu_cache_and_memory_queries_cover_cache_hit_and_miss_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # Cache hit path in get_gpu_available_memory.
    GPU_CACHE_TTL=999
    _file_mtime_epoch() { echo "0"; }
    _now_epoch() { echo "0"; }
    printf "free_mem=77\nis_idle=true\n" > "${GPU_RESERVATION_DIR}/.gpu_cache_0"
    assert_eq "77" "$(get_gpu_available_memory 0)" "cache hit returns cached value"

    # Cache miss: non-empty nvidia-smi output + non-empty pid listing updates cache.
    rm -f "${GPU_RESERVATION_DIR}/.gpu_cache_0"
    mock_nvidia_smi_set_mem_free_mib 0 20480
    mock_nvidia_smi_set_pids 0 $'123\n'  # non-idle
    assert_eq "20" "$(get_gpu_available_memory 0)" "free memory converts MiB->GB"
    grep -q '^is_idle=false' "${GPU_RESERVATION_DIR}/.gpu_cache_0" || t_fail "expected is_idle cached false"

    # Free memory query returning empty triggers explicit error-path.
    rm -f "${GPU_RESERVATION_DIR}/.gpu_cache_0"
    fixture_write "nvidia-smi/memory_free.0" ""
    if get_gpu_available_memory 0 >/dev/null; then
        t_fail "expected get_gpu_available_memory to fail when nvidia-smi output is empty"
    fi

    # is_gpu_idle cache miss with non-empty pid listing.
    rm -f "${GPU_RESERVATION_DIR}/.gpu_cache_0"
    fixture_write "nvidia-smi/compute_pids.0" $'42\n'
    fixture_write "nvidia-smi/memory_free.0" "10240"
    is_gpu_idle 0 && t_fail "expected non-idle when pids present"

    # get_gpu_total_memory uses default when output is empty.
    fixture_write "nvidia-smi/memory_total.0" ""
    if [[ "$(get_gpu_total_memory 0)" != "180" ]]; then
        t_fail "expected default total memory when nvidia-smi output is empty"
    fi

    # Refresh path exercises raw_output non-empty and free_mib conversion.
    fixture_write "nvidia-smi/memory_free.0" "20480"
    fixture_write "nvidia-smi/compute_pids.0" $'999\n'
    _refresh_gpu_cache 0
}

test_get_gpu_available_memory_returns_zero_when_nvidia_smi_command_fails() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    GPU_RESERVATION_DIR=""
    _cmd_nvidia_smi() { return 1; }

    run get_gpu_available_memory 0
    assert_rc "1" "${RUN_RC}" "nvidia-smi failure returns non-zero"
    assert_eq "0" "${RUN_OUT}" "fallback memory is 0"
}

test_is_reservation_valid_returns_stale_when_meta_missing_timestamp() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    local task_id="t_missing_ts"
    echo '{}' > "${QUEUE_DIR}/ready/${task_id}.task"
    printf "owner_pid=123\ngpu_list=0\n" > "${GPU_RESERVATION_DIR}/task_${task_id}.meta"

    run _is_reservation_valid "${task_id}"
    assert_rc "1" "${RUN_RC}" "missing timestamp makes reservation stale"
}

test_get_available_gpus_returns_empty_when_not_enough_gpus_available() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    list_gpu_ids() { echo "0 1"; }
    is_gpu_usable() { return 0; }

    if get_available_gpus 3 false "" 0 >/dev/null; then
        t_fail "expected get_available_gpus to fail when fewer GPUs are available than requested"
    fi
}

test_scheduler_lock_file_and_acquire_lock_timeout_and_stale_owner_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    unset QUEUE_DIR
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    local lock_dir
    lock_dir="${GPU_RESERVATION_DIR}/scheduler.lock.d"

    # Timeout branch prints owner_pid when present.
    mkdir -p "${lock_dir}"
    echo "123" > "${lock_dir}/owner"
    _now_epoch() { echo "0"; }
    if acquire_scheduler_lock 0; then
        t_fail "expected acquire_scheduler_lock timeout when lock already held"
    fi
    rm -rf "${lock_dir}"

    # Stale owner PID branch removes lock and succeeds on retry.
    mkdir -p "${lock_dir}"
    echo "999" > "${lock_dir}/owner"
    _pid_is_alive() { return 1; }
    _now_epoch() { echo "0"; }
    acquire_scheduler_lock 1
    release_scheduler_lock
}

test_should_use_adaptive_gpus_counts_single_gpu_tasks_and_adapts_only_when_safe() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}

    # One single-GPU task waiting: do not adapt.
    jq -n '{task_id:"single", task_type:"T", model_id:"m", model_name:"n", status:"ready", required_gpus:1, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/single.task"
    should_use_adaptive_gpus 2 4 2 && t_fail "expected no adaptation when single-GPU tasks are waiting"

    # Only multi-GPU tasks waiting: adaptation is allowed.
    rm -f "${QUEUE_DIR}/ready"/*.task
    jq -n '{task_id:"multi", task_type:"T", model_id:"m", model_name:"n", status:"ready", required_gpus:2, dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/multi.task"
    should_use_adaptive_gpus 2 4 2 || t_fail "expected adaptation when no single-GPU tasks are waiting"
}

test_required_gpu_category_case_arms_cover_legacy_classifier() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    assert_eq "1" "$(get_required_gpus_from_category 70)" "70B category"
    assert_eq "1" "$(get_required_gpus_from_category moe)" "moe category"
    assert_eq "1" "$(get_required_gpus_from_category 40)" "40B category"
    assert_eq "1" "$(get_required_gpus_from_category other)" "default category"
}

test_task_reservation_lock_timeout_stale_owner_and_ownerless_grace_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    local task_id="t1"
    local lock_dir="${GPU_RESERVATION_DIR}/task_${task_id}.lock.d"

    # Timeout when lock dir already exists.
    mkdir -p "${lock_dir}"
    _now_epoch() { echo "0"; }
    if _acquire_task_reservation_lock "${task_id}" 0; then
        t_fail "expected task reservation lock timeout"
    fi
    rm -rf "${lock_dir}"

    # Stale owner gets cleaned up and lock is acquired.
    mkdir -p "${lock_dir}"
    echo "999" > "${lock_dir}/owner"
    _pid_is_alive() { return 1; }
    _now_epoch() { echo "0"; }
    _acquire_task_reservation_lock "${task_id}" 2
    _release_task_reservation_lock "${task_id}"

    # Ownerless lock uses grace parsing + mtime staleness cleanup.
    mkdir -p "${lock_dir}"
    export GPU_RESERVATION_LOCK_NOOWNER_STALE_SECONDS="bad"
    _file_mtime_epoch() { echo "0"; }
    _now_epoch() { echo "100"; }
    _acquire_task_reservation_lock "${task_id}" 2
    _release_task_reservation_lock "${task_id}"
}

test_reserve_gpus_failure_branches_for_lock_and_existing_valid_reservations() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # Fails when task reservation lock can't be acquired.
    _acquire_task_reservation_lock() { return 1; }
    reserve_gpus "t1" "0" || true

    # Fails when the task already has a valid reservation elsewhere.
    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _is_reservation_valid() { return 0; }
    reserve_gpus "t1" "0" || true

    # Fails when a requested GPU is reserved by another valid task.
    local other="t_other"
    echo "${other}" > "${GPU_RESERVATION_DIR}/gpu_0.lock"
    _is_reservation_valid() {
        [[ "$1" == "t1" ]] && return 1
        [[ "$1" == "${other}" ]] && return 0
        return 1
    }
    reserve_gpus "t1" "0" || true
}

test_reserve_gpus_rejects_empty_gpu_list() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _is_reservation_valid() { return 1; }

    run reserve_gpus "t1" ""
    assert_rc "1" "${RUN_RC}" "empty gpu list should fail"
    [[ ! -f "${GPU_RESERVATION_DIR}/task_t1.meta" ]] || t_fail "expected no meta file for empty gpu list"
}

test_reserve_gpus_errors_when_gpu_lock_file_write_fails() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _is_reservation_valid() { return 1; }
    _now_epoch() { echo "100"; }

    printf() {
        if [[ "${2:-}" == "t1" ]]; then
            return 1
        fi
        command printf "$@"
    }

    run reserve_gpus "t1" "0"
    assert_rc "1" "${RUN_RC}" "lock write failure returns non-zero"
    [[ ! -f "${GPU_RESERVATION_DIR}/gpu_0.lock" ]] || t_fail "expected no gpu lock file after failure"
    [[ ! -f "${GPU_RESERVATION_DIR}/task_t1.meta" ]] || t_fail "expected metadata cleaned on failure"
}

test_reserve_gpus_errors_when_gpu_list_file_move_fails() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _acquire_task_reservation_lock() { return 0; }
    _release_task_reservation_lock() { return 0; }
    _is_reservation_valid() { return 1; }
    _now_epoch() { echo "100"; }

    mv() {
        local dst="${2:-}"
        if [[ "${1:-}" == "-f" ]]; then
            dst="${3:-}"
        fi
        if [[ "${dst}" == "${GPU_RESERVATION_DIR}/task_t1.gpus" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run reserve_gpus "t1" "0"
    assert_rc "1" "${RUN_RC}" "gpu list file move failure returns non-zero"
    [[ ! -f "${GPU_RESERVATION_DIR}/gpu_0.lock" ]] || t_fail "expected gpu lock cleaned on gpu list file error"
    [[ ! -f "${GPU_RESERVATION_DIR}/task_t1.meta" ]] || t_fail "expected metadata cleaned on gpu list file error"
}

test_is_reservation_valid_fallback_to_gpus_file_mtime_when_metadata_missing() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{ready,running,completed,pending,failed}
    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/gpu_res"
    mkdir -p "${GPU_RESERVATION_DIR}"

    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo "95"; }
    GPU_RESERVATION_TTL=60

    jq -n '{task_id:"t1", task_type:"T", model_id:"m", model_name:"n", status:"ready", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"
    echo "0" > "${GPU_RESERVATION_DIR}/task_t1.gpus"

    _is_reservation_valid "t1"
}

test_is_gpu_usable_rejects_reserved_low_memory_and_busy_idle_required_paths() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    is_gpu_available() { return 1; }
    if is_gpu_usable 0; then
        t_fail "expected unusable when GPU is reserved"
    fi

    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "0"; }
    export GPU_MIN_FREE_GB="10"
    if is_gpu_usable 0; then
        t_fail "expected unusable when free memory below threshold"
    fi

    get_gpu_available_memory() { echo "99"; }
    export GPU_REQUIRE_IDLE="true"
    is_gpu_idle() { return 1; }
    if is_gpu_usable 0; then
        t_fail "expected unusable when GPU is not idle and idle required"
    fi
}

test_get_available_gpus_selection_branches_must_include_spread_and_short_selection_error_path() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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

test_cleanup_stale_reservations_skips_valid_and_cleans_stale_branches() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    assert_eq "$(calculate_required_gpus 200)" "$(get_required_gpus 200)" "delegates to calculate_required_gpus"
}

test_is_gpu_usable_returns_zero_when_available_and_has_free_memory() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    GPU_MIN_FREE_GB=10
    GPU_REQUIRE_IDLE="false"
    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "999"; }

    is_gpu_usable 0
}

test_is_gpu_usable_sanitizes_invalid_min_free_setting() {
    mock_reset
    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/lib/scheduler.sh"

    GPU_MIN_FREE_GB="nope"
    GPU_REQUIRE_IDLE="true"
    is_gpu_available() { return 0; }
    get_gpu_available_memory() { echo "15"; }
    is_gpu_idle() { return 0; }

    is_gpu_usable 0
}
