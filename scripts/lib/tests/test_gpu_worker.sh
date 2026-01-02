#!/usr/bin/env bash

test_should_shutdown_checks_global_and_worker_flags() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    run should_shutdown "0" "${out}"
    assert_rc "1" "${RUN_RC}" "no flags -> no shutdown"

    touch "${out}/workers/SHUTDOWN"
    run should_shutdown "0" "${out}"
    assert_rc "0" "${RUN_RC}" "global shutdown"

    rm -f "${out}/workers/SHUTDOWN"
    touch "${out}/workers/gpu_0.shutdown"
    run should_shutdown "0" "${out}"
    assert_rc "0" "${RUN_RC}" "worker shutdown"
}

test_gpu_worker_sets_waiting_deps_when_only_pending_tasks() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers" "${out}/logs/tasks"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    # Create a pending task so pending>0 and ready==0.
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:["dep"], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/t1.task"

    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo ""; return 1; }

    gpu_worker "0" "${out}" &
    local pid=$!

    local saw_waiting="false"
    local i
    for i in $(seq 1 200); do
        if [[ -f "${out}/workers/gpu_0.status" ]] && [[ "$(cat "${out}/workers/gpu_0.status")" == "waiting_deps" ]]; then
            saw_waiting="true"
            break
        fi
    done

    touch "${out}/workers/SHUTDOWN"
    wait "${pid}" || true

    assert_eq "true" "${saw_waiting}" "waiting_deps status observed"
}

test_gpu_worker_exits_on_poison_context_log() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers" "${out}/logs/tasks"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local task_id="poison"
    local task_file="${QUEUE_DIR}/running/${task_id}.task"
    jq -n '{task_id:"poison", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:"x", completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:"0", dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo "${task_file}"; }

    execute_task() {
        local task_path="$1"
        local t_id
        t_id="$(jq -r '.task_id' "${task_path}")"
        echo "device-side assert triggered" > "${out}/logs/tasks/${t_id}.log"
        return 1
    }

    fail_task() { return 0; }
    release_task_gpus() { return 0; }

    local rc=0
    if ( gpu_worker "0" "${out}" ); then
        rc=0
    else
        rc=$?
    fi
    assert_rc "1" "${rc}" "poison context exits 1"
}

test_gpu_worker_breaks_on_shutdown_signal_and_updates_info() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    touch "${out}/workers/SHUTDOWN"
    gpu_worker "0" "${out}"

    assert_file_exists "${out}/workers/gpu_0.info" "worker info written"
    assert_eq "stopped" "$(cat "${out}/workers/gpu_0.status")" "status stopped"
}

test_gpu_worker_updates_heartbeat_and_retries_when_gpu_mem_unavailable() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    WORKER_HEARTBEAT_INTERVAL=0
    local calls=0
    get_gpu_available_memory() {
        calls=$((calls + 1))
        if [[ ${calls} -ge 1 ]]; then
            touch "${out}/workers/SHUTDOWN"
        fi
        echo ""
    }

    gpu_worker "0" "${out}"
    assert_file_exists "${out}/workers/gpu_0.heartbeat" "heartbeat touched"
}

test_gpu_worker_shuts_down_when_queue_empty_and_no_task() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo ""; }
    is_queue_empty() { return 0; }

    gpu_worker "0" "${out}"
    assert_eq "stopped" "$(cat "${out}/workers/gpu_0.status")" "worker stopped"
}

test_gpu_worker_sets_idle_when_queue_not_empty_but_no_task_available() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    WORKER_IDLE_SLEEP=0
    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo ""; }
    is_queue_empty() { return 1; }
    count_tasks() {
        case "$1" in
            pending) echo "0" ;;
            ready) echo "1" ;;
            running) echo "0" ;;
            *) echo "0" ;;
        esac
    }

    _sleep() { touch "${out}/workers/SHUTDOWN"; }

    gpu_worker "0" "${out}"
    assert_eq "stopped" "$(cat "${out}/workers/gpu_0.status")" "worker stops after shutdown"
}

test_gpu_worker_success_path_with_oom_precheck_and_cleanup_hooks() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers" "${out}/logs/tasks"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local task_id="ok1"
    local task_file="${QUEUE_DIR}/running/${task_id}.task"
    jq -n '{task_id:"ok1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:"x", completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo "${task_file}"; }

    check_oom_safe() { return 1; }
    get_oom_risk_level() { echo "high"; }
    purge_multi_gpu_memory() { return 0; }
    release_task_gpus() { return 0; }
    complete_task() { return 0; }
    execute_task() { return 0; }

    start_heartbeat_thread() { ( : ) & }
    _sleep() {
        if [[ "${1:-}" == "1" ]]; then
            touch "${out}/workers/SHUTDOWN"
        fi
        return 0
    }

    gpu_worker "0" "${out}"
}

test_gpu_worker_survives_risk_level_probe_failure() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers" "${out}/logs/tasks"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local task_id="risk1"
    local task_file="${QUEUE_DIR}/running/${task_id}.task"
    jq -n '{task_id:"risk1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:"x", completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    get_gpu_available_memory() { echo "100"; }
    find_and_claim_task() { echo "${task_file}"; }

    check_oom_safe() { return 1; }
    get_oom_risk_level() { return 1; }
    purge_multi_gpu_memory() { return 0; }
    release_task_gpus() { return 0; }
    complete_task() { return 0; }
    execute_task() { return 0; }

    start_heartbeat_thread() { ( : ) & }
    _sleep() {
        if [[ "${1:-}" == "1" ]]; then
            touch "${out}/workers/SHUTDOWN"
        fi
        return 0
    }

    gpu_worker "0" "${out}"
    assert_eq "stopped" "$(cat "${out}/workers/gpu_0.status")" "worker stops cleanly after risk probe failure"
}

test_wait_for_workers_returns_zero_when_all_workers_succeed() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    ( exit 0 ) &
    local pid1=$!
    ( exit 0 ) &
    local pid2=$!

    local rc=0
    if wait_for_workers "${pid1}" "${pid2}"; then
        rc=0
    else
        rc=$?
    fi
    assert_rc "0" "${rc}" "all workers succeeding returns 0"
}

test_gpu_worker_failure_timeout_and_oom_branches_and_failure_threshold() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers" "${out}/logs/tasks"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    WORKER_MAX_FAILURES=1

    get_gpu_available_memory() { echo "100"; }
    local mode="${MODE:-timeout}"

    local task_id="fail1"
    local task_file="${QUEUE_DIR}/running/${task_id}.task"
    jq -n '{task_id:"fail1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:"x", completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:"0", dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    find_and_claim_task() { echo "${task_file}"; }

    purge_multi_gpu_memory() { return 0; }
    handle_oom_task() { return 0; }
    release_task_gpus() { return 0; }
    maybe_retry_task() { return 0; }
    fail_task() { return 0; }

    start_heartbeat_thread() { ( : ) & }

    execute_task() {
        if [[ "${MODE}" == "timeout" ]]; then
            return 124
        fi
        echo "CUDA out of memory" > "${out}/logs/tasks/${task_id}.log"
        return 2
    }

    MODE="timeout"
    gpu_worker "0" "${out}"

    # OOM path (uses task log).
    rm -f "${out}/workers/SHUTDOWN"
    MODE="oom"
    gpu_worker "0" "${out}"
}

test_launch_wait_monitor_and_summary_branches() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    gpu_worker() { return 0; }

    GPU_ID_LIST="2,3"
    launch_worker_pool "${out}" "1" >/dev/null
    unset GPU_ID_LIST
    launch_worker_pool "${out}" "1" >/dev/null

    ( exit 0 ) & local p1=$!
    ( exit 1 ) & local p2=$!
    local rc=0
    wait_for_workers "${p1}" "${p2}" || rc=$?
    assert_rc "1" "${rc}" "wait_for_workers returns non-zero when any worker fails"

    # monitor_workers: cover GPU_ID_LIST and non-list branches, dead and stuck workers.
    mkdir -p "${out}/workers"
    echo "111" > "${out}/workers/gpu_0.pid"
    echo "222" > "${out}/workers/gpu_2.pid"
    : > "${out}/workers/gpu_2.heartbeat"
    : > "${out}/workers/gpu_2.status"

    local iter=0
    is_queue_empty() { iter=$((iter + 1)); [[ ${iter} -ge 2 ]]; }
    is_queue_complete() { return 0; }
    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo "0"; }
    _cmd_kill() {
        if [[ "${1:-}" == "-0" ]]; then
            [[ "${2:-}" == "111" ]] && return 1
            return 0
        fi
        return 0
    }
    reclaim_orphaned_tasks() { return 0; }
    gpu_worker() { return 0; }

    GPU_ID_LIST="0,1,2"
    monitor_workers "${out}" "3" "0" "1"
    unset GPU_ID_LIST
    iter=0
    monitor_workers "${out}" "1" "0" "1"

    # get_worker_summary GPU_ID_LIST and default enumeration; alive branch.
    echo "idle" > "${out}/workers/gpu_0.status"
    echo "333" > "${out}/workers/gpu_0.pid"
    _cmd_kill() { [[ "${1:-}" == "-0" ]] && return 0; return 0; }
    GPU_ID_LIST="0"
    get_worker_summary "${out}" "1" >/dev/null
    unset GPU_ID_LIST
    get_worker_summary "${out}" "1" >/dev/null
}

test_worker_pool_sanitizes_invalid_numeric_args() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    gpu_worker() { return 0; }

    launch_worker_pool "${out}" "nope" >/dev/null
    assert_eq "1" "$(ls "${out}/workers"/gpu_*.pid 2>/dev/null | wc -l | tr -d ' ')" "invalid num_gpus defaults to 1"

    is_queue_empty() { return 0; }
    monitor_workers "${out}" "bad" "nope" "nope"
    assert_file_exists "${out}/workers/SHUTDOWN" "monitor signals shutdown on empty queue"

    run get_worker_summary "${out}" "bad"
    assert_match 'GPU 0' "${RUN_OUT}" "summary uses default gpu id"
}

test_gpu_worker_sanitizes_invalid_worker_config() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    WORKER_HEARTBEAT_INTERVAL="nope"
    WORKER_IDLE_SLEEP="nope"
    WORKER_MAX_FAILURES="nope"
    should_shutdown() { return 0; }

    gpu_worker "0" "${out}"
    assert_eq "stopped" "$(cat "${out}/workers/gpu_0.status")" "worker exits cleanly with invalid config"
}

test_monitor_workers_stops_when_queue_empty_even_with_failures() {
    mock_reset
    # shellcheck source=../gpu_worker.sh
    source "${TEST_ROOT}/scripts/lib/gpu_worker.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/workers"

    is_queue_empty() { return 0; }
    is_queue_complete() { return 1; }

    monitor_workers "${out}" "1" "0" "1"
    assert_file_exists "${out}/workers/SHUTDOWN" "monitor signals shutdown when queue is empty"
}
