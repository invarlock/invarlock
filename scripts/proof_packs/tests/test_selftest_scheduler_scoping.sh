#!/usr/bin/env bash

test_lock_ownerless_recovery() {
    mock_reset

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    export GPU_RESERVATION_DIR="${TEST_TMPDIR}/workers/gpu_reservations"
    mkdir -p "${GPU_RESERVATION_DIR}"

    # shellcheck source=../scheduler.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/scheduler.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/scheduler.sh"

    local gpu_id="SENTINEL_GPU_ID"
    reserve_gpus "task_scoping" "0,1" >/dev/null
    assert_eq "SENTINEL_GPU_ID" "${gpu_id}" "gpu_id clobbered after reserve_gpus()"

    release_gpus "task_scoping" >/dev/null || true
    assert_eq "SENTINEL_GPU_ID" "${gpu_id}" "gpu_id clobbered after release_gpus()"
}
