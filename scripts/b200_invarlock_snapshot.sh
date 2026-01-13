#!/usr/bin/env bash
# Snapshot diagnostics for an InvarLock B200 validation run.
# Usage:
#   bash scripts/b200_invarlock_snapshot.sh /path/to/invarlock_validation_b200_YYYYmmdd_HHMMSS
#   OUT=/path/to/output bash scripts/b200_invarlock_snapshot.sh

set -uo pipefail

OUT_DIR="${1:-${OUT:-${OUTPUT_DIR:-}}}"
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="$(ls -1dt invarlock_validation_b200_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${OUT_DIR}" || ! -d "${OUT_DIR}" ]]; then
    echo "ERROR: output dir not found." >&2
    echo "Usage: $0 /path/to/OUTPUT_DIR   (or set OUT/OUTPUT_DIR)" >&2
    exit 2
fi

abs_out="${OUT_DIR}"
abs_out="$(readlink -f "${OUT_DIR}" 2>/dev/null || realpath "${OUT_DIR}" 2>/dev/null || echo "${OUT_DIR}")"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

mtime_epoch() {
    local path="$1"
    stat -c %Y "${path}" 2>/dev/null || stat -f %m "${path}" 2>/dev/null || echo ""
}

count_tasks_dir() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -maxdepth 1 -name '*.task' -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

echo "=== InvarLock Snapshot ==="
echo "time: $(date -Is)"
echo "host: $(hostname 2>/dev/null || true)"
echo "user: $(whoami 2>/dev/null || true)"
echo "pwd:  $(pwd)"
echo "out:  ${abs_out}"
echo

echo "== Env =="
echo "HOME=${HOME:-}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "GPU_ID_LIST=${GPU_ID_LIST:-}  NUM_GPUS=${NUM_GPUS:-}"
echo "HF_TOKEN_set=$([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo no)"
home="${HOME:-}"
effective_hf_home="${HF_HOME:-}"
if [[ -z "${effective_hf_home}" && -n "${home}" ]]; then
    effective_hf_home="${home}/.cache/huggingface"
fi
effective_hf_hub_cache="${HF_HUB_CACHE:-${effective_hf_home}/hub}"
effective_hf_datasets_cache="${HF_DATASETS_CACHE:-${effective_hf_home}/datasets}"
effective_transformers_cache="${TRANSFORMERS_CACHE:-${effective_hf_home}/transformers}"

echo "HF_HOME=${HF_HOME:-}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE:-}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-}"
echo "HF_HOME_effective=${effective_hf_home}"
echo "HF_HUB_CACHE_effective=${effective_hf_hub_cache}"
echo "HF_DATASETS_CACHE_effective=${effective_hf_datasets_cache}"
echo "TRANSFORMERS_CACHE_effective=${effective_transformers_cache}"
echo

echo "== Disk =="
df -h "${OUT_DIR}" 2>/dev/null | sed -n '1,2p' || true
du -sh "${OUT_DIR}" 2>/dev/null || true
if [[ -n "${effective_hf_home}" ]]; then
    echo "-- hf cache filesystem --"
    df -h "${effective_hf_home}" 2>/dev/null \
        || df -h "$(dirname "${effective_hf_home}")" 2>/dev/null \
        || true
fi
echo

echo "== Locks =="
for lock in "${OUT_DIR}/queue/queue.lock.d" "${OUT_DIR}/queue/scheduler.lock.d"; do
    if [[ -d "${lock}" ]]; then
        owner="$(cat "${lock}/owner" 2>/dev/null || echo "")"
        age=""
        mt="$(mtime_epoch "${lock}/owner")"
        [[ -n "${mt}" ]] && age="$(( $(date +%s) - mt ))"
        echo "$(basename "${lock}") owner_pid=${owner:-unknown} age_s=${age:-unknown}"
    fi
done
echo

echo "== GPUs =="
if have_cmd nvidia-smi; then
    nvidia-smi -L 2>/dev/null || true
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,pstate,power.draw \
        --format=csv,noheader,nounits 2>/dev/null || true
    echo "-- compute apps --"
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${apps}" ]]; then
        printf '%s\n' "${apps}"
    else
        echo "(none)"
    fi
else
    echo "nvidia-smi: not found"
fi
echo

echo "== Queue =="
pending="$(count_tasks_dir "${OUT_DIR}/queue/pending")"
ready="$(count_tasks_dir "${OUT_DIR}/queue/ready")"
running="$(count_tasks_dir "${OUT_DIR}/queue/running")"
completed="$(count_tasks_dir "${OUT_DIR}/queue/completed")"
failed="$(count_tasks_dir "${OUT_DIR}/queue/failed")"
total=$((pending + ready + running + completed + failed))
printf '%-10s %s\n' "pending:" "${pending}"
printf '%-10s %s\n' "ready:" "${ready}"
printf '%-10s %s\n' "running:" "${running}"
printf '%-10s %s\n' "completed:" "${completed}"
printf '%-10s %s\n' "failed:" "${failed}"
printf '%-10s %s\n' "total:" "${total}"
echo

echo "== Running Tasks (up to 10) =="
if [[ -d "${OUT_DIR}/queue/running" ]]; then
    mapfile -t running_files < <(find "${OUT_DIR}/queue/running" -maxdepth 1 -name '*.task' -type f 2>/dev/null | sort | head -n 10)
    if [[ ${#running_files[@]} -eq 0 ]]; then
        echo "(none)"
    else
        if have_cmd jq; then
            for f in "${running_files[@]}"; do
                jq -r '[.task_id,.task_type,.model_name,(.assigned_gpus//""),(.started_at//"")] | @tsv' "${f}"
            done
        else
            printf '%s\n' "${running_files[@]}" | xargs -n 1 basename
        fi
    fi
else
    echo "(queue not initialized)"
fi
echo

echo "== Workers =="
shutdown_global="no"
[[ -f "${OUT_DIR}/workers/SHUTDOWN" ]] && shutdown_global="yes"
shutdown_count="$(find "${OUT_DIR}/workers" -maxdepth 1 -name 'gpu_*.shutdown' -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "shutdown_global=${shutdown_global} per_gpu_shutdown_files=${shutdown_count}"
mapfile -t status_files < <(find "${OUT_DIR}/workers" -maxdepth 1 -name 'gpu_*.status' -type f 2>/dev/null | sort)
if [[ ${#status_files[@]} -eq 0 ]]; then
    echo "(no worker status files)"
else
    for status_file in "${status_files[@]}"; do
        gpu="$(basename "${status_file}" .status)"
        pid_file="${OUT_DIR}/workers/${gpu}.pid"
        heartbeat_file="${OUT_DIR}/workers/${gpu}.heartbeat"
        status="$(cat "${status_file}" 2>/dev/null || echo "?")"
        pid="$(cat "${pid_file}" 2>/dev/null || echo "")"
        alive="no"
        [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && alive="yes"
        hb_age=""
        if [[ -f "${heartbeat_file}" ]]; then
            hb_mt="$(mtime_epoch "${heartbeat_file}")"
            [[ -n "${hb_mt}" ]] && hb_age="$(( $(date +%s) - hb_mt ))"
        fi
        echo "${gpu} alive=${alive} pid=${pid:-} hb_age_s=${hb_age:-unknown} status=${status}"
    done
fi
echo

echo "== main.log (tail 40) =="
tail -n 40 "${OUT_DIR}/logs/main.log" 2>/dev/null || echo "(no main.log)"
echo

echo "== Worker Logs (tail 30 each) =="
mapfile -t worker_logs < <(find "${OUT_DIR}/logs" -maxdepth 1 -name 'gpu_*.log' -type f 2>/dev/null | sort)
if [[ ${#worker_logs[@]} -eq 0 ]]; then
    echo "(no gpu_*.log files)"
else
    for f in "${worker_logs[@]}"; do
        echo "-- $(basename "${f}") --"
        tail -n 30 "${f}" 2>/dev/null || true
    done
fi
echo

echo "== Failed Tasks (up to 5) =="
if [[ -d "${OUT_DIR}/queue/failed" ]]; then
    mapfile -t failed_files < <(find "${OUT_DIR}/queue/failed" -maxdepth 1 -name '*.task' -type f 2>/dev/null | sort | head -n 5)
    if [[ ${#failed_files[@]} -eq 0 ]]; then
        echo "(none)"
    else
        for f in "${failed_files[@]}"; do
            if have_cmd jq; then
                tid="$(jq -r '.task_id' "${f}" 2>/dev/null || echo "$(basename "${f}" .task)")"
                ttype="$(jq -r '.task_type // ""' "${f}" 2>/dev/null || echo "")"
                mid="$(jq -r '.model_id // ""' "${f}" 2>/dev/null || echo "")"
                err="$(jq -r '.error_msg // ""' "${f}" 2>/dev/null || echo "")"
                echo "-- ${tid} | ${ttype} | ${mid} --"
                echo "  error: ${err}"
                task_log="${OUT_DIR}/logs/tasks/${tid}.log"
                if [[ -f "${task_log}" ]]; then
                    tail -n 30 "${task_log}" 2>/dev/null | sed 's/^/  /'
                else
                    echo "  (no task log at ${task_log})"
                fi
            else
                echo "-- $(basename "${f}") --"
            fi
        done
    fi
else
    echo "(queue not initialized)"
fi
