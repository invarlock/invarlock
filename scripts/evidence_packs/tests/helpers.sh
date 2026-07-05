#!/usr/bin/env bash

t_fail() {
    echo "FAIL: $*" >&2
    return 1
}

assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${expected}" != "${actual}" ]]; then
        t_fail "${msg} expected='${expected}' actual='${actual}'"
    fi
}

assert_ne() {
    local not_expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${not_expected}" == "${actual}" ]]; then
        t_fail "${msg} not_expected='${not_expected}' actual='${actual}'"
    fi
}

assert_match() {
    local pattern="$1"
    local actual="$2"
    local msg="${3:-}"
    if ! [[ "${actual}" =~ ${pattern} ]]; then
        t_fail "${msg} pattern='${pattern}' actual='${actual}'"
    fi
}

assert_file_exists() {
    local path="$1"
    local msg="${2:-}"
    [[ -f "${path}" ]] || t_fail "${msg} missing_file='${path}'"
}

assert_dir_exists() {
    local path="$1"
    local msg="${2:-}"
    [[ -d "${path}" ]] || t_fail "${msg} missing_dir='${path}'"
}

assert_rc() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${expected}" != "${actual}" ]]; then
        t_fail "${msg} expected_rc='${expected}' actual_rc='${actual}'"
    fi
}

run() {
    local out_file err_file
    out_file="$(mktemp "${TEST_TMPDIR}/run.out.XXXXXX")"
    err_file="$(mktemp "${TEST_TMPDIR}/run.err.XXXXXX")"
    RUN_RC=0
    if "$@" >"${out_file}" 2>"${err_file}"; then
        RUN_RC=0
    else
        RUN_RC=$?
    fi
    RUN_OUT="$(cat "${out_file}")"
    RUN_ERR="$(cat "${err_file}")"
    case "$-" in
        *x*)
            # When xtrace is enabled (coverage runs), forward captured stderr so the
            # test runner can attribute executed lines inside "$@" to the right files.
            cat "${err_file}" >&2 || true
            ;;
    esac
    rm -f "${out_file}" "${err_file}"
    return 0
}

fixture_write() {
    local rel_path="$1"
    local content="$2"
    local path="${TEST_TMPDIR}/fixtures/${rel_path}"
    mkdir -p "$(dirname "${path}")"
    printf "%s" "${content}" > "${path}"
}

fixture_append() {
    local rel_path="$1"
    local content="$2"
    local path="${TEST_TMPDIR}/fixtures/${rel_path}"
    mkdir -p "$(dirname "${path}")"
    printf "%s" "${content}" >> "${path}"
}

mock_reset() {
    rm -rf "${TEST_TMPDIR}/fixtures"
    mkdir -p "${TEST_TMPDIR}/fixtures"
}

queue_fixture_dirs() {
    local out_dir="${1:-${TEST_TMPDIR}/out}"
    export QUEUE_DIR="${out_dir}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}
}

gpu_reservation_fixture_dir() {
    export GPU_RESERVATION_DIR="${1:-${TEST_TMPDIR}/gpu_res}"
    mkdir -p "${GPU_RESERVATION_DIR}"
}

write_queue_task() {
    local state="$1"
    local task_id="$2"
    local task_type="${3:-SETUP_BASELINE}"
    local model_name="${4:-n}"
    local extra_json="${5:-}"
    [[ -n "${extra_json}" ]] || extra_json="{}"

    mkdir -p "${QUEUE_DIR}/${state}"
    jq -n \
        --arg id "${task_id}" \
        --arg state "${state}" \
        --arg type "${task_type}" \
        --arg name "${model_name}" \
        --argjson extra "${extra_json}" \
        '{
            task_id: $id,
            task_type: $type,
            model_id: "m",
            model_name: $name,
            status: $state,
            retries: 0,
            max_retries: 3,
            created_at: "x",
            started_at: null,
            completed_at: null,
            error_msg: null,
            assigned_gpus: null,
            dependencies: [],
            params: {},
            priority: 50
        } + $extra' > "${QUEUE_DIR}/${state}/${task_id}.task"
}

write_gpu_reservation() {
    local task_id="$1"
    local timestamp="${2:-0}"
    local owner_pid="${3:-123}"
    local gpu_list="${4:-0}"

    mkdir -p "${GPU_RESERVATION_DIR}"
    printf "timestamp=%s\nowner_pid=%s\ngpu_list=%s\n" \
        "${timestamp}" "${owner_pid}" "${gpu_list}" \
        > "${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    printf "%s\n" "${gpu_list}" > "${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
}

mock_install_bin_dir() {
    local bin_dir="${TEST_TMPDIR}/mocks/bin"
    local dispatcher="${bin_dir}/mock-command"
    mkdir -p "${bin_dir}"
    cat > "${dispatcher}" <<'EOF'
#!/bin/bash
set -euo pipefail

tool="$(basename "$0")"
fixtures="${TEST_TMPDIR:-}/fixtures"

get_flag_val() {
    local flag="$1"
    shift
    local -a args=("$@")
    local i
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "${flag}" ]]; then
            echo "${args[$((i+1))]:-}"
            return 0
        fi
    done
    return 1
}

case "${tool}" in
    date)
        fixtures_dir="${fixtures}/date"
        mkdir -p "${fixtures_dir}" 2>/dev/null || true
        for arg in "$@"; do
            case "${arg}" in
                -d|-j|-v*) exit 1 ;;
            esac
        done
        fmt="${*: -1}"
        case "${fmt}" in
            +%s)
                base="$(cat "${fixtures_dir}/epoch" 2>/dev/null || echo "1700000000")"
                counter_file="${fixtures_dir}/epoch.counter"
                if [[ -f "${counter_file}" ]]; then
                    cur="$(cat "${counter_file}" 2>/dev/null || echo "${base}")"
                else
                    cur="${base}"
                fi
                echo "${cur}"
                echo "$((cur + 1))" > "${counter_file}" 2>/dev/null || true
                ;;
            +%Y%m%d_%H%M%S) cat "${fixtures_dir}/compact" 2>/dev/null || echo "20250101_000000" ;;
            +%Y-%m-%d\ %H:%M:%S) cat "${fixtures_dir}/pretty" 2>/dev/null || echo "2025-01-01 00:00:00" ;;
            +%Y-%m-%dT%H:%M:%SZ) cat "${fixtures_dir}/iso" 2>/dev/null || echo "2025-01-01T00:00:00Z" ;;
            *) echo "2025-01-01T00:00:00Z" ;;
        esac
        ;;
    df)
        mode=""
        args=("$@")
        for arg in "${args[@]}"; do
            case "${arg}" in
                -P) mode="P" ;;
                -BG) mode="BG" ;;
            esac
        done
        path="${args[$(( ${#args[@]} - 1 ))]}"
        base="$(basename "${path}" 2>/dev/null || echo "")"
        if [[ -n "${TEST_TMPDIR:-}" && -n "${mode}" && -n "${base}" && -f "${fixtures}/df.${mode}.${base}" ]]; then
            cat "${fixtures}/df.${mode}.${base}"
        elif [[ -n "${TEST_TMPDIR:-}" && -f "${fixtures}/df.out" ]]; then
            cat "${fixtures}/df.out"
        else
            cat <<'DFEOF'
Filesystem  1G-blocks  Used Available Use% Mounted on
/dev/mock      1000    10       990   1% /
DFEOF
        fi
        ;;
    du)
        cat "${fixtures}/du.out" 2>/dev/null || printf '0\t.\n'
        ;;
    flock)
        exit 0
        ;;
    hostname)
        cat "${fixtures}/hostname" 2>/dev/null || echo "test-host"
        ;;
    invarlock)
        subcmd="${1:-}"
        shift || true
        mkdir -p "${fixtures}" 2>/dev/null || true
        if [[ -f "${fixtures}/invarlock.stub" ]]; then
            [[ -f "${fixtures}/invarlock.stdout" ]] && cat "${fixtures}/invarlock.stdout"
            [[ -f "${fixtures}/invarlock.stderr" ]] && cat "${fixtures}/invarlock.stderr" >&2
            if [[ -f "${fixtures}/invarlock.rc" ]]; then
                rc="$(cat "${fixtures}/invarlock.rc" 2>/dev/null || echo "0")"
                exit "${rc}"
            fi
            exit 0
        fi
        echo "invarlock ${subcmd} $*" >> "${fixtures}/invarlock.calls" 2>/dev/null || true
        if [[ -f "${fixtures}/invarlock.capture_env_keys" ]]; then
            while IFS= read -r key; do
                [[ -z "${key}" ]] && continue
                value=""
                [[ "${!key+x}" == "x" ]] && value="${!key}"
                printf '%s=%s\n' "${key}" "${value}" >> "${fixtures}/invarlock.env"
            done < "${fixtures}/invarlock.capture_env_keys"
        fi
        case "${subcmd}" in
            run)
                out_dir="$(get_flag_val --out "$@" || true)"
                if [[ -n "${out_dir:-}" ]]; then
                    if [[ -f "${fixtures}/invarlock.create_report_nested" ]]; then
                        nested_dir="${out_dir}/run_1"
                        mkdir -p "${nested_dir}"
                        printf '{"ok":true}\n' > "${nested_dir}/report.json"
                    elif [[ -f "${fixtures}/invarlock.create_report" ]]; then
                        mkdir -p "${out_dir}"
                        printf '{"ok":true}\n' > "${out_dir}/report.json"
                    fi
                fi
                ;;
            evaluate)
                cert_out="$(get_flag_val --report-out "$@" || true)"
                out_dir="$(get_flag_val --out "$@" || true)"
                target="${cert_out:-${out_dir:-}}"
                if [[ -n "${target:-}" && -f "${fixtures}/invarlock.create_cert" ]]; then
                    mkdir -p "${target}"
                    printf '{"ok":true}\n' > "${target}/evaluation.report.json"
                fi
                if [[ -n "${target:-}" && -f "${fixtures}/invarlock.create_report_for_evaluate" ]]; then
                    nested_dir="${target}/edited/000000"
                    mkdir -p "${nested_dir}"
                    printf '{"ok":true}\n' > "${nested_dir}/report.json"
                fi
                ;;
        esac
        if [[ -f "${fixtures}/invarlock.rc" ]]; then
            rc="$(cat "${fixtures}/invarlock.rc" 2>/dev/null || echo "0")"
            [[ "${rc}" =~ ^-?[0-9]+$ ]] && exit "${rc}"
        fi
        exit 0
        ;;
    kill)
        mkdir -p "${fixtures}" 2>/dev/null || true
        echo "kill $*" >> "${fixtures}/kill.calls" 2>/dev/null || true
        if [[ "${1:-}" == "-0" && -n "${2:-}" ]]; then
            pid="${2}"
            grep -q -E "^${pid}$" "${fixtures}/kill/alive" 2>/dev/null
            exit $?
        fi
        exit 0
        ;;
    nvidia-smi)
        fixtures_dir="${fixtures}/nvidia-smi"
        gpu_id=""
        query=""
        args=("$@")
        for ((i=0; i<${#args[@]}; i++)); do
            case "${args[$i]}" in
                -i) gpu_id="${args[$((i+1))]:-}" ;;
                --query-gpu=*) query="${args[$i]#--query-gpu=}" ;;
                --query-compute-apps=*) query="${args[$i]#--query-compute-apps=}" ;;
            esac
        done
        [[ -n "${gpu_id}" ]] || gpu_id="0"
        if [[ -z "${query}" && -f "${fixtures_dir}/invalid_ids" ]] && grep -q -E "^${gpu_id}$" "${fixtures_dir}/invalid_ids" 2>/dev/null; then
            exit 1
        fi
        case "${query}" in
            index) cat "${fixtures_dir}/indices" 2>/dev/null || echo "0" ;;
            memory.free) cat "${fixtures_dir}/memory_free.${gpu_id}" 2>/dev/null || echo "0" ;;
            memory.total) cat "${fixtures_dir}/memory_total.${gpu_id}" 2>/dev/null || echo "184320" ;;
            utilization.gpu) cat "${fixtures_dir}/utilization.${gpu_id}" 2>/dev/null || echo "0" ;;
            pid) cat "${fixtures_dir}/compute_pids.${gpu_id}" 2>/dev/null || true ;;
            *) ;;
        esac
        ;;
    ps)
        fixtures_dir="${fixtures}/ps"
        want_pgid="false"
        pid=""
        args=("$@")
        for ((i=0; i<${#args[@]}; i++)); do
            case "${args[$i]}" in
                -o)
                    [[ "${args[$((i+1))]:-}" == "pgid=" ]] && want_pgid="true"
                    ;;
                -p) pid="${args[$((i+1))]:-}" ;;
            esac
        done
        [[ -n "${pid}" ]] || exit 1
        grep -q -E "^${pid}$" "${fixtures_dir}/alive" 2>/dev/null || exit 1
        if [[ "${want_pgid}" == "true" ]]; then
            cat "${fixtures_dir}/pgid.${pid}" 2>/dev/null || echo "${pid}"
        fi
        ;;
    python3)
        real_python3="${TEST_REAL_PYTHON3:-}"
        if [[ -n "${TEST_TMPDIR:-}" && "${1:-}" == */run_from_config.py ]]; then
            mkdir -p "${fixtures}"
            echo "python3 $*" >> "${fixtures}/python3.calls" 2>/dev/null || true
            if [[ -f "${fixtures}/python3.capture_env_keys" ]]; then
                while IFS= read -r key; do
                    [[ -z "${key}" ]] && continue
                    value=""
                    [[ "${!key+x}" == "x" ]] && value="${!key}"
                    printf '%s=%s\n' "${key}" "${value}" >> "${fixtures}/python3.env"
                done < "${fixtures}/python3.capture_env_keys"
            fi
            out_dir="$(get_flag_val --out "$@" || true)"
            if [[ -n "${out_dir:-}" ]]; then
                if [[ -f "${fixtures}/python3.create_report_nested" ]]; then
                    nested_dir="${out_dir}/run_1"
                    mkdir -p "${nested_dir}"
                    printf '{"ok":true}\n' > "${nested_dir}/report.json"
                elif [[ -f "${fixtures}/python3.create_report" ]]; then
                    mkdir -p "${out_dir}"
                    printf '{"ok":true}\n' > "${out_dir}/report.json"
                fi
            fi
            [[ -f "${fixtures}/python3.stdout" ]] && cat "${fixtures}/python3.stdout"
            [[ -f "${fixtures}/python3.stderr" ]] && cat "${fixtures}/python3.stderr" >&2
            if [[ -f "${fixtures}/python3.rc" ]]; then
                rc="$(cat "${fixtures}/python3.rc" 2>/dev/null || echo "0")"
                exit "${rc}"
            fi
            exit 0
        fi
        if [[ -f "${fixtures}/python3.stub" ]]; then
            if [[ -f "${fixtures}/python3.real_passthrough" && -n "${1:-}" ]]; then
                while IFS= read -r script_name || [[ -n "${script_name}" ]]; do
                    [[ -n "${script_name}" ]] || continue
                    if [[ "${1}" == "${script_name}" || "${1}" == */"${script_name}" ]]; then
                        [[ -n "${real_python3}" && -x "${real_python3}" ]] || {
                            echo "ERROR: TEST_REAL_PYTHON3 is not set to an executable path" >&2
                            exit 127
                        }
                        exec "${real_python3}" "$@"
                    fi
                done < "${fixtures}/python3.real_passthrough"
            fi
            [[ -f "${fixtures}/python3.stdout" ]] && cat "${fixtures}/python3.stdout"
            [[ -f "${fixtures}/python3.stderr" ]] && cat "${fixtures}/python3.stderr" >&2
            if [[ -f "${fixtures}/python3.rc" ]]; then
                rc="$(cat "${fixtures}/python3.rc" 2>/dev/null || echo "0")"
                exit "${rc}"
            fi
            exit 0
        fi
        [[ -n "${real_python3}" && -x "${real_python3}" ]] || {
            echo "ERROR: TEST_REAL_PYTHON3 is not set to an executable path" >&2
            exit 127
        }
        exec "${real_python3}" "$@"
        ;;
    sleep)
        mkdir -p "${fixtures}" 2>/dev/null || true
        echo "sleep $*" >> "${fixtures}/sleep.calls" 2>/dev/null || true
        ;;
    stat)
        fixtures_file="${fixtures}/stat/mtime"
        fmt=""
        args=("$@")
        for ((i=0; i<${#args[@]}; i++)); do
            case "${args[$i]}" in
                -c|-f) fmt="${args[$((i+1))]:-}" ;;
            esac
        done
        path="${args[$(( ${#args[@]} - 1 ))]}"
        if [[ -f "${fixtures_file}" ]]; then
            value="$(awk -v p="${path}" 'BEGIN{found=0} $1==p {print $2; found=1; exit} END{if(!found) exit 1}' "${fixtures_file}" 2>/dev/null || true)"
            [[ -n "${value:-}" ]] && { echo "${value}"; exit 0; }
        fi
        echo "1700000000"
        ;;
    timeout)
        if [[ -f "${fixtures}/timeout.stub" ]]; then
            echo "timeout $*" >> "${fixtures}/timeout.calls" 2>/dev/null || true
            if [[ -f "${fixtures}/timeout.rc" ]]; then
                rc="$(cat "${fixtures}/timeout.rc" 2>/dev/null || echo "0")"
                exit "${rc}"
            fi
            exit 0
        fi
        [[ $# -ge 2 ]] || exit 1
        shift
        exec "$@"
        ;;
    *)
        echo "unknown mock command: ${tool}" >&2
        exit 127
        ;;
esac
EOF
    chmod +x "${dispatcher}"
    local command_name
    for command_name in date df du flock hostname invarlock kill nvidia-smi ps python3 sleep stat timeout; do
        ln -sf "mock-command" "${bin_dir}/${command_name}"
    done
    echo "${bin_dir}"
}

mock_python3_stub_enable() {
    fixture_write "python3.stub" ""
}

mock_python3_stub_allow_real_script() {
    local script_name="$1"
    local path="${TEST_TMPDIR}/fixtures/python3.real_passthrough"
    mkdir -p "$(dirname "${path}")"
    printf '%s\n' "${script_name}" >> "${path}"
}

mock_python3_force_real_cmd_python() {
    _cmd_python() { command "${TEST_REAL_PYTHON3}" "$@"; }
}

push_active_python_bin() {
    if [[ "${PYTHON_BIN+x}" == "x" ]]; then
        TEST_PREV_PYTHON_BIN="${PYTHON_BIN}"
        TEST_PREV_PYTHON_BIN_WAS_SET="1"
    else
        TEST_PREV_PYTHON_BIN=""
        TEST_PREV_PYTHON_BIN_WAS_SET="0"
    fi

    local active_python=""
    active_python="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
    [[ -n "${active_python}" ]] || t_fail "python executable not found for PYTHON_BIN override"
    export PYTHON_BIN="${active_python}"
}

pop_active_python_bin() {
    if [[ "${TEST_PREV_PYTHON_BIN_WAS_SET:-0}" == "1" ]]; then
        export PYTHON_BIN="${TEST_PREV_PYTHON_BIN}"
    else
        unset PYTHON_BIN
    fi
    unset TEST_PREV_PYTHON_BIN
    unset TEST_PREV_PYTHON_BIN_WAS_SET
}

mock_nvidia_smi_set_mem_free_mib() {
    local gpu_id="$1"
    local mib="$2"
    fixture_write "nvidia-smi/memory_free.${gpu_id}" "$(printf '%s\n' "${mib}")"
}

mock_nvidia_smi_set_mem_total_mib() {
    local gpu_id="$1"
    local mib="$2"
    fixture_write "nvidia-smi/memory_total.${gpu_id}" "$(printf '%s\n' "${mib}")"
}

mock_nvidia_smi_set_pids() {
    local gpu_id="$1"
    local pids_text="$2"
    fixture_write "nvidia-smi/compute_pids.${gpu_id}" "${pids_text}"
}

mock_df_set_output() {
    local text="$1"
    fixture_write "df.out" "${text}"
}

mock_ps_set_alive() {
    local pid="$1"
    fixture_append "ps/alive" "$(printf '%s\n' "${pid}")"
}

mock_ps_set_pgid() {
    local pid="$1"
    local pgid="$2"
    fixture_write "ps/pgid.${pid}" "$(printf '%s\n' "${pgid}")"
}
