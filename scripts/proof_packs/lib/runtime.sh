#!/usr/bin/env bash
# runtime.sh - Deterministic hooks + command wrappers for proof pack scripts.
#
# This file is intentionally tiny and safe-to-source. Tests can override any
# function here via function redefinition after sourcing.

if ! declare -F _cmd_date >/dev/null 2>&1; then
    :
    _cmd_date() { command date "$@"; }
fi

if ! declare -F _cmd_sleep >/dev/null 2>&1; then
    :
    _cmd_sleep() { command sleep "$@"; }
fi

if ! declare -F _cmd_hostname >/dev/null 2>&1; then
    :
    _cmd_hostname() { command hostname "$@"; }
fi

if ! declare -F _cmd_nvidia_smi >/dev/null 2>&1; then
    :
    _cmd_nvidia_smi() { command nvidia-smi "$@"; }
fi

if ! declare -F _cmd_df >/dev/null 2>&1; then
    :
    _cmd_df() { command df "$@"; }
fi

if ! declare -F _cmd_du >/dev/null 2>&1; then
    :
    _cmd_du() { command du "$@"; }
fi

if ! declare -F _cmd_ps >/dev/null 2>&1; then
    :
    _cmd_ps() { command ps "$@"; }
fi

if ! declare -F _cmd_kill >/dev/null 2>&1; then
    :
    _cmd_kill() { command kill "$@"; }
fi

if ! declare -F _cmd_python >/dev/null 2>&1; then
    :
    _cmd_python() { command python3 "$@"; }
fi

if ! declare -F _cmd_stat >/dev/null 2>&1; then
    :
    _cmd_stat() { command stat "$@"; }
fi

if ! declare -F _now_epoch >/dev/null 2>&1; then
    :
    _now_epoch() { _cmd_date +%s; }
fi

if ! declare -F _now_iso >/dev/null 2>&1; then
    :
    _now_iso() { _cmd_date -u +"%Y-%m-%dT%H:%M:%SZ"; }
fi

if ! declare -F _sleep >/dev/null 2>&1; then
    :
    _sleep() { _cmd_sleep "$@"; }
fi

if ! declare -F _rand_jitter_ms >/dev/null 2>&1; then
    # Returns an integer jitter in ms in the inclusive range [-max_abs_ms, +max_abs_ms].
    :
    _rand_jitter_ms() {
        local max_abs_ms="${1:-0}"
        if ! [[ "${max_abs_ms}" =~ ^[0-9]+$ ]] || [[ "${max_abs_ms}" -le 0 ]]; then
            echo "0"
            return 0
        fi
        local span=$((max_abs_ms * 2 + 1))
        local offset=$((RANDOM % span))
        echo "$((offset - max_abs_ms))"
    }
fi

if ! declare -F _hostname >/dev/null 2>&1; then
    :
    _hostname() { _cmd_hostname; }
fi

if ! declare -F _has_proc_dir >/dev/null 2>&1; then
    :
    _has_proc_dir() { [[ -d /proc ]]; }
fi

if ! declare -F _pid_is_alive_backend >/dev/null 2>&1; then
    :
    _pid_is_alive_backend() {
        if _has_proc_dir; then
            echo "proc"
        else
            echo "ps"
        fi
    }
fi

if ! declare -F _pid_is_alive_proc >/dev/null 2>&1; then
    :
    _pid_is_alive_proc() {
        local pid="$1"
        [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ && -d "/proc/${pid}" ]]
    }
fi

if ! declare -F _pid_is_alive_ps >/dev/null 2>&1; then
    :
    _pid_is_alive_ps() {
        local pid="$1"
        [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 1
        _cmd_ps -p "${pid}" >/dev/null 2>&1
    }
fi

if ! declare -F _pid_is_alive >/dev/null 2>&1; then
    :
    _pid_is_alive() {
        local pid="$1"
        case "$(_pid_is_alive_backend)" in
            proc)
                _pid_is_alive_proc "${pid}"
                ;;
            ps)
                _pid_is_alive_ps "${pid}"
                ;;
            *)
                _pid_is_alive_ps "${pid}"
                ;;
        esac
    }
fi

if ! declare -F _file_mtime_epoch >/dev/null 2>&1; then
    :
    _file_mtime_epoch() {
        local path="$1"
        local out=""

        out=$(_cmd_stat -c %Y "${path}" 2>/dev/null) && [[ "${out}" =~ ^[0-9]+$ ]] && { echo "${out}"; return 0; }
        out=$(_cmd_stat -f %m "${path}" 2>/dev/null) && [[ "${out}" =~ ^[0-9]+$ ]] && { echo "${out}"; return 0; }
        return 1
    }
fi

if ! declare -F _iso_to_epoch >/dev/null 2>&1; then
    :
    _iso_to_epoch() {
        local iso="$1"
        local out=""

        if [[ -z "${iso}" || "${iso}" == "null" ]]; then
            echo "0"
            return 0
        fi

        out=$(_cmd_date -d "${iso}" +%s 2>/dev/null) && [[ "${out}" =~ ^[0-9]+$ ]] && { echo "${out}"; return 0; }
        out=$(_cmd_date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "${iso}" +%s 2>/dev/null) && [[ "${out}" =~ ^[0-9]+$ ]] && { echo "${out}"; return 0; }
        # Fallback to python (portable).
        _cmd_python - "${iso}" <<'PY' 2>/dev/null || echo "0"
import datetime, sys
iso = sys.argv[1] if len(sys.argv) > 1 else ""
if not iso:
    print("0")
    raise SystemExit(0)
try:
    dt = datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    print(int(dt.timestamp()))
except Exception:
    print("0")
PY
    }
fi

if ! declare -F _now_iso_plus_seconds >/dev/null 2>&1; then
    :
    _now_iso_plus_seconds() {
        local seconds="${1:-0}"
        if ! [[ "${seconds}" =~ ^-?[0-9]+$ ]]; then
            seconds=0
        fi

        local delta_d="+${seconds} seconds"
        local delta_v="-v+${seconds}S"
        if [[ ${seconds} -lt 0 ]]; then
            local abs=$((0 - seconds))
            delta_d="-${abs} seconds"
            delta_v="-v-${abs}S"
        fi

        local out=""
        out=$(_cmd_date -u -d "${delta_d}" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null) && { echo "${out}"; return 0; }
        out=$(_cmd_date -u "${delta_v}" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null) && { echo "${out}"; return 0; }

        # Fallback to python (portable).
        _cmd_python - "${seconds}" <<'PY' 2>/dev/null || _now_iso
import datetime, sys
seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
print(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
    }
fi
