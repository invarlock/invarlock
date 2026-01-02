#!/usr/bin/env bash

test_b200_bootstrap_help_exits_zero_and_prints_usage() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local log_file="${work_dir}/setup.log"

    local out rc
    set +e
    out="$(env WORK_DIR="${work_dir}" LOG_FILE="${log_file}" bash -x ./scripts/b200_bootstrap_and_validate.sh --help)"
    rc=$?
    set -e
    assert_rc "0" "${rc}" "help should exit 0"
    assert_match "B200 Setup and Validation Runner" "${out}" "help header"
    assert_dir_exists "${work_dir}" "work dir created"
}

test_b200_bootstrap_install_system_deps_branches() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${work_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    local bash_path=""
    bash_path="$(command -v bash)"
    cat > "${bin_dir}/bash" <<EOF
#!/bin/sh
exec "${bash_path}" "\$@"
EOF
    chmod +x "${bin_dir}/bash"

    cat > "${bin_dir}/sudo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
calls="${TEST_TMPDIR:-}/sudo.calls"
echo "$*" >> "${calls}"
exec "$@"
EOF
    chmod +x "${bin_dir}/sudo"

    cat > "${bin_dir}/apt-get" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/apt-get"

    cat > "${bin_dir}/yum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/yum"

    cat > "${bin_dir}/date" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "2024-01-01 00:00:00"
EOF
    chmod +x "${bin_dir}/date"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    LOG_FILE="${log_file}"

    _bootstrap_is_root() { return 0; }
    check_root

    # apt-get branch (root, no sudo)
    hash -r
    PATH="${bin_dir}" install_system_deps

    # apt-get branch (non-root uses sudo)
    _bootstrap_is_root() { return 1; }
    : > "${TEST_TMPDIR}/sudo.calls"
    hash -r
    PATH="${bin_dir}" install_system_deps
    assert_file_exists "${TEST_TMPDIR}/sudo.calls" "sudo used when not root"

    # apt-get branch (non-root without sudo errors)
    mv "${bin_dir}/sudo" "${bin_dir}/sudo.off"
    local rc=0
    set +e
    (
        hash -r
        PATH="${bin_dir}"
        _bootstrap_is_root() { return 1; }
        install_system_deps
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "missing sudo causes failure"

    # yum branch
    mv "${bin_dir}/apt-get" "${bin_dir}/apt-get.off"
    _bootstrap_is_root() { return 0; }
    hash -r
    PATH="${bin_dir}" install_system_deps

    # yum branch (non-root without sudo errors)
    set +e
    (
        hash -r
        PATH="${bin_dir}"
        _bootstrap_is_root() { return 1; }
        install_system_deps
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "missing sudo causes failure on yum"

    # unknown package manager branch
    mv "${bin_dir}/yum" "${bin_dir}/yum.off"
    rc=0
    set +e
    (
        hash -r
        PATH="${bin_dir}"
        install_system_deps
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "unknown package manager exits non-zero"
}

test_b200_bootstrap_setup_python_venv_creates_and_reuses_venv() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${work_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/python3.12" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" && -n "${3:-}" ]]; then
    venv_dir="${3}"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python3.12"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
    echo "Python 3.12.0"
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"
    export HF_HOME="${WORK_DIR}/hf_home"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

    setup_python_venv
    setup_python_venv
}

test_b200_bootstrap_run_validation_error_branches() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${work_dir}/.venv/bin"
    : > "${work_dir}/.venv/bin/activate"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

queue="${TEST_TMPDIR:-}/fixtures/python.rc_queue"

if [[ "${1:-}" == "--version" ]]; then
    echo "Python 3.12.0"
    exit 0
fi

if [[ "${1:-}" == "-m" ]]; then
    # pip/invarlock invocations: succeed deterministically.
    exit 0
fi

if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi

if [[ "${1:-}" == "-" ]]; then
    cat >/dev/null || true
    if [[ -f "${queue}" ]]; then
        rc="$(head -n 1 "${queue}" 2>/dev/null || echo "0")"
        tail -n +2 "${queue}" > "${queue}.tmp" 2>/dev/null || true
        mv "${queue}.tmp" "${queue}" 2>/dev/null || true
        exit "${rc}"
    fi
    exit 0
fi

exit 0
EOF
    chmod +x "${bin_dir}/python"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"

    # Patch verification fails (metrics verification).
    fixture_write "python.rc_queue" "$(printf '0\n1\n')"
    local rc=0
    set +e
    ( run_validation )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "metrics verification failure exits non-zero"
}

test_b200_bootstrap_install_invarlock_deps_and_patch_branches() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    # Local repo + requirements.txt fixtures.
    mkdir -p "${work_dir}/src/invarlock"
    : > "${work_dir}/pyproject.toml"
    printf "torch==0.0.0\n" > "${work_dir}/requirements.txt"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

queue="${TEST_TMPDIR:-}/fixtures/python.rc_queue"

if [[ "${1:-}" == "-m" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-" ]]; then
    cat >/dev/null || true
    if [[ -f "${queue}" ]]; then
        rc="$(head -n 1 "${queue}" 2>/dev/null || echo "0")"
        tail -n +2 "${queue}" > "${queue}.tmp" 2>/dev/null || true
        mv "${queue}.tmp" "${queue}" 2>/dev/null || true
        exit "${rc}"
    fi
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"
    INVARLOCK_SRC="${work_dir}"

    # CLI missing branch.
    fixture_write "python.rc_queue" "$(printf '0\n0\n')"
    (
        PATH="${bin_dir}:/usr/bin:/bin"
        install_invarlock_deps
    )

    # CLI present branch.
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    fixture_write "python.rc_queue" "$(printf '0\n0\n')"
    install_invarlock_deps

    # Patch warning branches (python exit non-zero).
    fixture_write "python.rc_queue" "$(printf '1\n')"
    patch_invarlock_metrics_gather
}

test_b200_bootstrap_configure_gpu_env_and_validate_environment_branches() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"
    export HF_HOME="${WORK_DIR}/hf_home"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

    # configure_gpu_env: raw_list branch + requested default.
    nvidia-smi() { return 0; }
    CUDA_VISIBLE_DEVICES="0,1"
    NUM_GPUS=""
    configure_gpu_env

    # requested not numeric
    CUDA_VISIBLE_DEVICES="0,1"
    NUM_GPUS="nope"
    configure_gpu_env

    # requested < 1 coerces to 1
    CUDA_VISIBLE_DEVICES="0,1"
    NUM_GPUS="0"
    configure_gpu_env

    # requested > available clamps
    CUDA_VISIBLE_DEVICES="0,1"
    NUM_GPUS="99"
    configure_gpu_env

    # invalid non-numeric id exits 1
    local rc=0
    set +e
    (
        CUDA_VISIBLE_DEVICES="0,bad"
        NUM_GPUS=""
        nvidia-smi() { return 0; }
        configure_gpu_env
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "non-numeric id exits"

    # invalid id rejected by nvidia-smi -i exits 1
    set +e
    (
        CUDA_VISIBLE_DEVICES="0"
        NUM_GPUS=""
        nvidia-smi() { [[ "${1:-}" == "-i" ]] && return 1; return 0; }
        configure_gpu_env
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "invalid gpu id exits"

    # no GPUs detected exits 1 (raw_list empty, empty index list)
    set +e
    (
        unset CUDA_VISIBLE_DEVICES GPU_ID_LIST
        NUM_GPUS=""
        unset -f nvidia-smi || true
        fixture_write "nvidia-smi/indices" ""
        configure_gpu_env
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "no gpus exits"

    # validate_environment: exercise warning + script/lib present/missing branches.
    nvidia-smi() {
        case "$*" in
            *"--query-gpu=name"*) printf "B200\n" ;;
            *"--query-gpu=index"*) printf "0\n1\n" ;;
            *) return 0 ;;
        esac
        return 0
    }

    rm -f "${work_dir}/b200_validation_suite.sh"
    rm -rf "${work_dir}/lib"
    validate_environment

    : > "${work_dir}/b200_validation_suite.sh"
    mkdir -p "${work_dir}/lib"
    : > "${work_dir}/lib/x.sh"
    validate_environment

    # validate_environment: exit branch when nvidia-smi probe fails.
    local rc=0
    set +e
    (
        nvidia-smi() { return 1; }
        validate_environment
    )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "nvidia-smi failure exits non-zero"
}

test_b200_bootstrap_run_validation_missing_script_and_exit_code_branches() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
queue="${TEST_TMPDIR:-}/fixtures/python.rc_queue"
if [[ "${1:-}" == "-m" || "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-" ]]; then
    cat >/dev/null || true
    if [[ -f "${queue}" ]]; then
        rc="$(head -n 1 "${queue}" 2>/dev/null || echo "0")"
        tail -n +2 "${queue}" > "${queue}.tmp" 2>/dev/null || true
        mv "${queue}.tmp" "${queue}" 2>/dev/null || true
        exit "${rc}"
    fi
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python"
    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"
    export HF_HOME="${WORK_DIR}/hf_home"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

    # metrics verification fails.
    fixture_write "python.rc_queue" "$(printf '0\n0\n0\n1\n')"
    local rc=0
    set +e
    ( run_validation )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "metrics verification failure exits"

    # validation script missing.
    fixture_write "python.rc_queue" "$(printf '0\n0\n0\n0\n')"
    rm -f "${work_dir}/b200_validation_suite.sh"
    set +e
    ( run_validation )
    rc=$?
    set -e
    assert_ne "0" "${rc}" "missing validation script exits"

    # chmod branch + exit_code branches.
    cat > "${work_dir}/b200_validation_suite.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod -x "${work_dir}/b200_validation_suite.sh"
    fixture_write "python.rc_queue" "$(printf '0\n0\n0\n0\n')"
    run_validation

    cat > "${work_dir}/b200_validation_suite.sh" <<'EOF'
#!/usr/bin/env bash
exit 3
EOF
    chmod +x "${work_dir}/b200_validation_suite.sh"
    fixture_write "python.rc_queue" "$(printf '0\n0\n0\n0\n')"
    rc=0
    run_validation || rc=$?
    assert_rc "3" "${rc}" "non-zero exit code propagated"
}

test_b200_bootstrap_post_run_diagnostics_and_main_case_arms() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"

    post_run_diagnostics

    local out_dir="${work_dir}/invarlock_validation_b200_zzz"
    mkdir -p "${out_dir}/workers" "${out_dir}/queue/completed" "${out_dir}/queue/failed" "${out_dir}/queue/pending" "${out_dir}/queue/ready" "${out_dir}/logs"
    echo "123" > "${out_dir}/workers/gpu_0.pid"
    : > "${out_dir}/queue/completed/a.task"
    : > "${out_dir}/logs/a.log"
    post_run_diagnostics

    # Cover main() case arms without executing heavy operations.
    install_system_deps() { return 0; }
    setup_python_venv() { return 0; }
    install_pytorch_b200() { return 0; }
    install_invarlock_deps() { return 0; }
    validate_environment() { return 0; }
    run_validation() { return 0; }
    post_run_diagnostics() { return 0; }

    main "setup-only"
    main "run-only"
    main "diagnostics"
    main "unknown-mode"
}

test_b200_bootstrap_install_pytorch_b200_runs_offline_with_stubbed_python() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-m" ]]; then
    # pip install invocations (offline).
    exit 0
fi

if [[ "${1:-}" == "-" ]]; then
    # heredoc verification script
    cat >/dev/null || true
    exit 0
fi

exit 0
EOF
    chmod +x "${bin_dir}/python"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"

    install_pytorch_b200
}

test_b200_bootstrap_install_invarlock_deps_logs_warning_when_import_fails() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-m" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
    exit 1
fi
if [[ "${1:-}" == "-" ]]; then
    cat >/dev/null || true
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python"

    export PATH="${bin_dir}:$PATH"

    source ./scripts/b200_bootstrap_and_validate.sh
    WORK_DIR="${work_dir}"
    VENV_DIR="${venv_dir}"
    LOG_FILE="${log_file}"

    install_invarlock_deps
    assert_match "WARNING: Could not import invarlock" "$(cat "${log_file}")" "import warning logged"
}

test_b200_bootstrap_entrypoint_invokes_main_when_executed() {
    mock_reset

    local work_dir="${TEST_TMPDIR}/work"
    local venv_dir="${work_dir}/.venv"
    local log_file="${work_dir}/setup.log"
    mkdir -p "${venv_dir}/bin"
    : > "${venv_dir}/bin/activate"

    run env WORK_DIR="${work_dir}" VENV_DIR="${venv_dir}" LOG_FILE="${log_file}" \
        bash -x ./scripts/b200_bootstrap_and_validate.sh diagnostics
    assert_rc "0" "${RUN_RC}" "diagnostics mode exits successfully"
}
