#!/usr/bin/env bash

test_setup_remote_install_torch_uses_index_url() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    pack_activate_venv() { :; }
    pack_run_cmd() { echo "$*" > "${TEST_TMPDIR}/cmd"; }

    TORCH_INDEX_URL="https://example.com/simple"
    TORCH_PACKAGES="torch torchvision"
    PACK_SKIP_TORCH_CHECK=1

    install_torch

    local cmd
    cmd="$(cat "${TEST_TMPDIR}/cmd")"
    assert_match "--index-url https://example.com/simple" "${cmd}" "index url applied"
    assert_match "torch torchvision" "${cmd}" "packages included"
}

test_setup_remote_install_invarlock_stack_uses_pinned_requirement_repairs() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    local cmd_log="${TEST_TMPDIR}/install.log"
    : > "${cmd_log}"

    pack_activate_venv() { :; }
    pack_run_cmd() { printf '%s\n' "$*" >> "${cmd_log}"; }

    REPO_DIR="${TEST_TMPDIR}/opt-invarlock"
    mkdir -p "${REPO_DIR}/requirements/evidence-packs"
    touch \
        "${REPO_DIR}/requirements/evidence-packs/huggingface_hub.txt" \
        "${REPO_DIR}/requirements/evidence-packs/accelerate.txt" \
        "${REPO_DIR}/requirements/evidence-packs/pyyaml.txt" \
        "${REPO_DIR}/requirements/evidence-packs/protobuf.txt" \
        "${REPO_DIR}/requirements/evidence-packs/sentencepiece.txt"

    install_invarlock_stack

    local log_text
    log_text="$(cat "${cmd_log}")"
    assert_match "python -m pip install -e \\.\[hf\\]" "${log_text}" "editable hf install"
    assert_match "requirements/evidence-packs/huggingface_hub.txt" "${log_text}" "huggingface_hub pinned repair"
    assert_match "requirements/evidence-packs/accelerate.txt" "${log_text}" "accelerate pinned repair"
    assert_match "requirements/evidence-packs/pyyaml.txt" "${log_text}" "pyyaml pinned repair"
    assert_match "requirements/evidence-packs/protobuf.txt" "${log_text}" "protobuf pinned repair"
    assert_match "requirements/evidence-packs/sentencepiece.txt" "${log_text}" "sentencepiece pinned repair"
}

test_setup_remote_verify_remote_stack_runs_package_native_smoke() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    pack_activate_venv() { :; }
    pack_run_cmd() { echo "$*" > "${TEST_TMPDIR}/smoke.cmd"; }

    REPO_DIR="/opt/invarlock"

    verify_remote_stack

    local cmd
    cmd="$(cat "${TEST_TMPDIR}/smoke.cmd")"
    assert_match "python /opt/invarlock/scripts/evidence_packs/python/remote_setup_smoke.py" "${cmd}" "remote smoke helper invoked"
    assert_match "--repo-root /opt/invarlock" "${cmd}" "repo root forwarded to smoke helper"
}

test_setup_remote_ensure_runtime_image_builds_cuda_local_when_missing() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    local cmd_log="${TEST_TMPDIR}/runtime-image.log"
    : > "${cmd_log}"

    pack_activate_venv() { :; }
    pack_run_cmd() { printf '%s\n' "$*" >> "${cmd_log}"; }

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
  exit 1
fi
exit 0
EOF
    chmod +x "${bin_dir}/docker"

    cat > "${bin_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/nvidia-smi"

    PATH="${bin_dir}:${PATH}"
    export PATH

    REPO_DIR="${TEST_TMPDIR}/opt-invarlock"
    mkdir -p "${REPO_DIR}"
    unset INVARLOCK_RUNTIME_IMAGE

    ensure_runtime_image

    local log_text
    log_text="$(cat "${cmd_log}")"
    assert_match "make runtime-image-cuda" "${log_text}" "cuda runtime image build invoked"
    assert_eq "invarlock-runtime:cuda-local" "${INVARLOCK_RUNTIME_IMAGE}" "cuda runtime image exported"
}

test_setup_remote_ensure_runtime_image_respects_explicit_override() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    local cmd_log="${TEST_TMPDIR}/runtime-image.log"
    : > "${cmd_log}"

    pack_activate_venv() { :; }
    pack_run_cmd() { printf '%s\n' "$*" >> "${cmd_log}"; }

    INVARLOCK_RUNTIME_IMAGE="ghcr.io/example/custom:latest"

    ensure_runtime_image

    local log_text
    log_text="$(cat "${cmd_log}")"
    assert_eq "" "${log_text}" "no runtime image build when explicit override is present"
    assert_eq "ghcr.io/example/custom:latest" "${INVARLOCK_RUNTIME_IMAGE}" "explicit runtime image preserved"
}

test_setup_remote_pack_install_pinned_requirement_requires_file() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    REPO_DIR="${TEST_TMPDIR}/repo"
    mkdir -p "${REPO_DIR}/requirements/evidence-packs"
    pack_run_cmd() { echo "unexpected" > "${TEST_TMPDIR}/unexpected.cmd"; }

    run pack_install_pinned_requirement "missing"
    assert_rc "1" "${RUN_RC}" "missing pinned requirement file fails"
    assert_match "Missing pinned evidence-pack requirement file" "${RUN_ERR}" "error names missing requirement file"
    [[ ! -f "${TEST_TMPDIR}/unexpected.cmd" ]] || t_fail "pack_run_cmd should not run when pinned requirement file is absent"
}

test_setup_remote_post_setup_marks_entrypoints_executable() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    pack_run_cmd() { echo "$*" > "${TEST_TMPDIR}/chmod.cmd"; }

    REPO_DIR="/opt/invarlock"
    post_setup

    local cmd
    cmd="$(cat "${TEST_TMPDIR}/chmod.cmd")"
    assert_match "chmod" "${cmd}" "chmod invoked"
    assert_match "/opt/invarlock/scripts/evidence_packs/run_suite.sh" "${cmd}" "run_suite path"
    assert_match "/opt/invarlock/scripts/evidence_packs/run_pack.sh" "${cmd}" "run_pack path"
    assert_match "/opt/invarlock/scripts/evidence_packs/verify_pack.sh" "${cmd}" "verify_pack path"
    assert_match "/opt/invarlock/scripts/evidence_packs/run_mini_pack_gate.sh" "${cmd}" "mini gate path"
}


test_setup_remote_ensure_repo_alias_links_canonical_root_when_repo_dir_differs() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    local cmd_log="${TEST_TMPDIR}/alias.log"
    : > "${cmd_log}"

    pack_run_cmd() { printf '%s\n' "$*" >> "${cmd_log}"; }

    REPO_DIR="/opt/invarlock-a100"
    CANONICAL_REPO_ALIAS="/root/invarlock-public"

    ensure_repo_alias

    local log_text
    log_text="$(cat "${cmd_log}")"
    assert_match "mkdir -p /root" "${log_text}" "alias parent directory created"
    assert_match "ln -sfn /opt/invarlock-a100 /root/invarlock-public" "${log_text}" "canonical alias refreshed"
}


test_setup_remote_clone_and_torch_check_branches() {
    mock_reset

    source ./scripts/evidence_packs/lib/setup_remote.sh

    local cmd_log="${TEST_TMPDIR}/cmds.log"
    : > "${cmd_log}"

    pack_activate_venv() { :; }
    pack_run_cmd() {
        if [[ "${1:-}" == "git" && "${2:-}" == "clone" ]]; then
            mkdir -p "${REPO_DIR}/.git"
        fi
        printf '%s\n' "$*" >> "${cmd_log}"
    }

    REPO_DIR="${TEST_TMPDIR}/repo"
    REPO_URL="https://example.com/repo.git"
    BRANCH="main"

    clone_or_update_repo
    assert_dir_exists "${REPO_DIR}/.git" "repo cloned"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "called" >> "${TEST_TMPDIR}/python.called"
exit 0
EOF
    chmod +x "${bin_dir}/python"

    cat > "${bin_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--query-gpu=name"* ]]; then
  echo "NVIDIA B200"
  exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/nvidia-smi"

    PATH="${bin_dir}:${PATH}"
    export PATH

    TORCH_PACKAGES="torch"
    PACK_SKIP_TORCH_CHECK=0

    install_torch
    assert_file_exists "${TEST_TMPDIR}/python.called" "torch check executed"
    local call_count
    call_count="$(wc -l < "${TEST_TMPDIR}/python.called" | tr -d ' ')"
    assert_eq "2" "${call_count}" "B200 torch arch warning branch executed"
}

test_setup_remote_main_runs_with_stubbed_commands() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/apt-get" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/apt-get"

    cat > "${bin_dir}/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
if [[ "${cmd}" == "clone" ]]; then
    dest="${@: -1}"
    mkdir -p \
        "${dest}/.git" \
        "${dest}/requirements/evidence-packs" \
        "${dest}/scripts/evidence_packs" \
        "${dest}/scripts/evidence_packs/python"
    touch "${dest}/scripts/evidence_packs/run_suite.sh" \
        "${dest}/scripts/evidence_packs/run_pack.sh" \
        "${dest}/scripts/evidence_packs/verify_pack.sh" \
        "${dest}/scripts/evidence_packs/python/remote_setup_smoke.py" \
        "${dest}/requirements/evidence-packs/huggingface_hub.txt" \
        "${dest}/requirements/evidence-packs/accelerate.txt" \
        "${dest}/requirements/evidence-packs/pyyaml.txt" \
        "${dest}/requirements/evidence-packs/protobuf.txt" \
        "${dest}/requirements/evidence-packs/sentencepiece.txt"
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/git"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    dest="${3:-}"
    mkdir -p "${dest}/bin"
    cat > "${dest}/bin/activate" <<'ACT'
:
ACT
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/python"
    ln -s "${bin_dir}/python" "${bin_dir}/python3.12"

    PATH="${bin_dir}:/usr/bin:/bin"
    export PATH

    export REPO_DIR="${TEST_TMPDIR}/repo"
    export VENV_DIR="${TEST_TMPDIR}/venv"
    export PYTHON_BIN="${bin_dir}/python"
    export REPO_URL="https://example.com/repo.git"
    export BRANCH="main"
    export TORCH_PACKAGES="torch"
    export PACK_SKIP_TORCH_CHECK=0

    run bash -x ./scripts/evidence_packs/lib/setup_remote.sh
    assert_rc "0" "${RUN_RC}" "setup_remote main succeeds"
    assert_dir_exists "${REPO_DIR}/.git" "repo initialized"
    assert_file_exists "${VENV_DIR}/bin/activate" "venv activate created"
}

test_setup_remote_main_bootstrap_satisfies_validation_suite_dependency_check() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}" "${TEST_TMPDIR}/state"

    cat > "${bin_dir}/apt-get" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/apt-get"

    cat > "${bin_dir}/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
if [[ "${cmd}" == "clone" ]]; then
    dest="${@: -1}"
    mkdir -p \
        "${dest}/.git" \
        "${dest}/requirements/evidence-packs" \
        "${dest}/scripts/evidence_packs" \
        "${dest}/scripts/evidence_packs/python"
    touch "${dest}/scripts/evidence_packs/run_suite.sh" \
        "${dest}/scripts/evidence_packs/run_pack.sh" \
        "${dest}/scripts/evidence_packs/verify_pack.sh" \
        "${dest}/scripts/evidence_packs/python/remote_setup_smoke.py" \
        "${dest}/requirements/evidence-packs/huggingface_hub.txt" \
        "${dest}/requirements/evidence-packs/accelerate.txt" \
        "${dest}/requirements/evidence-packs/pyyaml.txt" \
        "${dest}/requirements/evidence-packs/protobuf.txt" \
        "${dest}/requirements/evidence-packs/sentencepiece.txt"
    exit 0
fi
exit 0
EOF
    chmod +x "${bin_dir}/git"

    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

state_dir="${TEST_TMPDIR}/state"
mkdir -p "${state_dir}"

touch_marker() {
    : > "${state_dir}/$1"
}

has_marker() {
    [[ -f "${state_dir}/$1" ]]
}

if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    dest="${3:-}"
    mkdir -p "${dest}/bin"
    cat > "${dest}/bin/activate" <<ACT
export PATH="${dest}/bin:\$PATH"
ACT
    exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
    exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
    printf '%s\n' "$*" >> "${TEST_TMPDIR}/pip.calls"

    for arg in "$@"; do
        case "${arg}" in
            ".[hf]")
                touch_marker invarlock
                touch_marker transformers
                touch_marker datasets
                touch_marker huggingface_hub
                touch_marker yaml
                touch_marker safetensors
                touch_marker tiktoken
                mkdir -p "${VENV_DIR}/bin"
                cat > "${VENV_DIR}/bin/invarlock" <<'CLI'
#!/usr/bin/env bash
exit 0
CLI
                chmod +x "${VENV_DIR}/bin/invarlock"
                ;;
            *requirements/evidence-packs/huggingface_hub.txt)
                touch_marker huggingface_hub
                ;;
            *requirements/evidence-packs/accelerate.txt)
                touch_marker accelerate
                ;;
            *requirements/evidence-packs/pyyaml.txt)
                touch_marker yaml
                ;;
            *requirements/evidence-packs/protobuf.txt)
                touch_marker protobuf
                ;;
            *requirements/evidence-packs/sentencepiece.txt)
                touch_marker sentencepiece
                ;;
        esac
    done

    if [[ "$*" == *"--force-reinstall"* ]]; then
        touch_marker torch
    fi
    exit 0
fi

if [[ "${1:-}" == "-c" ]]; then
    code="${2:-}"
    case "${code}" in
        *"import torch; assert torch.cuda.is_available"*)
            has_marker torch
            exit $?
            ;;
        *"import transformers"*)
            has_marker transformers
            exit $?
            ;;
        *"import huggingface_hub"*)
            has_marker huggingface_hub
            exit $?
            ;;
        *"import accelerate"*)
            has_marker accelerate
            exit $?
            ;;
        *"import yaml"*)
            has_marker yaml
            exit $?
            ;;
        *"import google.protobuf"*)
            has_marker protobuf
            exit $?
            ;;
        *"import sentencepiece"*)
            has_marker sentencepiece
            exit $?
            ;;
        *"import invarlock"*)
            has_marker invarlock
            exit $?
            ;;
        *"import flash_attn; print('Flash Attention OK')"*)
            exit 1
            ;;
        *)
            exit 0
            ;;
    esac
fi

if [[ "${1:-}" == "${REPO_DIR}/scripts/evidence_packs/python/remote_setup_smoke.py" ]]; then
    has_marker invarlock || exit 1
    has_marker torch || exit 1
    has_marker transformers || exit 1
    has_marker datasets || exit 1
    has_marker huggingface_hub || exit 1
    has_marker accelerate || exit 1
    has_marker yaml || exit 1
    has_marker protobuf || exit 1
    has_marker sentencepiece || exit 1
    has_marker safetensors || exit 1
    has_marker tiktoken || exit 1
    [[ -x "${VENV_DIR}/bin/invarlock" ]] || exit 1
    printf '%s\n' "$*" > "${TEST_TMPDIR}/remote_smoke.cmd"
    exit 0
fi

exit 0
EOF
    chmod +x "${bin_dir}/python"
    ln -s "${bin_dir}/python" "${bin_dir}/python3"
    ln -s "${bin_dir}/python" "${bin_dir}/python3.12"

    for tool in jq nvidia-smi flock timeout; do
        cat > "${bin_dir}/${tool}" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
        chmod +x "${bin_dir}/${tool}"
    done

    PATH="${bin_dir}:/usr/bin:/bin"
    export PATH

    export REPO_DIR="${TEST_TMPDIR}/repo"
    export VENV_DIR="${TEST_TMPDIR}/venv"
    export PYTHON_BIN="${bin_dir}/python"
    export REPO_URL="https://example.com/repo.git"
    export BRANCH="main"
    export TORCH_PACKAGES="torch"
    export PACK_SKIP_TORCH_CHECK=1

    run bash -x ./scripts/evidence_packs/lib/setup_remote.sh
    assert_rc "0" "${RUN_RC}" "setup_remote main succeeds"
    assert_file_exists "${TEST_TMPDIR}/remote_smoke.cmd" "remote smoke helper ran"

    PATH="${VENV_DIR}/bin:${bin_dir}:/usr/bin:/bin"
    export PATH
    PACK_NET="1"
    SKIP_FLASH_ATTN="true"
    OUTPUT_DIR="${TEST_TMPDIR}/validation-out"
    source ./scripts/evidence_packs/lib/validation_suite.sh
    pack_setup_output_dirs
    check_dependencies

    local pip_log
    pip_log="$(cat "${TEST_TMPDIR}/pip.calls")"
    assert_match "requirements/evidence-packs/huggingface_hub.txt" "${pip_log}" "huggingface_hub pin installed"
    assert_match "requirements/evidence-packs/accelerate.txt" "${pip_log}" "accelerate pin installed"
    assert_match "requirements/evidence-packs/pyyaml.txt" "${pip_log}" "pyyaml pin installed"
    assert_match "requirements/evidence-packs/protobuf.txt" "${pip_log}" "protobuf pin installed"
    assert_match "requirements/evidence-packs/sentencepiece.txt" "${pip_log}" "sentencepiece pin installed"
}
