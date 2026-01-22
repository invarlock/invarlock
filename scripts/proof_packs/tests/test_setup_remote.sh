#!/usr/bin/env bash

test_setup_remote_install_torch_uses_index_url() {
    mock_reset

    source ./scripts/proof_packs/lib/setup_remote.sh

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

test_setup_remote_post_setup_marks_entrypoints_executable() {
    mock_reset

    source ./scripts/proof_packs/lib/setup_remote.sh

    pack_run_cmd() { echo "$*" > "${TEST_TMPDIR}/chmod.cmd"; }

    REPO_DIR="/opt/invarlock"
    post_setup

    local cmd
    cmd="$(cat "${TEST_TMPDIR}/chmod.cmd")"
    assert_match "chmod" "${cmd}" "chmod invoked"
    assert_match "/opt/invarlock/scripts/proof_packs/run_suite.sh" "${cmd}" "run_suite path"
    assert_match "/opt/invarlock/scripts/proof_packs/run_pack.sh" "${cmd}" "run_pack path"
    assert_match "/opt/invarlock/scripts/proof_packs/verify_pack.sh" "${cmd}" "verify_pack path"
}


test_setup_remote_clone_and_torch_check_branches() {
    mock_reset

    source ./scripts/proof_packs/lib/setup_remote.sh

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
    mkdir -p "${dest}/.git" "${dest}/scripts/proof_packs"
    touch "${dest}/scripts/proof_packs/run_suite.sh" \
        "${dest}/scripts/proof_packs/run_pack.sh" \
        "${dest}/scripts/proof_packs/verify_pack.sh"
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

    run bash -x ./scripts/proof_packs/lib/setup_remote.sh
    assert_rc "0" "${RUN_RC}" "setup_remote main succeeds"
    assert_dir_exists "${REPO_DIR}/.git" "repo initialized"
    assert_file_exists "${VENV_DIR}/bin/activate" "venv activate created"
}
