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
