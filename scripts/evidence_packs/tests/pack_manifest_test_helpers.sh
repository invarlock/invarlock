#!/usr/bin/env bash

pack_test_sign_manifest() {
    local pack_dir="$1"
    local repo_root="${TEST_ROOT:-$(pwd)}"
    PYTHONPATH="${repo_root}" python3 -m tests._support_evidence_pack_signing \
        "${pack_dir}/manifest.json" \
        >/dev/null
}
