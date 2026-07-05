#!/usr/bin/env bash

pack_test_sign_manifest() {
    local pack_dir="$1"
    local repo_root="${TEST_ROOT:-$(pwd)}"
    python3 "${repo_root}/scripts/evidence_packs/python/manifest_writer.py" sign \
        --manifest "${pack_dir}/manifest.json" \
        --generate-ephemeral \
        >/dev/null
}
