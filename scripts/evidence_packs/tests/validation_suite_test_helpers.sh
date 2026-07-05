#!/usr/bin/env bash

_make_validation_suite_sandbox() {
    local sandbox
    sandbox="$(mktemp -d "${TEST_TMPDIR}/pack_validation_suite.XXXXXX")"
    mkdir -p "${sandbox}/lib"
    cp -R "${TEST_ROOT}/scripts/evidence_packs/lib/." "${sandbox}/lib/"
    echo "${sandbox}"
}
