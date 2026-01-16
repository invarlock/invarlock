#!/usr/bin/env bash

test_suites_list_returns_known_suites() {
    mock_reset

    source ./scripts/proof_packs/suites.sh

    local out
    out="$(pack_list_suites)"
    assert_match "subset" "${out}" "lists subset"
    assert_match "full" "${out}" "lists full"
}

test_suites_apply_subset_sets_models() {
    mock_reset

    source ./scripts/proof_packs/suites.sh

    pack_apply_suite subset

    assert_eq "subset" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "" "${MODEL_2}" "model 2 cleared"
    assert_eq "" "${MODEL_3}" "model 3 cleared"
}

test_suites_apply_full_sets_models() {
    mock_reset

    source ./scripts/proof_packs/suites.sh

    pack_apply_suite full

    assert_eq "full" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "Qwen/Qwen2.5-14B" "${MODEL_2}" "model 2 set"
    assert_eq "Qwen/Qwen2.5-32B" "${MODEL_3}" "model 3 set"
    assert_eq "01-ai/Yi-34B" "${MODEL_4}" "model 4 set"
    assert_eq "mistralai/Mixtral-8x7B-v0.1" "${MODEL_5}" "model 5 set"
}

test_suites_apply_invalid_suite_returns_error() {
    mock_reset

    source ./scripts/proof_packs/suites.sh

    run pack_apply_suite nope
    assert_rc "2" "${RUN_RC}" "invalid suite returns 2"
}
