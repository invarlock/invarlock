#!/usr/bin/env bash
# suites.sh - Model suite definitions for proof packs.
# NOTE: Keep suites ungated/public.

PACK_SUITE="${PACK_SUITE:-subset}"

pack_list_suites() {
    printf '%s\n' subset full
}

pack_apply_suite() {
    local suite="${1:-${PACK_SUITE:-subset}}"
    case "${suite}" in
        subset)
            # Single-GPU friendly: one 7B model (fits 24GB consumer GPUs).
            # License: Apache-2.0 (permissive, business-friendly).
            MODEL_1="mistralai/Mistral-7B-v0.1"
            MODEL_2=""
            MODEL_3=""
            MODEL_4=""
            MODEL_5=""
            MODEL_6=""
            MODEL_7=""
            MODEL_8=""
            ;;
        full)
            # Multi-GPU: ungated medium/large models.
            MODEL_1="mistralai/Mistral-7B-v0.1"
            MODEL_2="Qwen/Qwen2.5-14B"
            MODEL_3="Qwen/Qwen2.5-32B"
            MODEL_4="01-ai/Yi-34B"
            MODEL_5="mistralai/Mixtral-8x7B-v0.1"
            MODEL_6="Qwen/Qwen1.5-72B"
            MODEL_7=""
            MODEL_8=""
            ;;
        *)
            local available
            available="$(pack_list_suites | paste -sd ', ' -)"
            echo "ERROR: Unknown suite '${suite}'. Available suites: ${available}" >&2
            return 2
            ;;
    esac

    PACK_SUITE="${suite}"
    export PACK_SUITE MODEL_1 MODEL_2 MODEL_3 MODEL_4 MODEL_5 MODEL_6 MODEL_7 MODEL_8
}
