#!/usr/bin/env bash
# dataset_provider_config.sh - YAML helpers for dataset.provider in evidence packs.
#
# Evidence packs historically emitted `dataset.provider: "wikitext2"` (string).
# To support HuggingFace text datasets (hf_text) and local JSONL datasets
# (local_jsonl), we optionally emit a mapping:
#
#   dataset:
#     provider:
#       kind: hf_text
#       dataset_name: c4
#       config_name: en
#       text_field: text
#       max_samples: 512
#
# The evidence-pack execution flows (repo-only `run_from_config.py` plus
# `invarlock evaluate`)
# support both representations.

pack_dataset_provider_kind() {
    local kind="${1:-${INVARLOCK_DATASET:-}}"
    kind="$(echo "${kind}" | xargs)"
    if [[ -z "${kind}" ]]; then
        kind="wikitext2"
    fi
    echo "${kind}"
}

pack_truthy() {
    local value="${1:-}"
    value="$(echo "${value}" | tr '[:upper:]' '[:lower:]' | xargs)"
    [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "y" || "${value}" == "on" ]]
}

pack_remote_code_allowed() {
    pack_truthy "${INVARLOCK_ALLOW_REMOTE_CODE:-}"
}

pack_model_trust_remote_code_yaml() {
    local indent="${1:-}"
    if pack_remote_code_allowed; then
        echo "${indent}trust_remote_code: true"
    else
        echo "${indent}trust_remote_code: false"
    fi
}

_pack_indent_lines() {
    local prefix="$1"
    local content="$2"
    local line
    while IFS= read -r line; do
        # Preserve empty lines as YAML comments to keep structure stable.
        if [[ -z "${line}" ]]; then
            echo "${prefix}#"
        else
            echo "${prefix}${line}"
        fi
    done < <(printf '%s\n' "${content}")
}

pack_render_dataset_provider_yaml() {
    # Emit YAML lines (indented 2 spaces) for the `dataset.provider` field.
    #
    # Output examples:
    # - "  provider: \"wikitext2\""
    # - "  provider:\n    kind: hf_text\n    dataset_name: \"c4\" ..."
    #
    # Inputs:
    # - arg1: provider kind (optional; defaults to INVARLOCK_DATASET).
    # - INVARLOCK_DATASET_PROVIDER_YAML: raw YAML mapping contents (no "provider:" line).
    local kind
    kind="$(pack_dataset_provider_kind "${1:-}")"

    if [[ -n "${INVARLOCK_DATASET_PROVIDER_YAML:-}" ]]; then
        echo "  provider:"
        _pack_indent_lines "    " "${INVARLOCK_DATASET_PROVIDER_YAML}"
        return 0
    fi

    if [[ "${kind}" == "hf_text" ]]; then
        local dataset_name="${INVARLOCK_HF_DATASET_NAME:-${INVARLOCK_HF_DATASET:-allenai/c4}}"
        # Migrate legacy "c4" to "allenai/c4" (script-based c4 deprecated in datasets 4.x)
        if [[ "${dataset_name}" == "c4" ]]; then
            dataset_name="allenai/c4"
        fi
        local config_name="${INVARLOCK_HF_CONFIG_NAME:-${INVARLOCK_HF_DATASET_CONFIG_NAME:-}}"
        if [[ -z "${config_name}" && "${dataset_name}" == "allenai/c4" ]]; then
            config_name="en"
        fi
        local text_field="${INVARLOCK_HF_TEXT_FIELD:-text}"
        local max_samples="${INVARLOCK_HF_MAX_SAMPLES:-2000}"
        local cache_dir="${INVARLOCK_HF_CACHE_DIR:-${HF_DATASETS_CACHE:-}}"
        # trust_remote_code no longer needed for allenai/c4 (Parquet-based)
        local trust_remote_code="${INVARLOCK_HF_TRUST_REMOTE_CODE:-}"

        echo "  provider:"
        echo "    kind: hf_text"
        echo "    dataset_name: \"${dataset_name}\""
        if [[ -n "${config_name}" ]]; then
            echo "    config_name: \"${config_name}\""
        fi
        echo "    text_field: \"${text_field}\""
        if [[ "${max_samples}" =~ ^[0-9]+$ ]]; then
            echo "    max_samples: ${max_samples}"
        else
            echo "    max_samples: 2000"
        fi
        if [[ -n "${trust_remote_code}" ]]; then
            case "$(echo "${trust_remote_code}" | tr '[:upper:]' '[:lower:]' | xargs)" in
                1|true|yes|y|on)
                    if ! pack_remote_code_allowed; then
                        echo "evidence-pack dataset provider remote code requires INVARLOCK_ALLOW_REMOTE_CODE=1" >&2
                        return 2
                    fi
                    echo "    trust_remote_code: true"
                    ;;
                0|false|no|n|off)
                    echo "    trust_remote_code: false"
                    ;;
            esac
        fi
        if [[ -n "${cache_dir}" ]]; then
            echo "    cache_dir: \"${cache_dir}\""
        fi
        return 0
    fi

    if [[ "${kind}" == "local_jsonl" ]]; then
        local file="${INVARLOCK_LOCAL_JSONL_FILE:-}"
        local path="${INVARLOCK_LOCAL_JSONL_PATH:-}"
        local data_files="${INVARLOCK_LOCAL_JSONL_DATA_FILES:-}"
        local text_field="${INVARLOCK_LOCAL_JSONL_TEXT_FIELD:-text}"
        local max_samples="${INVARLOCK_LOCAL_JSONL_MAX_SAMPLES:-2000}"

        echo "  provider:"
        echo "    kind: local_jsonl"
        if [[ -n "${file}" ]]; then
            echo "    file: \"${file}\""
        elif [[ -n "${path}" ]]; then
            echo "    path: \"${path}\""
        elif [[ -n "${data_files}" ]]; then
            echo "    data_files: \"${data_files}\""
        fi
        echo "    text_field: \"${text_field}\""
        if [[ "${max_samples}" =~ ^[0-9]+$ ]]; then
            echo "    max_samples: ${max_samples}"
        else
            echo "    max_samples: 2000"
        fi
        return 0
    fi

    echo "  provider: \"${kind}\""
}
