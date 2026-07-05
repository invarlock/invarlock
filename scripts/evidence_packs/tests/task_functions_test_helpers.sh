#!/usr/bin/env bash

stub_resolve_edit_params() {
    resolve_edit_params() {
        local model_output_dir="$1"
        local edit_spec="$2"
        local version="${3:-}"

        local edit_type param1 param2 scope
        IFS=':' read -r edit_type param1 param2 scope <<< "${edit_spec}"
        if [[ -z "${scope}" && "${edit_type}" != "quant_rtn" ]]; then
            scope="${param2}"
            param2=""
        fi
        if [[ "${edit_type}" == "quant_rtn" && -z "${scope}" ]]; then
            scope="${param2}"
            param2=""
        fi

        local status="selected"
        local edit_dir_name=""
        case "${edit_type}" in
            quant_rtn)
                edit_dir_name="quant_${param1}bit_${version}"
                ;;
            fp8_quant)
                edit_dir_name="fp8_${param1}_${version}"
                ;;
            magnitude_prune)
                local pct
                pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
                edit_dir_name="prune_${pct}pct_${version}"
                ;;
            lowrank_svd)
                edit_dir_name="svd_rank${param1}_${version}"
                ;;
            lora_merge)
                edit_dir_name="lora_rank${param1}_${version}"
                ;;
            fine_tune)
                edit_dir_name="fine_tune_step${param2}_${version}"
                ;;
            *)
                status="invalid"
                ;;
        esac

        jq -n \
            --arg status "${status}" \
            --arg edit_type "${edit_type}" \
            --arg param1 "${param1}" \
            --arg param2 "${param2}" \
            --arg scope "${scope}" \
            --arg version "${version}" \
            --arg edit_dir_name "${edit_dir_name}" \
            '{status:$status, edit_type:$edit_type, param1:$param1, param2:$param2, scope:$scope, version:$version, edit_dir_name:$edit_dir_name}'
    }
}

write_validation_edit_metadata() {
    local edit_path="$1"
    local edit_type="${2:-quant_rtn}"
    local storage_format="float_dequantized"
    case "${edit_type}" in
        magnitude_prune)
            storage_format="dense_float_with_zeros"
            ;;
        lowrank_svd)
            storage_format="dense_float_lowrank_approximated"
            ;;
        lora_merge)
            storage_format="merged_dense_checkpoint"
            ;;
        fine_tune)
            storage_format="fine_tuned_dense_checkpoint"
            ;;
    esac
    cat > "${edit_path}/edit_metadata.json" <<JSON
{
  "schema": "invarlock/evidence-pack-edit-metadata-v1",
  "artifact_class": "validation_subject_checkpoint",
  "edit_type": "${edit_type}",
  "edit_semantics": "external_subject_validation_edit",
  "deployable_as_hf_checkpoint": true,
  "optimized_deployment_backend": false,
  "backend": null,
  "storage_format": "${storage_format}",
  "actual_storage_format": "${storage_format}",
  "packed_quantized_storage": false,
  "runtime_memory_reduction": false,
  "scope": "ffn",
  "parameters": {},
  "coverage": {
    "edited_tensors": 1,
    "edited_params": 1,
    "total_params": 1,
    "coverage_ratio": 1.0
  }
}
JSON
}

write_minimal_validation_edit_artifact() {
    local edit_path="$1"
    local edit_type="${2:-quant_rtn}"
    mkdir -p "${edit_path}"
    echo "{}" > "${edit_path}/config.json"
    echo "weights" > "${edit_path}/pytorch_model.bin"
    echo "{}" > "${edit_path}/tokenizer_config.json"
    write_validation_edit_metadata "${edit_path}" "${edit_type}"
}

write_minimal_evaluate_baseline_report() {
    local report_path="$1"
    local seq_len="${2:-128}"
    local stride="${3:-128}"
    local preview_n="${4:-192}"
    local final_n="${5:-192}"
    cat > "${report_path}" <<JSON
{
  "data": {
    "seq_len": ${seq_len},
    "stride": ${stride},
    "preview_n": ${preview_n},
    "final_n": ${final_n}
  },
  "evaluation_windows": {
    "preview": {"window_ids": [1], "input_ids": [[1]]},
    "final": {"window_ids": [1], "input_ids": [[1]]}
  },
  "edit": {"name": "noop"}
}
JSON
}
