#!/usr/bin/env bash
# model_creation.sh - Shared model creation helpers for workers and main script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"

# Provide basic logging helpers when not sourced from the main script.
if ! declare -F log >/dev/null 2>&1; then
    :
    log() {
        echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] $*"
    }
fi

if ! declare -F error_exit >/dev/null 2>&1; then
    :
    error_exit() {
        echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
        exit 1
    }
fi

_model_creation_run_python() {
    local parent_dir="$1"
    local cuda_devices="$2"
    shift 2

    if ! mkdir -p "${parent_dir}"; then
        return 1
    fi

    CUDA_VISIBLE_DEVICES="${cuda_devices}" _cmd_python "$@"
}

create_model_variant() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local param1="${4:-}"
    local param2="${5:-}"
    local scope="${6:-}"
    local gpu_id="${7:-0}"

    [[ "${param1}" == "null" ]] && param1=""
    [[ "${param2}" == "null" ]] && param2=""
    [[ "${scope}" == "null" ]] && scope=""

    case "${edit_type}" in
        "quant_rtn")
            if [[ -z "${param1}" || -z "${param2}" || -z "${scope}" ]]; then
                echo "ERROR: quant_rtn requires bits, group_size, scope" >&2
                return 1
            fi
            create_edited_model "${baseline_path}" "${output_path}" "quant_rtn" "${param1}" "${param2}" "${scope}" "${gpu_id}"
            ;;
        "fp8_quant")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: fp8_quant requires format and scope" >&2
                return 1
            fi
            create_fp8_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "magnitude_prune")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: magnitude_prune requires sparsity and scope" >&2
                return 1
            fi
            create_pruned_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "lowrank_svd")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: lowrank_svd requires rank and scope" >&2
                return 1
            fi
            create_lowrank_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "error_injection")
            if [[ -z "${param1}" ]]; then
                echo "ERROR: error_injection requires error_type" >&2
                return 1
            fi
            create_error_model "${baseline_path}" "${output_path}" "${param1}" "${gpu_id}"
            ;;
        *)
            echo "ERROR: Unknown edit type: ${edit_type}" >&2
            return 1
            ;;
    esac
}
export -f create_model_variant

create_edited_model() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local bits="$4"
    local group_size="$5"
    local scope="$6"
    local gpu_id="${7:-0}"

    log "Creating edited model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Edit: ${edit_type} bits=${bits} group_size=${group_size} scope=${scope}"

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        local parent_dir
        parent_dir="$(dirname "${output_path}")"
        local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python "${parent_dir}" "${cuda_devices}" - "${baseline_path}" "${output_path}" "${bits}" "${group_size}" "${scope}" <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import os
import sys

try:
    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    baseline_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    bits = int(sys.argv[3])
    group_size = int(sys.argv[4])
    scope = sys.argv[5]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    flash_available = os.environ.get("FLASH_ATTENTION_AVAILABLE", "false") == "true"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if flash_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)

    @torch.no_grad()
    def round_to_nearest_gpu(tensor, bits, group_size):
        """Group-wise RTN quantization (per-output-channel groups along input dim)."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        orig_shape = tensor.shape
        flat = tensor.reshape(orig_shape[0], -1)
        in_features = flat.shape[1]
        if group_size <= 0 or group_size >= in_features:
            group_size = in_features
        num_groups = (in_features + group_size - 1) // group_size
        pad = (num_groups * group_size) - in_features
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        grouped = flat.reshape(orig_shape[0], num_groups, group_size)
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=1e-10)
        quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
        quantized = quantized.reshape(orig_shape[0], num_groups * group_size)
        if pad > 0:
            quantized = quantized[:, :in_features]
        return quantized.reshape(orig_shape).to(tensor.dtype)

    def should_quantize(name, scope):
        """Check if parameter should be quantized based on name and scope.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        - Generic: linear, dense, proj, fc, mlp, attn
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "dense", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"  # MPT/Falcon attention
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "dense", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon attention
            ])
        return False

    print(f"Quantizing to {bits}-bit on GPU (scope={scope})...")
    quantized_count = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if should_quantize(name, scope) and param.dim() >= 2:
            param.data = round_to_nearest_gpu(param.data, bits, group_size)
            quantized_count += 1
            edited_params += param.numel()
            if quantized_count <= 3:
                print(f"  Quantized: {name} ({param.shape})")

    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Quantized {quantized_count} parameters ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "quant_rtn",
        "bits": bits,
        "group_size": group_size,
        "scope": scope,
        "quantized_params": quantized_count
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved edited model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create edited model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
    else
        error_exit "Unknown edit type: ${edit_type}"
    fi
}
export -f create_edited_model

# ============ MAGNITUDE PRUNING ============
create_pruned_model() {
    local baseline_path="$1"
    local output_path="$2"
    local sparsity="$3"  # 0.1 for clean, 0.5 for stress
    local scope="$4"     # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating pruned model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Sparsity: ${sparsity}, Scope: ${scope}"

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python "${parent_dir}" "${cuda_devices}" - "${baseline_path}" "${output_path}" "${sparsity}" "${scope}" <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    sparsity = float(sys.argv[3])
    scope = sys.argv[4]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def should_prune(name, scope):
        """Check if parameter should be pruned based on name and scope.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon
            ])
        return False

    @torch.no_grad()
    def magnitude_prune(weight, sparsity):
        """Set smallest magnitude weights to zero."""
        flat = weight.abs().flatten()
        k = int(flat.numel() * sparsity)
        if k == 0:
            return weight
        threshold = torch.kthvalue(flat, k).values
        mask = weight.abs() >= threshold
        return weight * mask.to(weight.dtype)

    print(f"Pruning with sparsity={sparsity} (scope={scope})...")
    pruned_count = 0
    total_zeros = 0
    total_edited_params = 0
    total_model_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        if should_prune(name, scope) and param.dim() >= 2:
            original_zeros = (param == 0).sum().item()
            param.data = magnitude_prune(param.data, sparsity)
            new_zeros = (param == 0).sum().item()
            pruned_count += 1
            total_zeros += new_zeros
            total_edited_params += param.numel()
            if pruned_count <= 3:
                print(f"  Pruned: {name} ({original_zeros} -> {new_zeros} zeros)")

    actual_sparsity = total_zeros / total_edited_params if total_edited_params > 0 else 0
    coverage_pct = 100.0 * total_edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Pruned {pruned_count} parameters ({total_edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Actual sparsity within edited params: {actual_sparsity:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "magnitude_prune",
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "scope": scope,
        "pruned_params": pruned_count
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved pruned model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create pruned model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
}
export -f create_pruned_model

# ============ LOW-RANK SVD APPROXIMATION ============
create_lowrank_model() {
    local baseline_path="$1"
    local output_path="$2"
    local rank="$3"      # 256 for clean, 32 for stress
    local scope="$4"     # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating low-rank model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Rank: ${rank}, Scope: ${scope}"

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python "${parent_dir}" "${cuda_devices}" - "${baseline_path}" "${output_path}" "${rank}" "${scope}" <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    rank = int(sys.argv[3])
    scope = sys.argv[4]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def _parse_scope(raw_scope: str):
        base = (raw_scope or "").strip()
        layer_limit = None
        layer_exact = None
        if "@" in base:
            base, rest = base.split("@", 1)
            base = base.strip()
            for item in (s.strip() for s in rest.split(",") if s.strip()):
                if item.startswith("layers="):
                    try:
                        layer_limit = int(item.split("=", 1)[1])
                    except Exception:
                        layer_limit = None
                elif item.startswith("layer="):
                    try:
                        layer_exact = int(item.split("=", 1)[1])
                    except Exception:
                        layer_exact = None
        return base, layer_limit, layer_exact

    base_scope, layer_limit, layer_exact = _parse_scope(scope)
    if base_scope != scope:
        print(
            f"Parsed scope={scope} -> base_scope={base_scope}, layer_limit={layer_limit}, layer={layer_exact}"
        )

    def _extract_layer_index(name: str):
        marker = ".layers."
        pos = name.find(marker)
        if pos < 0:
            return None
        start = pos + len(marker)
        end = start
        while end < len(name) and name[end].isdigit():
            end += 1
        if end == start:
            return None
        try:
            return int(name[start:end])
        except Exception:
            return None

    def should_lowrank(name, base_scope):
        """Check if parameter should have low-rank approximation.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        """
        name_lower = name.lower()
        if base_scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif base_scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif base_scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon
            ])
        return False

    @torch.no_grad()
    def truncated_svd(weight, rank):
        """Apply truncated SVD to approximate weight matrix using randomized algorithm.

        Uses torch.svd_lowrank for efficiency on large matrices:
        - Full SVD: O(n^3) time, OOM risk on large weights
        - Randomized SVD: O(n^2 * rank) time, memory-efficient
        """
        if weight.dim() < 2:
            return weight

        original_shape = weight.shape
        weight_2d = weight.view(weight.shape[0], -1).float()

        max_rank = min(weight_2d.shape)
        effective_rank = min(rank, max_rank)

        # Use randomized SVD (O(n^2 * rank)) instead of full SVD (O(n^3))
        # niter=2 provides good accuracy while staying fast
        # q parameter is the target rank
        U, S, V = torch.svd_lowrank(weight_2d, q=effective_rank, niter=2)

        # Reconstruct: (U * S) @ V^T (avoid materializing diag(S))
        lowrank = (U * S) @ V.T
        return lowrank.to(weight.dtype).view(original_shape)

    print(f"Applying low-rank SVD with rank={rank} (scope={scope})...")
    modified_count = 0
    total_energy_retained = 0
    num_matrices = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if layer_limit is not None or layer_exact is not None:
            idx = _extract_layer_index(name)
            if idx is None:
                continue
            if layer_exact is not None and idx != layer_exact:
                continue
            if layer_limit is not None and idx >= layer_limit:
                continue

        if should_lowrank(name, base_scope) and param.dim() >= 2:
            original_norm = param.data.norm()
            param.data = truncated_svd(param.data, rank)
            new_norm = param.data.norm()
            energy_retained = (new_norm / original_norm).item() if original_norm > 0 else 1.0
            modified_count += 1
            total_energy_retained += energy_retained
            num_matrices += 1
            edited_params += param.numel()
            if modified_count <= 3:
                print(f"  Low-rank: {name}, energy retained: {energy_retained:.4f}")

    avg_energy = total_energy_retained / num_matrices if num_matrices > 0 else 1.0
    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Modified {modified_count} matrices ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Average energy retained: {avg_energy:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "lowrank_svd",
        "rank": rank,
        "scope": scope,
        "modified_matrices": modified_count,
        "avg_energy_retained": avg_energy
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved low-rank model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create low-rank model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
}
export -f create_lowrank_model

# ============ FP8 QUANTIZATION (SIMULATED) ============
create_fp8_model() {
    local baseline_path="$1"
    local output_path="$2"
    local format="$3"      # e4m3fn or e5m2
    local scope="$4"       # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating FP8 model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Format: ${format}, Scope: ${scope}"

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python "${parent_dir}" "${cuda_devices}" - "${baseline_path}" "${output_path}" "${format}" "${scope}" <<'PY'
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

baseline_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
format_type = sys.argv[3]
scope = sys.argv[4]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print(f"Loading baseline from {baseline_path}...")
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "trust_remote_code": True,
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}
model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)

def should_quantize(name, scope):
    name_lower = name.lower()
    if scope == "all":
        return "weight" in name_lower and any(x in name_lower for x in [
            "linear", "proj", "fc", "mlp", "attn",
            "wqkv", "query_key_value"
        ])
    if scope == "ffn":
        return "weight" in name_lower and any(x in name_lower for x in [
            "mlp", "fc", "gate", "up_proj", "down_proj",
            "dense_h_to_4h", "dense_4h_to_h"
        ])
    if scope == "attn":
        return "weight" in name_lower and any(x in name_lower for x in [
            "attn", "q_proj", "k_proj", "v_proj", "o_proj",
            "wqkv", "out_proj", "query_key_value"
        ])
    return False

if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
else:
    fp8_dtype = getattr(torch, "float8_e5m2", None)

if fp8_dtype is None:
    print("WARNING: torch float8 dtype not available; falling back to float16 quantization")

@torch.no_grad()
def quantize_fp8(tensor):
    if fp8_dtype is None:
        return tensor.to(torch.float16).to(tensor.dtype)
    return tensor.to(fp8_dtype).to(tensor.dtype)

print(f"Applying FP8 quantization (format={format_type}, scope={scope})...")
quantized_count = 0
num_tensors = 0
rel_error_total = 0.0
edited_params = 0
for name, param in model.named_parameters():
    if not should_quantize(name, scope) or param.dim() < 2:
        continue
    original = param.data.clone()
    param.data = quantize_fp8(param.data)
    num_tensors += 1
    quantized_count += 1
    edited_params += param.numel()
    denom = original.abs().mean() + 1e-10
    rel_error_total += float((param.data - original).abs().mean() / denom)
    if quantized_count <= 3:
        print(f"  FP8: {name}")

avg_error = rel_error_total / max(num_tensors, 1)
print(f"Quantized {quantized_count} tensors, avg relative error: {avg_error:.4f}")

model = model.cpu()
gc.collect()
torch.cuda.empty_cache()

output_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

metadata = {
    "edit_type": "fp8_quant",
    "format": format_type,
    "scope": scope,
    "quantized_tensors": quantized_count,
    "avg_relative_error": avg_error,
}
with open(output_path / "edit_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved FP8-quantized model to {output_path}")
PY
}
export -f create_fp8_model

# ============ ERROR MODEL CREATION ============
create_error_model() {
    local baseline_path="$1"
    local output_path="$2"
    local error_type="$3"
    local gpu_id="${4:-0}"

    log "Creating error model (type=${error_type}, GPU ${gpu_id})"
    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python "${parent_dir}" "${cuda_devices}" - "${baseline_path}" "${output_path}" "${error_type}" <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    error_type = sys.argv[3]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)

    # Use GPU for error injection when possible (handles large models better)
    # Fall back to CPU for small models or if GPU has issues
    try:
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        use_gpu = True
    except Exception as gpu_err:
        print(f"GPU loading failed ({gpu_err}), falling back to CPU (may be slow for large models)")
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        use_gpu = False

    error_info = {"error_type": error_type, "injected": False}

    # Build list of transformer blocks for index-based targeting
    # This works across architectures (LLaMA, MPT, Falcon, Qwen, etc.)
    import re
    block_params = {}  # {block_idx: [(name, param), ...]}
    block_pattern = re.compile(r'(?:layers|blocks|h)\.(\d+)\.')

    for name, param in model.named_parameters():
        match = block_pattern.search(name)
        if match:
            block_idx = int(match.group(1))
            if block_idx not in block_params:
                block_params[block_idx] = []
            block_params[block_idx].append((name, param))

    num_blocks = max(block_params.keys()) + 1 if block_params else 0
    first_block = 0
    middle_block = num_blocks // 2 if num_blocks > 1 else 0

    print(f"Detected {num_blocks} transformer blocks")

    if error_type == "nan_injection":
        # Target first block - works across architectures
        target_block = first_block
        for name, param in block_params.get(target_block, []):
            if 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data[0, 0] = float('nan')
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["target_block"] = target_block
                print(f"Injected NaN into: {name} (block {target_block})")
                break

    elif error_type == "inf_injection":
        # Target attention in first block
        for name, param in model.named_parameters():
            if 'attn' in name.lower() and 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data[0, 0] = float('inf')
                error_info["injected"] = True
                error_info["target_param"] = name
                print(f"Injected Inf into: {name}")
                break

    elif error_type == "extreme_quant":
        def extreme_quant(tensor):
            qmin, qmax = -2, 1
            scale = tensor.abs().max() / max(abs(qmin), abs(qmax))
            scale = torch.clamp(scale, min=1e-10)
            quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
            return (quantized * scale).to(tensor.dtype)

        count = 0
        for name, param in model.named_parameters():
            if 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data = extreme_quant(param.data)
                    count += 1
        error_info["injected"] = True
        error_info["quantized_params"] = count
        print(f"Applied extreme 2-bit quantization to {count} params")

    elif error_type == "shape_mismatch":
        # Resize token embeddings without updating tokenizer files. This should
        # trip invariants' tokenizer/vocab alignment checks during certify.
        try:
            emb = model.get_input_embeddings()
            old_vocab = int(getattr(emb, "num_embeddings", emb.weight.shape[0]))
            delta = 8
            new_vocab = old_vocab + delta
            model.resize_token_embeddings(new_vocab)
            error_info["injected"] = True
            error_info["old_vocab_size"] = old_vocab
            error_info["new_vocab_size"] = int(new_vocab)
            error_info["delta"] = int(delta)
            print(f"Resized token embeddings: {old_vocab} -> {new_vocab}")
        except Exception as e:
            print(f"WARNING: shape_mismatch not injected ({e})")

    elif error_type == "missing_tensors":
        # Simulate a missing-checkpoint scenario by shrinking the transformer
        # stack (drop the final block) and updating config accordingly.
        def _shrink_layers(container, attr):
            layers = getattr(container, attr, None)
            if layers is None:
                return False, 0, 0
            try:
                total = len(layers)
            except Exception:
                return False, 0, 0
            if total < 2:
                return False, total, total
            keep = total - 1
            try:
                if isinstance(layers, torch.nn.ModuleList):
                    new_layers = torch.nn.ModuleList(list(layers)[:keep])
                else:
                    new_layers = list(layers)[:keep]
                setattr(container, attr, new_layers)
                return True, total, keep
            except Exception:
                return False, total, total

        injected = False
        total_layers = 0
        kept_layers = 0

        # LLaMA/Mistral-style
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "layers"):
            injected, total_layers, kept_layers = _shrink_layers(base, "layers")
            if injected:
                error_info["arch"] = "llama"

        # GPT-2-style
        if not injected:
            tr = getattr(model, "transformer", None)
            if tr is not None and hasattr(tr, "h"):
                injected, total_layers, kept_layers = _shrink_layers(tr, "h")
                if injected:
                    error_info["arch"] = "gpt2"

        if injected:
            cfg = getattr(model, "config", None)
            for key in ("num_hidden_layers", "n_layer", "num_layers"):
                if cfg is not None and hasattr(cfg, key):
                    try:
                        setattr(cfg, key, int(kept_layers))
                    except Exception:
                        pass
            error_info["injected"] = True
            error_info["dropped_layers"] = int(total_layers - kept_layers)
            error_info["layers_before"] = int(total_layers)
            error_info["layers_after"] = int(kept_layers)
            print(f"Dropped transformer blocks: {total_layers} -> {kept_layers}")
        else:
            print("WARNING: missing_tensors not injected (no layer stack found)")

    elif error_type == "scale_explosion":
        # Target MLP/FFN in first block
        for name, param in model.named_parameters():
            if 'mlp' in name.lower() and 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data = param.data * 100.0
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["scale_factor"] = 100.0
                print(f"Scaled by 100x: {name}")
                break

    elif error_type == "rank_collapse":
        # Force multiple weight matrices to become rank-1 to trigger the spectral
        # degeneracy checks (stable-rank drop) across > max_caps modules.
        target_names = []
        patterns = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "c_attn.weight",
            "c_proj.weight",
            "out_proj.weight",
            "query_key_value.weight",
        )
        for name, param in model.named_parameters():
            if len(target_names) >= 8:
                break
            if param.dim() != 2 or "weight" not in name.lower():
                continue
            lname = name.lower()
            if any(p in lname for p in patterns):
                target_names.append((name, param))

        if not target_names:
            for name, param in model.named_parameters():
                if param.dim() == 2 and "weight" in name.lower():
                    target_names.append((name, param))
                if len(target_names) >= 8:
                    break

        applied = 0
        for name, param in target_names:
            with torch.no_grad():
                w = param.data
                if w.numel() < 4:
                    continue
                u = w[:, 0].clone()
                v = w[0, :].clone()
                w_new = u.unsqueeze(1) * v.unsqueeze(0)
                # Preserve Frobenius norm to avoid pure scale effects.
                denom = torch.norm(w_new) + 1e-12
                scale = torch.norm(w) / denom
                w.copy_(w_new * scale)
            applied += 1
            if applied <= 3:
                print(f"Rank-collapsed: {name}")

        if applied:
            error_info["injected"] = True
            error_info["rank_collapsed_params"] = applied
            error_info["targets"] = [n for n, _ in target_names[:applied]]
            print(f"Applied rank collapse to {applied} weight matrices")
        else:
            print("WARNING: rank_collapse not injected (no eligible weights found)")

    elif error_type == "norm_collapse":
        # Zero out entire matrices across multiple weight matrices to trigger
        # norm-collapse degeneracy (and typically a primary-metric failure)
        # across > max_caps modules.
        target_names = []
        patterns = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "c_attn.weight",
            "c_proj.weight",
            "out_proj.weight",
            "query_key_value.weight",
        )
        for name, param in model.named_parameters():
            if len(target_names) >= 32:
                break
            if param.dim() != 2 or "weight" not in name.lower():
                continue
            lname = name.lower()
            if any(p in lname for p in patterns):
                target_names.append((name, param))

        if not target_names:
            for name, param in model.named_parameters():
                if param.dim() == 2 and "weight" in name.lower():
                    target_names.append((name, param))
                if len(target_names) >= 32:
                    break

        applied = 0
        for name, param in target_names:
            with torch.no_grad():
                w = param.data
                if w.numel() < 4:
                    continue
                w.zero_()
            applied += 1
            if applied <= 3:
                print(f"Zeroed matrix: {name}")

        if applied:
            error_info["injected"] = True
            error_info["norm_collapsed_params"] = applied
            error_info["targets"] = [n for n, _ in target_names[:applied]]
            print(f"Applied norm collapse to {applied} weight matrices")
        else:
            print("WARNING: norm_collapse not injected (no eligible weights found)")

    elif error_type == "weight_tying_break":
        def _data_ptr(t):
            try:
                return int(t.data_ptr())
            except Exception:
                return None

        def _try_flip_tying(embed_weight, head_weight, label):
            if embed_weight is None or head_weight is None:
                return False
            embed_ptr = _data_ptr(embed_weight)
            head_ptr = _data_ptr(head_weight)
            if embed_ptr is None or head_ptr is None:
                return False

            try:
                is_tied = embed_ptr == head_ptr
            except Exception:
                is_tied = False

            cfg = getattr(model, "config", None)
            with torch.no_grad():
                if is_tied:
                    # Untie by cloning the head weight.
                    model.lm_head.weight = torch.nn.Parameter(head_weight.detach().clone())
                    if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
                        cfg.tie_word_embeddings = False
                    error_info["mode"] = "untie"
                else:
                    # Tie by re-aliasing head weight to embeddings.
                    model.lm_head.weight = embed_weight
                    if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
                        cfg.tie_word_embeddings = True
                    error_info["mode"] = "tie"

            error_info["injected"] = True
            error_info["target"] = label
            error_info["embed_ptr_before"] = embed_ptr
            error_info["head_ptr_before"] = head_ptr
            error_info["embed_ptr_after"] = _data_ptr(embed_weight)
            error_info["head_ptr_after"] = _data_ptr(getattr(getattr(model, "lm_head", None), "weight", None))
            print(f"Flipped weight tying ({label}): {error_info['mode']}")
            return True

        injected = False

        # LLaMA/Mistral style (model.embed_tokens <-> lm_head)
        try:
            llama_model = getattr(model, "model", None)
            embed_tokens = getattr(llama_model, "embed_tokens", None)
            injected = _try_flip_tying(getattr(embed_tokens, "weight", None), getattr(getattr(model, "lm_head", None), "weight", None), "llama")
        except Exception:
            injected = False

        # GPT-2 style (transformer.wte <-> lm_head)
        if not injected:
            try:
                transformer = getattr(model, "transformer", None)
                wte = getattr(transformer, "wte", None)
                injected = _try_flip_tying(getattr(wte, "weight", None), getattr(getattr(model, "lm_head", None), "weight", None), "gpt2")
            except Exception:
                injected = False

        if not injected:
            print("WARNING: Could not locate tied weights; weight_tying_break not injected")

    # Move to CPU for saving if loaded on GPU
    if use_gpu:
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    with open(output_path / "error_metadata.json", 'w') as f:
        json.dump(error_info, f, indent=2)

    del model
    gc.collect()
    print(f"Saved error model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create error model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
}
export -f create_error_model
