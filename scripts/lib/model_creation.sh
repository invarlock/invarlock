#!/usr/bin/env bash
# model_creation.sh - Shared model creation helpers for workers and main script

# Provide basic logging helpers when not sourced from the main script.
if ! type log &>/dev/null; then
    log() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    }
fi

if ! type error_exit &>/dev/null; then
    error_exit() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
        exit 1
    }
fi

create_edited_model() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local bits="$4"
    local group_size="$5"
    local scope="$6"
    local gpu_id="${7:-0}"

    log "Creating edited model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Edit: ${edit_type} bits=${bits} group_size=${group_size} scope=${scope}"

    mkdir -p "$(dirname "${output_path}")"

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
        CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import os
import sys

try:
    mode = os.environ.get("B200_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    bits = int("${bits}")
    group_size = int("${group_size}")
    scope = "${scope}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    flash_available = "${FLASH_ATTENTION_AVAILABLE}" == "true"

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
EOF
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

    log "Creating pruned model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Sparsity: ${sparsity}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    sparsity = float("${sparsity}")
    scope = "${scope}"

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
EOF
}
export -f create_pruned_model

# ============ LOW-RANK SVD APPROXIMATION ============
create_lowrank_model() {
    local baseline_path="$1"
    local output_path="$2"
    local rank="$3"      # 256 for clean, 32 for stress
    local scope="$4"     # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating low-rank model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Rank: ${rank}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    rank = int("${rank}")
    scope = "${scope}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def should_lowrank(name, scope):
        """Check if parameter should have low-rank approximation.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
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
        if should_lowrank(name, scope) and param.dim() >= 2:
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
EOF
}
export -f create_lowrank_model

# ============ FP4 QUANTIZATION (SIMULATED) ============
create_fp4_model() {
    local baseline_path="$1"
    local output_path="$2"
    local format="$3"      # e2m1 (standard) or aggressive
    local scope="$4"       # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating FP4 model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Format: ${format}, Scope: ${scope}"
    log "  FP4 Native Support: ${FP4_NATIVE_SUPPORT}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import os
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    format_type = "${format}"
    scope = "${scope}"

    # Check for B200 FP4 support
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device_name = torch.cuda.get_device_name(0)
    is_b200 = "B200" in device_name or "Blackwell" in device_name

    require_native = os.environ.get("INVARLOCK_REQUIRE_FP4_NATIVE", "false").strip().lower() in ("1", "true", "yes")
    te_available = False
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        te_available = True
    except Exception as e:
        if require_native:
            raise RuntimeError("TransformerEngine not available for native FP4 validation") from e

    fp4_native = bool(is_b200 and te_available)

    if not fp4_native:
        if is_b200:
            print("WARNING: B200 detected but TransformerEngine not available.")
            print("         FP4 quantization is simulated; no FP4 Tensor Core validation.")
        else:
            print(f"WARNING: FP4 is B200-native, current GPU: {device_name}")
            print("         FP4 quantization is simulated; results may not match true B200 behavior")

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def should_quantize(name, scope):
        """Check if parameter should be FP4 quantized.

        Supports multiple architectures (LLaMA, MPT, Falcon).
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
    def fp4_quantize(tensor, format_type):
        """
        FP4 quantization (E2M1 or aggressive).

        E2M1 format: 2 exponent bits, 1 mantissa bit
        Range: [-6, 6] with 7 distinct magnitudes + zero

        Aggressive: tighter clipping for stress testing
        """
        # FP4 E2M1 representable values (approximate)
        if format_type == "e2m1":
            # Standard E2M1: {0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}
            levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device, dtype=torch.float16)
        else:
            # Aggressive: tighter range for stress testing
            levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], device=tensor.device, dtype=torch.float16)

        # Memory-safe nearest-level quantization via bucketize (no NxK diff matrix).
        max_val = float(levels[-1].item())
        scale = tensor.abs().amax().float() / max_val
        scale = torch.clamp(scale, min=1e-10).to(device=tensor.device, dtype=torch.float16)

        thresholds = ((levels[:-1] + levels[1:]) / 2).to(device=tensor.device)

        flat = tensor.view(-1)
        n = flat.numel()
        chunk_elems = 5_000_000  # Bound peak temp memory (idx is int64)
        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = flat[start:end]
            scaled = chunk.to(torch.float16) / scale
            idx = torch.bucketize(scaled.abs(), thresholds)
            q = levels[idx] * scaled.sign()
            chunk.copy_((q * scale).to(chunk.dtype))

        return tensor

    print(f"Applying FP4 quantization (format={format_type}, scope={scope})...")
    quantized_count = 0
    total_error = 0
    num_tensors = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if should_quantize(name, scope) and param.dim() >= 2:
            original = param.data.clone()
            param.data = fp4_quantize(param.data, format_type)

            # Compute relative error
            error = (param.data - original).abs().mean() / (original.abs().mean() + 1e-10)
            total_error += error.item()
            quantized_count += 1
            num_tensors += 1
            edited_params += param.numel()

            if quantized_count <= 3:
                print(f"  FP4: {name}, rel_error: {error.item():.4f}")

    avg_error = total_error / num_tensors if num_tensors > 0 else 0
    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Quantized {quantized_count} tensors ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Average relative error: {avg_error:.4f}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "fp4_quant",
        "format": format_type,
        "scope": scope,
        "quantized_tensors": quantized_count,
        "avg_relative_error": avg_error,
        "b200_native": is_b200,
        "fp4_native": fp4_native,
        "transformer_engine": te_available
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved FP4-quantized model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create FP4 model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_fp4_model

# ============ ERROR MODEL CREATION ============
create_error_model() {
    local baseline_path="$1"
    local output_path="$2"
    local error_type="$3"
    local gpu_id="${4:-0}"

    log "Creating error model (type=${error_type}, GPU ${gpu_id})"
    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    error_type = "${error_type}"

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

    elif error_type == "zero_layer":
        # Target middle block - architecture agnostic
        target_block = middle_block
        for name, param in block_params.get(target_block, []):
            if 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data.zero_()
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["target_block"] = target_block
                print(f"Zeroed: {name} (block {target_block})")
                break

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
EOF
}
export -f create_error_model
