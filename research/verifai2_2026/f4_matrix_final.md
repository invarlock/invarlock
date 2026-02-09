# F4 Final Run Matrix (Locked)

This file locks the intended F4 experiment matrix for VerifAI-2 once the
end-to-end pipeline is proven.

Primary outcome is the external verifier (HumanEval/MBPP). InvarLock guard
signals are auxiliary features/predictors.

## Global Settings

- InvarLock preset: code-text canary, 256-token windows
  - Remote preset path: `/root/verifai2/f4/presets/code_canary_256.yaml`
- InvarLock profile/tier: `ci` / `balanced`
- Baseline reuse: enabled via `--baseline-report` (requires
  `INVARLOCK_STORE_EVAL_WINDOWS=1` for baseline runs)
- Verifiers: HumanEval (164) and MBPP (500)
- Generation contract:
  - decoding: greedy (`temperature=0`, `top_p=1`, `top_k=0`)
  - `max_new_tokens=256`
  - `seed=42`

## Model Families (6)

| Slug | HF id |
|---|---|
| `phi2` | `microsoft/phi-2` |
| `deepseek_coder_1_3b` | `deepseek-ai/deepseek-coder-1.3b-base` |
| `codegen_350m` | `Salesforce/codegen-350M-mono` |
| `codegen_2b` | `Salesforce/codegen-2B-mono` |
| `qwen2_5_coder_3b` | `Qwen/Qwen2.5-Coder-3B` |
| `qwen2_5_coder_7b` | `Qwen/Qwen2.5-Coder-7B` |

## Edit Variants Per Model (10)

### INT8 RTN (built-in edit, 6 variants)

| Edit slug | Parameters |
|---|---|
| `quant_rtn_int8_all_clamp0` | `scope=all`, `clamp_ratio=0.0` |
| `quant_rtn_int8_all_clamp0p1` | `scope=all`, `clamp_ratio=0.1` |
| `quant_rtn_int8_all_clamp0p25` | `scope=all`, `clamp_ratio=0.25` |
| `quant_rtn_int8_all_clamp0p5` | `scope=all`, `clamp_ratio=0.5` |
| `quant_rtn_int8_ffn_clamp0p25` | `scope=ffn`, `clamp_ratio=0.25` |
| `quant_rtn_int8_attn_clamp0p25` | `scope=attn`, `clamp_ratio=0.25` |

### Magnitude pruning (BYOE checkpoint, 4 variants)

Pruning uses masking/zeroing (tensor shapes preserved) so InvarLock guards and
code generation can run.

| Edit slug | Parameters |
|---|---|
| `prune_mag_s20_all` | `sparsity=0.20`, `scope=all` |
| `prune_mag_s40_all` | `sparsity=0.40`, `scope=all` |
| `prune_mag_s60_all` | `sparsity=0.60`, `scope=all` |
| `prune_mag_s40_ffn` | `sparsity=0.40`, `scope=ffn` |

## Data Products (per model-edit pair)

- `evaluation.report.json` (InvarLock)
- `invarlock verify --json` envelope (CI profile)
- `verifier_trace.v1.json` (HumanEval, baseline + edited)
- `verifier_trace.v1.json` (MBPP, baseline + edited)
- verifier-carrying artifact (schema-valid; references or embeds evidence)

