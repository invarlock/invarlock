# B200 Validation Suite

The InvarLock B200 Validation Suite (`scripts/b200_validation_suite.sh`) is a comprehensive validation harness optimized for 8x NVIDIA B200 180GB SXM6 GPUs. It validates InvarLock's edit detection capabilities across a diverse model suite spanning 7B to 72B parameters.

## Overview

| Aspect | Details |
|--------|---------|
| **Purpose** | Definitive Phase 0 validation for InvarLock edit detection |
| **Hardware** | 8× NVIDIA B200 180GB SXM6 (~4.5 TB/s bandwidth, ~2250 FP16 TFLOPS) |
| **Models** | 8 public models (7B–72B), no HuggingFace login required |
| **Edit Types** | 4 types × 2 versions (clean + stress) = 8 edits per model |
| **Scheduling** | Dynamic work-stealing with profile-based multi-GPU reservations |
| **Multi-GPU** | Profile-based planning; tasks scale to 2+ GPUs only when required memory exceeds per-GPU capacity |
| **Version** | v2.1.0-b200 |

## Quick Start

```bash
# Run validation suite
./scripts/b200_validation_suite.sh

# Resume a failed or interrupted run
OUTPUT_DIR=./invarlock_validation_b200_20241208_123456 \
  ./scripts/b200_validation_suite.sh --resume

# Optional: split calibration from the rest of the suite
# 1) Run calibration + preset generation only (checkpoint)
OUTPUT_DIR=./invarlock_validation_b200_20241208_123456 \
  ./scripts/b200_validation_suite.sh --calibrate-only

# 2) Continue from the calibration checkpoint (implies --resume when a queue exists)
OUTPUT_DIR=./invarlock_validation_b200_20241208_123456 \
  ./scripts/b200_validation_suite.sh --run-only
```text
                      ┌────────────────────┐

                      │  SETUP_BASELINE    │  Priority: 90
                      │  (download model)  │
                      └─────────┬──────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐      ┌────────────────┐      ┌─────────────────────┐
 │EVAL_BASELINE│      │CALIBRATION_RUN │      │ CREATE_EDITS_BATCH  │
 │ (lm-eval)   │      │    × 5 runs    │      │ (8 edits, 1 load)   │
 │ pri:80      │      │    pri:85      │      │ pri:70              │
 └─────────────┘      └────────┬───────┘      └─────────┬───────────┘
                               │                        │
                               ▼                        │
                      ┌────────────────┐                │
                      │GENERATE_PRESET │                │
                      │ (aggregate)    │                │
                      └────────┬───────┘                │
                               │                        │
                               │       ┌────────────────┘
                               │       │
                               │       ▼ (Split Eval: 4× parallel per edit)
                               │   ┌───────────────────────────────────┐
                               │   │ EVAL_MMLU ─────┐                  │
                               │   │ EVAL_HELLASWAG ├─ Can run in      │
                               │   │ EVAL_ARC ──────┤  parallel on     │
                               │   │ EVAL_WINOGRANDE┘  different GPUs  │
                               │   │      × 8 edits = 32 tasks         │
                               │   └───────────────────────────────────┘
                               │                        │
                               └──────────┬─────────────┘
                                          ▼
                                ┌──────────────────────────────┐
                                │        CERTIFY_EDIT          │
                                │ × 3/2 runs per edit (clean/   │
                                │ stress; depends on batch +    │
                                │ preset only)                  │
                                └──────────────────────────────┘
```

**Key Optimization Notes**:

- `CREATE_EDITS_BATCH` loads the model once and creates all 8 edits (vs 8 separate loads)
- Split eval tasks (`EVAL_MMLU`, `EVAL_HELLASWAG`, `EVAL_ARC`, `EVAL_WINOGRANDE`) can run on different GPUs
- `CERTIFY_EDIT` depends on `CREATE_EDITS_BATCH` and `GENERATE_PRESET`, but NOT on eval tasks
- Eval tasks run in parallel with certification tasks, maximizing GPU utilization

For 70B+ per-edit mode: `CREATE_EDIT → EVAL_EDIT → CERTIFY_EDIT` (no split eval tasks).

The following shows the per-edit dependency graph for 70B+ models:

```text
                      ┌────────────────────┐
                      │  SETUP_BASELINE    │  Priority: 90
                      │  (download model)  │
                      └─────────┬──────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐      ┌────────────────┐      ┌─────────────────────┐
 │EVAL_BASELINE│      │CALIBRATION_RUN │      │    CREATE_EDIT      │
 │ (lm-eval)   │      │    × 5 runs    │      │ (per-edit)          │
 │ pri:80      │      │    pri:85      │      │ pri:70              │
 └─────────────┘      └────────┬───────┘      └─────────┬───────────┘
                               │                        │
                               ▼                        ▼
                      ┌────────────────┐      ┌─────────────────────┐
                      │GENERATE_PRESET │      │      EVAL_EDIT      │
                      │ (aggregate)    │      │ (all benchmarks)    │
                      └────────┬───────┘      └─────────┬───────────┘
                               │                        │
                               └──────────┬─────────────┘
                                          ▼
                                ┌──────────────────────────────┐
                                │        CERTIFY_EDIT          │
                                │ × 3/2 runs per edit (clean/   │
                                │ stress; depends on create +   │
                                │ preset only)                  │
                                └──────────────────────────────┘
```

### GPU Worker Loop

Each GPU runs a continuous worker loop:

```text
                ┌─────────────────────────────┐
                │      START gpu_worker       │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │              ┌───────────────┐               │
        │              │ shutdown?     │──Yes──► EXIT  │
        │              └───────┬───────┘               │
        │                 No   │                       │
        │                      ▼                       │
        │              ┌───────────────┐               │
        │              │ query GPU     │               │
        │              │ free memory   │               │
        │              └───────┬───────┘               │
        │                      │                       │
        │                      ▼                       │
        │        ┌──────────────────────────────┐      │
        │        │ find_and_claim_task()       │      │
        │        │ • select + reserve under    │      │
        │        │   scheduler lock            │      │
        │        └──────────────┬──────────────┘      │
        │                     │                        │
        │          found? No  │  Yes                   │
        │                ┌────┴────┐                   │
        │                │         │                   │
        │                ▼         ▼                   │
        │         ┌──────────┐  ┌──────────────┐       │
        │         │ sleep(5s)│  │ task claimed │       │
        │         └────┬─────┘  │ (reserved)   │       │
        │              │        └──────┬───────┘       │
        │              │               │               │
        │              │               ▼               │
        │              │        ┌──────────────┐       │
        │              │        │execute_task()│       │
        │              │        └──────┬───────┘       │
        │              │               │               │
        │              │        success│fail           │
        │              │               │  │            │
        │              │               ▼  ▼            │
        │              │      ┌────────┐ ┌────────┐    │
        │              │      │complete│ │fail +  │    │
        │              │      │_task() │ │retry?  │    │
        │              │      └───┬────┘ └───┬────┘    │
        │              │          │          │         │
        │              │          ▼          ▼         │
        │              │    ┌────────────────────┐     │
        │              │    │ release_task_gpus()│     │
        │              │    │ (free GPU locks)   │     │
        │              │    └─────────┬──────────┘     │
        │              │              │                │
        └──────────────┴──────────────┘                │
                                                       │
        ◄──────────────────────────────────────────────┘
```

Note: Dependency promotion (pending→ready) is handled by the main monitor loop via `resolve_dependencies()`, not by individual GPU workers.

### Memory-Aware Task Selection

The scheduler only claims tasks that fit in available GPU memory:

```text
GPU 6: 180GB total, 35GB free (70B model loaded)

  Ready queue scan (includes safety margin):
  ┌────────────────────────────┬──────────┬──────────┬────────┐
  │ Task ID                    │ Required │ Priority │ Status │
  ├────────────────────────────┼──────────┼──────────┼────────┤
  │ llama-70b_CERTIFY_001      │ 169 GB   │    65    │ SKIP ✗ │
  │ mixtral_CALIBRATION_003    │ 108 GB   │    85    │ SKIP ✗ │
  │ qwen32b_EVAL_MMLU_002      │  84 GB   │    65    │ SKIP ✗ │
  │ mistral-7b_CERTIFY_002     │  16 GB   │    65    │ FITS ✓ │
  │ qwen-14b_CALIBRATION_02    │  31 GB   │    85    │ FITS ✓ │◄── SELECTED
  └────────────────────────────┴──────────┴──────────┴────────┘

  → Claims qwen-14b task (highest priority that fits in 35GB)
```

### Work-Stealing in Action

With dynamic scheduling, any GPU can claim any task. Small model tasks complete quickly, freeing GPUs to help with remaining large model tasks:

Example shown for an 8-GPU node (GPU_ID_LIST=0,1,2,3,4,5,6,7).

```text
Time→   T=0                    T=50%                  T=100%

GPU 0   ████ small ████ small ████ large (helping) ████░░░░░░
GPU 1   ████ small ████ medium ████ large (helping) ███░░░░░░
GPU 2   ████ small ████ medium ████ large ████░░░░░░░░░░░░░░░
GPU 3   ████ medium ████ medium ████ large ████░░░░░░░░░░░░░░
GPU 4   ████ medium ████ large ████████████████░░░░░░░░░░░░░░
GPU 5   ████ MoE ████████ large ████████████████░░░░░░░░░░░░░
GPU 6   ████ large (70B) ████████████████████████████████░░░░
GPU 7   ████ large (72B) ████████████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result: No GPU sits idle while work remains in the queue
```

## Library Modules

The dynamic scheduling system is implemented across the following library modules in `scripts/lib/`:

### [`task_serialization.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/task_serialization.sh)

JSON schema for tasks and safe read/write helpers:

- **Task Schema & Creation**:
  - `create_task()` - Create a task file with required fields (including `required_gpus`)
  - `validate_task()` - Validate required fields and task type
- **GPU Planning**:
  - `calculate_required_gpus()` - Compute GPUs needed from `model_size_gb` and per-device capacity
- **Task Field Access**:
  - `get_task_field()` / `update_task_field()` - Safe JSON field access
  - `get_task_required_gpus()` - Read required GPU count
  - `get_task_assigned_gpus()` - Read assigned GPU IDs
- **Lifecycle Helpers**:
  - `mark_task_started_multi()` - Mark task started with multi-GPU assignment
- **Memory Estimation**:
  - `estimate_model_memory()` - Estimate per-task memory (includes task-type multipliers + safety margin)

### [`queue_manager.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/queue_manager.sh)

File-backed queue operations and dependency tracking:

- **Queue Operations**:
  - `init_queue()` - Initialize queue directories
  - `add_task()` - Add task to pending queue
  - `claim_task()` - Atomically claim task for execution
  - `complete_task()` / `fail_task()` - Mark task completion status
  - `retry_task()` - Move failed task back to pending for retry
- **Dependency Handling**:
  - `resolve_dependencies()` - Auto-promote pending→ready when deps complete
  - `cancel_tasks_with_failed_dependencies()` - Fail dependents instead of stalling the queue
- **Monitoring & Inspection**:
  - `count_tasks()` / `is_queue_empty()` - Status queries
  - `get_queue_stats()` - Summary statistics
  - `find_task()` - Locate a task by ID across queue states

### [`scheduler.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/scheduler.sh)

GPU memory probes, multi-GPU reservation, adaptive allocation, and intelligent task selection:

- **Memory Management**:
  - `get_gpu_available_memory()` - Query available VRAM (cached)
  - `is_gpu_idle()` - Check for active compute processes (cached)
  - Uses adaptive safety margin: 2% for ≥160GB, 5% for ≥80GB, 10% for <80GB
  - Treats per-task `model_size_gb` as already safety-adjusted (see `estimate_model_memory()`)
- **Multi-GPU Distribution**:
  - `get_required_gpus()` - Calculate optimal GPUs from model size (default 1; scales up when required memory exceeds per-GPU capacity)
  - `get_minimum_gpus()` - Calculate minimum viable GPUs (fallback allocation)
  - `get_required_gpus_from_category()` - Calculate from model category (7b, 70b, moe)
- **GPU Reservation Protection**:
  - `acquire_scheduler_lock()` / `release_scheduler_lock()` - Serialize task selection and reservation
  - `init_gpu_reservations()` - Initialize reservation tracking directory
  - `reserve_gpus()` - Lock multiple GPUs for a task (serialized by scheduler lock)
  - `release_gpus()` - Release GPU locks when task completes
  - `is_gpu_available()` - Check if GPU is reserved
  - `get_available_gpus()` - Get N available GPUs
  - `cleanup_stale_reservations()` - Clean up orphaned reservations
- **Adaptive GPU Allocation**:
  - `should_use_adaptive_gpus()` - Decide when to use fewer than optimal GPUs
  - Prevents GPU idling when only large model tasks remain
  - Logs `[ADAPTIVE]` messages when using reduced parallelism
- **Task Selection**:
  - `find_best_task()` - Memory-aware and multi-GPU-aware task selection
  - `find_and_claim_task()` - Task claiming under scheduler lock with GPU reservation
  - `release_task_gpus()` - Release GPUs on task completion/failure
  - `apply_work_stealing_boost()` - Priority adjustments for lagging models

### [`gpu_worker.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/gpu_worker.sh)

Worker lifecycle, heartbeat management, and multi-GPU task execution:

- **Worker Lifecycle**:
  - `gpu_worker()` - Main worker loop (claims tasks and executes them)
  - `init_worker()` - Initialize per-worker directories and state
  - `should_shutdown()` / `signal_shutdown()` - Cooperative shutdown control
- **Heartbeat & Status**:
  - `update_heartbeat()` - Touch heartbeat file for liveness monitoring
  - `start_heartbeat_thread()` / `stop_heartbeat_thread()` - Background heartbeat loop
  - `update_worker_status()` - Write worker status to disk
- **Orchestration & Monitoring**:
  - `launch_worker_pool()` - Spawn one worker per selected GPU
  - `monitor_workers()` - Monitor heartbeats and reclaim orphaned tasks
  - `wait_for_workers()` - Wait for workers to exit
  - `get_worker_summary()` - Summarize worker status
- **GPU Reservations**:
  - Calls `release_task_gpus()` (from `scheduler.sh`) on task completion/failure

### [`task_functions.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/task_functions.sh)

Atomic task implementations for each task type:

- **Task Dispatch**:
  - `execute_task()` - Run a task by type with standard logging and timeouts
- **Setup & Calibration**:
  - `task_setup_baseline()` - Download and prepare model
  - `task_calibration_run()` - Run drift calibration
  - `task_generate_preset()` - Create calibration preset
- **lm-eval**:
  - `task_eval_baseline()` - Baseline lm-eval
  - `task_eval_edit()` - Edited-model lm-eval (monolithic: all benchmarks in one task)
  - `task_eval_single_benchmark()` - Split eval for one benchmark (MMLU, HellaSwag, ARC, WinoGrande)
- **Edit Creation**:
  - `task_create_edit()` - Apply a single quantization/pruning/SVD edit (per-edit)
  - `task_create_edits_batch()` - Create all 8 edits with a single model load (batch optimization)
- **Certification & Errors**:
  - `task_certify_edit()` - InvarLock certify for an edit
  - `task_create_error()` / `task_certify_error()` - Error injection variants + certification

### [`model_creation.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/model_creation.sh)

Shared model creation helpers used by workers:

- **Model Edits**:
  - `create_edited_model()` - RTN quantization edits
  - `create_pruned_model()` - Magnitude pruning edits
  - `create_lowrank_model()` - Low-rank SVD edits
  - `create_fp4_model()` - FP4 quantization edits (simulated)
- **Error Injection**:
  - `create_error_model()` - Error injection variants

### [`fault_tolerance.sh`](https://github.com/invarlock/invarlock/blob/main/scripts/lib/fault_tolerance.sh)

Retry policies and recovery hooks:

- **Error Detection**:
  - `detect_oom()` - Detect OOM signatures in task logs
  - `detect_transient_error()` - Detect transient failures eligible for retry
  - `detect_permanent_error()` - Detect fatal errors that should not be retried
  - `classify_error()` - Classify as `oom` / `transient` / `permanent` / `unknown`
- **Retry & Backoff**:
  - `calculate_backoff()` - Exponential backoff with jitter (capped)
  - `should_retry_task()` - Apply per-task retry limits (OOM gets fewer attempts)
  - `maybe_retry_task()` - Schedule a retry by setting `retry_after` and re-queueing
  - `is_retry_ready()` - Check whether the `retry_after` delay has elapsed
- **OOM Recovery**:
  - `handle_oom_task()` - Reduce batch size/sequence length and clear GPU memory before retry
- **Error Reporting**:
  - `record_error()` - Append to `state/errors.json` for summary statistics
  - `get_error_stats()` - Summarize total and recent errors
  - `print_error_summary()` - Print a concise error summary for the run
- **Health & Cleanup**:
  - `health_check()` - Preflight GPU/disk/PyTorch readiness
  - `cleanup_failed_task()` - Remove incomplete artifacts for a failed task
  - `cleanup_all_failed()` - Sweep failed queue and clean incomplete artifacts

## Execution Flow

The validation suite executes in these phases:

```text
┌─────────────────────────────────────────────────────────────┐
│  PHASE 0: Environment Setup                                 │
│  - Dependency check (torch, transformers, lm_eval, etc.)    │
│  - Flash Attention 2 detection/install                      │
│  - B200 optimization flags (TF32, cuDNN benchmark)          │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Task Queue Initialization                         │
│  - Generate tasks for all 8 models                          │
│  - ~54-71 tasks per model (~432-568 total)                  │
│  - Resume mode: reuse existing queue                        │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: GPU Worker Launch                                 │
│  - Start N parallel workers (one per selected GPU)          │
│  - Workers claim tasks dynamically                          │
│  - Progress monitoring every 60 seconds                     │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Analysis & Verdict                                │
│  - Compile lm-eval and InvarLock results                    │
│  - Correlation analysis (true positive/negative rates)      │
│  - Generate final Phase 0 verdict                           │
└─────────────────────────────────────────────────────────────┘
```

## Model-Size-Aware Configuration

The suite automatically adjusts InvarLock parameters based on model size:

| Model Size | seq_len | stride | windows | eval_batch |
|------------|---------|--------|---------|------------|
| 7B | 2048 | 1024 | 64+64 | 96 |
| 13-14B | 1536 | 768 | 48+48 | 64 |
| 30B | 1024 | 512 | 40+40 | 48 |
| 40B | 1024 | 512 | 36+36 | 32 |
| MoE | 1024 | 512 | 40+40 | 24 |
| 70-72B | 128 | 64 | 8+8 | 2 |

The conservative 70B settings prevent OOM during overhead checks where models may be loaded twice.
Per-task CI profile overrides are written under each task's `config_root/runtime/profiles/ci.yaml`
via `INVARLOCK_CONFIG_ROOT`, so window counts and bootstrap replicates always match the model size.
Bootstrap defaults are 2000 for small/medium models and 1000 for >=30B.

## Environment Variables

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | unset | Explicit GPU IDs to use (comma-separated numeric indices); if unset, uses all GPUs detected by `nvidia-smi` |
| `NUM_GPUS` | unset | Number of GPUs to use; if set, clamps to the first N GPUs from `CUDA_VISIBLE_DEVICES`/auto-detected GPUs |
| `GPU_MEMORY_GB` | `180` | Expected per-GPU memory for planning (live scheduling still uses `nvidia-smi`) |
| `OUTPUT_DIR` | auto-generated | Output directory path |
| `RESUME_MODE` | `true` | Skip completed work |

### Model Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_1` | `mistralai/Mistral-7B-v0.1` | Small model (~14 GB) |
| `MODEL_2` | `NousResearch/Llama-2-13b-hf` | Small model (~26 GB) |
| `MODEL_3` | `Qwen/Qwen2.5-14B` | Small model (~28 GB) |
| `MODEL_4` | `Qwen/Qwen2.5-32B` | Medium model (~64 GB) |
| `MODEL_5` | `01-ai/Yi-34B` | Medium model (~68 GB) |
| `MODEL_6` | `mistralai/Mixtral-8x7B-v0.1` | MoE model (~90 GB) |
| `MODEL_7` | `NousResearch/Llama-2-70b-hf` | Large model (~140 GB) |
| `MODEL_8` | `Qwen/Qwen1.5-72B` | Large model (~144 GB) |

### Evaluation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_TASKS` | `mmlu,hellaswag,arc_challenge,winogrande` | lm-eval benchmark tasks |
| `EVAL_NUM_FEWSHOT` | `5` | Few-shot examples |
| `EVAL_BATCH_SIZE` | `auto` | Auto-detect optimal batch |
| `EVAL_BATCH_SIZE_SMALL` | `auto:16` | 7B-14B models |
| `EVAL_BATCH_SIZE_MEDIUM` | `auto:8` | 30B-40B models |
| `EVAL_BATCH_SIZE_LARGE` | `auto:4` | 70B+ models |
| `EVAL_BATCH_SIZE_MOE` | `auto:6` | MoE models |
| `EVAL_CONTEXT_LEN` | `2048` | Context length cap used for memory planning |
| `LM_EVAL_PARALLELIZE` | `true` | Enable lm-eval `parallelize=True` when multiple GPUs are assigned |
| `LMEVAL_TORCH_COMPILE` | auto | Enable `torch.compile` in lm-eval (set by `B200_DETERMINISM`, overridable) |

### InvarLock Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INVARLOCK_DATASET` | `wikitext2` | Evaluation dataset |
| `INVARLOCK_TIER` | `balanced` | Guard tier preset |
| `B200_GUARDS_ORDER` | unset | Override InvarLock guard order for generated configs/presets (comma-separated; e.g. `invariants,spectral,rmt,variance,invariants`) |
| `INVARLOCK_BOOTSTRAP_N` | unset | Bootstrap replicates override (defaults: 2000, 1000 for ≥30B) |
| `INVARLOCK_PM_ACCEPTANCE_MIN` | `0.90` | Primary metric lower bound |
| `INVARLOCK_PM_ACCEPTANCE_MAX` | `1.20` | Primary metric upper bound |
| `INVARLOCK_SKIP_OVERHEAD_CHECK` | unset | Skip guard overhead measurement |
| `INVARLOCK_CONFIG_ROOT` | unset | Per-task runtime profile root (set automatically by the suite) |
| `INVARLOCK_ALLOW_NETWORK` | `1` | Allow dataset downloads |
| `INVARLOCK_REQUIRE_FP4_NATIVE` | `false` | Require TransformerEngine for native FP4 validation |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `B200_DETERMINISM` | `throughput` | Determinism mode: `throughput` or `strict` |
| `GPU_MIN_FREE_GB` | `10` | Minimum free GPU memory for task eligibility |
| `GPU_REQUIRE_IDLE` | `true` | Require GPUs with no compute processes when scheduling tasks |
| `SKIP_FLASH_ATTN` | `false` | Skip Flash Attention install |
| `CUDA_MEMORY_FRACTION` | `0.92` | Target memory utilization |
| `DRIFT_CALIBRATION_RUNS` | `5` | Calibration run count |
| `CLEAN_EDIT_RUNS` | `3` | Clean edit repetitions |
| `STRESS_EDIT_RUNS` | `2` | Stress edit repetitions |
| `RUN_ERROR_INJECTION` | `true` | Enable error injection tests |

### Reliability Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_MEM_TOLERANCE_GB` | `8` | Allow large-model tasks within this many GB of available GPU memory |
| `CANCEL_BLOCKED_TASKS_GRACE_SECONDS` | `90` | Grace period before failing tasks whose dependencies are in FAILED (prevents queue stall) |
| `MIN_FREE_DISK_GB` | `200` | Abort early if free disk space in `OUTPUT_DIR` filesystem drops below this threshold |
| `TASK_TIMEOUT_DEFAULT` | `21600` | Task timeout in seconds (0/empty disables) |
| `TASK_TIMEOUT_<TASKTYPE>` | unset | Per-task timeout override in seconds |
| `MODEL_LOAD_OVERHEAD_GB` | `4` | Added headroom (GB) for model load planning |
| `EDIT_OVERHEAD_GB` | `8` | Added headroom (GB) for per-edit tasks |
| `BATCH_EDIT_OVERHEAD_GB` | `8` | Added headroom (GB) for batch edit tasks |
| `EVAL_OVERHEAD_GB` | `6` | Added headroom (GB) for lm-eval tasks |
| `INVARLOCK_OVERHEAD_GB` | `6` | Added headroom (GB) for InvarLock run/certify tasks |

## Determinism vs Throughput

The suite supports two determinism modes:

### Throughput Mode (Default)

```bash
B200_DETERMINISM=throughput ./scripts/b200_validation_suite.sh
```

- Enables TF32 for faster matrix operations
- Enables cuDNN benchmark for optimal kernel selection
- Enables `torch.compile` for lm-eval and keeps `accelerator.compile=true`
- InvarLock's CI/Release determinism presets are still enforced within `invarlock run` / `invarlock certify`
- Recorded in certificate `meta.determinism`

### Strict Mode

```bash
B200_DETERMINISM=strict ./scripts/b200_validation_suite.sh
```

- Disables TF32 and cuDNN benchmark at harness level
- Disables `torch.compile` in lm-eval and sets `accelerator.compile=false`
- Sets `CUBLAS_WORKSPACE_CONFIG=:32768:8` for deterministic cuBLAS behavior
- Aligns with CI presets for reproducibility
- Use when certificates must match exact CI environment

## Output Structure

```text
invarlock_validation_b200_20241208_123456/
├── logs/
│   ├── main.log                    # Overall execution log
│   └── gpu_{id}.log                # Per-GPU worker logs (id = physical GPU index in the pool)
├── models/
│   └── {model_name}/               # Downloaded model weights
│       └── baseline/
│           └── model_profile.json  # Weights + config used for memory planning
├── presets/
│   └── calibrated_preset_{model}.yaml  # Calibration presets
├── queue/
│   ├── pending/                    # Tasks waiting on dependencies
│   ├── ready/                      # Tasks ready for execution
│   ├── running/                    # Currently executing tasks
│   ├── completed/                  # Successfully finished tasks
│   └── failed/                     # Failed tasks
├── state/
│   ├── progress.json               # Queue progress snapshots
│   └── disk_pressure.json          # Present if aborted due to low disk space
├── workers/
│   ├── gpu_{id}.pid                # Worker process IDs
│   └── gpu_{id}.heartbeat          # Worker liveness files
├── {model_name}/
│   ├── models/                     # Edited model variants
│   │   ├── quant_8bit_clean/
│   │   ├── quant_4bit_stress/
│   │   ├── fp4_e2m1_clean/
│   │   ├── prune_10pct_clean/
│   │   ├── svd_rank256_clean/
│   │   └── error_nan_injection/
│   ├── evals/                      # lm-eval results
│   │   ├── baseline_results.json
│   │   └── {edit}_results.json
│   └── certificates/
│       ├── calibration/
│       │   ├── run_{1-5}/          # Calibration runs
│       │   └── calibration_stats.json
│       ├── {edit_name}/
│       │   └── run_{1-3}/
│       │       └── evaluation.cert.json
│       └── errors/
│           └── {error_type}/
├── analysis/
│   ├── eval_results.csv            # Aggregated lm-eval scores
│   ├── invarlock_results.csv       # Aggregated InvarLock verdicts
│   ├── guard_sensitivity_matrix.csv
│   ├── calibration_summary.json
│   ├── policy_digest_summary.json
│   ├── determinism_summary.json
│   ├── memory_plan.csv             # Per-task memory + GPU plan
│   └── correlation_analysis.json
└── reports/
    ├── final_verdict.txt           # Human-readable summary
    └── final_verdict.json          # Machine-readable results
```

## Troubleshooting

### OOM on 70B+ Models

**Symptom**: CUDA out of memory during 70B or 72B model processing.

**Solutions**:

1. Skip overhead check:

   ```bash
   INVARLOCK_SKIP_OVERHEAD_CHECK=1 ./scripts/b200_validation_suite.sh
   ```

2. Ensure single model per GPU - check no other processes are using GPU memory:

   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
   ```

3. Use a lighter guard tier for reduced overhead:

   ```bash
   INVARLOCK_TIER=dev
   ```

4. Reduce eval batch caps for large models:

   ```bash
   EVAL_BATCH_SIZE_LARGE=auto:2 ./scripts/b200_validation_suite.sh
   ```

5. Multi-GPU sharding is supported automatically when the post-download memory plan
   says a task cannot fit within `GPU_MEMORY_PER_DEVICE`. If you still see OOMs,
   increase the overhead knobs (e.g., `INVARLOCK_OVERHEAD_GB`, `EDIT_OVERHEAD_GB`)
   or lower `GPU_MEMORY_PER_DEVICE` so the planner requests more GPUs.

### Task Timeouts

**Symptom**: Task fails with `Exit code 124` or `Timeout: task exceeded limit`.

**Solutions**:

1. Increase the default timeout:

   ```bash
   TASK_TIMEOUT_DEFAULT=28800 ./scripts/b200_validation_suite.sh
   ```

2. Override a specific task type:

   ```bash
   TASK_TIMEOUT_CREATE_EDIT=28800 ./scripts/b200_validation_suite.sh
   ```

### Preset Not Found

**Symptom**: `FileNotFoundError: calibrated_preset_{model}.yaml`

**Solutions**:

1. Rerun calibration to regenerate preset:

   ```bash
   # Remove existing calibration
   rm -rf ${OUTPUT_DIR}/{model_name}/certificates/calibration
   # Re-run with resume
   ./scripts/b200_validation_suite.sh --resume
   ```

2. Check preset directory exists and contains YAML files:

   ```bash
   ls -la ${OUTPUT_DIR}/presets/
   ```

### Resume Mode

**Usage**: Resume a failed or interrupted run without regenerating tasks:

```bash
OUTPUT_DIR=./invarlock_validation_b200_20241208_123456 \
  ./scripts/b200_validation_suite.sh --resume
```

**What happens**:

- Skips task queue generation
- Moves orphaned "running" tasks back to pending
- Moves failed tasks back to pending for retry
- Continues from where it left off

### Stuck Queue

**Symptom**: Progress stuck, queue not draining.

**Diagnosis**:

```bash
# Check queue status
ls ${OUTPUT_DIR}/queue/running/
ls ${OUTPUT_DIR}/queue/pending/

# Check for orphaned tasks
cat ${OUTPUT_DIR}/queue/running/*.task | jq -r '.task_id'

# Check worker logs for errors
tail -100 ${OUTPUT_DIR}/logs/gpu_0.log
grep -i error ${OUTPUT_DIR}/logs/*.log
```

**Solutions**:

1. Manually move stuck tasks back to pending:

   ```bash
   mv ${OUTPUT_DIR}/queue/running/*.task ${OUTPUT_DIR}/queue/pending/
   ```

2. Check for dependency cycles (should not occur but verify):

   ```bash
   cat ${OUTPUT_DIR}/queue/pending/*.task | jq -r '.dependencies[]'
   ```

### Flash Attention Failures

**Symptom**: `ImportError: cannot import flash_attn` or build failures.

**Solutions**:

1. Skip Flash Attention entirely:

   ```bash
   SKIP_FLASH_ATTN=true ./scripts/b200_validation_suite.sh
   ```

2. Install Python development headers:

   ```bash
   apt-get install python3-dev  # or python3.X-dev
   ```

3. Flash Attention will fall back to eager attention automatically for incompatible models (Falcon, MPT, etc.)

### Worker Crashes

**Symptom**: Workers exit unexpectedly, tasks stay in "running" state.

**Diagnosis**:

```bash
# Check worker PIDs
cat ${OUTPUT_DIR}/workers/gpu_*.pid | xargs ps -p

# Check for crash signatures
grep -E "(SIGKILL|OOM|killed)" ${OUTPUT_DIR}/logs/gpu_*.log
```

**Solutions**:

1. Check CUDA driver compatibility
2. Verify sufficient system RAM (models use CPU memory during loading)
3. Reduce `NUM_GPUS` if thermal throttling suspected

## Validation Verdict

The suite produces a Phase 0 validation verdict based on:

| Metric | Threshold | Weight |
|--------|-----------|--------|
| **Accuracy** | ≥60% | Required |
| **Error Detection** | ≥80% | Required |
| **Precision** | Reported | Info |
| **Recall** | Reported | Info |
| **F1 Score** | Reported | Info |

**Confidence Score** (0-100):

- **Sample confidence**: grows with the number of edit/eval pairs (log-scaled, capped at 25 points).
- **Error confidence**: grows with the number of error-injection experiments (log-scaled, capped at 25 points).
- **Accuracy confidence**: higher when the 95% Wilson confidence interval for accuracy is narrow (up to 25 points).
- **Balance confidence**: proportional to F1 score (up to 25 points).

**Verdict Outcomes**:

- `PHASE0_VALIDATED` + HIGH confidence: Ready for broader deployment
- `PHASE0_VALIDATED` + MEDIUM confidence: Functional, needs more testing
- `PHASE0_VALIDATED` + LOW confidence: Marginally passing
- `PHASE0_FAILED`: Does not meet Phase 0 requirements

## CI Integration

For CI pipelines, use reduced scope:

```bash
NUM_GPUS=2 \
MODEL_1="gpt2" \
MODEL_2="gpt2-medium" \
DRIFT_CALIBRATION_RUNS=2 \
CLEAN_EDIT_RUNS=1 \
STRESS_EDIT_RUNS=1 \
RUN_ERROR_INJECTION=false \
./scripts/b200_validation_suite.sh
```

## Related Documentation

- [Drift Gate (Quality Gate)](../assurance/04-guard-contracts.md#quality-gates-acceptance)
- [Device Drift Bands](../assurance/12-device-drift-bands.md)
- [Certificate Schema](../reference/certificate-schema.md)
- [Guard Reference](../reference/guards.md)
