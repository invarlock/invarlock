# Proof Pack Internals

This guide explains how the proof pack suite is wired internally: entrypoints,
task graph, scheduling, and artifact generation. It complements
[Proof Packs](proof-packs.md), which focuses on how to run a suite.

## Overview

| Aspect | Details |
| --- | --- |
| Purpose | Hardware-agnostic Phase 0 validation harness for edit detection |
| Version | `proof-packs-v1` |
| Hardware | NVIDIA GPUs where models fit VRAM; multi-GPU recommended for `full` |
| Models | `subset` (1 model) or `full` (6 models), ungated public |
| Edits | 4 types × 2 versions per model; clean variants use tuned presets |
| Scheduling | Dynamic work-stealing, `small_first` priority strategy |
| Multi-GPU | Profile-based; `required_gpus` grows only when memory requires it |
| Output | Proof pack with `manifest.json`, `checksums.sha256`, and cert bundles (`--layout v2` nests results + metadata) |

## Quick Start (Context)

```bash
# Run the subset suite (offline by default)
./scripts/proof_packs/run_suite.sh --suite subset

# Run the full suite and build a proof pack
./scripts/proof_packs/run_pack.sh --suite full --net 1

# Verify an existing proof pack
./scripts/proof_packs/verify_pack.sh --pack ./proof_pack_runs/subset_20250101_000000/proof_pack
```

## Hardware Target

- Hardware-agnostic by design; run on any NVIDIA GPU topology where the models
  fit in VRAM.
- Multi-GPU scheduling is enabled automatically when a task’s memory plan
  exceeds per-device capacity.
- Set `GPU_MEMORY_GB` or `GPU_MEMORY_PER_DEVICE` to match your hardware when
  running on GPUs with unusual memory sizes.

## Entrypoints and modules

### Entrypoints

- `scripts/proof_packs/run_suite.sh` runs a suite and sets `PACK_*` runtime flags
  before calling the main orchestrator.
- `scripts/proof_packs/run_pack.sh` runs a suite, then packages artifacts into a
  portable proof pack (manifest + checksums + certs).
- `scripts/proof_packs/verify_pack.sh` validates a proof pack: checksums,
  optional GPG signature, and `invarlock verify`.
- `scripts/proof_packs/suites.sh` defines the `subset` and `full` model sets and
  allows `MODEL_1`–`MODEL_8` overrides.
- `scripts/proof_packs/lib/validation_suite.sh` orchestrates the run: preflight,
  queue creation, worker launch, and monitoring.

### Library modules

- `lib/task_serialization.sh`: task schema, JSON helpers, GPU planning.
- `lib/queue_manager.sh`: queue states, dependency resolution, task generation.
- `lib/scheduler.sh`: dynamic priority, memory gating, reservations.
- `lib/gpu_worker.sh`: worker loop, heartbeats, task execution glue.
- `lib/task_functions.sh`: implementations for each task type.
- `lib/model_creation.sh`: edit and error-model creation helpers (`create_model_variant` dispatcher).
- `lib/config_generator.sh`: InvarLock config generation and wrapper helpers.
- `lib/result_compiler.sh`: analysis and verdict compilation.
- `lib/fault_tolerance.sh`: error classification and retry/backoff logic.
- `scripts/proof_packs/python/manifest_writer.py`: proof pack `manifest.json` writer.
- `scripts/proof_packs/python/preset_generator.py`: calibrated preset + edit-type variants.

### Module dependency graph

```text
┌─────────────────────────────────────────────────────────┐
│                         ENTRYPOINTS                      │
├──────────────┬──────────────┬────────────────────────────┤
│  run_pack.sh  │ run_suite.sh │ verify_pack.sh             │
│  (pack+run)   │ (run only)   │ (checksums+certs verify)   │
└──────┬────────┴──────┬───────┴────────────────────────────┘
       │               │
       ▼               ▼
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                    │
├──────────────────────────────────────────────────────────┤
│  lib/validation_suite.sh (main_dynamic)                   │
│  ├─ Phase 0: setup + preflight                            │
│  ├─ Phase 1: queue init      ───────────┐                 │
│  ├─ Phase 2: worker launch   │          │                 │
│  └─ Phase 3: monitor + retry │          │                 │
└──────────────────────────────┼──────────┼─────────────────┘
                               │          │
       ┌───────────────────────┘          └─────────────┐
       ▼                                                 ▼
┌──────────────────────────┐                   ┌──────────────────┐
│       TASK EXECUTION      │                   │  CORE SERVICES   │
├──────────────────────────┤                   ├──────────────────┤
│  lib/gpu_worker.sh        │◄──────────────────┤  queue_manager   │
│  ├─ Task claim            │                   │  scheduler       │
│  ├─ OOM pre-check         │                   │  task_serial.    │
│  ├─ execute_task()        │                   │  fault_tol.      │
│  └─ GPU cleanup           │                   └──────────────────┘
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│       TASK FUNCTIONS      │
├──────────────────────────┤
│  ├─ SETUP_BASELINE        │
│  ├─ CALIBRATION_RUN       │
│  ├─ GENERATE_PRESET       │
│  ├─ CREATE_EDITS(_BATCH)  │
│  ├─ CREATE_ERROR          │
│  └─ CERTIFY_*             │
└──────────────────────────┘
```

## Troubleshooting decision tree

```text
Proof pack issues?
│
├─ Missing manifest.json/checksums.sha256?
│  └─ Used run_suite.sh instead of run_pack.sh
│     → Run: ./scripts/proof_packs/run_pack.sh --suite ... --net ...
│
├─ Spectral guard failing “clean” quantization edits?
│  ├─ Check: caps_exceeded in certificate spectral.summary
│  │  └─ Use edit-type presets (generated from calibration) or increase max_caps
│  └─ Check: high z-scores in attention layers
│     └─ Expected for quantization; calibrate or adjust thresholds
│
├─ OOM errors?
│  ├─ Lower GPU_MEMORY_PER_DEVICE / GPU_MEMORY_GB
│  ├─ Disable batching: PACK_USE_BATCH_EDITS=false
│  └─ Reduce InvarLock batch/seq_len (INVARLOCK_EVAL_BATCH, INVARLOCK_SEQ_LEN)
│
└─ Disk pressure / ENOSPC?
   ├─ Check OUTPUT_DIR filesystem free space
   └─ Use a larger volume and rerun (suite writes caches under OUTPUT_DIR/.hf)
```

## Model Suite

Model suites are defined in `scripts/proof_packs/suites.sh` and applied by
`run_suite.sh`.

| Suite | Models | Notes |
| --- | --- | --- |
| `subset` | `mistralai/Mistral-7B-v0.1` | Single-GPU friendly |
| `full` | `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B`, `01-ai/Yi-34B`, `mistralai/Mixtral-8x7B-v0.1`, `Qwen/Qwen1.5-72B` | Multi-GPU recommended |

Default full-suite model sizes (weights-only, approximate):

| Model | VRAM | Category | Notes |
| --- | --- | --- | --- |
| `mistralai/Mistral-7B-v0.1` | ~14 GB | Small | Flash Attention 2 compatible |
| `Qwen/Qwen2.5-14B` | ~28 GB | Small | Flash Attention 2 compatible |
| `Qwen/Qwen2.5-32B` | ~64 GB | Medium | Flash Attention 2 compatible |
| `01-ai/Yi-34B` | ~68 GB | Medium | Flash Attention 2 compatible |
| `mistralai/Mixtral-8x7B-v0.1` | ~90 GB | MoE | MoE architecture |
| `Qwen/Qwen1.5-72B` | ~144 GB | Large | Flash Attention 2 compatible |

Notes:

- Override models via `MODEL_1`–`MODEL_8`; set an empty string to disable a slot.
- `validation_suite.sh` includes a fallback list of large causal models if it is
  run directly without `suites.sh`.

## Edit Types

Each model runs 8 edit experiments (4 types × 2 versions) plus optional error
injection tests.

### Clean edits (tuned)

Clean edits use tuned parameters supplied via `PACK_TUNED_EDIT_PARAMS_FILE`.
The suite uses `:clean:` as a sentinel in the edit spec and resolves concrete
parameters at runtime.

| Edit Type | Parameters | Scope |
| --- | --- | --- |
| Quantization RTN | tuned (`bits`, `group_size`) from tuned params file | FFN only |
| FP8 Quantization | tuned (`format`) from tuned params file | FFN only |
| Magnitude Pruning | tuned (`prune_level`) from tuned params file | FFN only |
| Low-Rank SVD | tuned (`rank`) from tuned params file | FFN only |

### Stress edits

Stress edits should trigger InvarLock guard failures:

| Edit Type | Parameters | Scope |
| --- | --- | --- |
| Quantization RTN | `quant_rtn:4:32:all` (4-bit, group size 32) | All layers |
| FP8 Quantization | `fp8_quant:e5m2:all` | All layers |
| Magnitude Pruning | `magnitude_prune:0.5:all` (50% sparsity) | All layers |
| Low-Rank SVD | `lowrank_svd:32:all` (rank 32) | All layers |

### Error injection tests

Enabled when `RUN_ERROR_INJECTION=true` (default):

- `nan_injection`
- `inf_injection`
- `extreme_quant`
- `scale_explosion`
- `weight_tying_break`

## Scheduling

The suite uses dynamic work-stealing scheduling with a file-backed task queue.
`validation_suite.sh` seeds the queue and launches one worker per GPU; workers
claim tasks under a scheduler lock with GPU reservation files.

### `small_first` priority strategy

Base task priorities (queue manager) are combined with dynamic boosts in
`scheduler.sh` (model size, blocked dependents, age, and fairness penalties).

```text
Priority (base)     Task type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  90 ┤ SETUP_BASELINE
  85 ┤ CALIBRATION_RUN
  75 ┤ GENERATE_PRESET
  70 ┤ CREATE_EDITS_BATCH / CREATE_EDIT
  65 ┤ CERTIFY_EDIT
  60 ┤ CREATE_ERROR
  55 ┤ CERTIFY_ERROR
```

Dynamic boosts (scheduler):

- Model size boosts: <30GB (+30), <70GB (+20), <100GB (+10).
- Critical tasks: `SETUP_BASELINE` (+50), `CALIBRATION_RUN` (+20).
- Unblock boost: +2 per dependent task (capped).
- Age boost: +1 per 5 minutes in the queue (capped).
- Fairness penalty: -3 per running task for the same model (capped).
- Work-stealing boost: raises priority for lagging models.

### Dynamic scheduling diagram

```text
run_pack.sh (optional)
  -> run_suite.sh
     -> validation_suite.sh (main_dynamic)
        -> init_queue + generate_all_tasks
        -> start gpu_worker per GPU
        -> monitor loop (resolve deps, progress, restarts)
```

### Work-stealing timeline (illustrative)

```text
Time→   T=0                     T=50%                  T=100%
GPU 0   ████ small ████ small ████ large (helping) ████░░░░░░
GPU 1   ████ small ████ medium ████ large (helping) ███░░░░░░
GPU 2   ████ small ████ medium ████ large ████░░░░░░░░░░░░░░░
GPU 3   ████ medium ████ medium ████ large ████░░░░░░░░░░░░░░
GPU 4   ████ medium ████ large ████████████████░░░░░░░░░░░░░░
GPU 5   ████ MoE ████████ large ████████████████░░░░░░░░░░░░░
```

Illustrative only; actual scheduling depends on queue state and memory.

## Multi-GPU Model Distribution

After baseline setup, the suite writes `model_profile.json` and updates per-task
memory estimates. `task_serialization.sh` calculates `required_gpus` based on
`GPU_MEMORY_PER_DEVICE` and `NUM_GPUS`:

- Tasks reserve multiple GPUs only when memory exceeds per-device capacity.
- Adaptive under-allocation is disabled by default (`get_minimum_gpus` matches
  `required_gpus`) to avoid OOM.
- Set `GPU_MEMORY_PER_DEVICE` explicitly for non-80/180GB hardware.

### Memory-aware selection example

```text
GPU 2: 80GB total, 28GB free

Ready queue scan (highest-priority fit):
  qwen-14b_CALIBRATION_RUN_002  req=24GB  pri=85  FITS ✓
  mixtral_CREATE_EDITS_BATCH_001 req=92GB pri=70  SKIP ✗
  yi-34b_CERTIFY_EDIT_001       req=72GB  pri=65  SKIP ✗
```

### GPU reservation protection

Reservations are stored under `OUTPUT_DIR/workers/gpu_reservations/` and guarded
by a `queue/scheduler.lock` (mkdir-based). The scheduler also expires stale
reservations by TTL (`GPU_RESERVATION_TTL`).

### Reservation state example

```text
GPU 0   GPU 1   GPU 2   GPU 3
FREE    RSVD    FREE    RSVD
        ^              ^
        |              |
      task_a     task_b (multi-GPU: 1,3)
```

```text
queue/scheduler.lock
workers/gpu_reservations/
├── gpu_1.lock
├── task_<task_id>.gpus
└── task_<task_id>.meta
```

## Task lifecycle

```text
┌─────────┐    ┌───────┐    ┌─────────┐    ┌───────────┐
│ PENDING │───▶│ READY │───▶│ RUNNING │───▶│ COMPLETED │
└─────────┘    └───────┘    └─────────┘    └───────────┘
                                 │
                                 ▼
                             ┌────────┐
                             │ FAILED │
                             └────────┘
```

### GPU worker loop

```text
START gpu_worker
  │
  ├─ check shutdown? ── yes → exit
  │
  ├─ query GPU memory
  ├─ find_and_claim_task (scheduler lock + reservation)
  │     ├─ none → sleep → loop
  │     └─ task → execute_task → complete/fail → release_gpus
  └─ update heartbeat/status → loop
```

## Batch optimizations

Small/medium models default to batch edit creation:

- **Batch edit creation**: `CREATE_EDITS_BATCH` loads a model once and creates
  all 8 edits (cuts repeated model loads).

Large or MoE models disable batch edits automatically (or via
`PACK_USE_BATCH_EDITS=false`) and fall back to per-edit tasks
(`CREATE_EDIT → CERTIFY_EDIT`).

### Task dependency graphs

Batch (default):

```text
SETUP_BASELINE
  ├─ CALIBRATION_RUN × N ──> GENERATE_PRESET ──┐
  ├─ CREATE_EDITS_BATCH ------------------------┴─> CERTIFY_EDIT × runs
  └─ CREATE_ERROR × types ----------------------┴─> CERTIFY_ERROR × types
```

Notes:

- Error injection tasks (`CREATE_ERROR` → `CERTIFY_ERROR`) branch off
  `SETUP_BASELINE` and require the preset for certification.

Per-edit path (large/MoE or `PACK_USE_BATCH_EDITS=false`):

```text
SETUP_BASELINE
  ├─ CALIBRATION_RUN × N ──> GENERATE_PRESET ──┐
  ├─ CREATE_EDIT × edits -----------------------┴─> CERTIFY_EDIT × runs
  └─ CREATE_ERROR × types ----------------------┴─> CERTIFY_ERROR × types
```

## Task breakdown per model (defaults)

Defaults: `DRIFT_CALIBRATION_RUNS=5`, `CLEAN_EDIT_RUNS=3`,
`STRESS_EDIT_RUNS=2`, `RUN_ERROR_INJECTION=true`.

Batch path (default for small/medium):

- Setup baseline: 1 task
- Calibration runs + preset: 6 tasks
- Batch edits: 1 task
- Certify edits: 20 tasks
- Error injection: 10 tasks

Total: ~38 tasks/model (varies with overrides).

Per-edit path (large/MoE or `PACK_USE_BATCH_EDITS=false`):

- Setup baseline: 1 task
- Calibration runs + preset: 6 tasks
- Create edits: 8 tasks
- Certify edits: 20 tasks
- Error injection: 10 tasks

Total: ~45 tasks/model (varies with overrides).

## Execution phases

```text
PHASE 0: Environment setup
  - Dependency checks, GPU pool configuration, disk preflight
PHASE 1: Task queue initialization
  - Generate tasks for all models, resolve initial dependencies
PHASE 2: GPU worker launch
  - Spawn one worker per GPU, dynamic scheduling in loop
PHASE 3: Reports + verdict
  - Compile certificates into final verdict reports
```

## Run directory layout

```text
OUTPUT_DIR/
  analysis/
    determinism_repeats.json          # optional (when --repeats is used)
  reports/
    final_verdict.txt
    final_verdict.json
  presets/
  state/
    model_revisions.json              # pinned HF revisions (when --net 1)
    progress.json
    disk_pressure.json
    tuned_edit_params.json            # copy of PACK_TUNED_EDIT_PARAMS_FILE
  queue/
    pending/ ready/ running/ completed/ failed/
    queue.lock
    scheduler.lock
  logs/
    gpu_<id>.log
    tasks/<task_id>.log
  workers/
    gpu_<id>.pid
    gpu_<id>.heartbeat
    gpu_<id>.status
    gpu_reservations/
    SHUTDOWN
  <model_name>/
    models/
      baseline/
      <edit_name>/
      error_<type>/
    certificates/
      calibration/
      <edit_name>/run_<n>/
      errors/<type>/
```

## Run modes

- `--calibrate-only` / `PACK_SUITE_MODE=calibrate-only`
  - Only promotes `SETUP_BASELINE`, `CALIBRATION_RUN`, and `GENERATE_PRESET`
    tasks.
  - The monitor exits after all `GENERATE_PRESET` tasks complete.
- `--run-only`
  - Continue a prior run after calibration. This is effectively `--resume` with
    `PACK_SUITE_MODE=full`.
- `--resume`
  - Reuses an existing queue and continues from where the run stopped.

## Determinism vs throughput

`PACK_DETERMINISM` controls harness-level determinism:

```bash
# Throughput (default)
PACK_DETERMINISM=throughput ./scripts/proof_packs/run_suite.sh --suite subset

# Strict
PACK_DETERMINISM=strict ./scripts/proof_packs/run_suite.sh --suite subset
```

- Throughput: `NVIDIA_TF32_OVERRIDE=1`, `CUDNN_BENCHMARK=1`.
- Strict: `NVIDIA_TF32_OVERRIDE=0`, `CUDNN_BENCHMARK=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Network mode and model revisions

Proof packs are offline by default:

- `PACK_NET=0` sets `INVARLOCK_ALLOW_NETWORK=0` and enables HF offline modes.
- `PACK_NET=1` enables downloads and writes `state/model_revisions.json` (ungated
  models only).
- Offline runs require `model_revisions.json`; missing revisions trigger a hard
  error during `SETUP_BASELINE`.

Use `PACK_MODEL_REVISIONS_FILE` to override the revisions path.

## Disk and cache behavior

Large runs can be storage-heavy (baseline + edits + error models):

- Disk preflight estimates required storage and aborts early when insufficient.
  - Override with `PACK_SKIP_DISK_PREFLIGHT=1` (not recommended).
  - The minimum free space guard is `MIN_FREE_DISK_GB` (default 200).
- `PACK_BASELINE_STORAGE_MODE=snapshot_symlink` stores baseline weights as
  symlinks to HF cache files to reduce duplication.
- HF caches default to `OUTPUT_DIR/.hf` (override with `HF_HOME`, `HF_HUB_CACHE`,
  `HF_DATASETS_CACHE`).

## Proof pack packaging and verification

`run_pack.sh` builds a portable pack:

- Copies `reports/final_verdict.{txt,json}` and key `analysis/*` artifacts.
- Collects all certificates into `proof_pack/certs/...`.
- Generates `manifest.json`, `checksums.sha256`, and optional
  `manifest.json.asc`.
- Optional HTML export can be disabled with `PACK_SKIP_HTML=1`.

### Packaging flow

```text
run_pack.sh
  ├─ run_suite.sh → OUTPUT_DIR
  ├─ collect certs + reports
  ├─ write manifest + checksums
  └─ optional HTML + GPG signature
```

`verify_pack.sh` checks the pack:

- Verifies `checksums.sha256`.
- Verifies the GPG signature if `manifest.json.asc` exists.
- Runs `invarlock verify` across all certs (JSON output optional).

## Remote setup helper

`scripts/proof_packs/lib/setup_remote.sh` is an optional bootstrap script for
fresh GPU hosts. It clones the repo, creates a venv, installs PyTorch and
InvarLock, and leaves the host ready to run `run_pack.sh`.

Common knobs for the setup script:

- `REPO_DIR`, `REPO_URL`, `BRANCH`, `PYTHON_BIN`, `VENV_DIR`.
- `TORCH_INDEX_URL`, `TORCH_PACKAGES`, `PACK_SKIP_TORCH_CHECK`.
- `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`.

## Tuning reference

### Core configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PACK_SUITE` | `subset` | Suite name (`subset` or `full`) |
| `PACK_NET` | `0` | Enable network preflight/downloads |
| `PACK_OUTPUT_DIR` | unset | Sets `OUTPUT_DIR` when provided |
| `OUTPUT_DIR` | auto | `./proof_pack_runs/<suite>_<timestamp>` via entrypoint |
| `PACK_OUTPUT_DIR_ABSOLUTE` | `false` | Normalize `OUTPUT_DIR` to absolute path |
| `PACK_SUITE_MODE` | `full` | `full`, `calibrate-only`, or `run-only` |
| `PACK_DETERMINISM` | `throughput` | Harness determinism mode |
| `PACK_REPEATS` | `0` | Determinism repeat metadata |
| `PACK_MODEL_REVISIONS_FILE` | `OUTPUT_DIR/state/model_revisions.json` | Revisions path |
| `PACK_USE_BATCH_EDITS` | `auto` | Force/disable batch edit creation |
| `RESUME_MODE` | `true` | Skip completed steps when outputs exist |

### Hardware selection

| Variable | Default | Description |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | unset | Explicit GPU pool (comma-separated) |
| `GPU_ID_LIST` | unset | Alternate GPU pool list |
| `NUM_GPUS` | auto | Number of GPUs to use (clamped to pool) |
| `GPU_MEMORY_GB` | auto | Per-GPU memory hint for planning |
| `GPU_MEMORY_PER_DEVICE` | `GPU_MEMORY_GB` | Per-device memory for `required_gpus` |
| `GPU_MIN_FREE_GB` | `10` | Minimum free VRAM for eligibility |
| `GPU_REQUIRE_IDLE` | `true` | Require GPUs with no compute processes |
| `GPU_CACHE_TTL` | `5` | GPU cache TTL (seconds) |
| `GPU_RESERVATION_TTL` | `60` | Reservation TTL (seconds) |
| `GPU_RESERVATION_LOCK_TIMEOUT` | `5` | Reservation lock timeout (seconds) |

### Model overrides

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_1`–`MODEL_8` | suite-defined | Override model slots; empty disables |

### InvarLock settings

| Variable | Default | Description |
| --- | --- | --- |
| `INVARLOCK_DATASET` | `wikitext2` | Dataset provider |
| `INVARLOCK_TIER` | `balanced` | Guard tier preset |
| `INVARLOCK_PREVIEW_WINDOWS` | `32` | Preview windows |
| `INVARLOCK_FINAL_WINDOWS` | `32` | Final windows |
| `INVARLOCK_SEQ_LEN` | `512` | Sequence length |
| `INVARLOCK_STRIDE` | `256` | Stride |
| `INVARLOCK_EVAL_BATCH` | `32` | InvarLock batch size |
| `INVARLOCK_PM_ACCEPTANCE_MIN` | `0.90` | Primary metric lower bound |
| `INVARLOCK_PM_ACCEPTANCE_MAX` | `1.20` | Primary metric upper bound |
| `PACK_GUARDS_ORDER` | `invariants,spectral,rmt,variance,invariants` | Guards included in calibration and presets |

### Tuned edit presets

| Variable | Default | Description |
| --- | --- | --- |
| `PACK_TUNED_EDIT_PARAMS_FILE` | unset | JSON file with tuned clean edit params (required when `CLEAN_EDIT_RUNS>0`). |

### Calibration preset reuse

| Variable | Default | Description |
| --- | --- | --- |
| `PACK_CALIBRATION_PRESET_DIR` | unset | Directory containing `calibrated_preset_<model>.yaml/json` to reuse; skips calibration runs. |
| `PACK_CALIBRATION_PRESET_FILE` | unset | Single preset file applied to all models (advanced). |

### Experiment controls

| Variable | Default | Description |
| --- | --- | --- |
| `DRIFT_CALIBRATION_RUNS` | `5` | Calibration run count |
| `CLEAN_EDIT_RUNS` | `3` | Clean edit certify runs |
| `STRESS_EDIT_RUNS` | `2` | Stress edit certify runs |
| `RUN_ERROR_INJECTION` | `true` | Enable error injection |

### Storage and memory planning

| Variable | Default | Description |
| --- | --- | --- |
| `PACK_BASELINE_STORAGE_MODE` | `snapshot_symlink` | Baseline storage mode |
| `MIN_FREE_DISK_GB` | `200` | Disk pressure threshold |
| `PACK_SKIP_DISK_PREFLIGHT` | `0` | Skip storage preflight |
| `CUDA_MEMORY_FRACTION` | `0.92` | Target GPU memory fraction |
| `MODEL_LOAD_OVERHEAD_GB` | `4` | Load overhead for planning |
| `EDIT_OVERHEAD_GB` | `8` | Per-edit overhead for planning |
| `BATCH_EDIT_OVERHEAD_GB` | `8` | Batch edit overhead |
| `INVARLOCK_OVERHEAD_GB` | `6` | InvarLock overhead |

### Worker + reliability controls

| Variable | Default | Description |
| --- | --- | --- |
| `WORKER_HEARTBEAT_INTERVAL` | `30` | Heartbeat interval (seconds) |
| `WORKER_IDLE_SLEEP` | `5` | Sleep when idle (seconds) |
| `WORKER_MAX_FAILURES` | `10` | Stop worker after N failures |
| `WORKER_TIMEOUT` | `2700` | Worker heartbeat timeout (seconds) |
| `CANCEL_BLOCKED_TASKS_GRACE_SECONDS` | `90` | Fail blocked tasks after grace |
| `TASK_TIMEOUT_DEFAULT` | `21600` | Default task timeout (seconds) |
| `TASK_TIMEOUT_<TASKTYPE>` | unset | Per-task timeout override |

### Packaging and verification

| Variable | Default | Description |
| --- | --- | --- |
| `PACK_DIR` | `OUTPUT_DIR/proof_pack` | Proof pack output dir |
| `PACK_GPG_SIGN` | `1` | Sign manifest if `gpg` available |
| `PACK_SKIP_HTML` | `0` | Skip HTML rendering |
| `PACK_VERIFY_PROFILE` | `dev` | Profile for `invarlock verify` |

## Troubleshooting

### Missing model revisions (offline)

If offline runs fail with “requires model revisions”, run a preflight:

```bash
./scripts/proof_packs/run_suite.sh --suite subset --net 1
```

Or point to an existing revisions file with `PACK_MODEL_REVISIONS_FILE`.

### OOM on large models

- Lower `GPU_MEMORY_PER_DEVICE` so the planner requests more GPUs.
- Disable batch edits: `PACK_USE_BATCH_EDITS=false`.
- Reduce InvarLock batch/seq_len (e.g., `INVARLOCK_EVAL_BATCH=16 INVARLOCK_SEQ_LEN=256`).
- Increase memory overhead knobs (`MODEL_LOAD_OVERHEAD_GB`, `EDIT_OVERHEAD_GB`).

### Disk pressure / preflight failures

Check `state/disk_pressure.json` and ensure the output filesystem has headroom.
Use `MIN_FREE_DISK_GB=0` or `PACK_SKIP_DISK_PREFLIGHT=1` only if you accept
risk of partial artifacts.

### Task timeouts

Increase the default or per-task timeout:

```bash
TASK_TIMEOUT_DEFAULT=28800 ./scripts/proof_packs/run_suite.sh --suite subset
TASK_TIMEOUT_CREATE_EDIT=28800 ./scripts/proof_packs/run_suite.sh --suite subset
```

### Stuck queues or dead workers

- Inspect `state/progress.json` and `workers/gpu_<id>.status`.
- Check worker logs: `logs/gpu_<id>.log` and `logs/tasks/<task_id>.log`.
- Re-run with `--resume` to recover from a crash.
