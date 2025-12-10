# B200 Validation

## Overview
- Purpose: definitive validation suite tuned for 8x NVIDIA B200 180GB nodes.
- Coverage: 8 public models (7B–72B), 4 edit types × 2 strengths each, error injection.
- Hardware: 8× B200 180GB SXM6; target VRAM 85–92% per GPU.
- Cost: ~\$100 for full run; runtime ~2.5 hours with dynamic scheduling.

## Quick Start
```bash
./scripts/invarlock_definitive_validation_b200.sh
./scripts/invarlock_definitive_validation_b200.sh --resume
```

## Model Suite & Edits
- GPU 0: mistralai/Mistral-7B-v0.1 (~14 GB)
- GPU 1: NousResearch/Llama-2-13b-hf (~26 GB)
- GPU 2: Qwen/Qwen2.5-14B (~28 GB)
- GPU 3: Qwen/Qwen2.5-32B (~64 GB)
- GPU 4: 01-ai/Yi-34B (~68 GB)
- GPU 5: mistralai/Mixtral-8x7B-v0.1 (~90 GB, MoE)
- GPU 6: NousResearch/Llama-2-70b-hf (~140 GB)
- GPU 7: Qwen/Qwen1.5-72B (~144 GB)

Edits exercised per model:
- Quantization RTN: 8-bit clean, 4-bit stress
- FP4 Quantization: E2M1 clean, aggressive stress (B200-native)
- Magnitude Pruning: 10% clean, 50% stress
- Low-Rank SVD: rank-256 clean, rank-32 stress

## Dynamic Scheduling Architecture
- Work-stealing GPU workers pull from a shared queue; idle GPUs grab pending work.
- Queue lifecycle: pending → ready → running → completed/failed with dependency resolution.
- Scheduler: fits tasks to GPU free memory; boosts priorities to prevent starvation.
- Worker loop: heartbeat files, per-GPU logs, failure backoff, optional retry/oom recovery.
- Progress monitor: periodic queue stats and work-stealing boosts until all tasks drain.

## Library Modules
- `scripts/lib/task_functions.sh`: atomic task runners (setup, calibrate, certify, error inject).
- `scripts/lib/queue_manager.sh`: file-backed queue operations and dependency updates.
- `scripts/lib/scheduler.sh`: GPU memory probes and task selection, work-stealing boosts.
- `scripts/lib/gpu_worker.sh`: worker lifecycle, heartbeats, failure handling hooks.
- `scripts/lib/task_serialization.sh`: JSON schema for tasks plus safe read/write helpers.
- `scripts/lib/fault_tolerance.sh`: retry/backoff policies, OOM recovery hooks.

## Troubleshooting
- OOM on 70B+: set `INVARLOCK_SKIP_OVERHEAD_CHECK=1` or keep profile `dev` for guard overhead, and ensure only one model loads per GPU.
- Preset not found: rerun calibration to regenerate `presets/calibrated_preset_<model>.yaml`; absolute preset paths are now resolved for subshells.
- Resume mode: rerun with `--resume` and `OUTPUT_DIR=<existing>` to reuse queues; running/failed tasks are re-queued automatically.
- Stuck queue: check `queue/running` for orphaned tasks and `logs/tasks/*.log` for `error_msg` details; progress monitor reports counts every minute.

## Environment Variables
- `INVARLOCK_SNAPSHOT_MODE={auto|bytes|chunked}`: force snapshot strategy during runs.
- `INVARLOCK_SKIP_OVERHEAD_CHECK=1`: skip guard-overhead measurement (avoids double-load on large models in ci/release profiles).
- `USE_DYNAMIC_SCHEDULING=true|false`: toggle work-stealing scheduler (default: true).
