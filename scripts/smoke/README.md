# Smoke Scripts

`scripts/smoke/` contains repo-maintainer smoke entry points. These are not
install-time APIs. Prefer invoking them through the listed workflow or `make`
target when one exists.

| Script | Primary caller | Network | Runtime mode | Main output |
| --- | --- | --- | --- | --- |
| `cli_smoke_fast.sh` | Manual CLI surface checks; lane in `cli_smoke_matrix.sh` | Optional | Host and container when available | Log file plus temporary reports under the work root |
| `cli_smoke_negative.sh` | Manual negative-path CLI checks; lane in `cli_smoke_matrix.sh` | Never | Host | Fixture reports and verify outputs under the work root |
| `cli_smoke_matrix.sh` | Manual lane dispatcher | Optional | Delegates per lane | Consolidated lane log |
| `run_gpt2_user_journey_smoke.sh` | `.github/workflows/gpt2-smoke.yml`; realistic lane | Required unless cache is seeded | Host, container, or both | `journey-results.tsv`, `final_verdict.json`, reports, HTML |
| `run_tiny_container_smoke.sh` | `.github/workflows/tiny-container-smoke.yml` | Required unless cache is seeded | Container by default; host with `INVARLOCK_SMOKE_MODE=local` | Evaluation report, HTML, optional evidence pack |
| `run_tiny_all_matrix.sh` | Manual tiny model matrix | Optional; `NET=1` enables downloads | Container by default through `evaluate`; dry-run by default | `checklist.md`, and reports when `RUN=1` |
| `run_tiny_fine_tune_byoe_smoke.py` | Manual BYOE fine-tune smoke | Never by default; `--allow-network` enables cache fill | Host CPU | Baseline/subject checkpoints, enriched evaluation report, verify JSON, smoke summary |
| `run_cpu_telemetry.sh` | Manual telemetry sweep | Required | Evaluate default execution path | Telemetry reports under `reports/telemetry/cpu-ci` |
| `check_device_drift.py` | Assurance docs and tests | Never | N/A | Exit status and drift message |
| `gpt2_journey_helpers.py` | Helper for `run_gpt2_user_journey_smoke.sh` | Never directly | N/A | TSV summaries, final verdicts, strict-bundle fixtures |
| `guard_validation_smoke.py` | `make guard-validation-smoke` | Never | N/A | Synthetic guard validation JSON and Markdown |
| `lib/smoke_common.sh` | Shared shell helper sourced by smoke scripts | Never directly | N/A | Python selection, timestamps, cache, runtime-image helpers |

## Maintenance Notes

- Keep front-door scripts small enough to read, but do not merge all smoke lanes
  into one script. Shared shell helpers are the preferred consolidation path.
- Shared runtime plumbing lives in `lib/smoke_common.sh`. Keep new smoke scripts
  on that helper instead of copying Python selection, timestamp, cache, or
  runtime-image functions.
- GPT-2 journey report shaping and strict-bundle fixture generation live in
  `gpt2_journey_helpers.py`; keep heavy JSON/TSV transforms out of shell.
- `cli_smoke_fast.sh` remains the main future extraction candidate because it
  still owns several embedded fixtures and command result helpers.
- `cli_smoke_matrix.sh` is a lane dispatcher. Keep it focused on orchestration;
  lane behavior belongs in the lane scripts it invokes.
- `run_cpu_telemetry.sh` is telemetry-oriented rather than a normal pass/fail
  CI smoke gate. Keep that distinction visible in docs and summaries.
- `run_tiny_fine_tune_byoe_smoke.py` is the local fine-tune BYOE realism lane:
  it should remain offline by default and should enrich a real evaluation report
  only after the baseline and subject checkpoints have been materialized.
