#!/usr/bin/env bash
# Exhaustive CLI smoke runner for InvarLock. Runs safe commands (help, list, dry-run)
# and captures outputs to a temporary log for review.

set -uo pipefail

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

# Resolve CLI runner: prefer installed `invarlock`, else use `python -m invarlock` with local src path.
if command -v invarlock >/dev/null 2>&1; then
  CLI="invarlock"
else
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
  CLI="python -m invarlock"
fi

LOG_FILE="$(mktemp -t invarlock_cli_smoke.XXXXXX.log)"

echo "[info] $(ts) CLI runner: $CLI" | tee -a "$LOG_FILE"
echo "[info] $(ts) Log file: $LOG_FILE"

smoke_timeout() {
  local var_name="$1"
  local default_value="$2"
  printf '%s' "${!var_name:-$default_value}"
}

RUN_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_RUN_TIMEOUT 180)"
EVALUATE_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_EVALUATE_TIMEOUT 420)"
CALIBRATE_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_CALIBRATE_TIMEOUT 420)"
echo "[info] $(ts) Timeout budget: run=${RUN_TIMEOUT_SECONDS}s evaluate=${EVALUATE_TIMEOUT_SECONDS}s calibrate=${CALIBRATE_TIMEOUT_SECONDS}s" | tee -a "$LOG_FILE"

ensure_writable_hf_cache() {
  local candidate_root=""
  if [[ -n "${HF_HOME:-}" ]]; then
    candidate_root="${HF_HOME}"
  else
    candidate_root="${HOME}/.cache/huggingface"
  fi

  local probe_dir="${HF_DATASETS_CACHE:-${candidate_root}/datasets}"
  if mkdir -p "$probe_dir" >/dev/null 2>&1 && touch "$probe_dir/.ivl_smoke_probe" >/dev/null 2>&1; then
    rm -f "$probe_dir/.ivl_smoke_probe" >/dev/null 2>&1 || true
    return
  fi

  local smoke_cache_root
  smoke_cache_root="$(mktemp -d -t invarlock_cli_hf_cache.XXXXXX)"
  export HF_HOME="$smoke_cache_root"
  export HF_DATASETS_CACHE="$smoke_cache_root/datasets"
  mkdir -p "$HF_DATASETS_CACHE"
  echo "[info] $(ts) Falling back to writable HF cache: $smoke_cache_root" | tee -a "$LOG_FILE"
}

ensure_writable_hf_cache

# Run a single command string via bash -lc, capturing stdout+stderr and exit code.
run() {
  local label="$1"
  local cmd="$2"
  {
    echo "\n==== BEGIN $label ===="
    echo "[cmd] $cmd"
    echo "[ts] $(ts)"
  } >>"$LOG_FILE"
  set +e
  bash -lc "$cmd" >>"$LOG_FILE" 2>&1
  local ec=$?
  set -e
  {
    echo "[exit_code] $ec"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

# Run with a timeout (seconds). Uses python subprocess + bash -lc for parity.
run_to() {
  local label="$1"; shift
  local seconds="$1"; shift
  local cmd="$1"
  {
    echo "\n==== BEGIN $label (timeout=${seconds}s) ===="
    echo "[cmd] $cmd"
    echo "[ts] $(ts)"
  } >>"$LOG_FILE"
  set +e
  SMOKE_CMD="$cmd" SMOKE_TIMEOUT="$seconds" python - <<'PY' >>"$LOG_FILE" 2>&1
import os, subprocess, sys
cmd = os.environ.get("SMOKE_CMD", "")
if not cmd:
    print("[error] SMOKE_CMD not set")
    sys.exit(1)
try:
    to = float(os.environ.get("SMOKE_TIMEOUT", "60"))
    cp = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, timeout=to )
    sys.stdout.write(cp.stdout)
    sys.stderr.write(cp.stderr)
    rc = cp.returncode
except subprocess.TimeoutExpired as te:
    out = te.stdout or b""
    err = te.stderr or b""
    if isinstance(out, bytes):
        try:
            out = out.decode("utf-8", errors="replace")
        except Exception:
            out = ""
    if isinstance(err, bytes):
        try:
            err = err.decode("utf-8", errors="replace")
        except Exception:
            err = ""
    sys.stdout.write(out)
    sys.stderr.write(err)
    rc = 124
    print(f"[timeout] command exceeded {os.environ.get('SMOKE_TIMEOUT', '60')}s")
sys.exit(rc)
PY
  local ec=$?
  set -e
  {
    echo "[exit_code] $ec"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

# Conditionally run long commands (model/dataset) only when adapters stack exists
have_adapters_stack() {
  bash -lc "$CLI plugins adapters --json >/dev/null 2>&1" || return 1
  # Try importing torch+transformers for a hard check (quick)
  python - <<'PY'
import sys
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

run_env() {
  local label="$1"
  shift
  # Remaining args: environment assignments + command
  run "$label" "$*"
}

# Top-level and core commands (help-only: safe)
run "invarlock --help"                "$CLI --help"
run "invarlock version"               "$CLI version"
run "invarlock evaluate --help"        "$CLI evaluate --help"
run "invarlock verify --help"         "$CLI verify --help"
run "invarlock run --help"            "$CLI run --help"
run "invarlock calibrate --help"      "$CLI calibrate --help"
run "invarlock calibrate null-sweep --help" "$CLI calibrate null-sweep --help"
run "invarlock calibrate ve-sweep --help" "$CLI calibrate ve-sweep --help"
run "invarlock proof-pack --help"     "$CLI proof-pack --help"
run "invarlock proof-pack build --help" "$CLI proof-pack build --help"
run "invarlock proof-pack inspect --help" "$CLI proof-pack inspect --help"
run "invarlock proof-pack verify --help" "$CLI proof-pack verify --help"
run "invarlock report --help"         "$CLI report --help"
run "invarlock report verify --help"  "$CLI report verify --help"
run "invarlock report explain --help" "$CLI report explain --help"
run "invarlock report html --help"    "$CLI report html --help"
run "invarlock report validate --help" "$CLI report validate --help"
run "invarlock doctor --help"         "$CLI doctor --help"

# Plugins listings (safe; JSON and text variants)
run "invarlock plugins --help"        "$CLI plugins --help"
run "invarlock plugins list --help"   "$CLI plugins list --help"
run "invarlock plugins list (text)"   "$CLI plugins list"
run "invarlock plugins list --json"   "$CLI plugins list --json"
run "invarlock plugins list adapters --json" "$CLI plugins list adapters --json"
run "invarlock plugins list guards --json"   "$CLI plugins list guards --json"
run "invarlock plugins list edits --json"    "$CLI plugins list edits --json"
run "invarlock plugins list datasets --json" "$CLI plugins list datasets --json"
run "invarlock plugins list plugins --json"  "$CLI plugins list plugins --json"

# Category-specific helpers
run "invarlock plugins adapters --help" "$CLI plugins adapters --help"
run "invarlock plugins adapters --json" "$CLI plugins adapters --json"
run "invarlock plugins guards --help"   "$CLI plugins guards --help"
run "invarlock plugins guards --json"   "$CLI plugins guards --json"
run "invarlock plugins edits --help"    "$CLI plugins edits --help"
run "invarlock plugins edits --json"    "$CLI plugins edits --json"

# Install/uninstall dry runs (safe, no side effects without --apply)
run "invarlock plugins install --dry-run gpu"  "$CLI plugins install --dry-run gpu"
run "invarlock plugins install --dry-run gptq" "$CLI plugins install --dry-run gptq"
run "invarlock plugins install --dry-run awq"  "$CLI plugins install --dry-run awq"
run "invarlock plugins uninstall --dry-run gpu"  "$CLI plugins uninstall --dry-run gpu"
run "invarlock plugins uninstall --dry-run gptq" "$CLI plugins uninstall --dry-run gptq"
run "invarlock plugins uninstall --dry-run awq"  "$CLI plugins uninstall --dry-run awq"

# Extended: verify, evaluate/run with and without network
# Create a tiny invalid report to exercise verify paths
TMP_DIR="$(mktemp -d -t invarlock_cli_smoke.XXXXXX.dir)"
echo '{"schema_version": "v1", "primary_metric": {}}' >"$TMP_DIR/report_invalid.json"
printf '%s\n' '{"verdict":"PASS","summary":{"status":"smoke"}}' >"$TMP_DIR/final_verdict.json"
printf '%s\n' '{"commit":"smoke","branch":"staging/next"}' >"$TMP_DIR/source_repo.json"
printf '%s\n' '{"platform":"cli-smoke","mode":"attested-fixture"}' >"$TMP_DIR/environment.json"
printf '%s\n' '{"models":{"sshleifer/tiny-gpt2":{"revision":"fixture"}}}' >"$TMP_DIR/model_revisions.json"
PROOF_PACK_REPORT_DIR="$TMP_DIR/proof_pack_report"
mkdir -p "$PROOF_PACK_REPORT_DIR"
PROOF_PACK_REPORT_DIR="$PROOF_PACK_REPORT_DIR" python - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

from invarlock.cli.commands import verify as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


report_dir = Path(os.environ["PROOF_PACK_REPORT_DIR"])
report_path = report_dir / "evaluation.report.json"
spectral_contract = {
    "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
}
rmt_contract = {
    "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
    "activation_sampling": {
        "windows": {"count": 8, "indices_policy": "evenly_spaced"}
    },
}
report_payload = {
    "schema_version": "v1",
    "run_id": "proof-pack-cli-smoke",
    "artifacts": {"generated_at": "2024-01-01T00:00:00"},
    "plugins": {},
    "meta": {},
    "provenance": {
        "provider_digest": {
            "ids_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
    },
    "dataset": {
        "provider": "unit",
        "seq_len": 8,
        "windows": {
            "preview": 2,
            "final": 2,
            "stats": {
                "window_match_fraction": 1.0,
                "window_overlap_fraction": 0.0,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                "paired_windows": 2,
            },
        },
    },
    "validation": {
        "primary_metric_acceptable": True,
        "preview_final_drift_acceptable": True,
        "invariants_pass": True,
        "spectral_stable": True,
        "rmt_stable": True,
    },
    "baseline_ref": {
        "run_id": "baseline-run",
        "model_id": "model",
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
    },
    "artifacts_extra": {},
    "primary_metric": {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
        "ratio_vs_baseline": 1.0,
        "ci": [1.0, 1.0],
        "display_ci": [1.0, 1.0],
    },
    "spectral": {
        "evaluated": True,
        "measurement_contract": spectral_contract,
        "measurement_contract_hash": verify_mod._measurement_contract_digest(
            spectral_contract
        ),
        "measurement_contract_match": True,
    },
    "rmt": {
        "evaluated": True,
        "measurement_contract": rmt_contract,
        "measurement_contract_hash": verify_mod._measurement_contract_digest(
            rmt_contract
        ),
        "measurement_contract_match": True,
    },
    "resolved_policy": {
        "spectral": {"measurement_contract": spectral_contract},
        "rmt": {"measurement_contract": rmt_contract},
    },
    "evaluation_windows": {
        "final": {
            "logloss": [math.log(10.0)],
            "token_counts": [1],
        }
    },
}
report_path.write_text(json.dumps(report_payload, sort_keys=True), encoding="utf-8")
runtime_manifest = {
    "manifest_version": 1,
    "generated_at_utc": "2026-03-21T00:00:00+00:00",
    "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
    "execution_mode": "container",
    "report": {
        "filename": report_path.name,
        "path": report_path.as_posix(),
        "sha256": sha256_file(report_path),
    },
    "config": {
        "path": None,
        "sha256": None,
        "source": "missing",
    },
    "runtime": {
        "container_execution": True,
        "image_digest": "sha256:" + ("a" * 64),
        "image_ref": "invarlock-runtime:local",
        "allow_network": False,
        "allow_remote_code": False,
        "allow_third_party_plugins": False,
    },
}
(report_dir / RUNTIME_MANIFEST_FILENAME).write_text(
    json.dumps(runtime_manifest, sort_keys=True), encoding="utf-8"
)
PY

run "invarlock verify (human, invalid)" "$CLI verify \"$TMP_DIR/report_invalid.json\""
run "invarlock verify --json (invalid)" "$CLI verify --json \"$TMP_DIR/report_invalid.json\""
run "invarlock proof-pack build" "$CLI proof-pack build \"$TMP_DIR/proof_pack_cli\" --final-verdict \"$TMP_DIR/final_verdict.json\" --source-repo \"$TMP_DIR/source_repo.json\" --environment \"$TMP_DIR/environment.json\" --material model_revisions=\"$TMP_DIR/model_revisions.json\" --report \"$PROOF_PACK_REPORT_DIR/evaluation.report.json\" --profile ci --json"
run "invarlock proof-pack inspect --json" "$CLI proof-pack inspect \"$TMP_DIR/proof_pack_cli\" --json"
run "invarlock proof-pack verify --json" "$CLI proof-pack verify \"$TMP_DIR/proof_pack_cli\" --json"

# Offline runs (force quick failure if uncached)
OFFLINE_ENV="HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"
OFFLINE_EVAL_ENV="$OFFLINE_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"
OFFLINE_HOST_ENV="$OFFLINE_ENV INVARLOCK_ALLOW_HOST_EXECUTION=1"
OFFLINE_HOST_EVAL_ENV="$OFFLINE_EVAL_ENV INVARLOCK_ALLOW_HOST_EXECUTION=1"

if have_adapters_stack; then
  run_to "invarlock run (offline)" "$RUN_TIMEOUT_SECONDS" "$OFFLINE_ENV $CLI run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci --device cpu --out \"$TMP_DIR/run_offline\""
  run_to "invarlock evaluate (offline)" "$EVALUATE_TIMEOUT_SECONDS" "$OFFLINE_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_offline\" --report-out \"$TMP_DIR/report_offline_out\""
  run_to "invarlock run (offline, host)" "$RUN_TIMEOUT_SECONDS" "$OFFLINE_HOST_ENV $CLI run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci --device cpu --out \"$TMP_DIR/run_offline_host\""
  run_to "invarlock evaluate (offline, host)" "$EVALUATE_TIMEOUT_SECONDS" "$OFFLINE_HOST_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_offline_host\" --report-out \"$TMP_DIR/report_offline_host_out\""
  run_to "invarlock calibrate null-sweep (network)" "$CALIBRATE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI calibrate null-sweep --allow-network --config configs/calibration/null_sweep_ci.yaml --out \"$TMP_DIR/calibrate_null\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
  run_to "invarlock calibrate ve-sweep (network)" "$CALIBRATE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI calibrate ve-sweep --allow-network --config configs/calibration/rmt_ve_sweep_ci.yaml --out \"$TMP_DIR/calibrate_ve\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
  run_to "invarlock calibrate null-sweep (network, host)" "$CALIBRATE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI calibrate null-sweep --allow-network --allow-host-execution --config configs/calibration/null_sweep_ci.yaml --out \"$TMP_DIR/calibrate_null_host\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
  run_to "invarlock calibrate ve-sweep (network, host)" "$CALIBRATE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI calibrate ve-sweep --allow-network --allow-host-execution --config configs/calibration/rmt_ve_sweep_ci.yaml --out \"$TMP_DIR/calibrate_ve_host\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
else
  {
    echo "\n==== BEGIN invarlock run (offline) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (offline) ====\n"
    echo "\n==== BEGIN invarlock evaluate (offline) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (offline) ====\n"
    echo "\n==== BEGIN invarlock run (offline, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (offline, host) ====\n"
    echo "\n==== BEGIN invarlock evaluate (offline, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (offline, host) ====\n"
    echo "\n==== BEGIN invarlock calibrate null-sweep (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock calibrate null-sweep (network) ====\n"
    echo "\n==== BEGIN invarlock calibrate ve-sweep (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock calibrate ve-sweep (network) ====\n"
    echo "\n==== BEGIN invarlock calibrate null-sweep (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock calibrate null-sweep (network, host) ====\n"
    echo "\n==== BEGIN invarlock calibrate ve-sweep (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock calibrate ve-sweep (network, host) ====\n"
  } >>"$LOG_FILE"
fi

# With network allowed (may still fail fast if extras missing)
NET_ENV="INVARLOCK_ALLOW_NETWORK=1 TOKENIZERS_PARALLELISM=false"
NET_EVAL_ENV="$NET_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"
NET_HOST_ENV="$NET_ENV INVARLOCK_ALLOW_HOST_EXECUTION=1"
NET_HOST_EVAL_ENV="$NET_EVAL_ENV INVARLOCK_ALLOW_HOST_EXECUTION=1"
if have_adapters_stack; then
  run_to "invarlock run (network)" "$RUN_TIMEOUT_SECONDS" "$NET_ENV $CLI run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci --device cpu --out \"$TMP_DIR/run_net\""
  run_to "invarlock evaluate (network)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_net\" --report-out \"$TMP_DIR/report_net_out\""
  run "invarlock verify (network output)" "if [ -f \"$TMP_DIR/report_net_out/evaluation.report.json\" ]; then $CLI verify --json \"$TMP_DIR/report_net_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
  run_to "invarlock run (network, host)" "$RUN_TIMEOUT_SECONDS" "$NET_HOST_ENV $CLI run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci --device cpu --out \"$TMP_DIR/run_net_host\""
  run_to "invarlock evaluate (network, host)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_HOST_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_net_host\" --report-out \"$TMP_DIR/report_net_host_out\""
  run "invarlock verify (network host output)" "if [ -f \"$TMP_DIR/report_net_host_out/evaluation.report.json\" ]; then $CLI verify --allow-unattested-artifacts --json \"$TMP_DIR/report_net_host_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
else
  {
    echo "\n==== BEGIN invarlock run (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (network) ====\n"
    echo "\n==== BEGIN invarlock evaluate (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (network) ====\n"
    echo "\n==== BEGIN invarlock verify (network output) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock verify (network output) ====\n"
    echo "\n==== BEGIN invarlock run (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (network, host) ====\n"
    echo "\n==== BEGIN invarlock evaluate (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (network, host) ====\n"
    echo "\n==== BEGIN invarlock verify (network host output) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock verify (network host output) ====\n"
  } >>"$LOG_FILE"
fi

echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"
