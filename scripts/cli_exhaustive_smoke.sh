#!/usr/bin/env bash
# Exhaustive CLI smoke runner for InvarLock. Runs safe commands (help, list, dry-run)
# and captures outputs to a temporary log for review.

set -uo pipefail

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${INVARLOCK_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(bash "$ROOT/scripts/select_python.sh")"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
printf -v CLI '%q ' "$PYTHON_BIN" -m invarlock
CLI="${CLI% }"

LOG_FILE="$(mktemp -t invarlock_cli_smoke.XXXXXX.log)"

echo "[info] $(ts) CLI runner: $CLI" | tee -a "$LOG_FILE"
echo "[info] $(ts) Log file: $LOG_FILE"

smoke_timeout() {
  local var_name="$1"
  local default_value="$2"
  printf '%s' "${!var_name:-$default_value}"
}

EVALUATE_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_EVALUATE_TIMEOUT 420)"
CALIBRATE_NULL_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_CALIBRATE_NULL_TIMEOUT 900)"
CALIBRATE_VE_TIMEOUT_SECONDS="$(smoke_timeout INVARLOCK_SMOKE_CALIBRATE_VE_TIMEOUT 1200)"
echo "[info] $(ts) Timeout budget: evaluate=${EVALUATE_TIMEOUT_SECONDS}s calibrate_null=${CALIBRATE_NULL_TIMEOUT_SECONDS}s calibrate_ve=${CALIBRATE_VE_TIMEOUT_SECONDS}s" | tee -a "$LOG_FILE"

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
  SMOKE_CMD="$cmd" SMOKE_TIMEOUT="$seconds" "$PYTHON_BIN" - <<'PY' >>"$LOG_FILE" 2>&1
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
  bash -lc "$CLI advanced plugins adapters --json >/dev/null 2>&1" || return 1
  # Try importing torch+transformers for a hard check (quick)
  "$PYTHON_BIN" - <<'PY'
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
run "invarlock evaluate --help"       "$CLI evaluate --help"
run "invarlock verify --help"         "$CLI verify --help"
run "invarlock report --help"         "$CLI report --help"
run "invarlock report generate --help" "$CLI report generate --help"
run "invarlock report explain --help" "$CLI report explain --help"
run "invarlock report html --help"    "$CLI report html --help"
run "invarlock report validate --help" "$CLI report validate --help"
run "invarlock doctor --help"         "$CLI doctor --help"
run "invarlock advanced --help"       "$CLI advanced --help"
run "invarlock advanced proof-pack --help" "$CLI advanced proof-pack --help"
run "invarlock advanced proof-pack build --help" "$CLI advanced proof-pack build --help"
run "invarlock advanced proof-pack inspect --help" "$CLI advanced proof-pack inspect --help"
run "invarlock advanced proof-pack verify --help" "$CLI advanced proof-pack verify --help"
run "invarlock advanced policy --help" "$CLI advanced policy --help"
run "invarlock advanced policy build --help" "$CLI advanced policy build --help"
run "invarlock advanced policy verify --help" "$CLI advanced policy verify --help"
run "invarlock advanced calibrate --help" "$CLI advanced calibrate --help"
run "invarlock advanced calibrate null-sweep --help" "$CLI advanced calibrate null-sweep --help"
run "invarlock advanced calibrate ve-sweep --help" "$CLI advanced calibrate ve-sweep --help"

# Plugins listings (safe; JSON and text variants)
run "invarlock advanced plugins --help"        "$CLI advanced plugins --help"
run "invarlock advanced plugins list --help"   "$CLI advanced plugins list --help"
run "invarlock advanced plugins list (text)"   "$CLI advanced plugins list"
run "invarlock advanced plugins list --json"   "$CLI advanced plugins list --json"

# Category-specific helpers
run "invarlock advanced plugins adapters --help" "$CLI advanced plugins adapters --help"
run "invarlock advanced plugins adapters --json" "$CLI advanced plugins adapters --json"
run "invarlock advanced plugins guards --help"   "$CLI advanced plugins guards --help"
run "invarlock advanced plugins guards --json"   "$CLI advanced plugins guards --json"
run "invarlock advanced plugins edits --help"    "$CLI advanced plugins edits --help"
run "invarlock advanced plugins edits --json"    "$CLI advanced plugins edits --json"

# Extended: verify, evaluate/run with and without network
# Create a tiny invalid report to exercise verify paths
TMP_DIR="$(mktemp -d -t invarlock_cli_smoke.XXXXXX.dir)"
echo '{"schema_version": "v1", "primary_metric": {}}' >"$TMP_DIR/report_invalid.json"
printf '%s\n' '{"verdict":"PASS","summary":{"status":"smoke"}}' >"$TMP_DIR/final_verdict.json"
printf '%s\n' '{"commit":"smoke","branch":"staging/next"}' >"$TMP_DIR/source_repo.json"
printf '%s\n' '{"platform":"cli-smoke","mode":"attested-fixture"}' >"$TMP_DIR/environment.json"
printf '%s\n' '{"models":{"sshleifer/tiny-gpt2":{"revision":"fixture"}}}' >"$TMP_DIR/model_revisions.json"
printf '%s\n' '{"metrics":{"pm_ratio":{"ratio_limit_base":1.1}}}' >"$TMP_DIR/resolved_policy.json"
printf '%s\n' '[{"path":"metrics.pm_ratio.ratio_limit_base","value":1.1}]' >"$TMP_DIR/policy_overrides.json"
printf '%s\n' '{"support_tiers":["published_basis"]}' >"$TMP_DIR/policy_compatibility.json"
PROOF_PACK_REPORT_DIR="$TMP_DIR/proof_pack_report"
mkdir -p "$PROOF_PACK_REPORT_DIR"
PROOF_PACK_REPORT_DIR="$PROOF_PACK_REPORT_DIR" "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

from invarlock.reporting import verify_contract as verify_mod
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
        "ci": [0.0, 0.0],
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
run "invarlock advanced proof-pack build" "$CLI advanced proof-pack build \"$TMP_DIR/proof_pack_cli\" --final-verdict \"$TMP_DIR/final_verdict.json\" --source-repo \"$TMP_DIR/source_repo.json\" --environment \"$TMP_DIR/environment.json\" --material model_revisions=\"$TMP_DIR/model_revisions.json\" --report \"$PROOF_PACK_REPORT_DIR/evaluation.report.json\" --profile ci --json"
run "invarlock advanced proof-pack inspect --json" "$CLI advanced proof-pack inspect \"$TMP_DIR/proof_pack_cli\" --json"
run "invarlock advanced proof-pack verify --json" "$CLI advanced proof-pack verify \"$TMP_DIR/proof_pack_cli\" --json"
run "invarlock advanced policy build" "$CLI advanced policy build --resolved-policy \"$TMP_DIR/resolved_policy.json\" --overrides \"$TMP_DIR/policy_overrides.json\" --compatibility \"$TMP_DIR/policy_compatibility.json\" --out \"$TMP_DIR/policy-pack.json\" --owner smoke"
run "invarlock advanced policy verify --json" "$CLI advanced policy verify \"$TMP_DIR/policy-pack.json\" --json"

# Offline runs (force quick failure if uncached)
OFFLINE_ENV="HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"
OFFLINE_EVAL_ENV="$OFFLINE_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"

if have_adapters_stack; then
  run_to "invarlock evaluate (offline)" "$EVALUATE_TIMEOUT_SECONDS" "$OFFLINE_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_offline\" --report-out \"$TMP_DIR/report_offline_out\""
  run_to "invarlock evaluate (offline, local)" "$EVALUATE_TIMEOUT_SECONDS" "$OFFLINE_EVAL_ENV $CLI evaluate --assurance trusted-local --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_offline_local\" --report-out \"$TMP_DIR/report_offline_local_out\""
  run_to "invarlock advanced calibrate null-sweep (network)" "$CALIBRATE_NULL_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate null-sweep --allow-network --config configs/calibration/null_sweep_ci.yaml --out \"$TMP_DIR/calibrate_null\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
  run_to "invarlock advanced calibrate ve-sweep (network)" "$CALIBRATE_VE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate ve-sweep --allow-network --config configs/calibration/rmt_ve_sweep_ci.yaml --out \"$TMP_DIR/calibrate_ve\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
  run_to "invarlock advanced calibrate null-sweep (network, host)" "$CALIBRATE_NULL_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate null-sweep --allow-network --allow-host-execution --config configs/calibration/null_sweep_ci.yaml --out \"$TMP_DIR/calibrate_null_host\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
  run_to "invarlock advanced calibrate ve-sweep (network, host)" "$CALIBRATE_VE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate ve-sweep --allow-network --allow-host-execution --config configs/calibration/rmt_ve_sweep_ci.yaml --out \"$TMP_DIR/calibrate_ve_host\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
else
  {
    echo "\n==== BEGIN invarlock evaluate (offline) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (offline) ====\n"
    echo "\n==== BEGIN invarlock evaluate (offline, local) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (offline, local) ====\n"
    echo "\n==== BEGIN invarlock advanced calibrate null-sweep (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock advanced calibrate null-sweep (network) ====\n"
    echo "\n==== BEGIN invarlock advanced calibrate ve-sweep (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock advanced calibrate ve-sweep (network) ====\n"
    echo "\n==== BEGIN invarlock advanced calibrate null-sweep (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock advanced calibrate null-sweep (network, host) ====\n"
    echo "\n==== BEGIN invarlock advanced calibrate ve-sweep (network, host) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock advanced calibrate ve-sweep (network, host) ====\n"
  } >>"$LOG_FILE"
fi

# With network allowed (may still fail fast if extras missing)
NET_ENV="INVARLOCK_ALLOW_NETWORK=1 TOKENIZERS_PARALLELISM=false"
NET_EVAL_ENV="$NET_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"
if have_adapters_stack; then
  run_to "invarlock evaluate (network)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_EVAL_ENV $CLI evaluate --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_net\" --report-out \"$TMP_DIR/report_net_out\""
  run "invarlock verify (network output)" "if [ -f \"$TMP_DIR/report_net_out/evaluation.report.json\" ]; then $CLI verify --json \"$TMP_DIR/report_net_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
  run_to "invarlock evaluate (network, local)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_EVAL_ENV $CLI evaluate --assurance trusted-local --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/presets/causal_lm/wikitext2_512.yaml --device cpu --out \"$TMP_DIR/report_net_local\" --report-out \"$TMP_DIR/report_net_local_out\""
  run "invarlock verify (network local output)" "if [ -f \"$TMP_DIR/report_net_local_out/evaluation.report.json\" ]; then $CLI verify --assurance trusted-local --json \"$TMP_DIR/report_net_local_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
else
  {
    echo "\n==== BEGIN invarlock evaluate (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (network) ====\n"
    echo "\n==== BEGIN invarlock verify (network output) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock verify (network output) ====\n"
    echo "\n==== BEGIN invarlock evaluate (network, local) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock evaluate (network, local) ====\n"
    echo "\n==== BEGIN invarlock verify (network local output) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock verify (network local output) ====\n"
  } >>"$LOG_FILE"
fi

echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"
