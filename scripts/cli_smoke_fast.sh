#!/usr/bin/env bash
# Fast CLI smoke lane for InvarLock.
#
# This lane is intentionally PR-friendly:
# - broad command-surface coverage
# - positive-path report/proof-pack/policy/calibration commands
# - tiny-model evaluate parity across trusted-local and container execution

set -euo pipefail

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${INVARLOCK_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(bash "$ROOT/scripts/select_workspace_python.sh")"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
printf -v CLI '%q ' "$PYTHON_BIN" -m invarlock
CLI="${CLI% }"

LOG_FILE="${INVARLOCK_SMOKE_LOG_FILE:-$(mktemp -t invarlock_cli_fast_smoke.XXXXXX.log)}"
TOTAL_COMMANDS=0
UNEXPECTED_FAILURES=0
SKIPPED_COMMANDS=0

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

have_network_access() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import socket
import sys

try:
    socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}

have_docker_daemon() {
  command -v docker >/dev/null 2>&1 || return 1
  docker info >/dev/null 2>&1
}

have_smoke_model_cache() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys

try:
    from transformers import AutoTokenizer
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
    sys.exit(1)

try:
    AutoTokenizer.from_pretrained(
        "sshleifer/tiny-gpt2",
        local_files_only=True,
        trust_remote_code=False,
    )
except Exception:
    sys.exit(1)

sys.exit(0)
PY
}

assert_tiny_eval_parity() {
  local container_report="$1"
  local local_report="$2"
  CONTAINER_REPORT="$container_report" LOCAL_REPORT="$local_report" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

from invarlock.reporting.report_policy import resolve_tiny_relax_from_report

container_path = Path(os.environ["CONTAINER_REPORT"])
local_path = Path(os.environ["LOCAL_REPORT"])

container = json.loads(container_path.read_text(encoding="utf-8"))
local = json.loads(local_path.read_text(encoding="utf-8"))

if not resolve_tiny_relax_from_report(container):
    raise SystemExit("container tiny smoke report is missing tiny_relax provenance")
if not resolve_tiny_relax_from_report(local):
    raise SystemExit("trusted-local tiny smoke report is missing tiny_relax provenance")

container_validation = container.get("validation") or {}
local_validation = local.get("validation") or {}
keys = (
    "primary_metric_acceptable",
    "preview_final_drift_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
)
for key in keys:
    if container_validation.get(key) != local_validation.get(key):
        raise SystemExit(
            f"container/local tiny smoke mismatch for validation.{key}: "
            f"{container_validation.get(key)!r} != {local_validation.get(key)!r}"
        )

container_metric = container.get("primary_metric") or {}
local_metric = local.get("primary_metric") or {}
att_ratio = container_metric.get("ratio_vs_baseline")
local_ratio = local_metric.get("ratio_vs_baseline")
if isinstance(att_ratio, (int, float)) and isinstance(local_ratio, (int, float)):
    if math.isfinite(att_ratio) and math.isfinite(local_ratio):
        if not math.isclose(float(att_ratio), float(local_ratio), rel_tol=1e-6, abs_tol=1e-6):
            raise SystemExit(
                "container/local tiny smoke mismatch for primary_metric.ratio_vs_baseline: "
                f"{att_ratio!r} != {local_ratio!r}"
            )
PY
}

run_tiny_eval_parity() {
  local label="$1"
  local container_report="$2"
  local local_report="$3"
  {
    echo "\n==== BEGIN $label ===="
    echo "[container_report] $container_report"
    echo "[local_report] $local_report"
    echo "[ts] $(ts)"
  } >>"$LOG_FILE"
  set +e
  assert_tiny_eval_parity "$container_report" "$local_report" >>"$LOG_FILE" 2>&1
  local ec=$?
  set -e
  {
    echo "[exit_code] $ec"
    record_result "$label" "$ec" "0"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

expected_exit_match() {
  local actual="$1"
  local expected_csv="${2:-0}"
  local expected=""
  IFS=',' read -r -a expected <<<"$expected_csv"
  for expected in "${expected[@]}"; do
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
  done
  return 1
}

record_result() {
  local label="$1"
  local ec="$2"
  local expected="${3:-0}"
  local status="pass"

  TOTAL_COMMANDS=$((TOTAL_COMMANDS + 1))
  if ! expected_exit_match "$ec" "$expected"; then
    status="fail"
    UNEXPECTED_FAILURES=$((UNEXPECTED_FAILURES + 1))
  fi

  {
    echo "[expected_exit_codes] $expected"
    echo "[status] $status"
  } >>"$LOG_FILE"
}

skip_run() {
  local label="$1"
  local reason="$2"
  TOTAL_COMMANDS=$((TOTAL_COMMANDS + 1))
  SKIPPED_COMMANDS=$((SKIPPED_COMMANDS + 1))
  {
    echo "\n==== BEGIN $label ===="
    echo "[skip] $reason"
    echo "[expected_exit_codes] skip"
    echo "[status] skip"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

SMOKE_MODEL_ID="${INVARLOCK_SMOKE_MODEL_ID:-sshleifer/tiny-gpt2}"
SMOKE_PRESET="${INVARLOCK_SMOKE_PRESET:-configs/presets/causal_lm/wikitext2_512.yaml}"
SMOKE_CALIBRATE_NULL_CONFIG="${INVARLOCK_SMOKE_CALIBRATE_NULL_CONFIG:-configs/calibration/null_sweep_smoke.yaml}"
SMOKE_CALIBRATE_VE_CONFIG="${INVARLOCK_SMOKE_CALIBRATE_VE_CONFIG:-configs/calibration/rmt_ve_sweep_smoke.yaml}"

# Run a single command string via bash -lc, capturing stdout+stderr and exit code.
run() {
  local label="$1"
  local cmd="$2"
  local expected="${3:-0}"
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
    record_result "$label" "$ec" "$expected"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

# Run with a timeout (seconds). Uses python subprocess + bash -lc for parity.
run_to() {
  local label="$1"; shift
  local seconds="$1"; shift
  local cmd="$1"
  local expected="${2:-0}"
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
        out = out.decode("utf-8", errors="replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", errors="replace")
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
    record_result "$label" "$ec" "$expected"
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
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
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
run "invarlock advanced proof-pack keygen --help" "$CLI advanced proof-pack keygen --help"
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
run "invarlock doctor --json"                    "$CLI doctor --json"

# Extended: verify and evaluate with and without network
TMP_DIR="$(mktemp -d -t invarlock_cli_smoke.XXXXXX.dir)"
printf '%s\n' '{"verdict":"PASS","summary":{"status":"smoke"}}' >"$TMP_DIR/final_verdict.json"
printf '%s\n' '{"commit":"smoke","branch":"staging/next"}' >"$TMP_DIR/source_repo.json"
printf '%s\n' '{"platform":"cli-smoke","mode":"container-fixture"}' >"$TMP_DIR/environment.json"
printf '%s\n' '{"models":{"sshleifer/tiny-gpt2":{"revision":"fixture"}}}' >"$TMP_DIR/model_revisions.json"
printf '%s\n' '{"metrics":{"pm_ratio":{"ratio_limit_base":1.1}}}' >"$TMP_DIR/resolved_policy.json"
printf '%s\n' '[{"path":"metrics.pm_ratio.ratio_limit_base","value":1.1}]' >"$TMP_DIR/policy_overrides.json"
printf '%s\n' '{"support_tiers":["published_basis"]}' >"$TMP_DIR/policy_compatibility.json"
PROOF_PACK_SIGNING_KEY="$TMP_DIR/proof_pack_signing_key.pem"
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
subject_report_path = report_dir.parent / "runs" / "subject" / "report.json"
baseline_report_path = report_dir.parent / "runs" / "source" / "report.json"
subject_report_path.parent.mkdir(parents=True, exist_ok=True)
baseline_report_path.parent.mkdir(parents=True, exist_ok=True)
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
subject_report = {
    "meta": {
        "model_id": "docs-demo-model",
        "adapter": "hf_causal",
        "commit": "docs-demo",
        "seed": 42,
        "device": "cpu",
        "ts": "2026-04-03T00:00:00+00:00",
        "auto": {
            "enabled": False,
            "tier": "balanced",
            "probes_used": 0,
            "target_pm_ratio": None,
        },
    },
    "data": {
        "dataset": "unit",
        "split": "validation",
        "seq_len": 8,
        "stride": 8,
        "preview_n": 2,
        "final_n": 2,
    },
    "edit": {
        "name": "noop",
        "plan_digest": "docs-demo",
        "deltas": {
            "params_changed": 0,
            "sparsity": None,
            "bitwidth_map": None,
            "layers_modified": 0,
        },
    },
    "guards": [],
    "metrics": {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
        },
        "bootstrap": {
            "method": "percentile",
            "replicates": 50,
            "alpha": 0.05,
            "seed": 0,
            "coverage": {
                "preview": {"used": 2},
                "final": {"used": 2},
            },
        },
        "paired_delta_summary": {"mean": 0.0},
        "preview_total_tokens": 50000,
        "final_total_tokens": 50000,
        "logloss_delta": 0.0,
        "logloss_delta_ci": [-0.01, 0.01],
    },
    "artifacts": {
        "events_path": "",
        "logs_path": "",
        "checkpoint_path": None,
    },
    "flags": {
        "guard_recovered": False,
        "rollback_reason": None,
    },
    "evaluation_windows": {
        "preview": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.31],
            "token_counts": [100, 100],
        },
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.31],
            "token_counts": [100, 100],
        },
    },
}
baseline_report = {
    "run_id": "docs-demo-base",
    "model_id": "docs-demo-model",
    "meta": {"seed": 0, "model_id": "docs-demo-model"},
    "evaluation_windows": {
        "preview": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.30],
            "token_counts": [100, 100],
        },
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.30],
            "token_counts": [100, 100],
        },
    },
    "data": {
        "seq_len": 8,
        "preview_n": 2,
        "final_n": 2,
        "dataset": "unit",
        "split": "validation",
        "stride": 8,
    },
    "edit": {
        "name": "none",
        "plan_digest": "0",
        "deltas": {
            "params_changed": 0,
            "layers_modified": 0,
            "sparsity": None,
            "bitwidth_map": None,
        },
    },
    "guards": [],
    "metrics": {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
        },
    },
    "artifacts": {
        "events_path": "",
        "logs_path": "",
        "checkpoint_path": None,
    },
    "flags": {
        "guard_recovered": False,
        "rollback_reason": None,
    },
}
subject_report_path.write_text(json.dumps(subject_report, sort_keys=True), encoding="utf-8")
baseline_report_path.write_text(json.dumps(baseline_report, sort_keys=True), encoding="utf-8")
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

run "invarlock verify --json (fixture report)" "$CLI verify --json --profile ci \"$PROOF_PACK_REPORT_DIR/evaluation.report.json\""
run "invarlock report generate (demo run reports)" "$CLI report generate --run \"$TMP_DIR/runs/subject/report.json\" --baseline-run-report \"$TMP_DIR/runs/source/report.json\" --format report -o \"$TMP_DIR/generated_report\""
run "invarlock report validate (demo generated report)" "$CLI report validate \"$TMP_DIR/generated_report/evaluation.report.json\""
run "invarlock report html (demo generated report)" "$CLI report html -i \"$TMP_DIR/generated_report/evaluation.report.json\" -o \"$TMP_DIR/generated_report/evaluation.html\" --force"
run "invarlock report explain (demo run reports)" "$CLI report explain --subject-report \"$TMP_DIR/runs/subject/report.json\" --baseline-report \"$TMP_DIR/runs/source/report.json\""
run "invarlock advanced proof-pack keygen --json" "$CLI advanced proof-pack keygen \"$PROOF_PACK_SIGNING_KEY\" --json"
run "invarlock advanced proof-pack build" "$CLI advanced proof-pack build \"$TMP_DIR/proof_pack_cli\" --final-verdict \"$TMP_DIR/final_verdict.json\" --source-repo \"$TMP_DIR/source_repo.json\" --environment \"$TMP_DIR/environment.json\" --material model_revisions=\"$TMP_DIR/model_revisions.json\" --report \"$PROOF_PACK_REPORT_DIR/evaluation.report.json\" --signing-key \"$PROOF_PACK_SIGNING_KEY\" --profile ci --json"
run "invarlock advanced proof-pack inspect --json" "$CLI advanced proof-pack inspect \"$TMP_DIR/proof_pack_cli\" --json"
run "invarlock advanced proof-pack verify --json" "$CLI advanced proof-pack verify \"$TMP_DIR/proof_pack_cli\" --json"
run "invarlock advanced policy build" "$CLI advanced policy build --resolved-policy \"$TMP_DIR/resolved_policy.json\" --overrides \"$TMP_DIR/policy_overrides.json\" --compatibility \"$TMP_DIR/policy_compatibility.json\" --out \"$TMP_DIR/policy-pack.json\" --owner smoke"
run "invarlock advanced policy verify --json" "$CLI advanced policy verify \"$TMP_DIR/policy-pack.json\" --json"

# Offline runs (force quick failure if uncached)
OFFLINE_ENV="HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"
OFFLINE_EVAL_ENV="$OFFLINE_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"

if have_adapters_stack; then
  if have_smoke_model_cache; then
    run_to "invarlock evaluate (offline, local)" "$EVALUATE_TIMEOUT_SECONDS" "$OFFLINE_EVAL_ENV $CLI evaluate --execution-mode trusted-local --baseline \"$SMOKE_MODEL_ID\" --subject \"$SMOKE_MODEL_ID\" --adapter auto --profile dev --preset \"$SMOKE_PRESET\" --device cpu --out \"$TMP_DIR/report_offline_local\" --report-out \"$TMP_DIR/report_offline_local_out\""
  else
    skip_run "invarlock evaluate (offline, local)" "smoke model cache not available"
  fi
else
  skip_run "invarlock evaluate (offline, local)" "adapters stack (torch/transformers) not available"
fi

# With network allowed (may still fail fast if extras missing)
NET_ENV="INVARLOCK_ALLOW_NETWORK=1 TOKENIZERS_PARALLELISM=false"
NET_EVAL_ENV="$NET_ENV INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1"
if have_adapters_stack; then
  if have_network_access; then
    if have_docker_daemon; then
      run_to "invarlock evaluate (network, container)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_EVAL_ENV $CLI evaluate --allow-network --baseline \"$SMOKE_MODEL_ID\" --subject \"$SMOKE_MODEL_ID\" --adapter auto --profile dev --preset \"$SMOKE_PRESET\" --device cpu --out \"$TMP_DIR/report_net\" --report-out \"$TMP_DIR/report_net_out\""
      run "invarlock verify (network container output)" "if [ -f \"$TMP_DIR/report_net_out/evaluation.report.json\" ]; then $CLI verify --json \"$TMP_DIR/report_net_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
      run "invarlock report validate (network container output)" "if [ -f \"$TMP_DIR/report_net_out/evaluation.report.json\" ]; then $CLI report validate \"$TMP_DIR/report_net_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
      run_to "invarlock advanced calibrate null-sweep (network, container)" "$CALIBRATE_NULL_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate null-sweep --allow-network --config \"$SMOKE_CALIBRATE_NULL_CONFIG\" --out \"$TMP_DIR/calibrate_null\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
      run_to "invarlock advanced calibrate ve-sweep (network, container)" "$CALIBRATE_VE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate ve-sweep --allow-network --config \"$SMOKE_CALIBRATE_VE_CONFIG\" --out \"$TMP_DIR/calibrate_ve\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
    else
      skip_run "invarlock evaluate (network, container)" "docker daemon not available"
      skip_run "invarlock verify (network container output)" "docker daemon not available"
      skip_run "invarlock report validate (network container output)" "docker daemon not available"
      skip_run "invarlock advanced calibrate null-sweep (network, container)" "docker daemon not available"
      skip_run "invarlock advanced calibrate ve-sweep (network, container)" "docker daemon not available"
    fi
    run_to "invarlock evaluate (network, local)" "$EVALUATE_TIMEOUT_SECONDS" "$NET_EVAL_ENV $CLI evaluate --allow-network --execution-mode trusted-local --baseline \"$SMOKE_MODEL_ID\" --subject \"$SMOKE_MODEL_ID\" --adapter auto --profile dev --preset \"$SMOKE_PRESET\" --device cpu --out \"$TMP_DIR/report_net_local\" --report-out \"$TMP_DIR/report_net_local_out\""
    run "invarlock verify (network local output)" "if [ -f \"$TMP_DIR/report_net_local_out/evaluation.report.json\" ]; then $CLI verify --runtime-provenance trusted-local --json \"$TMP_DIR/report_net_local_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
    run "invarlock report validate (network local output)" "if [ -f \"$TMP_DIR/report_net_local_out/evaluation.report.json\" ]; then $CLI report validate \"$TMP_DIR/report_net_local_out/evaluation.report.json\"; else echo '[skip] report missing'; fi"
    run_to "invarlock advanced calibrate null-sweep (network, host)" "$CALIBRATE_NULL_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate null-sweep --allow-network --allow-host-execution --config \"$SMOKE_CALIBRATE_NULL_CONFIG\" --out \"$TMP_DIR/calibrate_null_host\" --profile ci --device cpu --tier balanced --n-seeds 1 --seed-start 42"
    run_to "invarlock advanced calibrate ve-sweep (network, host)" "$CALIBRATE_VE_TIMEOUT_SECONDS" "TOKENIZERS_PARALLELISM=false $CLI advanced calibrate ve-sweep --allow-network --allow-host-execution --config \"$SMOKE_CALIBRATE_VE_CONFIG\" --out \"$TMP_DIR/calibrate_ve_host\" --profile ci --device cpu --tier balanced --window 6 --n-seeds 1 --seed-start 42"
    if have_docker_daemon; then
      if [[ -f "$TMP_DIR/report_net_out/evaluation.report.json" && -f "$TMP_DIR/report_net_local_out/evaluation.report.json" ]]; then
        run_tiny_eval_parity \
          "invarlock evaluate parity (container vs local)" \
          "$TMP_DIR/report_net_out/evaluation.report.json" \
          "$TMP_DIR/report_net_local_out/evaluation.report.json"
      else
        skip_run "invarlock evaluate parity (container vs local)" "parity inputs missing"
      fi
    else
      skip_run "invarlock evaluate parity (container vs local)" "docker daemon not available"
    fi
  else
    skip_run "invarlock evaluate (network, container)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock verify (network container output)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock report validate (network container output)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock evaluate (network, local)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock verify (network local output)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock report validate (network local output)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock advanced calibrate null-sweep (network, container)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock advanced calibrate ve-sweep (network, container)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock advanced calibrate null-sweep (network, host)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock advanced calibrate ve-sweep (network, host)" "network resolution for huggingface.co unavailable"
    skip_run "invarlock evaluate parity (container vs local)" "network resolution for huggingface.co unavailable"
  fi
else
  skip_run "invarlock evaluate (network, container)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock verify (network container output)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock report validate (network container output)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock evaluate (network, local)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock verify (network local output)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock report validate (network local output)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock advanced calibrate null-sweep (network, container)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock advanced calibrate ve-sweep (network, container)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock advanced calibrate null-sweep (network, host)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock advanced calibrate ve-sweep (network, host)" "adapters stack (torch/transformers) not available"
  skip_run "invarlock evaluate parity (container vs local)" "adapters stack (torch/transformers) not available"
fi

echo "[summary] $(ts) total=${TOTAL_COMMANDS} skipped=${SKIPPED_COMMANDS} unexpected_failures=${UNEXPECTED_FAILURES}" | tee -a "$LOG_FILE"
echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"

if [[ "$UNEXPECTED_FAILURES" -ne 0 ]]; then
  exit 1
fi
