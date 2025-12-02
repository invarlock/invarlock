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
  "$CLI" plugins adapters --json >/dev/null 2>&1 || return 1
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
run "invarlock certify --help"        "$CLI certify --help"
run "invarlock verify --help"         "$CLI verify --help"
run "invarlock run --help"            "$CLI run --help"
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

echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"

# Extended: verify, certify/run with and without network
# Create a tiny invalid certificate to exercise verify paths
TMP_DIR="$(mktemp -d -t invarlock_cli_smoke.XXXXXX.dir)"
echo '{"schema_version": "v1", "primary_metric": {}}' >"$TMP_DIR/cert_invalid.json"

run "invarlock verify (human, invalid)" "$CLI verify \"$TMP_DIR/cert_invalid.json\""
run "invarlock verify --json (invalid)" "$CLI verify --json \"$TMP_DIR/cert_invalid.json\""

# Offline runs (force quick failure if uncached)
OFFLINE_ENV="HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"

if have_adapters_stack; then
  run_to "invarlock run (offline)" 60 "$OFFLINE_ENV $CLI run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci --device cpu --out \"$TMP_DIR/run_offline\""
  run_to "invarlock certify (offline)" 60 "$OFFLINE_ENV $CLI certify --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/tasks/causal_lm/ci_cpu.yaml --out \"$TMP_DIR/cert_offline\" --cert-out \"$TMP_DIR/cert_offline_out\""
else
  {
    echo "\n==== BEGIN invarlock run (offline) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (offline) ====\n"
    echo "\n==== BEGIN invarlock certify (offline) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock certify (offline) ====\n"
  } >>"$LOG_FILE"
fi

# With network allowed (may still fail fast if extras missing)
NET_ENV="INVARLOCK_ALLOW_NETWORK=1 TOKENIZERS_PARALLELISM=false"
if have_adapters_stack; then
  run_to "invarlock run (network)" 60 "$NET_ENV $CLI run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci --device cpu --out \"$TMP_DIR/run_net\""
  run_to "invarlock certify (network)" 60 "$NET_ENV $CLI certify --source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto --profile ci --preset configs/tasks/causal_lm/ci_cpu.yaml --out \"$TMP_DIR/cert_net\" --cert-out \"$TMP_DIR/cert_net_out\""
else
  {
    echo "\n==== BEGIN invarlock run (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock run (network) ====\n"
    echo "\n==== BEGIN invarlock certify (network) ===="
    echo "[skip] adapters stack (torch/transformers) not available"
    echo "==== END invarlock certify (network) ====\n"
  } >>"$LOG_FILE"
fi
