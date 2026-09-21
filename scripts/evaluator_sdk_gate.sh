#!/usr/bin/env bash
# SDK capture probes are separate from SDK-free installed recipient journeys.
set -euo pipefail
SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_NAME="${1:-}"
SDK_SERIALIZER_ONLY=0
case "$SDK_NAME" in
  deepeval|ragas|lighteval|hugging-face-evaluate|autoevals|openevals|arize-phoenix-evals|opik)
    SDK_TEST="tests/evaluation_records/test_scalar_integrations.py::test_pinned_sdk_objects[$SDK_NAME]"
    SDK_SERIALIZER_ONLY=1 ;;
  pydantic-evals|azure-ai-evaluation|evidently|mlflow|garak|openai-evals|trulens)
    SDK_TEST="tests/evaluation_records/test_batch_integrations.py::test_installed_sdk_native_objects_offline[$SDK_NAME]"
    SDK_SERIALIZER_ONLY=1 ;;
  inspect-ai)
    SDK_TEST="tests/evaluation_records/test_sdk_capture.py::test_actual_inspect_log_uses_core_entrypoint" ;;
  langfuse)
    SDK_TEST="tests/evaluation_records/test_sdk_capture.py"
    export INVARLOCK_REQUIRE_LANGFUSE_SDK=1 ;;
  lm-evaluation-harness|promptfoo)
    SDK_TEST="tests/evaluation_records/test_original_sdk_capture.py::test_pinned_original_sdk_serialization[$SDK_NAME]" ;;
  *) echo "Choose a supported evaluator from the integration guide." >&2; exit 2 ;;
esac
SDK_TESTS=("$SDK_TEST" "tests/evaluation_records/test_sdk_installed_handoff.py")
SDK_TESTS+=("tests/evaluation_records/test_live_capture_cli.py::test_actual_capture_cli_sdk_guard_and_installed_recipient[$SDK_NAME]")
case "$SDK_NAME" in
  lm-evaluation-harness|inspect-ai|promptfoo|langfuse)
    SDK_TESTS+=("tests/evaluation_records/test_live_capture_cli.py::test_actual_http_capture_cli_sdk_guard_and_installed_recipient[$SDK_NAME]") ;;
esac
case "$SDK_NAME" in
  lm-evaluation-harness|inspect-ai|promptfoo|lighteval|garak|openai-evals|langfuse)
    for SDK_LIVE_TEST in \
      test_actual_framework_drives_callback_and_writes_native_export \
      test_actual_framework_prompt_mutation_is_refused_before_task \
      test_actual_framework_task_failure_is_retained; do
      SDK_TESTS+=("tests/evaluation_records/test_live_harness_capture.py::${SDK_LIVE_TEST}[$SDK_NAME]")
    done ;;
  deepeval|ragas|hugging-face-evaluate|autoevals|openevals|arize-phoenix-evals|opik)
    SDK_TESTS+=("tests/evaluation_records/test_live_scalar_capture.py::test_live_scalar_fresh_task_and_real_sdk_metric[$SDK_NAME]") ;;
  pydantic-evals|azure-ai-evaluation|evidently|mlflow|trulens)
    SDK_TESTS+=("tests/evaluation_records/test_live_batch_capture.py::test_real_sdk_execution_preserves_complete_synthetic_schedule[$SDK_NAME]") ;;
esac
case "$SDK_NAME" in
  deepeval|ragas|lighteval|hugging-face-evaluate|autoevals|openevals|arize-phoenix-evals|opik)
    SDK_TESTS+=("tests/evaluation_records/test_sdk_scalar_roundtrip.py") ;;
esac
SDK_RECIPE_DEPS=(--with pytest==9.1.1 --with jsonschema==4.26.0)
SDK_TEMP="$(mktemp -d)"
trap 'rm -rf "$SDK_TEMP"' EXIT
if [ "$SDK_NAME" = "deepeval" ]; then
  SDK_TESTS+=("tests/evaluation_records/test_scalar_integrations.py::test_documented_deepeval_python_capture")
  # Literal recipes import captured-file verification and request parsing.
  # Pin those dependencies only; this remains a source-level recipe check.
  SDK_RECIPE_DEPS+=(--with cryptography==50.0.0 --with pyyaml==6.0.3)
elif [ "$SDK_NAME" = "langfuse" ]; then
  SDK_TESTS+=("tests/examples/test_langfuse_export.py::test_documented_langfuse_native_recipe")
fi
cd "$SDK_ROOT"
export PYTHONPATH="$SDK_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export INVARLOCK_REQUIRE_EVALUATOR_SDK="$SDK_NAME"
export INVARLOCK_LIVE_EVALUATOR="$SDK_NAME"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DEEPEVAL_TELEMETRY_OPT_OUT=YES RAGAS_DO_NOT_TRACK=true OPIK_TRACK_DISABLE=true OTEL_SDK_DISABLED=true
# SDK serialization and recipient verification run in different environments.
# A caller may supply an already installed candidate; otherwise install the
# current built wheel into a fresh hash-locked core-only environment.
if [ -z "${INVARLOCK_EVALUATOR_PARITY_PYTHON:-}" ]; then
  shopt -s nullglob
  SDK_WHEELS=("$SDK_ROOT"/dist/invarlock-*.whl)
  if [ "${#SDK_WHEELS[@]}" -ne 1 ]; then
    echo "Build exactly one candidate wheel with make dist-check first." >&2
    exit 2
  fi
  uv venv --python 3.12 "$SDK_TEMP/recipient"
  uv pip install --python "$SDK_TEMP/recipient/bin/python" --require-hashes \
    -r "$SDK_ROOT/requirements/workflows/release-install-py312.txt"
  uv pip install --python "$SDK_TEMP/recipient/bin/python" --no-deps "${SDK_WHEELS[0]}"
  uv pip check --python "$SDK_TEMP/recipient/bin/python"
  export INVARLOCK_EVALUATOR_PARITY_PYTHON="$SDK_TEMP/recipient/bin/python"
fi
SDK_LOCK="$SDK_ROOT/examples/evaluator-qualification/locks/$SDK_NAME.txt"
if [ "$SDK_NAME" = "lighteval" ]; then
  # Preserve the historical qualification lock. The live pipeline needs the
  # compatible hash-library constraint, plus independently checked NLP assets.
  SDK_LOCK="$SDK_ROOT/examples/integrations/evaluator-live/locks/lighteval.txt"
  export INVARLOCK_NLTK_ARCHIVE_DIR="$SDK_TEMP/nltk-archives"
  export INVARLOCK_LIGHTEVAL_REGISTRY_ASSET="$SDK_TEMP/tinyBenchmarks.pkl"
  python3 - "$SDK_ROOT/examples/integrations/evaluator-live/harness.py" "$INVARLOCK_NLTK_ARCHIVE_DIR" <<'PY'
import hashlib
import pathlib
import runpy
import sys
import urllib.request

driver = runpy.run_path(sys.argv[1])
pins = driver["NLTK_ARCHIVES"]
destination = pathlib.Path(sys.argv[2]) / "tokenizers"
destination.mkdir(parents=True)
for name, expected in pins.items():
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{name}.zip"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(f"NLTK {name} archive differs from its independent pin")
    (destination / f"{name}.zip").write_bytes(raw)
resource = driver["LIGHTEVAL_REGISTRY_RESOURCE"]
with urllib.request.urlopen(resource["url"], timeout=60) as response:
    raw = response.read(resource["size"] + 1)
if len(raw) != resource["size"] or hashlib.sha256(raw).hexdigest() != resource["sha256"]:
    raise SystemExit("LightEval registry resource differs from its independent pin")
(destination.parent.parent / "tinyBenchmarks.pkl").write_bytes(raw)
PY
fi
test -f "$SDK_LOCK"
# The top-level SDK versions are pinned by the maintained evaluator inventory.
# These isolated test environments are not dependencies of the distributed core.
if [ "$SDK_NAME" = "promptfoo" ]; then
  # The maintained npm lock pins the package archive's SHA-512 and SHA-1.
  # Install verified bytes into a disposable evaluator environment only.
  if [ -z "${INVARLOCK_PROMPTFOO_PACKAGE:-}" ]; then
    SDK_PACKAGE="$(node -e 'const fs=require("fs"); const line=fs.readFileSync(process.argv[1],"utf8").split("\n").find(x=>x.startsWith("package=")); if(!line)throw Error("missing package pin"); process.stdout.write(line.slice(8));' "$SDK_LOCK")"
    npm pack "$SDK_PACKAGE" --json --pack-destination "$SDK_TEMP" > "$SDK_TEMP/archive.json"
    SDK_ARCHIVE="$(node -e '
      const fs=require("fs"), path=require("path"), crypto=require("crypto");
      const pins=Object.fromEntries(fs.readFileSync(process.argv[1],"utf8").trim().split("\n").map(line=>{const i=line.indexOf("=");return [line.slice(0,i),line.slice(i+1)];}));
      const root=process.argv[2], packed=JSON.parse(fs.readFileSync(path.join(root,"archive.json"),"utf8"));
      if(packed.length!==1 || path.basename(packed[0].filename)!==packed[0].filename)throw Error("invalid npm archive result");
      const file=path.join(root,packed[0].filename), bytes=fs.readFileSync(file);
      if("sha512-"+crypto.createHash("sha512").update(bytes).digest("base64")!==pins.integrity || crypto.createHash("sha1").update(bytes).digest("hex")!==pins.shasum)throw Error("Promptfoo archive differs from maintained integrity pins");
      process.stdout.write(file);
    ' "$SDK_LOCK" "$SDK_TEMP")"
    npm install --prefix "$SDK_TEMP" --ignore-scripts --no-audit --no-fund "$SDK_ARCHIVE"
    export INVARLOCK_PROMPTFOO_PACKAGE="$SDK_TEMP/node_modules/promptfoo"
  fi
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_ROOT/requirements/workflows/core-py312.txt" \
    --with pytest==9.1.1 \
    python -m pytest -q -o addopts='' --basetemp "$SDK_TEMP/pytest" "${SDK_TESTS[@]}"
elif [ "$SDK_NAME" = "lighteval" ]; then
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_LOCK" \
    "${SDK_RECIPE_DEPS[@]}" \
    python -c 'import os,runpy,sys; driver=runpy.run_path(sys.argv[1]); driver["lighteval_resource"](os.environ["INVARLOCK_LIGHTEVAL_REGISTRY_ASSET"]); import pytest; raise SystemExit(pytest.main(sys.argv[2:]))' \
    "$SDK_ROOT/examples/integrations/evaluator-live/harness.py" \
    -q -o addopts='' --basetemp "$SDK_TEMP/pytest" "${SDK_TESTS[@]}"
elif [ "${SDK_SERIALIZER_ONLY:-0}" = "1" ]; then
  # Probe native SDK serialization from source only. This is deliberately not
  # a core/SDK co-installation test: capture process and recipient may be separate.
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_LOCK" \
    "${SDK_RECIPE_DEPS[@]}" \
    python -m pytest -q -o addopts='' --basetemp "$SDK_TEMP/pytest" "${SDK_TESTS[@]}"
else
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_LOCK" \
    --with-requirements "$SDK_ROOT/requirements/workflows/core-py312.txt" \
    --with pytest==9.1.1 \
    python -m pytest -q -o addopts='' --basetemp "$SDK_TEMP/pytest" "${SDK_TESTS[@]}"
fi
