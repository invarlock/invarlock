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
cd "$SDK_ROOT"
export PYTHONPATH="$SDK_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export INVARLOCK_REQUIRE_EVALUATOR_SDK="$SDK_NAME"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DEEPEVAL_TELEMETRY_OPT_OUT=YES RAGAS_DO_NOT_TRACK=true OPIK_TRACK_DISABLE=true OTEL_SDK_DISABLED=true
SDK_LOCK="$SDK_ROOT/examples/evaluator-qualification/locks/$SDK_NAME.txt"
test -f "$SDK_LOCK"
# The top-level SDK versions are pinned by the maintained evaluator inventory.
# These isolated test environments are not dependencies of the distributed core.
if [ "$SDK_NAME" = "promptfoo" ]; then
  # The maintained npm lock pins the package archive's SHA-512 and SHA-1.
  # Install verified bytes into a disposable producer environment only.
  if [ -z "${INVARLOCK_PROMPTFOO_PACKAGE:-}" ]; then
    SDK_TEMP="$(mktemp -d)"
    trap 'rm -rf "$SDK_TEMP"' EXIT
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
    python -m pytest -q -o addopts='' "$SDK_TEST"
elif [ "${SDK_SERIALIZER_ONLY:-0}" = "1" ]; then
  # Probe native SDK serialization from source only. This is deliberately not
  # a core/SDK co-installation test: producer and recipient may be separate.
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_LOCK" \
    --with pytest==9.1.1 --with jsonschema==4.26.0 \
    python -m pytest -q -o addopts='' "$SDK_TEST"
else
  uv run --no-project --isolated --python 3.12 \
    --with-requirements "$SDK_LOCK" \
    --with-requirements "$SDK_ROOT/requirements/workflows/core-py312.txt" \
    --with pytest==9.1.1 \
    python -m pytest -q -o addopts='' "$SDK_TEST"
fi
