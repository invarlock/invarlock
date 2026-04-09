#!/bin/bash -eu

cd "$SRC/invarlock"

python3 -m pip install --require-hashes -r requirements/workflows/clusterfuzzlite-py311.txt

# ClusterFuzzLite's Python base image currently builds with Python 3.11.
# Build a local wheel, then install it via a hash-pinned direct-url
# requirements file so the fuzz environment stays reproducible.
python3 -m build --wheel --no-isolation

wheel_path="$(find dist -maxdepth 1 -name 'invarlock-*.whl' | sort | head -n 1)"
if [ -z "$wheel_path" ]; then
  echo "ClusterFuzzLite build failed: local wheel not found" >&2
  exit 1
fi

wheel_requirements="$(mktemp)"
python3 - "$wheel_path" "$wheel_requirements" <<'PY'
import hashlib
from pathlib import Path
import sys

wheel_path = Path(sys.argv[1]).resolve()
requirements_path = Path(sys.argv[2])
wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
requirements_path.write_text(
    f"invarlock @ {wheel_path.as_uri()} --hash=sha256:{wheel_hash}\n",
    encoding="utf-8",
)
PY
python3 -m pip install --ignore-requires-python --no-deps --require-hashes -r "$wheel_requirements"

for fuzzer in $(find "$SRC/invarlock/fuzzers" -name '*_fuzzer.py' | sort); do
  fuzzer_basename=$(basename -s .py "$fuzzer")
  fuzzer_package="${fuzzer_basename}.pkg"

  pyinstaller \
    --distpath "$OUT" \
    --onefile \
    --collect-data invarlock \
    --add-data "$SRC/invarlock/contracts:contracts" \
    --name "$fuzzer_package" \
    "$fuzzer"

  cat > "$OUT/$fuzzer_basename" <<EOF
#!/bin/sh
# LLVMFuzzerTestOneInput for fuzzer detection.
this_dir=\$(CDPATH= cd -- "\$(dirname "\$0")" && pwd)
workspace_root=\${GITHUB_WORKSPACE:-\$(CDPATH= cd -- "\$this_dir/.." && pwd)}
if [ -z "\${INVARLOCK_CONTRACTS_ROOT:-}" ] && [ -d "\$workspace_root/contracts" ]; then
  export INVARLOCK_CONTRACTS_ROOT="\$workspace_root/contracts"
fi
"\$this_dir/$fuzzer_package" "\$@"
EOF
  chmod +x "$OUT/$fuzzer_basename"

  corpus_dir="$SRC/invarlock/fuzzers/corpora/$fuzzer_basename"
  if [ -d "$corpus_dir" ]; then
    (cd "$corpus_dir" && zip -qr "$OUT/${fuzzer_basename}_seed_corpus.zip" .)
  fi
done
