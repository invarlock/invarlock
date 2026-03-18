#!/bin/bash -eu

cd "$SRC/invarlock"
python3 -m pip install .

for fuzzer in $(find "$SRC/invarlock/fuzzers" -name '*_fuzzer.py' | sort); do
  fuzzer_basename=$(basename -s .py "$fuzzer")
  fuzzer_package="${fuzzer_basename}.pkg"

  pyinstaller --distpath "$OUT" --onefile --name "$fuzzer_package" "$fuzzer"

  cat > "$OUT/$fuzzer_basename" <<EOF
#!/bin/sh
# LLVMFuzzerTestOneInput for fuzzer detection.
this_dir=\$(dirname "\$0")
"\$this_dir/$fuzzer_package" "\$@"
EOF
  chmod +x "$OUT/$fuzzer_basename"

  corpus_dir="$SRC/invarlock/fuzzers/corpora/$fuzzer_basename"
  if [ -d "$corpus_dir" ]; then
    (cd "$corpus_dir" && zip -qr "$OUT/${fuzzer_basename}_seed_corpus.zip" .)
  fi
done
