# llama.cpp Qwen3.5 GGUF quantization comparison

This one-command example downloads the official revision-pinned
`ggml-org/Qwen3.5-0.8B-GGUF` Q8 artifact and verifies its published byte length and
SHA-256 digest. It builds the maintained, source-pinned llama.cpp runtime,
converts that authenticated Q8 source to Q5_K_M offline with the pinned
quantizer, and compares the two artifacts through:

```text
invarlock evaluate -> invarlock verify -> invarlock report
```

Run it from the repository root:

```bash
make example-gguf-llama-cpp \
  EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/gguf-llama-cpp"
```

Docker or Podman, network access for the pinned download and image build, and
roughly 3 GB of temporary disk space are required. Model execution and Q5
derivation are offline. The quantizer comes from the same pinned llama.cpp
source as the runtime, and the evidence authenticates a transformation summary
containing the source and derived artifact identities. Both model workers run
as non-root users in network-disabled, read-only containers. The resulting
workspace contains the distinct GGUF files, authenticated backend resources,
signed evidence, a separately signed strict-verification receipt, and an HTML
report.

The 50-record schedule uses losslessly decoded, single-token Qwen3.5 targets. Its
illustrative policy requires at least 50 records, a paired interval no wider
than 20 percentage points, and a lower confidence bound above -15 percentage
points. The command also requires both models to solve at least 40% of those
records before reporting success. A passing result supports only that bounded
comparison; it is not a general model-quality or quantization-quality result.
