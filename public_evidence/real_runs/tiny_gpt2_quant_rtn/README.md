# tiny GPT-2 quant_rtn Real Run

This directory contains a real container-backed `invarlock evaluate` run using
the GPT-2-family checkpoint `sshleifer/tiny-gpt2` as both baseline and subject.
The subject is edited by the built-in `quant_rtn` RTN dequantized weight-edit
simulation with the attention-only 8-bit overlay.

The run is intentionally small enough to commit, but it is not hand-assembled:
`evaluation.report.json` and `runtime.manifest.json` were emitted by the CLI,
then packaged into a signed evidence pack.

Verify the report:

```bash
uv run invarlock verify \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evaluation.report.json \
  --profile release \
  --assurance strict
```

Verify the signed pack with signer pinning:

```bash
uv run invarlock advanced evidence-pack verify \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evidence_pack \
  --strict \
  --profile release \
  --report-assurance strict \
  --expected-fingerprint sha256:cc17b2af6579f5de01e74d91e93528b04670ff89f907ec3ba786a69065435605
```

Non-goals: this does not vendor model weights, and `quant_rtn` is a deterministic
dequantized weight perturbation rather than a deployable quantization backend.
