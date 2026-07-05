# tiny GPT-2 External Magnitude-Prune BYOE Real Run

This directory contains a real container-backed `invarlock evaluate` run using
`sshleifer/tiny-gpt2` as the baseline and a locally materialized subject
checkpoint created by `external_edit_recipe.py`.

The subject checkpoint is not produced by an InvarLock edit plugin. The recipe
loads the baseline checkpoint, applies deterministic magnitude pruning outside
the verifier, saves the subject under `<local-temp-dir>`, and then `invarlock
evaluate` consumes that subject with `--edit-label custom`.

Verify the report:

```bash
uv run invarlock verify \
  public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evidence_pack/reports/report-001/evaluation.report.json \
  --profile release \
  --assurance strict
```

Verify the signed pack with signer pinning:

```bash
uv run invarlock advanced evidence-pack verify \
  public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evidence_pack \
  --strict \
  --profile release \
  --report-assurance strict \
  --expected-fingerprint sha256:e01c40a94c89b22306a2670b032f623aa5428351d06e18f9b3e9e6a39b42c41b
```

Scope: this run records evidence for the real BYOE path, where a pre-edited
external subject checkpoint is verified against a baseline. Model-weight
vendoring, sparse runtime speedups, and production compression-backend behavior
are outside this artifact.
