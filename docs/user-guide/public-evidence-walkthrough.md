# Public Evidence Walkthrough

## Purpose

This walkthrough shows the shipped public evidence floor that reviewers can
verify without downloading model weights. It is deliberately BYOE-oriented:
InvarLock validates a baseline/subject comparison artifact; it does not produce
deployable quantized checkpoints.

## Published-basis pass

The repository ships strict-pass public-basis examples for GPT-2-style causal LM
and BERT-style masked LM lanes:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/gpt2/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/bert/evaluation.report.json
```

Each directory includes:

| File | Role |
| --- | --- |
| `evaluation.report.json` | Canonical verifier input with primary metric, guard evidence, policy digest, and assurance section. |
| `runtime.manifest.json` | Container runtime provenance manifest bound to the report by SHA-256. |
| `evidence_pack_recipe.json` | Recipe pointer for rebuilding a full validation evidence pack. |
| `artifact_package/` | Checkpoint references, report/runtime paths, signed-pack path, and exact verifier commands. |
| `evidence_pack/` | Signed, checksum-bound GPT-2 public evidence pack that verifies under strict release policy. |

The support matrix records these paths under
`contracts/support_matrix.json` as the `published_basis` evidence floor.

The GPT-2 `artifact_package/` is intentionally a checkpoint-reference package,
not a weight dump. It names the baseline and subject checkpoint references, binds
them to the report, runtime manifest, and signed pack, and keeps the exact
verification commands in `artifact_package/artifact_package.json`. Large model
weights remain external to the repository; the rebuild recipe is the source of
truth for materializing a fresh BYOE evidence drop.

The GPT-2 lane also ships a small signed pack so reviewers can exercise the
full offline evidence-pack verifier without rebuilding the suite:

```bash
FPR=$(python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("public_evidence/published_basis/gpt2/evidence_pack/manifest.json")
    .read_text(encoding="utf-8")
)
print(manifest["signing_key_fingerprint"])
PY
)

invarlock advanced evidence-pack verify \
  public_evidence/published_basis/gpt2/evidence_pack \
  --strict \
  --profile release \
  --report-assurance strict \
  --expected-fingerprint "$FPR"
```

The expected pack result is `ok=true` with `authenticity=pinned`. Without
`--expected-fingerprint`, the signature still proves integrity but not signer
authenticity.

## Caught regression

The caught-regression fixture keeps the naive primary metric acceptable
(`ratio_vs_baseline = 1.0`) but marks the spectral guard as failed:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/caught_regressions/spectral_guard_failure/evaluation.report.json
```

Expected outcome: verification fails. The failure is not a perplexity failure;
it is a guard/policy failure:

```text
Release verification requires validation.spectral_stable == true
spectral did not pass
```

That is the intended release-gate behavior: a clean summary metric is not enough
when a guard detects unstable weight geometry.

## Applying this to your checkpoint

Use your own edited checkpoint from a quantization, pruning, distillation, or
fine-tuning pipeline, then run `invarlock evaluate` or generate an
`evaluation.report.json` from paired run reports:

```bash
invarlock report generate \
  --run runs/subject/report.json \
  --baseline-run-report runs/baseline/report.json \
  --format report \
  -o reports/eval

invarlock verify --profile release --assurance strict \
  reports/eval/evaluation.report.json
```

Keep `evaluation.report.json` and `runtime.manifest.json` together. Use
`invarlock advanced runtime-verify` only when you specifically want to inspect
the manifest/report binding; use `invarlock verify` for the full release gate.
