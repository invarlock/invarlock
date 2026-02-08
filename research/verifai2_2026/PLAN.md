# Plan (Steps 1–4): F4 + S1

Scope: only Steps **1–4** (deliverables+scope, verifier-trace contract, pilot,
canary creation/release plan). Scaling experiments and paper writing happen
after this plan is executed.

## Step 1 — Deliverables + Frozen Scope (F4 + S1)

### Deliverables (repo artifacts)

1. **Verifier-trace contract** (shared for F4 + S1)
   - Spec: `research/verifai2_2026/verifier_trace_contract.md`
   - JSON Schema: `research/verifai2_2026/specs/verifier_trace.v1.schema.json`

2. **Verifier-carrying artifact schema** (S1)
   - JSON Schema: `research/verifai2_2026/specs/verifier_carrying_artifact.v1.schema.json`
   - `schema-verify` checker: `scripts/verifai2_2026/schema_verify.py`

3. **Pilot scaffolding** (F4 + S1)
   - Canary builder + manifest: `scripts/verifai2_2026/make_text_canary.py`
   - Artifact assembler (InvarLock report + verifier trace -> unified artifact):
     `scripts/verifai2_2026/pilot_assemble_artifact.py`
   - Example inputs:
     - `research/verifai2_2026/examples/verifier_trace_humaneval.example.json`
     - `research/verifai2_2026/examples/verifier_trace_mbpp.example.json`

### Frozen methodological commitments (paper-facing)

- External verifier outcome is the *headline metric* (HumanEval/MBPP pass rates).
- InvarLock evidence is an auxiliary signal used for **selective verification**:
  reduce verifier calls at a strict FNR bound on verifier FAIL (target proposed
  for the paper: `< 1%`).
- Canary is domain-aligned (code/proof text) via `local_jsonl`; generic text is
  ablation-only.
- Baseline evaluation reports are reused across edit variants via
  `--baseline-report` (requires stored windows).
- Verifier determinism is recorded via the verifier-trace contract (prompt set
  hashing, harness version, sandbox limits, decoding params).

### Draft scope targets (not executed yet in Steps 1–4)

These are placeholders to guide later scaling work; final model IDs/revisions
must be validated in a pilot before committing them in the paper.

- **Models (target for F4)**: 8 total across 0.35B–30B+ parameters.
- **Edits (target for F4)**: ~20–30 variants per model (INT8 sweeps + pruning
  severities; 4-bit GPTQ/AWQ/BNB as held-out families).

## Step 2 — Lock Down the Verifier-Trace Contract (review rounds)

We treat this like a verification interface: it must be minimal, stable, and
auditable.

### Round 0 (minimum viable)

Required:

- Prompt-set identity (dataset name/split/revision) + hash of the prompt list.
- Model identity (HF id + revision hash) + tokenizer identity.
- Decoding parameters (greedy vs sampling, temperature/top-p, max tokens).
- Harness identity (name + version/commit) + sandbox resource limits.
- Results: per-problem verdicts + aggregated pass rates.

Problems discovered:

- “Prompt list hash” must specify canonicalization rules (order, encoding).
- Need explicit support for “hash-only” prompts for release-safe sharing.
- Need a place to record *counterexample slices* (failing test names, error
  excerpts) without embedding full logs in JSON.

### Round 1 (make it reviewer-proof)

Add:

- Canonical hashing rules (UTF-8, normalized JSON, sorted lists).
- Artifact addressing by hash for large logs (optional external blobs).
- Execution environment metadata (OS/container image digest) for reproducibility.
- Clear trust boundary: the trace contract records determinism inputs; it does
  not “prove” determinism across platforms.

### Round 2 (final v1)

Finalize:

- JSON Schema v1 for trace records + wrapper artifact.
- `schema-verify` tool that checks:
  - schema conformance
  - required trace-contract fields present
  - hash fields are well-formed
  - optional file-hash verification when provided local paths

Artifacts produced in this step:

- Spec + schemas under `research/verifai2_2026/`
- Validation tool under `scripts/verifai2_2026/`

## Step 3 — Minimal End-to-End Pilot (before scaling)

Goal: prove that the pipeline can produce paired artifacts and mismatches, with
no hand-wavy “we will log X later”.

### Pilot inputs (recommendation)

- Model: a small code-capable HF checkpoint that loads with `hf_causal`.
- Edit: built-in INT8 `quant_rtn` (1–2 variants) + one pruning mask variant.
- Canary: a tiny `local_jsonl` file produced by `make_text_canary.py`.
- Verifier: a placeholder verifier trace record (until HumanEval harness is
  wired) to validate schema and assembly.

Notes for small/offline pilots:

- Prefer `--profile dev --tier none` for the pilot. `--profile ci` applies
  default dataset window counts (200/200) and will fail fast on small canaries.
- If you copy a Hugging Face cache snapshot for offline use, make sure the
  copied model directory contains *real files*, not symlinks into the HF cache.
  (Use `cp -aL` when copying snapshot contents.)

### Pilot outputs (pass criteria)

- An InvarLock `evaluation.report.json` and a passing `invarlock verify`.
- A verifier trace JSON that validates against `verifier_trace.v1`.
- A unified verifier-carrying artifact that validates via `schema_verify.py`.

## Step 4 — Canary Creation + Release-Safe Plan

### Canary creation requirements

- `local_jsonl` with `{"text": ...}` lines (metadata allowed but optional).
- Deterministic selection (seeded) and a **manifest** capturing:
  - source (dataset or local corpus)
  - selection rules + seed
  - content hash of the resulting JSONL
  - per-item hashes (optional, for audits)

### Release-safe sharing

Default posture: do **not** redistribute raw canary text unless licensing is
unambiguous. Instead, release:

- manifest + selection script + pinned dataset revision (or pinned local corpus
  commit in a public repo) so others can reconstruct.
- If redistribution is allowed, also release the JSONL file directly.

### Artifacts produced in this step

- `scripts/verifai2_2026/make_text_canary.py` (builder)
- `research/verifai2_2026/examples/` (tiny example canary + manifest)

## Validation Commands (for Steps 1–4)

```bash
# Validate schemas + examples
python scripts/verifai2_2026/schema_verify.py --help
python scripts/verifai2_2026/schema_verify.py research/verifai2_2026/examples/artifact.example.json

# Build a tiny local canary from a directory of code/text files
python scripts/verifai2_2026/make_text_canary.py \
  --input-dir /path/to/corpus \
  --glob '**/*.py' \
  --n 512 \
  --out /tmp/code_canary.jsonl \
  --manifest-out /tmp/code_canary.manifest.json

# Minimal offline pilot (end-to-end artifact): local_jsonl + local HF directory
HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src python -m invarlock evaluate \
  --baseline /path/to/local_hf_model_dir \
  --subject /path/to/local_hf_model_dir \
  --adapter hf_causal \
  --profile dev --tier none \
  --preset /tmp/pilot_local_jsonl.yaml \
  --edit-config configs/overlays/edits/quant_rtn/tiny_demo.yaml \
  --out /tmp/pilot_runs --report-out /tmp/pilot_reports

PYTHONPATH=src python -m invarlock verify --profile dev --json \
  /tmp/pilot_reports/evaluation.report.json > /tmp/pilot_reports/verify.json

python scripts/verifai2_2026/pilot_assemble_artifact.py \
  --evaluation-report /tmp/pilot_reports/evaluation.report.json \
  --verifier-trace /tmp/pilot_dummy_trace.json \
  --verify-json /tmp/pilot_reports/verify.json \
  --embed-evaluation-report \
  --out /tmp/pilot_artifact.json

python scripts/verifai2_2026/schema_verify.py --check-files /tmp/pilot_artifact.json
```
