# VerifAI-2 (ICLR 2026) × InvarLock — F4 + S1 Workbench

This folder contains the concrete planning artifacts and runnable scaffolding
for two intended VerifAI-2 workshop submissions:

- **F4 (8pp)**: verifier-first study of weight edits on code LMs where InvarLock
  evidence is used to drive a **selective verification / escalation** policy.
- **S1 (4pp)**: a **verifier-carrying artifact schema** + checker that links
  guard evidence (InvarLock) to external verifier traces (HumanEval/MBPP/Lean/SMT).

Start here:

- Plan: `research/verifai2_2026/PLAN.md`
- Verifier trace contract (spec): `research/verifai2_2026/verifier_trace_contract.md`
- Schemas:
  - `research/verifai2_2026/specs/verifier_trace.v1.schema.json`
  - `research/verifai2_2026/specs/verifier_carrying_artifact.v1.schema.json`
- Pilot helpers:
  - `scripts/verifai2_2026/make_text_canary.py`
  - `scripts/verifai2_2026/pilot_assemble_artifact.py`
  - `scripts/verifai2_2026/schema_verify.py`

