# Verifier-Trace Contract (v1)

Purpose: define a minimal, machine-checkable contract for recording *external
verifier* outcomes (tests, proof checking, SMT) with enough determinism and
provenance metadata to make results auditable and reproducible.

This contract is shared across:

- **F4** (verifier-first analysis; HumanEval/MBPP are ground truth)
- **S1** (schema that carries verifier traces as first-class blocks)

Non-goals:

- This does **not** prove determinism across platforms.
- This does **not** require embedding large logs; it supports hash-addressed
  external blobs.

## Design Principles

1. **Verifier-first:** the verifier outcome is the headline metric; the contract
   records the conditions under which that outcome was produced.
2. **Stable hashing:** prompt sets and key configs must be hashable with clear
   canonicalization rules.
3. **Release-safe:** prompt content may be recorded as hashes + dataset
   references rather than raw text.
4. **Portable:** records should be valid even when some heavy artifacts are
   stored externally (logs, generated code).

## Canonicalization & Hashing Rules

Unless explicitly noted otherwise:

- Strings are UTF-8.
- Hash algorithm is SHA-256 (hex lowercase).
- JSON objects are canonicalized for hashing by:
  - sorting keys
  - removing insignificant whitespace (no spaces after separators)
  - encoding as UTF-8 bytes

Normative reference implementation for canonical JSON bytes:

```py
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
```

### Prompt-Set Digest

Prompt-set digest is computed over a canonical JSON object (note: **do not**
include embedded prompt text in this digest; it must be stable across
`mode=hash_only` vs `mode=embedded`):

```json
{
  "dataset": {"name": "...", "config": "...", "split": "...", "revision": "..."},
  "items": [{"id": "...", "sha256": "..."}, "..."]
}
```

The `items` array order is significant and must be the order used by the harness.

### Item Hash Meaning

`items[*].sha256` is the SHA-256 of the **exact prompt string fed to the model**
(UTF-8 bytes), after any harness templating/normalization. When `mode=embedded`,
`items[*].text` must be present and `sha256(text)` must match `items[*].sha256`.

## Required Fields (v1)

### 1) Verifier identity

- `verifier.name`: e.g. `"humaneval"`, `"mbpp"`, `"lean4"`, `"z3"`
- `verifier.kind`: `"code_execution" | "proof_checker" | "smt_solver" | "static_analyzer"`
- `verifier.harness`: name + (**at least one of**) version, git commit, container
  image digest. `config_digest_sha256` should be recorded when a structured
  harness config exists.

### 2) Prompt set

Two supported modes:

- `mode="embedded"`: prompt text is included (only use if redistributable).
- `mode="hash_only"`: include per-item hashes + dataset identifiers so others
  can reconstruct.

Required:

- dataset identifiers: name/split/revision (plus config when relevant). For local
  prompt sets, record `dataset.name="local"` and set `dataset.revision` to a
  content-addressed identifier (e.g., the canary JSONL sha256); also record
  `dataset.manifest_sha256` when available.
- per-item id + sha256 of prompt text (or sha256 of a canonical prompt record)

**Contract invariant:** `results.cases[*].id` must exactly match the prompt-set
`items[*].id` list (same ids, same order). This makes traces stable and
machine-checkable.

### 3) Model + tokenizer identity

Required:

- `model.id` and `model.revision` (commit hash or immutable tag)
- `tokenizer.id` and `tokenizer.revision` (if separate)

Optional but recommended:

- `model.files[]` with per-file sha256 when practical (large models may omit)

### 4) Decoding parameters

Required:

- decoding method: `greedy|sample|beam`
- `temperature`, `top_p`, `top_k`, `max_new_tokens`, `stop` (if any)
- `seed` if sampling is used; for greedy runs record the seed anyway

### 5) Sandbox / execution environment

Required for code execution:

- `sandbox.network_enabled` (must be false for execution verifiers)
- timeouts and resource limits (cpu/mem/wall)

Recommended:

- container image digest, OS, python version

### 6) Results

Required:

- per-item verdicts (pass/fail/error) with stable ids
- aggregate summary (pass-rate + total, with an explicit metric name like
  `"pass@1"` or `"pass@10"`)

Optional:

- counterexample slices (failing test names, stderr excerpt hash) and/or
  hash-addressed external logs.

## Trust Boundary Notes (for paper text)

- The contract + `schema-verify` can validate that a trace record is *well-formed*
  and that configuration metadata is present.
- It cannot validate that the harness executed correctly without rerunning it.
- The point is auditability and reproducibility, not formal correctness.
