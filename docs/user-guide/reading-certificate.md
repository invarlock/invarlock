# Reading a Certificate (v1)

This guide highlights the key sections of a v1 certificate and how to
interpret them.

- Primary Metric row
  - Shows the task‑appropriate metric (ppl_* or accuracy), its point estimates,
    and paired CI. The ratio/Δpp vs baseline drives the gate.
- System Overhead row (when available)
  - Latency and throughput stats appear separate from quality and reflect the guarded run.
- pPL identity (ppl families)
  - Confirms `exp(mean Δlog)` ≈ `ratio_vs_baseline`; Δlog CI maps to ratio CI
    when reported.
- Provenance
  - Provider/environment/policy digests: `provider_digest`
    (ids/tokenizer/masking), `env_flags`, and `policy_digest` with thresholds
    snapshot.
- Confidence label
  - High/Medium/Low based on CI width and stability; see thresholds and `unstable` flag.

Tip: Use `invarlock verify` to recheck schema, pairing, and ratio math.
