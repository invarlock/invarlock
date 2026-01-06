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
- Measurement contract
  - `resolved_policy.spectral.measurement_contract` /
    `resolved_policy.rmt.measurement_contract` pin the estimator + sampling
    procedure used by guards.
  - `spectral.measurement_contract_hash` / `rmt.measurement_contract_hash` are
    compact digests for audit and baseline pairing.
  - In CI/Release, `invarlock verify` enforces baseline/subject pairing (`*_measurement_contract_match = true`).
- Confidence label
  - High/Medium/Low based on CI width and stability; see thresholds and `unstable` flag.

Tip: Use `invarlock verify` to recheck schema, pairing, and ratio math.
