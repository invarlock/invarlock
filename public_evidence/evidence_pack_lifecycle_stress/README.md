# Evidence-Pack Lifecycle Stress Summary

This directory records a summary-only validation run for evidence-pack task
lifecycle and queue-state handling. It does not include raw logs,
host-specific paths, model weights, or run directories.

The validation covers queue lock recovery, atomic task transitions, retry
sanitization, dependency readiness and cancellation, orphan reclamation, worker
restart paths, and structured queue-state helpers on a CUDA-capable validation
host.

Use `lifecycle_summary.json` for the validation outcomes and
`hash_inventory.json` for hashes of the public summary files.
