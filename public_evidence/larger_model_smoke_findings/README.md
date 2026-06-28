# Larger-Model Smoke Findings

This directory records summary-only findings from bounded model-catalog
CUDA/container smoke runs. It is intended to preserve which lanes reached strict
verification and which lane failed before report or verifier artifacts were
materialized.

The public files do not include raw logs, host-specific paths, run directories,
or model weights. The findings summary is not a support-matrix change and does
not establish model-quality or assurance results.

Primary files:

- `findings_summary.json`: lane-level outcome summary and counts.
- `hash_inventory.json`: SHA-256 and byte inventory for this directory.
- `evidence.meta.json`: public evidence metadata and verifier commands.
