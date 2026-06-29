# Larger-Model Queue Drain Findings

This directory records summary-only findings from post-cutoff model-catalog
CUDA/container queue-drain runs. It preserves which lanes reached strict
verification, which attempts failed before verifier artifacts were materialized,
and which clean reruns resolved earlier failed attempts.

The public files do not include raw logs, host-specific paths, run directories,
or model weights. The findings summary is not a support-matrix change and does
not establish model-quality or assurance results.

Primary files:

- `findings_summary.json`: lane-level outcome summary and counts.
- `late_clean_addendum.json`: late clean outcomes not included in the initial
  findings cutoff.
- `hash_inventory.json`: SHA-256 and byte inventory for this directory.
- `evidence.meta.json`: public evidence metadata and verifier commands.
