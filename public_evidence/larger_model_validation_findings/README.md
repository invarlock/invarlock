# Larger-Model Validation Findings

This directory records summary-only categorized findings from larger-model
CUDA/container smoke, validation, diagnostic, and published-basis verification
lanes. It preserves which lanes reached strict verification, which attempts
failed before verifier artifacts were materialized, which reruns resolved
earlier failed attempts, and which published-basis lanes completed cleanly.

The public files do not include raw logs, host-specific paths, run directories,
or model weights. The findings summary is not a support-matrix change and does
not establish model-quality or assurance results.

Primary files:

- `lane_outcomes.json`: categorized lane-level outcomes and counts across the
  bounded smoke matrix, initial validation matrix, clean resolutions, follow-up
  lanes, and published-basis verification lanes.
- `hash_inventory.json`: SHA-256 and byte inventory for this directory.
- `evidence.meta.json`: public evidence metadata and verifier commands.
