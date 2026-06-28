# FA2 Fallback Compatibility Summary

This directory records summary-only compatibility behavior for evidence-pack
Flash Attention 2 handling. It does not include raw logs, host-specific paths,
model weights, or run directories.

The evidence records a CUDA-capable validation host where CUDA was available
but Flash Attention 2 was unavailable, plus focused shell coverage showing that
the evidence-pack dependency and config paths fall back to eager attention.

Use `compatibility_summary.json` for the observed outcomes and
`hash_inventory.json` for hashes of the public summary files.
