# Attention Backend Compatibility

This directory records summary-only compatibility behavior for evidence-pack
attention backend handling. It does not include raw logs, host-specific paths,
model weights, or run directories.

The evidence records a CUDA-capable validation host where CUDA was available
but Flash Attention 2 was unavailable, plus focused shell coverage showing that
the evidence-pack dependency and config paths select eager attention when
optimized attention is unavailable or install/import is not clean.

Use `compatibility_summary.json` for the observed outcomes and
`hash_inventory.json` for hashes of the public summary files.
