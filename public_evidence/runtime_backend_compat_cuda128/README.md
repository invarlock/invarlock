# CUDA 12.8 Runtime Backend Compatibility Sweep

This directory records a summary-only compatibility sweep for the example CUDA
runtime images used by optional quantization and compression integrations. It
does not include model weights, raw logs, or host-specific run directories.

The sweep checks that each example image builds from its pinned CUDA 12.8 lock
file and that the matching runtime smoke imports the expected backend adapter
surface on a CUDA-capable validation host.

Use `compatibility_summary.json` for the backend-family outcomes and
`hash_inventory.json` for hashes of the public summary files.
