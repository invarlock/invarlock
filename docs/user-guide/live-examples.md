# Live Examples

The curated live-example lane keeps the public front door honest while limiting
CI to stable dependency surfaces.

## Curated CI-safe subset

These paths are exercised by `make docs-live-fast` and by GitHub Actions:

- `README.md`
- `docs/user-guide/getting-started.md`
- `docs/user-guide/quickstart.md`
- `notebooks/invarlock_python_api.ipynb`
- `notebooks/invarlock_policy_tiers.ipynb`

Use this lane when you need deterministic evidence that the documented
`evaluate -> verify -> report html` path still works.

For Markdown examples, this lane replays the documented commands in
host mode and skips heavyweight model-loading steps while reusing
seeded demo evidence for the later `verify` and `report html` steps.
For the curated notebooks, heavyweight evaluation cells are treated the same
way so the contract-reading and verification cells still run against stable demo
reports.

## Full local verifier

Run the broader local lane with:

```bash
make docs-live
```

That verifier replays runnable markdown examples and smoke-runs all notebooks
under `notebooks/`. The local lane runs the real model-loading and evaluation
steps for the runnable surfaces it selects.
For heavier public examples, the verifier rewrites onto smoke-sized shipped
assets such as `sshleifer/tiny-gpt2`, `gpt2_smoke_128`, and the smoke calibration
configs. It also normalizes heavy `ci` / `release` examples onto a local smoke
profile and trims calibration seed counts so the lane stays practical while
still exercising real load/evaluate paths. Expect network fetches, model
downloads, and materially longer runtimes than the curated CI subset.

## Trust boundary

Notebook examples are host examples. They are meant to be easy to run from a
checkout or an installed development environment. Use the runtime-container path
for the default CLI workflow and strict runtime provenance.
