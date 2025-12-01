# Certificate Telemetry Fields

InvarLock certificates now embed latency, memory, and token throughput details under
`/telemetry` for quick edge-readiness checks. Each field is copied directly from
`report.json` when available and is always accompanied by the device recorded in
`meta.device`.

| JSON Pointer | Meaning | Notes |
|--------------|---------|-------|
| `/telemetry/device` | Execution device reported by the run (e.g. `cpu`, `mps`, `cuda`). | Mirrors `meta.device`. |
| `/telemetry/latency_ms_per_tok` | Mean wall-clock latency per generated token. | Reported in milliseconds/token. |
| `/telemetry/memory_mb_peak` | Peak resident memory observed during the run. | Reported in MiB. |
| `/telemetry/preview_total_tokens` | Total tokens processed in the preview split. | Helpful for throughput calculations. |
| `/telemetry/final_total_tokens` | Total tokens processed in the final split. | Matches the tokens counted in the paired bootstrap. |
| `/telemetry/throughput_tok_per_s` | Average throughput when available (tokens/second). | Present when the adapter surfaces throughput statistics. |

Use `invarlock verify` to ensure the certificate remains valid once extracted:

```bash
invarlock verify reports/release/gpt2_small/ci/quant8_cert/evaluation.cert.json
jq '.telemetry' reports/release/gpt2_small/ci/quant8_cert/evaluation.cert.json
```

For CPU edge targets, the helper profile `ci_cpu` and script below capture a
minimal telemetry sweep:

```bash
bash scripts/run_cpu_telemetry.sh
# certs emitted to reports/telemetry/cpu-ci/**/evaluation.cert.json
```

The resulting certificates can be archived alongside the release matrix to
showcase latency/memory behavior across both GPU/MPS and CPU deployments.
