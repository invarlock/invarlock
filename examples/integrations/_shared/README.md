# Shared Integration Assets

These files define the common shape for target-specific integration examples.
They are intentionally independent of optional third-party backends.

## Files

| File | Role |
| --- | --- |
| `evidence-scope.md` | Wording and mode boundaries for public integration examples. |
| `expected-artifacts.md` | Artifact checklist for runnable examples. |
| `run_invarlock_compare.sh` | Shared compare/verify/render wrapper for HF-loadable baseline and subject paths. |

## Preflight Checklist

Run these from the repository root or from an environment where `invarlock` is
installed:

```bash
invarlock doctor
invarlock advanced plugins list --json
```

For optional target backends, verify the Python import before promising a
runnable example:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('gptqmodel') is not None)"
```

Use the relevant module name for each target, such as `torchao`, `peft`,
`optimum`, `llmcompressor`, `lm_eval`, `vllm`, or `bitsandbytes`.

## Shared Compare Wrapper

The shared script expects an already loadable baseline and subject:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-smoke \
  --allow-network
```

The default mode is strict/container-backed. For a host-side exploratory run,
pass both flags explicitly:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-host-smoke \
  --execution-mode host \
  --assurance off \
  --allow-network
```
