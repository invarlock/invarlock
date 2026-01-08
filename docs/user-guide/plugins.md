# Plugin Workflow: Adapters and Guards

InvarLock’s plugin system extends model loading and guard capabilities. Plugins do
not add additional edit algorithms beyond the built‑in RTN quantization.

- Adapters: add new model readers or inference backends (e.g., GPTQ/AWQ
  formats) so you can certify pre‑edited checkpoints via the Compare & Certify
  (BYOE) flow.
- Guards: add custom validation checks while keeping the certificate surface stable.

## List Installed Plugins

```bash
invarlock plugins adapters
invarlock plugins guards
```

## Use a Plugin Adapter (Compare & Certify)

Provide baseline and subject checkpoints in a supported format; the adapter
handles loading:

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline /path/to/baseline \
  --subject  /path/to/subject \
  --adapter hf_gptq \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
```

## Author a Guard Plugin

1. Implement the `Guard` protocol:

```python
from invarlock.core.api import Guard

class MyGuard(Guard):
    name = "my_guard"

    def prepare(self, model, adapter, calib, policy):
        return {"ready": True}

    def validate(self, model, adapter, context):
        return {
            "passed": True,
            "action": "continue",
            "message": "No issues detected",
            "metrics": {},
        }
```

`context` receives baseline metrics and guard policy data collected earlier in
the run. Return dictionaries should include at least `passed`, `action`, and
`message` keys so the runner can decide whether to continue, warn, or abort.

1. Register the guard in your `pyproject.toml`:

```toml
[project.entry-points."invarlock.guards"]
my_guard = "my_package.my_module:MyGuard"
```

1. Use it in a run:

```yaml
guards:
  order: ["invariants", "spectral", "my_guard", "variance"]
  my_guard:
    # plugin-specific settings
    enabled: true
```

Then invoke your config as usual:

```bash
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

> Retries reuse a single loaded model; the runner snapshot/restores state
> between attempts for reproducible bare/guarded comparisons.
