# Guards System Reference

InvarLock's guard system provides comprehensive safety mechanisms to prevent model
degradation during editing operations. This guide covers all available guards,
their configuration, and best practices.

## Overview

The guard system implements a multi-layered approach to model safety:

1. **Invariants Guard**: Validates structural integrity and parameter sanity
2. **Spectral Guard**: Controls weight matrix spectral properties for stability
3. **RMT Guard**: Enforces baseline-relative activation edge-risk ε-bands
4. **Variance Guard**: Equalizes activation variances for improved performance

Guards operate in a pipeline with configurable policies and automatic intervention capabilities.

## Guard Lifecycle

Each guard follows a standard lifecycle:

```text
1. prepare()    → Initialize baseline measurements
2. before_edit() → Pre-edit validation
3. after_edit()  → Post-edit validation and intervention
4. finalize()    → Final validation and reporting
```

## Available Guards

### Invariants Guard

**Purpose**: Validates fundamental model properties and structural integrity

**Key Checks**:

- Non-finite detection (parameters/buffers must remain finite)
- LayerNorm presence (baseline LayerNorm modules must still exist after edits)
- Tokenizer alignment (embedding vocab sizes stay in sync with tokenizer/config)
- Parameter count consistency and structure hash stability
- Weight tying preservation across supported architectures (GPT-2, BERT, LLaMA)
- Optional profile-specific checks (e.g., adapter-provided invariants)

**Configuration**:

```yaml
guards:
  invariants:
    strict_mode: false        # Enable strict validation
    on_fail: "abort"          # Action on failure: "warn", "rollback", or "abort"
    profile_checks:
      - "custom::sample_invariant"
```

**Usage Example**:

```python
from invarlock.guards import InvariantsGuard

guard = InvariantsGuard(strict_mode=True, on_fail="abort")

# Prepare baseline
result = guard.prepare(model, adapter, calib_data, policy)

# Check after editing
outcome = guard.finalize(model)
if not outcome.passed:
    print(f"Violations: {outcome.violations}")
```

**Validation Criteria**:

- ✅ No NaN or infinite values in parameters or buffers
- ✅ LayerNorm inventory matches the baseline capture
- ✅ Embedding vocab sizes match tokenizer/config metadata
- ✅ Parameter count unchanged (unless an explicit profile check allows it)
- ✅ Weight tying relationships preserved
- ✅ Model structure hash unchanged
- ✅ Any profile-specific checks return success (when configured)

### Spectral Guard

**Purpose**: Controls spectral properties of weight matrices for numerical stability.

> **Balanced defaults (policy shipped in tiers):** `scope: all`,
> `sigma_quantile: 0.95`, `deadband: 0.10`, `max_caps: 5`,
> `max_spectral_norm: null`,
> `multiple_testing: {method: bh, alpha: 0.05, m: 4}` with κ
> `{ffn: 3.849, attn: 3.018, embed: 1.05, other: 0.0}`.
>
> **Conservative note:** Bonferroni (`alpha = 0.000625`), `max_caps = 3`, κ
> `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}`.

**Key features**

- Monitors per-family spectral norms with calibrated κ caps.
- Budget-aware WARNs via BH/Bonferroni multiple testing.
- Optional rescaling/correction for tighter tiers.
- Emits policy + metrics so reviewers can recompute thresholds.

**Balanced quick-start (copy/paste).**

```yaml
guards:
  spectral:
    sigma_quantile: 0.95
    deadband: 0.10
    scope: all
    max_caps: 5
    multiple_testing: { method: bh, alpha: 0.05, m: 4 }
    family_caps: { ffn: {kappa: 3.849}, attn: {kappa: 3.018}, embed: {kappa: 1.05}, other: {kappa: 0.0} }
    max_spectral_norm: null
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    predictive_one_sided: true
    min_effect_lognll: 0.0
```

Use this baseline and trial calibrated κ shifts via small local overrides rather than
editing shared tier files. Preferred flow: emit a local guard-override YAML during
calibration (or copy the calibrated values into a local preset) and apply it by
adding a `guards:` override block to your run config. If you prefer to keep a
hand‑edited example under version control, use a pattern like
`configs/overrides/spectral_balanced_local.example.yaml` and copy it locally
before editing.

Prefer calibrating κ via the budget-aware recipe and keep `multiple_testing`,
`deadband = 0.10`, `max_caps = 5`, and `max_spectral_norm: null` unchanged
unless policy review explicitly approves a change. To exercise a local override
during release evidence, use a local preset that includes the guard overrides:

```bash
# configs/local/quant8_calibrated.yaml = base preset + guards overrides
invarlock run -c configs/local/quant8_calibrated.yaml \
  --profile release --tier balanced \
  --baseline runs/.../baseline/report.json \
  --out runs/.../quant8_calibrated
```

**Naming.** Standardise on `sigma_quantile` and `multiple_testing` (with
underscore) to keep
reports/overrides schema-aligned.

**Spectral is weight-based.** |z| distributions depend solely on weights;
re-sampling evaluation windows or dataset seeds will not move tails. Prefer
pooling per-module |z| across related baselines (e.g., 1B/3B/7B siblings) to
steady the estimate.

**Observability cheat sheet.**

- `/.guards[] | select(.name=="spectral") | .policy`
- `/.guards[] | select(.name=="spectral") | .metrics | {caps_applied,caps_exceeded,max_caps}`
- `/spectral.{summary,families,policy,multiple_testing}` in certificates
- `/.guards[] | select(.name=="spectral") | .final_z_scores` (emitted when calibration logging is enabled)
- **Checkpoints.**

  ```bash
  jq '.guards[] | select(.name=="spectral") | .policy' report.json
  jq '.guards[] | select(.name=="spectral") | .metrics | {caps_applied,caps_exceeded,max_caps}' report.json
  jq '.spectral' reports/quant8_small_cert.json
  ```

**Advanced configuration.**

```yaml
guards:
  spectral:
    # Conservative (quality-first)
    sigma_quantile: 0.90
    deadband: 0.05
    scope: ffn
    multiple_testing: { method: bonferroni, alpha: 0.000625, m: 4 }
    max_caps: 3

    # Aggressive (compression-first)
    correction_enabled: false
    sigma_quantile: 0.98
    deadband: 0.15
    scope: all

    # Per-family κ overrides
    family_caps:
      attn: {kappa: 2.6}
      ffn: {kappa: 2.9}
```

**Usage example**

```python
from invarlock.guards.spectral import SpectralGuard, capture_baseline_sigmas

guard = SpectralGuard(sigma_quantile=0.95, deadband=0.10, scope="all", correction_enabled=False)
baselines = capture_baseline_sigmas(model)

result = guard.validate(model, adapter, {"baseline_metrics": baselines})
if not result["passed"]:
    print(f"Spectral violation: {result['message']}")
```

**Spectral analysis helpers**

```python
from invarlock.guards.spectral import (
    compute_sigma_max,
    apply_spectral_control,
    apply_relative_spectral_cap,
    scan_model_gains,
)

sigma = compute_sigma_max(layer.weight)
baseline_sigmas = capture_baseline_sigmas(model)

apply_relative_spectral_cap(model, cap_ratio=1.5, baseline_sigmas=baseline_sigmas)
gains = scan_model_gains(model)
```

### RMT Guard

**Purpose**: Enforces baseline-relative activation edge-risk stability (ε-band).

**Key Features**:

- **Activation edge-risk scoring** on sampled, token-weighted activations under a fixed
  measurement contract.
- **Per-family ε thresholds** via `epsilon_by_family` (accept when each family stays within
  `(1 + ε_family) × baseline`).
- **Fail-closed in CI/Release** when activation evidence is required but unavailable.
- **Evidence emission**: `edge_risk_by_family_base`, `edge_risk_by_family`, `epsilon_violations`,
  and `measurement_contract`.

**Configuration**:

```yaml
guards:
  rmt:
    # ε-band thresholds (primary acceptance rule)
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }

    # Optional: pin the measurement contract for reproducibility
    estimator: { iters: 3, init: ones }
    activation:
      sampling:
        windows: { count: 8, indices_policy: evenly_spaced }

    # Weight-RMT knobs (diagnostics/correction helpers)
    q: auto
    deadband: 0.10
    margin: 1.5
    correct: false
```

**Usage Example**:

```python
from invarlock.guards.rmt import RMTGuard

guard = RMTGuard(
    epsilon_default=0.01,
    epsilon_by_family={"ffn": 0.01, "attn": 0.01, "embed": 0.01, "other": 0.01},
)
guard.prepare(
    model,
    adapter,
    calib_data,
    policy={
        "estimator": {"iters": 3, "init": "ones"},
        "activation": {
            "sampling": {"windows": {"count": 8, "indices_policy": "evenly_spaced"}}
        },
    },
)

# ...perform edit...

guard.after_edit(model)
outcome = guard.finalize(model)
print(f"RMT passed: {outcome.passed}")
print(f"ε violations: {outcome.metrics['epsilon_violations']}")
```

**Weight-RMT utilities (diagnostics / correction)**:

```python
from invarlock.guards.rmt import (
    mp_bulk_edge,
    layer_svd_stats,
    analyze_weight_distribution,
    clip_full_svd
)

# Compute Marchenko-Pastur bulk edge
edge = mp_bulk_edge(m=768, n=512, whitened=True)

# Analyze layer statistics
stats = layer_svd_stats(layer, baseline_sigmas, baseline_mp_stats, layer_name)

# Analyze weight distribution
dist_stats = analyze_weight_distribution(model, n_bins=50)

# Clip singular values
clipped_weight = clip_full_svd(weight_matrix, clip_val=2.0)
```

### Variance Guard

**Purpose**: Equalizes residual activation variances for improved stability

**Key Features**:

- **Variance Equalization**: Balances activation variances across layers
- **A/B Testing**: Validates improvements through empirical testing
- **Adaptive Scaling**: Applies learned scaling factors
- **Performance Gating**: Only enables if demonstrated improvement

**Configuration**:

```yaml
guards:
  variance:
    min_gain: 0.0              # Minimum improvement threshold (Balanced tier)
    max_calib: 200             # Maximum calibration samples (Balanced tier)
    scope: "ffn"               # Target scope: "ffn" for RTN demo
    clamp: [0.85, 1.12]        # Scaling factor limits (Balanced tier)
    deadband: 0.02             # Improvement deadband (Balanced tier)
    seed: 123                  # Random seed for reproducibility
```

**Usage Example**:

```python
from invarlock.guards.variance import VarianceGuard, equalise_residual_variance
from invarlock.guards.policies import get_variance_policy

# Create guard with balanced policy
policy = get_variance_policy("balanced")
guard = VarianceGuard(policy)

# Prepare and check readiness
result = guard.prepare(model, adapter, calib_data, policy)
if result["ready"]:
    # Enable variance equalization
    guard.enable(model)

    # Set A/B test results
    guard.set_ab_results(
        ppl_no_ve=3.5,
        ppl_with_ve=3.2,
        windows_used=50,
        seed_used=42
    )

    # Finalize and check if improvement was sufficient
    outcome = guard.finalize(model)
```

**Variance Analysis**:

```python
# Compute variance scaling factors
scales = equalise_residual_variance(
    model=model,
    dataloader=calib_data,
    windows=50,
    tol=0.01,
    clamp_range=(0.8, 1.2)
)

print(f"Computed scales: {scales}")
```

## Guard Policies

### Predefined Policies

**Conservative (Quality-First)**:

```yaml
guards:
  spectral:
    sigma_quantile: 0.90         # Tight spectral percentile
    deadband: 0.05               # Small tolerance
    correction_enabled: true     # Conservative tier enables correction
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
    # activation_required defaults to true in CI/Release profiles
  variance:
    min_gain: 0.01               # High improvement requirement
```

**Balanced (Default)**:

```yaml
guards:
  spectral:
    sigma_quantile: 0.95         # Balanced spectral percentile
    deadband: 0.10               # Standard deadband
    correction_enabled: false    # Monitor-only spectral caps
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    min_gain: 0.0                # VE gain floor (paired with min_effect_lognll)
```

**Aggressive (Compression-First)**:

```yaml
guards:
  spectral:
    sigma_quantile: 0.98         # Loose spectral percentile
    deadband: 0.15            # Large tolerance
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    min_gain: 0.0             # Lower improvement requirement
```

### Model-Size Adaptive Policies

```python
from invarlock.guards.policies import get_policy_for_model_size

# Automatic policy selection based on model size
total_params = sum(p.numel() for p in model.parameters())
policy = get_policy_for_model_size(total_params)

# Small models (< 100M params) use aggressive settings
# Large models (> 1B params) use conservative settings
```

### Custom Policies

```python
from invarlock.guards.policies import (
    create_custom_spectral_policy,
    create_custom_rmt_policy,
    create_custom_variance_policy
)

# Create custom spectral policy
spectral_policy = create_custom_spectral_policy(
    sigma_quantile=0.93,
    deadband=0.08,
    scope="ffn"
)

# Create custom RMT policy
rmt_policy = create_custom_rmt_policy(
    q=2.0,
    deadband=0.12,
    margin=1.6,
    correct=True,
)
rmt_policy["epsilon_default"] = 0.01
rmt_policy["epsilon_by_family"] = {"ffn": 0.01, "attn": 0.01, "embed": 0.01, "other": 0.01}

# Create custom variance policy
variance_policy = create_custom_variance_policy(
    min_gain=0.22,
    max_calib=120,
    scope="both",
    clamp=(0.75, 1.25)
)
```

## Guard Chain Configuration

### Sequential Guard Execution

```yaml
guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]

  # Individual guard configurations
  invariants:
    strict_mode: false
  spectral:
    sigma_quantile: 0.95
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    min_gain: 0.20
```

### Conditional Guard Execution

```yaml
guards:
  # Always run invariants
  invariants:
    enabled: true

  # Only run spectral for certain edit types
  spectral:
    enabled_for: ["quant_rtn"]

  # Skip RMT for small models
  rmt:
    min_params: 100000000     # 100M parameters minimum

  # Only run variance for FFN-targeting edits
  variance:
    required_scope: ["ffn", "both"]
```

### Failure Handling

```yaml
guards:
  on_failure: "abort"         # Global failure action

  # Per-guard failure actions
  spectral:
    on_failure: "warn"        # Just warn, don't abort
  rmt:
    on_failure: "correct"     # Attempt correction
  invariants:
    on_failure: "abort"       # Always abort on invariant violation
```

## Advanced Usage

### Guard Integration in Pipeline

```python
from invarlock.core.runner import CoreRunner
from invarlock.core.api import GuardChain

# Create guard chain
guards = GuardChain([
    InvariantsGuard(strict_mode=True),
    SpectralGuard(sigma_quantile=0.95),
    RMTGuard(margin=1.5),
    VarianceGuard(get_variance_policy("balanced"))
])

# Configure runner with guards
runner = CoreRunner(guard_chain=guards)

# Run with automatic guard validation
result = runner.run(config)
```

### Custom Guard Development

```python
from invarlock.core.api import Guard

class CustomGuard(Guard):
    name = "custom_guard"

    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.prepared = False

    def prepare(self, model, adapter, calib, policy):
        # Initialize baseline measurements
        self.baseline = self._compute_baseline(model)
        self.prepared = True
        return {"ready": True, "baseline_metrics": self.baseline}

    def validate(self, model, adapter, context):
        # Validate model state
        current = self._compute_current(model)
        passed = abs(current - self.baseline) < self.threshold

        return {
            "passed": passed,
            "action": "continue" if passed else "warn",
            "message": f"Custom metric: {current:.3f}",
            "metrics": {"current": current, "baseline": self.baseline}
        }

    def _compute_baseline(self, model):
        # Custom baseline computation
        pass

    def _compute_current(self, model):
        # Current state computation
        pass
```

### Guard Monitoring and Logging

```python
import time
from invarlock.observability import MonitoringManager

monitor = MonitoringManager()
monitor.start()

spectral_guard = SpectralGuard(sigma_quantile=0.95)

started = time.perf_counter()
result = spectral_guard.validate(model, adapter, context)
duration = time.perf_counter() - started

monitor.record_operation(
    "spectral.validate",
    duration,
    guard=spectral_guard.name,
    passed=result["passed"],
)
if not result["passed"]:
    monitor.record_error("spectral_guard", result["message"], guard=spectral_guard.name)

monitor.stop()
```

## Performance Considerations

### Guard Overhead

| Guard      | Typical Overhead | Memory Impact               |
| ---------- | ---------------- | --------------------------- |
| Invariants | < 0.1s           | Minimal                     |
| Spectral   | 0.1-0.5s         | Low                         |
| RMT        | 0.2-1.0s         | Moderate                    |
| Variance   | 0.5-2.0s         | High (requires calibration) |

### Optimization Tips

 1. **Selective Guard Usage**: Only enable necessary guards
2. **Scope Limitation**: Target specific modules (e.g., `scope: "ffn"`)
3. **Calibration Data Size**: Limit samples for variance guard
4. **Baseline Caching**: Cache baseline measurements when possible

```yaml
# Optimized configuration for large models
guards:
  invariants:
    enabled: true             # Always check invariants
  spectral:
    enabled: true
    scope: "ffn"              # Focus on FFN layers only
  rmt:
    enabled: false            # Skip for performance
  variance:
    enabled: true
    max_calib: 50             # Limit calibration samples
```

## Troubleshooting

### Common Issues

**1. Guard Preparation Failures**

```python
# Check guard compatibility
if not guard.can_handle(model):
    print(f"Guard {guard.name} cannot handle this model type")

# Verify calibration data format
if not hasattr(calib_data, '__iter__'):
    print("Calibration data must be iterable")
```

**2. Spectral Violations**

```python
# Debug spectral issues
from invarlock.guards.spectral import scan_model_gains

gains = scan_model_gains(model)
for layer, gain in gains.items():
    if gain > threshold:
        print(f"High spectral norm in {layer}: {gain:.3f}")
```

**3. RMT ε‑Band Violations**

```python
# Inspect ε-band violations (activation edge-risk)
guard.after_edit(model)
outcome = guard.finalize(model)
for violation in outcome.metrics["epsilon_violations"]:
    print(
        f"{violation['family']}: "
        f"edge_cur={violation['edge_cur']:.3f} "
        f"> allowed={violation['allowed']:.3f} "
        f"(ε={violation['epsilon']:.3f})"
    )
```

**4. Variance Guard Issues**

```python
# Check A/B test results
guard.set_ab_results(ppl_no_ve=3.5, ppl_with_ve=3.2)
should_enable, reason = guard._evaluate_ab_gate()
print(f"Variance equalization decision: {should_enable} ({reason})")
```

### Best Practices

1. **Start Conservative**: Begin with conservative policies and relax as needed
2. **Monitor Performance**: Track guard overhead and adjust accordingly
3. **Validate Improvements**: Use A/B testing to verify guard effectiveness
4. **Log Everything**: Enable comprehensive logging for debugging
5. **Test Combinations**: Validate guard interactions with different edit types

## Integration Examples

### CLI Integration

```bash
# Run with custom guard configuration embedded in the config YAML
invarlock run --config my_config.yaml --out runs/guard_demo

# Override specific guard settings by editing the `guards:` block in the config
```

### Programmatic Integration

```python
# Complete pipeline with guards
from invarlock.core.runner import CoreRunner
from invarlock.core.api import GuardChain
from invarlock.guards import InvariantsGuard, RMTGuard, SpectralGuard, VarianceGuard
from invarlock.guards.policies import get_variance_policy

guard_chain = GuardChain([
    InvariantsGuard(strict_mode=True),
    SpectralGuard(sigma_quantile=0.95, deadband=0.10),
    RMTGuard(
        epsilon_default=0.01,
        epsilon_by_family={"ffn": 0.01, "attn": 0.01, "embed": 0.01, "other": 0.01},
    ),
    VarianceGuard(get_variance_policy("balanced")),
])

# Run pipeline
runner = CoreRunner(guard_chain=guard_chain)
result = runner.run(config)

# Check guard results
for guard_name, guard_result in result["guard_results"].items():
    print(f"{guard_name}: {'PASS' if guard_result.passed else 'FAIL'}")
```

The guard system provides comprehensive protection while maintaining flexibility
for different use cases and performance requirements.
