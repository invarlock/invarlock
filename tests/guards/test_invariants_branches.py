from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.guards.invariants import InvariantsGuard


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden: int = 4):
        super().__init__()
        # LayerNorm to be tracked
        self.ln = nn.LayerNorm(hidden)
        # Embedding to track vocab size
        self.embed = nn.Embedding(vocab_size, hidden)
        # GPT-2 style names for weight tying check
        self.transformer = SimpleNamespace(wte=self.embed)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # Tie weights by reference for capture (no forward needed in tests)
        self.lm_head.weight = self.embed.weight  # parameter alias
        # Minimal config to satisfy profile checks
        self.config = SimpleNamespace(model_type="gpt2", is_decoder=True)

    def forward(self, x):  # noqa: D401
        return self.lm_head(self.ln(self.embed(x)))


def test_weight_tying_violation_detected():
    model = TinyModel()
    guard = InvariantsGuard()
    guard.prepare(model, adapter=None, calib=None, policy={})
    # Break weight tying
    with torch.no_grad():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone() + 0.0)
    outcome = guard.finalize(model)
    # Expect invariant_violation for weight_tying change
    assert any(
        v.get("type") == "invariant_violation" and v.get("check") == "weight_tying"
        for v in outcome.violations
    )


def test_strict_invariant_violation_is_fatal():
    model = TinyModel()
    guard = InvariantsGuard(strict_mode=True, on_fail="abort")
    guard.prepare(model, adapter=None, calib=None, policy={})
    # Modify structure to change structure_hash
    model.extra_layer = nn.Linear(1, 1)
    out = guard.finalize(model)
    assert out.passed is False and out.action in {"abort", "rollback"}


def test_invariants_not_prepared_reports_violation():
    model = TinyModel()
    guard = InvariantsGuard(strict_mode=False)
    outcome = guard.finalize(model)
    assert outcome.passed is False
    assert any(v.get("type") == "not_prepared" for v in outcome.violations)


def test_invariants_detects_non_finite_and_missing_layernorm():
    model = TinyModel()
    guard = InvariantsGuard(strict_mode=True, on_fail="abort")
    # Prepare captures baseline invariants
    guard.prepare(model, adapter=None, calib=None, policy={})

    # Inject non-finite into a parameter
    with torch.no_grad():
        model.embed.weight.view(-1)[0] = float("nan")

    # Remove LayerNorm to trigger missing check
    model.ln = nn.Identity()

    outcome = guard.finalize(model)
    assert outcome.passed is False
    assert outcome.action in {"abort", "rollback"}
    types = {v.get("type") for v in outcome.violations}
    assert "non_finite_tensor" in types
    assert "layer_norm_missing" in types
    assert outcome.metrics["fatal_violations"] >= 1


def test_invariants_strict_mode_rollback_vs_warn():
    model = TinyModel()
    # Strict mode with on_fail rollback
    strict_guard = InvariantsGuard(strict_mode=True, on_fail="rollback")
    strict_guard.prepare(model, adapter=None, calib=None, policy={})
    # Remove LayerNorm to force layer_norm_missing in strict mode
    model.ln = nn.Identity()
    out_strict = strict_guard.finalize(model)
    assert out_strict.passed is False and out_strict.action == "rollback"

    # Lenient mode yields warnings and passes
    model2 = TinyModel()
    lenient = InvariantsGuard(strict_mode=False)
    lenient.prepare(model2, adapter=None, calib=None, policy={})
    model2.ln = nn.Identity()
    out_lenient = lenient.finalize(model2)
    assert out_lenient.passed is True and out_lenient.action == "warn"


def test_invariants_lenient_abort_on_warning():
    # When on_fail=abort in lenient mode, warnings cause abort and passed=False
    model = TinyModel()
    guard = InvariantsGuard(strict_mode=False, on_fail="abort")
    guard.prepare(model, adapter=None, calib=None, policy={})
    model.ln = nn.Identity()
    out = guard.finalize(model)
    assert out.passed is False and out.action == "abort"


def test_invariants_non_finite_buffer_detected():
    model = TinyModel()
    guard = InvariantsGuard()
    guard.prepare(model, adapter=None, calib=None, policy={})
    # Register non-finite buffer post-prepare
    model.register_buffer("buf_bad", torch.tensor([float("nan")]))
    out = guard.finalize(model)
    assert any(v.get("type") == "non_finite_tensor" for v in out.violations)


def test_invariants_tokenizer_mismatch_detected():
    model = TinyModel(vocab_size=32, hidden=8)
    guard = InvariantsGuard()
    guard.prepare(model, adapter=None, calib=None, policy={})

    # Change embedding to different vocab size
    model.embed = nn.Embedding(24, 8)
    model.transformer.wte = model.embed
    model.lm_head = nn.Linear(8, 24, bias=False)
    model.lm_head.weight = model.embed.weight

    outcome = guard.finalize(model)
    assert outcome.passed in {True, False}
    types = [v.get("type") for v in outcome.violations]
    assert "tokenizer_mismatch" in types
    assert "tokenizer_mismatches" in outcome.metrics


def test_tokenizer_mismatch_when_embeddings_absent():
    model = TinyModel(vocab_size=16, hidden=4)
    guard = InvariantsGuard()
    guard.prepare(model, adapter=None, calib=None, policy={})
    # Remove embeddings so current vocab sizes dict is empty
    model.embed = nn.Identity()
    model.transformer.wte = None
    outcome = guard.finalize(model)
    types = [v.get("type") for v in outcome.violations]
    assert "tokenizer_mismatch" in types


def test_invariants_profile_checks_respected():
    model = TinyModel()
    guard = InvariantsGuard()
    # Request specific profile checks
    policy = {
        "profile_checks": ["causal_masking", "mlm_mask_alignment", "unknown_check"]
    }
    prep = guard.prepare(model, adapter=None, calib=None, policy=policy)
    assert prep["ready"] is True
    # Baseline capture includes profile:: keys
    keys = guard.baseline_checks.keys()
    assert any(k.startswith("profile::") for k in keys)


def test_invariants_validate_auto_prepares():
    model = TinyModel()
    guard = InvariantsGuard()
    result = guard.validate(model, adapter=None, context={})
    assert isinstance(result, dict) and "passed" in result


def test_invariants_validate_when_prepared():
    model = TinyModel()
    guard = InvariantsGuard()
    guard.prepare(model, adapter=None, calib=None, policy={})
    result = guard.validate(model, adapter=None, context={})
    assert isinstance(result, dict) and "violations" in result


def test_profile_rotary_embedding_detected():
    # Build a minimal LLaMA-like structure with rotary_emb present
    class SelfAttn:
        def __init__(self):
            self.rotary_emb = object()

    class Layer:
        def __init__(self):
            self.self_attn = SelfAttn()

    class LLamaModel:
        def __init__(self):
            self.layers = [Layer()]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = LLamaModel()

    model = M()
    guard = InvariantsGuard()
    policy = {"profile_checks": ["rotary_embedding"]}
    prep = guard.prepare(model, adapter=None, calib=None, policy=policy)
    assert prep["ready"] is True
    assert guard.baseline_checks.get("profile::rotary_embedding") is True


def test_adapter_aware_standard_invariants_violation():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([float("nan")]))

    model = M()
    from invarlock.guards.invariants import check_adapter_aware_invariants

    passed, results = check_adapter_aware_invariants(model)
    assert results.get("adapter_type") == "none"
    assert passed is False or results.get("violations")
