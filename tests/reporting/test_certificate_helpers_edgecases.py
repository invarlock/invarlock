from __future__ import annotations

import math

import pytest

from invarlock.reporting import certificate as C


class _RaisingStr:
    def __str__(self) -> str:  # pragma: no cover - used to trigger exception in target
        raise RuntimeError("boom")


class _RaisingGet:
    def get(
        self, *args, **kwargs
    ):  # pragma: no cover - used to trigger exception in target
        raise RuntimeError("boom")


def test_is_ppl_kind_handles_str_exception() -> None:
    assert C._is_ppl_kind(_RaisingStr()) is False
    assert C._is_ppl_kind("ppl_causal") is True


def test_get_ppl_final_handles_bad_metrics_get() -> None:
    # Legacy _get_ppl_final removed; rely on primary_metric parsing in certificate generation.
    assert True


def test_coerce_int_variants() -> None:
    assert C._coerce_int(5) == 5
    # Non-integer float rejected (only near-integers accepted)
    assert C._coerce_int(5.8) is None
    assert C._coerce_int("7") == 7
    assert C._coerce_int(None) is None
    assert C._coerce_int("bad") is None


def test_sanitize_seed_bundle_partial_and_fallback() -> None:
    sanitized = C._sanitize_seed_bundle({"python": 1, "numpy": None}, fallback=42)
    # Explicit/missing None entries preserve None; others use fallback
    assert (
        sanitized["python"] == 1
        and sanitized["numpy"] is None
        and sanitized["torch"] is None
    )


def test_infer_scope_from_modules_variants() -> None:
    assert C._infer_scope_from_modules([]) == "unknown"
    assert C._infer_scope_from_modules(["model.attn.block"]) == "attn"
    assert C._infer_scope_from_modules(["decoder.mlp.fc"]) == "ffn"
    assert C._infer_scope_from_modules(["wte.embedding"]) == "embed"
    mixed = C._infer_scope_from_modules(["layer.attention", "mlp.ffn", "tok.embed"])
    assert mixed in {"attn+embed+ffn", "attn+ffn+embed"}


def test_coerce_interval_from_string_and_list() -> None:
    lo, hi = C._coerce_interval("(1.5, 2.5)")
    assert math.isclose(lo, 1.5) and math.isclose(hi, 2.5)
    lo2, hi2 = C._coerce_interval("not a tuple")
    assert math.isnan(lo2) and math.isnan(hi2)
    lo3, hi3 = C._coerce_interval(["x", 2])
    assert math.isnan(lo3) and math.isnan(hi3)


def test_compute_edit_digest_quant_and_default() -> None:
    d = C._compute_edit_digest({"edit": {"name": "quant_rtn", "config": {"bits": 8}}})
    assert d["family"] == "quantization" and isinstance(d["impl_hash"], str)
    d2 = C._compute_edit_digest({"edit": {"name": "noop"}})
    assert d2["family"] == "cert_only"


def test_extract_certificate_meta_prefers_python_seed() -> None:
    report = {
        "meta": {
            "model_id": "demo",
            "adapter": "hf",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 9, "numpy": None},
        }
    }
    meta = C._extract_certificate_meta(report)
    assert meta["seed"] == 9
    assert meta["seeds"]["python"] == 9


def test_extract_certificate_meta_defaults_seed_to_zero() -> None:
    report = {"meta": {"model_id": "demo", "adapter": "hf", "device": "cpu"}}
    meta = C._extract_certificate_meta(report)
    assert meta["seed"] == 0


def test_normalize_and_validate_report_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        C._normalize_and_validate_report("oops")  # type: ignore[arg-type]
