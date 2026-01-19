from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _import_verify_module():
    # transformers stub so importing verify (which imports run) is safe
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self):
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto  # type: ignore[attr-defined]
        tr.GPT2Tokenizer = _GPT2  # type: ignore[attr-defined]
        sys.modules["transformers"] = tr
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object  # type: ignore[attr-defined]
        sys.modules["transformers.tokenization_utils_base"] = sub
    return importlib.import_module("invarlock.cli.commands.verify")


def test_primary_metric_validation_ppl_and_non_ppl() -> None:
    verify_mod = _import_verify_module()
    cert = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 20.0,
            "ratio_vs_baseline": 2.0,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }
    assert verify_mod._validate_primary_metric(cert) == []
    # Mismatch ratio
    cert_bad = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 20.0,
            "ratio_vs_baseline": 1.5,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }
    assert any(
        "ratio mismatch" in e for e in verify_mod._validate_primary_metric(cert_bad)
    )
    # Non-ppl requires ratio present
    cert_np = {"primary_metric": {"kind": "accuracy", "final": 0.9}}
    assert verify_mod._validate_primary_metric(cert_np)

    # Invalid primary metric (e.g., NaN/Inf weights during error injection) should not
    # be treated as malformed for integrity verification.
    cert_invalid = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": float("nan"),
            "ratio_vs_baseline": float("nan"),
            "invalid": True,
            "degraded_reason": "non_finite_pm",
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }
    assert verify_mod._validate_primary_metric(cert_invalid) == []

    cert_nonfinite_undeclared = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": float("nan"),
            "ratio_vs_baseline": float("nan"),
            "invalid": False,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }
    assert verify_mod._validate_primary_metric(cert_nonfinite_undeclared)


def test_primary_metric_validation_rejects_baseline_zero_and_missing_ratio() -> None:
    verify_mod = _import_verify_module()

    cert_zero = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
        },
        "baseline_ref": {"primary_metric": {"final": 0.0}},
    }
    assert verify_mod._validate_primary_metric(cert_zero)

    cert_missing_ratio = {
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "baseline_ref": {"primary_metric": {"final": 5.0}},
    }
    assert verify_mod._validate_primary_metric(cert_missing_ratio)


def test_pairing_counts_and_drift_band() -> None:
    verify_mod = _import_verify_module()
    cert = {
        "dataset": {
            "windows": {
                "preview": 2,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 1}},
                    "paired_windows": 2,
                },
            }
        }
    }
    assert verify_mod._validate_pairing(cert) == []
    assert verify_mod._validate_counts(cert) == []
    # Drift band fails out of range
    cert_pm = {"primary_metric": {"preview": 1.0, "final": 1.2}}
    errs = verify_mod._validate_drift_band(cert_pm)
    assert errs and "out of band" in errs[0]
    # Drift band is undefined for invalid primary metrics
    assert (
        verify_mod._validate_drift_band(
            {
                "primary_metric": {
                    "preview": float("nan"),
                    "final": float("nan"),
                    "invalid": True,
                }
            }
        )
        == []
    )


def test_pairing_rejects_low_match_high_overlap_and_missing_values() -> None:
    verify_mod = _import_verify_module()

    cert_bad = {
        "dataset": {
            "windows": {
                "stats": {
                    "window_match_fraction": 0.9,
                    "window_overlap_fraction": 0.1,
                    "window_pairing_reason": None,
                    "paired_windows": 1,
                }
            }
        }
    }
    assert len(verify_mod._validate_pairing(cert_bad)) == 2

    cert_missing = {"dataset": {"windows": {"stats": {}}}}
    assert len(verify_mod._validate_pairing(cert_missing)) == 3


def test_counts_mismatch_and_missing_values() -> None:
    verify_mod = _import_verify_module()

    cert_missing = {
        "dataset": {"windows": {"preview": 2, "final": 1, "stats": {"coverage": {}}}}
    }
    assert len(verify_mod._validate_counts(cert_missing)) == 2

    cert_mismatch = {
        "dataset": {
            "windows": {
                "preview": 2,
                "final": 1,
                "stats": {
                    "coverage": {"preview": {"used": 3}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            }
        }
    }
    assert len(verify_mod._validate_counts(cert_mismatch)) == 2


def test_validate_drift_band_boundaries_and_missing_values() -> None:
    verify_mod = _import_verify_module()

    # Boundary values are allowed: inclusive [0.95, 1.05]
    assert (
        verify_mod._validate_drift_band(
            {"primary_metric": {"preview": 10.0, "final": 9.5}}
        )
        == []
    )
    assert (
        verify_mod._validate_drift_band(
            {"primary_metric": {"preview": 10.0, "final": 10.5}}
        )
        == []
    )

    # Just outside the band must fail
    assert verify_mod._validate_drift_band(
        {"primary_metric": {"preview": 10.0, "final": 9.49}}
    )
    assert verify_mod._validate_drift_band(
        {"primary_metric": {"preview": 10.0, "final": 10.51}}
    )

    # Missing/invalid preview/final should report missing drift basis
    assert verify_mod._validate_drift_band({"primary_metric": {"final": 1.0}})
    assert verify_mod._validate_drift_band(
        {"primary_metric": {"preview": 0.0, "final": 1.0}}
    )
    assert verify_mod._validate_drift_band(
        {"primary_metric": {"preview": 10.0, "final": float("nan")}}
    )


def test_validate_drift_band_parses_override_shapes() -> None:
    verify_mod = _import_verify_module()

    cert_dict = {
        "primary_metric": {
            "preview": 10.0,
            "final": 12.0,
            "drift_band": {"min": 0.9, "max": 1.3},
        }
    }
    assert verify_mod._validate_drift_band(cert_dict) == []

    cert_list = {
        "primary_metric": {"preview": 10.0, "final": 12.0, "drift_band": [0.9, 1.3]}
    }
    assert verify_mod._validate_drift_band(cert_list) == []

    cert_invalid = {
        "primary_metric": {
            "preview": 10.0,
            "final": 12.0,
            "drift_band": {"min": "bad", "max": "1.3"},
        }
    }
    assert verify_mod._validate_drift_band(cert_invalid)


def test_coercion_helpers_and_measurement_contract_digest() -> None:
    verify_mod = _import_verify_module()

    assert verify_mod._coerce_float("1.5") == 1.5
    assert verify_mod._coerce_float(2) == 2.0
    assert verify_mod._coerce_float(None) is None
    assert verify_mod._coerce_float("nope") is None
    assert verify_mod._coerce_float(float("nan")) is None

    assert verify_mod._coerce_int("3") == 3
    assert verify_mod._coerce_int(0) == 0
    assert verify_mod._coerce_int(-1) is None
    assert verify_mod._coerce_int("nope") is None

    contract1 = {"a": 1, "b": {"c": 2}}
    contract2 = {"b": {"c": 2}, "a": 1}  # different ordering
    d1 = verify_mod._measurement_contract_digest(contract1)
    d2 = verify_mod._measurement_contract_digest(contract2)
    assert isinstance(d1, str) and len(d1) == 16
    assert d1 == d2

    assert verify_mod._measurement_contract_digest({}) is None
    assert verify_mod._measurement_contract_digest("nope") is None

    class _BadStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert verify_mod._measurement_contract_digest({"x": _BadStr()}) is None


def test_validate_measurement_contracts_ci_requires_baseline_match() -> None:
    verify_mod = _import_verify_module()

    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }

    cert = {
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": False,
        },
        "rmt": {
            "evaluated": True,
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": False,
        },
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
    }

    errs = verify_mod._validate_measurement_contracts(cert, profile="ci")
    assert len(errs) == 2
    assert any("spectral measurement contract must match baseline" in e for e in errs)
    assert any("rmt measurement contract must match baseline" in e for e in errs)


def test_validate_measurement_contracts_hash_and_resolved_policy_mismatches() -> None:
    verify_mod = _import_verify_module()

    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    resolved_policy_contract = {
        "estimator": {"type": "power_iter", "iters": 5, "init": "ones"}
    }

    spectral_hash = verify_mod._measurement_contract_digest(spectral_contract)
    resolved_hash = verify_mod._measurement_contract_digest(resolved_policy_contract)
    assert isinstance(spectral_hash, str) and spectral_hash
    assert isinstance(resolved_hash, str) and resolved_hash

    cert = {
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": "deadbeefdeadbeef",
            "measurement_contract_match": True,
        },
        "rmt": {"evaluated": False},
        "resolved_policy": {
            "spectral": {"measurement_contract": resolved_policy_contract},
        },
    }

    errs = verify_mod._validate_measurement_contracts(cert, profile="ci")
    assert len(errs) == 2
    assert any("measurement_contract_hash mismatch" in e for e in errs)
    assert any("differs between analysis and resolved_policy" in e for e in errs)


def test_validate_measurement_contracts_skips_unevaluated_guards() -> None:
    verify_mod = _import_verify_module()
    cert = {
        "spectral": {"evaluated": False},
        "rmt": {"evaluated": False},
        "resolved_policy": {},
    }
    assert verify_mod._validate_measurement_contracts(cert, profile="ci") == []


def test_pairing_rejects_pairing_reason_and_zero_pairs() -> None:
    verify_mod = _import_verify_module()
    cert_reason = {
        "dataset": {
            "windows": {
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_pairing_reason": "no_baseline_reference",
                    "paired_windows": 1,
                }
            }
        }
    }
    errs = verify_mod._validate_pairing(cert_reason)
    assert errs and any("window_pairing_reason" in e for e in errs)

    cert_zero = {
        "dataset": {
            "windows": {
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_pairing_reason": None,
                    "paired_windows": 0,
                }
            }
        }
    }
    errs2 = verify_mod._validate_pairing(cert_zero)
    assert errs2 and any("paired_windows" in e for e in errs2)


def test_tokenizer_hash_and_profile_lints() -> None:
    verify_mod = _import_verify_module()
    # Tokenizer hash checks: enforced only when both baseline and edited hashes exist
    cert_ok = {
        "meta": {"tokenizer_hash": "abc"},
        "baseline_ref": {"tokenizer_hash": "abc"},
    }
    assert verify_mod._validate_tokenizer_hash(cert_ok) == []

    cert_mismatch = {
        "meta": {"tokenizer_hash": "abc"},
        "baseline_ref": {"tokenizer_hash": "xyz"},
    }
    errs_tok = verify_mod._validate_tokenizer_hash(cert_mismatch)
    assert errs_tok and "Tokenizer hash mismatch" in errs_tok[0]

    cert_missing = {"meta": {"tokenizer_hash": "abc"}, "baseline_ref": {}}
    assert verify_mod._validate_tokenizer_hash(cert_missing) == []

    # Lints
    cert_lints_ok = {
        "meta": {
            "model_profile": {
                "cert_lints": [
                    {"type": "equals", "path": "x.y", "value": 1, "message": "E"},
                    {"type": "gte", "path": "a", "value": 2, "message": "G"},
                    {"type": "lte", "path": "b", "value": 5, "message": "L"},
                ]
            }
        },
        "x": {"y": 1},
        "a": 2.0,
        "b": 5.0,
    }
    assert verify_mod._apply_profile_lints(cert_lints_ok) == []

    cert_lints = {
        "meta": {
            "model_profile": {
                "cert_lints": [
                    {"type": "equals", "path": "x.y", "value": 1, "message": "E"},
                    {"type": "gte", "path": "a", "value": 2},
                    {"type": "lte", "path": "b", "value": 5},
                ]
            }
        },
        "x": {"y": 2},  # equals fails
        "a": 1,  # gte fails
        "b": 10,  # lte fails
    }
    errs = verify_mod._apply_profile_lints(cert_lints)
    assert any("Expected x.y ==" in e for e in errs)
    assert any("Expected a ≥" in e for e in errs)
    assert any("Expected b ≤" in e for e in errs)
    assert len(errs) == 3

    cert_non_numeric = {
        "meta": {
            "model_profile": {"cert_lints": [{"type": "gte", "path": "a", "value": 2}]}
        },
        "a": "not-a-number",
    }
    errs_num = verify_mod._apply_profile_lints(cert_non_numeric)
    assert errs_num and "numeric comparison" in errs_num[0]

    # Resolve path helper
    payload = {"a": {"b": {"c": 3}}}
    assert verify_mod._resolve_path(payload, "a.b.c") == 3
    assert verify_mod._resolve_path(payload, "a.b.missing") is None


def test_validate_certificate_payload_schema_failure(tmp_path: Path) -> None:
    verify_mod = _import_verify_module()
    bad = tmp_path / "bad_cert.json"
    bad.write_text("{}\n")
    errs = verify_mod._validate_certificate_payload(bad)
    assert errs and "schema" in errs[0].lower()


def test_warn_adapter_family_mismatch(tmp_path: Path) -> None:
    verify_mod = _import_verify_module()
    # Create a baseline report with adapter provenance
    baseline = tmp_path / "baseline_report.json"
    baseline.write_text(
        """
        {
          "meta": {
            "plugins": {
              "adapter": {
                "provenance": {"family": "hf", "library": "transformers", "version": "0.0"}
              }
            }
          }
        }
        """.strip()
    )

    cert = {
        "plugins": {
            "adapter": {
                "provenance": {"family": "ggml", "library": "ggml", "version": "0.0"}
            }
        },
        "provenance": {"baseline": {"report_path": str(baseline)}},
    }
    # Should not raise; may emit a soft warning to console
    verify_mod._warn_adapter_family_mismatch(baseline, cert)
