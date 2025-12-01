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


def test_tokenizer_hash_and_profile_lints(tmp_path: Path) -> None:
    verify_mod = _import_verify_module()
    cert = {
        "meta": {"tokenizer_hash": "abc"},
        "dataset": {"tokenizer": {"hash": "abc"}},
        "baseline_ref": {"tokenizer_hash": "xyz"},
    }
    # Mismatch triggers error when both present
    assert verify_mod._validate_tokenizer_hash(cert)

    # Lints
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
