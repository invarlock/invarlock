from __future__ import annotations

from invarlock.cli.commands import verify as v


def test_validate_counts_skips_when_expected_preview_final_missing() -> None:
    cert = {
        "dataset": {
            "windows": {
                "stats": {
                    "paired_windows": 1,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                }
            }
        }
    }
    assert v._validate_counts(cert) == []


def test_validate_counts_emits_paired_windows_mismatch() -> None:
    cert = {
        "dataset": {
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "paired_windows": 1,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                },
            }
        }
    }
    errs = v._validate_counts(cert)
    assert any("Paired window count mismatch" in e for e in errs)


def test_validate_tokenizer_hash_falls_back_to_dataset_tokenizer_hash() -> None:
    cert = {
        "meta": {},
        "dataset": {"tokenizer": {"hash": "aaa"}},
        "baseline_ref": {"tokenizer_hash": "bbb"},
    }
    errs = v._validate_tokenizer_hash(cert)
    assert any("Tokenizer hash mismatch" in e for e in errs)


def test_measurement_contract_digest_returns_none_on_stringify_error() -> None:
    class _Boom:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert v._measurement_contract_digest({"x": _Boom()}) is None


def test_validate_measurement_contracts_skips_non_dict_guard_block() -> None:
    cert = {
        "spectral": "not-a-dict",
        "rmt": {},
        "resolved_policy": {},
    }
    errs = v._validate_measurement_contracts(cert, profile="ci")
    assert isinstance(errs, list)
    assert not any("spectral.measurement_contract" in e for e in errs)


def test_validate_measurement_contracts_missing_hash_and_dev_profile_skip() -> None:
    cert = {
        "spectral": {
            "measurement_contract": {"estimator": {"type": "power_iter"}},
        },
        "rmt": {"evaluated": False},
        "resolved_policy": {
            "spectral": {"measurement_contract": {"estimator": {"type": "power_iter"}}}
        },
    }
    errs = v._validate_measurement_contracts(cert, profile="dev")
    assert any("spectral.measurement_contract_hash" in e for e in errs)


def test_apply_profile_lints_skips_when_lints_not_a_list() -> None:
    cert = {"meta": {"model_profile": {"cert_lints": "bad"}}}
    assert v._apply_profile_lints(cert) == []
