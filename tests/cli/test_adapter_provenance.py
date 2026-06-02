from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from invarlock.core.backend_inventory import extract_adapter_provenance


def test_extract_adapter_provenance_known_families():
    with patch("invarlock.core.backend_inventory.pkg_version", return_value="1.0.0"):
        for name, family in (
            ("hf_gptq", "gptq"),
            ("hf_awq", "awq"),
            ("hf_bnb", "bnb"),
            ("hf_torchao", "torchao_int8"),
            ("hf_hqq", "hqq"),
            ("hf_quanto", "quanto"),
            ("hf_ct", "compressed_tensors"),
            ("hf_causal", "hf"),
        ):
            prov = extract_adapter_provenance(name).to_dict()
            assert prov["family"] == family
            assert prov["version"] == "1.0.0"


def test_extract_adapter_provenance_missing_library_sets_fail_closed():
    def raise_not_found(_name: str):  # noqa: ANN001
        raise PackageNotFoundError("not installed")

    with patch(
        "invarlock.core.backend_inventory.pkg_version", side_effect=raise_not_found
    ):
        prov = extract_adapter_provenance("hf_gptq").to_dict()
        assert prov["supported"] is False
        assert prov["version"] is None
        assert "not available" in (prov.get("message") or "")


def test_extract_adapter_provenance_metadata_runtime_failure_sets_fail_closed():
    def raise_runtime(_name: str):  # noqa: ANN001
        raise RuntimeError("metadata backend failed")

    with patch(
        "invarlock.core.backend_inventory.pkg_version", side_effect=raise_runtime
    ):
        prov = extract_adapter_provenance("hf_gptq").to_dict()
        assert prov["supported"] is False
        assert prov["version"] is None
        assert "not available" in (prov.get("message") or "")
