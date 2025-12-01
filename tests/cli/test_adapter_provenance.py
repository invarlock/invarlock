from __future__ import annotations

from unittest.mock import patch

from invarlock.cli.provenance import extract_adapter_provenance


def test_extract_adapter_provenance_known_families():
    with patch("invarlock.cli.provenance.pkg_version", return_value="1.0.0"):
        for name, family in (
            ("hf_gptq", "gptq"),
            ("hf_awq", "awq"),
            ("hf_bnb", "bnb"),
            ("hf_gpt2", "hf"),
        ):
            prov = extract_adapter_provenance(name).to_dict()
            assert prov["family"] == family
            assert prov["version"] == "1.0.0"


def test_extract_adapter_provenance_missing_library_sets_fail_closed():
    def raise_not_found(_name: str):  # noqa: ANN001
        raise Exception("not installed")

    with patch("invarlock.cli.provenance.pkg_version", side_effect=raise_not_found):
        prov = extract_adapter_provenance("hf_gptq").to_dict()
        assert prov["supported"] is False
        assert prov["version"] is None
        assert "not available" in (prov.get("message") or "")
