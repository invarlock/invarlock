from __future__ import annotations

import copy
import json
from pathlib import Path

from invarlock.cli.commands import verify as verify_mod


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_warn_adapter_family_mismatch_covers_baseline_path_branches(
    tmp_path: Path,
) -> None:
    cert_path = tmp_path / "cert.json"
    cert_path.write_text("{}", encoding="utf-8")

    base_cert = {
        "plugins": {"adapter": {"provenance": {"family": "hf"}}},
        "provenance": {"baseline": {"report_path": ""}},
    }

    # provenance not a dict: baseline_report_path should remain None
    payload = copy.deepcopy(base_cert)
    payload["provenance"] = "not-a-dict"
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline path points to a missing file: p.exists() is False
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(tmp_path / "missing.json")
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline file exists but meta.plugins is not a dict
    baseline_path = _write_json(
        tmp_path / "base_plugins_not_dict.json", {"meta": {"plugins": []}}
    )
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(baseline_path)
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline file exists, adapter is not a dict
    baseline_path = _write_json(
        tmp_path / "base_adapter_not_dict.json",
        {"meta": {"plugins": {"adapter": "nope"}}},
    )
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(baseline_path)
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline file exists, provenance is not a dict
    baseline_path = _write_json(
        tmp_path / "base_prov_not_dict.json",
        {"meta": {"plugins": {"adapter": {"provenance": "nope"}}}},
    )
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(baseline_path)
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline file exists, family is missing/empty (val check branch)
    baseline_path = _write_json(
        tmp_path / "base_family_empty.json",
        {"meta": {"plugins": {"adapter": {"provenance": {"family": ""}}}}},
    )
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(baseline_path)
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)

    # baseline file exists, family present (no mismatch warning when equal)
    baseline_path = _write_json(
        tmp_path / "base_family_ok.json",
        {"meta": {"plugins": {"adapter": {"provenance": {"family": "hf"}}}}},
    )
    payload = copy.deepcopy(base_cert)
    payload["provenance"]["baseline"]["report_path"] = str(baseline_path)
    verify_mod._warn_adapter_family_mismatch(cert_path, payload)
