from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/checks/check_model_classification.py")


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_model_classification", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_model_classification"] = module
    spec.loader.exec_module(module)
    return module


def _write_contracts(
    tmp_path: Path, *, classification: dict, support: dict, catalog: dict
) -> tuple[Path, Path, Path]:
    classification_path = tmp_path / "model_classification.json"
    support_path = tmp_path / "support_matrix.json"
    catalog_path = tmp_path / "model_family_catalog.json"
    classification_path.write_text(json.dumps(classification), encoding="utf-8")
    support_path.write_text(json.dumps(support), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return classification_path, support_path, catalog_path


def _current_contracts() -> tuple[dict, dict, dict]:
    classification = json.loads(
        Path("contracts/model_classification.json").read_text(encoding="utf-8")
    )
    support = json.loads(
        Path("contracts/support_matrix.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        Path("contracts/model_family_catalog.json").read_text(encoding="utf-8")
    )
    return classification, support, catalog


def _patch_contract_paths(
    monkeypatch,
    mod: ModuleType,
    *,
    classification_path: Path,
    support_path: Path,
    catalog_path: Path,
) -> None:
    monkeypatch.setattr(mod, "MODEL_CLASSIFICATION_PATH", classification_path)
    monkeypatch.setattr(mod, "SUPPORT_MATRIX_PATH", support_path)
    monkeypatch.setattr(mod, "MODEL_FAMILY_CATALOG_PATH", catalog_path)


def test_model_classification_accepts_current_contracts() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "invarlock/model-classification-audit-v1"
    assert payload["ok"] is True
    assert payload["finding_count"] == 0


def test_model_classification_catches_blocked_checkpoint_reintroduction(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_script_module()
    classification, support, catalog = _current_contracts()
    support["lanes"][0]["representative_models"] = ["facebook/opt-1.3b"]
    classification_path, support_path, catalog_path = _write_contracts(
        tmp_path,
        classification=classification,
        support=support,
        catalog=catalog,
    )
    _patch_contract_paths(
        monkeypatch,
        mod,
        classification_path=classification_path,
        support_path=support_path,
        catalog_path=catalog_path,
    )

    findings = mod.audit()

    assert any(
        finding.scope == "blocked_named_checkpoint:facebook/opt-1.3b"
        and "support_matrix:gpt2-causal-hf" in finding.message
        for finding in findings
    )


def test_model_classification_catches_promotion_candidate_decision_drift(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_script_module()
    classification, support, catalog = _current_contracts()
    for entry in classification["entries"]:
        if entry.get("candidate_id") == "broader-bert-like-mlms":
            entry["classification"] = "backlog"
            break
    classification_path, support_path, catalog_path = _write_contracts(
        tmp_path,
        classification=classification,
        support=support,
        catalog=catalog,
    )
    _patch_contract_paths(
        monkeypatch,
        mod,
        classification_path=classification_path,
        support_path=support_path,
        catalog_path=catalog_path,
    )

    findings = mod.audit()

    assert any(
        finding.scope == "promotion_candidate:broader-bert-like-mlms"
        and "requires classification 'blocked'" in finding.message
        for finding in findings
    )


def test_model_classification_catches_suite_role_drift(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_script_module()
    classification, support, catalog = _current_contracts()
    for entry in classification["entries"]:
        if entry.get("id") == "mixtral-8x7b-moe-causal-hf":
            entry["suite_roles"] = ["support-matrix-backlog-gpu"]
            break
    classification_path, support_path, catalog_path = _write_contracts(
        tmp_path,
        classification=classification,
        support=support,
        catalog=catalog,
    )
    _patch_contract_paths(
        monkeypatch,
        mod,
        classification_path=classification_path,
        support_path=support_path,
        catalog_path=catalog_path,
    )

    findings = mod.audit()

    assert any(
        finding.scope == "model_evidence:repo-mentioned-gpu:mixtral_8x7b_public"
        and "does not list this suite role" in finding.message
        for finding in findings
    )
