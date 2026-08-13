from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/evaluator-qualification/measure_signed_transactions.py"
TRANSACTIONS = ROOT / "examples/evaluator-qualification/signed-transactions"
README = (TRANSACTIONS / "README.md").read_text(encoding="utf-8")


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("signed_transaction_costs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measurements_replay_both_real_signed_transactions(tmp_path: Path) -> None:
    module = _module()

    result = module.measure_all(root=ROOT, runs=2)

    assert result["format"] == module.FORMAT
    assert result["runs"] == 2
    transactions = result["transactions"]
    assert [item["profile_id"] for item in transactions] == list(module.PROFILE_IDS)
    for item in transactions:
        assert item["record_count"] == 400
        assert item["evidence_files"] > 20
        assert item["evidence_bytes"] > 100_000
        assert item["package_bytes"] > item["evidence_bytes"]
        assert item["verification_and_receipt_median_ms"] > 0
        assert item["report_render_median_ms"] > 0


def test_measurement_sizes_match_the_retained_directories() -> None:
    module = _module()
    transaction = TRANSACTIONS / "inspect-ai"
    files, total = module._tree_stats(transaction)

    expected = [path for path in transaction.rglob("*") if path.is_file()]
    assert files == len(expected)
    assert total == sum(path.stat().st_size for path in expected)


def test_public_cost_table_records_exact_sizes_and_claim_boundary() -> None:
    module = _module()
    for profile_id in module.PROFILE_IDS:
        _files, size = module._tree_stats(TRANSACTIONS / profile_id / "evidence")
        assert f"{size:,}" in README
    assert "complete semantic evidence replay" in README
    assert "not a performance guarantee" in README


def test_measurement_rejects_unsafe_or_invalid_inputs(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.MeasurementError, match="between 1 and 100"):
        module.measure_all(root=ROOT, runs=0)
    with pytest.raises(module.MeasurementError, match="real directory"):
        module._tree_stats(tmp_path / "missing")

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "linked").symlink_to(tmp_path / "missing")
    with pytest.raises(module.MeasurementError, match="symbolic links"):
        module._tree_stats(tree)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(module.MeasurementError, match="JSON object"):
        module._object(invalid, label="fixture")
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(module.MeasurementError, match="readable JSON"):
        module._object(invalid, label="fixture")


def test_measurement_rejects_invalid_record_and_transaction_metadata(
    tmp_path: Path,
) -> None:
    module = _module()
    evidence = tmp_path / "evidence/reports"
    evidence.mkdir(parents=True)
    evidence.joinpath("evaluation.report.json").write_text(
        '{"record_count":true}', encoding="utf-8"
    )
    with pytest.raises(module.MeasurementError, match="record count"):
        module._record_count(evidence.parent)

    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction.joinpath("transaction.json").write_text(
        '{"profile_id":"unknown"}', encoding="utf-8"
    )
    with pytest.raises(module.MeasurementError, match="retained flagship"):
        module.measure_transaction(transaction, runs=1, temporary_root=tmp_path)

    with pytest.raises(module.MeasurementError, match="verification anchors"):
        module._verification_anchors({"verification": []})


def test_measurement_rejects_a_retained_receipt_outside_its_trust_roots() -> None:
    module = _module()
    transaction_root = TRANSACTIONS / "inspect-ai"
    transaction = json.loads(transaction_root.joinpath("transaction.json").read_bytes())
    verification = dict(transaction["verification"])
    verification["verifier_fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(module.MeasurementError, match="retained receipt verification"):
        module._verify_retained_receipt(transaction_root, verification)


def test_measurement_runs_one_untimed_warmup_before_every_sample() -> None:
    module = _module()
    observed: list[int] = []

    timings = module._timings(observed.append, runs=3)

    assert observed == [-1, 0, 1, 2]
    assert len(timings) == 3
    assert all(value >= 0 for value in timings)


def test_measurement_main_prints_json_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "measure_all",
        lambda **_kwargs: {
            "environment": {},
            "format": module.FORMAT,
            "runs": 1,
            "transactions": [],
        },
    )
    assert module.main(["--runs", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == module.FORMAT

    def rejected(**_kwargs: object) -> None:
        raise module.MeasurementError("rejected")

    monkeypatch.setattr(module, "measure_all", rejected)
    assert module.main([]) == 2
    assert "FAIL rejected" in capsys.readouterr().err
