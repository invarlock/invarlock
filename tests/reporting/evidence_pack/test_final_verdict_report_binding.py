from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_binding as binding_mod
from invarlock.evidence_pack import (
    EvidencePackStatus,
    verify_final_verdict_report_binding,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(pack_dir: Path, relative_path: str, run_id: str) -> Path:
    report_path = pack_dir / relative_path
    _write_json(report_path, {"run_id": run_id, "ok": True})
    return report_path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_report_top_level_binding_matches(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    relative_path = "reports/report-001/evaluation.report.json"
    report = _write_report(pack_dir, relative_path, "run-1")
    _write_json(
        pack_dir / "results/final_verdict.json",
        {
            "verdict": "PASS",
            "report_path": relative_path,
            "report_sha256": f"sha256:{_digest(report)}",
            "run_id": "run-1",
        },
    )
    assert verify_final_verdict_report_binding(pack_dir, require_binding=True) == []


def test_binding_rejects_source_mutation_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack_dir = tmp_path / "pack"
    relative_path = "reports/report-001/evaluation.report.json"
    report = _write_report(pack_dir, relative_path, "run-1")
    _write_json(
        pack_dir / "results/final_verdict.json",
        {
            "verdict": "PASS",
            "report_path": relative_path,
            "report_sha256": _digest(report),
            "run_id": "run-1",
        },
    )
    original = binding_mod._verify_final_verdict_report_binding_snapshot

    def mutate_after_capture(
        snapshot_root: Path, *, require_binding: bool
    ) -> list[str]:
        report.write_text('{"run_id":"substituted"}\n', encoding="utf-8")
        return original(snapshot_root, require_binding=require_binding)

    monkeypatch.setattr(
        binding_mod,
        "_verify_final_verdict_report_binding_snapshot",
        mutate_after_capture,
    )

    errors = binding_mod.verify_final_verdict_report_binding(
        pack_dir,
        require_binding=True,
    )

    assert any("changed after capture" in error for error in errors)


def test_binding_surfaces_ambiguous_manifest_declaration(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    relative_path = "reports/report-001/evaluation.report.json"
    report = _write_report(pack_dir, relative_path, "run-1")
    _write_json(
        pack_dir / "results/final_verdict.json",
        {
            "verdict": "PASS",
            "report_path": relative_path,
            "report_sha256": _digest(report),
            "run_id": "run-1",
        },
    )
    (pack_dir / "manifest.json").write_text(
        '{"verification":{},"verification":{"report_assurance":"strict"}}\n',
        encoding="utf-8",
    )

    errors = verify_final_verdict_report_binding(pack_dir)

    assert len(errors) == 1
    assert "duplicate key" in errors[0]


def test_single_report_stale_hash_fails_closed(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    _write_report(
        pack_dir,
        "reports/report-001/evaluation.report.json",
        "run-1",
    )
    _write_json(
        pack_dir / "results/final_verdict.json",
        {"verdict": "PASS", "report_sha256": "0" * 64, "run_id": "run-1"},
    )

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert any("report_sha256 does not match" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("run_id", "other-run", "run_id does not match"),
        (
            "report_path",
            "reports/report-002/evaluation.report.json",
            "not present in the pack",
        ),
        ("report_path", "../../outside.json", "invalid report path"),
    ],
)
def test_single_report_identifier_and_path_claims_must_match(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    pack_dir = tmp_path / "pack"
    report = _write_report(
        pack_dir,
        "reports/report-001/evaluation.report.json",
        "run-1",
    )
    verdict = {"verdict": "PASS", "report_sha256": _digest(report), field: value}
    _write_json(pack_dir / "results/final_verdict.json", verdict)

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert any(expected in error for error in errors)


def test_multiple_reports_require_explicit_exact_bindings(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    first = _write_report(
        pack_dir,
        "reports/model/clean/noop/run_1/evaluation.report.json",
        "run-1",
    )
    second = _write_report(
        pack_dir,
        "reports/model/stress/prune/run_1/evaluation.report.json",
        "run-2",
    )
    verdict_path = pack_dir / "results/verdicts/final_verdict.json"
    _write_json(verdict_path, {"verdict": "PASS"})

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)
    assert any("requires exact report_bindings coverage" in error for error in errors)

    _write_json(
        verdict_path,
        {
            "verdict": "PASS",
            "report_bindings": [
                {
                    "path": first.relative_to(pack_dir).as_posix(),
                    "report_sha256": _digest(first),
                    "run_id": "run-1",
                },
                {
                    "path": second.relative_to(pack_dir).as_posix(),
                    "report_sha256": _digest(second),
                    "run_id": "run-2",
                },
            ],
            "records": [
                {
                    "path": "model/reports/clean/noop/run_1/evaluation.report.json",
                    "report_sha256": _digest(first),
                    "run_id": "run-1",
                }
            ],
        },
    )

    assert verify_final_verdict_report_binding(pack_dir, require_binding=True) == []


@pytest.mark.parametrize("failure", ["duplicate", "missing", "extra"])
def test_multiple_report_bindings_reject_non_exact_coverage(
    tmp_path: Path, failure: str
) -> None:
    pack_dir = tmp_path / "pack"
    first = _write_report(
        pack_dir,
        "reports/model/clean/a/evaluation.report.json",
        "run-1",
    )
    second = _write_report(
        pack_dir,
        "reports/model/clean/b/evaluation.report.json",
        "run-2",
    )
    bindings = [
        {
            "path": first.relative_to(pack_dir).as_posix(),
            "report_sha256": _digest(first),
        },
        {
            "path": second.relative_to(pack_dir).as_posix(),
            "report_sha256": _digest(second),
        },
    ]
    if failure == "duplicate":
        bindings[1] = dict(bindings[0])
    elif failure == "missing":
        bindings.pop()
    else:
        bindings.append(
            {
                "path": "reports/model/clean/other/evaluation.report.json",
                "report_sha256": "f" * 64,
            }
        )
    _write_json(
        pack_dir / "results/verdicts/final_verdict.json",
        {"verdict": "PASS", "report_bindings": bindings},
    )

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert errors
    assert any(
        marker in "\n".join(errors)
        for marker in ("duplicate path", "does not cover", "not present in the pack")
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("report_sha256", "0" * 64, "report_sha256 does not match"),
        ("run_id", "wrong", "run_id does not match"),
        (
            "path",
            "model/reports/clean/other/evaluation.report.json",
            "not present in the pack",
        ),
    ],
)
def test_verdict_records_are_bound_to_their_reports(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    pack_dir = tmp_path / "pack"
    report = _write_report(
        pack_dir,
        "reports/model/clean/noop/evaluation.report.json",
        "run-1",
    )
    binding = {
        "path": report.relative_to(pack_dir).as_posix(),
        "report_sha256": _digest(report),
        "run_id": "run-1",
    }
    record = {
        "path": "model/reports/clean/noop/evaluation.report.json",
        "report_sha256": _digest(report),
        "run_id": "run-1",
    }
    record[field] = value
    _write_json(
        pack_dir / "results/verdicts/final_verdict.json",
        {"verdict": "PASS", "report_bindings": [binding], "records": [record]},
    )

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert any(expected in error for error in errors)


def test_manifest_strict_assurance_requires_binding_without_cli_flag(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    _write_report(
        pack_dir,
        "reports/report-001/evaluation.report.json",
        "run-1",
    )
    _write_json(pack_dir / "results/final_verdict.json", {"verdict": "PASS"})
    _write_json(
        pack_dir / "manifest.json",
        {"verification": {"report_assurance": "strict"}},
    )

    errors = verify_final_verdict_report_binding(pack_dir)

    assert any("requires report_sha256" in error for error in errors)


@pytest.mark.parametrize("target", ["report", "verdict"])
def test_binding_rejects_file_symlinks_before_reading(
    tmp_path: Path, target: str
) -> None:
    pack_dir = tmp_path / "pack"
    external_dir = tmp_path / "external"
    external_report = external_dir / "evaluation.report.json"
    external_verdict = external_dir / "final_verdict.json"
    _write_json(external_report, {"run_id": "outside", "ok": True})
    _write_json(
        external_verdict,
        {"verdict": "PASS", "report_sha256": _digest(external_report)},
    )
    report_path = pack_dir / "reports/report-001/evaluation.report.json"
    verdict_path = pack_dir / "results/final_verdict.json"
    if target == "report":
        report_path.parent.mkdir(parents=True)
        report_path.symlink_to(external_report)
        _write_json(verdict_path, {"verdict": "PASS", "report_sha256": "0" * 64})
    else:
        _write_report(
            pack_dir,
            "reports/report-001/evaluation.report.json",
            "run-1",
        )
        verdict_path.parent.mkdir(parents=True)
        verdict_path.symlink_to(external_verdict)

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert any("must not contain symlinks" in error for error in errors)


@pytest.mark.parametrize("tree", ["reports", "results"])
def test_binding_rejects_symlinked_search_tree_ancestors(
    tmp_path: Path, tree: str
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    external_dir = tmp_path / f"external-{tree}"
    if tree == "reports":
        external_report = external_dir / "evaluation.report.json"
        _write_json(external_report, {"run_id": "outside"})
        (pack_dir / "reports").mkdir()
        (pack_dir / "reports/report-001").symlink_to(
            external_dir,
            target_is_directory=True,
        )
        _write_json(
            pack_dir / "results/final_verdict.json",
            {"verdict": "PASS", "report_sha256": "0" * 64},
        )
    else:
        _write_json(external_dir / "final_verdict.json", {"verdict": "PASS"})
        (pack_dir / "results").mkdir()
        (pack_dir / "results/verdicts").symlink_to(
            external_dir,
            target_is_directory=True,
        )
        _write_report(
            pack_dir,
            "reports/report-001/evaluation.report.json",
            "run-1",
        )

    errors = verify_final_verdict_report_binding(pack_dir, require_binding=True)

    assert any("search tree must not contain symlinks" in error for error in errors)


def test_core_verifier_rejects_report_symlink_before_checksum_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    external_report = tmp_path / "outside/evaluation.report.json"
    _write_json(external_report, {"run_id": "outside", "ok": True})
    report_path = pack_dir / "reports/report-001/evaluation.report.json"
    report_path.parent.mkdir(parents=True)
    report_path.symlink_to(external_report)
    _write_json(
        pack_dir / "results/final_verdict.json",
        {"verdict": "PASS", "report_sha256": _digest(external_report)},
    )
    checksums = pack_dir / "checksums.sha256"
    checksums.write_text("", encoding="utf-8")
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": "evidence-pack-v1",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": _digest(checksums),
        },
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "_verify_signature",
        lambda pack_dir, *, strict: ([], [], "sha256:" + "a" * 64),
    )

    def _unexpected_checksum_read(_pack_dir: Path):
        raise AssertionError("checksum verification must not read a report symlink")

    monkeypatch.setattr(
        evidence_pack_mod,
        "_verify_checksums",
        _unexpected_checksum_read,
    )

    result = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        strict=True,
        skip_verify=False,
        report_assurance="strict",
    )

    assert result.status == EvidencePackStatus.INTEGRITY
    assert result.payload["report_assurance"] == "strict"
    assert any(
        "must not contain symlinks" in error for error in result.payload["errors"]
    )
