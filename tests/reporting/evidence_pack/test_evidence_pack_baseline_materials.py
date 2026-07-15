from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.evidence_pack_baselines import (
    baseline_manifest_entries_from_mapping,
    discover_staged_baseline_materials,
    verify_baseline_materials,
)
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from tests.cli._support_verify_runtime_provenance import (
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_pack(tmp_path: Path) -> tuple[Path, Path, Path]:
    pack_dir = tmp_path / "pack"
    report_path = pack_dir / "reports/model/clean/evaluation.report.json"
    baseline_path = pack_dir / "baselines/model/evaluation.report.json"
    report = _strict_provenance_gate_cert()
    baseline = _matching_strict_ppl_baseline(report)
    _write_json(report_path, report)
    _write_json(baseline_path, baseline)
    scenarios_path = pack_dir / "metadata/scenarios.json"
    _write_json(
        scenarios_path,
        {
            "scenarios": [
                {
                    "id": "clean",
                    "strictness": "must_pass",
                    "artifact_class": "evidence_only_pack",
                    "generation": {"kind": "evidence_only"},
                }
            ]
        },
    )
    checksum_lines = [
        f"{_sha256(report_path)}  reports/model/clean/evaluation.report.json",
        f"{_sha256(baseline_path)}  baselines/model/evaluation.report.json",
        f"{_sha256(scenarios_path)}  metadata/scenarios.json",
    ]
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": "evidence-pack-v1",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": _sha256(pack_dir / "checksums.sha256"),
            "verification": {"report_assurance": "strict"},
            "verification_baselines": [
                {
                    "name": "baseline-001",
                    "path": "baselines/model/evaluation.report.json",
                    "digest": f"sha256:{_sha256(baseline_path)}",
                    "report_paths": ["reports/model/clean/evaluation.report.json"],
                }
            ],
        },
    )
    return pack_dir, report_path, baseline_path


def _rewrite_manifest(pack_dir: Path, mutate) -> None:  # noqa: ANN001
    path = pack_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def test_signed_baseline_material_resolves_to_subject_report(tmp_path: Path) -> None:
    pack_dir, report_path, baseline_path = _write_pack(tmp_path)

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert result.errors == ()
    assert result.required is True
    assert result.baseline_by_report == {report_path.resolve(): baseline_path.resolve()}


def test_strict_pack_rejects_missing_baseline_declaration(tmp_path: Path) -> None:
    pack_dir, _report_path, baseline_path = _write_pack(tmp_path)
    baseline_path.unlink()
    _rewrite_manifest(pack_dir, lambda payload: payload.pop("verification_baselines"))

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any(
        "requires signed verification_baselines" in error for error in result.errors
    )


def test_baseline_tree_rejects_undeclared_extra_material(tmp_path: Path) -> None:
    pack_dir, _report_path, _baseline_path = _write_pack(tmp_path)
    _write_json(pack_dir / "baselines/extra/evaluation.report.json", {})

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any("undeclared baseline material" in error for error in result.errors)


def test_baseline_declaration_rejects_path_escape(tmp_path: Path) -> None:
    pack_dir, _report_path, _baseline_path = _write_pack(tmp_path)
    _rewrite_manifest(
        pack_dir,
        lambda payload: payload["verification_baselines"][0].update(
            {"path": "../outside/evaluation.report.json"}
        ),
    )

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any("must be a canonical baselines/" in error for error in result.errors)


def test_baseline_material_rejects_symlink(tmp_path: Path) -> None:
    pack_dir, _report_path, baseline_path = _write_pack(tmp_path)
    external = tmp_path / "external-baseline.json"
    external.write_bytes(baseline_path.read_bytes())
    baseline_path.unlink()
    try:
        baseline_path.symlink_to(external)
    except OSError as exc:  # pragma: no cover - restricted Windows runners
        pytest.skip(f"symlinks unavailable: {exc}")

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any("must not be a symlink" in error for error in result.errors)


def test_baseline_material_requires_exact_checksum_binding(tmp_path: Path) -> None:
    pack_dir, report_path, _baseline_path = _write_pack(tmp_path)
    (pack_dir / "checksums.sha256").write_text(
        f"{_sha256(report_path)}  reports/model/clean/evaluation.report.json\n",
        encoding="utf-8",
    )

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any("exactly one checksums.sha256 entry" in error for error in result.errors)


def test_baseline_material_rejects_malformed_json(tmp_path: Path) -> None:
    pack_dir, _report_path, baseline_path = _write_pack(tmp_path)
    baseline_path.write_text("{", encoding="utf-8")
    digest = _sha256(baseline_path)
    checksums = (pack_dir / "checksums.sha256").read_text(encoding="utf-8")
    lines = [
        f"{digest}  baselines/model/evaluation.report.json"
        if line.endswith("baselines/model/evaluation.report.json")
        else line
        for line in checksums.splitlines()
    ]
    (pack_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")
    _rewrite_manifest(
        pack_dir,
        lambda payload: payload["verification_baselines"][0].update(
            {"digest": f"sha256:{digest}"}
        ),
    )

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any("not valid JSON" in error for error in result.errors)


def test_baseline_material_rejects_subject_mismatch(tmp_path: Path) -> None:
    pack_dir, _report_path, baseline_path = _write_pack(tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["metrics"]["primary_metric"]["final"] = 99.0
    _write_json(baseline_path, baseline)
    digest = _sha256(baseline_path)
    checksums = (pack_dir / "checksums.sha256").read_text(encoding="utf-8")
    lines = [
        f"{digest}  baselines/model/evaluation.report.json"
        if line.endswith("baselines/model/evaluation.report.json")
        else line
        for line in checksums.splitlines()
    ]
    (pack_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")
    _rewrite_manifest(
        pack_dir,
        lambda payload: payload["verification_baselines"][0].update(
            {"digest": f"sha256:{digest}"}
        ),
    )

    result = verify_baseline_materials(pack_dir, report_assurance="strict")

    assert any(
        "does not match subject baseline_ref" in error for error in result.errors
    )


def test_nested_report_verification_receives_discovered_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack_dir, report_path, baseline_path = _write_pack(tmp_path)
    baseline_result = verify_baseline_materials(pack_dir, report_assurance="strict")
    seen: list[tuple[list[Path], Path | None]] = []

    def _run_verify(
        reports: list[Path],
        *,
        baseline: Path | None = None,
        **_kwargs: object,
    ) -> VerifyExecutionResult:
        seen.append((reports, baseline))
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"summary": {"ok": True}, "results": []},
            diagnostics=(),
        )

    monkeypatch.setattr(evidence_pack_mod, "_run_verify_command", _run_verify)

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="strict",
        baseline_by_report=baseline_result.baseline_by_report,
    )

    assert errors == []
    assert seen == [([report_path], baseline_path.resolve())]


def test_staged_multi_model_baselines_map_each_subject_independently(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "staged-pack"
    for model in ("model-a", "model-b"):
        report = _strict_provenance_gate_cert()
        baseline = _matching_strict_ppl_baseline(report)
        _write_json(
            pack_dir / f"reports/{model}/clean/evaluation.report.json",
            report,
        )
        _write_json(
            pack_dir / f"baselines/{model}/evaluation.report.json",
            baseline,
        )

    result = discover_staged_baseline_materials(
        pack_dir,
        report_assurance="strict",
    )
    entries = baseline_manifest_entries_from_mapping(
        pack_dir,
        result.baseline_by_report,
    )

    assert result.errors == ()
    assert len(result.baseline_by_report) == 2
    assert len(entries) == 2
    assert {entry["path"] for entry in entries} == {
        "baselines/model-a/evaluation.report.json",
        "baselines/model-b/evaluation.report.json",
    }
