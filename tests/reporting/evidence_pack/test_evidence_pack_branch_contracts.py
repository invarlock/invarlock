from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import invarlock.evidence_pack_baselines as baselines
import invarlock.evidence_pack_edit_common as edit_common
import invarlock.evidence_pack_edit_validation as edit_validation
import invarlock.evidence_pack_policy as policy
import invarlock.reporting.verify_baseline as verify_baseline
import invarlock.reporting.verify_bootstrap as verify_bootstrap
import invarlock.reporting.verify_strict_schedule as verify_strict_schedule
from tests.reporting.evidence_pack._support_evidence_pack_branch_contracts import (
    _minimal_baseline_pack,
    _sha256,
    _write_json,
    _write_policy_fixture,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "\\bad",
        "/absolute",
        "reports//evaluation.report.json",
        "a/b",
        "other/a/evaluation.report.json",
    ],
)
def test_baseline_path_normalizer_rejects_noncanonical_values(value: object) -> None:
    assert (
        baselines._normalize_relative_path(
            value, root="reports", filename="evaluation.report.json"
        )
        is None
    )


def test_baseline_helpers_cover_missing_and_malformed_inputs(tmp_path: Path) -> None:
    assert baselines._canonical_report_paths(tmp_path) == set()
    (tmp_path / "reports").symlink_to(tmp_path / "elsewhere")
    assert baselines._canonical_report_paths(tmp_path) == set()
    (tmp_path / "reports").unlink()
    malformed = tmp_path / "reports/model/bad/evaluation.report.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    assert (
        baselines._strict_report_paths(
            tmp_path, {"reports/model/bad/evaluation.report.json"}
        )
        == set()
    )

    assert not baselines._manifest_requires_baselines(
        {}, report_assurance="off", strict_report_paths=set()
    )
    assert baselines._manifest_requires_baselines(
        {},
        report_assurance="report",
        strict_report_paths={"reports/m/s/evaluation.report.json"},
    )


def test_baseline_tree_and_metric_helpers_reject_unsafe_shapes(tmp_path: Path) -> None:
    assert baselines._baseline_tree_files_and_symlink_errors(tmp_path) == (set(), [])
    (tmp_path / "baselines").symlink_to(tmp_path / "outside")
    files, errors = baselines._baseline_tree_files_and_symlink_errors(tmp_path)
    assert files == set()
    assert errors == ["baselines/ must not be a symlink."]
    (tmp_path / "baselines").unlink()
    root = tmp_path / "baselines/model"
    root.mkdir(parents=True)
    external = tmp_path / "outside"
    external.write_text("x", encoding="utf-8")
    (root / "link").symlink_to(external)
    files, errors = baselines._baseline_tree_files_and_symlink_errors(tmp_path)
    assert files == set()
    assert errors == [
        "Baseline material tree must not contain symlinks: baselines/model/link"
    ]

    assert baselines._metric_kind_and_final({}) is None
    assert baselines._metric_kind_and_final({"metrics": "bad"}) is None
    for metric in (
        {"kind": "", "final": 1},
        {"kind": "ppl", "final": True},
        {"kind": "ppl", "final": float("inf")},
    ):
        assert baselines._metric_kind_and_final({"primary_metric": metric}) is None
    assert baselines._subject_baseline_metric({"baseline_ref": []}) is None
    assert baselines._metric_kind_and_final(
        {"metrics": {"primary_metric": {"kind": " PPL ", "final": 2}}}
    ) == ("ppl", 2.0)


def test_semantic_baseline_binding_rejects_each_malformed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    report.write_text("{", encoding="utf-8")
    baseline.write_text("{}", encoding="utf-8")
    assert (
        "not valid JSON"
        in baselines._semantic_binding_errors(
            report_path=report,
            baseline_path=baseline,
            relative_report_path="reports/m/s/evaluation.report.json",
        )[0]
    )

    _write_json(report, {})
    baseline.write_text("{", encoding="utf-8")
    assert (
        "not valid JSON"
        in baselines._semantic_binding_errors(
            report_path=report,
            baseline_path=baseline,
            relative_report_path="reports/m/s/evaluation.report.json",
        )[0]
    )

    _write_json(report, [])
    _write_json(baseline, {})
    assert (
        "must be a JSON object"
        in baselines._semantic_binding_errors(
            report_path=report,
            baseline_path=baseline,
            relative_report_path="reports/m/s/evaluation.report.json",
        )[0]
    )
    _write_json(report, {})
    _write_json(baseline, [])
    assert (
        "must be a JSON object"
        in baselines._semantic_binding_errors(
            report_path=report,
            baseline_path=baseline,
            relative_report_path="reports/m/s/evaluation.report.json",
        )[0]
    )

    # Malformed metric identities fail before any producer-controlled report can bind.
    _write_json(report, {"baseline_ref": {}})
    _write_json(baseline, {})
    errors = baselines._semantic_binding_errors(
        report_path=report,
        baseline_path=baseline,
        relative_report_path="reports/m/s/evaluation.report.json",
    )
    assert any("lacks a finite primary metric" in error for error in errors)
    assert any("lacks baseline_ref.primary_metric" in error for error in errors)

    monkeypatch.setattr(
        verify_baseline,
        "append_strict_baseline_contract_errors",
        lambda errors, **kwargs: None,
    )
    monkeypatch.setattr(
        verify_strict_schedule,
        "_append_strict_supplied_baseline_binding_errors",
        lambda errors, **kwargs: None,
    )
    monkeypatch.setattr(
        verify_bootstrap,
        "append_strict_ppl_bootstrap_replay_errors",
        lambda errors, **kwargs: None,
    )
    _write_json(
        report, {"baseline_ref": {"primary_metric": {"kind": "acc", "final": 3}}}
    )
    _write_json(baseline, {"primary_metric": {"kind": "ppl", "final": 2}})
    errors = baselines._semantic_binding_errors(
        report_path=report,
        baseline_path=baseline,
        relative_report_path="reports/m/s/evaluation.report.json",
    )
    assert any("metric kind does not match" in error for error in errors)
    assert any("final value does not match" in error for error in errors)


def test_verify_baseline_manifest_fail_closed_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack, report_rel, baseline_rel = _minimal_baseline_pack(tmp_path)
    monkeypatch.setattr(baselines, "_semantic_binding_errors", lambda **kwargs: [])

    (pack / "manifest.json").write_text("{", encoding="utf-8")
    assert (
        "not valid JSON"
        in baselines.verify_baseline_materials(pack, report_assurance="off").errors[0]
    )
    _write_json(pack / "manifest.json", [])
    assert baselines.verify_baseline_materials(pack, report_assurance="off").errors == (
        "manifest must decode to a JSON object",
    )

    _write_json(
        pack / "manifest.json", {"verification": {"report_assurance": "strict"}}
    )
    result = baselines.verify_baseline_materials(pack, report_assurance="off")
    assert result.required
    assert any("requires signed" in error for error in result.errors)
    assert any("undeclared baseline" in error for error in result.errors)

    for declaration in ([], "bad"):
        _write_json(pack / "manifest.json", {"verification_baselines": declaration})
        assert (
            "must be a non-empty list"
            in baselines.verify_baseline_materials(pack, report_assurance="off").errors[
                -1
            ]
        )

    variants: list[tuple[object, str]] = [
        ("bad", "must be an object"),
        (
            {
                "path": baseline_rel,
                "digest": "sha256:" + "0" * 64,
                "report_paths": [report_rel],
            },
            ".name must be",
        ),
        (
            {
                "name": "x",
                "path": "bad",
                "digest": "sha256:" + "0" * 64,
                "report_paths": [report_rel],
            },
            ".path must be",
        ),
        (
            {
                "name": "x",
                "path": baseline_rel,
                "digest": "bad",
                "report_paths": [report_rel],
            },
            ".digest must be a sha256",
        ),
        (
            {
                "name": "x",
                "path": baseline_rel,
                "digest": "sha256:ABC",
                "report_paths": [report_rel],
            },
            "lowercase sha256",
        ),
        (
            {
                "name": "x",
                "path": baseline_rel,
                "digest": "sha256:" + "0" * 64,
                "report_paths": [],
            },
            "report_paths must be",
        ),
    ]
    for index, (declaration, expected) in enumerate(variants):
        case = tmp_path / f"case-{index}"
        case_pack, _, _ = _minimal_baseline_pack(case)
        _write_json(
            case_pack / "manifest.json", {"verification_baselines": [declaration]}
        )
        assert any(
            expected in error
            for error in baselines.verify_baseline_materials(
                case_pack, report_assurance="off"
            ).errors
        )

    # Duplicate names, baseline paths, report mappings, missing files, checksum mismatch,
    # non-object JSON, and invalid/unavailable report paths are all independently rejected.
    digest = _sha256(pack / baseline_rel)
    declarations = [
        {
            "name": "same",
            "path": baseline_rel,
            "digest": f"sha256:{digest}",
            "report_paths": [
                report_rel,
                "bad",
                "reports/model/missing/evaluation.report.json",
            ],
        },
        {
            "name": "same",
            "path": baseline_rel,
            "digest": f"sha256:{digest}",
            "report_paths": [report_rel],
        },
        {
            "name": "missing",
            "path": "baselines/missing/evaluation.report.json",
            "digest": f"sha256:{digest}",
            "report_paths": [report_rel],
        },
    ]
    _write_json(pack / "manifest.json", {"verification_baselines": declarations})
    errors = baselines.verify_baseline_materials(pack, report_assurance="strict").errors
    for fragment in (
        "duplicated",
        "declared more than once",
        "is missing",
        "not a canonical report path",
        "not present as a canonical report",
    ):
        assert any(fragment in error for error in errors), (fragment, errors)

    # A second distinct baseline can reach the duplicate report binding check.
    second_rel = "baselines/model2/evaluation.report.json"
    _write_json(pack / second_rel, {})
    second_digest = _sha256(pack / second_rel)
    (pack / "checksums.sha256").write_text(
        f"{digest}  {baseline_rel}\n{second_digest}  {second_rel}\n", encoding="utf-8"
    )
    _write_json(
        pack / "manifest.json",
        {
            "verification_baselines": [
                {
                    "name": "one",
                    "path": baseline_rel,
                    "digest": f"sha256:{digest}",
                    "report_paths": [report_rel],
                },
                {
                    "name": "two",
                    "path": second_rel,
                    "digest": f"sha256:{second_digest}",
                    "report_paths": [report_rel],
                },
            ]
        },
    )
    errors = baselines.verify_baseline_materials(pack, report_assurance="strict").errors
    assert any("more than one verification baseline" in error for error in errors)


def test_baseline_discovery_and_build_preflight_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack, report_rel, baseline_rel = _minimal_baseline_pack(tmp_path)
    monkeypatch.setattr(
        baselines, "_semantic_binding_errors", lambda **kwargs: ["mismatch"]
    )
    result = baselines.discover_staged_baseline_materials(
        pack, report_assurance="strict"
    )
    assert any("no matching staged raw baseline" in error for error in result.errors)

    # Two semantically matching but byte-distinct baselines are ambiguous.
    other = pack / "baselines/model/other/evaluation.report.json"
    _write_json(other, {"different": True})
    monkeypatch.setattr(baselines, "_semantic_binding_errors", lambda **kwargs: [])
    result = baselines.discover_staged_baseline_materials(
        pack, report_assurance="strict"
    )
    assert any("multiple distinct staged baselines" in error for error in result.errors)

    report = pack / report_rel
    assert baselines.verify_build_baseline(
        baseline_path=report, report_paths=[report]
    ) == [
        "Verification baseline must be a file distinct from every subject report (report input 1)."
    ]
    (pack / baseline_rel).write_text("{", encoding="utf-8")
    report.write_text("{", encoding="utf-8")
    assert (
        baselines.verify_build_baseline(
            baseline_path=pack / baseline_rel, report_paths=[report]
        )
        == []
    )


def test_baseline_nonobject_material_and_optional_discovery_are_rejected_or_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack, report_rel, baseline_rel = _minimal_baseline_pack(tmp_path)
    _write_json(pack / baseline_rel, [])
    digest = _sha256(pack / baseline_rel)
    (pack / "checksums.sha256").write_text(
        f"{digest}  {baseline_rel}\n", encoding="utf-8"
    )
    _write_json(
        pack / "manifest.json",
        {
            "verification_baselines": [
                {
                    "name": "baseline-1",
                    "path": baseline_rel,
                    "digest": f"sha256:{digest}",
                    "report_paths": [report_rel],
                }
            ]
        },
    )
    monkeypatch.setattr(baselines, "_semantic_binding_errors", lambda **kwargs: [])
    errors = baselines.verify_baseline_materials(pack, report_assurance="off").errors
    assert any("must decode to a JSON object" in error for error in errors)

    monkeypatch.setattr(
        baselines, "_semantic_binding_errors", lambda **kwargs: ["no match"]
    )
    result = baselines.discover_staged_baseline_materials(pack, report_assurance="off")
    assert result.errors == ()


@pytest.mark.parametrize(
    "metadata,fragments",
    [
        ({"edit_type": "synthetic_lowrank_delta"}, ["requires edit_provenance"]),
        (
            {"edit_type": "synthetic_lowrank_delta", "edit_provenance": {}},
            [
                "synthetic=true",
                "trained_adapter=false",
                "adapter_merge_performed=false",
            ],
        ),
        (
            {"edit_type": "synthetic_dense_update", "edit_provenance": {}},
            [
                "synthetic=true",
                "optimization_performed=false",
                "training_data_used=false",
            ],
        ),
    ],
)
def test_synthetic_edit_metadata_cannot_masquerade_as_training(
    metadata: dict[str, Any], fragments: list[str]
) -> None:
    payload = {
        "schema": edit_common.EDIT_METADATA_SCHEMA,
        "artifact_class": edit_common.VALIDATION_SUBJECT_CHECKPOINT,
        "optimized_deployment_backend": False,
        "packed_quantized_storage": False,
        **metadata,
    }
    errors = edit_validation._metadata_consistency_errors(
        scenario_id="synthetic",
        spec={"artifact_class": edit_common.VALIDATION_SUBJECT_CHECKPOINT},
        metadata=payload,
    )
    for fragment in fragments:
        assert any(fragment in error for error in errors)


@pytest.mark.parametrize(
    "edit_type,provenance",
    [
        (
            "synthetic_lowrank_delta",
            {
                "synthetic": True,
                "trained_adapter": False,
                "adapter_merge_performed": False,
            },
        ),
        (
            "synthetic_dense_update",
            {
                "synthetic": True,
                "optimization_performed": False,
                "training_data_used": False,
            },
        ),
    ],
)
def test_synthetic_edit_metadata_accepts_only_explicit_fixture_provenance(
    edit_type: str, provenance: dict[str, bool]
) -> None:
    assert (
        edit_validation._metadata_consistency_errors(
            scenario_id="synthetic",
            spec={"artifact_class": edit_common.VALIDATION_SUBJECT_CHECKPOINT},
            metadata={
                "schema": edit_common.EDIT_METADATA_SCHEMA,
                "artifact_class": edit_common.VALIDATION_SUBJECT_CHECKPOINT,
                "edit_type": edit_type,
                "optimized_deployment_backend": False,
                "packed_quantized_storage": False,
                "coverage": {
                    "edited_tensors": 1,
                    "edited_params": 4,
                    "total_params": 8,
                    "coverage_ratio": 0.5,
                },
                "edit_provenance": provenance,
            },
        )
        == []
    )


def test_policy_material_fail_closed_matrix(tmp_path: Path) -> None:
    missing, errors = policy.load_valid_policy_pack(
        tmp_path / "missing", label="Policy"
    )
    assert missing is None and errors == [f"Policy not found: {tmp_path / 'missing'}"]
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert (
        "not valid JSON/YAML"
        in policy.load_valid_policy_pack(bad, label="Policy")[1][0]
    )
    _write_json(bad, {})
    assert policy.load_valid_policy_pack(bad, label="Policy")[0] is None

    pack, acceptance_policy = _write_policy_fixture(tmp_path)
    (pack / "manifest.json").write_text("{", encoding="utf-8")
    assert (
        "not valid JSON"
        in policy.verify_policy_material(
            pack, report_assurance="off", acceptance_policy_path=None
        ).errors[0]
    )
    _write_json(pack / "manifest.json", [])
    assert policy.verify_policy_material(
        pack, report_assurance="off", acceptance_policy_path=None
    ).errors == ("manifest must decode to a JSON object",)

    sealed = pack / policy.POLICY_RELATIVE_PATH
    base_entry = policy.policy_manifest_entry(sealed)
    cases: list[tuple[object, str]] = [
        (None, "requires signed"),
        ([], "must be an object"),
        ({**base_entry, "path": "policy/other.json"}, ".path must be"),
    ]
    for index, (declaration, fragment) in enumerate(cases):
        _write_json(
            pack / "manifest.json",
            {
                "verification": {"report_assurance": "strict"},
                policy.POLICY_MANIFEST_FIELD: declaration,
            },
        )
        errors = policy.verify_policy_material(
            pack, report_assurance="strict", acceptance_policy_path=None
        ).errors
        assert any(fragment in error for error in errors), (index, errors)

    _write_json(
        pack / "manifest.json",
        {
            "verification": {"report_assurance": "strict"},
            policy.POLICY_MANIFEST_FIELD: base_entry,
        },
    )
    sealed.unlink()
    errors = policy.verify_policy_material(
        pack, report_assurance="strict", acceptance_policy_path=acceptance_policy
    ).errors
    assert any("path is missing" in error for error in errors)

    pack, acceptance_policy = _write_policy_fixture(tmp_path / "fresh")
    sealed = pack / policy.POLICY_RELATIVE_PATH
    entry = policy.policy_manifest_entry(sealed)
    entry["policy_digest"] = "sha256:" + "0" * 64
    (pack / "checksums.sha256").write_text(
        f"{'0' * 64}  {policy.POLICY_RELATIVE_PATH}\n{'1' * 64}  {policy.POLICY_RELATIVE_PATH}\n",
        encoding="utf-8",
    )
    _write_json(pack / "manifest.json", {policy.POLICY_MANIFEST_FIELD: entry})
    errors = policy.verify_policy_material(
        pack, report_assurance="off", acceptance_policy_path=acceptance_policy
    ).errors
    assert any("exactly one" in error for error in errors)
    assert any("policy_digest does not match" in error for error in errors)


def test_policy_tree_symlinks_and_optional_acceptance_path(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "policy").symlink_to(tmp_path / "outside")
    assert policy._policy_tree_files_and_symlink_errors(pack) == (
        set(),
        ["policy/ must not be a symlink."],
    )
    (pack / "policy").unlink()
    (pack / "policy").mkdir()
    (pack / "policy/nested").mkdir()
    outside = tmp_path / "outside"
    outside.write_text("{}", encoding="utf-8")
    (pack / "policy/link.json").symlink_to(outside)
    files, errors = policy._policy_tree_files_and_symlink_errors(pack)
    assert files == set()
    assert errors == [
        "Policy material tree must not contain symlinks: policy/link.json"
    ]

    pack, _reviewer = _write_policy_fixture(tmp_path / "optional")
    # Independent anchoring is optional only when neither invocation nor manifest is strict.
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    manifest["verification"]["report_assurance"] = "off"
    _write_json(pack / "manifest.json", manifest)
    result = policy.verify_policy_material(
        pack, report_assurance="off", acceptance_policy_path=None
    )
    assert not result.required
    assert not any("independently supplied" in error for error in result.errors)


def test_policy_declaration_rejects_symlinked_material(tmp_path: Path) -> None:
    pack, acceptance_policy = _write_policy_fixture(tmp_path)
    sealed = pack / policy.POLICY_RELATIVE_PATH
    external = tmp_path / "sealed-copy.json"
    external.write_bytes(sealed.read_bytes())
    sealed.unlink()
    sealed.symlink_to(external)
    errors = policy.verify_policy_material(
        pack, report_assurance="strict", acceptance_policy_path=acceptance_policy
    ).errors
    assert any("must not be a symlink" in error for error in errors)
