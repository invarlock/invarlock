from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from invarlock.core.assurance_contract import CANONICAL_GUARD_CHAIN
from invarlock.reporting.verify_contract_types import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
)
from scripts.checks.check_public_evidence import check_public_evidence
from scripts.model_evidence import negative_fixture_generation as generation_module
from scripts.model_evidence.negative_fixture_generation import (
    ALL_SCENARIOS,
    EXPECTED_FAILURE_TEXT,
    GENUINE_SCENARIOS,
    NegativeFixtureError,
    generate,
)
from tests.cli._support_verify_runtime_provenance import (
    _bind_strict_baseline,
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    bind_runtime_policy_receipt,
)
from tests.reporting.validation._support_verify_assurance_guard_chain import (
    _report as _strict_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path, *, prefix: bool = True) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}" if prefix else digest


def _base_report() -> dict[str, object]:
    return json.loads(
        (
            REPO_ROOT / "tests/artifacts/golden_runs/gpt2/evaluation.report.json"
        ).read_text()
    )


def _scenario_report(scenario: str) -> dict[str, object]:
    report = copy.deepcopy(_base_report())
    validation = report["validation"]
    assert isinstance(validation, dict)
    if scenario == "spectral_guard_failure":
        validation["spectral_stable"] = False
        report["spectral"] = {
            "caps_applied": 6,
            "max_caps": 5,
            "caps_exceeded": True,
            "violations": [{"module": "transformer.h.0.mlp.c_fc"}],
        }
    elif scenario == "rmt_guard_failure":
        validation["rmt_stable"] = False
        report["rmt"] = {
            "stable": False,
            "epsilon_violations": [{"family": "ffn", "growth": 0.02}],
        }
    elif scenario == "variance_guard_failure":
        report["variance"] = {
            "enabled": True,
            "passed": False,
            "predictive_gate": {
                "evaluated": True,
                "passed": False,
                "reason": "minimum_effect_not_met",
            },
        }
    elif scenario == "invariants_failure":
        validation["invariants_pass"] = False
        report["invariants"] = {
            "status": "fail",
            "failures": [{"check": "weight_finiteness", "module": "lm_head"}],
        }
    elif scenario == "primary_metric_failure":
        validation["primary_metric_acceptable"] = False
        primary = report["primary_metric"]
        assert isinstance(primary, dict)
        primary["ratio_vs_baseline"] = 2.0
    elif scenario == "runtime_provenance_failure":
        report["invariants"] = {"status": "pass", "failures": []}
        report["rmt"] = {"stable": True}
        report["spectral"] = {"caps_applied": 0, "max_caps": 5}
    else:  # pragma: no cover
        raise AssertionError(scenario)
    return report


def _write_source(root: Path, scenario: str) -> dict[str, str]:
    source = root / scenario
    report_path = source / "evaluation.report.json"
    manifest_path = source / "runtime.manifest.json"
    baseline_path = source / "baseline.report.json"
    policy_path = source / "acceptance_policy_pack.json"
    _write_json(report_path, _scenario_report(scenario))
    _write_json(baseline_path, _base_report())
    _write_json(policy_path, {})
    _write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "execution_mode": "container",
            "report": {
                "filename": report_path.name,
                "path": str(report_path),
                "sha256": _sha(report_path, prefix=False),
            },
            "runtime": {
                "container_execution": True,
                "image_digest": "sha256:" + "a" * 64,
                "image_ref": "example.invalid/runtime@sha256:" + "a" * 64,
            },
        },
    )
    entry = {
        "report": str(report_path),
        "runtime_manifest": str(manifest_path),
        "baseline_report": str(baseline_path),
        "policy_pack": str(policy_path),
        "expected_runtime_image_digest": "sha256:" + "a" * 64,
    }
    if scenario in GENUINE_SCENARIOS:
        receipt_path = source / "execution.receipt.json"
        _write_json(
            receipt_path,
            {
                "schema": "invarlock.negative_fixture.execution_receipt.v1",
                "scenario": scenario,
                "execution_kind": GENUINE_SCENARIOS[scenario][1],
                "simulation": False,
                "command": "invarlock evaluate --config evaluated-scenario.yaml",
                "report_sha256": _sha(report_path),
                "runtime_manifest_sha256": _sha(manifest_path),
            },
        )
        entry["execution_receipt"] = str(receipt_path)
    return entry


def _write_evaluated_source(root: Path, scenario: str) -> dict[str, str]:
    source = root / scenario
    report = _strict_report(list(CANONICAL_GUARD_CHAIN))
    report["context"]["profile"] = "release"
    report["assurance"]["profile"] = "release"
    if scenario == "spectral_guard_failure":
        report["validation"]["spectral_stable"] = False
        report["spectral"].update(
            {
                "passed": False,
                "decision": "block",
                "status": "fail",
                "caps_applied": 6,
                "caps_exceeded": True,
                "violations": [{"module": "layer.0", "kind": "cap_exceeded"}],
            }
        )
    elif scenario == "rmt_guard_failure":
        report["validation"]["rmt_stable"] = False
        report["rmt"].update(
            {
                "passed": False,
                "decision": "block",
                "status": "fail",
                "stable": False,
                "epsilon_violations": [{"family": "ffn", "growth": 0.02}],
            }
        )
    elif scenario == "variance_guard_failure":
        report["variance"].update(
            {"enabled": True, "passed": False, "decision": "block"}
        )
        report["variance"]["predictive_gate"] = {
            "evaluated": True,
            "passed": False,
            "reason": "minimum_effect_not_met",
        }
    elif scenario == "invariants_failure":
        report["validation"]["invariants_pass"] = False
        report["invariants"].update(
            {
                "passed": False,
                "decision": "block",
                "status": "fail",
                "failures": [{"check": "weight_finiteness", "module": "lm_head"}],
                "violations": [{"kind": "weight_finiteness", "module": "lm_head"}],
            }
        )
    elif scenario == "primary_metric_failure":
        log_ratio = math.log(1.2)
        log_subject = math.log(2.4)
        report["evaluation_windows"]["final"]["logloss"] = [log_subject] * 180
        report["primary_metric"].update(
            {
                "final": 2.4,
                "ratio_vs_baseline": 1.2,
                "ci": [log_ratio, log_ratio],
                "display_ci": [1.2, 1.2],
                "analysis_point_final": log_subject,
            }
        )
        report["validation"]["primary_metric_acceptable"] = False
    bind_runtime_policy_receipt(report)
    report_path = source / "evaluation.report.json"
    baseline_path = source / "baseline.report.json"
    manifest_path = source / "runtime.manifest.json"
    policy_path = source / "acceptance_policy_pack.json"
    baseline = _matching_strict_ppl_baseline(report)
    baseline["context"]["profile"] = "release"
    _bind_strict_baseline(report, baseline)
    policy = _matching_strict_policy_pack(report)
    _write_json(report_path, report)
    _write_json(baseline_path, baseline)
    _write_json(policy_path, policy)
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": "2026-07-09T00:00:00+00:00",
        "verifier_contract_version": "runtime-manifest-v1",
        "execution_mode": "container",
        "config": {"path": None, "sha256": None, "source": "missing"},
        "report": {
            "filename": report_path.name,
            "path": str(report_path),
            "sha256": _sha(report_path, prefix=False),
        },
        "runtime": {
            "image_ref": "ghcr.io/invarlock/invarlock-runtime:test",
            "image_digest": "sha256:" + "a" * 64,
            "container_execution": True,
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }
    _write_json(manifest_path, manifest)
    entry = {
        "report": str(report_path),
        "runtime_manifest": str(manifest_path),
        "baseline_report": str(baseline_path),
        "policy_pack": str(policy_path),
        "expected_runtime_image_digest": "sha256:" + "a" * 64,
    }
    if scenario in GENUINE_SCENARIOS:
        receipt_path = source / "execution.receipt.json"
        _write_json(
            receipt_path,
            {
                "schema": "invarlock.negative_fixture.execution_receipt.v1",
                "scenario": scenario,
                "execution_kind": GENUINE_SCENARIOS[scenario][1],
                "simulation": False,
                "command": "invarlock evaluate --config evaluated-scenario.yaml",
                "report_sha256": _sha(report_path),
                "runtime_manifest_sha256": _sha(manifest_path),
            },
        )
        entry["execution_receipt"] = str(receipt_path)
    return entry


def _write_spec(root: Path) -> Path:
    spec_path = root / "generation_spec.json"
    _write_json(
        spec_path,
        {
            "schema": "invarlock.negative_fixture.generation_spec.v1",
            "scenarios": {
                scenario: _write_source(root / "sources", scenario)
                for scenario in ALL_SCENARIOS
            },
        },
    )
    return spec_path


def _write_evaluated_spec(root: Path) -> Path:
    spec_path = root / "evaluated_generation_spec.json"
    _write_json(
        spec_path,
        {
            "schema": "invarlock.negative_fixture.generation_spec.v1",
            "scenarios": {
                scenario: _write_evaluated_source(root / "evaluated", scenario)
                for scenario in ALL_SCENARIOS
            },
        },
    )
    return spec_path


def _fake_release_verifier(
    reports: list[Path], **_kwargs: object
) -> VerifyExecutionResult:
    scenario = reports[0].parent.name
    return VerifyExecutionResult(
        outcome=VerifyOutcome.POLICY_FAIL,
        payload={"ok": False},
        diagnostics=(
            VerifyDiagnostic(level="detail", message=EXPECTED_FAILURE_TEXT[scenario]),
        ),
    )


def _published_bundle(output: Path) -> Path:
    pointer = json.loads((output / "negative_fixtures.current.json").read_text())
    return output / pointer["bundle"]


def test_generation_promotes_bound_genuine_failures_and_labeled_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_module, "run_verify_reports", _fake_release_verifier)
    spec = _write_spec(tmp_path)
    output = tmp_path / "public_evidence"

    generate(spec, output)
    bundle = _published_bundle(output)

    for scenario in ALL_SCENARIOS:
        category = (
            GENUINE_SCENARIOS[scenario][0]
            if scenario in GENUINE_SCENARIOS
            else "policy_failures"
        )
        destination = bundle / category / scenario
        receipt = json.loads((destination / "generation.receipt.json").read_text())
        inventory = json.loads((destination / "hash_inventory.json").read_text())
        manifest = json.loads((destination / "runtime.manifest.json").read_text())
        assert manifest["report"]["sha256"] == _sha(
            destination / "evaluation.report.json", prefix=False
        )
        assert receipt["simulation"] is (scenario == "runtime_provenance_failure")
        assert receipt["authority"] == {
            "filename": "negative_fixtures.current.json",
            "schema": "invarlock.negative_fixture.current.v1",
            "evidence_kind": "negative_fixture",
        }
        for artifact in inventory["artifacts"]:
            path = destination / artifact["path"]
            assert artifact["bytes"] == path.stat().st_size
            assert artifact["sha256"] == _sha(path)

    simulated = bundle / "policy_failures/runtime_provenance_failure"
    simulated_manifest = json.loads((simulated / "runtime.manifest.json").read_text())
    assert simulated_manifest["execution_mode"] == "host-bypass"
    assert simulated_manifest["runtime"]["container_execution"] is False
    assert simulated_manifest["runtime"]["image_digest"] is None


def test_generation_rejects_fake_pm_passing_guard_fixture_without_replacing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_module, "run_verify_reports", _fake_release_verifier)
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    report_path = Path(spec["scenarios"]["spectral_guard_failure"]["report"])
    report = json.loads(report_path.read_text())
    report["validation"]["primary_metric_acceptable"] = False
    _write_json(report_path, report)
    output = tmp_path / "public_evidence"
    destination = output / "caught_regressions/spectral_guard_failure"
    _write_json(destination / "sentinel.json", {"preserved": True})

    with pytest.raises(
        NegativeFixtureError,
        match="validation.primary_metric_acceptable must be true",
    ):
        generate(spec_path, output)

    assert json.loads((destination / "sentinel.json").read_text()) == {
        "preserved": True
    }
    assert not (output / "negative_fixtures.current.json").exists()


def test_generation_rejects_receipt_that_labels_a_genuine_case_as_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_module, "run_verify_reports", _fake_release_verifier)
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    receipt_path = Path(spec["scenarios"]["invariants_failure"]["execution_receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["simulation"] = True
    _write_json(receipt_path, receipt)

    with pytest.raises(NegativeFixtureError, match="simulation"):
        generate(spec_path, tmp_path / "public_evidence")

    assert not (tmp_path / "public_evidence/negative_fixtures.current.json").exists()


def test_generation_rejects_duplicate_generation_spec_keys(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(
        payload.replace(
            '"schema": "invarlock.negative_fixture.generation_spec.v1"',
            '"schema": "invarlock.negative_fixture.generation_spec.v1",\n'
            '  "schema": "invarlock.negative_fixture.generation_spec.v1"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(NegativeFixtureError, match="cannot load generation spec"):
        generate(spec_path, tmp_path / "public_evidence")


def test_generation_rejects_named_failure_not_reproduced_by_release_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_spec(tmp_path)

    def unrelated_failure(
        _reports: list[Path], **_kwargs: object
    ) -> VerifyExecutionResult:
        return VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload={"ok": False},
            diagnostics=(
                VerifyDiagnostic(level="detail", message="unrelated policy failure"),
            ),
        )

    monkeypatch.setattr(generation_module, "run_verify_reports", unrelated_failure)

    with pytest.raises(
        NegativeFixtureError,
        match="release verifier did not reproduce spectral_guard_failure",
    ):
        generate(spec_path, tmp_path / "public_evidence")

    assert not (tmp_path / "public_evidence/negative_fixtures.current.json").exists()


def test_generation_rejects_source_mutation_during_verifier_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_spec(tmp_path)

    def mutate_baseline(reports: list[Path], **kwargs: object) -> VerifyExecutionResult:
        baseline = Path(str(kwargs["baseline"]))
        baseline.write_bytes(baseline.read_bytes() + b"\n")
        return _fake_release_verifier(reports)

    monkeypatch.setattr(generation_module, "run_verify_reports", mutate_baseline)

    with pytest.raises(NegativeFixtureError, match="trusted baseline report changed"):
        generate(spec_path, tmp_path / "public_evidence")

    assert not (tmp_path / "public_evidence/negative_fixtures.current.json").exists()


def test_bundle_publication_keeps_old_authority_when_pointer_swap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public_evidence"
    generate(_write_evaluated_spec(tmp_path / "first"), output)
    old_bundle = _published_bundle(output)
    pointer_path = output / "negative_fixtures.current.json"
    pointer_before = pointer_path.read_bytes()
    original_replace = generation_module.os.replace

    def fail_pointer_swap(source: Path, destination: Path) -> None:
        if Path(destination) == pointer_path:
            raise OSError("injected pointer-swap failure")
        original_replace(source, destination)

    monkeypatch.setattr(generation_module.os, "replace", fail_pointer_swap)

    with pytest.raises(OSError, match="pointer-swap failure"):
        generate(_write_evaluated_spec(tmp_path / "second"), output)

    assert pointer_path.read_bytes() == pointer_before
    assert _published_bundle(output) == old_bundle
    assert not list(output.glob(".negative-fixtures-pointer-*.tmp"))
    assert len(list((output / "negative_fixture_bundles").iterdir())) == 1


def test_bundle_files_are_fsynced_before_bundle_becomes_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public_evidence"
    stage = tmp_path / "stage"
    _write_json(stage / "nested/value.json", {"durable": True})
    events: list[str] = []
    original_fsync_tree = generation_module._fsync_tree
    original_replace = generation_module.os.replace

    def record_fsync(root: Path) -> None:
        original_fsync_tree(root)
        if root == stage:
            events.append("stage_fsynced")

    def record_replace(source: Path, destination: Path) -> None:
        if Path(source) == stage:
            events.append("bundle_published")
        original_replace(source, destination)

    monkeypatch.setattr(generation_module, "_fsync_tree", record_fsync)
    monkeypatch.setattr(generation_module.os, "replace", record_replace)

    generation_module._commit_immutable_bundle(stage, output)

    assert events[:2] == ["stage_fsynced", "bundle_published"]


def test_authority_pointer_switch_occurs_only_after_bundle_and_pointer_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public_evidence"
    spec_path = _write_evaluated_spec(tmp_path / "evaluated")
    pointer_path = output / "negative_fixtures.current.json"
    bundle_parent = output / "negative_fixture_bundles"
    state = {
        "bundle_parent_fsynced": False,
        "pointer_file_fsynced": False,
        "output_fsynced": False,
    }
    original_fsync = generation_module.os.fsync
    original_fsync_directory = generation_module._fsync_directory
    original_replace = generation_module.os.replace

    def record_fsync(descriptor: int) -> None:
        if list(output.glob(".negative-fixtures-pointer-*.tmp")):
            state["pointer_file_fsynced"] = True
        original_fsync(descriptor)

    def record_fsync_directory(path: Path) -> None:
        original_fsync_directory(path)
        if path == bundle_parent:
            state["bundle_parent_fsynced"] = True
        elif path == output:
            state["output_fsynced"] = True

    def record_replace(source: Path, destination: Path) -> None:
        if Path(destination) == pointer_path:
            assert state["bundle_parent_fsynced"] is True
            assert state["pointer_file_fsynced"] is True
        original_replace(source, destination)

    monkeypatch.setattr(generation_module.os, "fsync", record_fsync)
    monkeypatch.setattr(generation_module, "_fsync_directory", record_fsync_directory)
    monkeypatch.setattr(generation_module.os, "replace", record_replace)

    generate(spec_path, output)

    assert state["output_fsynced"] is True


def test_post_switch_directory_fsync_failure_does_not_orphan_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public_evidence"
    spec_path = _write_evaluated_spec(tmp_path / "evaluated")
    original_fsync_directory = generation_module._fsync_directory

    def fail_output_fsync(path: Path) -> None:
        if path == output:
            raise OSError("injected authority-directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(generation_module, "_fsync_directory", fail_output_fsync)

    with pytest.raises(RuntimeError, match="durability could not be confirmed"):
        generate(spec_path, output)

    pointer = json.loads((output / "negative_fixtures.current.json").read_text())
    assert (output / pointer["bundle"]).is_dir()


def test_bundle_fsync_failure_cannot_switch_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public_evidence"
    stage = tmp_path / "stage"
    _write_json(stage / "value.json", {"durable": False})

    def fail_fsync(_root: Path) -> None:
        raise OSError("injected staged fsync failure")

    monkeypatch.setattr(generation_module, "_fsync_tree", fail_fsync)

    with pytest.raises(OSError, match="staged fsync failure"):
        generation_module._commit_immutable_bundle(stage, output)

    assert stage.is_dir()
    assert not (output / "negative_fixtures.current.json").exists()
    bundle_parent = output / "negative_fixture_bundles"
    assert not bundle_parent.exists() or not list(bundle_parent.iterdir())


def test_real_release_verifier_reproduces_all_six_failures_and_late_error_is_atomic(
    tmp_path: Path,
) -> None:
    spec_path = _write_evaluated_spec(tmp_path)
    output = tmp_path / "public_evidence"

    generate(spec_path, output)
    bundle = _published_bundle(output)

    before: dict[str, str] = {}
    for scenario in ALL_SCENARIOS:
        category = (
            GENUINE_SCENARIOS[scenario][0]
            if scenario in GENUINE_SCENARIOS
            else "policy_failures"
        )
        destination = bundle / category / scenario
        receipt = json.loads((destination / "generation.receipt.json").read_text())
        assert receipt["release_verifier"]["outcome"] == "policy_fail"
        assert receipt["release_verifier"]["structural_contract"] == "clean"
        assert (
            receipt["release_verifier"]["expected_failure_text"]
            == EXPECTED_FAILURE_TEXT[scenario]
        )
        before[scenario] = _sha(destination / "hash_inventory.json")

    spec = json.loads(spec_path.read_text())
    late_receipt = Path(
        spec["scenarios"]["primary_metric_failure"]["execution_receipt"]
    )
    receipt = json.loads(late_receipt.read_text())
    receipt["simulation"] = True
    _write_json(late_receipt, receipt)
    pointer_before = (output / "negative_fixtures.current.json").read_bytes()

    with pytest.raises(NegativeFixtureError, match="simulation"):
        generate(spec_path, output)

    assert (output / "negative_fixtures.current.json").read_bytes() == pointer_before

    for scenario in ALL_SCENARIOS:
        category = (
            GENUINE_SCENARIOS[scenario][0]
            if scenario in GENUINE_SCENARIOS
            else "policy_failures"
        )
        assert (
            _sha(bundle / category / scenario / "hash_inventory.json")
            == before[scenario]
        )


def test_current_negative_index_is_a_strict_public_consumer_surface(
    tmp_path: Path,
) -> None:
    spec_path = _write_evaluated_spec(tmp_path)
    output = tmp_path / "public_evidence"
    output.mkdir()
    (output / "README.md").write_text("# public evidence\n", encoding="utf-8")

    generate(spec_path, output)

    pointer = json.loads((output / "negative_fixtures.current.json").read_text())
    assert pointer["schema"] == "invarlock.negative_fixture.current.v1"
    assert pointer["current_contract_status"] == "current_strict_negative_evidence"
    assert pointer["release_status"] == "negative_evidence_only_not_release_ready"
    assert pointer["strict_contract"] == {
        "profile": "release",
        "assurance_mode": "strict",
        "expected_outcome": "policy_fail",
    }
    assert [entry["scenario"] for entry in pointer["scenarios"]] == list(ALL_SCENARIOS)
    assert (
        check_public_evidence(
            output,
            require_current_negative_evidence=True,
        )
        == []
    )

    bundle = output / pointer["bundle"]
    (bundle / "unexpected.json").write_text("{}\n", encoding="utf-8")
    errors = check_public_evidence(
        output,
        require_current_negative_evidence=True,
    )
    assert any(
        "immutable bundle file set is not canonical" in error for error in errors
    )
    (bundle / "unexpected.json").unlink()

    pointer["scenarios"][0]["artifacts"]["evaluation_report"]["sha256"] = (
        "sha256:" + "0" * 64
    )
    _write_json(output / "negative_fixtures.current.json", pointer)

    errors = check_public_evidence(
        output,
        require_current_negative_evidence=True,
    )

    assert any("indexed sha256 does not match" in error for error in errors)


def test_current_negative_index_rejects_duplicate_keys_retired_v2_and_open_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_module, "run_verify_reports", _fake_release_verifier)
    output = tmp_path / "public_evidence"
    generate(_write_spec(tmp_path), output)
    pointer_path = output / "negative_fixtures.current.json"
    pointer_bytes = pointer_path.read_bytes()
    pointer_text = pointer_bytes.decode("utf-8")

    pointer_path.write_text(
        pointer_text.replace(
            '"schema": "invarlock.negative_fixture.current.v1",',
            '"schema": "invarlock.negative_fixture.current.v1",\n'
            '  "schema": "invarlock.negative_fixture.current.v1",',
            1,
        ),
        encoding="utf-8",
    )
    duplicate_errors = check_public_evidence(
        output, require_current_negative_evidence=True
    )
    assert any("duplicate key" in error for error in duplicate_errors)

    pointer = json.loads(pointer_bytes)
    pointer["schema"] = "invarlock.negative_fixture.current.v2"
    _write_json(pointer_path, pointer)
    retired_errors = check_public_evidence(
        output, require_current_negative_evidence=True
    )
    assert any(
        "current index requires invarlock.negative_fixture.current.v1" in error
        for error in retired_errors
    )

    pointer["schema"] = "invarlock.negative_fixture.current.v1"
    pointer["unreviewed_claim"] = True
    pointer["scenarios"][0]["unreviewed_claim"] = True
    _write_json(pointer_path, pointer)
    shape_errors = check_public_evidence(output, require_current_negative_evidence=True)
    assert any("exact v1 shape" in error for error in shape_errors)
