from __future__ import annotations

import importlib
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()


@dataclass(frozen=True)
class ModuleTombstone:
    old_path: str
    scan_filename: bool = True

    @property
    def module_name(self) -> str:
        path = self.old_path.removeprefix("src/").removesuffix(".py")
        return path.replace("/", ".")

    @property
    def filename(self) -> str:
        return Path(self.old_path).name

    @property
    def reference_names(self) -> tuple[str, ...]:
        if self.scan_filename:
            return (self.module_name, self.filename)
        return (self.module_name,)


# A retired module belongs here when restoring it would recreate a duplicate
# owner, compatibility facade, or misleading public import surface. This is a
# name-tombstone policy, not a general missing-path checker. Published evidence
# paths remain the responsibility of the evidence inventory and publication
# gates.
MODULE_TOMBSTONES = tuple(
    ModuleTombstone(path)
    for path in (
        "src/invarlock/adapters/hf_quantized.py",
        "src/invarlock/clean_pruning_selection_evidence.py",
        "src/invarlock/cli/run_overhead.py",
        "src/invarlock/clean_selection/evidence.py",
        "src/invarlock/clean_selection_evidence.py",
        "src/invarlock/core/run_orchestrator_execute.py",
        "src/invarlock/core/run_orchestrator_execute_attempt_results.py",
        "src/invarlock/core/run_orchestrator_execute_attempts.py",
        "src/invarlock/core/run_orchestrator_execute_environment.py",
        "src/invarlock/core/run_orchestrator_execute_execution.py",
        "src/invarlock/core/run_orchestrator_execute_helpers.py",
        "src/invarlock/core/guard_metric_impact_contract.py",
        "src/invarlock/core/runner_eval_latency.py",
        "src/invarlock/core/runner_eval_metrics.py",
        "src/invarlock/core/runner_eval_metrics_multimodal.py",
        "src/invarlock/core/runner_eval_metrics_stats.py",
        "src/invarlock/core/runner_eval_phase.py",
        "src/invarlock/core/runner_eval_windows.py",
        "src/invarlock/core/runner_execution_phases.py",
        "src/invarlock/core/runner_execution_plan.py",
        "src/invarlock/core/runner_finalize.py",
        "src/invarlock/core/runner_guards.py",
        "src/invarlock/core/runner_pairing.py",
        "src/invarlock/deployable_logical_coverage.py",
        "src/invarlock/evidence_pack_edit_metadata.py",
        "src/invarlock/reporting/normalizer.py",
        "src/invarlock/reporting/render.py",
        "src/invarlock/reporting/render_guard_sections.py",
        "src/invarlock/reporting/render_markdown.py",
        "src/invarlock/reporting/render_markdown_structure.py",
        "src/invarlock/reporting/render_markdown_tables.py",
        "src/invarlock/reporting/report_builder.py",
        "src/invarlock/reporting/report_files.py",
        "src/invarlock/reporting/report_make_support.py",
        "src/invarlock/reporting/report_overhead.py",
        "src/invarlock/reporting/report_validation.py",
        "src/invarlock/reporting/report_validation_guard_flags.py",
        "src/invarlock/reporting/report_validation_metric_impact.py",
        "src/invarlock/reporting/report_validation_overhead.py",
        "src/invarlock/reporting/report_validation_thresholds.py",
        "src/invarlock/reporting/verify_checks.py",
    )
)
# Exact repository-relative tombstones cover retired public documentation,
# executable helpers, and packaged-data locations.  Unlike Python-module
# tombstones, these deliberately do not ban a basename everywhere: for
# example, tuned_edit_params.json remains a valid run-state filename even
# though the former repository fixture with that name was removed.
REPOSITORY_PATH_TOMBSTONES = (
    "docs/assurance/10-guard-overhead-method.md",
    "examples/integrations/fine_tune/materialize_tiny_fine_tune_subject.py",
    "examples/integrations/peft_lora/materialize_tiny_peft_lora_subject.py",
    "scripts/evidence_packs/python/editing/attach_pruning_selection_receipt.py",
    "scripts/evidence_packs/python/editing/prepare_clean_selection_execution.py",
    "scripts/evidence_packs/python/editing/pruning_contract.py",
    "scripts/evidence_packs/tuned_edit_params.json",
    "tests/reporting/guards/test_report_guard_metric_impact_fallback_ratio.py",
    "tests/reporting/overhead/test_report_determinism_and_guard_metric_impact_skip_passthrough.py",
    "tests/reporting/overhead/test_report_metric_impact_helper_paths.py",
    "tests/reporting/overhead/test_report_metric_impact_row_presence.py",
)

_RETIRED_PATH_PATTERN = re.compile(
    "|".join(
        re.escape(value)
        for value in sorted(
            REPOSITORY_PATH_TOMBSTONES
            + tuple(tombstone.old_path for tombstone in MODULE_TOMBSTONES),
            key=len,
            reverse=True,
        )
    )
)
_RETIRED_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])(?:"
    + "|".join(
        re.escape(value)
        for value in sorted(
            {
                value
                for tombstone in MODULE_TOMBSTONES
                for value in tombstone.reference_names
            },
            key=len,
            reverse=True,
        )
    )
    + r")(?![A-Za-z0-9_.])"
)
# These names belonged to the retired PPL-only guard-overhead representation.
# Canonical reports now separate generic guard metric impact from measured
# system overhead. Retaining the old field names would silently recreate the
# ambiguous contract even if the retired Python modules stayed deleted.
OBSOLETE_GUARD_METRIC_TERMS = (
    "bare_ppl",
    "guarded_ppl",
    "impact_ratio",
    "impact_percent",
    "impact_threshold",
    "metric_direction",
    "impact_basis",
    "impact_value",
    "guard_overhead",
    "guard_overhead_acceptable",
    "overhead_ratio",
    "overhead_percent",
    "overhead_threshold",
)
OBSOLETE_GUARD_METRIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(term) for term in OBSOLETE_GUARD_METRIC_TERMS)
    + r")(?![A-Za-z0-9_])"
)

# Packed backend storage and dense logical edit coverage are intentionally
# separate contracts. These retired names collapsed those two quantities and
# must not return through fixtures, scripts, documentation, or generated data.
OBSOLETE_DEPLOYABLE_QUANTIZATION_TERMS = (
    "packed_storage_parameter_elements",
    "quantized_parameter_count",
    "total_parameter_count",
)
OBSOLETE_DEPLOYABLE_QUANTIZATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(term) for term in OBSOLETE_DEPLOYABLE_QUANTIZATION_TERMS)
    + r")(?![A-Za-z0-9_])"
)

# Narrow exemptions cover negative assertions and synthetic architecture
# fixtures whose purpose is to prove that a tombstone is rejected. Each
# exemption applies to one exact retired value rather than hiding the file.
REFERENCE_EXEMPTIONS = {
    (
        Path("tests/ci/test_make_coverage_targets.py"),
        "src/invarlock/clean_pruning_selection_evidence.py",
    ),
    (
        Path("tests/ci/test_make_coverage_targets.py"),
        "clean_pruning_selection_evidence.py",
    ),
    (
        Path("tests/reporting/schema/test_import_surface.py"),
        "invarlock.reporting.report_builder",
    ),
    (
        Path("tests/reporting/schema/test_import_surface.py"),
        "invarlock.reporting.report_make_support",
    ),
    (
        Path("tests/reporting/schema/test_import_surface.py"),
        "invarlock.reporting.verify_checks",
    ),
    (
        Path("tests/reporting/schema/test_import_surface.py"),
        "invarlock.reporting.render",
    ),
    (
        Path("tests/reporting/schema/test_import_surface.py"),
        "invarlock.reporting.report_files",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "src/invarlock/reporting/report_builder.py",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "report_builder.py",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "src/invarlock/reporting/report_make_support.py",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "report_make_support.py",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "src/invarlock/reporting/verify_checks.py",
    ),
    (
        Path("tests/lint/test_architecture_guardrails_failure_paths.py"),
        "verify_checks.py",
    ),
    (
        Path("tests/scripts/test_architecture_docs_refs.py"),
        "report_builder.py",
    ),
    (
        Path("tests/scripts/test_architecture_docs_refs.py"),
        "report_make_support.py",
    ),
}

# The runtime rejection tables and validation test use removed field literals
# only to prove the canonical contract fails closed. No documentation,
# configuration, script, or ordinary fixture is exempt.
OBSOLETE_TERM_EXEMPTIONS = {
    (Path(path), term)
    for path in (
        "src/invarlock/core/assurance_guard_validation_raw.py",
        "src/invarlock/core/assurance_guard_validation_runtime.py",
        "src/invarlock/reporting/report_metric_impact.py",
        "tests/reporting/validation/test_guard_metric_impact_report_surface.py",
    )
    for term in (
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    )
} | {(Path("tests/core/test_assurance_replay_fail_closed_edges.py"), "bare_ppl")}

OBSOLETE_TERM_QUARANTINE: dict[Path, dict[str, int]] = {}

HISTORICAL_TEXT_EXEMPTIONS = {Path("CHANGELOG.md")}


@lru_cache(maxsize=1)
def _tracked_text_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    paths: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        path = REPO_ROOT / relative
        if path.resolve() == THIS_FILE or relative in HISTORICAL_TEXT_EXEMPTIONS:
            continue
        if not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        paths.append(path)
    return tuple(paths)


def _reference_offenders() -> list[str]:
    offenders: list[str] = []
    for path in _tracked_text_files():
        relative = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        retired_paths = set(_RETIRED_PATH_PATTERN.findall(text))
        retired_names = set(_RETIRED_NAME_PATTERN.findall(text))
        for retired_path in REPOSITORY_PATH_TOMBSTONES:
            if retired_path in retired_paths:
                offenders.append(f"{relative} -> {retired_path}")
        observed_terms = Counter(OBSOLETE_GUARD_METRIC_PATTERN.findall(text))
        for retired_term in OBSOLETE_GUARD_METRIC_TERMS:
            observed = observed_terms[retired_term]
            quarantined = OBSOLETE_TERM_QUARANTINE.get(relative, {}).get(
                retired_term, 0
            )
            if observed == quarantined:
                continue
            if (relative, retired_term) in OBSOLETE_TERM_EXEMPTIONS:
                continue
            if quarantined:
                offenders.append(
                    f"{relative} -> {retired_term} count {observed}, expected {quarantined}"
                )
            elif observed:
                offenders.append(f"{relative} -> {retired_term}")
        for retired_term in set(OBSOLETE_DEPLOYABLE_QUANTIZATION_PATTERN.findall(text)):
            offenders.append(f"{relative} -> {retired_term}")
        for tombstone in MODULE_TOMBSTONES:
            if (
                tombstone.old_path in retired_paths
                and (
                    relative,
                    tombstone.old_path,
                )
                not in REFERENCE_EXEMPTIONS
            ):
                offenders.append(f"{relative} -> {tombstone.old_path}")
            if (
                relative,
                tombstone.module_name,
            ) not in REFERENCE_EXEMPTIONS and tombstone.module_name in retired_names:
                offenders.append(f"{relative} -> {tombstone.module_name}")
            if (
                tombstone.scan_filename
                and (
                    relative,
                    tombstone.filename,
                )
                not in REFERENCE_EXEMPTIONS
                and tombstone.filename in retired_names
            ):
                offenders.append(f"{relative} -> {tombstone.filename}")
    return sorted(set(offenders))


def test_retired_module_paths_remain_absent() -> None:
    assert not [
        tombstone.old_path
        for tombstone in MODULE_TOMBSTONES
        if (REPO_ROOT / tombstone.old_path).exists()
    ]


def test_retired_deployable_logical_coverage_import_fails() -> None:
    with pytest.raises(
        ModuleNotFoundError,
        match=r"No module named 'invarlock.deployable_logical_coverage'",
    ):
        importlib.import_module("invarlock.deployable_logical_coverage")


def test_retired_repository_paths_remain_absent() -> None:
    assert not [
        retired_path
        for retired_path in REPOSITORY_PATH_TOMBSTONES
        if (REPO_ROOT / retired_path).exists()
    ]


def test_retired_module_names_are_absent_from_tracked_text() -> None:
    assert _reference_offenders() == []


def test_obsolete_term_quarantine_is_empty() -> None:
    assert OBSOLETE_TERM_QUARANTINE == {}


def test_guard_metric_impact_tombstones_cover_all_removed_names() -> None:
    module_paths = {tombstone.old_path for tombstone in MODULE_TOMBSTONES}
    assert "src/invarlock/core/guard_metric_impact_contract.py" in module_paths
    assert {
        "tests/reporting/guards/test_report_guard_metric_impact_fallback_ratio.py",
        "tests/reporting/overhead/test_report_determinism_and_guard_metric_impact_skip_passthrough.py",
        "tests/reporting/overhead/test_report_metric_impact_helper_paths.py",
        "tests/reporting/overhead/test_report_metric_impact_row_presence.py",
    } <= set(REPOSITORY_PATH_TOMBSTONES)

    obsolete = Counter(OBSOLETE_GUARD_METRIC_PATTERN.findall("guard_overhead"))
    assert obsolete == {"guard_overhead": 1}
    assert (
        OBSOLETE_GUARD_METRIC_PATTERN.findall(
            "guard_runtime_overhead_threshold system_overhead"
        )
        == []
    )


def test_deployable_quantization_tombstones_cover_ambiguous_old_names() -> None:
    assert set(OBSOLETE_DEPLOYABLE_QUANTIZATION_TERMS) == {
        "packed_storage_parameter_elements",
        "quantized_parameter_count",
        "total_parameter_count",
    }


def test_tombstone_scan_covers_public_repository_surfaces() -> None:
    relative_paths = {path.relative_to(REPO_ROOT) for path in _tracked_text_files()}
    roots = {path.parts[0] for path in relative_paths}

    assert Path("Makefile") in relative_paths
    assert Path("README.md") in relative_paths
    assert Path("pyproject.toml") in relative_paths
    assert {
        ".github",
        "configs",
        "contracts",
        "docs",
        "examples",
        "public_evidence",
        "scripts",
        "src",
        "tests",
    } <= roots
