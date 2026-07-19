from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DOCS = (
    "README.md",
    "docs/README.md",
    "docs/user-guide/getting-started.md",
    "docs/reference/cli.md",
    "docs/reference/architecture.md",
)
JOURNEY_COMMANDS = (
    "invarlock evaluate",
    "invarlock verify",
    "invarlock report",
)
AUXILIARY_DOCS = (
    "SUPPORT.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    ".github/WORKFLOWS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    "examples/README.md",
    "public_evidence/README.md",
    "requirements/README.md",
    "scripts/README.md",
    "tests/README.md",
)
EXPECTED_DOC_PAGES = {
    "README.md",
    "assurance/acceptance-checklist.md",
    "assurance/assurance-case.md",
    "assurance/decision-semantics.md",
    "assurance/glossary.md",
    "assurance/pairing-and-replay.md",
    "assurance/reproducibility.md",
    "reference/api-guide.md",
    "reference/architecture.md",
    "reference/artifacts.md",
    "reference/cli.md",
    "reference/contracts.md",
    "reference/environment.md",
    "reference/lifecycle.md",
    "reference/release-verification.md",
    "reference/reports.md",
    "reference/runtime-security.md",
    "reference/runtime-providers.md",
    "reference/documentation.md",
    "security/best-practices.md",
    "security/dependency-audit.md",
    "security/threat-model.md",
    "security/trust-model.md",
    "user-guide/diagnostics.md",
    "user-guide/ci-integration.md",
    "user-guide/change-scenarios.md",
    "user-guide/evaluation-request.md",
    "user-guide/evidence-and-verification.md",
    "user-guide/public-evidence.md",
    "user-guide/getting-started.md",
    "user-guide/key-management.md",
    "user-guide/runtime-providers.md",
    "user-guide/schedule-and-policy.md",
    "user-guide/troubleshooting.md",
    "user-guide/verification-failure-lab.md",
}
DOCUMENT_TYPE_CONTRACTS = {
    "user-guide": (
        '!!! tip "User guide"',
        ("**Outcome:**", "**Audience:**", "**Prerequisites:**"),
    ),
    "assurance": (
        '!!! abstract "Assurance note"',
        (
            "**In plain language:**",
            "**Question:**",
            "**Decision use:**",
            "**Evidence:**",
        ),
    ),
    "reference": (
        '!!! info "Reference"',
        ("**Surface:**", "**Stability:**", "**Use this page when:**"),
    ),
    "security": (
        '!!! warning "Security guidance"',
        (
            "**In plain language:**",
            "**Objective:**",
            "**Assets or boundary:**",
            "**Use this page when:**",
        ),
    ),
}


def _read(relative: str) -> str:
    return REPO_ROOT.joinpath(relative).read_text(encoding="utf-8")


def _declared_dependency_names(relative: str, *, extra: str | None = None) -> set[str]:
    project = tomllib.loads(_read(relative))["project"]
    requirements = (
        project["dependencies"]
        if extra is None
        else project["optional-dependencies"][extra]
    )
    return {
        re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement).group(0).lower()
        for requirement in requirements
    }


def _code_table_names(section: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(r"^\| `([^`]+)` \|", section, re.MULTILINE)
    }


def test_core_docs_present_the_three_transaction_journey() -> None:
    for relative in CORE_DOCS:
        text = _read(relative)
        missing = [command for command in JOURNEY_COMMANDS if command not in text]
        assert missing == [], f"{relative} misses {missing}"


def test_public_docs_describe_the_release_assurance_surface() -> None:
    text = "\n".join(_read(relative) for relative in CORE_DOCS).lower()
    for phrase in (
        "baseline",
        "subject",
        "dataset",
        "runtime",
        "policy",
        "provider abi",
        "canonical evidence",
        "signed verification receipt",
        "hugging face",
        "gguf",
        "tensorrt-llm",
        "observation-only",
    ):
        assert phrase in text


def test_auxiliary_docs_track_the_product_and_release_surface() -> None:
    text_by_path = {relative: _read(relative) for relative in AUXILIARY_DOCS}
    all_text = "\n".join(text_by_path.values())

    assert "invarlock --version" in text_by_path["SUPPORT.md"]
    assert "signed verification receipt" in text_by_path["SUPPORT.md"]
    assert "invarlock --version" in text_by_path[".github/ISSUE_TEMPLATE/bug_report.md"]

    workflows = text_by_path[".github/WORKFLOWS.md"]
    notices = text_by_path["THIRD_PARTY_NOTICES.md"]
    distribution_projects = (
        "pyproject.toml",
        "addins/diagnostics/pyproject.toml",
        "addins/gguf/pyproject.toml",
        "addins/multimodal/pyproject.toml",
        "addins/tensorrt_llm/pyproject.toml",
    )
    distributions = {
        tomllib.loads(_read(relative))["project"]["name"]
        for relative in distribution_projects
    }
    assert distributions == {
        "invarlock",
        "invarlock-diagnostics",
        "invarlock-runtime-gguf",
        "invarlock-runtime-hf-vision-text",
        "invarlock-runtime-tensorrt-llm",
    }
    for distribution in distributions:
        assert distribution in workflows
        assert distribution in notices

    core_section = notices.split("## Core distribution", maxsplit=1)[1].split(
        "## Hugging Face extra", maxsplit=1
    )[0]
    hf_section = notices.split("## Hugging Face extra", maxsplit=1)[1].split(
        "## First-party optional distributions", maxsplit=1
    )[0]
    assert _code_table_names(core_section) == _declared_dependency_names(
        "pyproject.toml"
    )
    assert _code_table_names(hf_section) == _declared_dependency_names(
        "pyproject.toml", extra="hf"
    )

    scripts = text_by_path["scripts/README.md"]
    for argument in (
        "--release-sha",
        "--expected-version",
        "--hash-manifest",
    ):
        assert argument in scripts

    for retired in (
        "invarlock doctor",
        "invarlock version",
        "events.jsonl",
        "tests/adapters",
        "tests/calibration",
        "tests/edits",
        "tests/guards",
        "tests/plugins",
        "reports/eval/",
    ):
        assert retired not in all_text


def test_runtime_qualification_docs_use_authenticated_candidate_wheels() -> None:
    reference = _read("docs/reference/runtime-providers.md")
    assert "scripts/qualification_candidate_wheels.py" in reference
    assert "CANDIDATE_WHEEL_MANIFEST" in reference
    assert "third-party dependency environment" in reference

    guide = _read("docs/user-guide/runtime-providers.md")
    assert "scripts/qualification_candidate_wheels.py" in guide
    assert "--wheel dist/invarlock-*.whl" in guide

    addin_wheels = {
        "addins/gguf/README.md": "invarlock_runtime_gguf-*.whl",
        "addins/multimodal/README.md": ("invarlock_runtime_hf_vision_text-*.whl"),
        "addins/tensorrt_llm/README.md": ("invarlock_runtime_tensorrt_llm-*.whl"),
    }
    for relative, wheel in addin_wheels.items():
        text = _read(relative)
        assert "scripts/qualification_candidate_wheels.py" in text
        assert "CANDIDATE_WHEEL_MANIFEST" in text
        assert "dist/invarlock-*.whl" in text
        assert wheel in text


def test_documentation_lint_covers_maintained_markdown_surfaces() -> None:
    makefile = _read("Makefile")
    workflow = _read(".github/workflows/docs-ci.yml")
    for pattern in (
        "CODE_OF_CONDUCT.md",
        "SUPPORT.md",
        "THIRD_PARTY_NOTICES.md",
        '".github/**/*.md"',
        '"examples/**/*.md"',
        '"requirements/**/*.md"',
        '"tests/README.md"',
    ):
        assert makefile.count(pattern) == 2
    for pattern in (
        "- '*.md'",
        "- '.github/**/*.md'",
        "- 'examples/**/*.md'",
        "- 'requirements/**/*.md'",
        "- 'tests/README.md'",
        "- 'tests/docs/**'",
    ):
        assert workflow.count(pattern) == 2


def test_docs_describe_the_narrow_engine_and_embedding_facade() -> None:
    text = "\n".join(
        _read(path).lower()
        for path in (
            "README.md",
            "docs/README.md",
            "docs/reference/architecture.md",
            "docs/reference/api-guide.md",
        )
    )
    for required in (
        "invarlock.engine",
        "invarlock evaluate",
        "invarlock verify",
        "invarlock report",
        "runtime provider",
        "signed evidence",
    ):
        assert required in text


def test_navigation_contains_only_existing_pages() -> None:
    config = yaml.safe_load(_read("mkdocs.yml"))
    rendered = str(config["nav"])
    for path in (
        "user-guide/getting-started.md",
        "user-guide/evaluation-request.md",
        "user-guide/evidence-and-verification.md",
        "user-guide/verification-failure-lab.md",
        "user-guide/public-evidence.md",
        "user-guide/ci-integration.md",
        "user-guide/schedule-and-policy.md",
        "user-guide/key-management.md",
        "user-guide/runtime-providers.md",
        "user-guide/diagnostics.md",
        "user-guide/troubleshooting.md",
        "assurance/assurance-case.md",
        "assurance/decision-semantics.md",
        "assurance/pairing-and-replay.md",
        "assurance/reproducibility.md",
        "assurance/acceptance-checklist.md",
        "assurance/glossary.md",
        "reference/cli.md",
        "reference/architecture.md",
        "reference/contracts.md",
        "reference/api-guide.md",
        "reference/artifacts.md",
        "reference/reports.md",
        "reference/runtime-providers.md",
        "reference/runtime-security.md",
        "reference/environment.md",
        "reference/lifecycle.md",
        "reference/release-verification.md",
        "reference/documentation.md",
        "security/trust-model.md",
        "security/threat-model.md",
        "security/best-practices.md",
        "security/dependency-audit.md",
    ):
        assert path in rendered
        assert REPO_ROOT.joinpath("docs", path).is_file()


def test_docs_tree_has_only_the_documented_pages() -> None:
    actual = {
        path.relative_to(REPO_ROOT / "docs").as_posix()
        for path in (REPO_ROOT / "docs").rglob("*.md")
    }
    assert actual == EXPECTED_DOC_PAGES


def test_typed_docs_declare_their_reader_contract() -> None:
    for directory, (marker, fields) in DOCUMENT_TYPE_CONTRACTS.items():
        pages = sorted(REPO_ROOT.joinpath("docs", directory).glob("*.md"))
        assert pages, f"no pages found for {directory}"
        for page in pages:
            text = page.read_text(encoding="utf-8")
            opener = "\n".join(text.splitlines()[:32])
            assert text.count(marker) == 1, page
            assert marker in opener, f"reader contract is too late in {page}"
            positions = [opener.index(field) for field in fields]
            assert positions == sorted(positions), f"reader contract order in {page}"
            for field in fields:
                value = opener.split(field, maxsplit=1)[1].splitlines()[0].strip()
                assert value, f"empty {field} in {page}"


def test_workflow_diagram_tracks_current_transactions() -> None:
    svg = _read("docs/assets/evaluation-verification-flow.svg")
    diagram = svg.lower()
    for phrase in (
        "request.yaml",
        "baseline artifact",
        "subject artifact",
        "prepare schedule · run pinned oci sides or import evidence",
        "invarlock evaluate",
        "paired comparison and interval",
        "invarlock/evidence-pack-v1",
        "canonical signed evidence bundle",
        "baseline + subject artifact digests · schedule digest",
        "invarlock verify",
        "authenticate pack · replay pairs and interval under anchors",
        "signed verification receipt",
        "acceptance result",
        "scoped pass or rejection",
        "invarlock report",
        "console · optional HTML",
        "human-readable evidence view",
    ):
        assert phrase.lower() in diagram
    for stale in (
        "spectral -&gt; RMT",
        "evaluation.report.json",
        "report html",
        "nonzero: rejected",
    ):
        assert stale.lower() not in diagram
    for connection in (
        'd="M 580 99 L 620 99"',
        'd="M 1020 99 L 980 99"',
        'd="M 800 141 L 800 163"',
        'd="M 800 241 L 800 263"',
        'd="M 430 753 L 430 765"',
        'd="M 610 753 C 650 758 740 760 790 765"',
    ):
        assert connection in svg
    assert 'd="M 610 794 L 640 794"' not in svg

    dependency_svg = _read("docs/assets/reference-evidence-dependency.svg")
    assert "manifest + anchors + verdict" in dependency_svg
    assert 'd="M 680 598 L 770 598"' not in dependency_svg

    pairing_svg = _read("docs/assets/user-guide-pairing-contract.svg")
    for phrase in (
        "Newcombe v2",
        "legacy v1 replay by report format",
        "metric bound · record count",
        "interval-width precision",
    ):
        assert phrase in pairing_svg

    readme = _read("README.md")
    architecture = _read("docs/reference/architecture.md")
    assert 'src="docs/assets/evaluation-verification-flow.svg"' in readme
    assert "../assets/evaluation-verification-flow.svg" in architecture


def test_example_request_conforms_to_the_closed_schema_surface() -> None:
    request = yaml.safe_load(_read("examples/request.yaml"))
    schema = json.loads(_read("contracts/evaluation_request.schema.json"))
    jsonschema.Draft202012Validator(schema).validate(request)
    assert request["format_version"] == "invarlock/evaluation-request-v1"
    assert set(request) == {"format_version", "comparison", "execution", "output"}
    assert request["execution"]["mode"] == "import"
    assert request["execution"]["records"] == "import/paired-records.json"
    assert request["execution"]["schedule"] == "inputs/schedule.json"
    assert request["comparison"]["dataset"] == "inputs/schedule.json"
    assert request["comparison"]["policy"] == "policy/acceptance.json"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == (
        "hf_transformers"
    )
    assert request["comparison"]["subject"]["runtime"]["provider"] == (
        "hf_transformers"
    )


def test_public_example_includes_every_required_input_and_verify_anchor() -> None:
    for relative in (
        "examples/inputs/schedule.json",
        "examples/policy/acceptance.json",
        "examples/generate_keys.py",
    ):
        assert REPO_ROOT.joinpath(relative).is_file(), f"missing {relative}"

    example = _read("examples/README.md")
    for fragment in (
        "--signing-key",
        "--policy",
        "--expected-baseline-artifact",
        "--expected-subject-artifact",
        "--expected-schedule",
        "--expected-baseline-runtime",
        "--expected-subject-runtime",
        "--expected-signer",
        "--receipt",
        "--verifier-signing-key",
        "--verifier-identity",
    ):
        assert fragment in example

    policy = json.loads(_read("examples/policy/acceptance.json"))
    assert policy == {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -10.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 50,
                }
            }
        }
    }
    assert "one-record (`-2` percentage-point)" in example
    assert "`-50` percentage-point regression" not in example


def test_change_scenario_guide_preserves_the_external_artifact_boundary() -> None:
    guide = _read("docs/user-guide/change-scenarios.md")
    catalog = _read("examples/scenarios/README.md")
    for phrase in (
        "fine-tuned",
        "pruned",
        "quantized",
        "GGUF",
        "TensorRT-LLM",
        "multimodal",
        "external-harness",
        "serving-endpoint",
        "evidence-handoff",
    ):
        assert phrase in guide
        assert phrase in catalog
    assert re.search(r"at least 400\s+eligible paired records", guide)
    assert "make example-scenarios-check" in guide
    assert "does not install a training or compression framework" in guide


def test_complete_docs_cover_current_assurance_and_claim_limits() -> None:
    text = "\n".join(
        _read(path)
        for path in (
            "docs/assurance/assurance-case.md",
            "docs/assurance/decision-semantics.md",
            "docs/assurance/pairing-and-replay.md",
            "docs/assurance/reproducibility.md",
            "docs/security/trust-model.md",
            "docs/security/threat-model.md",
            "docs/reference/contracts.md",
        )
    ).lower()
    for phrase in (
        "exact_match",
        "normalized_nll_per_utf8_byte",
        "fixed schedule",
        "population inference",
        "execution attestation",
        "representative",
        "paired-records-v1",
        "comparison-report-v1",
        "comparison-report-v2",
        "runtime-side-report-v1",
        "minimum_record_count",
        "maximum_interval_width_pp",
        "maximum_interval_width_ratio",
        "verification receipt",
    ):
        assert phrase in text


def test_preflight_docs_name_current_format_and_pending_precision() -> None:
    text = "\n".join(
        _read(path)
        for path in (
            "docs/reference/api-guide.md",
            "docs/reference/cli.md",
            "docs/security/best-practices.md",
        )
    )
    assert "invarlock/evaluation-preflight-v2" in text
    assert "pending_execution" in text
    assert "invarlock/evaluation-preflight-v1" not in text


def test_receipt_reader_example_does_not_self_authorize_the_verifier() -> None:
    api_guide = _read("docs/reference/api-guide.md")
    assert "expected_verifier_fingerprint=verifier_fingerprint" in api_guide
    assert 'verification.payload["verifier_fingerprint"]' not in api_guide


def test_operational_guides_pin_current_failure_publication_and_release_paths() -> None:
    failure_lab = _read("docs/user-guide/verification-failure-lab.md")
    for fragment in (
        "tampered-evidence/reports/evaluation.report.json",
        "chmod u+w tampered-evidence/reports/evaluation.report.json",
        "chmod u+w extra-file-evidence",
        'expected_verifier_fingerprint="sha256:" + "0" * 64',
    ):
        assert fragment in failure_lab
    assert "tampered-evidence/report.json" not in failure_lab

    publication = _read("docs/user-guide/public-evidence.md")
    for fragment in (
        "public_evidence/evidence_index.json",
        '"external_asset"',
        "make public-evidence-sync",
        "make public-evidence-audit",
    ):
        assert fragment in publication

    ci_integration = _read("docs/user-guide/ci-integration.md")
    assert "pip install --require-hashes" in ci_integration
    assert "invarlock==$INVARLOCK_VERSION" not in ci_integration
    assert "INVARLOCK_POLICY_SHA256" in ci_integration
    assert "acceptance policy digest mismatch" in ci_integration

    release = _read("docs/reference/release-verification.md")
    for fragment in (
        "make dist-check",
        "make release-preflight",
        '--hash-manifest "$HASH_MANIFEST"',
        "scripts/release/make_offline_bundle.sh",
        "fix forward under a new version",
    ):
        assert fragment in release

    security = _read("docs/security/best-practices.md")
    for fragment in (
        "single orchestrating process",
        "does not currently expose a KMS, HSM, or remote-signer interface",
        "Data handling and retention",
    ):
        assert fragment in security


def test_unreleased_changelog_is_a_product_synthesis() -> None:
    changelog = _read("CHANGELOG.md")
    unreleased = changelog.split("## [0.12.1]", maxsplit=1)[0]
    assert "paired model release-regression evaluation" in unreleased
    assert "exact two-sided McNemar" in unreleased
    assert "perplexity ratio as a verifier-derived likelihood" in unreleased
    assert "paired schedule-resampling interval" in unreleased
    assert "host-to-OCI" in unreleased
    assert "canonical evidence bundles" in unreleased
    assert "invarlock.engine" in unreleased
    assert "invarlock-diagnostics" in unreleased
    assert "### Added" in unreleased
    assert "### Changed" in unreleased
    assert "### Removed" in unreleased
    assert "### Fixed" in unreleased
