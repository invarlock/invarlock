from __future__ import annotations

import json
import re
import subprocess
import sys
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
    "reference/acceptance-attestations.md",
    "reference/architecture.md",
    "reference/artifacts.md",
    "reference/cli.md",
    "reference/compatibility.md",
    "reference/contracts.md",
    "reference/pipeline-contracts.md",
    "reference/evaluator-qualification.md",
    "reference/environment.md",
    "reference/lifecycle.md",
    "reference/release-verification.md",
    "reference/reports.md",
    "reference/policy-engine-interop.md",
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
    "user-guide/pipeline-integration.md",
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


def test_readme_resources_use_absolute_urls_for_pypi() -> None:
    readme = _read("README.md")
    embedded_urls = re.findall(r'\b(?:href|src|srcset)="([^"]+)"', readme)
    markdown_urls = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    resource_urls = embedded_urls + markdown_urls

    assert resource_urls
    assert all(url.startswith(("https://", "mailto:")) for url in resource_urls)


def test_readme_links_to_the_schema_valid_public_request() -> None:
    readme = _read("README.md")
    request_url = (
        "https://github.com/invarlock/invarlock/blob/main/examples/request.yaml"
    )

    assert request_url in readme
    assert "omits `--runtime-image` and `--runtime-image-digest`" in readme
    request = yaml.safe_load(_read("examples/request.yaml"))
    schema = json.loads(_read("contracts/evaluation_request.schema.json"))
    jsonschema.Draft202012Validator(schema).validate(request)


def test_readme_hierarchy_promotes_evaluator_neutral_evidence_paths() -> None:
    readme = _read("README.md")
    headings = (
        "## Evidence paths",
        "## Decision boundary",
        "## Try the signed handoff locally",
        "## Inspect published evidence",
        "## Run, verify, and report",
        "## The release-regression decision",
        "## Import and qualify evaluator results",
        "## Hand off acceptance",
        "## Providers and diagnostics",
        "## Documentation",
    )
    positions = [readme.index(heading) for heading in headings]

    assert positions == sorted(positions)
    evidence_paths = readme[positions[0] : positions[1]]
    for phrase in (
        "Native execution",
        "Adapter support",
        "Replay authority",
        "Signed-journey maturity",
        "evaluator-neutral contracts",
    ):
        assert phrase in evidence_paths
    assert evidence_paths.index("evaluation-verification-flow.svg") < (
        evidence_paths.index("| Axis |")
    )

    introduction = readme[: positions[0]]
    assert "in-toto/DSSE" not in introduction
    decision_boundary = " ".join(readme[positions[1] : positions[2]].split())
    for phrase in (
        "one precise question",
        "authenticated evidence and independently supplied trust anchors",
        "reproducible, portable, and suitable for recipient-controlled approval",
        "Broader deployment, safety, compliance, and organizational decisions",
        "complete claim boundary and assumptions",
    ):
        assert phrase in decision_boundary
    assert "## Scope and non-goals" not in readme

    acceptance = readme[positions[7] : positions[8]]
    assert "in-toto/DSSE" in acceptance
    assert "**Compatibility note:**" in acceptance


def test_readme_first_run_commands_track_checked_in_surfaces() -> None:
    readme = _read("README.md")
    makefile = _read("Makefile")
    evidence_root = REPO_ROOT / "public_evidence/evidence/mistral-7b-weight-scale-hf"

    assert "python run.py --fixture golden" in readme
    assert "\nexample-quickstart:" in makefile
    assert REPO_ROOT.joinpath("examples/quickstart/run.py").is_file()
    assert REPO_ROOT.joinpath(
        "examples/acceptance-handoff/golden/technical-anchors.json"
    ).is_file()

    report_path = "public_evidence/evidence/mistral-7b-weight-scale-hf/evidence"
    receipt_path = (
        "public_evidence/evidence/mistral-7b-weight-scale-hf/verification.receipt.json"
    )
    assert report_path in readme
    assert receipt_path in readme
    assert evidence_root.joinpath("evidence/manifest.json").is_file()
    assert evidence_root.joinpath("verification.receipt.json").is_file()


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
    assert "EleutherAI/LAMBADA OpenAI" in notices
    assert "Software Copyright (c) 2019 OpenAI" in notices
    assert "evaluator qualification profiles" in notices
    assert "shared MMLU-Pro semantic artifact" in notices

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


def test_runtime_and_report_references_track_current_closed_contracts() -> None:
    runtime_reference = _read("docs/reference/runtime-providers.md")
    for setting in (
        "cpu_threads",
        "prompt_batch_size",
        "prompt_microbatch_size",
        "processor_metadata_sha256",
    ):
        assert f"`{setting}`" in runtime_reference

    gguf_addin = _read("addins/gguf/README.md")
    gguf_guide = _read("docs/user-guide/runtime-providers.md")
    for fragment in (
        "cpu_threads=16",
        "prompt_batch_size=512",
        "prompt_microbatch_size=512",
    ):
        assert fragment in gguf_addin
        assert fragment in gguf_guide

    contracts = _read("docs/reference/contracts.md")
    assert "The v3 comparison report is the current writer format" in contracts
    assert "| Behavioral schedule | `format_version`, `task`," in contracts
    assert "`minimum_side_accuracy`" in contracts

    cli = _read("docs/reference/cli.md")
    assert "[--json]" in cli.split("## `report`", maxsplit=1)[1]
    assert "invarlock/evidence-report-v1" in cli

    api = _read("docs/reference/api-guide.md")
    assert "expected_request_digest: str | None = None" in api


def test_documentation_lint_discovers_maintained_markdown_surfaces() -> None:
    makefile = _read("Makefile")
    assert makefile.count("git ls-files -z -- ':(icase,glob)**/*.md'") == 2
    assert makefile.count("xargs -0") == 2
    assert "scripts/checks/check_public_text.py" in makefile


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


def test_acceptance_docs_state_the_external_policy_engine_boundary() -> None:
    text = " ".join(
        "\n".join(
            _read(path)
            for path in (
                "docs/reference/acceptance-attestations.md",
                "docs/reference/policy-engine-interop.md",
                "examples/acceptance-handoff/README.md",
            )
        )
        .lower()
        .split()
    )

    assert "open policy agent" in text
    assert "cue" in text
    assert "without an invarlock service or policy-engine plugin" in text
    assert "standalone verifier" in text
    assert "do not themselves perform raw ed25519 verification" in text


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
        "reference/compatibility.md",
        "reference/acceptance-attestations.md",
        "reference/policy-engine-interop.md",
        "reference/evaluator-qualification.md",
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
        'd="M 530 99 L 565 99"',
        'd="M 1100 99 L 1065 99"',
        'd="M 815 141 L 815 163"',
        'd="M 815 241 L 815 263"',
        'd="M 510 680 L 510 700"',
        'd="M 440 775 L 440 795"',
        'd="M 650 775 C 700 783 790 788 840 795"',
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
    assert (
        'src="https://raw.githubusercontent.com/invarlock/invarlock/main/'
        'docs/assets/evaluation-verification-flow.svg"'
    ) in readme
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
        "make example-hf-transformers",
        "make example-hf-vision-text",
        "make example-peft-lora",
        "make example-evidence-handoff",
        "separately generated trust inputs",
        "human-readable report",
    ):
        assert fragment in example
    assert re.search(r"signed evidence\s+pack", example)

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
    handoff = _read("examples/run_trust_boundary_demo.py")
    assert 'expected_policy_verdict="fail"' in handoff
    assert "tampered_report.write_bytes" in handoff
    makefile = _read("Makefile")
    handoff_recipe = makefile.split("trust-boundary-demo:", 1)[1].split("\n\n", 1)[0]
    assert "rm -rf" not in handoff_recipe


def test_model_change_guide_preserves_the_external_artifact_boundary() -> None:
    guide = _read("docs/user-guide/change-scenarios.md")
    for phrase in (
        "trained",
        "pruned",
        "quantization",
        "GGUF",
        "TensorRT-LLM",
        "Vision-text",
        "harness",
        "endpoint",
    ):
        assert phrase in guide
    assert "use 400" in guide
    assert "make example-peft-lora" in guide
    assert re.search(r"remains responsible for creating that\s+artifact", guide)
    assert "InvarLock does not rebuild it" in guide


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
        "comparison-report-v3",
        "runtime-side-report-v1",
        "minimum_record_count",
        "minimum_side_accuracy",
        "maximum_interval_width_pp",
        "maximum_interval_width_ratio",
        "normalized-request digest",
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
        "make release-reference-journey",
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


def test_latest_release_changelog_is_a_product_synthesis() -> None:
    changelog = _read("CHANGELOG.md")
    unreleased, remainder = changelog.split("## [0.15.0]", maxsplit=1)
    release, remainder = remainder.split("## [0.14.0]", maxsplit=1)
    previous_release = remainder.split("## [0.13.0]", maxsplit=1)[0]
    normalized = " ".join(release.split())
    previous_normalized = " ".join(previous_release.split())
    assert "independently replayable deployment evidence" not in unreleased
    assert "independently replayable deployment evidence" in normalized
    assert "Comparison-report v3" in normalized
    assert "minimum-side accuracy" in normalized
    assert "BF16, GGUF, QAT, and OCI" in normalized
    assert "clean-consumer candidate-wheel replay" in normalized
    assert "SPDX 3.0.1 AI observation" in normalized
    assert "without expanding core contracts" in normalized
    assert "evaluator-neutral qualification" not in unreleased
    assert "evaluator-neutral qualification" in previous_normalized
    assert "recipient-controlled acceptance handoff" in previous_normalized
    assert "observation-only" in previous_normalized
    assert "canonical in-toto/DSSE acceptance envelope" in previous_normalized
    assert "OPA/Rego and CUE" in previous_normalized
    assert "clean-checkout v0.13 compatibility corpus" in previous_normalized
    for heading in ("### Added", "### Changed", "### Removed", "### Fixed"):
        assert heading in unreleased
        assert heading in release
        assert heading in previous_release


def test_evaluator_docs_preserve_qualification_and_integration_depth() -> None:
    text = _read("docs/reference/evaluator-qualification.md")
    normalized = " ".join(text.split())

    assert "Adapter support" in text
    assert "Replay authority" in text
    assert "Signed-journey maturity" in text
    assert "LM Evaluation Harness" in text
    assert "every retained independently replayable import" in normalized
    assert "102-record" in text
    assert "102 shared outputs" in text
    assert "Retained (2 signed transactions, 400 records each)" in text
    assert "agree on every baseline and subject output" in text
    assert "does not assign one cumulative" in text
    assert "retained current-model OCI journeys" in normalized
    assert "without claiming a native signed journey" in normalized
    assert "maximum 10-percentage-point paired interval width" in normalized
    assert "600 records" not in normalized
    assert "Benchmark harnesses" in text
    assert "Application evaluation SDKs" in text
    assert "Evaluation and observability platforms" in text
    assert "Microsoft PromptFlow" in text
    assert "Azure AI Evaluation" in text
    assert "Only LM Evaluation Harness" not in text
    assert "remains the only evaluator example" not in text


def test_evaluator_documentation_matrix_matches_retained_manifests() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "examples"
                / "evaluator-qualification"
                / "render_docs_matrix.py"
            ),
            "--check",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
