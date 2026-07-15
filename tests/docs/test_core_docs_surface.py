from __future__ import annotations

import re
from pathlib import Path

from invarlock.public_contracts import stable_cli_json_surfaces

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOVED_HOST_MODE_TOKEN = "trusted" + "-local"
REMOVED_RUN_COMMAND = "invarlock " + "run"
REMOVED_SOURCE_FLAG = "--" + "source "
REMOVED_EDITED_FLAG = "--" + "edited "


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_core_docs_promote_evaluate_verify_report_html():
    surfaces = [
        "README.md",
        "docs/user-guide/getting-started.md",
        "docs/user-guide/quickstart.md",
        "docs/reference/cli.md",
    ]
    for rel_path in surfaces:
        text = _read(rel_path)
        assert "invarlock evaluate" in text
        assert "invarlock verify" in text
        assert "report html" in text


def test_readme_and_cli_reference_cover_first_touch_and_runtime_verify_surfaces():
    readme = _read("README.md")
    cli_ref = _read("docs/reference/cli.md")

    assert "invarlock --help" in readme
    assert "invarlock --version" in readme
    assert "invarlock report --help" in readme
    assert "invarlock advanced --help" in readme
    assert "invarlock advanced runtime-verify" in readme

    assert "First-Touch Surfaces" in cli_ref
    assert "invarlock advanced calibrate --help" in cli_ref
    assert "invarlock advanced runtime-verify --help" in cli_ref


def test_core_docs_do_not_promote_removed_top_level_commands():
    surfaces = [
        "README.md",
        "docs/user-guide/getting-started.md",
        "docs/user-guide/quickstart.md",
    ]
    banned = [
        REMOVED_RUN_COMMAND,
        "invarlock evidence-pack",
        "invarlock policy",
        "plugins install",
        "plugins uninstall",
        "allow-host-execution",
        "INVARLOCK_ALLOW_HOST_EXECUTION=1",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        for needle in banned:
            assert needle not in text, f"{needle} still documented in {rel_path}"


def test_public_compare_examples_use_baseline_subject_terms() -> None:
    surfaces = [
        "README.md",
        "docs/user-guide/compare-and-evaluate.md",
        "docs/user-guide/knowledge-and-self-edit-workflows.md",
        "docs/user-guide/quickstart.md",
        "docs/reference/cli.md",
        "configs/presets/causal_lm/hf_text_c4_128.yaml",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert REMOVED_SOURCE_FLAG not in text, (
            f"removed source flag still present in {rel_path}"
        )
        assert REMOVED_EDITED_FLAG not in text, (
            f"removed edited flag still present in {rel_path}"
        )


def test_support_surfaces_use_host_mode_assurance_for_public_evaluate_examples():
    surfaces = [
        "CONTRIBUTING.md",
        "configs/README.md",
        "notebooks/invarlock_quickstart_cpu.ipynb",
        "notebooks/invarlock_compare_evaluate.ipynb",
        "notebooks/invarlock_custom_datasets.ipynb",
        "notebooks/invarlock_policy_tiers.ipynb",
        "notebooks/invarlock_python_api.ipynb",
        "notebooks/invarlock_evaluation_report_deep_dive.ipynb",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert "--execution-mode host" in text, (
            f"--execution-mode host missing from {rel_path}"
        )
        assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in text, (
            f"legacy host-execution env still documented in {rel_path}"
        )


def test_support_surfaces_do_not_teach_removed_public_top_level_commands():
    surfaces = [
        "CONTRIBUTING.md",
        "configs/README.md",
        "configs/presets/causal_lm/wikitext2_512.yaml",
        "configs/local/quant8_calibrated.example.yaml",
        "docs/assurance/glossary.md",
    ]

    banned = [
        REMOVED_RUN_COMMAND,
        "invarlock plugins",
        "invarlock calibrate",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        for needle in banned:
            assert needle not in text, f"{needle} still present in {rel_path}"


def test_public_security_and_reference_docs_use_host_mode_assurance_for_public_host_runs():
    surfaces = [
        "docs/reference/datasets.md",
        "docs/security/best-practices.md",
        "docs/security/threat-model.md",
        "docs/user-guide/example-reports.md",
        "docs/assurance/glossary.md",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert "--execution-mode host" in text, (
            f"--execution-mode host missing from {rel_path}"
        )


def test_public_docs_do_not_mention_removed_trusted_local_mode() -> None:
    surfaces = [
        "README.md",
        "docs/README.md",
        "docs/reference/cli.md",
        "docs/user-guide/quickstart.md",
        "docs/user-guide/compare-and-evaluate.md",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert REMOVED_HOST_MODE_TOKEN not in text, (
            f"removed legacy host-mode term still present in {rel_path}"
        )


def test_evidence_pack_docs_keep_repo_wrappers_advanced_and_use_current_verify_surface():
    text = _read("docs/user-guide/evidence-packs.md")
    assert "scripts/evidence_packs/verify_pack.sh" in text
    assert "invarlock advanced evidence-pack verify" in text
    assert "invarlock evidence-pack verify" not in text
    assert REMOVED_RUN_COMMAND not in text
    assert "--allow-host-execution" not in text


def test_notebook_links_and_docs_navigation_cover_curated_live_examples() -> None:
    mkdocs = _read("mkdocs.yml")
    docs_hub = _read("docs/README.md")
    deep_dive = _read("notebooks/invarlock_evaluation_report_deep_dive.ipynb")

    assert "invarlock_evaluation_report_deep_dive.ipynb" in deep_dive
    assert "notebooks/invarlock_python_api.ipynb" in docs_hub
    assert "notebooks/invarlock_policy_tiers.ipynb" in docs_hub
    assert "Live Examples" in mkdocs


def test_knowledge_self_edit_workflow_docs_are_discoverable() -> None:
    mkdocs = _read("mkdocs.yml")
    docs_hub = _read("docs/README.md")
    reference_index = _read("docs/reference/index.md")
    getting_started = _read("docs/user-guide/getting-started.md")
    compare = _read("docs/user-guide/compare-and-evaluate.md")

    expected = "user-guide/knowledge-and-self-edit-workflows.md"
    assert expected in mkdocs
    assert "Knowledge & Self-Edit Workflows" in mkdocs
    assert "knowledge-and-self-edit-workflows.md" in docs_hub
    assert "knowledge-and-self-edit-workflows.md" in reference_index
    assert "knowledge-and-self-edit-workflows.md" in getting_started
    assert "knowledge-and-self-edit-workflows.md" in compare


def test_programmatic_docs_mark_python_surface_as_advanced_not_contract_stable() -> (
    None
):
    quickstart = _read("docs/reference/programmatic-quickstart.md")
    api = _read("docs/reference/api-guide.md")
    index = _read("docs/reference/index.md")

    assert "advanced/non-stable" in quickstart
    assert "advanced/non-stable" in api
    assert "CoreRunner.execute and helpers" not in index


def test_contract_reference_docs_freeze_versioned_json_and_packaged_public_evidence():
    text = _read("docs/reference/contracts.md")
    normalized = " ".join(text.split())

    assert "Versioned sub-contracts" in text
    assert "InvarLock is pre-1.0 as a package" in text
    assert "stable within their declared contract versions" in text
    assert "Report schema" in text
    assert "Evidence-pack format" in text
    assert "Verifier output" in text
    assert "CLI stability policy" in text
    assert "Adapter support tiers" in text
    assert 'format_version: "verify-v1"' in text
    assert 'format_version: "runtime-verify-v1"' in text
    assert 'format_version: "evidence-pack-verify-v1"' in text
    assert 'verify.format_version: "verify-v1"' in text
    assert "invarlock/_data/public_evidence/published_basis_index.json" in text
    assert "Evidence not yet created" in text
    assert "empty index" in text
    assert "maintained public contract carriers" in normalized
    assert "make_public_contract_bundle.py" not in text


def test_cli_docs_list_stable_cli_json_surfaces() -> None:
    text = _read("docs/reference/contracts.md") + "\n" + _read("docs/reference/cli.md")

    for command, format_version in stable_cli_json_surfaces().items():
        assert command in text
        assert format_version in text


def test_cli_reference_lists_stable_and_experimental_command_classes() -> None:
    text = _read("docs/reference/cli.md")

    assert "Stable vs Experimental Commands" in text
    assert "Stable core workflow" in text
    assert "Stable JSON automation" in text
    assert "Stable advanced verifiers" in text
    assert "Experimental or maintainer-only" in text
    assert "not covered by the public CLI stability contract" in text


def test_byod_end_to_end_example_has_enough_rows_for_requested_windows() -> None:
    text = _read("docs/user-guide/bring-your-own-data.md")
    match = re.search(
        r"## End-to-end example \(local JSONL\)\n\n```bash\n(.*?)\n```",
        text,
        re.DOTALL,
    )
    assert match is not None
    block = match.group(1)
    preview = int(re.search(r"preview_n:\s*(\d+)", block).group(1))
    final = int(re.search(r"final_n:\s*(\d+)", block).group(1))
    rows = len(re.findall(r'\{"text":"[^"]+"\}', block))

    assert rows >= preview + final
