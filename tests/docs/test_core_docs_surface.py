from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_core_docs_do_not_promote_removed_top_level_commands():
    surfaces = [
        "README.md",
        "docs/user-guide/getting-started.md",
        "docs/user-guide/quickstart.md",
    ]
    banned = [
        "invarlock run",
        "invarlock proof-pack",
        "invarlock policy",
        "plugins install",
        "plugins uninstall",
        "allow-host-execution",
        "INVARLOCK_ALLOW_HOST_EXECUTION=1",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        for needle in banned:
            assert needle not in text, f"{needle} still promoted in {rel_path}"


def test_support_surfaces_use_local_mode_for_public_evaluate_examples():
    surfaces = [
        "CONTRIBUTING.md",
        "configs/README.md",
        "notebooks/invarlock_quickstart_cpu.ipynb",
        "notebooks/invarlock_compare_evaluate.ipynb",
        "notebooks/invarlock_custom_datasets.ipynb",
        "notebooks/invarlock_policy_tiers.ipynb",
        "notebooks/invarlock_evaluation_report_deep_dive.ipynb",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert "--mode local" in text, f"--mode local missing from {rel_path}"
        assert "INVARLOCK_ALLOW_HOST_EXECUTION=1" not in text, (
            f"legacy host-execution env still promoted in {rel_path}"
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
        "invarlock run",
        "invarlock plugins",
        "invarlock calibrate",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        for needle in banned:
            assert needle not in text, f"{needle} still present in {rel_path}"


def test_public_security_and_reference_docs_use_local_mode_for_public_host_runs():
    surfaces = [
        "docs/reference/datasets.md",
        "docs/security/best-practices.md",
        "docs/security/threat-model.md",
        "docs/user-guide/example-reports.md",
        "docs/assurance/glossary.md",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        assert "--mode local" in text, f"--mode local missing from {rel_path}"


def test_proof_pack_docs_keep_repo_wrappers_advanced_and_use_current_verify_surface():
    text = _read("docs/user-guide/proof-packs.md")
    assert "repo-only" in text
    assert "invarlock advanced proof-pack verify" in text
    assert "invarlock proof-pack verify" not in text
    assert "invarlock run" not in text
    assert "--allow-host-execution" not in text
