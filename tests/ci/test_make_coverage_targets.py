from __future__ import annotations

from pathlib import Path


def _coverage_tests_eval_block() -> str:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    marker = "COVERAGE_TESTS_EVAL := \\"
    start = text.index(marker)
    rest = text[start:].splitlines()

    lines: list[str] = []
    for line in rest:
        if line.startswith("COVERAGE_TESTS_CLI_COMMANDS :="):
            break
        lines.append(line)
    return "\n".join(lines)


def test_coverage_target_includes_active_eval_data_and_helper_tests() -> None:
    block = _coverage_tests_eval_block()

    expected_patterns = (
        "tests/eval/test_task_metrics.py",
        "tests/eval/test_eval_bootstrap_wrapper.py",
        "tests/eval/test_metric_tail_gate.py",
        "tests/eval/test_data*.py",
        "tests/eval/test_hf_text_provider*.py",
        "tests/eval/test_local_jsonl*.py",
        "tests/eval/test_synthetic_provider_cases.py",
        "tests/eval/test_wikitext2_fast_capacity.py",
        "tests/eval/test_provider_deterministic_loader_cases.py",
        "tests/eval/test_difficulty_scorer_modes.py",
        "tests/eval/providers",
    )

    for pattern in expected_patterns:
        assert pattern in block


def test_coverage_target_includes_probe_suite_for_plain_coverage_run() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "COVERAGE_TESTS_EVAL_PROBES :=" in text
    for pattern in (
        "tests/eval/test_fft.py",
        "tests/eval/test_fft_probe_cases.py",
        "tests/eval/test_mi*.py",
        "tests/eval/test_post_attention_probes.py",
        "tests/eval/test_post_attention_probe_cases.py",
    ):
        assert pattern in text
    assert "$(COVERAGE) run --append -m pytest -q -p no:cov" in text


def test_coverage_target_includes_core_cli_surface_and_runtime_security_tests() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    for pattern in (
        "tests/cli/test_core_command_surface.py",
        "tests/cli/test_execution_mode.py",
        "tests/cli/test_removed_command_migrations.py",
        "tests/cli/test_python_m_invarlock.py",
        "tests/cli/test_security_default_contract.py",
        "tests/cli/test_container_delegation.py",
        "invarlock/runtime_security.py",
    ):
        assert pattern in text


def test_coverage_include_does_not_embed_space_prefixed_cli_patterns() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    line = next(
        raw_line
        for raw_line in text.splitlines()
        if raw_line.startswith("COVERAGE_INCLUDE :=")
    )

    assert "\\" not in line
    assert ", src/invarlock/cli/*" not in line
    assert ", invarlock/cli/*" not in line
    assert "src/invarlock/cli/*" in line
    assert "src/invarlock/cli/commands/*" in line
    assert "src/invarlock/public_contracts.py" in line
    assert "src/invarlock/proof_pack.py" in line
    assert "src/invarlock/runtime_security.py" in line
    assert "invarlock/cli/commands/*" in line
