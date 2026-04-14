from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT_INIT_FILES = (
    REPO_ROOT / "src/invarlock/__init__.py",
    REPO_ROOT / "src/invarlock/core/__init__.py",
    REPO_ROOT / "src/invarlock/adapters/__init__.py",
    REPO_ROOT / "src/invarlock/guards/__init__.py",
)
OWNER_LAYER_ROOTS = (
    REPO_ROOT / "src/invarlock/core",
    REPO_ROOT / "src/invarlock/reporting",
)
REMOVED_REPORTING_MODULES = (
    REPO_ROOT / "src/invarlock/reporting/report_builder.py",
    REPO_ROOT / "src/invarlock/reporting/report_make_support.py",
    REPO_ROOT / "src/invarlock/reporting/verify_checks.py",
)
RUN_COMMAND_PATH = REPO_ROOT / "src/invarlock/cli/commands/run.py"
RUN_EXECUTION_PATH = REPO_ROOT / "src/invarlock/cli/run_execution.py"
REPORT_FILES_PATH = REPO_ROOT / "src/invarlock/reporting/report_files.py"
METRICS_PATH = REPO_ROOT / "src/invarlock/eval/metrics.py"
METRICS_LENS_PATH = REPO_ROOT / "src/invarlock/eval/metrics_lens.py"
CONFIG_RUNTIME_PATH = REPO_ROOT / "src/invarlock/core/config_runtime.py"
CONFIG_LOADER_PATH = REPO_ROOT / "src/invarlock/core/config_loader.py"
RUNTIME_SECURITY_PATH = REPO_ROOT / "src/invarlock/runtime_security.py"
BROAD_EXCEPTION = "except " + "Exception"
BROAD_EXCEPTION_AS_ERROR = BROAD_EXCEPTION + " as error"
BROAD_EXCEPTION_RETURN_ZERO = "except " + "Exception:\\n                return 0.0"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_hf_adapter_local_only_retry_uses_cache_miss_detection() -> None:
    path = REPO_ROOT / "src/invarlock/adapters/hf_mixin.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_pretrained_model":
            target = node
            break

    assert target is not None, "_load_pretrained_model not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_is_local_loader_cache_miss" in source
    assert "prefer_local_files_only" in source
    assert BROAD_EXCEPTION not in source


def test_guarded_benchmark_failures_raise_instead_of_continuing() -> None:
    path = REPO_ROOT / "src/invarlock/eval/bench_runner.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_single_run":
            target = node
            break

    assert target is not None, "execute_single_run not found"
    source = ast.get_source_segment(text, target) or ""
    assert "Guard construction failed" in source
    assert "_build_benchmark_run_report(" in source
    assert BROAD_EXCEPTION not in source


def test_execute_scenario_surfaces_benchmark_assembly_failures() -> None:
    path = REPO_ROOT / "src/invarlock/eval/bench_runner.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_scenario":
            target = node
            break

    assert target is not None, "execute_scenario not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_assign_dataset_provider(" in source
    assert "_extract_success_report_path(" in source
    assert "Evaluation report generation failed for" in source
    assert BROAD_EXCEPTION not in source


def test_spectral_validation_unexpected_failures_raise() -> None:
    path = REPO_ROOT / "src/invarlock/guards/spectral_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_guard":
            target = node
            break

    assert target is not None, "validate_guard not found"
    source = ast.get_source_segment(text, target) or ""
    assert BROAD_EXCEPTION_AS_ERROR not in source


def test_spectral_prepare_and_after_edit_do_not_swallow_runtime_failures() -> None:
    path = REPO_ROOT / "src/invarlock/guards/spectral_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    targets: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "prepare_guard",
            "after_edit_guard",
        }:
            targets[node.name] = node

    assert set(targets) == {"prepare_guard", "after_edit_guard"}

    prepare_source = ast.get_source_segment(text, targets["prepare_guard"]) or ""
    assert BROAD_EXCEPTION not in prepare_source
    assert 'raise RuntimeError("Failed to prepare spectral guard.")' in prepare_source

    after_edit_source = ast.get_source_segment(text, targets["after_edit_guard"]) or ""
    assert BROAD_EXCEPTION not in after_edit_source
    assert (
        'raise RuntimeError("Post-edit spectral analysis failed.")' in after_edit_source
    )


def test_latency_measurement_failures_raise_runtime_errors() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "measure_latency":
            target = node
            break

    assert target is not None, "measure_latency not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_latency_validation_error(" in source
    assert "non-empty evaluation window" in source
    assert "sequence longer than 10 tokens" in source
    assert "attended token" in source
    assert "Latency warmup failed." in source
    assert "Latency measurement failed." in source
    assert "return 0.0" not in source
    assert BROAD_EXCEPTION_RETURN_ZERO not in source


def test_metrics_runtime_does_not_hide_device_vocab_or_memory_failures() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    targets: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_resolve_eval_device",
            "_infer_model_vocab_size",
            "measure_memory",
        }:
            targets[node.name] = node

    assert set(targets) == {
        "_resolve_eval_device",
        "_infer_model_vocab_size",
        "measure_memory",
    }

    resolve_source = ast.get_source_segment(text, targets["_resolve_eval_device"]) or ""
    assert BROAD_EXCEPTION not in resolve_source

    vocab_source = (
        ast.get_source_segment(text, targets["_infer_model_vocab_size"]) or ""
    )
    assert BROAD_EXCEPTION not in vocab_source
    assert "return None" in vocab_source

    memory_source = ast.get_source_segment(text, targets["measure_memory"]) or ""
    assert 'logger.debug(f"Memory measurement failed' not in memory_source
    assert (
        'raise RuntimeError(\n                    f"Memory measurement failed for sample {i}."'
        in memory_source
        or 'raise RuntimeError(f"Memory measurement failed for sample {i}.")'
        in memory_source
    )
    assert "_memory_validation_error(" in memory_source


def test_metrics_validator_model_errors_raise_instead_of_debug_fallback() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_support.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_model":
            target = node
            break

    assert target is not None, "validate_model not found"
    source = ast.get_source_segment(text, target) or ""
    assert "Could not count model parameters" not in source
    assert "Model parameter iteration failed" in source
    assert (
        "except (AttributeError, TypeError, RuntimeError, ValueError) as exc:" in source
    )
