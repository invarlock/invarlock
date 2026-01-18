from __future__ import annotations

from pathlib import Path


def test_proof_pack_does_not_depend_on_lmeval() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    # Keep the check scoped to the execution path + task graph.
    paths = [
        repo_root / "scripts/proof_packs/run_pack.sh",
        repo_root / "scripts/proof_packs/run_suite.sh",
        repo_root / "scripts/proof_packs/lib/validation_suite.sh",
        repo_root / "scripts/proof_packs/lib/task_serialization.sh",
        repo_root / "scripts/proof_packs/lib/queue_manager.sh",
        repo_root / "scripts/proof_packs/lib/task_functions.sh",
        repo_root / "scripts/proof_packs/lib/result_compiler.sh",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "lm_eval" not in combined
    assert "lmeval_runner.sh" not in combined
    assert "EVAL_BASELINE" not in combined
    assert "EVAL_EDIT" not in combined
    assert "EVAL_SINGLE_BENCHMARK" not in combined
