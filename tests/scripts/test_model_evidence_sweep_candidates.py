from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

from tests.scripts._support_model_evidence_sweep import load_script_module


def test_promotion_gap_gpu_suite_covers_prepared_deferred_lanes() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.PROMOTION_GAP_GPU_SUITE,
            slugs=[],
            lane_ids=[],
            shard_index=0,
            shard_count=1,
        )
    }

    assert set(specs) == {
        "distilbert_base_uncased",
    }
    assert specs["distilbert_base_uncased"].adapter == "hf_mlm"
    assert specs["distilbert_base_uncased"].preset_relpath == (
        "configs/presets/masked_lm/distilbert_base_uncased_128.yaml"
    )


def test_support_matrix_backlog_gpu_suite_covers_prepared_candidate_rows() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.SUPPORT_MATRIX_BACKLOG_GPU_SUITE,
            slugs=[],
            lane_ids=[],
            shard_index=0,
            shard_count=1,
        )
    }

    assert set(specs) == {
        "google_gemma_4_12b_it",
        "google_gemma_4_e4b_it",
        "google_gemma_4_e2b_it_image_text",
        "qwen_qwen3_5_4b",
        "qwen_qwen3_5_2b",
        "huggingfacetb_smollm3_3b",
        "microsoft_phi_4_mini_instruct",
        "google_flan_t5_base",
        "qwen_qwen3_30b_a3b_instruct_2507",
        "mistralai_mixtral_8x7b_v0_1",
        "allenai_olmoe_1b_7b_0924",
        "google_gemma_4_26b_a4b_it",
    }
    assert specs["google_gemma_4_12b_it"].preset_relpath == (
        "configs/presets/multimodal/gemma4_12b_public_vqav2_256.yaml"
    )
    assert specs["google_gemma_4_12b_it"].adapter == "hf_multimodal"
    assert specs["google_gemma_4_12b_it"].verify_profile == "release"
    assert specs["google_gemma_4_12b_it"].vision_text_materialization is not None
    assert specs["google_gemma_4_12b_it"].vision_text_materialization["dataset"] == (
        "Multimodal-Fatima/VQAv2_sample_validation"
    )
    assert (
        specs["google_gemma_4_12b_it"].vision_text_materialization["max_samples"] == 800
    )
    for slug, preset in {
        "google_gemma_4_e4b_it": "configs/presets/multimodal/gemma4_e4b_public_vqav2_256.yaml",
        "google_gemma_4_e2b_it_image_text": "configs/presets/multimodal/gemma4_e2b_public_vqav2_256.yaml",
        "qwen_qwen3_5_4b": "configs/presets/multimodal/qwen3_5_4b_public_vqav2_256.yaml",
        "qwen_qwen3_5_2b": "configs/presets/multimodal/qwen3_5_2b_public_vqav2_256.yaml",
    }.items():
        assert specs[slug].preset_relpath == preset
        assert specs[slug].adapter == "hf_multimodal"
        assert specs[slug].verify_profile == "release"
        assert specs[slug].vision_text_materialization is not None
        assert specs[slug].vision_text_materialization["dataset"] == (
            "Multimodal-Fatima/VQAv2_sample_validation"
        )
    qwen4_prompt = str(
        specs["qwen_qwen3_5_4b"].vision_text_materialization["prompt_template"]
    )
    assert '{"answer":"short phrase"}' in qwen4_prompt
    assert "Do not explain or include thinking" in qwen4_prompt
    assert specs["microsoft_phi_4_mini_instruct"].preset_relpath == (
        "configs/presets/causal_lm/phi4_mini_512.yaml"
    )
    assert specs["google_flan_t5_base"].preset_relpath == (
        "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml"
    )
    assert specs["google_flan_t5_base"].adapter == "hf_seq2seq"
    assert specs["google_flan_t5_base"].verify_profile == "release"
    assert specs["mistralai_mixtral_8x7b_v0_1"].preset_relpath == (
        "configs/presets/causal_lm/mixtral_8x7b_512.yaml"
    )
    assert specs["mistralai_mixtral_8x7b_v0_1"].adapter == "hf_causal"
    assert specs["mistralai_mixtral_8x7b_v0_1"].verify_profile == "release"

    assert specs["google_gemma_4_e4b_it"].lane_id == "gemma4-e4b-image-text-hf"
    assert (
        specs["google_gemma_4_e2b_it_image_text"].lane_id == "gemma4-e2b-image-text-hf"
    )


def test_repo_mentioned_gpu_suite_phi4_host_dry_run_uses_lane_preset(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "candidate-phi4-reasoning-host"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "phi4_reasoning_plus_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    assert payload[0]["slug"] == "phi4_reasoning_plus_public"
    assert payload[0]["prefetch"][-1] == "microsoft/Phi-4-reasoning-plus"
    assert "--allow-remote-code" in payload[0]["evaluate"]
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert payload[0]["evaluate"][preset_idx] == str(
        repo_root / "configs/presets/causal_lm/phi4_reasoning_plus_512.yaml"
    )
    assert "--runtime-provenance" in payload[0]["verify"]

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == "repo-mentioned-gpu"
    assert manifest["lanes"][0]["slug"] == "phi4_reasoning_plus_public"


def test_support_matrix_backlog_gpu_suite_phi4_dry_run_uses_builtin_phi3_policy(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "candidate-phi4-mini-host"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "microsoft_phi_4_mini_instruct",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    assert payload[0]["slug"] == "microsoft_phi_4_mini_instruct"
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert payload[0]["evaluate"][preset_idx] == str(
        repo_root / "configs/presets/causal_lm/phi4_mini_512.yaml"
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == "support-matrix-backlog-gpu"
    assert manifest["lanes"][0]["slug"] == "microsoft_phi_4_mini_instruct"


def test_support_matrix_backlog_gemma_dry_run_materializes_public_vqav2(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "candidate-gemma-vqav2"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "google_gemma_4_12b_it",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    item = payload[0]
    assert item["slug"] == "google_gemma_4_12b_it"
    assert item["materialize_dataset"][0] == sys.executable
    assert "Multimodal-Fatima/VQAv2_sample_validation" in item["materialize_dataset"]
    assert "99487d2651df3799002b2fb3e455741744514a02" in item["materialize_dataset"]
    preset_idx = item["evaluate"].index("--preset") + 1
    assert item["evaluate"][preset_idx].endswith("prepared_preset.yaml")
    assert item["evaluate"][item["evaluate"].index("--profile") + 1] == "release"

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    lane = manifest["lanes"][0]
    assert lane["vision_text_materialization"]["dataset"] == (
        "Multimodal-Fatima/VQAv2_sample_validation"
    )
    assert lane["vision_text_materialization"]["max_samples"] == 800
    assert (
        '{"answer":"short phrase"}'
        in lane["vision_text_materialization"]["prompt_template"]
    )


def test_lane_requires_remote_code_uses_preset_model_flag() -> None:
    mod = load_script_module("model_evidence_sweep")
    phi4 = next(
        lane
        for lane in mod.SUITES[mod.REPO_MENTIONED_GPU_SUITE]
        if lane.slug == "phi4_reasoning_plus_public"
    )
    qwen = next(
        lane
        for lane in mod.SUITES[mod.REPO_MENTIONED_GPU_SUITE]
        if lane.slug == "qwen2_7b_public"
    )
    phi4_mini = next(
        lane
        for lane in mod.SUITES[mod.SUPPORT_MATRIX_BACKLOG_GPU_SUITE]
        if lane.slug == "microsoft_phi_4_mini_instruct"
    )

    assert mod.lane_requires_remote_code(phi4) is True
    assert mod.lane_requires_remote_code(phi4_mini) is False
    assert mod.lane_requires_remote_code(qwen) is False


def test_model_evidence_sweep_marks_remote_code_prefetch_as_skipped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "remote-code-fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  echo "Loading this model requires you to execute custom code. Please set trust_remote_code=True." >&2
  exit 1
fi
echo "unexpected invocation" >&2
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    output_root = tmp_path / "evidence-host-remote-code"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "phi4_reasoning_plus_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    summary_tsv = (output_root / "summary.tsv").read_text(encoding="utf-8")
    assert "skipped\tremote_code_required" in summary_tsv


def test_run_lane_sets_remote_code_env_for_matching_preset(tmp_path: Path) -> None:
    mod = load_script_module("model_evidence_sweep")
    import evidence_workflows.workflow_runner as workflow_runner

    spec = next(
        lane
        for lane in mod.SUITES[mod.REPO_MENTIONED_GPU_SUITE]
        if lane.slug == "phi4_reasoning_plus_public"
    )
    output_root = tmp_path / "evidence-remote-code"
    calls: list[tuple[list[str], str | None]] = []
    real_completed = subprocess.CompletedProcess

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout,
        stderr,
        text: bool,
        check: bool,
    ):
        calls.append((cmd, env.get("INVARLOCK_ALLOW_REMOTE_CODE")))
        if "-c" in cmd:
            return real_completed(cmd, 0)
        if cmd[:3] == [sys.executable, "-m", "invarlock"] and "evaluate" in cmd:
            report_dir = Path(cmd[cmd.index("--report-out") + 1])
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
            return real_completed(cmd, 0)
        if cmd[:3] == [sys.executable, "-m", "invarlock"] and "verify" in cmd:
            return real_completed(cmd, 0)
        raise AssertionError(f"unexpected command: {cmd}")

    original_run = workflow_runner.subprocess.run
    workflow_runner.subprocess.run = fake_run
    try:
        execution_root = mod._execution_root(output_root, execution_mode="host")
        result = mod.run_lane(
            spec,
            python_exe=sys.executable,
            profile=None,
            device="cuda",
            execution_mode="host",
            output_root=output_root,
            execution_root=execution_root,
            env={"PYTHONPATH": "src"},
        )
    finally:
        workflow_runner.subprocess.run = original_run

    assert result.evaluate_exit == 0
    assert result.verify_exit == 0
    assert calls
    assert all(flag == "1" for _cmd, flag in calls)
