from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZER = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "lm_eval_harness"
    / "normalize_lm_eval_results.py"
)
RUNNER = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "lm_eval_harness"
    / "run_tiny_lm_eval_sidecar.sh"
)


def test_lm_eval_sidecar_runner_wires_preflight_and_lane_label() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    text = RUNNER.read_text(encoding="utf-8")
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_lane_artifact_label" in text
    assert "--lane-label" in text
    assert "lm_eval_sidecar_summary.json" in text
    assert "run_summary.txt" in text
    assert "LM Eval sidecar run complete" in text
    assert 'write_run_summary "success"' in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "integration_log_kv" in text
    assert '} > "$report_out/run_command.txt"' not in text


def _write_lm_eval_report(path: Path, word_perplexity: float) -> None:
    payload = {
        "results": {
            "wikitext": {
                "alias": "wikitext",
                "sample_len": 1,
                "word_perplexity,none": word_perplexity,
                "word_perplexity_stderr,none": "N/A",
            }
        },
        "versions": {"wikitext": 2.0},
        "n-samples": {"wikitext": {"original": 62, "effective": 1}},
        "higher_is_better": {"wikitext": {"word_perplexity": False}},
        "config": {
            "device": "cpu",
            "limit": 1.0,
            "batch_size": "1",
            "model_sha": "abc123",
            "random_seed": 0,
            "numpy_seed": 1234,
            "torch_seed": 1234,
            "fewshot_seed": 1234,
        },
        "git_hash": "local",
        "lm_eval_version": "0.4.12",
        "model_source": "hf",
        "model_name": "tiny-model",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_lm_eval_sidecar_normalizer_preserves_raw_metric_keys(
    tmp_path: Path,
) -> None:
    baseline_json = tmp_path / "baseline.json"
    subject_json = tmp_path / "subject.json"
    output_json = tmp_path / "summary.json"
    command_log = tmp_path / "run_command.txt"
    command_log.write_text("commands\n", encoding="utf-8")
    _write_lm_eval_report(baseline_json, 100.0)
    _write_lm_eval_report(subject_json, 125.0)

    result = subprocess.run(
        [
            sys.executable,
            str(NORMALIZER),
            "--baseline-json",
            str(baseline_json),
            "--baseline-model",
            "baseline",
            "--subject-json",
            str(subject_json),
            "--subject-model",
            "subject",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
            "--device",
            "cpu",
            "--lane-label",
            "cpu-host-off",
            "--command-log",
            str(command_log),
            "--output",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    task = summary["baseline"]["tasks"]["wikitext"]

    assert task["metrics"]["word_perplexity,none"] == 100.0
    assert task["metric_aliases"]["word_perplexity,none"] == "word_perplexity"
    assert task["stderrs"]["word_perplexity,none"] == "N/A"
    assert summary["baseline"]["seeds"]["torch"] == 1234
    assert summary["lane_artifact_label"] == "cpu-host-off"
    assert summary["command_log"] == str(command_log)
    assert (
        summary["comparison"]["wikitext"]["word_perplexity,none"][
            "subject_minus_baseline"
        ]
        == 25.0
    )
    assert (
        summary["comparison"]["wikitext"]["word_perplexity,none"][
            "subject_over_baseline"
        ]
        == 1.25
    )


def test_lm_eval_sidecar_normalizer_rejects_missing_results(
    tmp_path: Path,
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(NORMALIZER),
            "--baseline-json",
            str(bad_json),
            "--baseline-model",
            "baseline",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
            "--device",
            "cpu",
            "--lane-label",
            "cpu-host-off",
            "--output",
            str(tmp_path / "summary.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "lacks a results object" in result.stderr
