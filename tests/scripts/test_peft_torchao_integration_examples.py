from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS_DIR = REPO_ROOT / "examples" / "integrations"
SOURCE_MATRIX = INTEGRATIONS_DIR / "source_matrix.json"
PEFT_DIR = REPO_ROOT / "examples" / "integrations" / "peft_lora"
TORCHAO_DIR = REPO_ROOT / "examples" / "integrations" / "torchao_int8_runtime"

EXAMPLE_RUNNERS = [
    INTEGRATIONS_DIR / "awq" / "run_tiny_awq.sh",
    INTEGRATIONS_DIR / "compressed_tensors" / "run_tiny_hf_ct.sh",
    INTEGRATIONS_DIR / "gptqmodel" / "run_tiny_gptqmodel.sh",
    INTEGRATIONS_DIR / "hf_bnb" / "run_tiny_hf_bnb_8bit.sh",
    INTEGRATIONS_DIR / "hqq" / "run_tiny_hf_hqq.sh",
    INTEGRATIONS_DIR / "lm_eval_harness" / "run_tiny_lm_eval_sidecar.sh",
    INTEGRATIONS_DIR / "peft_lora" / "run_tiny_peft_lora.sh",
    INTEGRATIONS_DIR / "quanto" / "run_tiny_hf_quanto.sh",
    INTEGRATIONS_DIR / "torchao_int8_runtime" / "run_tiny_hf_torchao_int8.sh",
]

README_EXAMPLES = [
    "awq",
    "compressed_tensors",
    "gptqmodel",
    "hf_bnb",
    "hqq",
    "lm_eval_harness",
    "peft_lora",
    "quanto",
    "torchao_int8_runtime",
]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_matrix() -> dict[str, dict[str, object]]:
    payload = json.loads(SOURCE_MATRIX.read_text(encoding="utf-8"))
    assert payload["schema"] == "invarlock.integration_source_matrix.v1"
    entries = payload["entries"]
    assert isinstance(entries, list)
    return {entry["target"]: entry for entry in entries}


def _write_matrix_artifact_set(report_dir: Path) -> None:
    report_dir.mkdir(parents=True)
    (report_dir / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": True, "reason": "ok"},
                "results": [
                    {
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "container",
                                "verified": True,
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "runtime.manifest.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "evaluation.html").write_text("<html></html>\n", encoding="utf-8")
    (report_dir / "backend_inventory.json").write_text(
        '{"backend": "hqq"}\n', encoding="utf-8"
    )
    (report_dir / "lane_artifact.json").write_text(
        json.dumps({"lane_artifact_label": "cuda-container-strict"}) + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_command.txt").write_text(
        "wrapper: run_tiny_hf_hqq.sh --lane cuda\n", encoding="utf-8"
    )
    (report_dir / "run_summary.txt").write_text(
        "status: success\n"
        "lane_artifact_label: cuda-container-strict\n"
        "verify_status: ok\n"
        "verify_runtime_provenance_declared: container\n"
        "verify_runtime_provenance_verified: true\n",
        encoding="utf-8",
    )
    (report_dir / "checkpoint_refs.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "adapter_runtime_summary.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "fixture_summary.json").write_text("{}\n", encoding="utf-8")


def _write_test_source_matrix(repo_root: Path) -> Path:
    matrix_path = repo_root / "examples" / "integrations" / "source_matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "invarlock.integration_source_matrix.v1",
                "entries": [
                    {
                        "target": "hqq",
                        "readme": "examples/integrations/hqq/README.md",
                        "runner": "examples/integrations/hqq/run_tiny_hf_hqq.sh",
                        "report_path": "reports/tiny-hf-hqq/<artifact-lane>",
                        "lane": "cuda-container-strict",
                        "expected": {
                            "lane_artifact_label": "cuda-container-strict",
                            "verify_status": "ok",
                            "runtime_provenance_declared": "container",
                            "runtime_provenance_verified": True,
                        },
                        "required_artifacts": [
                            "evaluation.report.json",
                            "verify.json",
                            "runtime.manifest.json",
                            "evaluation.html",
                            "backend_inventory.json",
                            "lane_artifact.json",
                            "run_command.txt",
                            "run_summary.txt",
                            "checkpoint_refs.json",
                            "adapter_runtime_summary.json",
                            "fixture_summary.json",
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix_path


def test_peft_lora_runner_wires_local_fixture() -> None:
    runner = PEFT_DIR / "run_tiny_peft_lora.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--runtime-provenance" in text
    assert "--device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_integration_example_readmes_document_run_lanes() -> None:
    expected_headings = {
        "awq": ["### cuda-host-off lane", "### cuda-container-strict lane"],
        "compressed_tensors": [
            "### cpu-host-off lane",
            "### cuda-container-strict lane",
        ],
        "gptqmodel": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "hf_bnb": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "hqq": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "lm_eval_harness": [
            "### cpu-host-off lane",
            "### cuda-host-off lane",
            "### mps-host-off lane",
        ],
        "peft_lora": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "quanto": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "torchao_int8_runtime": [
            "### cpu-host-off lane",
            "### cuda-container-strict lane",
        ],
    }

    integrations = REPO_ROOT / "examples" / "integrations"
    for example, headings in expected_headings.items():
        text = (integrations / example / "README.md").read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text, f"{example} missing run lane {heading!r}"

    for example in [
        "compressed_tensors",
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "peft_lora",
        "quanto",
        "torchao_int8_runtime",
    ]:
        text = (integrations / example / "README.md").read_text(encoding="utf-8")
        assert "`cpu-host-off`" in text
        assert "`cuda-host-off`" in text
        assert "`cuda-container-strict`" in text
        assert "--lane host" in text
        assert "--lane cuda" in text
        assert "--device cpu" in text
        assert "--device cuda" in text
        assert text.index("`cuda-container-strict`") < text.index("`cuda-host-off`")
        assert text.index("`cuda-container-strict`") < text.index("`cpu-host-off`")
        assert "run_summary.txt" in text
        assert "verifier status, runtime provenance status" in text
        assert "shared completion block" in text

    awq_text = (integrations / "awq" / "README.md").read_text(encoding="utf-8")
    assert "`cpu-host-off`" not in awq_text
    assert "`cuda-host-off`" in awq_text
    assert "`cuda-container-strict`" in awq_text
    assert "--lane host" in awq_text
    assert "--lane host --device cuda" in awq_text
    assert "--lane cuda" in awq_text
    assert "--device cpu" not in awq_text
    assert awq_text.index("`cuda-container-strict`") < awq_text.index("`cuda-host-off`")
    assert "run_summary.txt" in awq_text
    assert "verifier status, runtime provenance status" in awq_text

    lm_eval_text = (integrations / "lm_eval_harness" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "`mps-host-off`" in lm_eval_text
    assert "--device mps" in lm_eval_text
    assert lm_eval_text.index("`cuda-container-strict`") < lm_eval_text.index(
        "`cuda-host-off`"
    )
    assert "primary evidence" in lm_eval_text
    assert "run_summary.txt" in lm_eval_text
    assert "verifier status, runtime provenance status" in lm_eval_text


def test_integration_runners_default_reports_are_lane_scoped() -> None:
    for runner in EXAMPLE_RUNNERS:
        subprocess.run(["bash", "-n", str(runner)], check=True)

        text = runner.read_text(encoding="utf-8")
        assert "<artifact-lane>" in text, f"{runner} missing help contract"
        assert "report_out_was_default=1" in text, f"{runner} missing default flag"
        assert "report_out_was_default=0" in text, f"{runner} missing override flag"
        assert "integration_lane_report_out" in text, f"{runner} missing lane output"


def test_integration_readmes_use_run_lane_subsections() -> None:
    for example in README_EXAMPLES:
        readme = INTEGRATIONS_DIR / example / "README.md"
        text = readme.read_text(encoding="utf-8")

        assert "## Run\n\n## Lane Support" not in text
        assert "## Run\n\n### Lane Support" in text


def test_integration_readme_report_paths_are_lane_scoped() -> None:
    for example in README_EXAMPLES:
        readme = INTEGRATIONS_DIR / example / "README.md"
        text = readme.read_text(encoding="utf-8")
        report_paths = re.findall(r"`(reports/tiny-[^`]+)`", text)

        assert report_paths, f"{example} README has no report artifact paths"
        for report_path in report_paths:
            assert "/<artifact-lane>/" in report_path, (
                f"{example} report path is not lane-scoped: {report_path}"
            )


def test_strict_evidence_claim_readmes_have_artifact_source_matrix() -> None:
    matrix_entries = _load_source_matrix()
    common_artifacts = {
        "evaluation.report.json",
        "verify.json",
        "runtime.manifest.json",
        "evaluation.html",
        "lane_artifact.json",
        "run_command.txt",
        "run_summary.txt",
    }
    quantized_strict_targets = {
        "awq",
        "compressed_tensors",
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "quanto",
        "torchao_int8_runtime",
    }
    target_provenance_artifacts = {
        "awq": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "compressed_tensors": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
        "gptqmodel": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "hf_bnb": {"fixture_summary.json"},
        "hqq": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
        "peft_lora": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "quanto": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
        "torchao_int8_runtime": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
    }

    claimed_readmes = {}
    for readme in INTEGRATIONS_DIR.rglob("README.md"):
        text = readme.read_text(encoding="utf-8")
        if "strict container evidence is verified" in text:
            claimed_readmes[readme.parent.name] = text

    assert set(claimed_readmes) == set(matrix_entries)

    for example, text in claimed_readmes.items():
        entry = matrix_entries[example]
        readme = INTEGRATIONS_DIR / example / "README.md"
        runner = Path(entry["runner"])
        runtime_image = entry["runtime_image"]
        expected = entry["expected"]
        required_artifacts = set(entry["required_artifacts"])
        provenance_artifacts = set(entry["provenance_artifacts"])

        assert Path(entry["readme"]) == readme.relative_to(REPO_ROOT)
        assert (REPO_ROOT / runner).is_file()
        assert entry["strict_claim_phrase"] in text
        assert entry["lane"] == "cuda-container-strict"
        assert entry["command_shape"] == "--lane cuda"
        assert "`cuda-container-strict`" in text
        assert str(entry["report_path"]) in text
        assert runtime_image["source_command"] in text
        assert runtime_image["digest_source"] == "runtime.manifest.json"
        assert expected["lane_artifact_label"] == "cuda-container-strict"
        assert expected["verify_status"] == "ok"
        assert expected["runtime_provenance_declared"] == "container"
        assert expected["runtime_provenance_verified"] is True
        assert common_artifacts <= required_artifacts
        assert provenance_artifacts == target_provenance_artifacts[example]
        assert provenance_artifacts <= required_artifacts

        for artifact in required_artifacts:
            assert artifact in text

        runner_text = (REPO_ROOT / runner).read_text(encoding="utf-8")
        for artifact in provenance_artifacts:
            assert artifact in runner_text

        if example in quantized_strict_targets:
            assert "backend_inventory.json" in required_artifacts
            assert (
                entry["runner_enforcement"]["backend_inventory"]
                == "--require-backend-inventory"
            )
            assert "--require-backend-inventory" in runner_text
            assert 'lane_artifact_label" == "cuda-container-strict"' in runner_text
        else:
            assert "backend_inventory.json" not in required_artifacts
            assert entry["runner_enforcement"] == {}


def test_source_matrix_artifact_validator_accepts_complete_artifacts(
    tmp_path: Path,
) -> None:
    validator = _load_module(
        INTEGRATIONS_DIR / "_shared" / "validate_source_matrix_artifacts.py",
        "source_matrix_artifact_validator",
    )
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = (
        tmp_path
        / "examples"
        / "integrations"
        / "hqq"
        / "reports"
        / "tiny-hf-hqq"
        / "cuda-container-strict"
    )
    _write_matrix_artifact_set(report_dir)

    selected, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
    )

    assert selected == ["hqq"]
    assert issues == []


def test_source_matrix_artifact_validator_reports_artifact_and_status_mismatches(
    tmp_path: Path,
) -> None:
    validator = _load_module(
        INTEGRATIONS_DIR / "_shared" / "validate_source_matrix_artifacts.py",
        "source_matrix_artifact_validator_mismatch",
    )
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = (
        tmp_path
        / "examples"
        / "integrations"
        / "hqq"
        / "reports"
        / "tiny-hf-hqq"
        / "cuda-container-strict"
    )
    _write_matrix_artifact_set(report_dir)
    (report_dir / "backend_inventory.json").unlink()
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": False, "reason": "policy_fail"},
                "results": [
                    {
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "host",
                                "verified": False,
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
    )
    messages = [issue.message for issue in issues]

    assert "required artifact is missing" in messages
    assert any("verify status mismatch" in message for message in messages)
    assert any(
        "runtime provenance declared mode mismatch" in message for message in messages
    )
    assert any(
        "runtime provenance verified flag mismatch" in message for message in messages
    )


def test_materialized_subject_readmes_define_evidence_boundary() -> None:
    expectations = {
        "awq": ["`hf_awq`", "`external_edit_summary.json`"],
        "compressed_tensors": ["`hf_ct`", "`adapter_runtime_summary.json`"],
        "gptqmodel": ["`hf_gptq`", "`external_edit_summary.json`"],
        "peft_lora": ["`hf_causal`", "`external_edit_summary.json`"],
    }

    for example, phrases in expectations.items():
        text = (INTEGRATIONS_DIR / example / "README.md").read_text(encoding="utf-8")

        assert "## Evidence Boundary" in text, f"{example} lacks evidence boundary"
        assert "The subject checkpoint is materialized before" in text
        assert "verifier result for that\nproduced subject" in text
        for phrase in phrases:
            assert phrase in text


def test_torchao_readme_documents_backend_inventory_sidecar() -> None:
    text = (TORCHAO_DIR / "README.md").read_text(encoding="utf-8")

    assert (
        "`reports/tiny-hf-torchao-int8/<artifact-lane>/backend_inventory.json`" in text
    )
    assert "The shell runner relies on InvarLock report persistence to emit" in text
    assert "`backend_inventory.json` when adapter provenance is available" in text
    assert "adapter provenance is available" in text


def test_shared_example_docs_scope_source_archives_and_image_digests() -> None:
    shared_readme = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "README.md"
    ).read_text(encoding="utf-8")
    image_readme = (
        REPO_ROOT / "examples" / "integrations" / "_runtime_images" / "README.md"
    ).read_text(encoding="utf-8")

    assert "Use `--committed` when sharing an archive" in shared_readme
    assert "Use `--include-worktree` only" in shared_readme
    assert "deliberately including local changes" in shared_readme
    assert "may produce a different image digest" in image_readme
    assert "digest recorded in `runtime.manifest.json`" in image_readme


def test_peft_readme_scopes_strict_evidence_to_tiny_runtime() -> None:
    text = (PEFT_DIR / "README.md").read_text(encoding="utf-8")

    assert "strict container evidence is verified on CUDA for this tiny" in text
    assert "scoped to the configured tiny merged dense checkpoint" in text
    assert "shared integration evidence" in text


def test_integration_example_docs_use_canonical_lane_wording() -> None:
    scanned_paths = list((REPO_ROOT / "examples" / "integrations").rglob("README.md"))
    scanned_paths.extend(
        [
            REPO_ROOT
            / "examples"
            / "integrations"
            / "_shared"
            / "run_invarlock_compare.sh",
            REPO_ROOT / "examples" / "integrations" / "_shared" / "preflight.sh",
        ]
    )

    stale_phrases = [
        "host" + "/" + "off",
        "CPU host" + "/" + "off",
        "CUDA host" + "/" + "off",
        "CUDA" + "/" + "container",
        "strict-cuda" + "-container",
        "gpu-host" + "-off",
    ]

    for path in scanned_paths:
        text = path.read_text(encoding="utf-8")
        for phrase in stale_phrases:
            assert phrase not in text, f"{path} contains stale lane wording {phrase!r}"


def test_shared_compare_wrapper_checks_report_materialization() -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    subprocess.run(["bash", "-n", str(wrapper)], check=True)

    text = wrapper.read_text(encoding="utf-8")
    assert "Evaluate completed but did not write the expected report" in text
    assert "Evaluate completed but did not write the required backend inventory" in text
    assert '[[ ! -s "$report_json" ]]' in text
    assert (
        '[[ "$require_backend_inventory" -eq 1 && ! -s "$backend_inventory_json" ]]'
        in text
    )
    assert 'rm -f "$report_json" "$verify_json"' in text
    assert 'rm -f "$report_json" "$verify_json" "$backend_inventory_json"' in text
    assert 'CLI=("$PYTHON_BIN" -m invarlock)' in text
    assert '"${CLI[@]}" evaluate' in text
    assert '"${CLI[@]}" verify' in text
    assert "--lane MODE" in text
    assert 'execution_mode="container"' in text
    assert 'runtime_provenance="container"' in text
    assert 'device="cuda"' in text
    assert "lane_artifact.json" in text
    assert "lane_artifact_label" in text
    assert "run_summary.txt" in text
    assert "--require-backend-inventory" in text
    assert "InvarLock integration run complete" in text
    assert "InvarLock integration run failed" in text
    assert 'write_run_summary "success"' in text
    assert "emit_verify_summary_fields" in text
    assert "verify_status" in text
    assert "verify_runtime_provenance_status" in text
    assert "runtime provenance:" in text
    assert "integration_log_step" in text
    assert "integration_log_kv" in text
    assert "integration_default_host_device" in text
    assert "integration_lane_artifact_label" in text
    assert "integration_lane_report_out" in text
    assert "report_out_was_default=1" in text
    assert "report_out_was_default=0" in text
    assert "<artifact-lane>" in text


def test_shared_compare_wrapper_enforces_backend_inventory_sidecar(
    tmp_path: Path,
) -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REAL_PYTHON = {sys.executable!r}

if sys.argv[1:3] != ["-m", "invarlock"]:
    raise SystemExit(subprocess.run([REAL_PYTHON, *sys.argv[1:]]).returncode)

args = sys.argv[3:]
command = args[0]
if command == "evaluate":
    report_out = Path(args[args.index("--report-out") + 1])
    report_out.mkdir(parents=True, exist_ok=True)
    (report_out / "evaluation.report.json").write_text(
        json.dumps({{"schema": "fake", "results": []}}) + "\\n",
        encoding="utf-8",
    )
    if os.environ.get("FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY") == "1":
        (report_out / "backend_inventory.json").write_text(
            json.dumps({{"adapter": "fake"}}) + "\\n",
            encoding="utf-8",
        )
    raise SystemExit(0)
if command == "verify":
    payload = {{
        "summary": {{"ok": True, "reason": "ok"}},
        "results": [
            {{
                "verification": {{
                    "runtime_provenance": {{
                        "declared_mode": "container",
                        "status": "verified",
                        "verified": True,
                    }}
                }}
            }}
        ],
    }}
    print(json.dumps(payload))
    raise SystemExit(0)
if command == "report" and args[1] == "html":
    output = Path(args[args.index("-o") + 1])
    output.write_text("<html></html>\\n", encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(f"unexpected fake invarlock command: {{args!r}}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    base_cmd = [
        str(wrapper),
        "--baseline",
        "dense",
        "--subject",
        "quant",
        "--report-out",
        str(tmp_path / "reports"),
        "--lane",
        "cuda",
        "--require-backend-inventory",
        "--no-html",
    ]
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)

    missing = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing.returncode == 1
    assert "required backend inventory" in missing.stderr
    assert (tmp_path / "reports" / "evaluation.report.json").is_file()
    assert not (tmp_path / "reports" / "backend_inventory.json").exists()

    env["FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY"] = "1"
    ok = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert ok.returncode == 0
    assert "InvarLock integration run complete" in ok.stdout
    assert (tmp_path / "reports" / "backend_inventory.json").is_file()
    assert (tmp_path / "reports" / "verify.json").is_file()
    assert "status: success" in (tmp_path / "reports" / "run_summary.txt").read_text(
        encoding="utf-8"
    )


def test_shared_source_archive_helper_avoids_macos_xattrs() -> None:
    helper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "create_source_archive.sh"
    )
    subprocess.run(["bash", "-n", str(helper)], check=True)

    text = helper.read_text(encoding="utf-8")
    assert 'git -C "$REPO_ROOT" archive --format=tar.gz' in text
    assert "--include-worktree" in text
    assert "COPYFILE_DISABLE=1" in text
    assert "--no-xattrs" in text
    assert "ls-files -z --cached --modified --others --exclude-standard" in text


def test_shared_expected_artifacts_documents_backend_inventory() -> None:
    text = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "expected-artifacts.md"
    ).read_text(encoding="utf-8")

    assert "`backend_inventory.json`" in text
    assert "`external_edit_summary.json`" in text
    assert "`adapter_runtime_summary.json`" in text
    assert "`fixture_summary.json`" in text
    assert "InvarLock report persistence" in text
    assert "adapter provenance is available" in text
    assert "reports/<target>/<artifact-lane>/evaluation.report.json" in text
    assert "--runtime-provenance container" in text
    assert "For the primary CUDA/container strict lane" in text


def test_shared_evidence_scope_documents_source_matrix_contract() -> None:
    text = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "evidence-scope.md"
    ).read_text(encoding="utf-8")

    assert "`source_matrix.json` is the source-controlled contract" in text
    assert "strict container evidence is verified" in text
    assert "`source_matrix.json` has an entry" in text
    assert "`checkpoint_refs.json`, `external_edit_summary.json`" in text
    assert "`adapter_runtime_summary.json`, and `fixture_summary.json`" in text


def test_shared_preflight_helper_defines_host_lane_contract() -> None:
    helper = REPO_ROOT / "examples" / "integrations" / "_shared" / "preflight.sh"
    subprocess.run(["bash", "-n", str(helper)], check=True)

    text = helper.read_text(encoding="utf-8")
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_preflight_gptqmodel_host_runtime" in text
    assert "integration_lane_artifact_label" in text
    assert "integration_lane_report_out" in text
    assert "integration_effective_assurance" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "integration_log_kv" in text
    assert "cuda-container-strict" in text
    assert "cuda-host-off" in text
    assert "cpu-host-off" in text


def test_torchao_runner_wires_local_fixture() -> None:
    runner = TORCHAO_DIR / "run_tiny_hf_torchao_int8.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "prepare_tiny_hf_torchao_fixture.py" in text
    assert "--model-dir" in text
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert '--baseline "$model_dir"' in text
    assert '--subject "$model_dir"' in text
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_torchao" in text
    assert "--edit-label torchao_int8_runtime_quantization" in text
    assert "adapter_runtime_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_torchao_readme_frames_hf_torchao_as_primary_path() -> None:
    text = (TORCHAO_DIR / "README.md").read_text(encoding="utf-8")

    assert "torchao Int8 Runtime Integration Example" in text
    assert "`hf_torchao` adapter" in text
    assert "strict container evidence is verified" in text
    assert "this tiny\n`hf_torchao` runtime-load example" in text
    assert "runnable evidence path is the `hf_torchao` subject" in text
    assert "scoped to the configured tiny HF checkpoint" in text
    assert "shared integration evidence" in text
    assert "run_tiny_hf_torchao_int8.sh" in text


def test_peft_lora_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-gpt2-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-gpt2-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "peft-lora-fixture-v1"


def test_peft_lora_helper_isolates_dense_lora_from_quantized_dispatch(
    monkeypatch,
) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example_dispatch",
    )

    class DenseModel:
        config = object()
        is_quantized = False

    calls = {"count": 0}

    def fake_get_peft_model(_model, _config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ImportError(
                "cannot import name 'AwqGEMMQuantLinear' from "
                "'gptqmodel.nn_modules.qlinear.gemm_awq'"
            )
        return "peft-model"

    monkeypatch.setattr(
        helper,
        "_disable_quantized_peft_dispatch_for_dense_example",
        lambda: True,
    )

    assert (
        helper._get_dense_peft_model(DenseModel(), object(), fake_get_peft_model)
        == "peft-model"
    )
    assert calls["count"] == 2


def test_torchao_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-llama-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-llama-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "torchao-fixture-v1"


def test_torchao_helper_prefers_non_deprecated_config_version() -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_config_helper",
    )

    class _ModernConfig:
        def __init__(self, *, version=None):
            self.version = version

    class _LegacyConfig:
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("unexpected keyword")
            self.version = 1

    modern = helper._torchao_int8_weight_only_config(_ModernConfig)
    legacy = helper._torchao_int8_weight_only_config(_LegacyConfig)

    assert modern.version == 2
    assert legacy.version == 1
