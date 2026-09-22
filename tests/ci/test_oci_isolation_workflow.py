"""Require actual OCI isolation execution and retained outcomes in CI."""

from pathlib import Path

import pytest
import yaml


def test_front_door_runs_isolation_cases_and_retains_failed_outcomes():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-front-door-smoke.yml").read_text()
    )
    steps = workflow["jobs"]["engine-smoke"]["steps"]
    journey = next(
        step for step in steps if "signed canary journey" in step.get("name", "")
    )
    assert "tests/integration/test_oci_isolation.py" in journey["run"]
    assert "tests/integration/test_container_native_judge_journey.py" in journey["run"]
    assert "INVARLOCK_OCI_ISOLATION_RESULTS=" in journey["run"]
    upload = next(
        step for step in steps if step.get("name") == "Retain OCI isolation outcomes"
    )
    assert "always()" in upload["if"]
    assert upload["with"]["path"] == "artifacts/oci-isolation"


def test_front_door_requires_both_engines_without_skip_or_shortened_suite():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-front-door-smoke.yml").read_text()
    )
    job = workflow["jobs"]["engine-smoke"]
    assert job["strategy"] == {
        "fail-fast": False,
        "matrix": {"engine": ["docker", "podman"]},
    }
    assert job["env"]["CONTAINER_ENGINE"] == "${{ matrix.engine }}"
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    prepare = next(
        step for step in steps if step.get("name") == "Prepare selected engine"
    )
    assert "sudo apt-get install --yes podman" in prepare["run"]
    assert '"$CONTAINER_ENGINE" version' in prepare["run"]
    assert '"$CONTAINER_ENGINE" info' in prepare["run"]
    journey = next(
        step for step in steps if "signed canary journey" in step.get("name", "")
    )
    assert 'INVARLOCK_CONTAINER_ENGINE="$CONTAINER_ENGINE"' in journey["run"]
    assert "tests/integration/test_container_front_door_journey.py" in journey["run"]
    assert "tests/integration/test_oci_isolation.py" in journey["run"]
    assert "tests/integration/test_container_native_judge_journey.py" in journey["run"]
    assert not any(step.get("continue-on-error", False) for step in steps)
    upload = next(
        step for step in steps if step.get("name") == "Retain OCI isolation outcomes"
    )
    assert upload["with"]["name"] == "oci-isolation-outcomes-${{ matrix.engine }}"


def test_front_door_defines_exact_match_and_likelihood_canary_requests(tmp_path):
    import json

    from invarlock.core.evaluation_request import load_evaluation_request
    from tests.integration.test_container_front_door_journey import _request

    for metric in ("exact_match", "normalized_nll_per_utf8_byte"):
        workspace = tmp_path / metric
        workspace.mkdir()
        (workspace / "models/tiny-hf").mkdir(parents=True)
        request = _request(
            workspace,
            checkpoint_digest="a" * 64,
            tokenizer_digest="b" * 64,
            output="evidence",
            metric=metric,
        )
        parsed = load_evaluation_request(request)
        assert parsed.comparison.metric == metric
        dataset = json.loads((workspace / "inputs/records.jsonl").read_bytes())
        if metric == "normalized_nll_per_utf8_byte":
            assert dataset["expected"] == " token-4 token-5"
            assert parsed.comparison.baseline.runtime.settings["max_output_tokens"] == 2
        policy = json.loads((workspace / "inputs/policy.json").read_bytes())
        expected = (
            {"delta_min_pp": -100.0} if metric == "exact_match" else {"ratio_max": 1.1}
        )
        assert policy["resolved_policy"]["metrics"] == {metric: expected}


@pytest.mark.parametrize(
    "outcome",
    ["complete", "skipped", "failure", "missing-nll", "missing-judge", "empty"],
)
def test_parity_completion_rejects_partial_or_unsuccessful_execution(tmp_path, outcome):
    import subprocess
    from xml.etree import ElementTree

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-front-door-smoke.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["engine-smoke"]["steps"]
        if step.get("name") == "Require complete engine journey outcomes"
    )
    assert step["if"] == "${{ always() }}"
    suite = ElementTree.Element("testsuite")
    if outcome != "empty":
        for metric in ("exact_match", "normalized_nll_per_utf8_byte"):
            if outcome == "missing-nll" and metric != "exact_match":
                continue
            case = ElementTree.SubElement(suite, "testcase", name=f"journey[{metric}]")
            if outcome in ("skipped", "failure"):
                ElementTree.SubElement(case, outcome)
    if outcome not in ("empty", "missing-judge"):
        ElementTree.SubElement(
            suite,
            "testcase",
            name="test_native_judge_container_capture_evaluate_verify_report",
        )
    destination = tmp_path / "artifacts/oci-isolation/results.xml"
    destination.parent.mkdir(parents=True)
    ElementTree.ElementTree(suite).write(destination)
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (outcome == "complete"), result.stderr


def test_stable_smoke_status_requires_all_engine_lanes_to_succeed():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-front-door-smoke.yml").read_text()
    )
    summary = workflow["jobs"]["smoke"]
    assert summary["if"] == "${{ always() }}"
    assert summary["needs"] == ["engine-smoke"]
    assert "strategy" not in summary
    assert "name" not in summary
    step = summary["steps"][0]
    assert step["env"]["ENGINE_RESULT"] == "${{ needs.engine-smoke.result }}"
    assert step["run"] == 'test "$ENGINE_RESULT" = success'


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:2"])
def test_container_journey_device_selection_is_explicit(monkeypatch, device):
    from tests.integration.test_container_front_door_journey import _runtime_device

    monkeypatch.delenv("INVARLOCK_RUNTIME_DEVICE", raising=False)
    if device is not None:
        monkeypatch.setenv("INVARLOCK_RUNTIME_DEVICE", device)
    assert _runtime_device() == (device or "cpu")


@pytest.mark.parametrize("device", ["", "auto", "mps", "cuda:-1", "cuda:all"])
def test_container_journey_rejects_ambiguous_device_selection(monkeypatch, device):
    from tests.integration.test_container_front_door_journey import _runtime_device

    monkeypatch.setenv("INVARLOCK_RUNTIME_DEVICE", device)
    with pytest.raises(ValueError, match="must be cpu, cuda, or cuda:index"):
        _runtime_device()


@pytest.mark.parametrize(
    "device,observed,capability,changed,accepted",
    [
        ("cpu", "cpu", None, False, True),
        ("cuda", "cuda", "9.0", False, True),
        ("cuda:0", "cuda", "9.0", False, True),
        ("cuda", "cpu", None, False, False),
        ("cpu", "cuda", "9.0", False, False),
        ("cuda", "cuda", None, False, False),
        ("cuda", "cuda", "9.0", True, False),
    ],
)
def test_container_journey_requires_bound_observed_device_facts(
    device, observed, capability, changed, accepted
):
    import hashlib
    import json

    from tests.integration.test_container_front_door_journey import (
        _assert_runtime_device,
    )

    # Synthetic documents exercise assertion failures, not hardware qualification.
    receipt = json.dumps(
        {
            "plugin": {"name": "hf_transformers"},
            "device": {
                "device_kind": observed,
                "device_name": "CPU" if observed == "cpu" else "Test GPU",
                "compute_capability": capability,
            },
        }
    ).encode()
    image_digest = "sha256:" + "a" * 64
    manifest = json.dumps(
        {
            "execution_mode": "container",
            "outer_container": {"image_digest": image_digest},
            "runtime_provider": {
                "receipt": {"sha256": hashlib.sha256(receipt).hexdigest()}
            },
        }
    ).encode()
    if changed:
        receipt += b" "
    if accepted:
        assert (
            _assert_runtime_device(
                manifest, receipt, device=device, image_digest=image_digest
            )["device_kind"]
            == observed
        )
    else:
        with pytest.raises(AssertionError):
            _assert_runtime_device(
                manifest, receipt, device=device, image_digest=image_digest
            )
