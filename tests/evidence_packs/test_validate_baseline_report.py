from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from scripts.evidence_packs.python import task_tools


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _baseline_payload() -> dict[str, Any]:
    return {
        "edit": {"name": "noop"},
        "meta": {"adapter": "hf_causal"},
        "context": {
            "profile": "ci",
            "auto": {"tier": "balanced"},
            "assurance": {"mode": "off"},
        },
        "evaluation_windows": {
            "preview": {"window_ids": ["p1"], "input_ids": [[1, 2]]},
            "final": {"window_ids": ["f1"], "input_ids": [[3, 4]]},
        },
    }


def _validate(
    path: Path,
    *,
    expected_preview_n: int | None = None,
    expected_final_n: int | None = None,
    expected_model_identity: dict[str, str] | None = None,
) -> int:
    args = ["validate-baseline-report", str(path), "hf_causal", "ci", "balanced"]
    if expected_preview_n is not None:
        args.extend(["--expected-preview-n", str(expected_preview_n)])
    if expected_final_n is not None:
        args.extend(["--expected-final-n", str(expected_final_n)])
    if expected_model_identity is not None:
        args.extend(
            [
                "--expected-model-identity-json",
                json.dumps(expected_model_identity, sort_keys=True),
            ]
        )
    return task_tools.main(args)


def test_validate_baseline_report_accepts_expected_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline.report.json"
    _write_report(report_path, _baseline_payload())

    assert _validate(report_path) == 0


def test_checkpoint_identity_is_bound_to_tree_contents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"gpt2"}')
    (checkpoint / "model.safetensors").write_bytes(b"packed-weights")

    assert task_tools.main(["checkpoint-identity", str(checkpoint)]) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["kind"] == "local_checkpoint_tree"

    (checkpoint / "model.safetensors").write_bytes(b"tampered-weights")
    assert task_tools.main(["checkpoint-identity", str(checkpoint)]) == 0
    after = json.loads(capsys.readouterr().out)

    assert after["sha256"] != before["sha256"]


def test_reusable_baseline_requires_exact_typed_model_identity(tmp_path: Path) -> None:
    identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    payload = _baseline_payload()
    payload["meta"]["model_identity"] = identity
    report_path = tmp_path / "baseline.report.json"
    _write_report(report_path, payload)

    assert _validate(report_path, expected_model_identity=identity) == 0
    wrong = {**identity, "sha256": "sha256:" + "b" * 64}
    assert _validate(report_path, expected_model_identity=wrong) == 1
    del payload["meta"]["model_identity"]
    _write_report(report_path, payload)
    assert _validate(report_path, expected_model_identity=identity) == 1


def test_generated_model_profile_is_deterministic_and_precedes_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "n_embd": 64,
                "n_layer": 2,
                "n_head": 4,
                "n_positions": 128,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"packed-weights")

    assert task_tools.main(["write-model-profile", str(checkpoint), "local/model"]) == 0
    profile_path = checkpoint / "model_profile.json"
    first_profile = profile_path.read_bytes()
    first_digest = checkpoint_tree_sha256(checkpoint)

    profile_path.unlink()
    assert task_tools.main(["write-model-profile", str(checkpoint), "local/model"]) == 0

    assert profile_path.read_bytes() == first_profile
    assert checkpoint_tree_sha256(checkpoint) == first_digest


def test_validate_baseline_report_rejects_window_count_mismatch(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "baseline.report.json"
    _write_report(report_path, _baseline_payload())

    assert _validate(report_path, expected_preview_n=400, expected_final_n=400) == 1


def test_validate_baseline_report_requires_expected_metadata(
    tmp_path: Path,
) -> None:
    for field_path in (
        ("meta", "adapter"),
        ("context", "profile"),
        ("context", "auto"),
        ("context", "auto", "tier"),
    ):
        payload: dict[str, Any] = _baseline_payload()
        node: Any = payload
        for key in field_path[:-1]:
            assert isinstance(node, dict)
            node = node[key]
        assert isinstance(node, dict)
        node.pop(field_path[-1], None)

        report_path = tmp_path / ("baseline-" + "-".join(field_path) + ".json")
        _write_report(report_path, payload)

        assert _validate(report_path) == 1
