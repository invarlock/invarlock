from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from invarlock import clean_pruning_selection_runtime as pruning_runtime
from invarlock import clean_selection_runtime as selection_runtime
from invarlock.clean_pruning_selection_artifacts import (
    build_clean_pruning_candidate_report_binding,
)
from invarlock.clean_pruning_selection_common import (
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_selection.binding import build_candidate_report_binding
from invarlock.clean_selection.common import CleanSelectionEvidenceError
from scripts.evidence_packs.python.editing import validate_artifact
from tests.evidence_packs._support_clean_pruning_selection import (
    _candidate_mapping as _pruning_candidate_mapping,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _record as _pruning_record,
)
from tests.evidence_packs._support_clean_selection import (
    _candidate_mapping,
    _record,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _minimal_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", {"model_type": "gpt2"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 16})
    save_file({"weight": torch.ones((1, 1))}, path / "model.safetensors")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"bits":4,"bits":8}', "duplicate JSON key"),
        ('{"bits":NaN}', "non-standard JSON constant"),
        ("[4, 8]", "must be a JSON object"),
    ],
)
def test_transform_parameters_reject_ambiguous_or_non_object_json(
    raw: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_artifact._parse_cli_json_object(
            raw,
            argument_name="--parameters-json",
        )


def test_transform_cli_fails_closed_before_replay_on_ambiguous_parameters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        validate_artifact.main(
            [
                "validate_artifact.py",
                "transform",
                str(tmp_path / "artifact"),
                "--baseline",
                str(tmp_path / "baseline"),
                "--edit-type",
                "quant_rtn",
                "--parameters-json",
                '{"bits":4,"bits":8}',
                "--scope",
                "ffn",
            ]
        )

    assert "duplicate JSON key" in capsys.readouterr().err


def test_transformation_sidecar_cannot_mutate_an_authenticated_checkpoint_tree(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    baseline = tmp_path / "baseline"
    artifact.mkdir()
    baseline.mkdir()

    for output in (artifact / "replay.json", baseline / "replay.json"):
        with pytest.raises(ValueError, match="outside the baseline and artifact"):
            validate_artifact._write_transformation_replay_sidecar(
                output,
                payload={"ok": True},
                artifact_dir=artifact,
                baseline_dir=baseline,
            )
        assert not output.exists()


def test_transformation_sidecar_rejects_symlink_and_occupied_atomic_temp(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    baseline = tmp_path / "baseline"
    artifact.mkdir()
    baseline.mkdir()
    target = tmp_path / "target.json"
    target.write_text("old", encoding="utf-8")
    output = tmp_path / "replay.json"
    output.symlink_to(target.name)

    with pytest.raises(ValueError, match="regular file path"):
        validate_artifact._write_transformation_replay_sidecar(
            output,
            payload={"ok": True},
            artifact_dir=artifact,
            baseline_dir=baseline,
        )
    assert target.read_text(encoding="utf-8") == "old"

    output.unlink()
    occupied = tmp_path / ".replay.json.tmp"
    occupied.write_text("attacker-controlled", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly occupied"):
        validate_artifact._write_transformation_replay_sidecar(
            output,
            payload={"ok": True},
            artifact_dir=artifact,
            baseline_dir=baseline,
        )
    assert occupied.read_text(encoding="utf-8") == "attacker-controlled"
    assert not output.exists()


def test_failed_edit_save_preserves_previous_output_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "retained.txt").write_text("previous", encoding="utf-8")

    class _Tokenizer:
        def save_pretrained(self, path: Path) -> None:
            _write_json(path / "tokenizer_config.json", {})

    class _Model:
        def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization is True
            _write_json(path / "config.json", {})
            (path / "pytorch_model.bin").write_bytes(b"not-a-real-checkpoint")

    monkeypatch.setattr(
        validate_artifact,
        "validate_edit_artifact",
        lambda *args, **kwargs: validate_artifact.EditArtifactValidationResult(
            ok=False,
            issues=["injected post-save validation failure"],
        ),
    )

    with pytest.raises(RuntimeError, match="post-save validation failure"):
        validate_artifact.save_edited_subject_artifact(
            model=_Model(),
            tokenizer=_Tokenizer(),
            output_path=output,
            metadata={"edit_type": "quant_rtn", "artifact_class": "subject"},
        )

    assert (output / "retained.txt").read_text(encoding="utf-8") == "previous"
    assert not validate_artifact.staging_path_for(output).exists()


def test_generic_artifact_validation_reports_bad_metadata_without_losing_shape(
    tmp_path: Path,
) -> None:
    missing = validate_artifact.validate_edit_artifact(tmp_path / "missing")
    assert missing.ok is False
    assert missing.issues == [
        f"edit artifact directory not found: {tmp_path / 'missing'}"
    ]

    checkpoint = tmp_path / "checkpoint"
    _minimal_checkpoint(checkpoint)
    (checkpoint / "edit_metadata.json").write_text("{", encoding="utf-8")
    result = validate_artifact.validate_edit_artifact(
        checkpoint,
        require_metadata=True,
        expected_edit_type="quant_rtn",
    )

    assert result.ok is False
    assert result.has_config is True
    assert result.has_tokenizer is True
    assert result.has_weights is True
    assert result.has_metadata is True
    assert any("edit_metadata.json invalid" in issue for issue in result.issues or [])


class _FakeParams4bit:
    def numel(self) -> int:
        return 2


class _FakeLinear4bit:
    pass


class _FakeRuntimeModel:
    def __init__(
        self,
        *,
        footprint: int,
        packed: bool = True,
        quantization_config: object = None,
        logits: torch.Tensor | None = None,
    ) -> None:
        self._footprint = footprint
        self._packed = packed
        self.config = types.SimpleNamespace(quantization_config=quantization_config)
        self._logits = logits if logits is not None else torch.ones((1, 1, 2))
        self._dense_weight = types.SimpleNamespace(numel=lambda: 4)

    def eval(self) -> _FakeRuntimeModel:
        return self

    def get_memory_footprint(self) -> int:
        return self._footprint

    def named_parameters(self, *, remove_duplicate: bool) -> list[tuple[str, object]]:
        assert remove_duplicate is False
        return [("layer.weight", self._dense_weight)]

    def named_modules(self) -> list[tuple[str, object]]:
        if not self._packed:
            return [("", self)]
        packed_module = _FakeLinear4bit()
        packed_module.weight = _FakeParams4bit()
        return [("", self), ("layer", packed_module)]

    def parameters(self) -> Any:
        yield types.SimpleNamespace(device=torch.device("cpu"))

    def __call__(self, **inputs: torch.Tensor) -> object:
        assert inputs
        return types.SimpleNamespace(logits=self._logits)


def _install_fake_runtime_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_footprint: int = 1_000,
    quantized_footprint: int = 400,
    packed: bool = True,
    quantization_config: object = None,
    logits: torch.Tensor | None = None,
) -> None:
    if quantization_config is None:
        quantization_config = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "load_in_8bit": False,
        }
    models = [
        _FakeRuntimeModel(footprint=baseline_footprint),
        _FakeRuntimeModel(
            footprint=quantized_footprint,
            packed=packed,
            quantization_config=quantization_config,
            logits=logits,
        ),
    ]

    class _AutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _FakeRuntimeModel:
            assert path
            assert kwargs
            return models.pop(0)

    class _Tokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, Any]:
            assert prompt == validate_artifact.DEPLOYABLE_SMOKE_PROMPT
            assert return_tensors == "pt"
            return {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _Tokenizer:
            assert path
            assert kwargs == {"trust_remote_code": False}
            return _Tokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _AutoModel
    fake_transformers.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        "invarlock.evidence_pack_contracts.deployable_coverage._bitsandbytes_type_contract",
        lambda bits: (
            (_FakeLinear4bit, _FakeParams4bit)
            if bits == 4
            else (type("UnusedLinear8bitLt", (), {}), type("UnusedInt8Params", (), {}))
        ),
    )


def test_runtime_quantization_reproof_records_observed_packing_and_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    baseline = tmp_path / "baseline"
    _minimal_checkpoint(artifact)
    _minimal_checkpoint(baseline)
    _install_fake_runtime_loaders(monkeypatch)

    proof = validate_artifact._runtime_bitsandbytes_proof(
        artifact,
        baseline_dir=baseline,
        expected_bits=4,
        trust_remote_code=False,
    )

    assert proof["quantized_module_count"] == 1
    assert len(proof["quantized_module_types"]) == 1
    assert proof["quantized_module_types"][0].endswith("._FakeLinear4bit")
    assert proof["runtime_memory_reduction_observed"] is True
    assert proof["reduction_bytes"] == 600
    assert proof["all_logits_finite"] is True


@pytest.mark.parametrize(
    ("loader_options", "message"),
    [
        ({"baseline_footprint": 0}, "baseline model reported a non-positive"),
        ({"quantized_footprint": 1_000}, "did not independently observe memory"),
        ({"packed": False}, "no bitsandbytes packed linear modules"),
        ({"quantization_config": {}}, "quantization config bit flags mismatch"),
        ({"logits": torch.tensor([float("nan")])}, "finite logits"),
    ],
)
def test_runtime_quantization_reproof_rejects_missing_independent_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader_options: dict[str, object],
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    baseline = tmp_path / "baseline"
    _minimal_checkpoint(artifact)
    _minimal_checkpoint(baseline)
    _install_fake_runtime_loaders(monkeypatch, **loader_options)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=message):
        validate_artifact._runtime_bitsandbytes_proof(
            artifact,
            baseline_dir=baseline,
            expected_bits=4,
            trust_remote_code=False,
        )


def test_runtime_quantization_reproof_requires_cuda_and_immutable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    baseline = tmp_path / "baseline"
    _minimal_checkpoint(artifact)
    _minimal_checkpoint(baseline)
    _install_fake_runtime_loaders(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        validate_artifact._runtime_bitsandbytes_proof(
            artifact,
            baseline_dir=baseline,
            expected_bits=4,
            trust_remote_code=False,
        )

    _install_fake_runtime_loaders(monkeypatch)
    real_identity = validate_artifact.checkpoint_tree_sha256
    artifact_reads = 0

    def changed_identity(path: Path) -> str:
        nonlocal artifact_reads
        digest = real_identity(path)
        if path == artifact:
            artifact_reads += 1
            if artifact_reads == 2:
                return "sha256:" + "f" * 64
        return digest

    monkeypatch.setattr(validate_artifact, "checkpoint_tree_sha256", changed_identity)
    with pytest.raises(RuntimeError, match="tree changed during runtime smoke"):
        validate_artifact._runtime_bitsandbytes_proof(
            artifact,
            baseline_dir=baseline,
            expected_bits=4,
            trust_remote_code=False,
        )


def test_failed_deployable_cli_payload_returns_nonzero_and_persists_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def reject(path: Path, **kwargs: object) -> dict[str, object]:
        captured.update({"path": path, **kwargs})
        return {"ok": False, "issues": ["packed modules were not observed"]}

    monkeypatch.setattr(validate_artifact, "validate_deployable_artifact", reject)
    output = tmp_path / "proofs" / "deployable.json"
    exit_code = validate_artifact.main(
        [
            "validate_artifact.py",
            "deployable",
            str(tmp_path / "artifact"),
            "--backend",
            "bitsandbytes",
            "--report-dir",
            str(tmp_path / "reports"),
            "--smoke",
            "--expected-bits",
            "4",
            "--require-publication",
            "--baseline",
            str(tmp_path / "baseline"),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is False
    assert captured == {
        "path": tmp_path / "artifact",
        "backend": "bitsandbytes",
        "report_dir": tmp_path / "reports",
        "smoke": True,
        "expected_bits": 4,
        "trust_remote_code": False,
        "require_publication": True,
        "baseline_dir": tmp_path / "baseline",
    }


def test_failed_pruning_cli_payload_returns_nonzero_with_bounded_worker_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def reject(path: Path, **kwargs: object) -> dict[str, object]:
        captured.update({"path": path, **kwargs})
        return {"ok": False, "issues": ["out-of-scope tensor changed"]}

    monkeypatch.setattr(validate_artifact, "validate_pruning_artifact", reject)
    exit_code = validate_artifact.main(
        [
            "validate_artifact.py",
            "pruning",
            str(tmp_path / "artifact"),
            "--baseline",
            str(tmp_path / "baseline"),
            "--scope",
            "ffn",
            "--target-sparsity",
            "0.5",
            "--workers",
            "3",
            "--worker-threads",
            "2",
        ]
    )

    assert exit_code == 1
    assert captured == {
        "path": tmp_path / "artifact",
        "baseline_dir": tmp_path / "baseline",
        "scope": "ffn",
        "target_sparsity": 0.5,
        "workers": 3,
        "worker_threads": 2,
    }


def _clean_binding_inputs(root: Path) -> dict[str, object]:
    record = _record(root)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    first_report = reports[0]
    assert isinstance(first_report, dict)
    report_reference = first_report["report"]
    assert isinstance(report_reference, dict)
    replay_reference = evaluation["replay"]
    runtime_reference = evaluation["runtime"]
    execution_reference = evaluation["execution"]
    assert isinstance(replay_reference, dict)
    assert isinstance(runtime_reference, dict)
    assert isinstance(execution_reference, dict)

    def read(reference: dict[str, object]) -> dict[str, object]:
        payload = json.loads(
            (root / str(reference["path"])).read_text(encoding="utf-8")
        )
        assert isinstance(payload, dict)
        return payload

    return {
        "report": read(report_reference),
        "replay": read(replay_reference),
        "runtime": read(runtime_reference),
        "original_model_key": record["original_model_key"],
        "candidate_id": candidate["candidate_id"],
        "transformation": candidate["transformation"],
        "selection_config": record["selection_config"],
        "execution_receipt": read(execution_reference),
        "execution_receipt_sha256": execution_reference["sha256"],
        "repeat_index": 0,
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"candidate_id": "../forged"}, "candidate_id is invalid"),
        ({"repeat_index": True}, "repeat_index must be an integer"),
        ({"repeat_index": 2}, "outside the selection schedule"),
    ],
)
def test_clean_selection_binding_rejects_forged_identity_or_repeat(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    inputs = _clean_binding_inputs(tmp_path)
    inputs.update(override)

    with pytest.raises(CleanSelectionEvidenceError, match=message):
        build_candidate_report_binding(**inputs)  # type: ignore[arg-type]


def test_clean_pruning_binding_rejects_forged_candidate_identity(
    tmp_path: Path,
) -> None:
    record = _pruning_record(tmp_path)
    candidate = _pruning_candidate_mapping(record, 0)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    report_run = reports[0]
    assert isinstance(report_run, dict)
    report_reference = report_run["report"]
    replay_reference = evaluation["replay"]
    runtime_reference = evaluation["runtime"]
    execution_reference = evaluation["execution"]
    assert isinstance(report_reference, dict)
    assert isinstance(replay_reference, dict)
    assert isinstance(runtime_reference, dict)
    assert isinstance(execution_reference, dict)

    def read(reference: dict[str, object]) -> dict[str, object]:
        payload = json.loads(
            (tmp_path / str(reference["path"])).read_text(encoding="utf-8")
        )
        assert isinstance(payload, dict)
        return payload

    with pytest.raises(
        CleanPruningSelectionEvidenceError,
        match="candidate_id is invalid",
    ):
        build_clean_pruning_candidate_report_binding(
            report=read(report_reference),
            replay=read(replay_reference),
            runtime=read(runtime_reference),
            original_model_key=str(record["original_model_key"]),
            candidate_id="../forged",
            pruning=candidate["pruning"],  # type: ignore[arg-type]
            selection_config=record["selection_config"],  # type: ignore[arg-type]
            execution_receipt=read(execution_reference),
            execution_receipt_sha256=str(execution_reference["sha256"]),
            repeat_index=0,
        )


@pytest.mark.parametrize("runtime", [selection_runtime, pruning_runtime])
def test_selection_runtime_rejects_post_start_sidecar_mutation(
    tmp_path: Path,
    runtime: Any,
) -> None:
    path = tmp_path / "selection.json"
    original = b'{"seed":1}\n'
    path.write_bytes(original)
    path.write_bytes(b'{"seed":2}\n')

    with pytest.raises(
        runtime.__dict__[
            next(name for name in runtime.__dict__ if name.endswith("RuntimeError"))
        ],
        match="changed after evaluator startup",
    ):
        runtime._snapshot_unchanged(path, expected=original, label="selection input")


@pytest.mark.parametrize("runtime", [selection_runtime, pruning_runtime])
def test_selection_runtime_rejects_checkpoint_symlink_and_identity_drift(
    tmp_path: Path,
    runtime: Any,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"trusted")
    link = tmp_path / "checkpoint-link"
    link.symlink_to(checkpoint.name, target_is_directory=True)
    error_type = runtime.__dict__[
        next(name for name in runtime.__dict__ if name.endswith("RuntimeError"))
    ]

    with pytest.raises(error_type, match="regular checkpoint directory"):
        runtime._assert_checkpoint_identity(
            link,
            expected={"sha256": "sha256:" + "0" * 64},
            label="candidate checkpoint",
        )
    with pytest.raises(error_type, match="does not match"):
        runtime._assert_checkpoint_identity(
            checkpoint,
            expected={"sha256": "sha256:" + "0" * 64},
            label="candidate checkpoint",
        )
