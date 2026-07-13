from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from invarlock import transformation_runtime_proof as runtime_proof_contract
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from scripts.evidence_packs.python.editing import runtime_reload_proof as runtime_proof


def test_runtime_reload_script_reexports_package_owned_contract() -> None:
    assert runtime_proof.RuntimeReloadProofError is (
        runtime_proof_contract.RuntimeReloadProofError
    )
    assert runtime_proof._PROOF_KEYS == runtime_proof_contract.PROOF_KEYS
    assert runtime_proof.RUNTIME_RELOAD_PROOF_SCHEMA == (
        runtime_proof_contract.RUNTIME_RELOAD_PROOF_SCHEMA
    )


def test_runtime_reload_rejects_duplicate_replay_keys_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    raw = replay.read_text(encoding="utf-8")
    replay.write_text(raw.replace('"schema":', '"schema":"forged", "schema":', 1))
    monkeypatch.setattr(
        runtime_proof,
        "_load_runtime_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not load")),
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError, match="not strict UTF-8 JSON"
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


_STORAGE_KEY = "model.layers.0.mlp.up_proj.weight"


def _clean_loading_info() -> dict[str, list[object]]:
    return {
        "unexpected_keys": [],
        "missing_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }


def _write_artifact(path: Path) -> dict[str, str]:
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    save_file(
        {_STORAGE_KEY: torch.zeros((1, 1), dtype=torch.float32)},
        path / "model.safetensors",
    )
    return {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(path),
    }


def _write_replay(path: Path, identity: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "invarlock/generated-transformation-replay-v1",
                "ok": True,
                "edit_type": "quant_rtn",
                "baseline_identity": identity,
                "artifact_identity": identity,
                "issues": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass
class _FakeRuntime:
    non_deterministic: bool = False
    loading_info: dict[str, object] = field(default_factory=_clean_loading_info)
    tokenizer_calls: list[dict[str, object]] = field(default_factory=list)
    model_calls: list[dict[str, object]] = field(default_factory=list)
    model_to_calls: list[object] = field(default_factory=list)

    def dependencies(self) -> runtime_proof.RuntimeReloadDependencies:
        runtime = self

        class Tokenizer:
            def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, Any]:
                assert prompt == runtime_proof.RUNTIME_RELOAD_PROMPT
                assert return_tensors == "pt"
                return {
                    "input_ids": torch.tensor([[4, 8, 15, 16]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long),
                }

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, path: Path, **kwargs: object) -> Tokenizer:
                runtime.tokenizer_calls.append({"path": path, **kwargs})
                return Tokenizer()

        class Model:
            def __init__(self, run_number: int) -> None:
                self.run_number = run_number
                self.device = torch.device("cpu")

            def eval(self) -> Model:
                return self

            def state_dict(self) -> dict[str, torch.Tensor]:
                return {_STORAGE_KEY: torch.zeros((1, 1), dtype=torch.float32)}

            def to(self, device: Any) -> Model:
                runtime.model_to_calls.append(device)
                self.device = torch.device(device)
                return self

            def __call__(self, **inputs: Any) -> SimpleNamespace:
                assert inputs["input_ids"].device == self.device
                offset = 0.1 * self.run_number if runtime.non_deterministic else 0.0
                logits = torch.tensor(
                    [[[1.0 + offset, 2.0], [3.0, 4.0]]],
                    dtype=torch.float32,
                    device=self.device,
                )
                return SimpleNamespace(logits=logits)

        class AutoModel:
            @classmethod
            def from_pretrained(
                cls, path: Path, **kwargs: object
            ) -> tuple[Model, dict[str, list[object]]]:
                runtime.model_calls.append({"path": path, **kwargs})
                return Model(len(runtime.model_calls)), dict(runtime.loading_info)

        return runtime_proof.RuntimeReloadDependencies(
            torch=torch,
            auto_model=AutoModel,
            auto_tokenizer=AutoTokenizer,
        )


def test_runtime_reload_proof_uses_two_fresh_local_reload_paths_and_writes_no_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )

    proof = runtime_proof.run_runtime_reload_proof(
        artifact,
        replay_path=replay,
        expected_identity=identity,
        device="cpu",
    )

    assert proof["ok"] is True
    assert proof["artifact_identity"] == identity
    assert proof["replay_artifact_identity"] == identity
    assert proof["device"] == "cpu"
    assert proof["input_device"] == "cpu"
    assert proof["reload_runs"] == 2
    assert proof["load_diagnostics"] == {
        "schema": runtime_proof.RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
        "reloads": [_clean_loading_info(), _clean_loading_info()],
    }
    assert proof["token_ids_shape"] == [1, 4]
    assert proof["logits_shape"] == [1, 2, 2]
    assert len(fake_runtime.tokenizer_calls) == 2
    assert len(fake_runtime.model_calls) == 2
    assert fake_runtime.model_to_calls == []
    for call in [*fake_runtime.tokenizer_calls, *fake_runtime.model_calls]:
        assert call["path"] == artifact
        assert call["local_files_only"] is True
        assert call["trust_remote_code"] is False
    for call in fake_runtime.model_calls:
        assert set(call) == {
            "path",
            "local_files_only",
            "trust_remote_code",
            "output_loading_info",
        }
        assert call["output_loading_info"] is True

    output = tmp_path / "reports" / "runtime_reload.json"
    runtime_proof.write_runtime_reload_proof(
        output,
        proof,
        artifact_dir=artifact,
        replay_path=replay,
    )
    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == proof
    assert str(tmp_path) not in serialized


def test_runtime_reload_proof_dispatches_bf16_loads_and_routes_inputs_to_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    model_calls: list[dict[str, object]] = []
    observed_input_devices: list[str] = []

    class CudaFacade:
        bfloat16 = torch.bfloat16

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def empty_cache() -> None:
                return None

        def __getattr__(self, name: str) -> Any:
            return getattr(torch, name)

    class Tokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, Any]:
            assert prompt == runtime_proof.RUNTIME_RELOAD_PROMPT
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> Tokenizer:
            del cls, path
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return Tokenizer()

    class Model:
        hf_device_map = {"model.embed_tokens": "cpu", "model.layers.0": "cuda:0"}

        def eval(self) -> Model:
            return self

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {_STORAGE_KEY: torch.zeros((1, 1), dtype=torch.float32)}

        def to(self, device: Any) -> Model:
            raise AssertionError(f"dispatch-managed model must not move to {device}")

        def get_input_embeddings(self) -> SimpleNamespace:
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))

        def __call__(self, **inputs: Any) -> SimpleNamespace:
            observed_input_devices.append(str(inputs["input_ids"].device))
            assert str(inputs["input_ids"].device) == "cpu"
            return SimpleNamespace(
                logits=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
            )

    class AutoModel:
        @classmethod
        def from_pretrained(
            cls, path: Path, **kwargs: object
        ) -> tuple[Model, dict[str, list[object]]]:
            model_calls.append({"path": path, **kwargs})
            return Model(), _clean_loading_info()

    monkeypatch.setattr(
        runtime_proof,
        "_load_runtime_dependencies",
        lambda: runtime_proof.RuntimeReloadDependencies(
            torch=CudaFacade(),
            auto_model=AutoModel,
            auto_tokenizer=AutoTokenizer,
        ),
    )

    proof = runtime_proof.run_runtime_reload_proof(
        artifact,
        replay_path=replay,
        device="auto",
    )

    assert proof["device"] == "cuda"
    assert proof["input_device"] == "cpu"
    assert observed_input_devices == ["cpu", "cpu"]
    assert len(model_calls) == 2
    for call in model_calls:
        assert call == {
            "path": artifact,
            "local_files_only": True,
            "trust_remote_code": False,
            "output_loading_info": True,
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }


@pytest.mark.parametrize(
    ("missing_name", "expected_error"),
    [
        ("tokenizer.json", "artifact tokenizer files are missing"),
        ("model.safetensors", "artifact model weights are missing"),
    ],
)
def test_runtime_reload_proof_fails_closed_when_required_artifact_inputs_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
    expected_error: str,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    (artifact / missing_name).unlink()
    replay = tmp_path / "replay.json"
    _write_replay(
        replay,
        {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(artifact),
        },
    )
    monkeypatch.setattr(
        runtime_proof,
        "_load_runtime_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not load")),
    )

    with pytest.raises(runtime_proof.RuntimeReloadProofError, match=expected_error):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_rejects_missing_replay_identity_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    payload = json.loads(replay.read_text(encoding="utf-8"))
    del payload["artifact_identity"]
    replay.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_proof,
        "_load_runtime_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not load")),
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="replay artifact identity is invalid",
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


@pytest.mark.parametrize(
    ("missing_component", "expected_error"),
    [
        ("tokenizer", "artifact tokenizer could not be loaded"),
        ("model", "artifact model could not be loaded"),
    ],
)
def test_runtime_reload_proof_fails_closed_when_a_runtime_loader_returns_no_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_component: str,
    expected_error: str,
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)

    class Tokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, Any]:
            del prompt, return_tensors
            return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> Tokenizer | None:
            del cls, path, kwargs
            return None if missing_component == "tokenizer" else Tokenizer()

    class AutoModel:
        @classmethod
        def from_pretrained(
            cls, path: Path, **kwargs: object
        ) -> tuple[None, dict[str, list[object]]]:
            del cls, path, kwargs
            return None, _clean_loading_info()

    monkeypatch.setattr(
        runtime_proof,
        "_load_runtime_dependencies",
        lambda: runtime_proof.RuntimeReloadDependencies(
            torch=torch,
            auto_model=AutoModel,
            auto_tokenizer=AutoTokenizer,
        ),
    )

    with pytest.raises(runtime_proof.RuntimeReloadProofError, match=expected_error):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_rejects_non_deterministic_fresh_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime(non_deterministic=True)
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="runtime reload was not deterministic",
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_rejects_injected_unexpected_checkpoint_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime(
        loading_info={
            "unexpected_keys": {"model.layers.999.mlp.up_proj.weight"},
            "missing_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }
    )
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="loading diagnostics report unexpected_keys",
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_rejects_unrecognized_loading_diagnostic_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime(
        loading_info={**_clean_loading_info(), "skipped_keys": []}
    )
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="did not return loading diagnostics",
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_normalizes_real_transformers_empty_set_diagnostics(
    tmp_path: Path,
) -> None:
    """Pinned Transformers returns empty sets for several loading-info keys."""

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.Qwen2ForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        output_loading_info=True,
    )
    assert isinstance(loaded, tuple) and len(loaded) == 2
    _, loading_info = loaded
    assert isinstance(loading_info, dict)
    assert any(
        isinstance(loading_info[field], set)
        for field in ("unexpected_keys", "missing_keys", "mismatched_keys")
    )
    assert runtime_proof._clean_load_diagnostics(loading_info) == _clean_loading_info()


def test_storage_key_audit_rejects_real_gpt2_key_filtered_from_loading_info(
    tmp_path: Path,
) -> None:
    """A loader-ignore pattern must not turn an injected safetensors key green."""

    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file

    config = transformers.GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    transformers.GPT2LMHeadModel(config).save_pretrained(
        tmp_path, safe_serialization=True
    )
    weights_path = tmp_path / "model.safetensors"
    tensors = load_file(weights_path)
    tensors["transformer.h.0.attn.bias.mlp.c_fc.weight"] = torch.ones(
        (1, 1), dtype=torch.float32
    )
    save_file(tensors, weights_path)

    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        output_loading_info=True,
    )
    assert isinstance(loaded, tuple) and len(loaded) == 2
    model, loading_info = loaded
    assert runtime_proof._clean_load_diagnostics(loading_info) == _clean_loading_info()
    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="keys absent from loaded model state",
    ):
        runtime_proof._storage_key_audit(tmp_path, model=model)


@pytest.mark.parametrize(
    ("weight_map", "include_second_shard", "expected_error"),
    (
        (
            {"model.layers.0.mlp.up_proj.weight": "model-00001.safetensors"},
            True,
            "files do not exactly match the index",
        ),
        (
            {"model.layers.0.mlp.up_proj.weight": "missing.safetensors"},
            True,
            "files do not exactly match the index",
        ),
        (
            {"model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors"},
            False,
            "keys do not exactly match the index",
        ),
    ),
    ids=("omits-existing-shard", "references-missing-shard", "mismatches-shard-keys"),
)
def test_storage_key_audit_rejects_incomplete_or_mismatched_sharded_index(
    tmp_path: Path,
    weight_map: dict[str, str],
    include_second_shard: bool,
    expected_error: str,
) -> None:
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.zeros((1, 1))},
        tmp_path / "model-00001.safetensors",
    )
    if include_second_shard:
        save_file(
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 1))},
            tmp_path / "model-00002.safetensors",
        )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    with pytest.raises(runtime_proof.RuntimeReloadProofError, match=expected_error):
        runtime_proof._artifact_storage_keys(tmp_path)


def test_runtime_reload_proof_rejects_artifact_identity_change_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )
    original_identity = runtime_proof.checkpoint_tree_sha256
    calls = 0

    def changed_identity(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            return "sha256:" + "f" * 64
        return original_identity(path)

    monkeypatch.setattr(runtime_proof, "checkpoint_tree_sha256", changed_identity)

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="artifact tree changed during runtime reload",
    ):
        runtime_proof.run_runtime_reload_proof(
            artifact, replay_path=replay, device="cpu"
        )


def test_runtime_reload_proof_writer_rejects_artifact_tree_and_replay_sidecar_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )
    proof = runtime_proof.run_runtime_reload_proof(artifact, replay_path=replay)

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="outside the artifact tree",
    ):
        runtime_proof.write_runtime_reload_proof(
            artifact / "reports" / "runtime.json",
            proof,
            artifact_dir=artifact,
            replay_path=replay,
        )
    assert not (artifact / "reports").exists()

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="must not replace replay evidence",
    ):
        runtime_proof.write_runtime_reload_proof(
            replay,
            proof,
            artifact_dir=artifact,
            replay_path=replay,
        )


def test_runtime_reload_proof_writer_rejects_disagreeing_reload_storage_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )
    proof = runtime_proof.run_runtime_reload_proof(artifact, replay_path=replay)
    audit = proof["storage_key_audit"]
    assert isinstance(audit, dict)
    reloads = audit["reloads"]
    assert isinstance(reloads, list)
    second = reloads[1]
    assert isinstance(second, dict)
    second["model_state_keys_sha256"] = "sha256:" + "f" * 64

    with pytest.raises(runtime_proof.RuntimeReloadProofError, match="do not agree"):
        runtime_proof.write_runtime_reload_proof(
            tmp_path / "runtime.json",
            proof,
            artifact_dir=artifact,
            replay_path=replay,
        )


def test_runtime_reload_proof_producer_rejects_impossible_storage_key_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )
    impossible_audit = {
        "artifact_storage_key_count": 2,
        "artifact_storage_keys_sha256": "sha256:" + "a" * 64,
        "model_state_key_count": 1,
        "model_state_keys_sha256": "sha256:" + "b" * 64,
        "unexpected_storage_keys": [],
    }
    monkeypatch.setattr(
        runtime_proof,
        "_storage_key_audit",
        lambda *_args, **_kwargs: dict(impossible_audit),
    )

    with pytest.raises(
        runtime_proof.RuntimeReloadProofError,
        match="more artifact storage keys than model state keys",
    ):
        runtime_proof.run_runtime_reload_proof(artifact, replay_path=replay)


def test_runtime_reload_proof_cli_writes_external_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    identity = _write_artifact(artifact)
    replay = tmp_path / "replay.json"
    _write_replay(replay, identity)
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        runtime_proof, "_load_runtime_dependencies", fake_runtime.dependencies
    )
    output = tmp_path / "runtime.json"

    assert (
        runtime_proof.main(
            [
                "--artifact",
                str(artifact),
                "--replay",
                str(replay),
                "--out",
                str(output),
                "--expected-identity-json",
                json.dumps(identity),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["artifact_identity"] == identity
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True
