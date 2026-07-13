from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as functional

from scripts.evidence_packs.python.editing import training_runtime as runtime


class TinyTokenizer:
    calls: list[dict[str, Any]] = []
    pad_token_id = 0
    eos_token = "<eos>"

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @pad_token.setter
    def pad_token(self, _value: str) -> None:
        self.pad_token_id = 1

    def __call__(self, texts: list[str], **options: Any) -> dict[str, torch.Tensor]:
        type(self).calls.append({"texts": list(texts), **options})
        length = int(options["max_length"])
        ids = torch.zeros((len(texts), length), dtype=torch.long)
        attention = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            encoded = [2 + (ord(character) % 29) for character in text][:length]
            ids[row, : len(encoded)] = torch.tensor(encoded)
            attention[row, : len(encoded)] = 1
        return {"input_ids": ids, "attention_mask": attention}

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text(
            json.dumps(
                {"kind": "tiny", "pad_token_id": self.pad_token_id}, sort_keys=True
            )
        )


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 32)
        self.config = SimpleNamespace(pad_token_id=None)
        self.loss_type: str | None = None

    @property
    def loss_function(self) -> Any:
        if self.loss_type != "ForCausalLM":
            raise ValueError("unsupported loss function")
        return functional.cross_entropy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        logits = self.projection(self.embedding(input_ids))
        loss = None
        if labels is not None:
            if self.loss_type != "ForCausalLM":
                raise ValueError("labeled forward did not bind ForCausalLM")
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "config.json").write_text('{"model_type":"tiny"}\n')


class FakeAutoModel:
    source_calls: list[tuple[str, dict[str, Any]]] = []
    reload_baseline = False
    source_state: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_pretrained(cls, source: Any, **options: Any) -> Any:
        path = (
            Path(source)
            if not isinstance(source, str) or Path(source).exists()
            else None
        )
        model = TinyCausalLM()
        diagnostics = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        if path is not None and path.is_dir():
            if not cls.reload_baseline:
                model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
            return (model, diagnostics) if options.get("output_loading_info") else model
        cls.source_calls.append((str(source), dict(options)))
        diagnostics["unexpected_keys"] = [
            "transformer.h.0.attn.masked_bias",
            "transformer.h.1.attn.masked_bias",
        ]
        if cls.source_state is None:
            cls.source_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            model.load_state_dict(cls.source_state)
        return (model, diagnostics) if options.get("output_loading_info") else model


class FakeAutoTokenizer:
    source_calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, source: str, **options: Any) -> TinyTokenizer:
        cls.source_calls.append((source, dict(options)))
        return TinyTokenizer()


class RecordingAdamW(torch.optim.AdamW):
    constructions: list[dict[str, Any]] = []
    completed_steps = 0

    def __init__(self, params: Any, **options: Any) -> None:
        type(self).constructions.append(dict(options))
        super().__init__(params, **options)

    def step(self, closure: Any = None) -> Any:
        type(self).completed_steps += 1
        return super().step(closure)


class NoOpOptimizer:
    def __init__(self, params: Any, **_options: Any) -> None:
        self.parameters = list(params)

    def zero_grad(self, *, set_to_none: bool) -> None:
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def step(self) -> None:
        return None


class FakeLoraConfig:
    last_options: dict[str, Any] = {}

    def __init__(self, **options: Any) -> None:
        type(self).last_options = dict(options)
        self.options = options
        self.r = int(options["r"])
        self.lora_alpha = int(options["lora_alpha"])
        self.lora_dropout = float(options["lora_dropout"])
        self.target_modules = set(options["target_modules"])
        self.bias = str(options["bias"])
        self.task_type = str(options["task_type"])
        self.fan_in_fan_out = bool(options["fan_in_fan_out"])
        self._custom_modules: dict[type[Any], Any] | None = None

    def _register_custom_module(self, mapping: dict[type[Any], Any]) -> None:
        self._custom_modules = dict(mapping)

    @classmethod
    def from_pretrained(
        cls,
        path: Path,
        *,
        local_files_only: bool,
    ) -> FakeLoraConfig:
        assert local_files_only is True
        saved = json.loads((Path(path) / "adapter_config.json").read_text())
        return cls(**saved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": set(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
            "fan_in_fan_out": self.fan_in_fan_out,
        }


class FakePeftModel(torch.nn.Module):
    def __init__(self, base_model: TinyCausalLM, config: FakeLoraConfig) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        rank = config.r
        self.lora_A = torch.nn.Parameter(torch.randn(rank, 8) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(32, rank))
        self.peft_config = {"default": config}

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {
            "base_model.embedding.weight": self.base_model.embedding.weight,
            "base_model.projection.base_layer.weight": (
                self.base_model.projection.weight
            ),
            "base_model.projection.base_layer.bias": self.base_model.projection.bias,
            "base_model.projection.lora_A.default.weight": self.lora_A,
            "base_model.projection.lora_B.default.weight": self.lora_B,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        if self.base_model.loss_type != "ForCausalLM":
            raise ValueError("LoRA forward did not bind ForCausalLM")
        hidden = self.base_model.embedding(input_ids)
        logits = self.base_model.projection(hidden)
        logits = logits + (hidden @ self.lora_A.T) @ self.lora_B.T
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss)

    def save_pretrained(
        self,
        path: Path,
        *,
        safe_serialization: bool,
        save_embedding_layers: bool,
    ) -> None:
        assert safe_serialization is True
        assert save_embedding_layers is False
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(fake_adapter_state(self), path / "adapter_model.pt")
        config_payload = self.peft_config["default"].to_dict()
        config_payload["target_modules"] = sorted(config_payload["target_modules"])
        (path / "adapter_config.json").write_text(
            json.dumps(config_payload, sort_keys=True) + "\n"
        )

    @classmethod
    def from_pretrained(
        cls,
        base_model: TinyCausalLM,
        path: Path,
        *,
        is_trainable: bool,
        local_files_only: bool,
        config: FakeLoraConfig | None = None,
    ) -> FakePeftModel:
        assert is_trainable is False
        assert local_files_only is True
        state = torch.load(Path(path) / "adapter_model.pt", weights_only=True)
        selected_config = config or FakeLoraConfig.from_pretrained(
            path,
            local_files_only=local_files_only,
        )
        model = cls(base_model, selected_config)
        with torch.no_grad():
            model.lora_A.copy_(state["lora_A"])
            model.lora_B.copy_(state["lora_B"])
        return model

    def merge_and_unload(self) -> TinyCausalLM:
        with torch.no_grad():
            self.base_model.projection.weight.add_(self.lora_B @ self.lora_A)
        return self.base_model


def fake_adapter_state(
    model: FakePeftModel, *, save_embedding_layers: bool = False
) -> dict[str, torch.Tensor]:
    assert save_embedding_layers is False
    return {"lora_A": model.lora_A, "lora_B": model.lora_B}


def fake_peft_dependencies() -> runtime.PeftDependencies:
    return runtime.PeftDependencies(
        lora_config_cls=FakeLoraConfig,
        get_peft_model=lambda model, config: FakePeftModel(model, config),
        get_peft_model_state_dict=fake_adapter_state,
        peft_model_cls=FakePeftModel,
        version="0.19.1",
    )


def reset_training_fakes() -> None:
    TinyTokenizer.calls = []
    FakeAutoModel.source_calls = []
    FakeAutoModel.reload_baseline = False
    FakeAutoModel.source_state = None
    FakeAutoTokenizer.source_calls = []
    RecordingAdamW.constructions = []
    RecordingAdamW.completed_steps = 0
    FakeLoraConfig.last_options = {}
