from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.runtime_providers import _hf_transformers_identity as identity


@pytest.mark.parametrize(
    ("imported", "installed"),
    [
        ("2.11.0", "2.11.0"),
        ("2.11.0+cpu", "2.11.0"),
        ("2.11.0", "2.11.0+cpu"),
        ("2.11.0+CUDA_12-8", "2.11.0+cuda.12.8"),
    ],
)
def test_runtime_version_accepts_only_equivalent_local_build_suffixes(
    imported: str, installed: str
) -> None:
    assert identity._runtime_version_matches_distribution(imported, installed)


@pytest.mark.parametrize(
    ("imported", "installed"),
    [
        ("2.11.1+cpu", "2.11.0"),
        ("2.11.0+cu128", "2.11.0+cpu"),
        ("2.11.0+", "2.11.0"),
        ("2.11.0+cuda@12", "2.11.0"),
        ("2.11.0+cu128+debug", "2.11.0"),
        (" 2.11.0", "2.11.0"),
        (None, "2.11.0"),
    ],
)
def test_runtime_version_rejects_non_equivalent_or_invalid_versions(
    imported: object, installed: str
) -> None:
    assert not identity._runtime_version_matches_distribution(imported, installed)


class _TokenizerBackend:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def to_str(self) -> str:
        if isinstance(self.payload, BaseException):
            raise self.payload
        assert isinstance(self.payload, str)
        return self.payload


class _Tokenizer:
    backend_tokenizer = _TokenizerBackend(
        json.dumps(
            {
                "model": {"type": "test"},
                "padding": {"strategy": "BatchLongest"},
                "truncation": {"max_length": 4},
            }
        )
    )
    chat_template = None
    clean_up_tokenization_spaces = False
    model_max_length = 64
    padding_side = "right"
    special_tokens_map = {"eos_token": "</s>"}
    truncation_side = "right"

    def get_vocab(self) -> dict[str, int]:
        return {"</s>": 1, "hello": 2}

    def get_added_vocab(self) -> dict[str, int]:
        return {"</s>": 1}


def test_tokenizer_contract_is_path_free_and_ignores_request_local_backend_state() -> (
    None
):
    tokenizer = _Tokenizer()
    first = identity.hf_tokenizer_contract_sha256(tokenizer)
    tokenizer.backend_tokenizer = _TokenizerBackend(
        json.dumps(
            {
                "model": {"type": "test"},
                "padding": None,
                "truncation": None,
            }
        )
    )

    assert identity.hf_tokenizer_contract_sha256(tokenizer) == first


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (SimpleNamespace(), "does not expose"),
        (
            SimpleNamespace(
                get_vocab=lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
            ),
            "vocabulary is unavailable",
        ),
        (SimpleNamespace(get_vocab=lambda: {}), "vocabulary is unavailable"),
        (
            SimpleNamespace(get_vocab=lambda: {"bad": True}),
            "vocabulary is invalid",
        ),
        (
            SimpleNamespace(
                get_vocab=lambda: {"ok": 1},
                backend_tokenizer=_TokenizerBackend(""),
            ),
            "backend contract is unavailable",
        ),
        (
            SimpleNamespace(
                get_vocab=lambda: {"ok": 1},
                get_added_vocab=lambda: (_ for _ in ()).throw(
                    RuntimeError("unavailable")
                ),
            ),
            "added vocabulary is unavailable",
        ),
        (
            SimpleNamespace(
                get_vocab=lambda: {"ok": 1},
                get_added_vocab=lambda: [],
            ),
            "added vocabulary is invalid",
        ),
    ],
)
def test_tokenizer_contract_rejects_incomplete_or_unstable_inputs(
    tokenizer: object, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        identity.hf_tokenizer_contract_sha256(tokenizer)


def test_json_safe_tokenizer_value_handles_structured_and_opaque_values() -> None:
    class _Serializable:
        def to_dict(self) -> dict[str, object]:
            return {"nested": {3, 1}}

    class _Opaque:
        def to_dict(self) -> object:
            raise ValueError("not serializable")

        def __str__(self) -> str:
            return "opaque"

    assert identity._json_safe_tokenizer_value(_Serializable()) == {"nested": [1, 3]}
    assert identity._json_safe_tokenizer_value(_Opaque()) == "opaque"


def test_regular_file_identity_accepts_bytes_and_rejects_missing_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_bytes(b"authenticated source\n")

    assert identity._regular_file_sha256(source.as_posix(), label="module") == (
        identity._sha256(source.read_bytes())
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        identity._regular_file_sha256(None, label="module")
    with pytest.raises(RuntimeError, match="unavailable"):
        identity._regular_file_sha256(
            (tmp_path / "missing.py").as_posix(), label="module"
        )


def test_distribution_identity_requires_complete_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Distribution:
        version = "1.2.3"

        def __init__(self, metadata: str | None, record: str | None) -> None:
            self.values = {"METADATA": metadata, "RECORD": record}

        def read_text(self, name: str) -> str | None:
            return self.values[name]

    monkeypatch.setattr(
        identity.importlib.metadata,
        "distribution",
        lambda _name: _Distribution("Name: example\n", "example.py,,1\n"),
    )
    assert identity._distribution_identity("example") == {
        "metadata_sha256": identity._sha256(b"Name: example\n"),
        "name": "example",
        "record_sha256": identity._sha256(b"example.py,,1\n"),
        "version": "1.2.3",
    }

    monkeypatch.setattr(
        identity.importlib.metadata,
        "distribution",
        lambda _name: _Distribution(None, "record"),
    )
    with pytest.raises(RuntimeError, match="lacks METADATA or RECORD"):
        identity._distribution_identity("example")

    class _VersionlessDistribution(_Distribution):
        version = ""

    monkeypatch.setattr(
        identity.importlib.metadata,
        "distribution",
        lambda _name: _VersionlessDistribution("metadata", "record"),
    )
    with pytest.raises(RuntimeError, match="lacks a version"):
        identity._distribution_identity("example")

    def missing(_name: str) -> object:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(identity.importlib.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="is not installed"):
        identity._distribution_identity("example")


def test_installed_backend_identity_rejects_mismatched_and_incomplete_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = SimpleNamespace(__version__="1.0", __file__=__file__)
    torch_c = SimpleNamespace(__file__=__file__)
    torch = SimpleNamespace(
        __version__="1.0",
        __file__=__file__,
        _C=torch_c,
        __config__=SimpleNamespace(show=lambda: "build"),
    )

    def import_module(name: str) -> object:
        return {
            "transformers": transformers,
            "torch": torch,
            "torch._C": torch_c,
            __name__: SimpleNamespace(__file__=__file__),
        }[name]

    monkeypatch.setattr(identity.importlib, "import_module", import_module)
    versions = {"transformers": "2.0", "torch": "1.0"}
    monkeypatch.setattr(
        identity,
        "_distribution_identity",
        lambda name: {
            "metadata_sha256": "a" * 64,
            "name": name,
            "record_sha256": "b" * 64,
            "version": versions[name],
        },
    )

    class Model:
        pass

    with pytest.raises(RuntimeError, match="transformers version"):
        identity._installed_backend_identity(Model())

    versions["transformers"] = "1.0"
    versions["torch"] = "2.0"
    with pytest.raises(RuntimeError, match="torch version"):
        identity._installed_backend_identity(Model())

    versions["torch"] = "1.0"
    with pytest.raises(RuntimeError, match="implementation identity"):
        identity._installed_backend_identity(object())

    torch.__config__ = SimpleNamespace(show=None)
    with pytest.raises(RuntimeError, match="build configuration"):
        identity._installed_backend_identity(Model())

    torch.__config__ = SimpleNamespace(show=lambda: "")
    with pytest.raises(RuntimeError, match="build configuration"):
        identity._installed_backend_identity(Model())


def test_observed_device_facts_cover_cpu_accelerator_and_error_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Module:
        pass

    class Tensor:
        def __init__(self, kind: str, index: int | None = None) -> None:
            self.device = SimpleNamespace(type=kind, index=index)

    class Model(Module):
        def __init__(
            self,
            parameters: tuple[Tensor, ...] = (),
            buffers: tuple[Tensor, ...] = (),
        ) -> None:
            self._parameters = parameters
            self._buffers = buffers

        def parameters(self) -> tuple[Tensor, ...]:
            return self._parameters

        def buffers(self) -> tuple[Tensor, ...]:
            return self._buffers

    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(
            name=f"GPU {index}", major=9, minor=0
        ),
    )
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module),
        cuda=cuda,
        _C=SimpleNamespace(_cuda_getDriverVersion=lambda: 12080),
    )
    monkeypatch.setattr(identity.importlib, "import_module", lambda _name: fake_torch)

    with pytest.raises(RuntimeError, match="must be a torch module"):
        identity._observed_device_facts(object(), expected_device_kind="cpu")
    with pytest.raises(RuntimeError, match="no tensors"):
        identity._observed_device_facts(Model(), expected_device_kind="cpu")
    with pytest.raises(RuntimeError, match="one model execution device"):
        identity._observed_device_facts(
            Model((Tensor("cpu"), Tensor("cuda", 0))),
            expected_device_kind="cpu",
        )
    with pytest.raises(ValueError, match="does not match"):
        identity._observed_device_facts(
            Model((Tensor("cpu"),)), expected_device_kind="cuda"
        )

    monkeypatch.setattr(identity.platform, "processor", lambda: "test-cpu")
    cpu = identity._observed_device_facts(
        Model(buffers=(Tensor("cpu"),)), expected_device_kind="cpu"
    )
    assert cpu.device_name == "CPU test-cpu"
    assert (
        identity._observed_device_facts(
            Model((Tensor("mps"),)), expected_device_kind="mps"
        ).device_name
        == "Apple Metal Performance Shaders"
    )
    assert (
        identity._observed_device_facts(
            Model((Tensor("xpu"),)), expected_device_kind="xpu"
        ).device_name
        == "xpu"
    )

    observed_cuda = identity._observed_device_facts(
        Model((Tensor("cuda"),)), expected_device_kind="cuda"
    )
    assert observed_cuda.device_name == "GPU 2"
    assert observed_cuda.compute_capability == "9.0"
    assert observed_cuda.driver_version == "12080"

    cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA model device is unavailable"):
        identity._observed_device_facts(
            Model((Tensor("cuda", 0),)), expected_device_kind="cuda"
        )
