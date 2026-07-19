from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from PIL import Image

from invarlock.core.runtime_provider.behavioral_schedule import (
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.evidence_pack_integrity import public_key_fingerprint
from scripts import qualification_render_preflight as preflight


def _text_schedule(*, expected: str = " A"):
    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "qualification-fixture",
            "config_name": "records-jsonl-v1",
            "revision": "a" * 64,
            "split": "validation",
        },
        records=[
            {
                "record_id": "example-001",
                "input_text": "Question?",
                "expected_output": expected,
            }
        ],
    )


def _multimodal_schedule():
    prompt = "What is shown?"
    return build_runtime_behavioral_schedule_from_material(
        task="vision_text_generation",
        dataset_identity={
            "provider": "local",
            "dataset_name": "qualification-vision-fixture",
            "config_name": "records-jsonl-v1",
            "revision": "b" * 64,
            "split": "validation",
        },
        records=[
            {
                "record_id": "vision-001",
                "input_parts": [
                    {
                        "kind": "text",
                        "role": "prompt",
                        "text": prompt,
                        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    },
                    {
                        "kind": "content",
                        "role": "image",
                        "content_id": "image_001",
                        "media_type": "image/png",
                        "byte_length": 100,
                        "sha256": "c" * 64,
                    },
                ],
                "expected_output": "A",
            }
        ],
    )


class _CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_tensors: object,
    ) -> Mapping[str, object]:
        assert truncation is False
        assert return_tensors is None
        ids = [1000 + ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 0)
        return {"input_ids": ids}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [1000 + ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 0)
        return ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        output = []
        for token_id in token_ids:
            if token_id == 0:
                if not skip_special_tokens:
                    output.append("<s>")
                continue
            output.append(chr(token_id - 1000))
        return "".join(output)


class _BrokenContinuationTokenizer(_CharacterTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        value = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return value.replace(" A", "A")


def _assert_self_authenticated(result: Mapping[str, object]) -> None:
    body = dict(result)
    observed = body.pop("result_sha256")
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert observed == hashlib.sha256(payload).hexdigest()


def _schedule_cli_args(tmp_path: Path, schedule) -> list[str]:
    payload = canonical_runtime_behavioral_schedule_json(schedule) + b"\n"
    path = tmp_path / "schedule.json"
    path.write_bytes(payload)
    return [
        "--schedule",
        str(path),
        "--schedule-file-sha256",
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        "--schedule-sha256",
        schedule.schedule_sha256,
    ]


def test_load_bound_schedule_authenticates_file_and_semantic_digests(
    tmp_path: Path,
) -> None:
    schedule = _text_schedule()
    payload = canonical_runtime_behavioral_schedule_json(schedule) + b"\n"
    path = tmp_path / "schedule.json"
    path.write_bytes(payload)

    loaded, file_digest = preflight.load_bound_schedule(
        path,
        expected_file_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        expected_schedule_sha256=schedule.schedule_sha256,
    )

    assert loaded == schedule
    assert file_digest == "sha256:" + hashlib.sha256(payload).hexdigest()
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="file digest"
    ):
        preflight.load_bound_schedule(
            path,
            expected_file_sha256="sha256:" + "0" * 64,
            expected_schedule_sha256=schedule.schedule_sha256,
        )


@pytest.mark.parametrize("metric", ["exact_match", "normalized_nll_per_utf8_byte"])
def test_hf_text_preflight_authenticates_full_untruncated_rendering(
    metric: str,
) -> None:
    schedule = _text_schedule()
    result = preflight.preflight_hf_text(
        schedule,
        schedule_file_sha256="sha256:" + "d" * 64,
        tokenizer=_CharacterTokenizer(),
        expected_tokenizer_sha256="e" * 64,
        context_length=64,
        max_output_tokens=8,
        metric=metric,
        tokenizer_digest=lambda _tokenizer: "e" * 64,
    )

    assert result["ok"] is True
    assert result["profile"] == "hf_text"
    assert result["maximum_prompt_tokens"] == len("Question?") + 1
    _assert_self_authenticated(result)


def test_hf_text_preflight_rejects_truncation_and_continuation_drift() -> None:
    schedule = _text_schedule()
    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "expected_tokenizer_sha256": "e" * 64,
        "max_output_tokens": 8,
        "tokenizer_digest": lambda _tokenizer: "e" * 64,
    }

    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="would be truncated"
    ):
        preflight.preflight_hf_text(
            schedule,
            tokenizer=_CharacterTokenizer(),
            context_length=3,
            metric="exact_match",
            **common,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError,
        match="not an exact tokenizer continuation",
    ):
        preflight.preflight_hf_text(
            schedule,
            tokenizer=_BrokenContinuationTokenizer(),
            context_length=64,
            metric="normalized_nll_per_utf8_byte",
            **common,
        )


def _engine_config(*, max_input_len: int = 64, max_seq_len: int = 80):
    return {
        "version": "1",
        "build_config": {
            "max_input_len": max_input_len,
            "max_seq_len": max_seq_len,
        },
        "pretrained_config": {
            "mapping": {
                "world_size": 1,
                "tp_size": 1,
                "pp_size": 1,
                "cp_size": 1,
            }
        },
    }


def test_tensorrt_preflight_authenticates_tokenizer_and_engine_limits() -> None:
    result = preflight.preflight_tensorrt(
        _text_schedule(expected="A"),
        schedule_file_sha256="sha256:" + "d" * 64,
        tokenizer=_CharacterTokenizer(),
        tokenizer_contract_sha256="e" * 64,
        engine_config=_engine_config(),
        engine_config_sha256="f" * 64,
        context_length=64,
        max_output_tokens=8,
    )

    assert result["profile"] == "tensorrt_llm"
    assert result["bindings"] == {
        "tokenizer_contract_sha256": "e" * 64,
        "engine_config_sha256": "f" * 64,
        "context_length": 64,
        "max_output_tokens": 8,
        "engine_max_input_len": 64,
        "engine_max_seq_len": 80,
    }
    _assert_self_authenticated(result)


@pytest.mark.parametrize(
    ("engine", "context", "output", "message"),
    [
        (_engine_config(max_input_len=8), 9, 1, "input limit"),
        (_engine_config(max_seq_len=10), 9, 2, "sequence limit"),
    ],
)
def test_tensorrt_preflight_rejects_incompatible_declared_limits(
    engine: Mapping[str, object], context: int, output: int, message: str
) -> None:
    with pytest.raises(preflight.QualificationRenderPreflightError, match=message):
        preflight.preflight_tensorrt(
            _text_schedule(expected="A"),
            schedule_file_sha256="sha256:" + "d" * 64,
            tokenizer=_CharacterTokenizer(),
            tokenizer_contract_sha256="e" * 64,
            engine_config=engine,
            engine_config_sha256="f" * 64,
            context_length=context,
            max_output_tokens=output,
        )


class _ClosableImage:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Processor:
    def __init__(self, *, token_count: int) -> None:
        self.tokenizer = _CharacterTokenizer()
        self.token_count = token_count

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert messages
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<image>What is shown?<assistant>"

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        assert kwargs["truncation"] is False
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": [list(range(self.token_count))]}


def test_multimodal_preflight_renders_real_media_without_truncation() -> None:
    image = _ClosableImage()
    result = preflight.preflight_multimodal(
        _multimodal_schedule(),
        schedule_file_sha256="sha256:" + "d" * 64,
        processor=_Processor(token_count=32),
        expected_processor_sha256="e" * 64,
        context_length=64,
        max_output_tokens=8,
        processor_digest=lambda _processor: "e" * 64,
        image_resolver=lambda _record: image,
    )

    assert result["profile"] == "multimodal"
    assert result["maximum_prompt_tokens"] == 32
    assert image.closed is True
    _assert_self_authenticated(result)


def test_multimodal_preflight_rejects_rendered_token_truncation() -> None:
    image = _ClosableImage()
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="would be truncated"
    ):
        preflight.preflight_multimodal(
            _multimodal_schedule(),
            schedule_file_sha256="sha256:" + "d" * 64,
            processor=_Processor(token_count=65),
            expected_processor_sha256="e" * 64,
            context_length=64,
            max_output_tokens=8,
            processor_digest=lambda _processor: "e" * 64,
            image_resolver=lambda _record: image,
        )
    assert image.closed is True


def _gguf_statement(schedule) -> dict[str, object]:
    expected = schedule.records[0]
    output = expected.expected_output
    return {
        "format_version": preflight.GGUF_PREFIX_FORMAT,
        "schedule_sha256": schedule.schedule_sha256,
        "artifact_sha256": "a" * 64,
        "backend_binary_sha256": "b" * 64,
        "runtime_image_digest": "sha256:" + "c" * 64,
        "prefix_record_count": 1,
        "records": [
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "output_text": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            }
        ],
    }


def test_gguf_live_prefix_is_signed_pinned_and_schedule_bound() -> None:
    schedule = _text_schedule(expected="A")
    statement = _gguf_statement(schedule)
    key = ed25519.Ed25519PrivateKey.generate()
    signature = key.sign(preflight._canonical_json(statement))

    result = preflight.verify_gguf_prefix(
        schedule,
        schedule_file_sha256="sha256:" + "d" * 64,
        statement=statement,
        signature=signature,
        public_key=key.public_key(),
        expected_signer_fingerprint=public_key_fingerprint(key.public_key()),
        minimum_exact_matches=1,
    )

    assert result["profile"] == "gguf_live_prefix"
    assert result["bindings"]["exact_matches"] == 1
    _assert_self_authenticated(result)


@pytest.mark.parametrize("attack", ["signature", "pairing", "output"])
def test_gguf_live_prefix_rejects_tampering(attack: str) -> None:
    schedule = _text_schedule(expected="A")
    statement = _gguf_statement(schedule)
    key = ed25519.Ed25519PrivateKey.generate()
    signature = key.sign(preflight._canonical_json(statement))
    if attack == "signature":
        signature = b"0" * len(signature)
        message = "signature"
    else:
        if attack == "pairing":
            statement["records"][0]["input_sha256"] = "0" * 64  # type: ignore[index]
            message = "pairing"
        else:
            statement["records"][0]["output_text"] = "B"  # type: ignore[index]
            message = "output digest"
        signature = key.sign(preflight._canonical_json(statement))

    with pytest.raises(preflight.QualificationRenderPreflightError, match=message):
        preflight.verify_gguf_prefix(
            schedule,
            schedule_file_sha256="sha256:" + "d" * 64,
            statement=statement,
            signature=signature,
            public_key=key.public_key(),
            expected_signer_fingerprint=public_key_fingerprint(key.public_key()),
            minimum_exact_matches=1,
        )


def test_hf_cli_emits_authenticated_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule = _text_schedule()
    tokenizer = _CharacterTokenizer()
    monkeypatch.setattr(preflight, "_load_hf_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            hf_tokenizer_contract_sha256=lambda _tokenizer: "e" * 64
        ),
    )

    assert (
        preflight.main(
            [
                "hf-text",
                *_schedule_cli_args(tmp_path, schedule),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--tokenizer-sha256",
                "e" * 64,
                "--context-length",
                "64",
                "--max-output-tokens",
                "8",
                "--metric",
                "exact_match",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == "hf_text"
    _assert_self_authenticated(result)


def test_tensorrt_cli_emits_authenticated_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule = _text_schedule(expected="A")
    contract = tmp_path / "tokenizer.json"
    contract.write_text("{}")
    engine = tmp_path / "config.json"
    engine_payload = json.dumps(
        _engine_config(), sort_keys=True, separators=(",", ":")
    ).encode()
    engine.write_bytes(engine_payload)
    monkeypatch.setattr(
        preflight,
        "_load_tensorrt_contract",
        lambda _path: (_CharacterTokenizer(), hashlib.sha256(b"{}").hexdigest()),
    )

    assert (
        preflight.main(
            [
                "tensorrt-llm",
                *_schedule_cli_args(tmp_path, schedule),
                "--tokenizer-contract",
                str(contract),
                "--tokenizer-sha256",
                hashlib.sha256(b"{}").hexdigest(),
                "--engine-config",
                str(engine),
                "--engine-config-sha256",
                hashlib.sha256(engine_payload).hexdigest(),
                "--context-length",
                "64",
                "--max-output-tokens",
                "8",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == "tensorrt_llm"
    _assert_self_authenticated(result)


def test_multimodal_cli_emits_authenticated_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule = _multimodal_schedule()
    processor = _Processor(token_count=32)
    image = _ClosableImage()
    monkeypatch.setattr(preflight, "_load_processor", lambda _path: processor)
    monkeypatch.setattr(
        preflight, "_image_resolver", lambda _path: lambda _record: image
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            processor_contract_sha256=lambda _processor: "e" * 64
        ),
    )

    assert (
        preflight.main(
            [
                "multimodal",
                *_schedule_cli_args(tmp_path, schedule),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--processor-sha256",
                "e" * 64,
                "--content-store",
                str(tmp_path / "content"),
                "--context-length",
                "64",
                "--max-output-tokens",
                "8",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == "multimodal"
    assert image.closed is True
    _assert_self_authenticated(result)


def test_gguf_cli_verifies_canonical_signed_statement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schedule = _text_schedule(expected="A")
    statement = _gguf_statement(schedule)
    statement_payload = preflight._canonical_json(statement) + b"\n"
    statement_path = tmp_path / "gguf-prefix.json"
    statement_path.write_bytes(statement_payload)
    key = ed25519.Ed25519PrivateKey.generate()
    signature_path = tmp_path / "gguf-prefix.sig"
    signature_path.write_bytes(
        base64.b64encode(key.sign(preflight._canonical_json(statement))) + b"\n"
    )
    public_key_path = tmp_path / "gguf-prefix.pub.pem"
    public_key_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    assert (
        preflight.main(
            [
                "gguf-prefix",
                *_schedule_cli_args(tmp_path, schedule),
                "--statement",
                str(statement_path),
                "--statement-sha256",
                "sha256:" + hashlib.sha256(statement_payload).hexdigest(),
                "--signature",
                str(signature_path),
                "--public-key",
                str(public_key_path),
                "--signer-fingerprint",
                public_key_fingerprint(key.public_key()),
                "--minimum-exact-matches",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == "gguf_live_prefix"
    _assert_self_authenticated(result)


def test_tensorrt_tokenizer_contract_loader_replays_runner_constructor(
    tmp_path: Path,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "Question": 2, "?": 3, "A": 4},
            unk_token="<pad>",
        )
    )
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    contract = {
        "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
        "add_special_tokens": False,
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "tokenizer_json": json.loads(raw.to_str()),
    }
    payload = preflight._canonical_json(contract) + b"\n"
    path = tmp_path / "tokenizer.json"
    path.write_bytes(payload)

    tokenizer, digest = preflight._load_tensorrt_contract(path)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 0


def test_image_resolver_authenticates_bytes_and_closes_source(tmp_path: Path) -> None:
    output = io.BytesIO()
    source = Image.new("RGB", (2, 2), color=(1, 2, 3))
    source.save(output, format="PNG")
    source.close()
    payload = output.getvalue()
    content_store = tmp_path / "content"
    content_store.mkdir()
    (content_store / "image_001").write_bytes(payload)
    prompt = "What is shown?"
    schedule = build_runtime_behavioral_schedule_from_material(
        task="vision_text_generation",
        dataset_identity={
            "provider": "local",
            "dataset_name": "qualification-image-fixture",
            "config_name": "records-jsonl-v1",
            "revision": "b" * 64,
            "split": "validation",
        },
        records=[
            {
                "record_id": "vision-001",
                "input_parts": [
                    {
                        "kind": "text",
                        "role": "prompt",
                        "text": prompt,
                        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    },
                    {
                        "kind": "content",
                        "role": "image",
                        "content_id": "image_001",
                        "media_type": "image/png",
                        "byte_length": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                ],
                "expected_output": "A",
            }
        ],
    )

    image = preflight._image_resolver(content_store)(schedule.records[0])
    try:
        assert image.mode == "RGB"
        assert image.size == (2, 2)
    finally:
        image.close()


def test_token_helpers_fail_closed_on_malformed_or_unavailable_apis() -> None:
    with pytest.raises(preflight.QualificationRenderPreflightError, match="SHA-256"):
        preflight._digest("BAD", label="fixture")
    for value in (True, 0, "1"):
        with pytest.raises(
            preflight.QualificationRenderPreflightError, match="positive integer"
        ):
            preflight._positive(value, label="fixture")
    assert preflight._single_token_ids([[1, 2]], label="fixture") == [1, 2]
    for value in ("1", [True]):
        with pytest.raises(
            preflight.QualificationRenderPreflightError, match="invalid token IDs"
        ):
            preflight._single_token_ids(value, label="fixture")
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="not callable"
    ):
        preflight._tokenize(object(), "A", add_special_tokens=False, label="fixture")
    with pytest.raises(preflight.QualificationRenderPreflightError, match="failed"):
        preflight._tokenize(
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            "A",
            add_special_tokens=False,
            label="fixture",
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="no input_ids"
    ):
        preflight._tokenize(
            lambda *args, **kwargs: {},
            "A",
            add_special_tokens=False,
            label="fixture",
        )
    with pytest.raises(preflight.QualificationRenderPreflightError, match="no decode"):
        preflight._decode(object(), [1], skip_special_tokens=True, label="fixture")


def test_hf_preflight_rejects_unsupported_or_unauthenticated_inputs() -> None:
    schedule = _text_schedule()
    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "tokenizer": _CharacterTokenizer(),
        "expected_tokenizer_sha256": "e" * 64,
        "context_length": 64,
        "max_output_tokens": 8,
        "metric": "exact_match",
    }
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="digest mismatch"
    ):
        preflight.preflight_hf_text(
            schedule, tokenizer_digest=lambda _tokenizer: "f" * 64, **common
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="authenticated"
    ):
        preflight.preflight_hf_text(
            schedule,
            tokenizer_digest=lambda _tokenizer: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
            **common,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="unsupported"
    ):
        preflight.preflight_hf_text(
            schedule,
            tokenizer_digest=lambda _tokenizer: "e" * 64,
            **{**common, "metric": "other"},
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"build_config": {}, "pretrained_config": {}}, "not closed"),
        (
            {"build_config": [], "pretrained_config": {}, "version": "1"},
            "invalid",
        ),
        (
            {
                "build_config": {"max_input_len": 1, "max_seq_len": 2},
                "pretrained_config": {"mapping": {"tp_size": 2}},
                "version": "1",
            },
            "single-rank",
        ),
        (
            {
                "build_config": {"max_input_len": 0, "max_seq_len": 2},
                "pretrained_config": {"mapping": {}},
                "version": "1",
            },
            "positive integer",
        ),
    ],
)
def test_tensorrt_engine_contract_fails_closed(
    config: Mapping[str, object], message: str
) -> None:
    with pytest.raises(preflight.QualificationRenderPreflightError, match=message):
        preflight._engine_limits(config)


def test_tensorrt_preflight_rejects_live_tokenization_mismatch() -> None:
    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "tokenizer_contract_sha256": "e" * 64,
        "engine_config": _engine_config(),
        "engine_config_sha256": "f" * 64,
        "context_length": 64,
        "max_output_tokens": 8,
    }
    with pytest.raises(preflight.QualificationRenderPreflightError, match="no encode"):
        preflight.preflight_tensorrt(_text_schedule(), tokenizer=object(), **common)
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="tokenization failed"
    ):
        preflight.preflight_tensorrt(
            _text_schedule(),
            tokenizer=SimpleNamespace(
                encode=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
            ),
            **common,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="token bound"
    ):
        preflight.preflight_tensorrt(
            _text_schedule(expected="A" * 9),
            tokenizer=_CharacterTokenizer(),
            **common,
        )


def test_multimodal_preflight_rejects_unbound_or_invalid_processor_contract() -> None:
    schedule = _multimodal_schedule()
    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "processor": _Processor(token_count=32),
        "expected_processor_sha256": "e" * 64,
        "context_length": 64,
        "max_output_tokens": 8,
        "image_resolver": lambda _record: _ClosableImage(),
    }
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="digest mismatch"
    ):
        preflight.preflight_multimodal(
            schedule, processor_digest=lambda _processor: "f" * 64, **common
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="authenticated"
    ):
        preflight.preflight_multimodal(
            schedule,
            processor_digest=lambda _processor: (_ for _ in ()).throw(RuntimeError()),
            **common,
        )
    with pytest.raises(preflight.QualificationRenderPreflightError, match="APIs"):
        preflight.preflight_multimodal(
            schedule,
            processor=object(),
            processor_digest=lambda _processor: "e" * 64,
            **{key: value for key, value in common.items() if key != "processor"},
        )


def test_gguf_prefix_rejects_unpinned_or_insufficient_statements() -> None:
    schedule = _text_schedule(expected="A")
    key = ed25519.Ed25519PrivateKey.generate()
    statement = _gguf_statement(schedule)
    signature = key.sign(preflight._canonical_json(statement))
    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "statement": statement,
        "signature": signature,
        "public_key": key.public_key(),
        "minimum_exact_matches": 1,
    }
    with pytest.raises(preflight.QualificationRenderPreflightError, match="not pinned"):
        preflight.verify_gguf_prefix(
            schedule,
            expected_signer_fingerprint="sha256:" + "0" * 64,
            **common,
        )
    altered = dict(statement)
    altered["schedule_sha256"] = "0" * 64
    with pytest.raises(preflight.QualificationRenderPreflightError, match="schedule"):
        preflight.verify_gguf_prefix(
            schedule,
            statement=altered,
            signature=key.sign(preflight._canonical_json(altered)),
            public_key=key.public_key(),
            expected_signer_fingerprint=public_key_fingerprint(key.public_key()),
            schedule_file_sha256="sha256:" + "d" * 64,
            minimum_exact_matches=1,
        )
    wrong = _gguf_statement(schedule)
    wrong["records"][0]["output_text"] = "B"  # type: ignore[index]
    wrong["records"][0]["output_sha256"] = hashlib.sha256(b"B").hexdigest()  # type: ignore[index]
    with pytest.raises(preflight.QualificationRenderPreflightError, match="minimum"):
        preflight.verify_gguf_prefix(
            schedule,
            statement=wrong,
            signature=key.sign(preflight._canonical_json(wrong)),
            public_key=key.public_key(),
            expected_signer_fingerprint=public_key_fingerprint(key.public_key()),
            schedule_file_sha256="sha256:" + "d" * 64,
            minimum_exact_matches=1,
        )


def test_schedule_and_record_helpers_reject_semantic_substitution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schedule.json"
    path.write_text("[]")
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="JSON object"
    ):
        preflight.load_bound_schedule(
            path,
            expected_file_sha256="sha256:" + hashlib.sha256(b"[]").hexdigest(),
            expected_schedule_sha256="a" * 64,
        )
    schedule = _text_schedule()
    payload = canonical_runtime_behavioral_schedule_json(schedule)
    path.write_bytes(payload)
    with pytest.raises(preflight.QualificationRenderPreflightError, match="semantic"):
        preflight.load_bound_schedule(
            path,
            expected_file_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            expected_schedule_sha256="0" * 64,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="prompt text"
    ):
        preflight._text_record(_multimodal_schedule().records[0], profile="fixture")
    with pytest.raises(preflight.QualificationRenderPreflightError, match="expected"):
        preflight._text_record(
            replace(_text_schedule().records[0], expected_output=None),
            profile="fixture",
        )
    with pytest.raises(preflight.QualificationRenderPreflightError, match="invalid"):
        preflight._result_integer({"count": True}, "count")


def test_decode_and_array_adapters_reject_runtime_shape_drift() -> None:
    class _Array:
        def tolist(self) -> list[int]:
            return [1, 2]

    class _RaisingDecoder:
        def decode(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("boom")

    class _NonTextDecoder:
        def decode(self, *args: object, **kwargs: object) -> int:
            return 1

    assert preflight._single_token_ids(_Array(), label="fixture") == [1, 2]
    with pytest.raises(preflight.QualificationRenderPreflightError, match="failed"):
        preflight._decode(
            _RaisingDecoder(), [1], skip_special_tokens=True, label="fixture"
        )
    with pytest.raises(preflight.QualificationRenderPreflightError, match="non-text"):
        preflight._decode(
            _NonTextDecoder(), [1], skip_special_tokens=True, label="fixture"
        )


def test_provider_profiles_reject_wrong_tasks_and_output_roundtrip() -> None:
    multimodal = _multimodal_schedule()
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="text_causal"
    ):
        preflight.preflight_hf_text(
            multimodal,
            schedule_file_sha256="sha256:" + "d" * 64,
            tokenizer=_CharacterTokenizer(),
            expected_tokenizer_sha256="e" * 64,
            context_length=64,
            max_output_tokens=8,
            metric="exact_match",
            tokenizer_digest=lambda _tokenizer: "e" * 64,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="text_causal"
    ):
        preflight.preflight_tensorrt(
            multimodal,
            schedule_file_sha256="sha256:" + "d" * 64,
            tokenizer=_CharacterTokenizer(),
            tokenizer_contract_sha256="e" * 64,
            engine_config=_engine_config(),
            engine_config_sha256="f" * 64,
            context_length=64,
            max_output_tokens=8,
        )
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="vision_text_generation"
    ):
        preflight.preflight_multimodal(
            _text_schedule(),
            schedule_file_sha256="sha256:" + "d" * 64,
            processor=_Processor(token_count=32),
            expected_processor_sha256="e" * 64,
            context_length=64,
            max_output_tokens=8,
            processor_digest=lambda _processor: "e" * 64,
            image_resolver=lambda _record: _ClosableImage(),
        )
    with pytest.raises(preflight.QualificationRenderPreflightError, match="round-trip"):
        preflight.preflight_hf_text(
            _text_schedule(),
            schedule_file_sha256="sha256:" + "d" * 64,
            tokenizer=_BrokenContinuationTokenizer(),
            expected_tokenizer_sha256="e" * 64,
            context_length=64,
            max_output_tokens=8,
            metric="exact_match",
            tokenizer_digest=lambda _tokenizer: "e" * 64,
        )


def test_multimodal_rendering_failures_close_images_and_fail_closed() -> None:
    schedule = _multimodal_schedule()

    class _BadProcessor(_Processor):
        def __init__(self, *, mode: str) -> None:
            super().__init__(token_count=32)
            self.mode = mode

        def apply_chat_template(self, messages: object, **kwargs: object) -> object:
            if self.mode == "render_error":
                raise RuntimeError("boom")
            if self.mode == "empty_render":
                return ""
            return super().apply_chat_template(messages, **kwargs)

        def __call__(self, **kwargs: object) -> object:
            if self.mode == "encode_error":
                raise RuntimeError("boom")
            if self.mode == "no_ids":
                return {}
            return super().__call__(**kwargs)

    common = {
        "schedule_file_sha256": "sha256:" + "d" * 64,
        "expected_processor_sha256": "e" * 64,
        "context_length": 64,
        "max_output_tokens": 8,
        "processor_digest": lambda _processor: "e" * 64,
    }
    for mode, message in (
        ("render_error", "rendering failed"),
        ("empty_render", "no text"),
        ("encode_error", "encoding failed"),
        ("no_ids", "no input_ids"),
    ):
        image = _ClosableImage()
        with pytest.raises(preflight.QualificationRenderPreflightError, match=message):
            preflight.preflight_multimodal(
                schedule,
                processor=_BadProcessor(mode=mode),
                image_resolver=lambda _record, image=image: image,
                **common,
            )
        if mode in {"encode_error", "no_ids"}:
            assert image.closed is True


def test_gguf_statement_shape_and_thresholds_fail_closed() -> None:
    schedule = _text_schedule(expected="A")
    key = ed25519.Ed25519PrivateKey.generate()
    fingerprint = public_key_fingerprint(key.public_key())

    def verify(statement: Mapping[str, object], minimum: int = 1) -> None:
        preflight.verify_gguf_prefix(
            schedule,
            schedule_file_sha256="sha256:" + "d" * 64,
            statement=statement,
            signature=key.sign(preflight._canonical_json(statement)),
            public_key=key.public_key(),
            expected_signer_fingerprint=fingerprint,
            minimum_exact_matches=minimum,
        )

    invalid = _gguf_statement(schedule)
    invalid["format_version"] = "other"
    with pytest.raises(preflight.QualificationRenderPreflightError, match="invalid"):
        verify(invalid)
    invalid = _gguf_statement(schedule)
    invalid["prefix_record_count"] = 2
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="record count"
    ):
        verify(invalid)
    with pytest.raises(preflight.QualificationRenderPreflightError, match="exceeds"):
        verify(_gguf_statement(schedule), minimum=2)
    invalid = _gguf_statement(schedule)
    invalid["records"] = [{}]
    with pytest.raises(preflight.QualificationRenderPreflightError, match="record 0"):
        verify(invalid)
    invalid = _gguf_statement(schedule)
    invalid["records"][0]["output_text"] = None  # type: ignore[index]
    invalid["records"][0]["output_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(
        preflight.QualificationRenderPreflightError, match="output is invalid"
    ):
        verify(invalid)
