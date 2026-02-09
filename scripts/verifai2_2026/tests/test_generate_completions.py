from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.verifai2_2026 import generate_completions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


class _NoGrad:
    def __enter__(self):  # noqa: ANN001
        return None

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Cuda:
    def __init__(self, *, available: bool) -> None:
        self._available = bool(available)
        self.seeded: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.seeded.append(int(seed))


class _Tensor:
    def __init__(self, n_rows: int, n_cols: int) -> None:
        self._shape = (int(n_rows), int(n_cols))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def to(self, _device: Any) -> _Tensor:
        return self

    def __getitem__(self, _key: Any) -> _Tensor:
        # Used for seq[:, in_len:]
        return _Tensor(self._shape[0], 3)


class _Model:
    def __init__(self, *, device: str = "cuda:0") -> None:
        self.device = device
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> _Tensor:
        input_ids = kwargs.get("input_ids")
        assert isinstance(input_ids, _Tensor)
        n_ret = int(kwargs.get("num_return_sequences", 1))
        return _Tensor(input_ids.shape[0] * n_ret, input_ids.shape[1] + 3)


class _Tokenizer:
    eos_token_id = 0
    eos_token = "<eos>"

    def __init__(self, *, bad_decode_len: bool = False) -> None:
        self.pad_token = None
        self.padding_side = "right"
        self._bad_decode_len = bool(bad_decode_len)

    def __call__(self, batch_prompts: list[str], **_kwargs: Any) -> dict[str, _Tensor]:
        return {"input_ids": _Tensor(len(batch_prompts), 5)}

    def batch_decode(self, gen_tokens: _Tensor, **_kwargs: Any) -> list[str]:
        n = gen_tokens.shape[0]
        if self._bad_decode_len:
            n = max(0, n - 1)
        return [f"c{i}" for i in range(n)]


class _AutoModelForCausalLM:
    def __init__(self, model: _Model) -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    def from_pretrained(self, *args: Any, **kwargs: Any) -> _Model:  # noqa: D401
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._model


class _AutoTokenizer:
    def __init__(self, tok: _Tokenizer) -> None:
        self._tok = tok
        self.calls: list[dict[str, Any]] = []

    def from_pretrained(self, *args: Any, **kwargs: Any) -> _Tokenizer:  # noqa: D401
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._tok


def test_read_jsonl_skips_blank_and_rejects_invalid(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text("\n" + json.dumps({"a": 1}) + "\n", encoding="utf-8")
    assert len(generate_completions._read_jsonl(p)) == 1

    p2 = tmp_path / "bad.jsonl"
    p2.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        generate_completions._read_jsonl(p2)

    p3 = tmp_path / "bad2.jsonl"
    p3.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected JSON object"):
        generate_completions._read_jsonl(p3)


def test_parse_device() -> None:
    assert generate_completions._parse_device("cpu") == ("cpu", None)
    assert generate_completions._parse_device("cuda") == ("cuda", 0)
    assert generate_completions._parse_device("cuda:3") == ("cuda", 3)
    with pytest.raises(ValueError, match=r"Invalid --device"):
        generate_completions._parse_device("cudaX")
    with pytest.raises(ValueError, match=r"Invalid --device"):
        generate_completions._parse_device("cuda:nope")
    with pytest.raises(ValueError, match=r"Invalid --device"):
        generate_completions._parse_device("tpu:0")


def test_validate_decoding_and_gen_kwargs() -> None:
    assert (
        generate_completions._validate_decoding(
            method="greedy",
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_samples=1,
            num_beams=1,
        )
        == []
    )
    assert generate_completions._validate_decoding(
        method="sample",
        temperature=0.0,
        top_p=0.95,
        top_k=0,
        num_samples=1,
        num_beams=0,
    )
    # Cover: sample num_samples<1 error path.
    assert generate_completions._validate_decoding(
        method="sample",
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        num_samples=0,
        num_beams=0,
    )
    assert generate_completions._validate_decoding(
        method="beam",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_samples=2,
        num_beams=1,
    )
    # Cover: greedy top_p/top_k/num_samples/num_beams error paths.
    assert generate_completions._validate_decoding(
        method="greedy",
        temperature=0.1,
        top_p=0.5,
        top_k=1,
        num_samples=2,
        num_beams=2,
    )
    # Cover: beam num_beams>=2 branch and num_samples in {0,1} branch.
    assert (
        generate_completions._validate_decoding(
            method="beam",
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_samples=1,
            num_beams=2,
        )
        == []
    )

    tok = _Tokenizer()
    g = generate_completions._gen_kwargs(
        tok=tok,
        method="greedy",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=4,
        num_samples=1,
        num_beams=0,
    )
    assert g["do_sample"] is False
    s = generate_completions._gen_kwargs(
        tok=tok,
        method="sample",
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        max_new_tokens=4,
        num_samples=3,
        num_beams=0,
    )
    assert s["do_sample"] is True
    b = generate_completions._gen_kwargs(
        tok=tok,
        method="beam",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=4,
        num_samples=1,
        num_beams=2,
    )
    assert b["num_beams"] == 2
    with pytest.raises(ValueError, match=r"Unknown decoding method"):
        generate_completions._gen_kwargs(
            tok=tok,
            method="nope",
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=4,
            num_samples=1,
            num_beams=0,
        )


def test_torch_dtype_and_seeding() -> None:
    cuda = _Cuda(available=False)
    torch = type(
        "TorchMod",
        (),
        {},
    )()
    torch.float16 = "f16"
    torch.bfloat16 = "bf16"
    torch.float32 = "f32"
    torch.cuda = cuda
    torch.manual_seed = lambda _s: None  # noqa: E731

    assert generate_completions._torch_dtype(torch, "fp16") == "f16"
    assert generate_completions._torch_dtype(torch, "bf16") == "bf16"
    assert generate_completions._torch_dtype(torch, "fp32") == "f32"
    with pytest.raises(ValueError, match=r"Invalid --dtype"):
        generate_completions._torch_dtype(torch, "int8")

    generate_completions._seed_everything(torch, 123)
    assert cuda.seeded == []

    cuda2 = _Cuda(available=True)
    torch2 = type(
        "TorchMod2",
        (),
        {},
    )()
    torch2.float16 = "f16"
    torch2.bfloat16 = "bf16"
    torch2.float32 = "f32"
    torch2.cuda = cuda2
    torch2.manual_seed = lambda _s: None  # noqa: E731
    generate_completions._seed_everything(torch2, 7)
    assert cuda2.seeded == [7]


def test_import_helpers_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(name: str):  # noqa: ANN001
        raise ImportError(name)

    monkeypatch.setattr(generate_completions.importlib, "import_module", _boom)

    with pytest.raises(RuntimeError, match=r"torch is required"):
        generate_completions._import_torch()
    with pytest.raises(RuntimeError, match=r"transformers is required"):
        generate_completions._import_transformers()


def test_load_hf_cpu_and_cuda_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = _Tokenizer()
    model = _Model(device="cpu")
    am = _AutoModelForCausalLM(model)
    at = _AutoTokenizer(tok)

    def _imp(name: str):  # noqa: ANN001
        if name == "transformers":
            return type(
                "Tr",
                (),
                {"AutoModelForCausalLM": am, "AutoTokenizer": at},
            )()
        raise AssertionError("unexpected import")

    monkeypatch.setattr(generate_completions.importlib, "import_module", _imp)

    torch = type(
        "Torch",
        (),
        {"float16": "f16", "bfloat16": "bf16", "float32": "f32"},
    )()

    m1, t1 = generate_completions._load_hf(
        torch=torch,
        model="m",
        revision="",
        tokenizer="tok",
        tokenizer_revision="rev",
        device="cpu",
        dtype="fp16",
        trust_remote_code=False,
    )
    assert m1 is model
    assert t1 is tok
    assert tok.pad_token == tok.eos_token
    assert tok.padding_side == "left"
    assert model.eval_called
    assert am.calls[-1]["kwargs"]["device_map"] == {"": "cpu"}

    # Cover: tok.pad_token already set branch.
    tok2 = _Tokenizer()
    tok2.pad_token = "PAD"
    at2 = _AutoTokenizer(tok2)

    def _imp_tok2(name: str):  # noqa: ANN001
        if name == "transformers":
            return type(
                "Tr",
                (),
                {"AutoModelForCausalLM": am, "AutoTokenizer": at2},
            )()
        raise AssertionError("unexpected import")

    monkeypatch.setattr(generate_completions.importlib, "import_module", _imp_tok2)
    _m_tok2, t_tok2 = generate_completions._load_hf(
        torch=torch,
        model="m",
        revision="",
        tokenizer="tok",
        tokenizer_revision="rev",
        device="cpu",
        dtype="fp16",
        trust_remote_code=False,
    )
    assert t_tok2.pad_token == "PAD"

    model2 = _Model(device="cuda:0")
    am2 = _AutoModelForCausalLM(model2)
    at2 = _AutoTokenizer(_Tokenizer())

    def _imp2(name: str):  # noqa: ANN001
        if name == "transformers":
            return type(
                "Tr",
                (),
                {"AutoModelForCausalLM": am2, "AutoTokenizer": at2},
            )()
        raise AssertionError("unexpected import")

    monkeypatch.setattr(generate_completions.importlib, "import_module", _imp2)
    m2, _t2 = generate_completions._load_hf(
        torch=torch,
        model="m",
        revision="abc",
        tokenizer="tok",
        tokenizer_revision="rev",
        device="cuda:3",
        dtype="bf16",
        trust_remote_code=True,
    )
    assert m2 is model2
    assert am2.calls[-1]["kwargs"]["device_map"] == {"": 3}
    assert am2.calls[-1]["kwargs"]["revision"] == "abc"
    assert am2.calls[-1]["kwargs"]["torch_dtype"] == "bf16"
    assert am2.calls[-1]["kwargs"]["trust_remote_code"] is True


def test_main_smoke_and_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "p1"}, {"id": "b", "prompt": "p2"}])
    out = tmp_path / "out.jsonl"

    # Stub torch/transformers for the full main() path.
    cuda = _Cuda(available=True)
    torch_mod = type("TorchMod", (), {})()
    torch_mod.float16 = "f16"
    torch_mod.bfloat16 = "bf16"
    torch_mod.float32 = "f32"
    torch_mod.cuda = cuda
    torch_mod.manual_seed = lambda _s: None  # noqa: E731
    torch_mod.no_grad = lambda: _NoGrad()  # noqa: E731

    tok_ok = _Tokenizer()
    model = _Model()
    am = _AutoModelForCausalLM(model)
    at = _AutoTokenizer(tok_ok)

    def _imp(name: str):  # noqa: ANN001
        if name == "torch":
            return torch_mod
        if name == "transformers":
            return type(
                "Tr",
                (),
                {"AutoModelForCausalLM": am, "AutoTokenizer": at},
            )()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(generate_completions.importlib, "import_module", _imp)

    rc = generate_completions.main(
        [
            "--tasks",
            str(tasks),
            "--out",
            str(out),
            "--model",
            "m",
            "--revision",
            "",
            "--tokenizer",
            "tok",
            "--tokenizer-revision",
            "rev",
            "--device",
            "cuda:0",
            "--dtype",
            "fp16",
            "--batch-size",
            "1",
            "--decoding-method",
            "sample",
            "--temperature",
            "0.2",
            "--top-p",
            "0.95",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "42",
            "--num-samples",
            "2",
            "--limit",
            "1",
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2  # limit=1 and num_samples=2

    # Size mismatch triggers RuntimeError.
    tok_bad = _Tokenizer(bad_decode_len=True)
    at_bad = _AutoTokenizer(tok_bad)

    def _imp_bad(name: str):  # noqa: ANN001
        if name == "torch":
            return torch_mod
        if name == "transformers":
            return type(
                "Tr",
                (),
                {"AutoModelForCausalLM": am, "AutoTokenizer": at_bad},
            )()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(generate_completions.importlib, "import_module", _imp_bad)
    with pytest.raises(RuntimeError, match=r"Unexpected generate\(\) output size"):
        generate_completions.main(
            [
                "--tasks",
                str(tasks),
                "--out",
                str(out),
                "--model",
                "m",
                "--decoding-method",
                "sample",
                "--temperature",
                "0.2",
                "--top-p",
                "0.95",
                "--top-k",
                "0",
                "--max-new-tokens",
                "4",
                "--seed",
                "42",
                "--num-samples",
                "2",
            ]
        )


def test_main_decoding_or_task_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"prompt": "x"}])
    out = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match=r"Task missing id field"):
        generate_completions.main(
            [
                "--tasks",
                str(tasks),
                "--out",
                str(out),
                "--model",
                "m",
                "--decoding-method",
                "greedy",
                "--temperature",
                "0",
                "--top-p",
                "1",
                "--top-k",
                "0",
                "--max-new-tokens",
                "4",
                "--seed",
                "1",
            ]
        )

    tasks2 = tmp_path / "tasks2.jsonl"
    _write_jsonl(tasks2, [{"id": "a", "prompt": "x"}])
    rc = generate_completions.main(
        [
            "--tasks",
            str(tasks2),
            "--out",
            str(out),
            "--model",
            "m",
            "--decoding-method",
            "greedy",
            "--temperature",
            "0.1",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "1",
        ]
    )
    assert rc == 2
    assert "requires temperature=0.0" in capsys.readouterr().err


def test_main_no_tasks_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("", encoding="utf-8")
    out = tmp_path / "out.jsonl"

    rc = generate_completions.main(
        [
            "--tasks",
            str(tasks),
            "--out",
            str(out),
            "--model",
            "m",
            "--decoding-method",
            "greedy",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "1",
        ]
    )
    assert rc == 2
    assert "No tasks found." in capsys.readouterr().err


def test_main_rejects_missing_prompt_field(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": None}])
    out = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match=r"missing prompt field"):
        generate_completions.main(
            [
                "--tasks",
                str(tasks),
                "--out",
                str(out),
                "--model",
                "m",
                "--decoding-method",
                "greedy",
                "--temperature",
                "0",
                "--top-p",
                "1",
                "--top-k",
                "0",
                "--max-new-tokens",
                "4",
                "--seed",
                "1",
            ]
        )
