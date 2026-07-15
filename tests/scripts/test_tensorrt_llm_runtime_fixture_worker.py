from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.scripts._tensorrt_llm_fixture_support import load_script

worker = load_script("tensorrt_llm_runtime_fixture_worker")


def test_build_uses_pinned_tp1_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(worker, "_tokenizer_contract", lambda _model: b"{}")

    def fake_run(command: tuple[str, ...], *, timeout: int) -> bytes:
        assert timeout == 3600
        commands.append(tuple(command))
        if command[0] == "trtllm-build":
            engine = Path(command[command.index("--output_dir") + 1])
            engine.mkdir()
            (engine / "config.json").write_text("{}", encoding="utf-8")
            (engine / "rank0.engine").write_bytes(b"engine")
        return b""

    monkeypatch.setattr(worker, "_bounded_run", fake_run)
    result = worker.build_fixture(
        model=model,
        output=output,
        repository=worker.MODEL_REPOSITORY,
        revision=worker.MODEL_REVISION,
    )
    assert result["ok"] is True
    assert len(commands) == 2
    conversion, build = commands
    assert conversion[conversion.index("--dtype") + 1] == "float16"
    assert conversion[conversion.index("--tp_size") + 1] == "1"
    assert build[build.index("--max_batch_size") + 1] == "1"
    assert build[build.index("--max_input_len") + 1] == "8"
    assert build[build.index("--max_seq_len") + 1] == "9"
    assert build[build.index("--max_num_tokens") + 1] == "9"
    assert build[build.index("--opt_num_tokens") + 1] == "8"
    assert build[build.index("--output_timing_cache") + 1] == (
        "/tmp/invarlock-tensorrt-llm-model.cache"
    )


def test_build_rejects_unpinned_source(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(worker.FixtureWorkerError, match="pinned"):
        worker.build_fixture(
            model=model,
            output=tmp_path / "output",
            repository="other/model",
            revision=worker.MODEL_REVISION,
        )


def test_probe_uses_closed_runner_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["request"] = json.loads(kwargs["input"])
        return SimpleNamespace(
            returncode=0,
            stderr=b"",
            stdout=json.dumps(
                {
                    "format_version": worker.RESPONSE_FORMAT,
                    "output_text": "token",
                }
            ).encode(),
        )

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    result = worker.probe_fixture(engine=Path("/engine"), tokenizer=Path("/tokenizer"))
    request = captured["request"]
    assert captured["command"] == (str(worker.RUNNER), "--invarlock-score-v1")
    assert request["input_text"] == worker.PROMPT
    assert request["settings"] == {
        "allow_network": False,
        "batch_size": 1,
        "context_length": 8,
        "max_output_tokens": 1,
        "seed": 0,
        "timeout_seconds": 300,
    }
    assert result["output_text"] == "token"


def test_probe_rejects_open_response_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr=b"",
            stdout=b'{"format_version":"invarlock/tensorrt-llm-runner-response-v1",'
            b'"output_text":"token","extra":true}',
        ),
    )
    with pytest.raises(worker.FixtureWorkerError, match="schema"):
        worker.probe_fixture(engine=Path("/engine"), tokenizer=Path("/tokenizer"))


def test_tokenizer_contract_is_canonical_and_uses_eos_as_pad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = SimpleNamespace(to_str=lambda: '{"model":{"type":"BPE"}}')
    tokenizer = SimpleNamespace(
        backend_tokenizer=backend,
        eos_token_id=2,
        pad_token_id=None,
    )
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    payload = worker._tokenizer_contract(tmp_path)
    assert payload == worker._canonical_json(json.loads(payload))
    decoded = json.loads(payload)
    assert decoded["eos_token_id"] == 2
    assert decoded["pad_token_id"] == 2
    assert decoded["add_special_tokens"] is False
    assert decoded["skip_special_tokens"] is True


@pytest.mark.parametrize(
    "tokenizer",
    [
        SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"),
            eos_token_id=2,
            pad_token_id=2,
        ),
        SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: '{"model":{}}'),
            eos_token_id=None,
            pad_token_id=2,
        ),
        SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: '{"model":{}}'),
            eos_token_id=2,
            pad_token_id=-1,
        ),
    ],
)
def test_tokenizer_contract_rejects_invalid_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: SimpleNamespace
) -> None:
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(worker.FixtureWorkerError):
        worker._tokenizer_contract(tmp_path)


def test_bounded_run_enforces_status_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"ok", stderr=b""
        ),
    )
    assert worker._bounded_run(("tool",), timeout=1) == b"ok"
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9, stdout=b"", stderr=b"bad"
        ),
    )
    with pytest.raises(worker.FixtureWorkerError, match="status 9"):
        worker._bounded_run(("tool",), timeout=1)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"x" * (worker._MAX_OUTPUT + 1), stderr=b""
        ),
    )
    with pytest.raises(worker.FixtureWorkerError, match="output limit"):
        worker._bounded_run(("tool",), timeout=1)


def test_main_dispatches_build_and_probe(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        worker,
        "build_fixture",
        lambda **_kwargs: {"format_version": worker.BUILD_RESULT_FORMAT, "ok": True},
    )
    assert (
        worker.main(
            [
                "build",
                "--model",
                "/model",
                "--output",
                "/output",
                "--repository",
                worker.MODEL_REPOSITORY,
                "--revision",
                worker.MODEL_REVISION,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    monkeypatch.setattr(
        worker,
        "probe_fixture",
        lambda **_kwargs: {"format_version": worker.PROBE_RESULT_FORMAT, "ok": True},
    )
    assert (
        worker.main(["probe", "--engine", "/engine", "--tokenizer", "/tokenizer"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_main_reports_worker_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def fail(**_kwargs: object):
        raise worker.FixtureWorkerError("closed")

    monkeypatch.setattr(worker, "probe_fixture", fail)
    assert (
        worker.main(["probe", "--engine", "/engine", "--tokenizer", "/tokenizer"]) == 2
    )
    assert "failed" in capsys.readouterr().err


def test_worker_io_and_local_model_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(worker.FixtureWorkerError, match="directory"):
        worker._require_new_directory(existing)
    target = tmp_path / "target"
    target.write_bytes(b"exists")
    with pytest.raises(worker.FixtureWorkerError, match="written safely"):
        worker._write_new(target, b"new")
    with pytest.raises(worker.FixtureWorkerError, match="unavailable"):
        worker._validate_local_model(tmp_path / "missing")
    with pytest.raises(worker.FixtureWorkerError, match="directory"):
        worker._validate_local_model(target)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("closed")),
    )
    with pytest.raises(worker.FixtureWorkerError, match="subprocess failed"):
        worker._bounded_run(("tool",), timeout=1)


def test_tokenizer_loader_failure_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_a, **_k: (_ for _ in ()).throw(OSError("closed"))
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(worker.FixtureWorkerError, match="cannot be loaded"):
        worker._tokenizer_contract(tmp_path)


def test_build_rejects_missing_or_open_engine_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(worker, "_tokenizer_contract", lambda _model: b"{}")
    monkeypatch.setattr(worker, "_bounded_run", lambda *_a, **_k: b"")
    with pytest.raises(worker.FixtureWorkerError, match="cannot be inspected"):
        worker.build_fixture(
            model=model,
            output=tmp_path / "first",
            repository=worker.MODEL_REPOSITORY,
            revision=worker.MODEL_REVISION,
        )

    def make_open_layout(command: tuple[str, ...], **_kwargs: object) -> bytes:
        if command[0] == "trtllm-build":
            engine = Path(command[command.index("--output_dir") + 1])
            engine.mkdir()
            (engine / "config.json").write_text("{}", encoding="utf-8")
            (engine / "rank0.engine").write_bytes(b"engine")
            (engine / "extra").write_bytes(b"bad")
        return b""

    monkeypatch.setattr(worker, "_bounded_run", make_open_layout)
    with pytest.raises(worker.FixtureWorkerError, match="closed TP=1"):
        worker.build_fixture(
            model=model,
            output=tmp_path / "second",
            repository=worker.MODEL_REPOSITORY,
            revision=worker.MODEL_REVISION,
        )


def test_probe_translates_execution_and_response_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("closed")),
    )
    with pytest.raises(worker.FixtureWorkerError, match="probe failed"):
        worker.probe_fixture(engine=Path("/engine"), tokenizer=Path("/tokenizer"))
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=2, stderr=b"bad", stdout=b""),
    )
    with pytest.raises(worker.FixtureWorkerError, match="complete cleanly"):
        worker.probe_fixture(engine=Path("/engine"), tokenizer=Path("/tokenizer"))
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stderr=b"", stdout=b"not-json"),
    )
    with pytest.raises(worker.FixtureWorkerError, match="response is invalid"):
        worker.probe_fixture(engine=Path("/engine"), tokenizer=Path("/tokenizer"))
