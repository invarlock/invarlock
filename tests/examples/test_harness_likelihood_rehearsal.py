"""Adversarial validation of the standalone, real Harness capture boundary."""

import importlib.util
from pathlib import Path

import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes


@pytest.fixture
def rehearsal():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/captured-results/harness_likelihood_rehearsal.py"
    )
    spec = importlib.util.spec_from_file_location("harness_rehearsal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_binding_and_native_pair_preserve_multibyte_reference(rehearsal):
    case = {"id": "unicode", "input": "A", "expected": " café"}
    assert rehearsal.canonical_bytes(case) == canonical_json_bytes(case)
    manifest = {
        "model": {"artifact_digest": "sha256:" + "a" * 64},
        "tokenizer": {"tokenizer_digest": "sha256:" + "b" * 64},
        "configuration_digest": "sha256:" + "c" * 64,
    }
    tokens = {
        "context_token_ids": [1],
        "continuation_token_ids": [2, 3],
        "joined_token_ids": [1, 2, 3],
        "decoded_continuation": " café",
    }
    rehearsal.validate_case(case)
    rehearsal.validate_encoding([1], [2, 3], [1, 2, 3], " café", case["expected"], 2)
    row = rehearsal.capture_record(case, (-5.25, False), tokens, manifest, 0)
    assert row["input"] == "A"
    assert row["expected"] == " café"
    assert row["output"] is None
    assert row["likelihood"]["utf8_byte_count"] == 6
    assert row["likelihood"]["token_count"] == 2
    assert row["context"]["harness"]["result"] == [-5.25, False]
    assert row["likelihood"]["input_digest"] == rehearsal.digest("A")
    assert row["likelihood"]["reference_digest"] == rehearsal.digest(" café")


@pytest.mark.parametrize(
    "context", ["", "ends ", "ends\n", "ends\t", "ends\u00a0", None]
)
def test_context_boundary_rejects_harness_whitespace_transfer(rehearsal, context):
    with pytest.raises(ValueError, match="context"):
        rehearsal.validate_case({"id": "case", "input": context, "expected": " answer"})


@pytest.mark.parametrize("reference", ["", None, {"answer": "text"}])
def test_missing_or_structured_reference_fails(rehearsal, reference):
    with pytest.raises(ValueError, match="reference"):
        rehearsal.validate_case({"id": "case", "input": "A", "expected": reference})


@pytest.mark.parametrize(
    "context,continuation,joined,decoded,limit",
    [
        ([1], [], [1], " answer", 128),
        ([True], [2], [True, 2], " answer", 128),
        ([1], [2], [3, 2], " answer", 128),
        ([1], [2], [1, 2], "answer", 128),
        ([1, 2], [3, 4], [1, 2, 3, 4], " answer", 2),
    ],
)
def test_token_coverage_and_truncation_fail_closed(
    rehearsal, context, continuation, joined, decoded, limit
):
    with pytest.raises(ValueError):
        rehearsal.validate_encoding(
            context, continuation, joined, decoded, " answer", limit
        )


@pytest.mark.parametrize(
    "result",
    [
        None,
        [1.0, False],
        [float("nan"), False],
        [float("-inf"), False],
        [-1.0, 0],
        [True, True],
        [-1],
    ],
)
def test_invalid_native_result_rejected_before_capture(rehearsal, result):
    with pytest.raises(ValueError):
        rehearsal.capture_record({}, result, {}, {}, 0)


def test_output_reuse_fails_before_import_or_model_calls(rehearsal, tmp_path):
    with pytest.raises(FileExistsError, match="must not already exist"):
        rehearsal.run(tmp_path / "missing-model", tmp_path)


def test_socket_audit_guard_blocks_network(rehearsal):
    for event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        with pytest.raises(RuntimeError, match="network"):
            rehearsal.block_network(event, ())
    rehearsal.block_network("open", ())


def test_fixed_cases_are_bounded_and_distinct(rehearsal):
    assert 4 <= len(rehearsal.CASES) <= 8
    assert len({row["id"] for row in rehearsal.CASES}) == len(rehearsal.CASES)
    assert len({row["input"] for row in rehearsal.CASES}) == len(rehearsal.CASES)
    assert len({row["expected"] for row in rehearsal.CASES}) == len(rehearsal.CASES)
    for case in rehearsal.CASES:
        rehearsal.validate_case(case)
    assert any(not row["expected"].isascii() for row in rehearsal.CASES)


def test_inventory_rejects_extra_or_changed_pinned_files(
    rehearsal, tmp_path, monkeypatch
):
    import hashlib
    import json

    config = {
        "model_type": "gpt2",
        "n_embd": 2,
        "n_head": 2,
        "n_layer": 2,
        "n_positions": 1024,
        "vocab_size": 50257,
    }
    for name in rehearsal.MODEL_FILES:
        (tmp_path / name).write_bytes(
            json.dumps(config).encode() if name == "config.json" else b"fixture bytes"
        )
    hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in rehearsal.MODEL_FILES
    }
    monkeypatch.setattr(rehearsal, "MODEL_HASHES", hashes)
    facts = rehearsal.inventory(tmp_path)
    assert [fact["path"] for fact in facts] == list(rehearsal.MODEL_FILES)
    (tmp_path / "model.safetensors").write_bytes(b"alternate selection")
    with pytest.raises(ValueError, match="exactly"):
        rehearsal.inventory(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "pytorch_model.bin").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="differs"):
        rehearsal.inventory(tmp_path)


def test_inventory_size_cap_before_reading_weights(rehearsal, tmp_path, monkeypatch):
    for name in rehearsal.MODEL_FILES:
        (tmp_path / name).write_bytes(b"too large")
    monkeypatch.setitem(rehearsal.CONFIGURATION, "maximum_model_file_bytes", 1)
    with pytest.raises(ValueError, match="size cap"):
        rehearsal.inventory(tmp_path)


def test_invalid_case_id_and_text_cap(rehearsal):
    with pytest.raises(ValueError, match="ID"):
        rehearsal.validate_case({"id": "", "input": "A", "expected": "B"})
    with pytest.raises(ValueError, match="size cap"):
        rehearsal.validate_case({"id": "case", "input": "A" * 4096, "expected": "B"})


def test_deadline_fails_closed(rehearsal):
    with pytest.raises(TimeoutError, match="wall-clock"):
        rehearsal.deadline(None, None)


@pytest.fixture
def orchestration(rehearsal, tmp_path, monkeypatch):
    """Isolate launcher control flow; these doubles never supply public evidence."""
    import contextlib
    import hashlib
    import sys
    from types import SimpleNamespace

    implementation = tmp_path / "implementation.py"
    implementation.write_bytes(b"unit-test dependency implementation")
    expected_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
    monkeypatch.setattr(
        rehearsal, "UPSTREAM_HASHES", {"test_dependency": expected_hash}
    )
    monkeypatch.setattr(
        rehearsal.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(implementation)),
    )
    monkeypatch.setattr(rehearsal.importlib.metadata, "version", lambda name: "0.4.12")
    monkeypatch.setattr(rehearsal.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(rehearsal.signal, "signal", lambda *args: None)
    monkeypatch.setattr(rehearsal.signal, "alarm", lambda *args: None)
    # Restore the process flag changed by the production standalone launcher.
    monkeypatch.setattr(rehearsal.sys, "dont_write_bytecode", False)
    monkeypatch.setattr(
        rehearsal,
        "inventory",
        lambda model: [
            {"path": "vocab.json", "byte_size": 1, "sha256": "sha256:" + "a" * 64}
        ],
    )
    state = SimpleNamespace(instances=[], mode=None)

    class LocalControlFlowDouble:
        def __init__(self, **kwargs):
            state.instances.append(self)
            self.backend = "causal"
            self.max_length = kwargs["max_length"]
            self.logits_cache = state.mode == "cache"
            self.cache_hook = SimpleNamespace(dbdict=None)
            self.model = SimpleNamespace(
                parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cpu"))]
            )
            self.tokenizer = SimpleNamespace(
                decode=lambda ids, **kwargs: "".join(map(chr, ids))
            )

        def tok_encode(self, text):
            return list(map(ord, text))

        def _encode_pair(self, context, continuation):
            context_ids = self.tok_encode(context)
            if state.mode == "changed_context":
                context_ids = [999]
            return context_ids, self.tok_encode(continuation)

        def loglikelihood(self, requests, **kwargs):
            if state.mode == "missing_results":
                return []
            return [(-3.5, False) for _ in requests]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda n: None,
            use_deterministic_algorithms=lambda b: None,
            manual_seed=lambda n: None,
            inference_mode=contextlib.nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.api.instance",
        SimpleNamespace(Instance=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.models.huggingface",
        SimpleNamespace(HFLM=LocalControlFlowDouble),
    )
    return state


def test_launcher_independent_calls_and_immutable_publication(
    rehearsal, orchestration, tmp_path
):
    import json

    output = tmp_path / "output"
    manifest, raw = rehearsal.run(tmp_path / "model", output)
    assert len(orchestration.instances) == 2
    assert orchestration.instances[0] is not orchestration.instances[1]
    assert manifest["configuration_digest"] == rehearsal.digest(
        manifest["configuration"]
    )
    for name, expected in (("manifest.json", manifest), ("raw-results.json", raw)):
        path = output / name
        assert path.read_bytes() == rehearsal.canonical_bytes(expected)
        assert json.loads(path.read_bytes()) == expected
        assert path.stat().st_mode & 0o222 == 0
    assert list(raw["runs"]) == ["baseline", "subject"]
    assert [row["id"] for row in raw["runs"]["baseline"]["records"]] == [
        case["id"] for case in rehearsal.CASES
    ]


@pytest.mark.parametrize(
    "mode,match",
    [
        ("cache", "honor"),
        ("changed_context", "altered"),
        ("missing_results", "omitted"),
    ],
)
def test_launcher_fails_closed_before_publication(
    rehearsal, orchestration, tmp_path, mode, match
):
    orchestration.mode = mode
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=match):
        rehearsal.run(tmp_path / "model", output)
    assert not output.exists()


def test_launcher_rejects_modified_upstream(
    rehearsal, orchestration, tmp_path, monkeypatch
):
    monkeypatch.setattr(rehearsal, "UPSTREAM_HASHES", {"test_dependency": "0" * 64})
    with pytest.raises(ValueError, match="upstream"):
        rehearsal.run(tmp_path / "model", tmp_path / "output")
    assert not orchestration.instances


def test_launcher_rejects_wrong_package(
    rehearsal, orchestration, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        rehearsal.importlib.metadata, "version", lambda name: "modified"
    )
    with pytest.raises(ValueError, match="unmodified"):
        rehearsal.run(tmp_path / "model", tmp_path / "output")
    assert not orchestration.instances


def test_launcher_rejects_inventory_mutation(
    rehearsal, orchestration, tmp_path, monkeypatch
):
    snapshots = iter([[{"path": "vocab.json"}], [{"path": "changed"}]])
    monkeypatch.setattr(rehearsal, "inventory", lambda model: next(snapshots))
    with pytest.raises(ValueError, match="changed during"):
        rehearsal.run(tmp_path / "model", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_argument_handoff(rehearsal, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        rehearsal.sys,
        "argv",
        ["capture", "--model", str(tmp_path), "--output", str(tmp_path / "out")],
    )
    monkeypatch.setattr(
        rehearsal,
        "run",
        lambda model, output: (
            {"model": {"artifact_digest": "pinned"}},
            {"runs": {"baseline": {}, "subject": {}}},
        ),
    )
    rehearsal.main()
    assert '"artifact_digest": "pinned"' in capsys.readouterr().out


def test_inventory_configuration_cap_is_independent_of_file_pin(
    rehearsal, tmp_path, monkeypatch
):
    import hashlib

    for name in rehearsal.MODEL_FILES:
        (tmp_path / name).write_bytes(b'{"model_type":"gpt2","n_embd":4096}')
    monkeypatch.setattr(
        rehearsal,
        "MODEL_HASHES",
        {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in rehearsal.MODEL_FILES
        },
    )
    with pytest.raises(ValueError, match="tiny GPT-2"):
        rehearsal.inventory(tmp_path)
