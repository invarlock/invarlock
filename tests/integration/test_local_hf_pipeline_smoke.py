from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def _require_local_hf_runtime() -> tuple[object, object, object]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except Exception as exc:  # pragma: no cover - host dependency guard
        message = f"local HF runtime is unavailable: {exc}"
        if os.environ.get("INVARLOCK_REQUIRE_LOCAL_HF") in {"1", "true", "yes", "on"}:
            pytest.fail(message)
        pytest.skip(message)

    return (
        GPT2Config,
        GPT2LMHeadModel,
        (Tokenizer, WordLevel, Whitespace, PreTrainedTokenizerFast),
    )


def test_local_hf_runtime_requirement_fails_when_ci_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tokenizers" or name.startswith(("tokenizers.", "transformers")):
            raise ImportError("blocked local hf runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("INVARLOCK_REQUIRE_LOCAL_HF", "1")
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(pytest.fail.Exception, match="blocked local hf runtime"):
        _require_local_hf_runtime()


def test_local_hf_runtime_requirement_can_skip_on_developer_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tokenizers" or name.startswith(("tokenizers.", "transformers")):
            raise ImportError("blocked local hf runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("INVARLOCK_REQUIRE_LOCAL_HF", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(pytest.skip.Exception, match="blocked local hf runtime"):
        _require_local_hf_runtime()


def _materialize_tiny_gpt2(model_dir: Path) -> None:
    GPT2Config, GPT2LMHeadModel, tokenizer_types = _require_local_hf_runtime()
    Tokenizer, WordLevel, Whitespace, PreTrainedTokenizerFast = tokenizer_types

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "epsilon": 8,
        "zeta": 9,
        "eta": 10,
        "theta": 11,
    }
    raw_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=vocab["<bos>"],
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def _write_local_inputs(tmp_path: Path, model_dir: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset.jsonl"
    samples = [
        {"text": "alpha beta gamma delta"},
        {"text": "epsilon zeta eta theta"},
        {"text": "alpha gamma epsilon eta"},
        {"text": "beta delta zeta theta"},
        {"text": "theta eta zeta epsilon"},
        {"text": "delta gamma beta alpha"},
        {"text": "alpha theta beta eta"},
        {"text": "gamma zeta delta epsilon"},
    ]
    dataset.write_text(
        "\n".join(json.dumps(sample) for sample in samples) + "\n",
        encoding="utf-8",
    )

    preset = tmp_path / "preset.yaml"
    preset.write_text(
        f"""
dataset:
  provider:
    kind: local_jsonl
    file: "{dataset.as_posix()}"
    text_field: text
    max_samples: 8
  split: validation
  seq_len: 8
  stride: 8
  preview_n: 3
  final_n: 3
  seed: 7
eval:
  metric:
    kind: ppl_causal
  loss:
    type: causal
guards:
  order: ["invariants", "spectral", "rmt"]
output:
  dir: "{(tmp_path / "unused-runs").as_posix()}"
  save_model: false
  save_report: true
model:
  id: "{model_dir.as_posix()}"
  adapter: hf_causal
edit:
  name: noop
  plan: {{}}
""",
        encoding="utf-8",
    )

    edit_config = tmp_path / "quant_rtn.yaml"
    edit_config.write_text(
        f"""
model:
  id: "{model_dir.as_posix()}"
  adapter: hf_causal
edit:
  name: quant_rtn
  plan:
    bitwidth: 8
    per_channel: true
    clamp_ratio: 0.0
    scope: attn
    max_modules: 1
    seed: 7
""",
        encoding="utf-8",
    )
    return preset, edit_config


def _offline_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "INVARLOCK_ALLOW_NETWORK": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "INVARLOCK_EVALUATE_TMP_DIR": str(tmp_path / "eval-tmp"),
        }
    )
    return env


def _run_cli(
    args: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "invarlock", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def test_local_hf_pipeline_evaluate_verify_and_report_html(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2-local"
    _materialize_tiny_gpt2(model_dir)
    preset, edit_config = _write_local_inputs(tmp_path, model_dir)

    report_dir = tmp_path / "reports"
    env = _offline_env(tmp_path)
    _run_cli(
        [
            "evaluate",
            "--baseline",
            str(model_dir),
            "--subject",
            str(model_dir),
            "--baseline-adapter",
            "hf_causal",
            "--subject-adapter",
            "hf_causal",
            "--profile",
            "dev",
            "--tier",
            "balanced",
            "--device",
            "cpu",
            "--preset",
            str(preset),
            "--edit-config",
            str(edit_config),
            "--execution-mode",
            "host",
            "--assurance",
            "off",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(report_dir),
            "--defer-report-rendering",
            "--quiet",
            "--no-color",
        ],
        env=env,
    )

    report_path = report_dir / "evaluation.report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["meta"]["adapter"] == "hf_causal"
    assert report["edit"]["name"] == "quant_rtn"
    assert report["primary_metric"]["kind"] == "ppl_causal"
    assert {guard["name"] for guard in report["guards"]} >= {
        "invariants",
        "spectral",
        "rmt",
    }

    verify = _run_cli(
        [
            "verify",
            str(report_path),
            "--json",
            "--runtime-provenance",
            "host",
            "--assurance",
            "off",
        ],
        env=env,
    )
    verify_payload = json.loads(verify.stdout)
    assert verify_payload["summary"]["ok"] is True

    html_path = tmp_path / "evaluation.html"
    _run_cli(
        [
            "report",
            "html",
            "--input",
            str(report_path),
            "--output",
            str(html_path),
            "--force",
        ],
        env=env,
    )
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "InvarLock" in html
