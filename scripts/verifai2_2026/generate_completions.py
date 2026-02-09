#!/usr/bin/env python3
"""
generate_completions.py
=======================

Deterministic-ish completion generation for code-verifier benchmarks (F4/S1).

Inputs:
- tasks JSONL: each line at least {"id": "...", "prompt": "..."} (tests ignored)

Outputs:
- completions JSONL: {"id": "...", "attempt_id": 0..k-1, "completion": "..."}

Notes:
- Transformers 5.x does not accept `generate(..., generator=...)`; we seed
  globally and record determinism inputs in the verifier-trace contract.
- This script does not download datasets; the caller provides tasks JSONL.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}")
            out.append(obj)
    return out


def _import_torch():  # noqa: ANN001
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "torch is required for --backend hf_causal (install the HF extra)."
        ) from exc


def _import_transformers() -> tuple[Any, Any]:
    try:
        tr = importlib.import_module("transformers")
        return tr.AutoModelForCausalLM, tr.AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for --backend hf_causal (install the HF extra)."
        ) from exc


def _parse_device(device: str) -> tuple[str, int | None]:
    d = device.strip().lower()
    if d == "cpu":
        return "cpu", None
    if d.startswith("cuda"):
        if d == "cuda":
            return "cuda", 0
        if d.startswith("cuda:"):
            try:
                idx = int(d.split(":", 1)[1])
            except Exception as exc:
                raise ValueError(f"Invalid --device {device!r}") from exc
            return "cuda", idx
    raise ValueError(f"Invalid --device {device!r} (expected cpu|cuda|cuda:N)")


def _validate_decoding(
    *,
    method: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_samples: int,
    num_beams: int,
) -> list[str]:
    errors: list[str] = []
    if method == "greedy":
        if temperature != 0.0:
            errors.append("decoding.method=greedy requires temperature=0.0")
        if top_p != 1.0:
            errors.append("decoding.method=greedy requires top_p=1.0")
        if top_k != 0:
            errors.append("decoding.method=greedy requires top_k=0")
        if num_samples not in {0, 1}:
            errors.append("decoding.method=greedy requires num_samples unset/1")
        if num_beams not in {0, 1}:
            errors.append("decoding.method=greedy requires num_beams unset/1")
    if method == "sample":
        if temperature <= 0.0:
            errors.append("decoding.method=sample requires temperature>0.0")
        if num_samples < 1:
            errors.append("decoding.method=sample requires num_samples>=1")
    if method == "beam":
        if num_beams < 2:
            errors.append("decoding.method=beam requires num_beams>=2")
        if num_samples not in {0, 1}:
            errors.append("decoding.method=beam currently supports num_samples unset/1")
    return errors


def _torch_dtype(torch, name: str):  # noqa: ANN001
    n = name.strip().lower()
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Invalid --dtype {name!r} (expected fp16|bf16|fp32)")


def _seed_everything(torch, seed: int) -> None:  # noqa: ANN001
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_hf(
    *,
    torch,  # noqa: ANN001
    model: str,
    revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tok = AutoTokenizer.from_pretrained(
        tokenizer,
        revision=tokenizer_revision,
        use_fast=True,
        trust_remote_code=bool(trust_remote_code),
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device_kind, device_idx = _parse_device(device)
    torch_dtype = _torch_dtype(torch, dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": bool(trust_remote_code),
    }
    if revision:
        model_kwargs["revision"] = revision

    if device_kind == "cpu":
        m = AutoModelForCausalLM.from_pretrained(
            model, device_map={"": "cpu"}, **model_kwargs
        )
    else:
        # Prefer using CUDA_VISIBLE_DEVICES for isolation. If not, allow explicit
        # index selection.
        idx = 0 if device_idx is None else int(device_idx)
        m = AutoModelForCausalLM.from_pretrained(
            model, device_map={"": idx}, **model_kwargs
        )
    m.eval()
    return m, tok


def _gen_kwargs(
    *,
    tok: Any,
    method: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    num_samples: int,
    num_beams: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": getattr(tok, "eos_token_id", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
    }
    if method == "greedy":
        kwargs.update(
            {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "num_return_sequences": 1,
            }
        )
        return kwargs
    if method == "sample":
        kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "num_return_sequences": int(num_samples),
            }
        )
        return kwargs
    if method == "beam":
        kwargs.update(
            {
                "do_sample": False,
                "num_beams": int(num_beams),
                "num_return_sequences": 1,
            }
        )
        return kwargs
    raise ValueError(f"Unknown decoding method {method!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf_causal"], default="hf_causal")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--id-field", type=str, default="id")
    p.add_argument("--prompt-field", type=str, default="prompt")
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--model", type=str, required=True, help="HF id or local dir.")
    p.add_argument("--revision", type=str, default="", help="HF revision/commit.")
    p.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HF tokenizer id (default: --model).",
    )
    p.add_argument(
        "--tokenizer-revision",
        type=str,
        default="",
        help="HF tokenizer revision (default: --revision).",
    )
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument(
        "--decoding-method",
        choices=["greedy", "sample", "beam"],
        required=True,
    )
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--num-beams", type=int, default=0)
    args = p.parse_args(argv)

    errors = _validate_decoding(
        method=str(args.decoding_method),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_samples=int(args.num_samples),
        num_beams=int(args.num_beams),
    )
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    tasks = _read_jsonl(args.tasks)
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        return 2

    id_field = str(args.id_field)
    prompt_field = str(args.prompt_field)
    lim = int(args.limit)
    tasks = tasks[:lim] if lim > 0 else tasks

    ids: list[str] = []
    prompts: list[str] = []
    for t in tasks:
        tid = t.get(id_field)
        prompt = t.get(prompt_field)
        if not isinstance(tid, str) or not tid:
            raise ValueError(f"Task missing id field {id_field!r}: {t!r}")
        if not isinstance(prompt, str):
            raise ValueError(f"Task id={tid} missing prompt field {prompt_field!r}")
        ids.append(tid)
        prompts.append(prompt)

    torch = _import_torch()
    _seed_everything(torch, int(args.seed))

    tok_id = str(args.tokenizer).strip() or str(args.model)
    tok_rev = str(args.tokenizer_revision).strip() or str(args.revision)
    m, tok = _load_hf(
        torch=torch,
        model=str(args.model),
        revision=str(args.revision),
        tokenizer=tok_id,
        tokenizer_revision=tok_rev,
        device=str(args.device),
        dtype=str(args.dtype),
        trust_remote_code=bool(args.trust_remote_code),
    )

    method = str(args.decoding_method)
    n = int(args.num_samples) if int(args.num_samples) > 0 else 1
    gen_kwargs = _gen_kwargs(
        tok=tok,
        method=method,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_new_tokens=int(args.max_new_tokens),
        num_samples=n,
        num_beams=int(args.num_beams) if int(args.num_beams) > 0 else 0,
    )

    bs = max(1, int(args.batch_size))
    n_ret = int(gen_kwargs.get("num_return_sequences", 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as out:
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i : i + bs]
            batch_ids = ids[i : i + bs]

            enc = tok(batch_prompts, return_tensors="pt", padding=True)
            enc = {k: v.to(m.device) for k, v in enc.items()}
            in_len = int(enc["input_ids"].shape[1])

            with torch.no_grad():
                seq = m.generate(**enc, **gen_kwargs)

            gen_tokens = seq[:, in_len:]
            texts = tok.batch_decode(gen_tokens, skip_special_tokens=True)

            if len(texts) != len(batch_ids) * n_ret:
                raise RuntimeError("Unexpected generate() output size")

            for j, tid in enumerate(batch_ids):
                for k in range(n_ret):
                    idx = j * n_ret + k
                    rec = {"id": tid, "attempt_id": k, "completion": texts[idx]}
                    out.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written += 1

    print(f"wrote={written} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
