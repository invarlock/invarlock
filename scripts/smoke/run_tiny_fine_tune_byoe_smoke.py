#!/usr/bin/env python3
"""Run a local tiny fine-tune BYOE smoke through evaluate and verify.

This is a maintainer smoke, not an install-time API. It performs one CPU
fine-tuning step on a cached tiny GPT-2 model, evaluates the saved baseline
against the saved subject, enriches the report with descriptive BYOE metadata,
and verifies the enriched report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

MODEL_ID = "sshleifer/tiny-gpt2"
TRAINING_TEXTS = [
    "InvarLock local fine tune smoke alpha.",
    "InvarLock local fine tune smoke beta.",
    "Evidence metadata remains descriptive.",
    "Tiny CPU training creates a real subject checkpoint.",
]
PROMPT_TEMPLATE = "teacher-forced causal LM over local_jsonl text rows\n"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _sha256_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = file_path.relative_to(path).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} did not contain a JSON object")
    return payload


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"text": text}, sort_keys=True) for text in texts]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_preset(path: Path, data_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            dataset:
              provider:
                kind: local_jsonl
              file: {json.dumps(str(data_file))}
              split: validation
              seq_len: 16
              stride: 16
              preview_n: 2
              final_n: 2
              seed: 42
            guards:
              order: []
            eval:
              metric: {{kind: ppl_causal}}
              loss: {{type: auto}}
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _disable_torchvision_for_text_only_transformers() -> None:
    """Keep text-only GPT-2 loading away from optional torchvision imports."""
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False


def _write_text_only_sitecustomize(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            try:
                import transformers.utils as transformers_utils
                import transformers.utils.import_utils as import_utils
            except Exception:
                pass
            else:
                import_utils.is_torchvision_available = lambda: False
                transformers_utils.is_torchvision_available = lambda: False
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _prepare_env(
    repo_root: Path, *, allow_network: bool, sitecustomize_dir: Path
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(sitecustomize_dir), str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["INVARLOCK_DEDUP_TEXTS"] = "1"
    env["INVARLOCK_CAPACITY_FAST"] = "1"
    env["TRANSFORMERS_NO_TORCHVISION"] = "1"
    if allow_network:
        env["INVARLOCK_ALLOW_NETWORK"] = "1"
        env.pop("TRANSFORMERS_OFFLINE", None)
        env.pop("HF_HUB_OFFLINE", None)
        env["HF_DATASETS_OFFLINE"] = "0"
    else:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def _run_command(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\nLog: {log_path}\n{tail}"
        )


def _materialize_fine_tune(
    *,
    work_root: Path,
    allow_network: bool,
    seed: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    _disable_torchvision_for_text_only_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    local_files_only = not allow_network
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, local_files_only=local_files_only
        )
    except Exception as exc:  # pragma: no cover - exercised by smoke operators
        if allow_network:
            raise
        raise RuntimeError(
            f"{MODEL_ID!r} could not be loaded offline from the local Hugging "
            "Face cache. Re-run with --allow-network only when downloads are "
            f"approved. Original error: {exc}"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_dir = work_root / "models" / "baseline"
    subject_dir = work_root / "models" / "subject"
    tokenizer.save_pretrained(baseline_dir)
    model.save_pretrained(baseline_dir)

    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_floating_point(tensor)
    }
    encoded = tokenizer(
        TRAINING_TEXTS,
        max_length=32,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**encoded, labels=labels).loss
    loss.backward()
    optimizer.step()
    model.eval()

    tokenizer.save_pretrained(subject_dir)
    model.save_pretrained(subject_dir)

    changed_tensors = 0
    max_abs_delta = 0.0
    changed_names: list[str] = []
    for name, tensor in model.state_dict().items():
        if name not in before or not torch.is_floating_point(tensor):
            continue
        diff = tensor.detach().cpu() - before[name]
        tensor_max = float(diff.abs().max().item())
        if tensor_max > 0.0:
            changed_tensors += 1
            max_abs_delta = max(max_abs_delta, tensor_max)
            changed_names.append(name)

    if changed_tensors == 0:
        raise RuntimeError("fine-tune smoke did not change any floating tensors")

    return {
        "baseline_dir": baseline_dir,
        "subject_dir": subject_dir,
        "loss": float(loss.detach().cpu().item()),
        "changed_tensors": changed_tensors,
        "changed_tensor_examples": changed_names[:12],
        "max_abs_delta": max_abs_delta,
        "baseline_digest": _sha256_tree(baseline_dir),
        "subject_digest": _sha256_tree(subject_dir),
    }


def _locate_report(report_root: Path) -> Path:
    reports = sorted(
        report_root.rglob("evaluation.report.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise RuntimeError(f"no evaluation.report.json found under {report_root}")
    return reports[-1]


def _enrich_report(
    *,
    report_path: Path,
    summary_path: Path,
    data_file: Path,
    materialized: dict[str, Any],
) -> dict[str, Any]:
    report = _read_json(report_path)
    edit = report.setdefault("edit", {})
    if not isinstance(edit, dict):
        raise RuntimeError("generated report has a non-object edit block")
    edit["edit_provenance"] = {
        "edit_family": "fine_tune",
        "edit_method": "cpu_tiny_single_step",
        "edit_count": 1,
        "target_set_digest": _sha256_file(data_file),
        "editor_artifact_digest": _sha256_file(summary_path),
        "dynamic_runtime_required": False,
    }
    edit["edit_impact"] = {
        "scenario_types": ["general_ability_sentinel", "sequential_edit_stress"]
    }
    edit["edit_topology"] = {
        "artifact_kind": "checkpoint",
        "module_hashes": {
            "baseline_checkpoint": str(materialized["baseline_digest"]),
            "subject_checkpoint": str(materialized["subject_digest"]),
        },
        "runtime_activation_policy": "load saved subject checkpoint",
        "training_or_edit_data_ref": _sha256_file(data_file),
    }
    edit["delta_privacy"] = {
        "delta_available": "hash_only",
        "privacy_sensitivity": "public",
        "public_raw_delta_approved": False,
    }
    report["evaluation_realism"] = {
        "mode": "teacher_forced",
        "prompt_template_hash": _sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
        "decoding_config": {},
        "max_tokens": 16,
        "truncation_policy": "truncate local_jsonl rows to seq_len=16",
        "dataset_or_task_id": "local-jsonl-tiny-fine-tune-smoke",
        "metric_is_generation_realistic": False,
        "proxy_metric_warning": (
            "Teacher-forced perplexity is a regression proxy, not live "
            "generation behavior."
        ),
    }
    _write_json(report_path, report)
    return report


def _assert_outline_sections(repo_root: Path, report: dict[str, Any]) -> list[str]:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from invarlock.reporting.report_outline import build_evaluation_report_outline

    outline = build_evaluation_report_outline(report)
    section_keys = list(outline.section_keys)
    required = {"evaluation_realism", "edit_provenance"}
    missing = required.difference(section_keys)
    if missing:
        raise RuntimeError(f"report outline missing section(s): {sorted(missing)}")
    return section_keys


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    work_root = Path(args.work_root).resolve() if args.work_root else None
    if work_root is None:
        work_root = Path(
            tempfile.mkdtemp(
                prefix="invarlock-tiny-fine-tune-byoe-", dir="/private/tmp"
            )
        )
    work_root.mkdir(parents=True, exist_ok=True)

    data_file = work_root / "data" / "tiny_fine_tune.jsonl"
    preset_path = work_root / "preset" / "local_jsonl.yaml"
    _write_jsonl(data_file, TRAINING_TEXTS)
    _write_preset(preset_path, data_file)

    materialized = _materialize_fine_tune(
        work_root=work_root,
        allow_network=args.allow_network,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    edit_summary_path = work_root / "external_edit_summary.json"
    _write_json(
        edit_summary_path,
        {
            "schema": "invarlock/local-tiny-fine-tune-smoke-v1",
            "model_id": MODEL_ID,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "loss": materialized["loss"],
            "changed_tensors": materialized["changed_tensors"],
            "changed_tensor_examples": materialized["changed_tensor_examples"],
            "max_abs_delta": materialized["max_abs_delta"],
            "baseline_digest": materialized["baseline_digest"],
            "subject_digest": materialized["subject_digest"],
        },
    )

    sitecustomize_dir = work_root / "python_compat"
    _write_text_only_sitecustomize(sitecustomize_dir)
    env = _prepare_env(
        repo_root,
        allow_network=args.allow_network,
        sitecustomize_dir=sitecustomize_dir,
    )
    report_root = work_root / "report"
    evaluate_command = [
        sys.executable,
        "-m",
        "invarlock",
        "evaluate",
        "--baseline",
        str(materialized["baseline_dir"]),
        "--subject",
        str(materialized["subject_dir"]),
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
        str(preset_path),
        "--out",
        str(work_root / "runs"),
        "--report-out",
        str(report_root),
        "--edit-label",
        "custom",
        "--execution-mode",
        "host",
        "--assurance",
        "off",
        "--no-banner",
        "--quiet",
        "--no-progress",
    ]
    if args.allow_network:
        evaluate_command.append("--allow-network")
    _run_command(
        command=evaluate_command,
        cwd=repo_root,
        env=env,
        log_path=work_root / "logs" / "evaluate.log",
    )

    report_path = _locate_report(report_root)
    enriched_report = _enrich_report(
        report_path=report_path,
        summary_path=edit_summary_path,
        data_file=data_file,
        materialized=materialized,
    )
    section_keys = _assert_outline_sections(repo_root, enriched_report)

    verify_json_path = work_root / "verify.json"
    verify_command = [
        sys.executable,
        "-m",
        "invarlock",
        "verify",
        "--json",
        "--profile",
        "dev",
        "--runtime-provenance",
        "host",
        "--assurance",
        "off",
        str(report_path),
    ]
    _run_command(
        command=verify_command,
        cwd=repo_root,
        env=env,
        log_path=verify_json_path,
    )
    verify_payload = _read_json(verify_json_path)
    verify_summary = verify_payload.get("summary")
    verify_summary = verify_summary if isinstance(verify_summary, dict) else {}

    result = {
        "ok": True,
        "work_root": str(work_root),
        "report_path": str(report_path),
        "verify_path": str(verify_json_path),
        "verify_ok": verify_summary.get("ok"),
        "verify_reason": verify_summary.get("reason"),
        "outline_sections": section_keys,
        "model_id": MODEL_ID,
        "changed_tensors": materialized["changed_tensors"],
        "max_abs_delta": materialized["max_abs_delta"],
        "offline": not args.allow_network,
    }
    _write_json(work_root / "smoke_summary.json", result)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local CPU tiny fine-tune BYOE smoke and verify the enriched "
            "evaluation report."
        )
    )
    parser.add_argument(
        "--work-root",
        help="Output directory. Defaults to a new /private/tmp smoke directory.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model downloads if the local Hugging Face cache is missing.",
    )
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_smoke(args)
    except Exception as exc:
        print(f"[tiny-fine-tune-byoe-smoke] FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
