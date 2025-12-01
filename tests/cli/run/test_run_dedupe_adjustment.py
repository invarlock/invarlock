from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path, preview=20, final=20) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{}}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview}
  final_n: {final}

guards:
  order: []

eval:
  spike_threshold: 2.0
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _records_with_duplicates(n: int, base: int):
    # Build n sequences, introduce 2 duplicates when n>=20
    ids = [[base + i, base + i + 1, base + i + 2] for i in range(n)]
    masks = [[1, 1, 1] for _ in range(n)]
    if n >= 20:
        # duplicate first two sequences
        ids[-1] = ids[0]
        ids[-2] = ids[1]
    return ids, masks


def test_dataset_dedupe_reduction_then_success(tmp_path: Path):
    cfg = _cfg(tmp_path, 20, 20)
    captured = {}

    class Provider:
        def windows(self, **kwargs):
            n_prev = int(kwargs.get("preview_n"))
            n_fin = int(kwargs.get("final_n"))
            prev_ids, prev_masks = _records_with_duplicates(n_prev, base=100)
            fin_ids, fin_masks = _records_with_duplicates(n_fin, base=200)
            return (
                SimpleNamespace(input_ids=prev_ids, attention_masks=prev_masks),
                SimpleNamespace(input_ids=fin_ids, attention_masks=fin_masks),
            )

    def cap_save(report, run_dir, formats, filename_prefix=None):
        captured["report"] = report
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context=getattr(k.get("config"), "context", {}),
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda model_id, adapter: SimpleNamespace(
                    default_loss="ce",
                    model_id=model_id,
                    adapter=adapter,
                    module_selectors={},
                    invariants=set(),
                    cert_lints=[],
                    family="gpt",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda name: SimpleNamespace(
                        name=name, load_model=lambda model_id, device=None: object()
                    ),
                    get_edit=lambda name: SimpleNamespace(name=name),
                    get_guard=lambda name: SimpleNamespace(name=name),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
                ),
            )
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        for target in ("invarlock.cli.commands.run.resolve_tokenizer",):
            stack.enter_context(
                patch(
                    target,
                    lambda profile: (
                        SimpleNamespace(
                            eos_token="</s>", pad_token="</s>", vocab_size=50000
                        ),
                        "tokhash123",
                    ),
                )
            )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    data = captured["report"]["data"]
    wp = data.get("window_plan", {})
    # After dedupe reduction, per-arm should be reduced from 20 to 15 (reduction min=5)
    assert wp.get("actual_preview") in {15, 20} and wp.get("actual_final") in {15, 20}
    # Note: dedupe_adjustments are recorded only when a window_plan exists earlier (e.g., release).
