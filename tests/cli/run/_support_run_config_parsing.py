from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.cli.run._support_run_common import canonical_ppl_metrics


def config_parsing_detect_profile(model_id: str, adapter: str) -> SimpleNamespace:
    return SimpleNamespace(
        default_loss="ce",
        default_provider=None,
        default_metric=None,
        model_id=model_id,
        adapter=adapter,
        family="gpt",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )


def config_parsing_tokenizer():
    return (
        SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50_000),
        "tokhash123",
    )


def config_parsing_core_report(
    *, evaluation_windows: dict[str, object] | None
) -> SimpleNamespace:
    return SimpleNamespace(
        edit={"plan_digest": "abcd", "deltas": {"heads_pruned": 0}},
        metrics=canonical_ppl_metrics(
            preview=10.0,
            final=10.0,
            window_overlap_fraction=0.0,
            window_match_fraction=1.0,
            paired_windows=1,
            loss_type="ce",
        ),
        guards={},
        context={"dataset_meta": {}},
        evaluation_windows=evaluation_windows,
        status="success",
    )


class _ConfigParsingEval:
    def __init__(self, *, spike_threshold: float, loss_type: str, capacity_fast: bool):
        self.spike_threshold = float(spike_threshold)
        self.loss = SimpleNamespace(type=loss_type)
        self.capacity_fast = bool(capacity_fast)

    def model_dump(self) -> dict[str, object]:
        return {
            "spike_threshold": float(self.spike_threshold),
            "loss": {"type": str(getattr(self.loss, "type", "auto"))},
            "capacity_fast": bool(self.capacity_fast),
        }


class ConfigParsingCfg:
    def __init__(
        self,
        *,
        outdir: Path,
        dataset_provider: object,
        loss_type: str = "ce",
        edit_plan: object | None = None,
        output: dict[str, object] | None = None,
    ) -> None:
        self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
        self.edit = SimpleNamespace(name="quant_rtn", plan=(edit_plan or {}))
        self.auto = SimpleNamespace(
            enabled=False, tier="balanced", probes=0, target_pm_ratio=None
        )
        self.guards = SimpleNamespace(order=[])
        self.dataset = SimpleNamespace(
            provider=dataset_provider,
            id="synthetic",
            split="validation",
            seq_len=8,
            stride=4,
            preview_n=2,
            final_n=2,
            seed=42,
        )
        self.eval = _ConfigParsingEval(
            spike_threshold=2.0, loss_type=loss_type, capacity_fast=True
        )
        out = {"dir": outdir}
        if output:
            out.update(output)
        self.output = SimpleNamespace(**out)

    def model_dump(self) -> dict[str, object]:
        out = {
            "dir": str(getattr(self.output, "dir", "")),
            "save_model": getattr(self.output, "save_model", False),
            "model_dir": getattr(self.output, "model_dir", None),
            "model_subdir": getattr(self.output, "model_subdir", None),
        }
        return {
            "model": {
                "id": self.model.id,
                "adapter": self.model.adapter,
                "device": self.model.device,
            },
            "edit": {
                "name": self.edit.name,
                "plan": getattr(self.edit, "plan", {}),
            },
            "auto": {
                "enabled": self.auto.enabled,
                "tier": self.auto.tier,
                "probes": self.auto.probes,
                "target_pm_ratio": self.auto.target_pm_ratio,
            },
            "guards": {"order": list(self.guards.order)},
            "dataset": {
                "provider": self.dataset.provider,
                "id": self.dataset.id,
                "split": self.dataset.split,
                "seq_len": self.dataset.seq_len,
                "stride": self.dataset.stride,
                "preview_n": self.dataset.preview_n,
                "final_n": self.dataset.final_n,
                "seed": self.dataset.seed,
            },
            "eval": {
                "spike_threshold": self.eval.spike_threshold,
                "loss": {"type": getattr(self.eval.loss, "type", None)},
            },
            "output": out,
        }
