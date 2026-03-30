from __future__ import annotations

import time
from typing import Any


def _raise_latency_error(summary: str, *, error: Exception | None = None) -> None:
    if error is None:
        raise RuntimeError(summary)
    raise RuntimeError(summary) from error


def measure_latency(model: Any, sample_data: Any, device: Any) -> float:
    """Simple latency measurement for a sample."""
    import torch

    if not sample_data:
        return 0.0

    sample = sample_data[0] if sample_data else None
    if sample is None:
        return 0.0

    if isinstance(sample, dict):
        input_ids = sample.get("input_ids", sample.get("inputs"))
    else:
        input_ids = sample

    if input_ids is None:
        return 0.0

    if not isinstance(input_ids, torch.Tensor):
        try:
            input_ids = torch.tensor(input_ids)
        except Exception as error:
            _raise_latency_error(
                "Latency measurement input tensor conversion failed.",
                error=error,
            )

    try:
        dim_val = input_ids.dim()
    except Exception as error:
        _raise_latency_error(
            "Latency measurement input shape inspection failed.",
            error=error,
        )
    if dim_val == 1:
        try:
            input_ids = input_ids.unsqueeze(0)
        except Exception as error:
            _raise_latency_error(
                "Latency measurement batch shaping failed.",
                error=error,
            )

    try:
        input_ids = input_ids.to(device)
    except Exception as error:
        _raise_latency_error(
            "Latency measurement device transfer failed.",
            error=error,
        )

    def maybe_sync() -> None:
        try:
            is_cuda = False
            if hasattr(device, "type"):
                is_cuda = device.type == "cuda"
            elif isinstance(device, str):
                is_cuda = device.startswith("cuda")
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as error:
            _raise_latency_error(
                "Latency measurement device synchronization failed.",
                error=error,
            )

    with torch.no_grad():
        try:
            labels_t = input_ids
            attn_t = None
            token_type_t = None
            if isinstance(sample, dict) and "attention_mask" in sample:
                try:
                    attn_t = torch.tensor(sample["attention_mask"])
                    try:
                        if attn_t.dim() == 1:
                            attn_t = attn_t.unsqueeze(0)
                    except Exception:
                        _raise_latency_error(
                            "Latency measurement attention-mask shaping failed."
                        )
                    try:
                        attn_t = attn_t.to(device)
                    except Exception as error:
                        _raise_latency_error(
                            "Latency measurement attention-mask device transfer failed.",
                            error=error,
                        )
                except Exception as error:
                    _raise_latency_error(
                        "Latency measurement attention-mask preparation failed.",
                        error=error,
                    )
            if isinstance(sample, dict) and "token_type_ids" in sample:
                try:
                    token_type_t = torch.tensor(sample["token_type_ids"])
                    if token_type_t.dim() == 1:
                        token_type_t = token_type_t.unsqueeze(0)
                    token_type_t = token_type_t.to(device)
                except Exception as error:
                    _raise_latency_error(
                        "Latency measurement token-type preparation failed.",
                        error=error,
                    )

            def call_model() -> Any:
                kwargs: dict[str, Any] = {"labels": labels_t}
                if attn_t is not None:
                    kwargs["attention_mask"] = attn_t
                if token_type_t is not None:
                    kwargs["token_type_ids"] = token_type_t
                return model(input_ids, **kwargs)

            for _ in range(3):
                _ = call_model()

            maybe_sync()
            start_time = time.time()
            for _ in range(10):
                _ = call_model()
            maybe_sync()
            end_time = time.time()

            total_time = (end_time - start_time) * 1000
            try:
                total_tokens = input_ids.numel() * 10
            except Exception as error:
                _raise_latency_error(
                    "Latency measurement token counting failed.",
                    error=error,
                )
            return total_time / total_tokens if total_tokens > 0 else 0.0
        except Exception:
            raise


def samples_to_dataloader(samples: list[Any]) -> Any:
    """Convert list of samples to a minimal DataLoader-compatible iterable."""

    class SampleDataLoader:
        def __init__(self, samples: list[Any]):
            self.samples = samples

        def __iter__(self):
            import torch

            for sample in self.samples:
                input_ids = sample.get("input_ids", sample.get("inputs"))
                attention_mask = sample.get("attention_mask")
                if input_ids is None:
                    continue

                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                if attention_mask is not None:
                    if not isinstance(attention_mask, torch.Tensor):
                        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                    if attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)

                batch = {"input_ids": input_ids}
                if attention_mask is not None:
                    batch["attention_mask"] = attention_mask

                token_type = sample.get("token_type_ids")
                if token_type is not None:
                    if not isinstance(token_type, torch.Tensor):
                        token_type = torch.tensor(token_type, dtype=torch.long)
                    if token_type.dim() == 1:
                        token_type = token_type.unsqueeze(0)
                    batch["token_type_ids"] = token_type

                labels = sample.get("labels")
                if labels is None:
                    labels = input_ids.clone()
                    if attention_mask is not None:
                        labels = labels.masked_fill(attention_mask == 0, -100)
                else:
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels, dtype=torch.long)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                batch["labels"] = labels
                yield batch

        def __len__(self) -> int:
            return len(self.samples)

    return SampleDataLoader(samples)


__all__ = ["measure_latency", "samples_to_dataloader"]
