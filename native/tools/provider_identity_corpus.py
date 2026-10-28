"""Retain provider identity contracts from the Python implementation (no SDK calls)."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

from invarlock.core.runtime_provider import ModelRuntimeSpec
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider
from invarlock.runtime_providers.hf_vision_text import HFVisionTextProvider
from invarlock.runtime_providers.llama_cpp import LlamaCppProvider
from invarlock.runtime_providers.tensorrt_llm import TensorRTLLMProvider

ROOT = Path(__file__).resolve().parents[2]
profiles = json.loads(
    (ROOT / "native/cli/src/assets/provider-profiles.json").read_text()
)
rows = []
for provider in [
    HFTransformersProvider(),
    HFVisionTextProvider(),
    LlamaCppProvider(),
    TensorRTLLMProvider(),
]:
    name = provider.name
    profile = profiles[name]
    settings = dict.fromkeys(profile["positive"], 1)
    settings.update(dict.fromkeys(profile["digests"], "a" * 64))
    settings["seed"] = 0
    model = "fixture-model"
    if name.startswith("hf_"):
        settings.update(
            checkpoint_tree_sha256="a" * 64,
            tokenizer_metadata_sha256="b" * 64,
            offline=True,
        )
        if name == "hf_vision_text":
            settings["processor_metadata_sha256"] = "c" * 64
    elif name == "llama_cpp":
        model = "gguf-sha256-" + "a" * 64 + ".gguf"
        settings["backend_version"] = (
            "version: 4242 (abcdef) built with clang 18 for Linux x86_64"
        )
    else:
        model = "tensorrt-llm-sha256-" + "a" * 64
        settings.update(backend_version="1.0", target_compute_capability="9.0")
    variants = [("valid", settings)]
    for key in settings:
        for value in [
            None,
            True,
            False,
            0,
            -1,
            1.0,
            "",
            " bad ",
            "sha256:" + "a" * 64,
            "A" * 64,
        ]:
            changed = copy.deepcopy(settings)
            changed[key] = value
            variants.append((f"{key}={value!r}", changed))
        changed = copy.deepcopy(settings)
        del changed[key]
        variants.append((f"missing:{key}", changed))
    for label, selected in variants:
        side = {
            "artifact": {"model_id": model, "locator": "fixture"},
            "runtime": {"provider": name, "settings": selected},
        }
        try:
            result = asdict(
                provider.identify_artifact(
                    ModelRuntimeSpec(
                        provider_name=name, model_id=model, settings=selected
                    )
                )
            )
        except (ValueError, TypeError) as error:
            rows.append(
                {"label": name + ":" + label, "side": side, "error": str(error)}
            )
        else:
            rows.append({"label": name + ":" + label, "side": side, "result": result})
path = ROOT / "native/cli/src/assets/provider-identity-corpus.json"
path.write_text(json.dumps(rows, indent=2) + "\n")
print(len(rows))
