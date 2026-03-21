# Third-Party Notices

InvarLock relies on several open-source projects, datasets, and reference model weights. This document
summarizes the direct Python dependencies declared in [`pyproject.toml`](./pyproject.toml), the common
optional runtime extras used by supported adapter flows, and the reference assets used in examples as of
2026-03-21.

This file is informational and is not an exhaustive transitive dependency manifest. Always consult the
official upstream license text for the final terms that apply to your build, container image, or
redistributed artifact.

## Direct Python Dependencies

| Component | Upstream | License | Notes |
|-----------|----------|---------|-------|
| Typer | [fastapi/typer](https://github.com/fastapi/typer) | MIT | CLI framework |
| Click | [pallets/click](https://github.com/pallets/click) | BSD 3-Clause | CLI command parsing and option handling |
| Shellingham | [sarugaku/shellingham](https://github.com/sarugaku/shellingham) | ISC | Shell detection used by CLI helpers |
| pandas | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | BSD 3-Clause | Tabular result processing and export helpers |
| scikit-learn | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause | Metrics and auxiliary ML utilities |
| Pydantic | [pydantic/pydantic](https://github.com/pydantic/pydantic) | MIT | Structured config and validation helpers |
| Rich | [Textualize/rich](https://github.com/Textualize/rich) | MIT | Terminal rendering and formatting |
| PyYAML | [yaml/pyyaml](https://github.com/yaml/pyyaml) | MIT | YAML configuration parsing |
| Markdown | [Python-Markdown/markdown](https://github.com/Python-Markdown/markdown) | BSD 3-Clause | Markdown rendering utilities |
| psutil | [giampaolo/psutil](https://github.com/giampaolo/psutil) | BSD-style | Process and system telemetry |
| Hypothesis | [HypothesisWorks/hypothesis](https://github.com/HypothesisWorks/hypothesis) | MPL 2.0 | Property-based testing support shipped in the runtime dependency set |
| typing_extensions | [python/typing_extensions](https://github.com/python/typing_extensions) | PSF-2.0 | Forward-compatible typing helpers |
| jsonschema | [python-jsonschema/jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | Report and manifest schema validation |

## Optional Runtime Extras

| Component | Upstream | License | Notes |
|-----------|----------|---------|-------|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | Core tensor runtime for adapters and edits |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache License 2.0 | Model loading, tokenizers, and generation utilities |
| safetensors | [huggingface/safetensors](https://github.com/huggingface/safetensors) | Apache License 2.0 | Tensor-only serialization used by secure snapshot flows |
| Datasets | [huggingface/datasets](https://github.com/huggingface/datasets) | Apache License 2.0 | Dataset ingestion and evaluation helpers |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | BSD 3-Clause | Numerical kernels and array helpers |
| Hugging Face Hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache License 2.0 | Model and dataset registry access |
| Accelerate | [huggingface/accelerate](https://github.com/huggingface/accelerate) | Apache License 2.0 | Device placement and distributed helpers |
| aiohttp | [aio-libs/aiohttp](https://github.com/aio-libs/aiohttp) | Apache-2.0 AND MIT | Optional async HTTP transport stack |
| h2 | [python-hyper/h2](https://github.com/python-hyper/h2) | MIT | Optional HTTP/2 support for Hub traffic |
| Pillow | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | Optional image handling for multimodal/runtime helpers |
| bitsandbytes | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | MIT | Optional GPU quantization/runtime kernels |
| AutoGPTQ | [PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | Apache License 2.0 | Optional GPTQ quantization backend (Linux-only extra) |
| AutoAWQ | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | Apache License 2.0 | Optional AWQ quantization backend (Linux-only extra) |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton) | MIT | Optional GPU kernel compilation/runtime support |

## Rust Components

The `invarlock-runtime-verify` workspace currently uses only the Rust standard library. As of the current
[`Cargo.lock`](./Cargo.lock), there are no third-party Cargo crates to list here.

## Reference Models

| Model | Publisher | Source | License |
|-------|-----------|--------|---------|
| `gpt2` | OpenAI | [Hugging Face](https://huggingface.co/gpt2) | [Modified MIT](https://huggingface.co/gpt2/resolve/main/README.md) |
| `bert-base-uncased` | Google | [Hugging Face](https://huggingface.co/bert-base-uncased) | [Apache License 2.0](https://huggingface.co/bert-base-uncased/resolve/main/README.md) |

Users are responsible for ensuring they comply with the upstream model licenses when redistributing
weights or deploying downstream products.

## Reference Datasets

| Dataset | Publisher | Source | License |
|---------|-----------|--------|---------|
| WikiText-2 | Salesforce Research | [Hugging Face](https://huggingface.co/datasets/wikitext) | [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

## Additional Notes

- This file should be refreshed whenever direct dependencies, optional runtime extras, reference assets,
  or shipped runtime components change.
- Release bundles, SBOMs, and upstream license files remain authoritative if this summary becomes stale.
