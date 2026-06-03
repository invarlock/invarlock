# Third-Party Notices

InvarLock relies on several open-source projects, datasets, and reference model weights. This document
summarizes the direct Python dependencies declared in [`pyproject.toml`](./pyproject.toml), common
optional extras used by adapter, evaluation, evidence-pack, and observability flows, and a representative
set of third-party assets referenced by public docs, presets, and smoke scripts in this repository
revision.

This file is informational and is not an exhaustive transitive dependency manifest. Always consult the
official upstream license text for the final terms that apply to your build, container image, or
redistributed artifact.

## Direct Python Dependencies

| Component | Upstream | License | Notes |
|-----------|----------|---------|-------|
| typer | [fastapi/typer](https://github.com/fastapi/typer) | MIT | CLI framework |
| click | [pallets/click](https://github.com/pallets/click) | BSD 3-Clause | CLI command parsing and option handling |
| shellingham | [sarugaku/shellingham](https://github.com/sarugaku/shellingham) | ISC | Shell detection used by CLI helpers |
| cryptography | [pyca/cryptography](https://github.com/pyca/cryptography) | Apache-2.0 OR BSD-3-Clause | Signature and key handling for evidence-pack integrity verification |
| pydantic | [pydantic/pydantic](https://github.com/pydantic/pydantic) | MIT | Structured config and validation helpers |
| rich | [Textualize/rich](https://github.com/Textualize/rich) | MIT | Terminal rendering and formatting |
| pyyaml | [yaml/pyyaml](https://github.com/yaml/pyyaml) | MIT | YAML configuration parsing |
| markdown | [Python-Markdown/markdown](https://github.com/Python-Markdown/markdown) | BSD 3-Clause | Markdown rendering utilities |
| psutil | [giampaolo/psutil](https://github.com/giampaolo/psutil) | BSD-style | Process and system telemetry |
| jsonschema | [python-jsonschema/jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | Report and manifest schema validation |
| idna | [kjd/idna](https://github.com/kjd/idna) | BSD 3-Clause | Internationalized domain-name validation used by URL/security checks |

## Optional Runtime Extras

This table focuses on the runtime-oriented optional dependencies used by
adapter, evaluation, evidence-pack, and observability flows. It intentionally
excludes dev- and CI-only extras such as `pytest`, `ruff`, `mkdocs`, and
release tooling.

| Component | Upstream | License | Notes |
|-----------|----------|---------|-------|
| torch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | Core tensor runtime for adapters and edits |
| torchvision | [pytorch/vision](https://github.com/pytorch/vision) | BSD 3-Clause | Optional vision/model-definition dependency used by GPTQModel |
| transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache License 2.0 | Model loading, tokenizers, and generation utilities |
| safetensors | [huggingface/safetensors](https://github.com/huggingface/safetensors) | Apache License 2.0 | Tensor-only serialization used by secure snapshot flows |
| datasets | [huggingface/datasets](https://github.com/huggingface/datasets) | Apache License 2.0 | Dataset ingestion and evaluation helpers |
| requests | [psf/requests](https://github.com/psf/requests) | Apache License 2.0 | Optional HTTP client used by evaluation extras and observability exporters |
| numpy | [numpy/numpy](https://github.com/numpy/numpy) | BSD 3-Clause | Numerical kernels and array helpers |
| scikit-learn | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause | Optional mutual-information/probe utilities |
| huggingface_hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache License 2.0 | Model and dataset registry access |
| accelerate | [huggingface/accelerate](https://github.com/huggingface/accelerate) | Apache License 2.0 | Device placement and distributed helpers |
| protobuf | [protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf) | BSD 3-Clause | Serialization support required by many Hugging Face model and tokenizer stacks |
| sentencepiece | [google/sentencepiece](https://github.com/google/sentencepiece) | Apache License 2.0 | Optional tokenizer runtime for SentencePiece-based model families |
| tiktoken | [openai/tiktoken](https://github.com/openai/tiktoken) | MIT | Optional tokenizer runtime for GPT-style families and compatibility probes |
| aiohttp | [aio-libs/aiohttp](https://github.com/aio-libs/aiohttp) | Apache-2.0 AND MIT | Optional async HTTP transport stack |
| h2 | [python-hyper/h2](https://github.com/python-hyper/h2) | MIT | Optional HTTP/2 support for Hub traffic |
| pillow | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | Optional image handling for multimodal/runtime helpers |
| bitsandbytes | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | MIT | Optional GPU quantization/runtime kernels |
| gptqmodel | [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) | Apache License 2.0 | Optional GPTQ and AWQ backend loading |
| optimum-quanto | [huggingface/optimum-quanto](https://github.com/huggingface/optimum-quanto) | Apache License 2.0 | Optional Quanto runtime quantization backend |
| compressed-tensors | [neuralmagic/compressed-tensors](https://github.com/neuralmagic/compressed-tensors) | Apache License 2.0 | Optional pre-quantized checkpoint loading backend |
| triton | [triton-lang/triton](https://github.com/triton-lang/triton) | MIT | Optional GPU kernel compilation/runtime support |

## Representative Reference Models

These rows are representative public examples and smoke assets rather than an
exhaustive inventory of every external model identifier referenced by configs,
tests, or support metadata.

| Model | Publisher | Source | License |
|-------|-----------|--------|---------|
| `gpt2` | OpenAI | [Hugging Face](https://huggingface.co/gpt2) | [Modified MIT](https://huggingface.co/gpt2/resolve/main/README.md) |
| `bert-base-uncased` | Google | [Hugging Face](https://huggingface.co/bert-base-uncased) | [Apache License 2.0](https://huggingface.co/bert-base-uncased/resolve/main/README.md) |

Users are responsible for ensuring they comply with the upstream model licenses when redistributing
weights or deploying downstream products.

## Representative Reference Datasets

These rows are representative public datasets referenced by examples and smoke
flows rather than an exhaustive inventory of every dataset name used across
presets and internal test fixtures.

| Dataset | Publisher | Source | License |
|---------|-----------|--------|---------|
| WikiText-2 | Salesforce Research | [Hugging Face](https://huggingface.co/datasets/wikitext) | [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

## Additional Notes

- This file should be refreshed whenever direct dependencies, optional runtime extras, reference assets,
  or included runtime components change.
- Release bundles, SBOMs, and upstream license files remain authoritative if this summary becomes stale.
