# Third-party notices

InvarLock relies on open-source dependencies and optional runtime components.
This summary tracks the direct dependencies declared by the current source
tree. It is informational, not an exhaustive transitive dependency or license
manifest. Upstream license files, built-artifact metadata, container contents,
and release SBOMs remain authoritative for a particular installation.

## Core distribution

The `invarlock` distribution declares these direct runtime dependencies in
`pyproject.toml`:

| Component | Upstream | License | Use |
| --- | --- | --- | --- |
| `typer` | [fastapi/typer](https://github.com/fastapi/typer) | MIT | Installed CLI framework |
| `click` | [pallets/click](https://github.com/pallets/click) | BSD 3-Clause | CLI parsing |
| `cryptography` | [pyca/cryptography](https://github.com/pyca/cryptography) | Apache-2.0 OR BSD-3-Clause | Evidence and receipt signatures |
| `rich` | [Textualize/rich](https://github.com/Textualize/rich) | MIT | Terminal rendering |
| `pyyaml` | [yaml/pyyaml](https://github.com/yaml/pyyaml) | MIT | Evaluation request parsing |
| `jsonschema` | [python-jsonschema/jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | Public contract validation |

## Hugging Face extra

The built-in Hugging Face provider is installed with the `hf` extra. Its direct
optional dependencies are:

| Component | Upstream | License | Use |
| --- | --- | --- | --- |
| `accelerate` | [huggingface/accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 | Device placement and quantized checkpoint loading support |
| `torch` | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | Tensor runtime |
| `transformers` | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Model and tokenizer loading |
| `safetensors` | [huggingface/safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Tensor-only artifact loading |
| `protobuf` | [protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf) | BSD 3-Clause | Model and tokenizer serialization support |
| `sentencepiece` | [google/sentencepiece](https://github.com/google/sentencepiece) | Apache-2.0 | SentencePiece tokenizer runtime |
| `tiktoken` | [openai/tiktoken](https://github.com/openai/tiktoken) | MIT | GPT-style tokenizer runtime |

## First-party optional distributions

The release builds five separately installable Python distributions:

| Distribution | Direct runtime dependency boundary |
| --- | --- |
| `invarlock` | Core dependencies above; the Hugging Face stack is an optional extra |
| `invarlock-diagnostics` | NumPy, licensed under BSD 3-Clause |
| `invarlock-runtime-gguf` | A compatible `invarlock` core distribution |
| `invarlock-runtime-hf-vision-text` | A compatible `invarlock` core distribution; its `runtime` extra adds the Hugging Face stack and Pillow (HPND) |
| `invarlock-runtime-tensorrt-llm` | A compatible `invarlock` core distribution |

The GGUF, vision-text, and TensorRT-LLM connector wheels do not bundle model
weights or native backends. The vision-text image adds Pillow from a hash-pinned
lock to the selected digest-pinned InvarLock CUDA base. The repository's
optional GGUF image builds a pinned
[llama.cpp](https://github.com/ggml-org/llama.cpp) source archive. The optional
TensorRT-LLM image inherits a digest-pinned NVIDIA TensorRT-LLM release image,
whose CUDA, TensorRT, and other components remain subject to their upstream
license terms. Inspect the relevant Dockerfile, pinned requirements, image
contents, and generated SBOM before redistributing an image.

## External artifacts

The repository's runnable offline example uses synthetic records and does not
bundle external model weights or datasets. Users who evaluate third-party
models, tokenizers, datasets, containers, or other artifacts are responsible
for the terms that apply to those inputs and to redistributed outputs.

Refresh this notice whenever declared dependencies, first-party distribution
boundaries, runtime images, or bundled external assets change.
