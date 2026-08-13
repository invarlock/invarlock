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

## Public qualification data

The repository contains deterministic 400-record qualification schedules
derived from these pinned public datasets:

| Dataset | Upstream license | Material represented in evidence |
| --- | --- | --- |
| [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT | Selected question text, answer choices, expected answers, categories, and stable record identities |
| [MMMU/MMMU-Pro Vision](https://huggingface.co/datasets/MMMU/MMMU_Pro) | Apache-2.0 | Selected record identities, expected answers, media digests, and the authenticated vision-text prompt; original media bytes remain external |
| [EleutherAI/LAMBADA OpenAI](https://huggingface.co/datasets/EleutherAI/lambada_openai) | MIT metadata; dataset card references Modified MIT | Selected English prompts, final-word targets, and stable record identities |

The checked-in
[qualification-suite manifest](docs/reference/qualification-suites.manifest.json)
binds those details for MMLU-Pro and MMMU-Pro Vision. The
[flagship evaluator corpus descriptor](examples/integrations/evaluator_transaction/flagship_corpus.json)
binds the corresponding revision, source digest, eligibility criteria,
selection seed, selected identities, and derived dataset digest for LAMBADA
OpenAI. The upstream terms continue to apply to the represented material.

### MMLU-Pro selected dataset material

The selected MMLU-Pro dataset material is distributed under the MIT License
declared by the upstream dataset card.

Copyright (c) 2024 MMLU-Pro authors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### LAMBADA OpenAI selected dataset material

The selected English LAMBADA material comes from the OpenAI-preprocessed test
split distributed by EleutherAI. The dataset card declares `mit` in its
metadata and identifies the following Modified MIT License in its licensing
section. It attributes the original LAMBADA dataset to
[Paperno et al.](https://doi.org/10.5281/zenodo.2630551) and the preprocessing
to OpenAI's GPT-2 work.

Modified MIT License

Software Copyright (c) 2019 OpenAI

We don't claim ownership of the content you create with GPT-2, so it is yours
to do with as you please. We only ask that you use GPT-2 responsibly and
clearly indicate your content was created using GPT-2.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## External artifacts

The repository's runnable offline example uses synthetic records. Apart from
the selected public qualification schedules described above, the repository
does not bundle external model weights or complete datasets. Users who evaluate
third-party models, tokenizers, datasets, containers, or other artifacts are
responsible for the terms that apply to those inputs and to redistributed
outputs.

Refresh this notice whenever declared dependencies, first-party distribution
boundaries, runtime images, or bundled external assets change.
