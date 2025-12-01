# Third-Party Notices

InvarLock relies on several open-source projects, datasets, and model weights. This document lists upstream
artifacts, their maintainers, and license terms as of 2025-12-01.

## Frameworks and Libraries

| Component | Upstream | License | Notes |
|-----------|----------|---------|-------|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | [BSD 3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Core tensor and autograd runtime |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) | Model loading, tokenizers, generation utilities |
| Datasets | [huggingface/datasets](https://github.com/huggingface/datasets) | [Apache License 2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) | Dataset ingestion (e.g., WikiText-2) |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Numerical kernels |
| SciPy | [scipy/scipy](https://github.com/scipy/scipy) | [BSD 3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Linear algebra utilities |
| tqdm | [tqdm/tqdm](https://github.com/tqdm/tqdm) | [MPL 2.0](https://github.com/tqdm/tqdm/blob/master/LICENSE) | Progress bars |
| Hugging Face Hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | [Apache License 2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) | Model and dataset registry, artifact resolution |
| Accelerate | [huggingface/accelerate](https://github.com/huggingface/accelerate) | [Apache License 2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE) | Device placement and distributed helpers (optional) |
| bitsandbytes | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | [MIT License](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE) | Optional GPU quantization/runtime kernels |
| AutoGPTQ | [PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | [Apache License 2.0](https://github.com/PanQiWei/AutoGPTQ/blob/main/LICENSE) | Optional GPTQ quantization backend (Linux-only extra) |
| AutoAWQ | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | [Apache License 2.0](https://github.com/casper-hansen/AutoAWQ/blob/main/LICENSE) | Optional AWQ quantization backend (Linux-only extra) |
| Triton | [openai/triton](https://github.com/openai/triton) | [MIT License](https://github.com/openai/triton/blob/main/LICENSE) | Optional GPU kernel compilation (Linux-only extra) |
| Optimum | [huggingface/optimum](https://github.com/huggingface/optimum) | [Apache License 2.0](https://github.com/huggingface/optimum/blob/main/LICENSE) | Optional ONNX/accelerated runtime integration |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | Optional ONNX inference runtime |

## Reference Models

| Model | Publisher | Source | License |
|-------|-----------|--------|---------|
| `gpt2` | OpenAI | [Hugging Face](https://huggingface.co/gpt2) | [Modified MIT](https://huggingface.co/gpt2/resolve/main/README.md) |
| `bert-base-uncased` | Google | [Hugging Face](https://huggingface.co/bert-base-uncased) | [Apache License 2.0](https://huggingface.co/bert-base-uncased/resolve/main/README.md) |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | TinyLlama team | [Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | [Apache License 2.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/LICENSE) |

Users are responsible for ensuring they comply with the upstream model licenses when redistributing weights
or deploying downstream products.

## Datasets

| Dataset | Publisher | Source | License |
|---------|-----------|--------|---------|
| WikiText-2 | Salesforce Research | [Hugging Face](https://huggingface.co/datasets/wikitext) | [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

## Additional Notes

- These notices are informational and do not replace the official upstream license terms.
- If you add or update dependencies, models, or datasets, please refresh this document before release.
