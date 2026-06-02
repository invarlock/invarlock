# Example Runtime Images

These images are only for integration example evidence runs. They are not the
regular InvarLock runtime images used by normal container-backed evaluation.

The split images keep optional quant backends out of the default runtime path:

| Family | Image tag | Used by |
| --- | --- | --- |
| `cuda-bnb` | `invarlock-example-runtime:cuda-bnb` | `hf_bnb` |
| `cuda-compressed-tensors` | `invarlock-example-runtime:cuda-compressed-tensors` | `compressed_tensors` |
| `cuda-gptqmodel` | `invarlock-example-runtime:cuda-gptqmodel` | `awq`, `gptqmodel` |
| `cuda-hqq` | `invarlock-example-runtime:cuda-hqq` | `hqq` |
| `cuda-quanto` | `invarlock-example-runtime:cuda-quanto` | `quanto` |
| `cuda-torchao` | `invarlock-example-runtime:cuda-torchao` | `torchao_int8_runtime` |

CUDA compatibility is split across host and image. The host supplies the NVIDIA
driver through the container runtime; the image supplies the pinned CUDA
user-space libraries expected by the Python stack. It is not enough for the
host to have some CUDA toolkit installed, because the strict evidence run needs
the container's Python packages, CUDA user-space libraries, and image digest to
be reproducible together. The smoke script detects whether a GPU is visible at
runtime, but the image does not mutate itself by reusing host CUDA libraries or
installing host-specific packages.

The slim families use PyTorch CUDA wheels and do not include a system CUDA
toolkit. `cuda-gptqmodel` keeps a CUDA-devel base because GPTQModel/AWQ strict
runs need toolchain surfaces that the slim wheel-only images do not provide.

Build and smoke one image from the repository root:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-hqq
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-hqq
```

On Docker hosts with `nvidia-smi`, the smoke automatically passes `--gpus all`
and requires a visible CUDA device. Set
`INVARLOCK_EXAMPLE_RUNTIME_REQUIRE_GPU=0` for import-only smoke checks.

Then run the matching example with:

```bash
INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-hqq \
examples/integrations/hqq/run_tiny_hf_hqq.sh --allow-network --force --lane cuda
```

Strict shared artifacts should include the digest-pinned image reference recorded
in `runtime.manifest.json`.
