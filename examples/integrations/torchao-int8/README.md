# TorchAO INT8

This example measures the behavioral effect of a real TorchAO INT8 weight-only
transformation with InvarLock's standard Hugging Face runtime.

From the repository root:

```bash
make example-torchao-int8
```

The command:

1. downloads the official Apache-2.0 `Qwen/Qwen3-0.6B` checkpoint at immutable
   revision `c1899de289a04d12100db370d81485cdf75e47ca`;
2. applies TorchAO's `Int8WeightOnlyConfig(version=2)` transformation;
3. confirms that TorchAO created quantized tensor subclasses while preserving
   Qwen's tied output projection;
4. materializes those quantized tensors exactly as a portable Hugging Face
   subject checkpoint and confirms that save/reload preserves every transformed
   dense tensor;
5. evaluates 50 paired records in the authenticated InvarLock runtime;
6. verifies the signed evidence with a separately generated verifier key; and
7. renders an HTML comparison report.

The acceptance policy allows at most a 1% normalized-NLL increase for this
compact demonstration. The resulting bundle records the compared baseline and
materialized subject checkpoint identities. Its authenticated TorchAO
observation records the source model and revision, complete quantization and
module-selection settings, software versions, transformed-tensor commitment,
exact materialization checks, and subject digest without influencing
acceptance. It also records finite next-token differences between TorchAO's
live INT8 kernel and the dense checkpoint across all 50 inputs. Those values are
diagnostic: different floating-point kernels need not be bitwise identical.

The acceptance verdict applies to the saved Hugging Face checkpoint. It does
not claim that the dense runtime reproduces every numerical detail of TorchAO's
live INT8 kernel.

This example evaluates the behavioral result of quantization. GGUF and
TensorRT-LLM examples exercise those runtime formats directly when runtime
performance and format-specific execution are the subject of the comparison.

Use `EXAMPLE_ARGS` to select a new output directory or another container
engine:

```bash
make example-torchao-int8 \
  EXAMPLE_ARGS="--workspace /tmp/torchao-evidence --container-engine podman"
```

The workspace must not already exist. Generated signing keys remain inside that
disposable workspace and are intended only for the example transaction.
CUDA is selected when available and CPU remains supported. The first run needs
several gigabytes of download, cache, and workspace capacity.
