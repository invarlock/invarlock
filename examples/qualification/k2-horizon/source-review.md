# K2 candidate runtime source review

!!! abstract "Assurance note"
    **In plain language:** the chosen native implementation can load the
    declared architecture without executing code supplied by the model
    repository, but its complete runnable image is not qualified yet.
    **Question:** which code and configuration boundaries must be preserved?
    **Decision use:** review a candidate build before authorizing qualification.
    **Evidence:** pinned public source inspection and dependency resolution;
    no GPU execution evidence.

The candidate SGLang source commit is
`392841f47cb7ef214601eeb528906a0abba02471`. It includes native K2 support and its
subsequent correctness fix. The contemporaneous `v0.5.19` release history does
not include these native files; selecting a newer-looking release label does
not establish equivalent support.

The reviewed boundary is explicit:

- [Native configuration](https://github.com/sgl-project/sglang/blob/392841f47cb7ef214601eeb528906a0abba02471/python/sglang/srt/configs/k2_horizon.py)
  registers the native model configuration. Model-repository Python is retained
  for artifact identity but is not trusted for execution.
- [Native model loading](https://github.com/sgl-project/sglang/blob/392841f47cb7ef214601eeb528906a0abba02471/python/sglang/srt/models/xllm.py)
  maps the K2 schema, rejects conflicting aliases and unsupported architecture
  settings, requires BF16, rejects quantization, and requires explicit source
  router partition provenance for MoVA. The campaign fixes source router
  partitions to 2 independently of tensor parallelism and keeps expert
  parallelism at 1. Expert load balancing, alternate expert placement, redundant
  experts, and two-batch overlap remain disabled.
- [Reasoning parsing](https://github.com/sgl-project/sglang/blob/392841f47cb7ef214601eeb528906a0abba02471/python/sglang/srt/parser/reasoning_parser.py)
  maps low effort to the corresponding IFM delimiter pair and starts inside the
  reasoning section. The campaign uses the native final content field without
  stripping reasoning itself or changing output text to obtain a match.
- [Native HTTP observations](https://github.com/sgl-project/sglang/blob/392841f47cb7ef214601eeb528906a0abba02471/python/sglang/srt/entrypoints/http_server.py)
  expose flattened resolved settings at `/server_info` and current model
  identity at `/model_info`. The capture worker checks both before and after capture.
  These responses are measurements attributed to the capture worker, not independent
  hardware attestation.
- [The chat protocol](https://github.com/sgl-project/sglang/blob/392841f47cb7ef214601eeb528906a0abba02471/python/sglang/srt/entrypoints/openai/protocol.py)
  accepts the frozen sampling and reasoning parameters. Truncated or unsupported
  native replies become explicit errors. No tool-call qualification follows
  from enabling a parser in upstream code.

This is a bounded review of the selected execution and parsing boundary, not
an assertion that every transitive runtime source line has been audited. The
whole immutable image and dependency inventory must be retained and reviewed.

## Source derivation and runtime gate

The maintained Python 3.12/Linux x86_64 lock contains 204 runtime, core, build,
and kernel distributions, including Torch 2.13.0, Transformers 5.12.1, and
FlashInfer 0.6.18 with CUDA 13 dependencies. The native path avoids checkpoint
`trust_remote_code`; the model-card Transformers runtime is a separate
configuration.

The unmodified source requires `outlines==0.1.11`, which depends on
`diskcache==5.6.3`. DiskCache has an unpatched unsafe-deserialization advisory,
[a published advisory](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v).
Writable cache contents can trigger pickle deserialization. A generic claim
that the chosen task does not use caching is insufficient to remove this
finding: the actual import and execution paths must be established.

The source helper authenticates the full archive, verifies the original bytes
of exactly three changed files, removes the optional Outlines dependency, and
replaces its two backend modules with explicit rejection. Native K2, HTTP,
configuration, reasoning, and grammar-dispatch source remain unchanged. Three
known source symlinks become regular files with the same target contents.
The resulting distribution has a distinct derived version; it must not be
represented as an unchanged upstream release. An advisory lookup for that
derived version does not establish source-level safety.

The maintained lock excludes Outlines and DiskCache. It binds the actual
NVIDIA CUDA Tile wheel rather than the PyPI download stub, plus both official
FlashInfer CUDA kernel binaries and CUDA 13 JIT wheel artifacts. Installation rejects source
distributions and requires hashes. No advisory suppression is included.
The image's Ubuntu packages use signed repository metadata and authenticated
local package artifacts. Package installation and source compilation run with
network access disabled; the hashed Python wheel installation is the separate
networked build step. Before that step, the image verifies and installs the
separately supplied pip wheel offline. The initial Ubuntu pip never downloads
the locked dependency set. The candidate core wheel retains its validated
filename and distribution version in the prepared manifest.
After runtime wheel installation, an offline purge removes only
`python3-pip-whl`, `python3.12-venv`, and `python3-venv`, without automatic
dependency removal. The final OS inventory is recorded afterward; compiler,
headers, native dependencies, and bootstrap provenance records are retained.
Remaining OS findings still require their own review and disposition.

To review a dependency update, prepare the authenticated source first, then
resolve the same Linux/Python target with the core and explicit build/kernel
requirements:

```bash
uv pip compile runtime-source/python/pyproject.toml pyproject.toml \
  examples/qualification/k2-horizon/runtime/build-requirements.in \
  examples/qualification/k2-horizon/runtime/kernel-requirements.in \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --only-binary :all: --generate-hashes --no-header --no-annotate \
  --find-links https://pypi.nvidia.com/cuda-tile/ \
  --find-links https://flashinfer.ai/whl/flashinfer-cubin/ \
  --find-links https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/ \
  --constraints requirements/workflows/k2-campaign-py312.txt \
  --emit-find-links --output candidate-runtime-lock.txt
```

Review the resulting versions and hashes against the maintained lock. Some
vendor indexes require streaming an entire wheel before its hash is available;
never accept a version-only line or a stub download. The maintained vendor
hashes come from the official NVIDIA index and FlashInfer release artifacts,
and installation independently recomputes them. Audit the changed closure
before replacing the maintained lock or rebuilding its image.

Before producing a ready build receipt:

1. Exercise the derived source's actual installed imports and unchanged grammar
   dispatcher. Confirm that excluded operations fail explicitly, the excluded
   distributions are absent, and the selected native modules retain their
   reviewed hashes. Source-level tests alone do not establish these results.
2. Archive the exact source commit. Bind any derived source change, the
   Dockerfile, a digest-pinned CUDA 13 base, the complete hashed dependency lock,
   candidate InvarLock wheel, and campaign source bytes in the build manifest.
3. Build for Linux x86_64, run dependency consistency and vulnerability checks
   for both Python and operating-system packages,
   observe installed source hashes and package versions, and retain the image
   ID and exported image identity. The source-review hashes in the catalog must
   match the installed native K2 and reasoning implementation.
4. Run the native CPU import and help checks in the final image with network
   access disabled. A ready runtime receipt means this image is ready for the
   separately authorized GPU preflight; it does not mean that a model or GPU
   configuration has been qualified.

After the ready image has an approved compute budget, run the bounded H200
preflight with real tensors. Check dtype, architecture, parsers, selected
parallelism, driver, CUDA kernels, GPU memory, topology, and resource bounds
before decisions. GPU startup and inference failures remain failed preflights;
they cannot inherit success from the earlier CPU image checks.

The maintained finalization command reconstructs the prepared context, observes the exact
local image, and retains the raw dependency reports and every OS applicability
disposition. Its receipt remains blocked without an explicit scoped security
decision, even when CPU checks pass. An unresolved finding cannot become an
accepted applicability row without an attributed rationale and supporting
evidence. The receipt is an unsigned local observation; it does not establish
independent execution attestation. See [finalization](README.md#finalize-the-exact-image)
for the command and review fields.
