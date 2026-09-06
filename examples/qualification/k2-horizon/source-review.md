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

## Build strategy and current blocker

The pinned source's Python 3.12/Linux x86_64 dependency closure resolves to 202
packages, including Torch 2.13.0, Transformers 5.12.1, and FlashInfer 0.6.18 with
CUDA 13 dependencies. The native path avoids checkpoint `trust_remote_code`;
the model-card Transformers runtime is a separate configuration.

The resolved closure includes `outlines==0.1.11`, which depends on
`diskcache==5.6.3`. DiskCache has an unpatched unsafe-deserialization advisory,
[a published advisory](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v).
Writable cache contents can trigger pickle deserialization. A generic claim
that the chosen task does not use caching is insufficient to remove this
finding: the actual import and execution paths must be established.

No vulnerable runtime lock or advisory suppression is included in the
maintained campaign. Before producing a ready build receipt:

1. Resolve the optional grammar/cache dependency through a reviewed source
   change or upstream fix. Preserve native K2 code bytes and make excluded
   operations fail explicitly. Exercise real imports and generation, rather
   than changing dependency metadata alone.
2. Archive the exact source commit. Bind any derived source change, the
   Dockerfile, a digest-pinned CUDA 13 base, the complete hashed dependency lock,
   candidate InvarLock wheel, and campaign source bytes in the build manifest.
3. Build for Linux x86_64, run dependency consistency and vulnerability checks,
   observe installed source hashes and package versions, and retain the image
   ID and exported image identity. The source-review hashes in the catalog must
   match the installed native K2 and reasoning implementation.
4. Run native import/startup tests, then the bounded H200 preflight with real
   tensors. Check dtype, architecture, parsers, selected parallelism, driver,
   CUDA kernels, GPU memory, topology, and resource bounds before decisions.

The runtime readiness file deliberately remains blocked until those concrete
requirements are fulfilled. A valid ready receipt must come from the reviewed
build and checks; manually filling digest-shaped strings does not establish
readiness or qualification.
