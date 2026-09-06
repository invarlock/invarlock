# K2 Horizon candidate campaign

!!! tip "User guide"
    **Outcome:** prepare frozen comparisons, capture native responses, and verify
    signed pipeline results independently.
    **Audience:** maintainers qualifying the declared K2 configurations.
    **Prerequisites:** a candidate InvarLock wheel for CPU preparation; a reviewed,
    immutable runtime image and an approved compute budget before GPU execution.

All five configurations are **candidates, not qualified**. No K2 weights have
been executed to establish the claims described here. The harness can prepare
and test its protocol on a CPU. The restricted runtime source excludes the
optional dependency finding recorded in [runtime readiness](runtime-readiness.json).
Do not provision GPU time until the exact image passes its build, dependency,
and native-import checks. Actual GPU startup is the first bounded preflight.

This route uses SGLang's native K2 implementation to generate responses and
InvarLock's public pipeline interface to compare the captures. A signed result
authenticates the captured inputs, attributed measurements, and deterministic
scoring. It is not native InvarLock isolated-transaction evidence, proof of GPU
execution, or a general endorsement of a model's quality. The core provider's
remote-code restrictions remain unchanged.

## Frozen model pairs

The [catalog](catalog.json) contains immutable revisions and all 477 selected
file identities across the five pairs. Publisher-advertised file hashes are
inputs to verification, not measurements already made by InvarLock.

| Model | Baseline revision | Candidate revision | Tensor parallelism |
| --- | --- | --- | --- |
| `IFM/K2-Horizon-0.9B` | `92959a6834719d4fac3b8e3e080f48605c5cbe1f` | `02d0da0fefe5a2f8dc3db091cad29b15c9d8e4fa` | 1 |
| `IFM/K2-Horizon-3.7B` | `2cd32eff5bdcae05fea9cd665295e69c7d0d62fe` | `633f52ad28b17edeabd82afc61d2d13b4c59a561` | 1 |
| `IFM/K2-Horizon-7B` | `c0bab2cc96a6953d7fad4d828938c729ae8088b2` | `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10` | 1 |
| `IFM/K2-Horizon-MoVA-36B-A4B` | `a37d42a0475d090feba15a69bef441f4b5caf43b` | `05cab0a4d7150c1c460a000b37ff40cc1af2feaa` | 2 |
| `IFM/K2-Horizon-32B` | `ac2f1a7b9fad9cf2d3815b3b5cc81501cf15dbb0` | `466db5f23c8a7c96b0b320b688612ee6f4446a35` | 2 |

The 7B label corresponds to approximately 9B total parameters in the published
metadata. The MoVA model stores approximately 37.4B parameters; its active
parameter label does not describe weight residency. The 32B pair compares
training stage 4 with the available Stage-1 release. Each pair's specific
change is recorded in the catalog.

The [model-card observations](model-card-observations.json) retain the license
fields at these exact revisions. The 0.9B candidate declares both `apache-2.0`
and `license_name: internal-only`, with a missing linked `LICENSE` file. The
five baseline revisions have no model-card license metadata. These facts need
clarification before commercial reference promotion; successful technical
qualification does not establish applicable model terms.

Materialization recomputes file hashes and logical BF16 tensor hashes, including
tensor names and shapes. Changing shard layout alone cannot satisfy the
required checkpoint change. A tensor-name or shape mismatch stops the campaign
for review instead of silently comparing a different architecture. Model
repository Python files are authenticated as artifact bytes; they are not
executed by this route.

## Protocol and limits

Each model has 576 original synthetic workflow cases: 192 routing decisions,
192 structured invoice extractions, and 192 integer arithmetic answers. Each
cohort contains 96 plain cases and 96 cases with irrelevant context. The stable IDs,
prompts, references, context labels, order, metric settings, and policy are
stored together in the generated plan. These cases qualify a workflow; they
are not a representative customer dataset or a general benchmark.

Routing uses literal match; extraction checks three declared JSON fields;
arithmetic uses zero-tolerance numeric comparison. All cohorts also record
request latency. Each overall result and context slice requires 96 usable
pairs, a candidate quality floor of 0.8, a maximum regression of 0.05, and a
95% interval no wider than 0.25. Latency permits at most a 5-second regression,
a 10-second interval width, and a 30-second candidate mean. Passing these
illustrative thresholds is not a production service-level commitment. Intervals
apply separately to each metric and slice; they do not provide a simultaneous
campaign-wide confidence guarantee or make these synthetic cases representative
of a customer population.

Generation fixes BF16, native `k2_horizon` reasoning parsing, low reasoning
effort, greedy sampling, seed 20260905, one request at a time, a 4096-token
context, and at most 512 generated tokens. Truncation, missing records, request
errors, and unsupported responses remain errors; they cannot become successful
answers through cleanup. Greedy sampling is not a claim of bitwise deterministic
GPU execution. Tool calling, FP8, alternative runtimes, expert parallelism
beyond 1, and maximum advertised context are outside this campaign.

The separate throughput preflight has 24 different cases. It retains its raw
responses and reports timing rather than quality decisions. A decision run
requires the matching completed preflight and a projected duration within the
frozen budget. A failed preflight requires a separately reviewed new plan;
retain the failed attempt and never change thresholds or select another pair
after observing a decision result.

## CPU preparation

Install the candidate wheel and its Hugging Face extra for later downloads.
Run commands from this repository, with the installed wheel available to
Python. Ordinary InvarLock onboarding remains GPU-free.

```bash
python -m examples.qualification.k2_campaign plan \
  --model 0.9b --output draft-0.9b.json

python -m pytest tests/examples/test_k2_campaign.py \
  tests/examples/test_k2_producer.py
```

The model selectors are `0.9b`, `3.7b`, `7b`, `mova-36b-a4b`, and `32b`.
The plan command neither downloads weights nor starts a container. Every
output path must be new.

## Build the candidate runtime

The source helper authenticates the complete SGLang archive before extraction.
It changes only the optional Outlines dependency and its two backend modules;
selecting either excluded operation raises an explicit error. Native K2 code,
reasoning parsing, and the HTTP interface retain their reviewed bytes.

```bash
curl --fail --location --output sglang-source.tar.gz \
  https://codeload.github.com/sgl-project/sglang/tar.gz/392841f47cb7ef214601eeb528906a0abba02471
python -m examples.qualification.k2_runtime_source \
  --archive sglang-source.tar.gz --output runtime-source
```

Resolve the Ubuntu packages using the exact base image and signed Ubuntu
repositories. The resolution container downloads packages but does not execute
models. Inspect the retained package versions, repository signatures, and
artifact hashes before accepting the resulting manifest identity. The index
helper uses `gpgv` and the public Ubuntu keyring pinned from that base, fetches
exact compressed indexes by their signed hashes, and retains them for replay.
Context preparation rechecks the signature-to-index-to-package chain for every
selected package; it rejects changed metadata or an unmatched artifact.

```bash
mkdir runtime-apt
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD/runtime-apt,dst=/out" \
  --mount "type=bind,src=$PWD/examples/qualification/k2-horizon/runtime/resolve-os.sh,dst=/resolve-os.sh,readonly" \
  --mount "type=bind,src=$PWD/examples/qualification/k2-horizon/runtime/os-security-pins.txt,dst=/security-pins.txt,readonly" \
  nvidia/cuda@sha256:a85c9f5af049f0ab679c1669ae6fa8393022886739af7361e85bb96878e8cdd4 \
  bash /resolve-os.sh
python -m examples.qualification.k2_runtime_apt --bundle runtime-apt

python -m examples.qualification.k2_runtime_build \
  --archive sglang-source.tar.gz --core-wheel "$CORE_WHEEL" \
  --expected-core-wheel-sha256 "$EXPECTED_CORE_WHEEL" \
  --apt-bundle runtime-apt \
  --expected-apt-manifest-sha256 "$EXPECTED_OS_MANIFEST" \
  --output runtime-context
docker build --platform linux/amd64 --iidfile runtime-image-id.txt \
  --tag k2-candidate:reviewed runtime-context
docker run --rm --platform linux/amd64 --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user 65532:65532 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
  --env HF_HOME=/tmp/huggingface \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 2 --memory 8g --pids-limit 512 --entrypoint timeout \
  "$(cat runtime-image-id.txt)" --signal=TERM --kill-after=10s 360 \
  python /opt/campaign/native_probe.py > native-imports.json
```

The image installs its Ubuntu bundle with network access disabled, and verifies
every selected artifact. Python installation permits only hashed wheels from
the maintained Linux x86_64/Python 3.12 lock. This includes pinned NVIDIA CUDA
Tile, FlashInfer CUDA kernel binaries and CUDA 13 JIT assets. Runtime execution has no network
access; bundled kernels must suffice. Rust extensions are disabled for this
Python HTTP route, and the FA3 selection uses the bundled kernel implementation.

The native probe checks installed source identities, actual imports, server help,
dependency consistency, and rejection of the excluded grammar operation through
the upstream test context. CPU imports do not exercise GPU-conditional kernels. Its
result explicitly has no GPU qualification authority. Retain both Python and
OS vulnerability reports for the final image, including applicability decisions
and unresolved findings. A prepared context or successful build alone cannot
produce a ready runtime receipt.

## Materialize and freeze before evaluation

After the runtime is ready and execution has an approved budget, provision a
Linux x86_64 host with two full H200 141GB GPUs, verified NVLink connectivity,
512GB host RAM, 32 vCPUs, and 2TB free NVMe space. Run checkpoint roles and model
pairs sequentially. These host resource figures are planning estimates, not
measurements. The worker checks H200 identity, memory, GPU count, and the reviewed R580
branch at version 580.159.03 or later. Request 580.178.04; the minimum follows
[NVIDIA's driver security bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5821).
Newer driver branches require their own verified security minimum before being
accepted. The image's CUDA requirements, actual loaded host driver libraries,
and NVLink topology must also pass host preflight.

Download only the catalog's selected files into fresh, regular-file snapshots:

```bash
python -m examples.qualification.k2_campaign download \
  --model 0.9b --role baseline --output baseline-model
python -m examples.qualification.k2_campaign download \
  --model 0.9b --role candidate --output candidate-model
python -m examples.qualification.k2_campaign measure \
  --model 0.9b --role baseline --snapshot baseline-model \
  --output baseline-measurement.json
python -m examples.qualification.k2_campaign measure \
  --model 0.9b --role candidate --snapshot candidate-model \
  --output candidate-measurement.json
```

The reviewed image builder must produce `runtime-build.json` with format
`invarlock/k2-runtime-build-v1`, status `ready`, the exact SGLang source commit
and reviewed source-file hashes from the catalog, its Docker image ID, source
archive digest, complete dependency-inventory digest, and security-review
digest. A Docker image ID is an OCI configuration digest; do not substitute a
registry manifest digest for it. Retain the image archive and any registry
manifest relationship separately. The image must contain the candidate
InvarLock wheel and both campaign modules plus the catalog under an importable
`examples.qualification` directory. The plan binds both modules' actual bytes.

Do not fabricate a ready receipt. The reviewed runtime closure and executable
image are still required; see the build strategy in
[the source review](source-review.md).

Set the two budget variables to the separately approved per-role limits. A
campaign-wide spending ceiling and explicit stop time must also be agreed
before paid execution; this example does not infer them.

```bash
python -m examples.qualification.k2_campaign freeze \
  --model 0.9b --runtime-build runtime-build.json \
  --baseline-measurement baseline-measurement.json \
  --candidate-measurement candidate-measurement.json \
  --maximum-wall-seconds "$APPROVED_ROLE_SECONDS" \
  --maximum-output-tokens "$APPROVED_ROLE_TOKENS" \
  --output plan.json
```

Give the independent recipient the plan and its canonical digest before
decision runs. Keep signing keys outside all model, capture, and container
mounts. The signing key is used only by the later host-side publication step.

## Native preflight and decision capture

The capture worker starts a new network-disabled, read-only container using the
frozen image ID and read-only model and plan mounts. It checks actual model and
server configuration before and after requests, observes latency, retains
native JSON responses, and removes only its own named container after success
or timeout. It rechecks model bytes after the run. These are enforced capture worker
controls; the resulting external evidence still relies on the capture worker's
attributed observations.

For each role, run the preflight, inspect its resource estimate, then run the
decision capture only within the approved ceiling:

```bash
python -m examples.qualification.k2_producer run \
  --plan plan.json --role baseline --snapshot baseline-model \
  --phase preflight --output baseline-preflight
python -m examples.qualification.k2_producer run \
  --plan plan.json --role candidate --snapshot candidate-model \
  --phase preflight --output candidate-preflight

python -m examples.qualification.k2_producer run \
  --plan plan.json --role baseline --snapshot baseline-model \
  --phase decision --preflight baseline-preflight/capture.json \
  --output baseline-decision
python -m examples.qualification.k2_producer run \
  --plan plan.json --role candidate --snapshot candidate-model \
  --phase decision --preflight candidate-preflight/capture.json \
  --output candidate-decision
```

Each request has a timeout of at most 120 seconds. Wall-time and output-token
budgets stop further requests. Remaining scheduled records retain explicit
errors, producing insufficient evidence instead of a smaller selected sample.
The host terminates the container at the wall-time ceiling, including startup.
A hard timeout may leave only diagnostic logs; it never produces qualification.

## Publish and independently verify

Publish all three cohort decisions, including regressions and insufficient
evidence. Publication exit codes are 0 for pass, 1 for regression, 2 for an
input/runtime error, and 3 for insufficient evidence. Verification exits 0 when
evidence is valid, even when it records a regression or insufficient evidence;
its output retains each cohort decision. Invalid evidence exits 2.

```bash
python -m examples.qualification.k2_campaign publish \
  --plan plan.json --baseline baseline-decision/capture.json \
  --candidate candidate-decision/capture.json \
  --key evidence-private.pem --output evidence.json

python -m examples.qualification.k2_campaign report \
  --evidence evidence.json --output reports
```

Transfer the plan, raw captures, evidence, and separately trusted public key to
a clean recipient environment containing the installed candidate wheel and
this example. Supply independently obtained canonical digests for the plan
and both captures. Do not derive those expected values from the evidence being
verified. The public SDK's `invarlock.pipeline.contracts.digest` computes this
canonical JSON identity; a raw-file `sha256sum` has different semantics.

```bash
python -m examples.qualification.k2_campaign verify \
  --plan plan.json --baseline baseline-decision/capture.json \
  --candidate candidate-decision/capture.json --evidence evidence.json \
  --key evidence-public.pem --expected-plan "$EXPECTED_PLAN" \
  --expected-baseline-capture "$EXPECTED_BASELINE_CAPTURE" \
  --expected-candidate-capture "$EXPECTED_CANDIDATE_CAPTURE" \
  --output verification.json
```

Verification reconstructs each public pipeline run from the raw native
response, checks the frozen request and configuration, verifies the signature
against the recipient's key, and replays each declared policy. It must reject
wrong keys, changed requests, substituted captures, changed model bindings,
and reordered or missing records. The tests include these boundary controls
and synthetic rejection/error cases; they are not K2 inference evidence.

Retain the exact source, image, dependency inventory, materializations, frozen
plan, preflights, raw captures, signed results, independent verification,
negative controls, and observed resource use for every model. Publication of a
qualified reference requires reviewing those actual artifacts. A quality
rejection can demonstrate a working workflow; insufficient evidence cannot.
Keep the existing Qwen and other historical evidence and CPU onboarding path.
